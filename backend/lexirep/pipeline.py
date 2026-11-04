"""
backend.lexirep.pipeline

End-to-end orchestration pipeline for LexiRep integration in ProsodyLab.
Connects:
1. Audio preprocessing & Whisper transcription
2. Acoustic pause measurement & word timestamp realignment
3. Canonical syllable count & identity engine
4. Equal-duration syllable interval partitioning
5. Full-utterance Wav2Vec 2.0 Layer-12 extraction (single pass)
6. Discrete frame-center syllable pooling
7. Frozen LexiRep inference service (GER / ITA)
8. Structured LexiRep result generation

Constraints:
- Preserves existing WhiStress word prominence (does not overwrite word['stressed']).
- Bypasses LexiRep for monosyllables (K = 1).
- Requires explicit nativity parameter ("GER" or "ITA").
- Reuses existing isolated services without duplicating inference logic.
"""

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple, Union
import time
import io
import numpy as np
import soundfile as sf
import torch

from whistress_service import service as whistress_service
from pitch_analysis import extract_pitch, realign_word_timestamps
from backend.syllable import SyllableEngine, partition_syllable_intervals, SyllableInterval
from backend.lexirep.feature_extractor import Wav2VecFeatureExtractor
from backend.lexirep.pooling import pool_syllables
from backend.lexirep.service import LexiRepService, SUPPORTED_LANGUAGES


@dataclass
class LexiRepSyllable:
    """Syllable-level representation with LexiRep inference metrics."""
    index: int
    text: str
    start: float
    end: float
    duration: float
    stress: Optional[int] = None
    contrast_score: Optional[float] = None
    confidence: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LexiRepWord:
    """Word-level representation containing both WhiStress and LexiRep predictions."""
    word: str
    start: float
    end: float
    duration: float
    is_polysyllabic: bool
    syllable_count: int
    whistress_stressed: bool
    lexirep_stressed_syllable: Optional[int]
    syllables: List[LexiRepSyllable]
    syllable_source: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["syllables"] = [s.to_dict() for s in self.syllables]
        d["syllable_source"] = self.syllable_source
        return d


@dataclass
class LexiRepPipelineResult:
    """Structured result of the complete LexiRep end-to-end pipeline."""
    nativity: str
    transcription: str
    audio_duration: float
    words: List[LexiRepWord]
    linguistic_compliance: bool
    num_polysyllabic_words: int
    total_syllables_pooled: int
    wav2vec_frames: int
    execution_stats: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nativity": self.nativity,
            "transcription": self.transcription,
            "audio_duration": round(self.audio_duration, 4),
            "words": [w.to_dict() for w in self.words],
            "linguistic_compliance": self.linguistic_compliance,
            "num_polysyllabic_words": self.num_polysyllabic_words,
            "total_syllables_pooled": self.total_syllables_pooled,
            "wav2vec_frames": self.wav2vec_frames,
            "execution_stats": {k: round(v, 4) for k, v in self.execution_stats.items()},
        }


class LexiRepPipeline:
    """
    End-to-End Orchestrator connecting Whisper, Realignment, Syllabification,
    Wav2Vec2, and LexiRep.
    """

    def __init__(
        self,
        syllable_engine: Optional[SyllableEngine] = None,
        feature_extractor: Optional[Wav2VecFeatureExtractor] = None,
        lexirep_service: Optional[LexiRepService] = None,
    ) -> None:
        self.syllable_engine = syllable_engine or SyllableEngine()
        self.feature_extractor = feature_extractor or Wav2VecFeatureExtractor()
        self.lexirep_service = lexirep_service or LexiRepService()

    def _validate_nativity(self, nativity: str) -> str:
        if not nativity or not isinstance(nativity, str):
            raise ValueError(
                f"nativity must be explicitly specified as one of {SUPPORTED_LANGUAGES}. Got: {nativity}"
            )
        lang = nativity.upper().strip()
        if lang not in SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Unsupported nativity: '{nativity}'. Must be one of {SUPPORTED_LANGUAGES}"
            )
        return lang

    def analyze_realigned_words(
        self,
        audio_array: np.ndarray,
        sr: int,
        realigned_words: List[Tuple[str, int, float, float]],
        nativity: str,
        execution_times: Optional[Dict[str, float]] = None,
    ) -> LexiRepPipelineResult:
        """
        Executes syllable extraction, Wav2Vec2 forward pass, pooling, and LexiRep
        inference on already-realigned word tuples: [(word_text, whistress_stress, start, end), ...].
        """
        lang = self._validate_nativity(nativity)
        t_start = time.perf_counter()
        stats: Dict[str, float] = execution_times.copy() if execution_times else {}

        audio_duration = len(audio_array) / float(sr)

        # ---------------------------------------------------------------------
        # 1. Syllable Engine & Equal-Duration Partitioning
        # ---------------------------------------------------------------------
        t0 = time.perf_counter()
        transcription_words: List[str] = []
        structured_words: List[LexiRepWord] = []

        poly_intervals_flat: List[Tuple[float, float]] = []
        poly_word_groupings: List[List[int]] = []
        poly_word_indices: List[int] = []
        poly_word_texts: List[str] = []
        global_syl_counter = 0

        for w_idx, (word_text, whistress_stress, start_s, end_s) in enumerate(realigned_words):
            start = float(start_s)
            end = float(end_s)
            w_dur = end - start
            transcription_words.append(word_text)

            # Analyze canonical syllable structure
            try:
                syl_res = self.syllable_engine.analyze(word_text)
            except Exception:
                syl_res = None

            if syl_res is None or syl_res.count <= 1:
                # MONOSYLLABLE POLICY: K=1 bypasses LexiRep
                single_syl = LexiRepSyllable(
                    index=0,
                    text=word_text,
                    start=start,
                    end=end,
                    duration=w_dur,
                    stress=None,
                    contrast_score=None,
                    confidence=None,
                )
                structured_words.append(
                    LexiRepWord(
                        word=word_text,
                        start=start,
                        end=end,
                        duration=w_dur,
                        is_polysyllabic=False,
                        syllable_count=1,
                        whistress_stressed=bool(whistress_stress),
                        lexirep_stressed_syllable=None,
                        syllables=[single_syl],
                        syllable_source=syl_res.source if syl_res else "fallback",
                    )
                )
            else:
                # POLYSYLLABIC WORD: partition into K equal-duration intervals
                K = syl_res.count
                try:
                    intervals = partition_syllable_intervals(start, end, syl_res.syllables)
                except Exception:
                    intervals = [SyllableInterval(0, word_text, start, end)]
                    K = 1

                if K >= 2:
                    word_syl_objs: List[LexiRepSyllable] = []
                    current_group: List[int] = []

                    for syl_idx, s_int in enumerate(intervals):
                        word_syl_objs.append(
                            LexiRepSyllable(
                                index=syl_idx,
                                text=s_int.syllable_text,
                                start=s_int.start,
                                end=s_int.end,
                                duration=s_int.duration,
                            )
                        )
                        poly_intervals_flat.append((s_int.start, s_int.end))
                        current_group.append(global_syl_counter)
                        global_syl_counter += 1

                    poly_word_groupings.append(current_group)
                    poly_word_indices.append(len(structured_words))
                    poly_word_texts.append(word_text)

                    structured_words.append(
                        LexiRepWord(
                            word=word_text,
                            start=start,
                            end=end,
                            duration=w_dur,
                            is_polysyllabic=True,
                            syllable_count=K,
                            whistress_stressed=bool(whistress_stress),
                            lexirep_stressed_syllable=None,
                            syllables=word_syl_objs,
                            syllable_source=syl_res.source if syl_res else "fallback",
                        )
                    )
                else:
                    single_syl = LexiRepSyllable(
                        index=0,
                        text=word_text,
                        start=start,
                        end=end,
                        duration=w_dur,
                    )
                    structured_words.append(
                        LexiRepWord(
                            word=word_text,
                            start=start,
                            end=end,
                            duration=w_dur,
                            is_polysyllabic=False,
                            syllable_count=1,
                            whistress_stressed=bool(whistress_stress),
                            lexirep_stressed_syllable=None,
                            syllables=[single_syl],
                            syllable_source="fallback",
                        )
                    )

        stats["syllable_engine_time_s"] = time.perf_counter() - t0

        # ---------------------------------------------------------------------
        # 2. Full-Utterance Wav2Vec2 Feature Extraction (EXACTLY ONCE)
        # ---------------------------------------------------------------------
        t0 = time.perf_counter()
        hidden_states = self.feature_extractor.extract(audio_array, sample_rate=sr)
        wav2vec_frames = hidden_states.shape[0]
        stats["wav2vec_extraction_time_s"] = time.perf_counter() - t0

        # ---------------------------------------------------------------------
        # 3. Syllable Frame Pooling & Compact Global Matrix Construction
        # ---------------------------------------------------------------------
        t0 = time.perf_counter()
        num_poly_words = len(poly_word_groupings)
        total_syllables_pooled = len(poly_intervals_flat)
        linguistic_compliance = True

        if num_poly_words > 0 and total_syllables_pooled > 0:
            X_poly, _ = pool_syllables(hidden_states, poly_intervals_flat, sample_rate=sr)
            stats["pooling_time_s"] = time.perf_counter() - t0

            # -----------------------------------------------------------------
            # 4. LexiRep Prediction on Compact Global Feature Matrix
            # -----------------------------------------------------------------
            t0 = time.perf_counter()
            inference_result = self.lexirep_service.predict(
                language=lang,
                features=X_poly,
                word_boundaries=poly_word_groupings,
                word_texts=poly_word_texts,
            )
            stats["lexirep_inference_time_s"] = time.perf_counter() - t0
            linguistic_compliance = inference_result.linguistic_compliance

            # Map predictions back onto the structured words hierarchy
            for p_idx, pred_word in enumerate(inference_result.words):
                target_w_idx = poly_word_indices[p_idx]
                word_obj = structured_words[target_w_idx]
                group_offset = poly_word_groupings[p_idx][0]
                local_stressed_idx = pred_word.stressed_syllable_index - group_offset
                word_obj.lexirep_stressed_syllable = local_stressed_idx

                for s_idx, syl_pred in enumerate(pred_word.syllables):
                    syl_obj = word_obj.syllables[s_idx]
                    syl_obj.stress = syl_pred.stress
                    syl_obj.contrast_score = round(float(syl_pred.contrast_score), 4)
                    syl_obj.confidence = round(float(syl_pred.confidence), 4)
        else:
            stats["pooling_time_s"] = 0.0
            stats["lexirep_inference_time_s"] = 0.0

        stats["total_pipeline_time_s"] = time.perf_counter() - t_start

        return LexiRepPipelineResult(
            nativity=lang,
            transcription=" ".join(transcription_words),
            audio_duration=audio_duration,
            words=structured_words,
            linguistic_compliance=linguistic_compliance,
            num_polysyllabic_words=num_poly_words,
            total_syllables_pooled=total_syllables_pooled,
            wav2vec_frames=wav2vec_frames,
            execution_stats=stats,
        )

    def process(
        self,
        audio: Union[bytes, np.ndarray, str],
        nativity: str,
        sample_rate: int = 16000,
    ) -> LexiRepPipelineResult:
        """
        Executes the full LexiRep analysis pipeline on input audio.
        """
        lang = self._validate_nativity(nativity)
        stats: Dict[str, float] = {}

        # 1. Audio Preprocessing
        t0 = time.perf_counter()
        if isinstance(audio, str):
            with open(audio, "rb") as f:
                audio_bytes = f.read()
            audio_dict = whistress_service.process_audio(audio_bytes)
            audio_array = audio_dict["array"]
            sr = audio_dict["sampling_rate"]
        elif isinstance(audio, bytes):
            audio_bytes = audio
            audio_dict = whistress_service.process_audio(audio_bytes)
            audio_array = audio_dict["array"]
            sr = audio_dict["sampling_rate"]
        elif isinstance(audio, np.ndarray):
            audio_array = audio.astype(np.float32)
            if audio_array.ndim > 1:
                audio_array = audio_array.mean(axis=-1)
            sr = sample_rate
            wav_io = io.BytesIO()
            sf.write(wav_io, audio_array, sr, format="WAV")
            audio_bytes = wav_io.getvalue()
        else:
            raise TypeError(f"Unsupported audio type: {type(audio).__name__}")
        stats["audio_prep_time_s"] = time.perf_counter() - t0

        # 2. Whisper Transcription & Acoustic Word Timestamp Realignment
        t0 = time.perf_counter()
        whistress_service.load_model()
        raw_words = whistress_service.client.predict(
            audio={"sampling_rate": sr, "array": audio_array},
            transcription=None,
            return_pairs=True,
        )
        t_whisper = time.perf_counter() - t0
        stats["whisper_time_s"] = t_whisper

        t0_realign = time.perf_counter()
        pitch_times, pitch_f0 = extract_pitch(audio_array, sr=sr)
        realigned_words, _ = realign_word_timestamps(
            raw_words, pitch_times, pitch_f0, audio_array, sr
        )
        t_realign = time.perf_counter() - t0_realign
        stats["realign_time_s"] = t_realign
        stats["whisper_realign_time_s"] = t_whisper + t_realign

        return self.analyze_realigned_words(
            audio_array=audio_array,
            sr=sr,
            realigned_words=realigned_words,
            nativity=lang,
            execution_times=stats,
        )

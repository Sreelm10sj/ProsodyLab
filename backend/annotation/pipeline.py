import os
import sys
import json
import argparse
import time

# Ensure we can import from backend and repo root
_backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_root_dir = os.path.dirname(_backend_dir)
for _p in [_root_dir, _backend_dir]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from whistress_service import service
from pitch_analysis import compute_pause_before, build_pitch_contour

from typing import Optional, Any, Dict, List

_lexirep_pipeline = None

def get_lexirep_pipeline():
    global _lexirep_pipeline
    if _lexirep_pipeline is None:
        from backend.lexirep.pipeline import LexiRepPipeline
        _lexirep_pipeline = LexiRepPipeline()
    return _lexirep_pipeline


def run_annotation_pipeline(
    audio_bytes: bytes,
    filename: str,
    speaker_nativity: Optional[str] = None,
    lexirep_pipeline: Optional[Any] = None,
) -> dict:
    """
    Orchestrates the existing ProsodyLab models into a structured JSON annotation.
    Optionally enriches words with LexiRep syllable-level lexical stress when
    speaker_nativity ("GER" or "ITA") is provided.
    """
    t_start = time.time()
    
    from pitch_analysis import PAUSE_ANNOTATION_THRESHOLD_S

    SUPPORTED_LANGUAGES = {"GER", "ITA"}
    nativity_clean = None
    if speaker_nativity is not None:
        if not isinstance(speaker_nativity, str) or not speaker_nativity.strip():
            raise ValueError("speaker_nativity must be a non-empty string or None")
        nativity_clean = speaker_nativity.strip().upper()
        if nativity_clean not in SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Invalid speaker_nativity: '{speaker_nativity}'. Must be one of {sorted(list(SUPPORTED_LANGUAGES))}"
            )

    # Run the core ML models and extraction
    pairs, pauses, phrases, pitch_times, pitch_f0, pitch_stylized, k_array, P_val, audio_array, sr = service.predict_with_intonation(audio_bytes)
    
    # Calculate duration
    duration = len(audio_array) / sr if sr else 0.0

    # Execute LexiRep if speaker_nativity was explicitly specified
    lexirep_result = None
    lexirep_error = None
    if nativity_clean:
        pipeline = lexirep_pipeline or get_lexirep_pipeline()
        try:
            lexirep_result = pipeline.analyze_realigned_words(
                audio_array=audio_array,
                sr=sr,
                realigned_words=pairs,
                nativity=nativity_clean,
            )
        except Exception as e:
            import logging
            logging.error(f"LexiRep analysis failed during annotation: {e}", exc_info=True)
            lexirep_error = str(e)
            lexirep_result = None
    
    words_out = []
    pauses_out = []
    timeline_out = []
    
    transcription_text = " ".join(w for w, *_ in pairs)
    
    for idx, (word, stress, start, end) in enumerate(pairs):
        start = float(start)
        end = float(end)
        
        # 1. Check for pause before this word
        pause_before = pauses[idx - 1] if idx > 0 else None
        
        if pause_before is not None and pause_before["duration"] > PAUSE_ANNOTATION_THRESHOLD_S:
            pause_start = pause_before["start"]
            pause_end = pause_before["end"]
            pause_duration = pause_before["duration"]
            
            pauses_out.append({
                "start": pause_start,
                "end": pause_end,
                "duration": pause_duration
            })
            
            # Add to timeline
            timeline_out.append({
                "type": "pause",
                "start": pause_start,
                "end": pause_end,
                "duration": pause_duration
            })
            
        # 2. Add the word itself
        word_dict = {
            "id": idx + 1,
            "text": word,
            "start": start,
            "end": end,
            "stress": "stressed" if stress else "unstressed"
        }

        # 3. Add additive LexiRep fields if nativity was specified
        if nativity_clean is not None and lexirep_result is not None and idx < len(lexirep_result.words):
            lx_word = lexirep_result.words[idx]
            word_dict["is_polysyllabic"] = bool(lx_word.is_polysyllabic)
            word_dict["syllable_count"] = int(lx_word.syllable_count)
            word_dict["lexical_stress_index"] = (
                int(lx_word.lexirep_stressed_syllable)
                if lx_word.lexirep_stressed_syllable is not None
                else None
            )
            if lx_word.syllable_source:
                word_dict["syllable_source"] = str(lx_word.syllable_source)

            syllables_list = []
            for s in lx_word.syllables:
                s_dict = {
                    "index": int(s.index),
                    "text": str(s.text),
                    "start": round(float(s.start), 3),
                    "end": round(float(s.end), 3),
                    "duration": round(float(s.duration), 3),
                    "stress": int(s.stress) if s.stress is not None else None,
                }
                if s.contrast_score is not None:
                    s_dict["contrast_score"] = round(float(s.contrast_score), 4)
                if s.confidence is not None:
                    s_dict["confidence"] = round(float(s.confidence), 4)
                syllables_list.append(s_dict)

            word_dict["syllables"] = syllables_list

            # Optional compact model diagnostics for polysyllabic words
            if lx_word.is_polysyllabic and lx_word.lexirep_stressed_syllable is not None:
                st_syl = lx_word.syllables[lx_word.lexirep_stressed_syllable]
                word_dict["lexirep"] = {
                    "model": nativity_clean,
                    "contrast_score": round(float(st_syl.contrast_score), 4) if st_syl.contrast_score is not None else None,
                    "confidence": round(float(st_syl.confidence), 4) if st_syl.confidence is not None else None,
                }

        words_out.append(word_dict)
        
        # Add to timeline
        timeline_out.append({
            "type": "word",
            "id": idx + 1,
            "start": start,
            "end": end
        })
        
    # Sort timeline by start time to ensure chronological ordering
    timeline_out.sort(key=lambda x: x["start"])
    
    # Build Intonation
    intonation_out = []
    for p_idx, phrase in enumerate(phrases):
        # Handle cases where voicing was insufficient
        label = phrase["pitch"]["pattern"]
        slope = phrase["pitch"]["overall_slope"]
        
        item = {
            "phrase_id": p_idx + 1,
            "start": phrase["start"],
            "end": phrase["end"],
            "slope": slope,
            "label": label
        }
        if "word_indices" in phrase:
            item["word_indices"] = phrase["word_indices"]
            if phrase["word_indices"]:
                item["end_word_idx"] = phrase["word_indices"][-1]
        intonation_out.append(item)
        
    # Sort pauses and intonation by start time as well
    pauses_out.sort(key=lambda x: x["start"])
    intonation_out.sort(key=lambda x: x["start"])

    # Compile final JSON (pitch contour timeline omitted from public annotation JSON)
    metadata = {
        "audio_file": os.path.basename(filename),
        "duration": round(duration, 3)
    }
    if nativity_clean:
        metadata["speaker_nativity"] = nativity_clean
    if lexirep_error:
        metadata["lexirep_error"] = lexirep_error

    annotation = {
        "metadata": metadata,
        "transcription": {
            "text": transcription_text,
            "language": "en"
        },
        "words": words_out,
        "pauses": pauses_out,
        "intonation": intonation_out,
        "timeline": timeline_out,
    }
    
    return annotation

def main():
    parser = argparse.ArgumentParser(description="ProsodyLab Automated Annotation Pipeline")
    parser.add_argument("audio_path", help="Path to input audio file")
    parser.add_argument("--out-dir", default="output", help="Output directory")
    parser.add_argument("--speaker-nativity", choices=["GER", "ITA"], default=None, help="Optional LexiRep speaker nativity model ('GER' or 'ITA')")
    args = parser.parse_args()
    
    if not os.path.exists(args.audio_path):
        print(f"Error: Input file '{args.audio_path}' not found.")
        sys.exit(1)
        
    os.makedirs(args.out_dir, exist_ok=True)
    filename = os.path.basename(args.audio_path)
    base_name, _ = os.path.splitext(filename)
    
    json_path = os.path.join(args.out_dir, f"{base_name}_annotation.json")
    txt_path = os.path.join(args.out_dir, f"{base_name}_transcript.txt")
    
    print(f"Loading models...")
    service.load_model()
    
    with open(args.audio_path, "rb") as f:
        audio_bytes = f.read()
        
    print(f"Processing '{args.audio_path}'...")
    try:
        annotation = run_annotation_pipeline(audio_bytes, filename, speaker_nativity=args.speaker_nativity)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Annotation failed: {e}")
        sys.exit(1)
        
    # Write JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(annotation, f, indent=4)
        
    # Write TXT
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(annotation["transcription"]["text"])
        
    print("\nAnnotation completed successfully.")
    print(f"Audio: {filename}")
    print(f"Duration: {annotation['metadata']['duration']} s")
    print(f"Words: {len(annotation['words'])}")
    print(f"Pauses: {len(annotation['pauses'])}")
    print(f"Intonation phrases: {len(annotation['intonation'])}")
    if "speaker_nativity" in annotation["metadata"]:
        print(f"LexiRep Speaker Nativity: {annotation['metadata']['speaker_nativity']}")
    print(f"Output saved to {args.out_dir}")

if __name__ == "__main__":
    main()

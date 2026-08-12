import React, { useState, useEffect, useRef } from 'react';
import AudioInput from './components/AudioInput';
import TranscriptView from './components/TranscriptView';
import ProsodyPanel from './components/ProsodyPanel';
import AnnotationViewer from './components/AnnotationViewer';
import { checkHealth, transcribeAudio, generateAnnotation } from './api';

export default function App() {
  const [transcription, setTranscription] = useState('');
  const [words, setWords] = useState([]);
  const [phrases, setPhrases] = useState([]);
  const [pitchContour, setPitchContour] = useState([]);
  const [pitchStylized, setPitchStylized] = useState([]);
  const [utteranceMedianHz, setUtteranceMedianHz] = useState(null);
  // Prosody panel: true = expanded (default), false = collapsed.
  // Controls both the ProsodyPanel body and the inline phrase arrows.
  const [showProsody, setShowProsody] = useState(true);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [health, setHealth] = useState({ status: 'checking', device: '', model_loaded: false });
  const [audioUrl, setAudioUrl] = useState(null);
  
  const [audioBlob, setAudioBlob] = useState(null);
  const [audioFilename, setAudioFilename] = useState('');
  const [annotation, setAnnotation] = useState(null);
  const [isAnnotating, setIsAnnotating] = useState(false);
  const [annotationError, setAnnotationError] = useState('');
  
  const audioRef = useRef(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);

  useEffect(() => {
    checkHealth()
      .then(data => {
        setHealth(data);
      })
      .catch(() => {
        setHealth({ status: 'error', device: '', model_loaded: false });
      });
  }, []);

  const handleAudioSubmit = async (audioBlob, filename, url) => {
    setIsLoading(true);
    setError('');
    setTranscription('');
    setWords([]);
    setAudioUrl(url); 
    setAudioBlob(audioBlob);
    setAudioFilename(filename);
    setAnnotation(null);
    setAnnotationError('');

    try {
      const data = await transcribeAudio(audioBlob, filename);
      setTranscription(data.transcription);
      setWords(data.words);
      // phrases — Phase 1–3 field
      setPhrases(data.phrases ?? []);
      // pitch_contour / utterance_median_hz — Phase 4A fields
      // ?? guards ensure backward compatibility with pre-4A backend.
      setPitchContour(data.pitch_contour ?? []);
      setPitchStylized(data.pitch_stylized ?? []);
      setUtteranceMedianHz(data.utterance_median_hz ?? null);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const handleAudioClear = () => {
    setTranscription('');
    setWords([]);
    setPhrases([]);
    setPitchContour([]);
    setPitchStylized([]);
    setUtteranceMedianHz(null);
    setError('');
    setAudioUrl(null);
    setAudioBlob(null);
    setAudioFilename('');
    setAnnotation(null);
    setAnnotationError('');
    setCurrentTime(0);
    setDuration(0);
  };
  
  const handleSeek = (time) => {
    if (audioRef.current) {
      audioRef.current.currentTime = time;
      audioRef.current.play().catch(e => console.log('Playback prevented', e));
    }
  };

  const handleGenerateAnnotation = async () => {
    if (!audioBlob) return;
    setIsAnnotating(true);
    setAnnotationError('');
    
    try {
      const result = await generateAnnotation(audioBlob, audioFilename);
      setAnnotation(result);
    } catch (err) {
      setAnnotationError('Unable to generate annotation. Please try again.');
      console.error('Annotation error:', err);
    } finally {
      setIsAnnotating(false);
    }
  };

  return (
    <div className="min-h-screen bg-paper text-ink font-sans flex flex-col items-center py-12 px-4 sm:px-6 lg:px-8 selection:bg-highlight selection:text-ink">
      <div className="w-full max-w-3xl flex flex-col gap-8">
        
        <div className="flex flex-col sm:flex-row justify-between items-start sm:items-end border-b-4 border-ink pb-4">
          <div>
            <h1 className="text-3xl font-bold font-mono tracking-tighter text-ink uppercase">
              Prosody<span className="text-highlight bg-ink px-1 rounded-sm ml-1">Lab</span>
            </h1>
            <p className="font-mono text-xs uppercase tracking-widest text-ink-light mt-2 font-medium">
              Speech Prosody Analysis & Visualization
            </p>
          </div>
          
          <div className="mt-4 sm:mt-0 flex items-center gap-2 font-mono text-[10px] uppercase tracking-wider font-bold">
            <span className={`w-2 h-2 rounded-full ${
              health.status === 'ok' ? 'bg-emerald-500' : 
              health.status === 'error' ? 'bg-red-500' : 'bg-highlight animate-pulse'
            }`} />
            STATUS: {health.status === 'ok' ? `ONLINE [${health.device}]` : health.status === 'error' ? 'OFFLINE' : 'CHECKING...'}
          </div>
        </div>

        <AudioInput onAudioSubmit={handleAudioSubmit} onAudioClear={handleAudioClear} isLoading={isLoading} />

        {audioUrl && (
          <div className="w-full bg-ink text-paper p-4 border-2 border-ink shadow-[4px_4px_0_0_#F4C430]">
            <div className="font-mono text-xs text-highlight uppercase tracking-widest mb-2 flex justify-between font-bold">
              <span>Master Playback</span>
              <span>{currentTime.toFixed(1)}s / {duration > 0 ? duration.toFixed(1) : '0.0'}s</span>
            </div>
            <audio 
              ref={audioRef}
              src={audioUrl} 
              controls 
              className="w-full h-8 outline-none focus-visible:ring-2 focus-visible:ring-highlight focus-visible:ring-offset-2 focus-visible:ring-offset-ink filter invert brightness-110 contrast-125 grayscale rounded-none"
              onTimeUpdate={(e) => setCurrentTime(e.target.currentTime)}
              onLoadedMetadata={(e) => setDuration(e.target.duration)}
            />
          </div>
        )}

        {error && (
          <div className="p-4 bg-red-100 border-2 border-red-500 text-red-900 font-mono text-sm uppercase tracking-wide font-bold">
            <strong>System Error:</strong> {error}. Please try again.
          </div>
        )}

        <div className={`transition-opacity duration-300 ${isLoading ? 'opacity-50 pointer-events-none' : 'opacity-100'}`}>
          {isLoading && words.length === 0 ? (
            <div className="w-full bg-paper border border-ink shadow-[4px_4px_0_0_#1E2530] p-12 text-center relative">
               <div className="absolute top-0 left-0 right-0 h-2 bg-highlight animate-pulse border-b border-ink" />
               <p className="font-mono text-sm text-ink font-bold uppercase tracking-widest flex items-center justify-center gap-3">
                 <span className="w-3 h-3 bg-highlight border-2 border-ink animate-ping rounded-full"></span>
                 Listening for stress...
               </p>
            </div>
          ) : (
            <TranscriptView 
              transcription={transcription} 
              words={words}
              phrases={phrases}
              showPhraseMarkers={showProsody}
              currentTime={currentTime}
              duration={duration}
              onSeek={handleSeek}
            />
          )}
        </div>

        {/* ── Phase 4B: Prosody Panel (legend + pitch contour, collapsible) ── */}
        <ProsodyPanel
          isExpanded={showProsody}
          onToggle={() => setShowProsody(v => !v)}
          words={words}
          phrases={phrases}
          pitchContour={pitchContour}
          pitchStylized={pitchStylized}
          utteranceMedianHz={utteranceMedianHz}
          currentTime={currentTime}
          duration={duration}
          onSeek={handleSeek}
        />

        {audioUrl && !isLoading && words.length > 0 && (
          <div className="flex flex-col gap-4 items-center mb-8">
            <button
              onClick={handleGenerateAnnotation}
              disabled={isAnnotating}
              className={`w-full max-w-sm px-6 py-4 font-mono text-sm font-bold uppercase tracking-widest border-2 transition-all flex justify-center items-center gap-3
                ${isAnnotating 
                  ? 'bg-paper text-ink-light border-ink/30 cursor-not-allowed shadow-none' 
                  : 'bg-highlight text-ink border-ink shadow-[4px_4px_0_0_#1E2530] hover:translate-x-[2px] hover:translate-y-[2px] hover:shadow-[2px_2px_0_0_#1E2530]'
                }
              `}
            >
              {isAnnotating ? (
                <>
                  <span className="w-4 h-4 bg-ink/50 rounded-full animate-ping"></span>
                  Generating Annotation...
                </>
              ) : 'Generate Annotation'}
            </button>
            
            {annotationError && (
              <p className="text-red-500 font-mono text-sm font-bold">{annotationError}</p>
            )}
          </div>
        )}

        {annotation && !isLoading && (
          <AnnotationViewer 
            annotation={annotation} 
            filename={audioFilename}
          />
        )}

      </div>
    </div>
  );
}

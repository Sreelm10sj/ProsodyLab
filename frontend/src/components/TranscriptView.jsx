import React, { useState, useMemo } from 'react';
import { ToggleLeft, ToggleRight } from 'lucide-react';
import PhraseMarker from './PhraseIntonation';

export default function TranscriptView({ transcription, words, phrases = [], showPhraseMarkers = true, currentTime, duration, onSeek }) {
  const [showLinguist, setShowLinguist] = useState(false);

  // Build a map from the last word-index of each phrase → that phrase object.
  // This lets the render loop insert a <PhraseMarker> after the correct word
  // in O(1) without restructuring any of the existing word-level logic.
  const phraseMarkerMap = useMemo(() => {
    const map = new Map();
    (phrases ?? []).forEach((phrase) => {
      const indices = phrase.word_indices;
      if (indices && indices.length > 0) {
        const lastIdx = indices[indices.length - 1];
        map.set(lastIdx, phrase);
      }
    });
    return map;
  }, [phrases]);

  if (!words || words.length === 0) {
    return (
      <div className="w-full bg-paper border border-ink shadow-[4px_4px_0_0_#1E2530] p-12 text-center relative">
        <div className="absolute top-0 left-0 right-0 h-1 bg-ink/20" />
        <p className="font-mono text-sm text-ink-light uppercase tracking-widest leading-relaxed">
          Record or drop a clip to see which words carry the stress
        </p>
      </div>
    );
  }

  return (
    <div className="w-full bg-paper border border-ink shadow-[4px_4px_0_0_#1E2530] p-6 sm:p-10 relative">
      <div className="absolute top-0 left-0 right-0 h-2 bg-highlight border-b border-ink" />
      
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center mb-8 gap-4 border-b-2 border-ink pb-4 pt-2">
        <h2 className="font-mono text-sm uppercase tracking-widest text-ink font-bold">
          Transcript Output
        </h2>
        
        <button 
          onClick={() => setShowLinguist(!showLinguist)}
          aria-label="Toggle linguist view"
          className="flex items-center gap-2 font-mono text-xs text-ink hover:text-ink-light focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ink p-1 -mx-1 transition-colors"
        >
          {showLinguist ? (
            <ToggleRight className="w-6 h-6 text-ink fill-highlight" />
          ) : (
            <ToggleLeft className="w-6 h-6 text-ink/40" />
          )}
          <span className="uppercase tracking-wide font-medium">Linguist View (IPA)</span>
        </button>
      </div>

      <div className="font-serif text-[22px] leading-relaxed text-ink selection:bg-highlight selection:text-ink">
        {words.map((item, idx) => {
          const startTime = item.start || 0;
          const endTime = item.end || 0;
          const isPlaying = duration > 0 && currentTime > 0 && currentTime >= startTime && currentTime < endTime;
          
          let pauseElement = null;
          if (idx > 0) {
            const prevItem = words[idx - 1];
            const prevEndTime = prevItem.end || 0;

            // Prefer the authoritative acoustic pause object computed server-side.
            // It contains { start, end, duration } and accurately represents silences
            // even if they overlap Whisper word boundaries.
            let pauseDuration = 0;
            let src = 'Whisper';

            if (item.pause_before != null) {
              if (typeof item.pause_before === 'object' && item.pause_before.duration !== undefined) {
                pauseDuration = item.pause_before.duration;
              } else if (typeof item.pause_before === 'number') {
                pauseDuration = item.pause_before;
              }
              src = 'acoustic';
            } else {
              // Legacy fallback only if backend completely lacks the field
              pauseDuration = startTime - prevEndTime;
            }

            if (pauseDuration > 0) {
              pauseElement = (
                <span 
                  className="text-ink-light/50 mx-1 cursor-help tracking-widest" 
                  title={`Pause: ${pauseDuration.toFixed(2)}s (${src})`}
                >
                  ...
                </span>
              );
            }
          }
          
          return (
            <React.Fragment key={idx}>
              {pauseElement}
              <span className="word-tooltip-container inline-block">
                <span
                  onClick={() => onSeek(startTime)}
                  style={{ animationDelay: `${idx * 80}ms` }}
                  className={`
                    cursor-pointer px-[2px] mx-[1px]
                    ${isPlaying ? 'bg-ink text-paper highlight-sweep-override' : 'hover:bg-black/5'} 
                    ${item.stressed && !showLinguist ? 'highlight-sweep font-bold' : ''}
                    ${item.stressed && showLinguist ? 'font-bold' : ''}
                  `}
                >
                  {item.stressed && showLinguist && <span className="text-highlight mr-[1px] font-sans">ˈ</span>}
                  {item.word}
                </span>
                <span className="word-tooltip shadow-lg text-left min-w-[120px]">
                  <div className="mb-1 border-b border-white/20 pb-1 font-sans">
                    Stressed: <span className={item.stressed ? 'text-highlight font-bold' : ''}>{item.stressed ? 'Yes' : 'No'}</span>
                  </div>
                  <div className="text-[10px] text-white/70 font-mono mt-1">
                    [{startTime.toFixed(2)}s - {endTime.toFixed(2)}s]
                  </div>
                </span>
              </span>
              {/* Phrase intonation marker — rendered after the last word of each phrase */}
              {showPhraseMarkers && phraseMarkerMap.has(idx) && (
                <PhraseMarker phrase={phraseMarkerMap.get(idx)} />
              )}
            </React.Fragment>
          );
        })}
      </div>
    </div>
  );
}

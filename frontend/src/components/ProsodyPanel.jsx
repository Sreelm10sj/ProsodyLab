/**
 * ProsodyPanel.jsx
 * ================
 * Collapsible panel that surfaces the phrase-level and contour-level
 * prosody features together in one place.
 *
 * Collapsed  →  nothing shown here; phrase arrows also hidden in TranscriptView
 * Expanded   →  legend  +  pitch contour
 *
 * Accessibility
 * -------------
 * - Toggle button exposes aria-expanded / aria-controls.
 * - The collapsible body is marked inert when collapsed so keyboard users
 *   cannot tab into hidden interactive content (the PitchContour canvas).
 * - Chevron rotates 180° via CSS transform; motion is suppressed when
 *   prefers-reduced-motion: reduce is set.
 */

import React, { useRef, useEffect, useState } from 'react';
import { ChevronDown } from 'lucide-react';
import PitchContour from './PitchContour';

// ── Legend configuration ──────────────────────────────────────────────────────

const LEGEND_ITEMS = [
  {
    id: 'stress',
    label: 'Stress',
    desc: 'Primary lexical stress',
    preview: (
      <span className="legend-preview-stress" aria-hidden="true">
        word
      </span>
    ),
  },
  {
    id: 'pause',
    label: 'Pause',
    desc: 'Inter-word gap &gt; 200 ms',
    preview: (
      <span className="legend-preview-pause" aria-hidden="true">
        ...
      </span>
    ),
  },
  {
    id: 'raw',
    label: 'Raw F0',
    desc: 'Unprocessed F0 contour',
    preview: (
      <svg width="36" height="16" aria-hidden="true" style={{ display: 'block', overflow: 'visible' }}>
        <line x1="2" y1="8" x2="34" y2="8" stroke="rgba(30,37,48,0.3)" strokeWidth="1" />
      </svg>
    ),
  },
  {
    id: 'stylized',
    label: 'Stylized F0 — used for intonation classification',
    desc: 'DP-MAE piecewise polynomial stylization',
    preview: (
      <svg width="36" height="16" aria-hidden="true" style={{ display: 'block', overflow: 'visible' }}>
        <line x1="2" y1="8" x2="14" y2="8" stroke="#059669" strokeWidth="2.5" strokeLinecap="round" />
        <line x1="14" y1="8" x2="24" y2="8" stroke="#94a3b8" strokeWidth="2.5" strokeLinecap="round" />
        <line x1="24" y1="8" x2="34" y2="8" stroke="#3b82f6" strokeWidth="2.5" strokeLinecap="round" />
      </svg>
    ),
  },
  {
    id: 'rising',
    label: 'Rising Phrase',
    desc: 'Phrase slope ≥ +1.5 st/s',
    preview: (
      <span className="legend-arrow legend-arrow--rising" aria-hidden="true">
        ↗
      </span>
    ),
  },
  {
    id: 'falling',
    label: 'Falling Phrase',
    desc: 'Phrase slope ≤ −1.5 st/s',
    preview: (
      <span className="legend-arrow legend-arrow--falling" aria-hidden="true">
        ↘
      </span>
    ),
  },
  {
    id: 'level',
    label: 'Level Phrase',
    desc: 'Within ±1.5 st/s',
    preview: (
      <span className="legend-arrow legend-arrow--level" aria-hidden="true">
        →
      </span>
    ),
  },
];

// ── Component ─────────────────────────────────────────────────────────────────

/**
 * ProsodyPanel
 *
 * Props
 * -----
 * isExpanded        : boolean  — controlled by parent (App.jsx)
 * onToggle          : ()=>void — flip isExpanded in parent
 * words             : words array (passed through to PitchContour)
 * phrases           : phrases array (passed through to PitchContour)
 * pitchContour      : pitch_contour array
 * pitchStylized     : DP-MAE pitch_stylized array
 * utteranceMedianHz : utterance_median_hz (may be null)
 * currentTime       : current playback position (seconds)
 * duration          : audio duration (seconds)
 * onSeek(t)         : shared seek callback
 */
export default function ProsodyPanel({
  isExpanded,
  onToggle,
  words,
  phrases,
  pitchContour,
  pitchStylized,
  utteranceMedianHz,
  currentTime,
  duration,
  onSeek,
}) {
  const [viewMode, setViewMode] = useState('both'); // 'both' | 'raw' | 'stylized'
  const [pixelsPerSec, setPixelsPerSec] = useState(100);

  // Don't render the panel at all if there is no prosody data yet
  const hasProsody = pitchContour.length > 0 || phrases.length > 0;
  if (!hasProsody) return null;

  return (
    <div className="w-full bg-paper border border-ink shadow-[4px_4px_0_0_#1E2530] relative overflow-hidden">
      {/* Highlight accent bar */}
      <div className="absolute top-0 left-0 right-0 h-2 bg-highlight border-b border-ink" />

      {/* ── Toggle button ──────────────────────────────────────────────── */}
      <button
        id="prosody-toggle"
        onClick={onToggle}
        aria-expanded={isExpanded}
        aria-controls="prosody-details"
        className="
          w-full px-6 sm:px-10 pt-6 pb-3
          flex items-center justify-between
          text-left
          hover:bg-ink/[0.02] transition-colors duration-150
          focus-visible:outline-none focus-visible:ring-2
          focus-visible:ring-ink focus-visible:ring-inset
        "
      >
        <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1">
          <span className="font-mono text-sm uppercase tracking-widest text-ink font-bold">
            Prosody Details
          </span>
          {utteranceMedianHz && (
            <span className="font-mono text-[10px] text-ink-lighter uppercase tracking-wider select-none">
              Utterance median: {utteranceMedianHz.toFixed(1)} Hz
            </span>
          )}
        </div>

        {/* Chevron: rotates 180° when expanded */}
        <span
          className="ml-4 text-ink-light flex-shrink-0 prosody-chevron"
          style={{ transform: isExpanded ? 'rotate(180deg)' : 'rotate(0deg)' }}
          aria-hidden="true"
        >
          <ChevronDown className="w-4 h-4" strokeWidth={2.5} />
        </span>
      </button>

      {/* ── Collapsible body ───────────────────────────────────────────── */}
      <CollapsibleBody isExpanded={isExpanded}>
        {/* Legend & Controls */}
        <div className="px-6 sm:px-10 pt-1 pb-5 border-b border-ink/10 flex flex-col xl:flex-row justify-between items-start xl:items-end gap-4 flex-wrap">
          <div>
            <p className="font-mono text-[10px] uppercase tracking-widest text-ink-lighter mb-3 font-bold select-none">
              Legend
            </p>
            <div className="legend-grid" role="list">
              {LEGEND_ITEMS.map(item => (
                <div
                  key={item.id}
                  className="legend-item"
                  role="listitem"
                  title={item.desc}
                >
                  <div className="legend-swatch-wrap">{item.preview}</div>
                  <span className="legend-label">{item.label}</span>
                </div>
              ))}
            </div>
          </div>
          
          <div className="flex flex-col sm:flex-row gap-4">
            <div className="flex items-center gap-3 font-mono text-[10px] uppercase tracking-widest font-bold">
              <span className="text-ink-lighter select-none">Zoom:</span>
              <div className="flex bg-ink/5 p-1 rounded-sm gap-1">
                {[50, 100, 200, 400].map(zoom => (
                  <button 
                    key={zoom}
                    onClick={() => setPixelsPerSec(zoom)}
                    className={`px-2 py-1 rounded-sm transition-colors ${pixelsPerSec === zoom ? 'bg-white shadow-sm text-ink' : 'text-ink-light hover:text-ink'}`}
                  >
                    {zoom}
                  </button>
                ))}
              </div>
            </div>
            
            <div className="flex items-center gap-3 font-mono text-[10px] uppercase tracking-widest font-bold">
              <span className="text-ink-lighter select-none">View:</span>
              <div className="flex bg-ink/5 p-1 rounded-sm gap-1">
                <button 
                  onClick={() => setViewMode('both')}
                  className={`px-3 py-1 rounded-sm transition-colors ${viewMode === 'both' ? 'bg-white shadow-sm text-ink' : 'text-ink-light hover:text-ink'}`}
                >Both</button>
                <button 
                  onClick={() => setViewMode('raw')}
                  className={`px-3 py-1 rounded-sm transition-colors ${viewMode === 'raw' ? 'bg-white shadow-sm text-ink' : 'text-ink-light hover:text-ink'}`}
                >Raw</button>
                <button 
                  onClick={() => setViewMode('stylized')}
                  className={`px-3 py-1 rounded-sm transition-colors ${viewMode === 'stylized' ? 'bg-white shadow-sm text-ink' : 'text-ink-light hover:text-ink'}`}
                >Stylized</button>
              </div>
            </div>
          </div>
        </div>

        {/* Pitch contour */}
        {pitchContour.length > 0 && (
          <div className="px-3 pt-2 pb-4">
            <PitchContour
              words={words}
              phrases={phrases}
              pitchContour={pitchContour}
              pitchStylized={pitchStylized}
              viewMode={viewMode}
              pixelsPerSec={pixelsPerSec}
              utteranceMedianHz={utteranceMedianHz}
              currentTime={currentTime}
              duration={duration}
              onSeek={onSeek}
            />
          </div>
        )}
      </CollapsibleBody>
    </div>
  );
}

// ── CollapsibleBody ───────────────────────────────────────────────────────────

/**
 * Smooth height animation using the CSS grid-template-rows trick:
 *   collapsed → grid-template-rows: 0fr
 *   expanded  → grid-template-rows: 1fr
 *
 * The inner wrapper must have overflow:hidden for the clip to work.
 *
 * The `inert` attribute is set imperatively via a ref so that keyboard
 * navigation cannot reach hidden content when collapsed.
 */
function CollapsibleBody({ isExpanded, children }) {
  const innerRef = useRef(null);

  // Apply / remove the HTML `inert` attribute to block focus + pointer events
  useEffect(() => {
    const el = innerRef.current;
    if (!el) return;
    if (isExpanded) {
      el.removeAttribute('inert');
    } else {
      el.setAttribute('inert', '');
    }
  }, [isExpanded]);

  return (
    <div
      id="prosody-details"
      role="region"
      aria-labelledby="prosody-toggle"
      className={`prosody-panel-body${isExpanded ? ' prosody-panel-body--open' : ''}`}
    >
      <div ref={innerRef} className="prosody-panel-inner">
        {children}
      </div>
    </div>
  );
}

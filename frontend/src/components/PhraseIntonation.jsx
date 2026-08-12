import React from 'react';

// ---------------------------------------------------------------------------
// Per-pattern display metadata
// ---------------------------------------------------------------------------
const PATTERN_META = {
  rising: {
    arrow:     '↗',
    label:     'Rising',
    cssClass:  'phrase-marker--rising',
  },
  falling: {
    arrow:     '↘',
    label:     'Falling',
    cssClass:  'phrase-marker--falling',
  },
  level: {
    arrow:     '→',
    label:     'Level',
    cssClass:  'phrase-marker--level',
  },
  insufficient_voicing: {
    arrow:     '~',
    label:     'Insufficient voicing',
    cssClass:  'phrase-marker--insuf',
  },
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function fmtSlope(val) {
  if (val === null || val === undefined) return '—';
  const sign = val >= 0 ? '+' : '';
  return `${sign}${val.toFixed(2)} st/s`;
}

function fmtVoiced(val) {
  if (val === null || val === undefined) return '—';
  return `${(val * 100).toFixed(0)}%`;
}

function fmtTime(val) {
  if (val === null || val === undefined) return '—';
  return `${val.toFixed(2)} s`;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

/**
 * PhraseMarker
 *
 * Renders an inline intonation-contour badge immediately after the last word
 * of a phrase. On hover, a tooltip shows the detailed pitch features.
 *
 * Props
 * -----
 * phrase : object  — one entry from the /api/transcribe `phrases` array
 */
export default function PhraseMarker({ phrase }) {
  if (!phrase) return null;

  const pitch   = phrase.pitch_stylized ?? {};
  const pattern = pitch.pattern ?? 'insufficient_voicing';
  const meta    = PATTERN_META[pattern] ?? PATTERN_META.insufficient_voicing;

  return (
    <span className="phrase-marker-container" aria-label={`Intonation: ${meta.label}`}>
      {/* Inline arrow badge */}
      <span className={`phrase-marker ${meta.cssClass}`}>
        {meta.arrow}
      </span>

      {/* Hover tooltip */}
      <span className="phrase-tooltip" role="tooltip">
        <div className="phrase-tooltip-header">
          {meta.arrow}&nbsp;{meta.label}
        </div>
        <div className="phrase-tooltip-divider" />
        <div className="phrase-tooltip-row">
          <span className="phrase-tooltip-key">Overall slope</span>
          <span className="phrase-tooltip-val">{fmtSlope(pitch.overall_slope)}</span>
        </div>
        <div className="phrase-tooltip-row">
          <span className="phrase-tooltip-key">Terminal slope</span>
          <span className="phrase-tooltip-val">{fmtSlope(pitch.terminal_slope)}</span>
        </div>
        <div className="phrase-tooltip-row">
          <span className="phrase-tooltip-key">Voiced</span>
          <span className="phrase-tooltip-val">{fmtVoiced(pitch.voiced_fraction)}</span>
        </div>
        <div className="phrase-tooltip-divider" />
        <div className="phrase-tooltip-row">
          <span className="phrase-tooltip-key">Span</span>
          <span className="phrase-tooltip-val">
            {fmtTime(phrase.start)}–{fmtTime(phrase.end)}
          </span>
        </div>
      </span>
    </span>
  );
}

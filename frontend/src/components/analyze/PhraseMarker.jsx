/**
 * PhraseMarker — inline intonation arrow badge with hover tooltip.
 *
 * Real backend phrase shape (from build_phrase_response):
 *   phrase.pitch.pattern:       "rising" | "falling" | "level" | "insufficient_voicing"
 *   phrase.pitch.overall_slope: float (semitones/second)
 *   phrase.pitch.terminal_slope: float
 *   phrase.pitch.voiced_fraction: float 0–1
 *   phrase.start / phrase.end: float (seconds)
 *
 * Also supports old mock shape: phrase.intonation string field.
 */

const TYPES = {
  rising:               { symbol: '↗', color: '#2F9E6F', label: 'Rising', cssClass: 'phrase-marker--rise' },
  falling:              { symbol: '↘', color: '#4A7EC7', label: 'Falling', cssClass: 'phrase-marker--fall' },
  level:                { symbol: '→', color: '#8B96AA', label: 'Level', cssClass: 'phrase-marker--level' },
  insufficient_voicing: { symbol: '~', color: '#A8B0BE', label: 'Insufficient voicing', cssClass: 'phrase-marker--none' },
}

function fmtSlope(v) {
  if (v == null) return '—'
  return `${v >= 0 ? '+' : ''}${v.toFixed(2)} st/s`
}

function fmtPct(v) {
  if (v == null) return '—'
  return `${(v * 100).toFixed(0)}%`
}

export default function PhraseMarker({ phrase, id }) {
  // Support real shape (phrase.pitch.pattern), pipeline shape (phrase.label), and legacy mock shape (phrase.intonation)
  const pitch      = phrase.pitch ?? phrase.pitch_stylized ?? {}
  const rawPattern = pitch.pattern ?? phrase.label ?? phrase.intonation ?? 'insufficient_voicing'
  const pattern    = String(rawPattern).toLowerCase()
  const def        = TYPES[pattern] ?? TYPES.insufficient_voicing
  const tooltipId  = `phrase-tooltip-${id}`
  const slope      = pitch.overall_slope ?? phrase.slope

  return (
    <span
      className="phrase-marker-container"
      style={{ display: 'inline-flex', alignItems: 'center', marginLeft: 3, marginRight: 2, verticalAlign: 'middle' }}
    >
      <span
        className={`phrase-marker ${def.cssClass ?? ''}`}
        aria-label={`${def.label} intonation`}
        aria-describedby={tooltipId}
        role="img"
        tabIndex={0}
        style={{
          color: def.color,
          fontWeight: 700,
          fontSize: '0.82em',
          lineHeight: 1,
          cursor: 'help',
          background: `${def.color}18`,
          border: `1px solid ${def.color}44`,
          borderRadius: '3px',
          padding: '1px 4px',
        }}
      >
        {def.symbol}
      </span>

      {/* Hover tooltip */}
      <span
        id={tooltipId}
        className="phrase-tooltip"
        role="tooltip"
      >
        <span className="block font-bold" style={{ color: def.color }}>{def.label} Intonation</span>
        <span className="block">
          {phrase.start?.toFixed(2)}s → {phrase.end?.toFixed(2)}s
        </span>
        {slope != null && (
          <span className="block" style={{ color: '#A8B0BE' }}>
            slope: {fmtSlope(slope)}
          </span>
        )}
        {pitch.terminal_slope != null && (
          <span className="block" style={{ color: '#A8B0BE' }}>
            terminal: {fmtSlope(pitch.terminal_slope)}
          </span>
        )}
        {pitch.voiced_fraction != null && (
          <span className="block" style={{ color: '#A8B0BE' }}>
            voiced: {fmtPct(pitch.voiced_fraction)}
          </span>
        )}
      </span>
    </span>
  )
}

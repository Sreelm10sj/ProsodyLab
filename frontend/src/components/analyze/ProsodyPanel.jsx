import { useState, useId } from 'react'
import { ChevronDown, ZoomIn, ZoomOut } from 'lucide-react'
import PitchContour from './PitchContour.jsx'

/**
 * ProsodyPanel — collapsible panel housing legend + controls + PitchContour.
 *
 * Frozen ARIA (docs/frontend-audit.md §5.4):
 *   aria-expanded, aria-controls on toggle button
 *   inert on body when collapsed
 */

const LEGEND = [
  { color: 'var(--color-pitch-rise)',  label: 'Rising (↗)' },
  { color: 'var(--color-pitch-fall)',  label: 'Falling (↘)' },
  { color: 'var(--color-pitch-level)', label: 'Level (→)' },
  { color: 'var(--color-pitch-none)',  label: 'Unvoiced (~)' },
  { color: 'var(--color-stress)',      label: 'Median F0', dashed: true },
]

const VIEW_MODES = ['both', 'raw', 'stylized']

export default function ProsodyPanel({
  pitchContour,
  pitchStylized,
  utteranceMedianHz,
  currentTime,
  duration,
  onSeek,
}) {
  const [open, setOpen] = useState(true)
  const [zoom, setZoom]   = useState(1)
  const [mode, setMode]   = useState('both')
  const bodyId = useId()

  return (
    <div
      className="border-2"
      style={{
        borderColor: '#1C2128',
        background: '#F5F4F0',
        boxShadow: 'var(--shadow-2)',
      }}
    >
      {/* Card header / toggle */}
      <div
        className="border-b-2"
        style={{ borderColor: '#1C2128', background: '#1C2128' }}
      >
        <button
          id="prosody-panel-toggle"
          onClick={() => setOpen(o => !o)}
          aria-expanded={open}
          aria-controls={bodyId}
          className="w-full px-5 py-3 flex items-center justify-between"
          style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#F5F4F0' }}
        >
          <span className="font-mono font-semibold text-[11px] uppercase tracking-widest">
            Pitch Contour
          </span>
          <ChevronDown
            size={14}
            aria-hidden="true"
            className={`prosody-chevron ${open ? 'prosody-chevron--open' : ''}`}
          />
        </button>
      </div>

      {/* Collapsible body */}
      <div
        id={bodyId}
        className={`prosody-panel-body ${open ? 'prosody-panel-body--open' : ''}`}
        {...(!open && { inert: '' })}
      >
        <div className="prosody-panel-inner">
          {/* Legend row */}
          <div
            className="flex flex-wrap items-center gap-x-5 gap-y-2 px-5 py-3 border-b"
            style={{ borderColor: '#A8B0BE' }}
          >
            {LEGEND.map(item => (
              <span key={item.label} className="flex items-center gap-1.5">
                <span
                  style={{
                    display: 'inline-block',
                    width: 20, height: 2,
                    background: item.color,
                    borderTop: item.dashed ? `2px dashed ${item.color}` : undefined,
                    opacity: item.dashed ? 0.8 : 1,
                  }}
                  aria-hidden="true"
                />
                <span className="font-mono text-[10px]" style={{ color: '#4E5766' }}>
                  {item.label}
                </span>
              </span>
            ))}
          </div>

          {/* Controls row */}
          <div className="flex items-center gap-4 px-5 py-3 border-b" style={{ borderColor: '#A8B0BE' }}>
            {/* View mode */}
            <div className="flex" role="group" aria-label="View mode">
              {VIEW_MODES.map(m => (
                <button
                  key={m}
                  id={`view-mode-${m}`}
                  onClick={() => setMode(m)}
                  className="font-mono font-semibold text-[10px] uppercase tracking-wider px-3 py-1.5 transition-colors duration-100"
                  style={{
                    background: mode === m ? '#1C2128' : 'transparent',
                    color: mode === m ? '#F5F4F0' : '#4E5766',
                    border: '1px solid #A8B0BE',
                    borderRadius: 2,
                    marginLeft: m !== 'both' ? -1 : 0,
                    position: 'relative',
                    zIndex: mode === m ? 1 : 0,
                    cursor: 'pointer',
                  }}
                  aria-pressed={mode === m}
                >
                  {m}
                </button>
              ))}
            </div>

            {/* Zoom */}
            <div className="flex items-center gap-1 ml-auto" role="group" aria-label="Zoom">
              <button
                id="zoom-out"
                onClick={() => setZoom(z => Math.max(0.5, z - 0.25))}
                aria-label="Zoom out"
                className="p-1.5 transition-colors"
                style={{ color: '#4E5766', cursor: 'pointer', background: 'none', border: 'none' }}
              >
                <ZoomOut size={14} />
              </button>
              <span className="font-mono text-[10px]" style={{ color: '#4E5766' }}>
                {Math.round(zoom * 100)}%
              </span>
              <button
                id="zoom-in"
                onClick={() => setZoom(z => Math.min(4, z + 0.25))}
                aria-label="Zoom in"
                className="p-1.5 transition-colors"
                style={{ color: '#4E5766', cursor: 'pointer', background: 'none', border: 'none' }}
              >
                <ZoomIn size={14} />
              </button>
            </div>
          </div>

          {/* Canvas */}
          <div className="px-2 py-4">
            <PitchContour
              pitchContour={pitchContour}
              pitchStylized={pitchStylized}
              utteranceMedianHz={utteranceMedianHz}
              currentTime={currentTime}
              duration={duration}
              onSeek={onSeek}
              viewMode={mode}
              zoom={zoom}
            />
            <p className="font-mono text-[10px] mt-2 text-center" style={{ color: '#A8B0BE' }}>
              Use ← → keys to seek · hover for pitch value
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}

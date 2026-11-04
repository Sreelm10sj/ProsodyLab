import { useState, useEffect, useRef, useCallback, useMemo } from 'react'
import { Play, Pause, Volume2, VolumeX } from 'lucide-react'

/**
 * AudioWaveformPlayer — Custom dual-color audio waveform scrubber & player
 * Inspired by ProsodyLab reference screenshot 2.
 *
 * Visual design:
 *   - Vertical amplitude bars across the duration
 *   - Played portion: warm golden yellow (#FDB813)
 *   - Unplayed portion: slate gray (#CBD5E1)
 *   - Black scrubber needle cursor (#0F172A)
 *   - Control bar: circular play/pause, time slider, volume, speed cycle (1x)
 *
 * Props:
 *   audioUrl     — Object URL or remote URL of the audio file
 *   audioBlob    — File or Blob object (optional, used for real Web Audio peak extraction)
 *   duration     — Total duration in seconds
 *   currentTime  — Current playback time in seconds
 *   isPlaying    — Boolean play state
 *   onSeek       — Callback(seconds) when user seeks on waveform or slider
 *   onPlayToggle — Callback(shouldPlay)
 *   audioRef     — Forwarded ref to the underlying HTMLAudioElement
 */
export default function AudioWaveformPlayer({
  audioUrl,
  audioBlob,
  duration = 0,
  currentTime = 0,
  isPlaying = false,
  onSeek,
  onPlayToggle,
  audioRef: externalAudioRef,
}) {
  const localAudioRef = useRef(null)
  const audio = externalAudioRef?.current || localAudioRef.current
  const waveformRef = useRef(null)

  const [volume, setVolume] = useState(1.0)
  const [isMuted, setIsMuted] = useState(false)
  const [playbackRate, setPlaybackRate] = useState(1.0)
  const [waveformPeaks, setWaveformPeaks] = useState([])
  const [isDragging, setIsDragging] = useState(false)

  const safeDuration = duration > 0 ? duration : (audio?.duration || 1)
  const progressRatio = Math.max(0, Math.min(1, currentTime / safeDuration))

  // ── 1. Extract Real Audio Waveform Peaks (fallback to deterministic envelope) ──
  useEffect(() => {
    let isCancelled = false
    const numBars = 100

    async function extractPeaks() {
      if (audioBlob && window.AudioContext) {
        try {
          const arrayBuffer = await audioBlob.arrayBuffer()
          const audioCtx = new (window.AudioContext || window.webkitAudioContext)()
          const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer)
          const channelData = audioBuffer.getChannelData(0)
          const blockSize = Math.floor(channelData.length / numBars)
          const peaks = []

          for (let i = 0; i < numBars; i++) {
            let sum = 0
            const start = i * blockSize
            const end = Math.min(start + blockSize, channelData.length)
            for (let j = start; j < end; j++) {
              sum += Math.abs(channelData[j])
            }
            const avg = sum / (end - start || 1)
            // Normalize with a gentle speech curve
            peaks.push(Math.min(1, Math.max(0.12, avg * 3.5)))
          }
          if (!isCancelled) {
            setWaveformPeaks(peaks)
          }
          audioCtx.close().catch(() => {})
          return
        } catch (e) {
          // AudioContext decoding might fail in headless or unsupported environments; fallback below
        }
      }

      // Deterministic speech-like envelope generator
      const fallback = []
      for (let i = 0; i < numBars; i++) {
        const x = i / numBars
        // Natural speech cadence envelope with vowels and pauses
        const speechEnvelope = Math.sin(x * Math.PI) * 0.5 + 0.3
        const modulation = Math.sin(x * 24) * 0.2 + Math.cos(x * 12) * 0.15
        const noise = (Math.sin(i * 997) % 0.2)
        const height = Math.max(0.14, Math.min(0.95, speechEnvelope + modulation + noise))
        fallback.push(height)
      }
      if (!isCancelled) {
        setWaveformPeaks(fallback)
      }
    }

    extractPeaks()
    return () => {
      isCancelled = true
    }
  }, [audioBlob, audioUrl])

  // ── 2. Playback Speed Cycle ────────────────────────────────────────────────
  const speeds = [0.75, 1.0, 1.25, 1.5]
  const handleCycleSpeed = () => {
    const nextIdx = (speeds.indexOf(playbackRate) + 1) % speeds.length
    const nextRate = speeds[nextIdx]
    setPlaybackRate(nextRate)
    if (audio) {
      audio.playbackRate = nextRate
    }
  }

  // ── 3. Volume & Mute ───────────────────────────────────────────────────────
  const handleToggleMute = () => {
    if (!audio) return
    const nextMuted = !isMuted
    setIsMuted(nextMuted)
    audio.muted = nextMuted
  }

  // ── 4. Waveform Click & Drag to Seek ───────────────────────────────────────
  const handleSeekFromClientX = useCallback(
    (clientX) => {
      if (!waveformRef.current) return
      const rect = waveformRef.current.getBoundingClientRect()
      const clickX = Math.max(0, Math.min(rect.width, clientX - rect.left))
      const ratio = clickX / rect.width
      const targetTime = ratio * safeDuration
      if (onSeek) onSeek(targetTime)
      if (audio) audio.currentTime = targetTime
    },
    [safeDuration, onSeek, audio]
  )

  const handleMouseDown = (e) => {
    setIsDragging(true)
    handleSeekFromClientX(e.clientX)
  }

  useEffect(() => {
    const handleMouseMove = (e) => {
      if (isDragging) {
        handleSeekFromClientX(e.clientX)
      }
    }
    const handleMouseUp = () => {
      if (isDragging) setIsDragging(false)
    }
    if (isDragging) {
      window.addEventListener('mousemove', handleMouseMove)
      window.addEventListener('mouseup', handleMouseUp)
    }
    return () => {
      window.removeEventListener('mousemove', handleMouseMove)
      window.removeEventListener('mouseup', handleMouseUp)
    }
  }, [isDragging, handleSeekFromClientX])

  const formatTime = (seconds) => {
    if (!seconds || isNaN(seconds)) return '0:00'
    const mins = Math.floor(seconds / 60)
    const secs = (seconds % 60).toFixed(2)
    const [wholeSecs, frac] = secs.split('.')
    return `${mins}:${wholeSecs.padStart(2, '0')}`
  }

  const formatTotalTime = (seconds) => {
    if (!seconds || isNaN(seconds)) return '0.00'
    return Number(seconds).toFixed(2)
  }

  return (
    <div className="w-full bg-white rounded-2xl border border-slate-200/90 shadow-sm p-6 space-y-5 select-none">
      
      {/* ── Waveform Track with Timestamps ──────────────────────────────────── */}
      <div className="space-y-2">
        {/* Waveform Canvas Area */}
        <div
          ref={waveformRef}
          onMouseDown={handleMouseDown}
          className="relative h-20 w-full flex items-center justify-between gap-[3px] cursor-pointer group py-2"
          role="slider"
          aria-label="Audio waveform scrubber"
          aria-valuemin={0}
          aria-valuemax={safeDuration}
          aria-valuenow={currentTime}
          tabIndex={0}
          onKeyDown={(e) => {
            if (e.key === 'ArrowLeft') {
              const target = Math.max(0, currentTime - 0.5)
              onSeek?.(target)
              if (audio) audio.currentTime = target
            } else if (e.key === 'ArrowRight') {
              const target = Math.min(safeDuration, currentTime + 0.5)
              onSeek?.(target)
              if (audio) audio.currentTime = target
            }
          }}
        >
          {/* Vertical Amplitude Bars */}
          {waveformPeaks.map((peak, idx) => {
            const barRatio = idx / waveformPeaks.length
            const isPlayed = barRatio <= progressRatio
            const heightPercent = Math.max(12, Math.round(peak * 100))

            return (
              <div
                key={idx}
                className="flex-1 rounded-full transition-colors duration-75"
                style={{
                  height: `${heightPercent}%`,
                  minHeight: '6px',
                  backgroundColor: isPlayed ? '#FDB813' : '#CBD5E1',
                }}
              />
            )
          })}

          {/* Scrubber Playhead Line (Black vertical needle) */}
          <div
            className="absolute top-0 bottom-0 w-[2.5px] bg-[#0F172A] rounded-full pointer-events-none shadow-sm transition-[left] duration-75 z-10"
            style={{ left: `${progressRatio * 100}%` }}
          >
            {/* Playhead Top Knob */}
            <div className="absolute -top-1 -left-[3px] w-2.5 h-2.5 rounded-full bg-[#0F172A]" />
          </div>
        </div>

        {/* Timestamps Row */}
        <div className="flex justify-between items-center text-xs font-mono font-medium text-slate-400 px-0.5">
          <span>{formatTime(currentTime)}</span>
          <span>{formatTotalTime(safeDuration)}</span>
        </div>
      </div>

      {/* ── Player Controls Row ─────────────────────────────────────────────── */}
      <div className="flex items-center justify-between gap-4 pt-1">
        
        {/* Play/Pause Main Circular Button */}
        <button
          type="button"
          onClick={() => {
            if (!audio) return
            if (isPlaying) {
              audio.pause()
              onPlayToggle?.(false)
            } else {
              audio.play().catch(() => {})
              onPlayToggle?.(true)
            }
          }}
          className="w-11 h-11 rounded-full flex items-center justify-center bg-[#0F172A] hover:bg-slate-800 text-white transition-transform active:scale-95 shadow-sm flex-shrink-0"
          aria-label={isPlaying ? 'Pause audio' : 'Play audio'}
        >
          {isPlaying ? (
            <Pause size={18} fill="currentColor" />
          ) : (
            <Play size={18} fill="currentColor" className="ml-0.5" />
          )}
        </button>

        {/* Scrubber Progress Slider */}
        <div className="flex-1 flex items-center gap-3">
          <input
            type="range"
            min={0}
            max={safeDuration || 1}
            step={0.01}
            value={currentTime}
            onChange={(e) => {
              const val = parseFloat(e.target.value)
              onSeek?.(val)
              if (audio) audio.currentTime = val
            }}
            className="w-full accent-[#0F172A] h-1.5 bg-slate-200 rounded-lg appearance-none cursor-pointer"
            aria-label="Time scrubber"
          />
          <span className="text-xs font-mono text-slate-500 font-medium flex-shrink-0 min-w-[70px]">
            {formatTime(currentTime)} / {formatTotalTime(safeDuration)}
          </span>
        </div>

        {/* Volume & Speed Actions */}
        <div className="flex items-center gap-2 flex-shrink-0">
          {/* Mute Toggle */}
          <button
            type="button"
            onClick={handleToggleMute}
            className="w-8 h-8 rounded-md flex items-center justify-center text-slate-500 hover:text-slate-800 hover:bg-slate-100 transition-colors"
            title={isMuted ? 'Unmute' : 'Mute'}
            aria-label={isMuted ? 'Unmute' : 'Mute'}
          >
            {isMuted ? <VolumeX size={17} /> : <Volume2 size={17} />}
          </button>

          {/* Speed Pill (1x) */}
          <button
            type="button"
            onClick={handleCycleSpeed}
            className="px-2.5 py-1 rounded-md text-xs font-mono font-bold text-slate-700 bg-slate-100 hover:bg-slate-200 border border-slate-200/80 transition-colors"
            title="Playback speed"
            aria-label={`Playback speed ${playbackRate}x`}
          >
            {playbackRate}x
          </button>
        </div>

      </div>

      {/* Embedded Real Audio Element (when external ref is not provided) */}
      {!externalAudioRef && (
        <audio
          ref={localAudioRef}
          src={audioUrl}
          className="sr-only"
          preload="metadata"
        />
      )}
    </div>
  )
}

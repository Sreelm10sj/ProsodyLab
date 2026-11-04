import { useEffect, useRef, useCallback } from 'react'
import { C } from '../../tokens.js'

/**
 * OrbCanvas — 2D canvas audio-reactive blob.
 *
 * Receives analyserNode from Recorder.jsx. Draws a 64-point Catmull-Rom
 * closed spline whose control points are displaced by smoothed audio data.
 *
 * Audio → visual mappings (from design-plan.md §3B):
 *   RMS amplitude        → scaleY = 1 + (rms/255) * 0.6
 *   Dominant freq bin    → scaleX = 1 - (dominantBinRatio * 0.25)
 *   Rate-of-change       → border width lerp 2px → 4px
 *   Silence              → retreat to circle over 600ms
 *   Loud burst           → 3-4 offset ripple rings, 400ms decay
 *
 * Design: monochromatic (paper fill, ink stroke). No glows, no color shifts.
 * Fallback: if analyserNode is null, renders static circle.
 */

const N = 64        // control points
const SIZE = 220    // canvas logical size (px)
const BASE_R = 72   // base radius

function catmullRomClosed(ctx, pts) {
  const n = pts.length
  ctx.beginPath()
  for (let i = 0; i < n; i++) {
    const p0 = pts[(i - 1 + n) % n]
    const p1 = pts[i]
    const p2 = pts[(i + 1) % n]
    const p3 = pts[(i + 2) % n]
    const steps = 8
    for (let t = 0; t < steps; t++) {
      const s = t / steps
      const s2 = s * s, s3 = s2 * s
      const x = 0.5 * (
        2 * p1[0] + (-p0[0] + p2[0]) * s +
        (2*p0[0] - 5*p1[0] + 4*p2[0] - p3[0]) * s2 +
        (-p0[0] + 3*p1[0] - 3*p2[0] + p3[0]) * s3
      )
      const y = 0.5 * (
        2 * p1[1] + (-p0[1] + p2[1]) * s +
        (2*p0[1] - 5*p1[1] + 4*p2[1] - p3[1]) * s2 +
        (-p0[1] + 3*p1[1] - 3*p2[1] + p3[1]) * s3
      )
      i === 0 && t === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)
    }
  }
  ctx.closePath()
}

export default function OrbCanvas({ analyserNode, isRecording }) {
  const canvasRef = useRef(null)
  const rafRef    = useRef(null)
  const stateRef  = useRef({
    scaleX: 1, scaleY: 1,
    borderW: 2, prevRms: 0,
    ripples: [],
    noiseOffsets: Array.from({ length: N }, (_, i) => i * 0.15),
  })

  // Build base ring of N points
  const buildPts = useCallback((cx, cy, rx, ry, noise) => {
    return Array.from({ length: N }, (_, i) => {
      const angle = (i / N) * Math.PI * 2
      const nx = Math.sin(i * 0.8 + noise) * 6  // organic perturb
      const ny = Math.cos(i * 0.9 + noise) * 5
      return [
        cx + Math.cos(angle) * rx + nx,
        cy + Math.sin(angle) * ry + ny,
      ]
    })
  }, [])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const dpr = Math.min(window.devicePixelRatio, 2)
    canvas.width  = SIZE * dpr
    canvas.height = SIZE * dpr
    const ctx = canvas.getContext('2d')
    ctx.scale(dpr, dpr)

    const s = stateRef.current
    const freqData = analyserNode ? new Uint8Array(analyserNode.frequencyBinCount) : null

    function lerp(a, b, t) { return a + (b - a) * t }

    function draw(ts) {
      const cx = SIZE / 2, cy = SIZE / 2
      ctx.clearRect(0, 0, SIZE, SIZE)

      let rms = 0, dominantBinRatio = 0

      if (analyserNode && freqData) {
        analyserNode.getByteFrequencyData(freqData)
        // RMS
        let sum = 0
        for (let i = 0; i < freqData.length; i++) sum += freqData[i] ** 2
        rms = Math.sqrt(sum / freqData.length)

        // Dominant frequency bin index (normalised)
        let maxVal = 0, maxIdx = 0
        for (let i = 0; i < freqData.length; i++) {
          if (freqData[i] > maxVal) { maxVal = freqData[i]; maxIdx = i }
        }
        dominantBinRatio = maxIdx / freqData.length

        // Loud burst → ripple
        const prevRms = s.prevRms
        if (rms - prevRms > 40 && s.ripples.length < 4) {
          s.ripples.push({ r: BASE_R, opacity: 0.4, ts })
        }
        s.prevRms = rms
      }

      const isSilent = rms < 8
      const targetScaleX = isSilent ? 1 : 1 - dominantBinRatio * 0.25
      const targetScaleY = isSilent ? 1 : 1 + (rms / 255) * 0.6
      const targetBW = isSilent ? 2 : lerp(2, 4, Math.min(rms / 80, 1))

      // Smooth approach rates
      const silentRate = 0.02  // slow retreat when silent
      const activeRate = 0.10  // faster response when active
      const rate = isSilent ? silentRate : activeRate

      s.scaleX   = lerp(s.scaleX,   targetScaleX, rate)
      s.scaleY   = lerp(s.scaleY,   targetScaleY, rate)
      s.borderW  = lerp(s.borderW,  targetBW,     rate)

      const noiseT = ts * 0.0008
      const pts = buildPts(cx, cy, BASE_R * s.scaleX, BASE_R * s.scaleY, noiseT)

      // Draw ripples
      for (let i = s.ripples.length - 1; i >= 0; i--) {
        const rip = s.ripples[i]
        const age = (ts - rip.ts) / 400  // 0 → 1 over 400ms
        if (age >= 1) { s.ripples.splice(i, 1); continue }
        rip.r = BASE_R + age * 30
        rip.opacity = 0.4 * (1 - age)
        ctx.beginPath()
        ctx.arc(cx, cy, rip.r, 0, Math.PI * 2)
        ctx.strokeStyle = C.ink
        ctx.lineWidth = 1
        ctx.globalAlpha = rip.opacity
        ctx.stroke()
        ctx.globalAlpha = 1
      }

      // Draw blob
      catmullRomClosed(ctx, pts)
      ctx.fillStyle = C.paper
      ctx.fill()
      ctx.strokeStyle = C.ink
      ctx.lineWidth = s.borderW
      ctx.stroke()

      rafRef.current = requestAnimationFrame(draw)
    }

    rafRef.current = requestAnimationFrame(draw)

    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current)
    }
  }, [analyserNode, buildPts])

  return (
    <div className="flex flex-col items-center gap-4">
      {!analyserNode && (
        <p className="font-mono text-xs text-center" style={{ color: '#4E5766' }}>
          Microphone unavailable
        </p>
      )}
      <canvas
        ref={canvasRef}
        width={SIZE}
        height={SIZE}
        style={{ width: SIZE, height: SIZE }}
        aria-hidden="true"
      />
    </div>
  )
}

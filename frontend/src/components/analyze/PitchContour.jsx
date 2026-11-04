import { useEffect, useRef, useCallback } from 'react'
import { C } from '../../tokens.js'

/**
 * PitchContour — two-layer canvas (static + RAF playhead).
 *
 * Accepts two data formats:
 *   A) Backend format:  Array<{time, semitone, K?, P?}>   (current backend)
 *   B) Object format:   {times: number[], hz: number[]}   (forward compat)
 *
 * Format A is normalised to B internally using utteranceMedianHz.
 *   hz = medianHz * 2^(semitone/12)
 *   unvoiced (semitone === null) -> hz = 0
 */

const ML = 40, MR = 20, MT = 12, MB = 24

/**
 * Normalise either data format to {times[], hz[]}.
 * Returns null if data is absent or empty.
 */
function normalise(data, medianHz) {
  if (!data) return null

  // Format B: already parallel arrays
  if (!Array.isArray(data) && data.times && data.hz) {
    return data.times.length > 0 ? data : null
  }

  // Format A: array of {time, semitone?} objects
  if (Array.isArray(data) && data.length > 0) {
    const times = []
    const hz    = []
    for (const pt of data) {
      times.push(pt.time ?? 0)
      if (pt.semitone != null && medianHz > 0) {
        hz.push(medianHz * Math.pow(2, pt.semitone / 12))
      } else {
        hz.push(0)  // unvoiced
      }
    }
    return { times, hz }
  }

  return null
}

export default function PitchContour({
  pitchContour,
  pitchStylized,
  utteranceMedianHz,
  currentTime,
  duration,
  onSeek,
  viewMode = 'both',
  zoom = 1,
}) {
  const staticRef    = useRef(null)
  const dynamicRef   = useRef(null)
  const tooltipRef   = useRef(null)
  const containerRef = useRef(null)
  const currentTimeRef = useRef(currentTime)
  const rafRef = useRef(null)

  // Normalise once per data change — avoids repeated per-frame conversion
  const medHz = utteranceMedianHz ?? 200
  const contourNorm   = normalise(pitchContour,   medHz)
  const stylizedNorm  = normalise(pitchStylized,  medHz)

  // Keep refs for RAF access without re-closuring
  const contourRef  = useRef(contourNorm)
  const stylizedRef = useRef(stylizedNorm)
  useEffect(() => { contourRef.current  = contourNorm },  [contourNorm])
  useEffect(() => { stylizedRef.current = stylizedNorm }, [stylizedNorm])
  useEffect(() => { currentTimeRef.current = currentTime }, [currentTime])

  // ── Static layer ──────────────────────────────────────────────────────────
  const drawStatic = useCallback(() => {
    const canvas = staticRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    const dpr = Math.min(window.devicePixelRatio, 2)
    const W = canvas.width / dpr
    const H = canvas.height / dpr
    const plotW = W - ML - MR
    const plotH = H - MT - MB

    ctx.clearRect(0, 0, W, H)
    ctx.fillStyle = C.paper
    ctx.fillRect(0, 0, W, H)

    // Pick primary data for axis scaling
    const data = viewMode === 'stylized'
      ? (stylizedNorm ?? contourNorm)
      : contourNorm

    if (!data || data.times.length === 0) {
      ctx.fillStyle = C.inkFaint
      ctx.font = `12px 'IBM Plex Mono', monospace`
      ctx.textAlign = 'center'
      ctx.fillText('No pitch data', W / 2, H / 2)
      return
    }

    const voiced = data.hz.filter(v => v > 0)
    if (voiced.length === 0) {
      ctx.fillStyle = C.inkFaint
      ctx.font = `12px 'IBM Plex Mono', monospace`
      ctx.textAlign = 'center'
      ctx.fillText('No voiced frames', W / 2, H / 2)
      return
    }

    const maxHz = Math.max(...voiced, medHz * 1.5)
    const minHz = Math.max(0, Math.min(...voiced, medHz * 0.5) - 20)
    const maxT  = data.times[data.times.length - 1]

    // Horizontal grid + Hz labels
    ctx.lineWidth = 0.5
    ctx.globalAlpha = 0.4
    const hzStep = maxHz > 300 ? 50 : 25
    for (let h = Math.ceil(minHz / hzStep) * hzStep; h <= maxHz + hzStep; h += hzStep) {
      const y = MT + plotH - ((h - minHz) / (maxHz - minHz)) * plotH
      if (y < MT - 2 || y > MT + plotH + 2) continue
      ctx.strokeStyle = C.inkFaint
      ctx.beginPath(); ctx.moveTo(ML, y); ctx.lineTo(ML + plotW, y); ctx.stroke()
      ctx.globalAlpha = 0.6
      ctx.fillStyle = C.inkFaint
      ctx.font = `9px 'IBM Plex Mono', monospace`
      ctx.textAlign = 'right'
      ctx.fillText(`${h}Hz`, ML - 4, y + 3)
      ctx.globalAlpha = 0.4
    }
    ctx.globalAlpha = 1

    // Median dashed line
    if (medHz > 0) {
      const medY = MT + plotH - ((medHz - minHz) / (maxHz - minHz)) * plotH
      if (medY >= MT && medY <= MT + plotH) {
        ctx.strokeStyle = C.stress
        ctx.lineWidth = 1
        ctx.setLineDash([4, 4])
        ctx.beginPath(); ctx.moveTo(ML, medY); ctx.lineTo(ML + plotW, medY); ctx.stroke()
        ctx.setLineDash([])
        ctx.fillStyle = C.stress
        ctx.font = `9px 'IBM Plex Mono', monospace`
        ctx.textAlign = 'left'
        ctx.fillText(`med ${medHz.toFixed(0)}Hz`, ML + 4, medY - 3)
      }
    }

    // Time axis
    ctx.fillStyle = C.inkFaint
    ctx.font = `9px 'IBM Plex Mono', monospace`
    ctx.textAlign = 'center'
    const tickCount = Math.min(10, Math.floor(maxT) || 1)
    for (let i = 0; i <= tickCount; i++) {
      const t = (i / tickCount) * maxT
      const x = ML + (t / maxT) * plotW
      ctx.fillText(`${t.toFixed(1)}s`, x, MT + plotH + MB - 4)
    }

    // Draw a contour line
    function drawContour(pts, color) {
      ctx.strokeStyle = color
      ctx.lineWidth = 1.5
      ctx.lineJoin = 'round'
      let pen = false
      ctx.beginPath()
      for (let i = 0; i < pts.times.length; i++) {
        const h = pts.hz[i]
        if (h <= 0) { pen = false; continue }
        const x = ML + (pts.times[i] / maxT) * plotW
        const y = MT + plotH - ((h - minHz) / (maxHz - minHz)) * plotH
        if (y < MT - 4 || y > MT + plotH + 4) { pen = false; continue }
        pen ? ctx.lineTo(x, y) : ctx.moveTo(x, y)
        pen = true
      }
      ctx.stroke()
    }

    if ((viewMode === 'both' || viewMode === 'raw') && contourNorm) {
      ctx.globalAlpha = viewMode === 'both' ? 0.55 : 1
      drawContour(contourNorm, C.pitchFall)
      ctx.globalAlpha = 1
    }
    if ((viewMode === 'both' || viewMode === 'stylized') && stylizedNorm) {
      drawContour(stylizedNorm, C.pitchRise)
    }

    // Border
    ctx.strokeStyle = C.inkFaint; ctx.lineWidth = 1; ctx.globalAlpha = 0.3
    ctx.strokeRect(ML, MT, plotW, plotH)
    ctx.globalAlpha = 1

  }, [contourNorm, stylizedNorm, medHz, viewMode, zoom])

  // ── Dynamic layer (RAF playhead) ──────────────────────────────────────────
  useEffect(() => {
    const canvas = dynamicRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')

    function drawDynamic() {
      const dpr = Math.min(window.devicePixelRatio, 2)
      const W = canvas.width / dpr
      const H = canvas.height / dpr
      ctx.clearRect(0, 0, W, H)

      const data = contourRef.current ?? stylizedRef.current
      if (!data || duration <= 0) {
        rafRef.current = requestAnimationFrame(drawDynamic)
        return
      }

      const plotW = W - ML - MR
      const plotH = H - MT - MB
      const maxT  = data.times[data.times.length - 1] ?? duration
      const t = currentTimeRef.current
      const x = ML + (t / maxT) * plotW

      ctx.strokeStyle = C.stress; ctx.lineWidth = 1.5
      ctx.beginPath(); ctx.moveTo(x, MT); ctx.lineTo(x, MT + plotH); ctx.stroke()
      ctx.fillStyle = C.stress
      ctx.beginPath(); ctx.moveTo(x - 4, MT); ctx.lineTo(x + 4, MT); ctx.lineTo(x, MT + 7)
      ctx.closePath(); ctx.fill()

      rafRef.current = requestAnimationFrame(drawDynamic)
    }

    rafRef.current = requestAnimationFrame(drawDynamic)
    return () => { if (rafRef.current) cancelAnimationFrame(rafRef.current) }
  }, [contourNorm, stylizedNorm, duration])

  // ── Resize + redraw static ────────────────────────────────────────────────
  useEffect(() => {
    function resize() {
      const container = containerRef.current
      if (!container) return
      const dpr = Math.min(window.devicePixelRatio, 2)
      for (const canvas of [staticRef.current, dynamicRef.current]) {
        if (!canvas) continue
        const rect = container.getBoundingClientRect()
        const logW = rect.width * zoom
        const logH = 200
        canvas.width  = logW * dpr
        canvas.height = logH * dpr
        canvas.style.width  = `${logW}px`
        canvas.style.height = `${logH}px`
        const ctx = canvas.getContext('2d')
        ctx.setTransform(1, 0, 0, 1, 0, 0)
        ctx.scale(dpr, dpr)
      }
      drawStatic()
    }
    resize()
    const ro = new ResizeObserver(resize)
    if (containerRef.current) ro.observe(containerRef.current)
    return () => ro.disconnect()
  }, [drawStatic, zoom])

  useEffect(() => { drawStatic() }, [drawStatic])

  // ── Tooltip ───────────────────────────────────────────────────────────────
  function handleMouseMove(e) {
    const tooltip = tooltipRef.current
    const canvas  = staticRef.current
    if (!tooltip || !canvas) return

    const rect   = canvas.getBoundingClientRect()
    const dpr    = Math.min(window.devicePixelRatio, 2)
    const relX   = e.clientX - rect.left
    const relY   = e.clientY - rect.top
    const W      = canvas.width / dpr
    const plotW  = W - ML - MR

    const rawData      = contourNorm
    const stylizedData = stylizedNorm
    const data = rawData ?? stylizedData
    if (!data || plotW <= 0) return

    const maxT = data.times[data.times.length - 1] ?? 1
    const t    = ((relX - ML) / plotW) * maxT
    if (t < 0 || t > maxT) { tooltip.style.display = 'none'; return }

    const idx = data.times.findIndex(v => v >= t)
    if (idx < 0) { tooltip.style.display = 'none'; return }

    const h   = data.hz[idx]
    // K/P values only in stylized contour points (may be undefined)
    const rawPt  = (pitchContour  && Array.isArray(pitchContour))  ? pitchContour[Math.min(idx, pitchContour.length-1)]  : null
    const stPt   = (pitchStylized && Array.isArray(pitchStylized)) ? pitchStylized[Math.min(idx, pitchStylized.length-1)] : null
    const K = stPt?.K ?? rawPt?.K
    const P = stPt?.P ?? rawPt?.P
    const semitone = stPt?.semitone ?? rawPt?.semitone

    let html = h > 0
      ? `<b>${t.toFixed(2)}s</b> &middot; ${h.toFixed(0)} Hz`
      : `<b>${t.toFixed(2)}s</b> &middot; unvoiced`
    if (semitone != null) html += `<br>${semitone >= 0 ? '+' : ''}${semitone.toFixed(1)} st`
    if (K != null) html += `<br>K=${K}`
    if (P != null) html += ` P=${P}`

    tooltip.innerHTML  = html
    tooltip.style.display = 'block'

    // Position relative to the canvas wrapper (not viewport)
    // Push tooltip right if near left edge, up if near bottom edge
    const canvasH = canvas.getBoundingClientRect().height
    const ttW = 120  // approximate tooltip width
    const leftOff = relX + ttW + 16 > rect.width ? relX - ttW - 6 : relX + 12
    const topOff  = relY - 36
    tooltip.style.left = `${leftOff}px`
    tooltip.style.top  = `${Math.max(4, topOff)}px`
  }

  function handleMouseLeave() {
    if (tooltipRef.current) tooltipRef.current.style.display = 'none'
  }

  function handleKeyDown(e) {
    if (e.key === 'ArrowLeft')  { e.preventDefault(); onSeek?.(-0.5) }
    if (e.key === 'ArrowRight') { e.preventDefault(); onSeek?.(0.5) }
  }

  const headingId = 'pitch-contour-heading'

  return (
    <>
      <p id={headingId} className="sr-only">Pitch contour visualization</p>
      <div
        ref={containerRef}
        className="pitch-contour-scroll"
        role="region"
        aria-labelledby={headingId}
        tabIndex={0}
        onKeyDown={handleKeyDown}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
        style={{ width: '100%' }}
      >
        {/* position:relative so tooltip uses this as containing block */}
        <div style={{ position: 'relative', width: '100%', height: 200 }}>
          <canvas ref={staticRef}  aria-hidden="true" style={{ position: 'absolute', top: 0, left: 0 }} />
          <canvas ref={dynamicRef} aria-hidden="true" style={{ position: 'absolute', top: 0, left: 0 }} />
          {/* Tooltip is inside the relative wrapper — no viewport-fixed issues */}
          <div
            ref={tooltipRef}
            aria-hidden="true"
            style={{
              display: 'none',
              position: 'absolute',
              background: '#1C2128',
              color: '#F5F4F0',
              fontFamily: 'IBM Plex Mono, monospace',
              fontSize: '0.7rem',
              lineHeight: 1.5,
              padding: '4px 8px',
              pointerEvents: 'none',
              zIndex: 10,
              whiteSpace: 'nowrap',
              borderRadius: 2,
              boxShadow: '2px 2px 0 rgba(0,0,0,0.3)',
            }}
          />
        </div>
      </div>
    </>
  )
}
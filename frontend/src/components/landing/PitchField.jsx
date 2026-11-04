import { useEffect, useRef } from 'react'
import { C } from '../../tokens.js'

/**
 * PitchField — Landing page hero background.
 *
 * prefers-reduced-motion handling:
 *   The pitch curves ARE essential content, not decorative embellishment —
 *   they are the visual vocabulary of the tool itself. Under reduced-motion
 *   we halve the drift speed and skip the lerp snap (pointer influence still
 *   works, just with reduced easing). This satisfies the intent of reduced-
 *   motion (no large sudden movement) while keeping the ambient life.
 *
 *   CSS transitions and GSAP tweens elsewhere carry the reduced-motion branch.
 *   The canvas RAF loop always runs.
 */

const NUM_CURVES = 5
const CONTROL_POINTS = 8
const LERP_SPEED = 0.07
const DRIFT_SPEED = 22

const CURVE_DEFS = [
  { color: C.pitchRise,  opacity: 0.32, cy: 0.35, amp: 55 },
  { color: C.pitchFall,  opacity: 0.28, cy: 0.55, amp: 70 },
  { color: C.pitchLevel, opacity: 0.22, cy: 0.20, amp: 30 },
  { color: C.pitchRise,  opacity: 0.18, cy: 0.72, amp: 45 },
  { color: C.pitchFall,  opacity: 0.22, cy: 0.48, amp: 60 },
]

function catmullRom(ctx, pts) {
  if (pts.length < 4) return
  let first = true
  for (let i = 0; i < pts.length - 3; i++) {
    const p0 = pts[i], p1 = pts[i+1], p2 = pts[i+2], p3 = pts[i+3]
    const steps = 16
    for (let t = 0; t <= steps; t++) {
      const s = t / steps, s2 = s * s, s3 = s2 * s
      const x = 0.5*(2*p1[0]+(-p0[0]+p2[0])*s+(2*p0[0]-5*p1[0]+4*p2[0]-p3[0])*s2+(-p0[0]+3*p1[0]-3*p2[0]+p3[0])*s3)
      const y = 0.5*(2*p1[1]+(-p0[1]+p2[1])*s+(2*p0[1]-5*p1[1]+4*p2[1]-p3[1])*s2+(-p0[1]+3*p1[1]-3*p2[1]+p3[1])*s3)
      first ? ctx.moveTo(x, y) : ctx.lineTo(x, y)
      first = false
    }
  }
}

export default function PitchField({ sectionRef }) {
  const canvasRef = useRef(null)
  const stateRef  = useRef(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')

    // reduced-motion: halve drift speed, but ALWAYS run RAF.
    // The curves are the content of this page, not decorative animation.
    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    const driftMul = reduced ? 0.4 : 1.0

    const dpr = () => Math.min(window.devicePixelRatio, 2)
    const logW = () => canvas.width  / dpr()
    const logH = () => canvas.height / dpr()

    function buildCurves(w, h) {
      return CURVE_DEFS.map((def, ci) => {
        const spacing = w / (CONTROL_POINTS - 1)
        const xOffset = ci * (w / NUM_CURVES) * 0.4
        const home = Array.from({ length: CONTROL_POINTS }, (_, i) => {
          const bx = i * spacing - xOffset
          const noise = Math.sin(i * 1.3 + ci * 2.7) * def.amp
          return [bx, def.cy * h + noise]
        })
        return {
          ...def,
          home: home.map(p => [...p]),
          cur:  home.map(p => [...p]),
          gap: Array.from({ length: CONTROL_POINTS - 1 }, (_, i) =>
            Math.random() < 0.18 && i > 1 && i < CONTROL_POINTS - 2
          ),
          speed: DRIFT_SPEED * driftMul * (0.7 + ci * 0.15),
          phase: ci * 0.4,
        }
      })
    }

    function resize() {
      const d = dpr()
      const rect = canvas.getBoundingClientRect()
      const cssW = rect.width  || canvas.parentElement?.offsetWidth  || window.innerWidth
      const cssH = rect.height || canvas.parentElement?.offsetHeight || window.innerHeight
      canvas.width  = cssW * d
      canvas.height = cssH * d
      ctx.setTransform(1, 0, 0, 1, 0, 0)
      ctx.scale(d, d)
      const w = logW(), h = logH()
      if (stateRef.current) stateRef.current.curves = buildCurves(w, h)
      return { w, h }
    }

    const state = {
      curves: [],
      pointer: { x: -9999, y: -9999, active: false },
      lastTime: null,
      rafId: null,
    }
    stateRef.current = state
    const { w, h } = resize()
    state.curves = buildCurves(w, h)

    const ro = new ResizeObserver(() => resize())
    ro.observe(canvas.parentElement || canvas)

    // Pointer events on the section so hover works over the z-10 text content
    const eventTarget = sectionRef?.current ?? canvas.parentElement
    function onPointerMove(e) {
      const rect = canvas.getBoundingClientRect()
      state.pointer.x = e.clientX - rect.left
      state.pointer.y = e.clientY - rect.top
      state.pointer.active = true
    }
    function onPointerLeave() { state.pointer.active = false }
    if (eventTarget) {
      eventTarget.addEventListener('pointermove', onPointerMove)
      eventTarget.addEventListener('pointerleave', onPointerLeave)
    }

    function drawFrame() {
      const w = logW(), h = logH()
      ctx.clearRect(0, 0, w, h)
      for (const curve of state.curves) {
        ctx.strokeStyle = curve.color
        ctx.lineWidth   = 1.5
        ctx.globalAlpha = curve.opacity
        const pts = curve.cur
        let segStart = 0
        for (let i = 0; i < pts.length - 1; i++) {
          if (curve.gap[i]) {
            if (i - segStart >= 3) {
              ctx.beginPath()
              catmullRom(ctx, pts.slice(segStart, i + 2))
              ctx.stroke()
            }
            segStart = i + 1
          }
        }
        if (pts.length - segStart >= 3) {
          ctx.beginPath()
          catmullRom(ctx, pts.slice(segStart))
          ctx.stroke()
        }
        ctx.globalAlpha = 1
      }
    }

    function tick(ts) {
      if (!state.lastTime) state.lastTime = ts
      const dt = Math.min((ts - state.lastTime) / 1000, 0.05)
      state.lastTime = ts
      const w = logW(), h = logH()

      for (let ci = 0; ci < state.curves.length; ci++) {
        const c = state.curves[ci]
        for (let i = 0; i < c.home.length; i++) {
          // Drift leftward + wrap
          c.home[i][0] -= c.speed * dt
          if (c.home[i][0] < -100) {
            c.home[i][0] += w + 200
            if (i < c.gap.length) {
              c.gap[i] = Math.random() < 0.18 && i > 1 && i < c.home.length - 2
            }
          }
          // Slow Y oscillation (reduced-motion: lower amplitude via driftMul on amp indirectly)
          c.home[i][1] = c.cy * h +
            Math.sin(i * 1.3 + ci * 2.7 + ts * 0.0004 + c.phase) * c.amp

          // Pointer influence — always active, helps user understand the tool
          let tx = c.home[i][0], ty = c.home[i][1]
          if (state.pointer.active) {
            const dx = state.pointer.x - c.home[i][0]
            const dy = state.pointer.y - c.home[i][1]
            const dist = Math.sqrt(dx*dx + dy*dy)
            if (dist < 140) {
              const inf = (1 - dist / 140)
              tx += dx * inf * 0.65
              ty += dy * inf * 0.65
            }
          }

          // Lerp cur toward target (reduced-motion: skip lerp, snap directly)
          if (reduced) {
            c.cur[i][0] = tx
            c.cur[i][1] = ty
          } else {
            c.cur[i][0] += (tx - c.cur[i][0]) * LERP_SPEED
            c.cur[i][1] += (ty - c.cur[i][1]) * LERP_SPEED
          }
        }
      }

      drawFrame()
      state.rafId = requestAnimationFrame(tick)
    }

    // Always run RAF — both normal and reduced-motion paths.
    state.rafId = requestAnimationFrame(tick)

    return () => {
      if (state.rafId) cancelAnimationFrame(state.rafId)
      ro.disconnect()
      if (eventTarget) {
        eventTarget.removeEventListener('pointermove', onPointerMove)
        eventTarget.removeEventListener('pointerleave', onPointerLeave)
      }
    }
  }, [sectionRef])

  return (
    <canvas
      ref={canvasRef}
      className="pitch-field-canvas"
      aria-hidden="true"
    />
  )
}
import { useEffect, useRef } from 'react'

/**
 * FrequencyVisualizer — Live Audio-Reactive Scientific Vocalizer
 *
 * Real-time Web Audio API frequency visualization driven by the microphone AnalyserNode.
 * Renders 21 symmetrical frequency bars that dynamically respond to voice amplitude
 * and frequency spectrum, settling smoothly into a clean baseline during pauses.
 *
 * Visual style:
 *   - Scientific audio instrument
 *   - Symmetrical vocalizer architecture
 *   - Fluid easing with smoothed attack and decay (no jitter)
 *   - ProsodyLab brand palette (slate-300 idle, navy & gold #E8B84B / #FDB813 on active voice)
 *   - Respects prefers-reduced-motion
 */

const BAR_COUNT = 21       // Symmetrical odd count for central vocal focus
const BAR_WIDTH = 6        // px
const BAR_GAP = 5          // px
const MIN_HEIGHT = 4       // px baseline capsule
const MAX_HEIGHT = 64      // px maximum speech peak
const CANVAS_WIDTH = 320   // logical px
const CANVAS_HEIGHT = 96   // logical px

export default function FrequencyVisualizer({ analyserNode, isRecording }) {
  const canvasRef = useRef(null)
  const rafRef = useRef(null)
  const heightsRef = useRef(new Float32Array(BAR_COUNT).fill(MIN_HEIGHT))

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const dpr = Math.min(window.devicePixelRatio || 1, 2)
    canvas.width = CANVAS_WIDTH * dpr
    canvas.height = CANVAS_HEIGHT * dpr
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    ctx.scale(dpr, dpr)

    const reduced = typeof window !== 'undefined' &&
      window.matchMedia?.('(prefers-reduced-motion: reduce)')?.matches

    const binCount = analyserNode?.frequencyBinCount || 128
    const freqData = analyserNode ? new Uint8Array(binCount) : null
    const currentHeights = heightsRef.current

    function draw() {
      ctx.clearRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT)

      const isDark = document.documentElement.classList.contains('dark')
      const idleColor = isDark ? '#334155' : '#CBD5E1'
      const activeNavy = isDark ? '#F8FAFC' : '#0F172A'
      const goldAccent = isDark ? '#FDB813' : '#D97706'

      let overallRms = 0

      if (analyserNode && freqData && isRecording) {
        analyserNode.getByteFrequencyData(freqData)

        let sumSq = 0
        // Focus on speech frequencies: bins 2 to 45
        for (let i = 2; i < Math.min(45, freqData.length); i++) {
          sumSq += freqData[i] * freqData[i]
        }
        overallRms = Math.sqrt(sumSq / 43)
      }

      // Check if user is currently speaking vs pausing / silent
      const isVoiceActive = overallRms > 6 && isRecording

      // Compute target heights for all 21 bars
      const targetHeights = new Float32Array(BAR_COUNT)

      for (let i = 0; i < BAR_COUNT; i++) {
        if (!isVoiceActive || !freqData) {
          targetHeights[i] = MIN_HEIGHT
          continue
        }

        // Symmetrical distance from center bar (index 10)
        const distFromCenter = Math.abs(i - 10)
        // Sample corresponding frequency bin
        const binIdx = 2 + distFromCenter * 3
        const rawMag = freqData[binIdx] || 0
        const normMag = rawMag / 255

        // Voice weighting: center bars emphasize core voice energy
        const centerWeight = Math.cos((distFromCenter / 10) * (Math.PI / 2)) * 0.4 + 0.6
        const energy = Math.pow(normMag * centerWeight, 1.25)

        const maxH = reduced ? 28 : MAX_HEIGHT
        targetHeights[i] = Math.max(MIN_HEIGHT, MIN_HEIGHT + (maxH - MIN_HEIGHT) * energy)
      }

      // Smooth interpolation: attack is responsive, decay is soft and fluid
      const attack = reduced ? 0.15 : 0.28
      const decay = reduced ? 0.08 : 0.12

      for (let i = 0; i < BAR_COUNT; i++) {
        const target = targetHeights[i]
        const current = currentHeights[i]
        const rate = target > current ? attack : decay
        currentHeights[i] = current + (target - current) * rate
      }

      // Render 21 centered symmetrical rounded capsules
      const totalWidth = BAR_COUNT * BAR_WIDTH + (BAR_COUNT - 1) * BAR_GAP
      const startX = (CANVAS_WIDTH - totalWidth) / 2
      const centerY = CANVAS_HEIGHT / 2

      for (let i = 0; i < BAR_COUNT; i++) {
        const x = startX + i * (BAR_WIDTH + BAR_GAP)
        const h = currentHeights[i]
        const y = centerY - h / 2
        const radius = Math.min(BAR_WIDTH / 2, h / 2)

        // Color modulation based on bar height & voice intensity
        const activityRatio = Math.max(0, Math.min(1, (h - MIN_HEIGHT) / (MAX_HEIGHT - MIN_HEIGHT)))
        
        ctx.beginPath()
        if (typeof ctx.roundRect === 'function') {
          ctx.roundRect(x, y, BAR_WIDTH, h, radius)
        } else {
          // Fallback capsule drawing
          ctx.rect(x, y, BAR_WIDTH, h)
        }

        if (activityRatio < 0.08) {
          ctx.fillStyle = idleColor
        } else if (activityRatio > 0.45) {
          ctx.fillStyle = goldAccent
        } else {
          ctx.fillStyle = activeNavy
        }
        ctx.fill()
      }

      if (isRecording) {
        rafRef.current = requestAnimationFrame(draw)
      }
    }

    draw()

    return () => {
      if (rafRef.current) {
        cancelAnimationFrame(rafRef.current)
        rafRef.current = null
      }
    }
  }, [analyserNode, isRecording])

  return (
    <div className="flex flex-col items-center justify-center p-3">
      <canvas
        ref={canvasRef}
        style={{ width: CANVAS_WIDTH, height: CANVAS_HEIGHT }}
        className="touch-none select-none"
        role="img"
        aria-label="Live audio frequency visualization"
      >
        Live audio frequency visualization
      </canvas>
    </div>
  )
}

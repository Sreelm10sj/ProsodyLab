import { useEffect, useState } from 'react'
import { getHealth } from '../api.js'

/**
 * BackendStatus — health dot + model_loaded badge.
 * Polls /api/health on mount. Shows:
 *   - green dot if status === "ok"
 *   - amber dot if model not loaded
 *   - red dot on error
 * Plus a mono badge for model_loaded state.
 */

export default function BackendStatus() {
  const [health, setHealth] = useState(null)
  const [error, setError] = useState(false)

  useEffect(() => {
    let cancelled = false
    async function check() {
      try {
        const data = await getHealth()
        if (!cancelled) { setHealth(data); setError(false) }
      } catch {
        if (!cancelled) setError(true)
      }
    }
    check()
    const id = setInterval(check, 30_000)
    return () => { cancelled = true; clearInterval(id) }
  }, [])

  const dotColor = error
    ? 'bg-red-500'
    : health?.status === 'ok'
      ? (health.model_loaded ? 'bg-[#2F9E6F]' : 'bg-[#E8B84B]')
      : 'bg-[#A8B0BE]'

  const label = error
    ? 'Backend unreachable'
    : health
      ? `Backend ${health.status} · ${health.model_loaded ? 'model loaded' : 'model loading'}`
      : 'Checking backend…'

  return (
    <div className="flex items-center gap-2" role="status" aria-label={label}>
      <span
        className={`inline-block w-2 h-2 rounded-full ${dotColor}`}
        aria-hidden="true"
      />
      {health && (
        <span className="font-mono text-[10px] uppercase tracking-wider text-[#4E5766] dark:text-slate-400">
          {health.model_loaded ? 'model loaded' : 'model loading'}
        </span>
      )}
    </div>
  )
}

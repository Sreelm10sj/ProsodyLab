/**
 * Wordmark — renders "ProsodyLab" with intonation arrow.
 * Used on: Landing hero, Analyze header, Train header.
 */

export default function Wordmark({ size = 'sm', light = false, variant = 'refined' }) {
  const isLg = size === 'lg'

  if (variant === 'refined') {
    return (
      <span
        className={`inline-flex items-center tracking-tight select-none ${
          isLg ? 'text-4xl md:text-5xl font-extrabold' : 'text-xl font-bold'
        } ${light ? 'text-slate-50' : 'text-slate-900 dark:text-slate-100'}`}
        style={{ fontFamily: 'var(--font-sans)' }}
        aria-label="ProsodyLab"
      >
        <span
          aria-hidden="true"
          className={`tracking-tight ${light ? 'text-slate-50' : 'text-slate-900 dark:text-slate-100'}`}
        >
          Prosody
        </span>
        <span
          aria-hidden="true"
          className={`tracking-tight font-extrabold ${light ? 'text-slate-50' : 'text-slate-900 dark:text-slate-100'}`}
        >
          Lab
        </span>
        <span
          aria-hidden="true"
          className="font-bold text-emerald-600 dark:text-emerald-400 ml-1"
          style={{
            fontSize: isLg ? '1.5rem' : '1.05rem',
            lineHeight: 1,
          }}
        >
          ↗
        </span>
      </span>
    )
  }

  // Classic brutalist fallback
  const baseClass = light ? 'text-[#F5F4F0]' : 'text-[#1C2128] dark:text-slate-100'
  const sizeClass = size === 'lg' ? 'text-5xl md:text-7xl font-bold' : 'text-2xl font-bold'

  return (
    <span
      className={`font-mono tracking-tight inline-flex items-baseline ${sizeClass} ${baseClass}`}
      aria-label="ProsodyLab"
    >
      <span aria-hidden="true">Prosody</span>
      <span className="font-mono font-bold" aria-hidden="true">Lab</span>
      <span
        className="ml-1 text-[#2F9E6F] dark:text-emerald-400 text-xs font-mono font-bold leading-none"
        aria-hidden="true"
        style={{ fontSize: size === 'lg' ? '1.1rem' : '0.65rem' }}
      >
        ↗
      </span>
    </span>
  )
}


/**
 * Format separation score consistently whether provided as fraction (0.6822)
 * or pre-calculated percentage (68.2231).
 */
export function formatSeparationScore(score) {
  if (score === null || score === undefined) return '—'
  const num = Number(score)
  if (isNaN(num)) return '—'
  const pct = num <= 1.0 ? num * 100 : num
  return `${pct.toFixed(2)}%`
}

/**
 * Format training accuracy consistently whether provided as decimal fraction (0.842)
 * or pre-calculated percentage (84.2).
 */
export function formatAccuracy(acc) {
  if (acc === null || acc === undefined) return '—'
  const num = Number(acc)
  if (isNaN(num)) return '—'
  const pct = num <= 1.0 ? num * 100 : num
  return `${pct.toFixed(1)}%`
}


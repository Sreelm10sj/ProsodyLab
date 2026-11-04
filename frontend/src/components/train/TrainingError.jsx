import React, { useState } from 'react'

export default function TrainingError({ title = 'Error', message, error, details, onDismiss }) {
  const [showDetails, setShowDetails] = useState(false)

  const text = message || error
  if (!text) return null

  return (
    <div
      role="alert"
      className="p-4 rounded-xl border border-red-200 bg-red-50/70 text-red-900 text-xs font-mono space-y-2"
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <strong className="text-red-700 uppercase font-bold">{title}:</strong>
          <span className="font-sans text-xs text-red-900">{text}</span>
        </div>
        {onDismiss && (
          <button
            type="button"
            onClick={onDismiss}
            className="text-red-600 hover:text-red-900 font-bold ml-4 cursor-pointer"
            aria-label="Dismiss error"
          >
            ✕
          </button>
        )}
      </div>

      {details && (
        <div className="pt-2 border-t border-red-200/80">
          <button
            type="button"
            onClick={() => setShowDetails((prev) => !prev)}
            className="text-[11px] text-red-700 hover:underline flex items-center gap-1 cursor-pointer font-mono"
          >
            {showDetails ? '▼ Hide Technical Details' : '▶ Show Technical Details'}
          </button>
          {showDetails && (
            <pre className="mt-2 p-3 bg-white/90 border border-red-200 rounded-lg text-[11px] text-red-950 overflow-x-auto max-h-48 whitespace-pre-wrap font-mono">
              {typeof details === 'string' ? details : JSON.stringify(details, null, 2)}
            </pre>
          )}
        </div>
      )}
    </div>
  )
}

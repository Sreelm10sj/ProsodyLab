import { useState, useCallback } from 'react'

/**
 * BatchProgress — Real-time progress screen for batch audio annotation
 * Refined to match ProsodyLab visual system:
 *   - Clean white card with subtle 1px border
 *   - Clean typography and metrics
 *   - Gold/navy progress bar
 *   - Current file indicator with spinner
 *
 * Props:
 *   jobState     — current batch state object
 *   onCancel     — callback when cancellation is confirmed
 *   isCancelling — boolean while cancel request is in-flight
 */
export default function BatchProgress({ jobState, onCancel, isCancelling }) {
  const [showConfirmCancel, setShowConfirmCancel] = useState(false)

  const totalFiles = jobState?.total_files ?? 0
  const completedFiles = jobState?.completed_files ?? 0
  const failedFiles = jobState?.failed_files ?? 0
  const remainingFiles = Math.max(0, totalFiles - (completedFiles + failedFiles))
  const progressPercent = jobState?.progress_percent ?? 0.0
  const currentFile = jobState?.current_file ?? 'Initializing worker…'

  const handleConfirmCancel = useCallback(() => {
    setShowConfirmCancel(false)
    onCancel()
  }, [onCancel])

  return (
    <div className="w-full bg-white rounded-2xl border border-slate-200/90 shadow-sm p-6 md:p-8 space-y-6 select-none">
      
      {/* Header & Status */}
      <div className="flex items-center justify-between flex-wrap gap-4 pb-4 border-b border-slate-100">
        <div>
          <div className="flex items-center gap-2">
            <span className="inline-block w-2.5 h-2.5 rounded-full bg-amber-500 animate-pulse" />
            <h2 className="text-sm font-bold uppercase tracking-wider text-slate-900 font-mono">
              Batch Analysis In Progress
            </h2>
          </div>
          <p className="text-xs text-slate-500 font-mono mt-1">
            {completedFiles} of {totalFiles} files completed
          </p>
        </div>

        {!showConfirmCancel ? (
          <button
            type="button"
            id="btn-cancel-batch"
            onClick={() => setShowConfirmCancel(true)}
            disabled={isCancelling}
            className="btn-white-action px-3 py-1.5 text-xs text-slate-600 hover:text-red-600 hover:border-red-200"
          >
            {isCancelling ? 'Cancelling…' : 'Cancel Batch'}
          </button>
        ) : (
          <div className="flex items-center gap-3 p-2 rounded-lg bg-red-50/60 border border-red-200">
            <span className="text-xs text-red-900 font-medium">
              Cancel batch? Completed files remain downloadable.
            </span>
            <button
              type="button"
              id="btn-confirm-cancel"
              onClick={handleConfirmCancel}
              className="px-2.5 py-1 rounded bg-red-600 hover:bg-red-700 text-white text-xs font-bold transition-colors"
            >
              Yes, Cancel
            </button>
            <button
              type="button"
              onClick={() => setShowConfirmCancel(false)}
              className="px-2 py-1 rounded border border-slate-300 text-slate-600 text-xs hover:bg-white"
            >
              Keep Running
            </button>
          </div>
        )}
      </div>

      {/* Progress Bar */}
      <div className="space-y-2">
        <div className="flex justify-between items-center text-xs font-mono font-bold text-slate-900">
          <span>Overall Progress</span>
          <span className="text-amber-600">{progressPercent.toFixed(1)}%</span>
        </div>
        <div className="w-full h-2.5 bg-slate-100 rounded-full overflow-hidden border border-slate-200/60">
          <div
            className="h-full bg-gradient-to-r from-amber-400 to-[#FDB813] transition-all duration-300 ease-out rounded-full"
            style={{
              width: `${Math.min(100, Math.max(0, progressPercent))}%`,
            }}
          />
        </div>
      </div>

      {/* Current File Banner */}
      <div className="px-4 py-3 rounded-xl bg-slate-50/80 border border-slate-200/80 flex items-center gap-3 overflow-hidden">
        <svg
          className="animate-spin flex-shrink-0 text-amber-500"
          width="15" height="15"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="3"
          strokeLinecap="round"
          aria-hidden="true"
        >
          <path d="M21 12a9 9 0 1 1-6.219-8.56" />
        </svg>
        <div className="text-xs font-mono truncate text-slate-800">
          <span className="text-slate-400 font-medium">Current:&nbsp;</span>
          <span className="font-bold">{currentFile}</span>
        </div>
      </div>

      {/* Metrics Row */}
      <div className="grid grid-cols-3 gap-3 text-center">
        <div className="p-3 rounded-xl border border-slate-200/80 bg-slate-50/40">
          <div className="text-[11px] uppercase tracking-wider text-slate-400 font-mono">Completed</div>
          <div className="text-xl font-bold text-emerald-600 font-mono mt-0.5">{completedFiles}</div>
        </div>
        <div className="p-3 rounded-xl border border-slate-200/80 bg-slate-50/40">
          <div className="text-[11px] uppercase tracking-wider text-slate-400 font-mono">Failed</div>
          <div className={`text-xl font-bold font-mono mt-0.5 ${failedFiles > 0 ? 'text-red-500' : 'text-slate-700'}`}>
            {failedFiles}
          </div>
        </div>
        <div className="p-3 rounded-xl border border-slate-200/80 bg-slate-50/40">
          <div className="text-[11px] uppercase tracking-wider text-slate-400 font-mono">Remaining</div>
          <div className="text-xl font-bold text-slate-700 font-mono mt-0.5">{remainingFiles}</div>
        </div>
      </div>

    </div>
  )
}

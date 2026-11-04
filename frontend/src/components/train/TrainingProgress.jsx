import React, { useState } from 'react'
import { ChevronDown } from 'lucide-react'

export default function TrainingProgress({
  jobId,
  jobState,
  onCancelJob,
  isCancelling,
}) {
  const [showConfirmCancel, setShowConfirmCancel] = useState(false)
  const [showHistory, setShowHistory] = useState(false)

  const status = jobState?.status || 'queued'
  const currentStep = jobState?.current_step || 'Initializing worker...'
  const percent = Math.min(100, Math.max(0, jobState?.progress_percent || 0))
  const loop = jobState?.loop || 0
  const totalLoops = jobState?.total_loops || 13
  const elapsed = jobState?.elapsed_seconds || 0
  const changes = jobState?.pseudo_label_changes || 0
  const lossHistory = jobState?.loss_history || []

  // Resolve IDEC loss: prefer direct state property, current_metrics, final_metrics, then lossHistory
  let latestIdec = jobState?.idec_loss ?? jobState?.current_metrics?.idec_loss ?? jobState?.final_metrics?.final_idec_loss ?? null
  if (latestIdec === null) {
    for (let i = lossHistory.length - 1; i >= 0; i--) {
      const item = lossHistory[i]
      if (item.idec_loss !== null && item.idec_loss !== undefined) {
        latestIdec = item.idec_loss
        break
      }
    }
  }

  function handleConfirmCancel() {
    setShowConfirmCancel(false)
    onCancelJob()
  }

  function formatSeconds(secs) {
    const s = Math.floor(secs)
    const m = Math.floor(s / 60)
    const rem = s % 60
    return `${m}m ${rem.toString().padStart(2, '0')}s`
  }

  return (
    <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-6 md:p-8 space-y-6">
      <div className="flex items-start justify-between gap-4 flex-wrap">
        <div>
          <h2 className="text-xl font-bold text-slate-900 tracking-tight flex items-center gap-2.5">
            <span className="w-2.5 h-2.5 rounded-full bg-[#FDB813] animate-pulse" />
            <span>Training LexiRep Model</span>
          </h2>
          <p className="font-mono text-xs text-slate-500 mt-1">
            Job ID: <span className="text-slate-900 font-semibold">{jobId}</span>
          </p>
        </div>

        {/* Cancel Button */}
        <div>
          <button
            type="button"
            onClick={() => setShowConfirmCancel(true)}
            disabled={isCancelling || status === 'completed' || status === 'failed' || status === 'cancelled'}
            className="btn-white-action px-3.5 py-1.5 text-xs font-semibold text-slate-600 hover:text-red-600 hover:border-red-200 disabled:opacity-40 cursor-pointer"
          >
            {isCancelling ? 'Cancelling...' : 'Cancel Training'}
          </button>
        </div>
      </div>

      {/* Cancellation Confirmation Dialog */}
      {showConfirmCancel && (
        <div className="p-4 rounded-xl border border-red-200 bg-red-50/80 text-red-900 space-y-3 shadow-2xs">
          <div className="text-xs font-bold uppercase tracking-wider text-red-950 font-mono">
            Cancel this training run?
          </div>
          <p className="font-sans text-xs text-red-900 leading-relaxed">
            The worker process will be terminated immediately. Partially trained weights and logs will remain available for inspection.
          </p>
          <div className="flex gap-2 pt-1">
            <button
              type="button"
              onClick={handleConfirmCancel}
              className="px-3.5 py-1.5 bg-red-600 text-white rounded-lg text-xs font-semibold hover:bg-red-700 cursor-pointer shadow-2xs"
            >
              Confirm Cancel
            </button>
            <button
              type="button"
              onClick={() => setShowConfirmCancel(false)}
              className="btn-white-action px-3.5 py-1.5 text-xs font-semibold"
            >
              Resume
            </button>
          </div>
        </div>
      )}

      {/* Failure Banner */}
      {status === 'failed' && (
        <div className="p-4 rounded-xl border border-red-200 bg-red-50/70 text-red-950 font-mono text-xs space-y-1">
          <div className="font-bold uppercase text-red-900 flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-red-600" />
            <span>Training Failed</span>
          </div>
          <p className="font-sans text-xs text-red-900 mt-1">
            {jobState?.error || 'Worker encountered an unexpected error.'}
          </p>
        </div>
      )}

      {/* Main Progress Container */}
      <div className="space-y-6 pt-2">
        {/* Phase & Loop status */}
        <div className="flex flex-wrap items-baseline justify-between gap-2">
          <div>
            <div className="text-xs font-medium text-slate-500 uppercase tracking-wider font-mono">Current Phase</div>
            <div className="font-sans font-bold text-lg md:text-xl text-slate-900 mt-0.5">{currentStep}</div>
          </div>
          <div className="font-mono text-xs text-slate-500">
            Loop <strong className="text-slate-900 text-base">{loop}</strong> of {totalLoops}
          </div>
        </div>

        {/* Progress Bar */}
        <div className="space-y-2">
          <div className="w-full bg-slate-100 rounded-full h-3 overflow-hidden p-0.5 border border-slate-200/60">
            <div
              className="h-full bg-[#FDB813] rounded-full transition-all duration-300"
              style={{ width: `${percent}%` }}
              role="progressbar"
              aria-valuenow={percent}
              aria-valuemin="0"
              aria-valuemax="100"
            />
          </div>
          <div className="flex justify-between font-mono text-xs text-slate-500">
            <span>Overall Progress</span>
            <span className="font-bold text-slate-900">{percent.toFixed(1)}%</span>
          </div>
        </div>

        {/* Live Metrics Grid (SupCon Loss is explicitly NOT rendered) */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 pt-2">
          <div className="bg-slate-50/70 p-4 rounded-xl border border-slate-200 font-mono shadow-2xs">
            <span className="text-[10px] text-slate-500 uppercase font-semibold block">IDEC Loss</span>
            <span className="font-bold text-xl text-slate-900 mt-1 block">
              {latestIdec !== null ? Number(latestIdec).toFixed(4) : '—'}
            </span>
          </div>

          <div className="bg-slate-50/70 p-4 rounded-xl border border-slate-200 font-mono shadow-2xs">
            <span className="text-[10px] text-slate-500 uppercase font-semibold block">Label Changes</span>
            <span className="font-bold text-xl text-slate-900 mt-1 block">
              {changes ? changes.toLocaleString() : '0'}
            </span>
          </div>

          <div className="bg-slate-50/70 p-4 rounded-xl border border-slate-200 font-mono shadow-2xs">
            <span className="text-[10px] text-slate-500 uppercase font-semibold block">Elapsed Time</span>
            <span className="font-bold text-xl text-slate-900 mt-1 block">
              {formatSeconds(elapsed)}
            </span>
          </div>
        </div>

        {/* Expandable Live History Table */}
        {lossHistory.length > 0 && (
          <div className="pt-2 border-t border-slate-200/80">
            <button
              type="button"
              onClick={() => setShowHistory((p) => !p)}
              className="text-xs font-semibold text-slate-600 hover:text-slate-900 flex items-center gap-1.5 cursor-pointer font-sans"
            >
              <ChevronDown
                size={14}
                className={`transition-transform duration-150 ${showHistory ? 'rotate-180' : ''}`}
              />
              <span>{showHistory ? 'Hide Live Epoch History' : `Show Live Epoch History (${lossHistory.length} events)`}</span>
            </button>

            {showHistory && (
              <div className="mt-3 rounded-xl border border-slate-200 overflow-hidden bg-white shadow-2xs max-h-56 overflow-y-auto">
                <table className="w-full text-left border-collapse font-mono text-xs">
                  <thead className="bg-slate-50/70 border-b border-slate-200/80 sticky top-0 text-[10px] text-slate-500 uppercase">
                    <tr>
                      <th className="py-2.5 px-3 font-semibold">Phase</th>
                      <th className="py-2.5 px-3 font-semibold">Loop</th>
                      <th className="py-2.5 px-3 font-semibold">Epoch</th>
                      <th className="py-2.5 px-3 font-semibold">IDEC Loss</th>
                      <th className="py-2.5 px-3 font-semibold">Changes</th>
                      <th className="py-2.5 px-3 font-semibold">Time</th>
                    </tr>
                  </thead>
                  <tbody>
                    {lossHistory.slice().reverse().map((item, idx) => (
                      <tr key={idx} className="border-b border-slate-100 hover:bg-slate-50/60 transition-colors">
                        <td className="py-2 px-3 text-slate-500">{item.phase}</td>
                        <td className="py-2 px-3 font-bold text-slate-900">{item.loop}</td>
                        <td className="py-2 px-3 text-slate-500">{item.epoch}</td>
                        <td className="py-2 px-3 text-slate-900 font-semibold">
                          {item.idec_loss !== null && item.idec_loss !== undefined
                            ? Number(item.idec_loss).toFixed(4)
                            : '—'}
                        </td>
                        <td className="py-2 px-3 text-slate-800">{item.label_changes || 0}</td>
                        <td className="py-2 px-3 text-slate-500">{item.elapsed ?? item.elapsed_sec ?? 0}s</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

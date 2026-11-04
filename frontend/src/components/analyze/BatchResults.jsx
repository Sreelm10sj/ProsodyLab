import { useState, useCallback } from 'react'
import { Download, CheckCircle2, AlertTriangle, XCircle, ChevronDown, ChevronUp, ArrowRight } from 'lucide-react'

/**
 * BatchResults — Displays terminal batch outcomes and provides ZIP download.
 * Refined to match ProsodyLab visual system:
 *   - Soft tinted outcome banners
 *   - Gold action button for ZIP download
 *   - Refined status table and expandable error drawers
 *
 * Props:
 *   jobState     — batch state object from API
 *   onDownload   — callback to trigger download of annotations ZIP
 *   onReset      — callback to reset batch state and analyze another archive
 *   isDownloading— boolean flag while ZIP is downloading
 */
export default function BatchResults({ jobState, onDownload, onReset, isDownloading }) {
  const [expandedRow, setExpandedRow] = useState(null)

  const status = jobState?.status ?? 'completed'
  const totalFiles = jobState?.total_files ?? 0
  const completedFiles = jobState?.completed_files ?? 0
  const failedFiles = jobState?.failed_files ?? 0
  const elapsed = jobState?.elapsed_seconds ?? 0
  const files = jobState?.files ?? []

  const isSuccess = status === 'completed'
  const isPartial = status === 'completed_with_errors'
  const isCancelled = status === 'cancelled'
  const isFailed = status === 'failed'

  const canDownload = completedFiles > 0

  const toggleRow = (idx) => {
    setExpandedRow(expandedRow === idx ? null : idx)
  }

  return (
    <div className="w-full bg-white rounded-2xl border border-slate-200/90 shadow-sm p-6 md:p-8 space-y-6 select-none">
      
      {/* Banner / Header */}
      <div
        className={`p-6 rounded-xl border flex items-center justify-between flex-wrap gap-4 ${
          isSuccess
            ? 'bg-emerald-50/70 border-emerald-200/80 text-emerald-950'
            : isPartial
              ? 'bg-amber-50/70 border-amber-200/80 text-amber-950'
              : 'bg-slate-50 border-slate-200 text-slate-900'
        }`}
      >
        <div className="flex items-center gap-3.5">
          {isSuccess && <CheckCircle2 size={26} className="text-emerald-600 flex-shrink-0" />}
          {isPartial && <AlertTriangle size={26} className="text-amber-600 flex-shrink-0" />}
          {(isCancelled || isFailed) && <XCircle size={26} className="text-red-500 flex-shrink-0" />}

          <div>
            <h2 className="text-base font-bold tracking-tight">
              {isSuccess && 'Batch Complete'}
              {isPartial && 'Batch Complete with Errors'}
              {isCancelled && 'Batch Cancelled'}
              {isFailed && 'Batch Failed'}
            </h2>
            <p className="text-xs font-mono text-slate-500 mt-1">
              {completedFiles} annotations generated · {failedFiles} failed · {elapsed}s elapsed
            </p>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex items-center gap-3 flex-wrap">
          {canDownload && (
            <button
              type="button"
              id="btn-download-batch-zip"
              onClick={onDownload}
              disabled={isDownloading}
              className="btn-gold-action px-5 py-2.5 text-xs font-bold shadow-gold"
            >
              <Download size={14} />
              <span>{isDownloading ? 'Preparing ZIP…' : 'Download Annotations (.zip)'}</span>
            </button>
          )}

          <button
            type="button"
            onClick={onReset}
            className="btn-white-action px-4 py-2.5 text-xs font-semibold"
          >
            Analyze Another Batch
          </button>
        </div>
      </div>

      {/* File Status Table */}
      <div className="space-y-2">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-500 font-mono">
          File Status Details ({files.length} files)
        </h3>

        <div className="rounded-xl border border-slate-200/90 overflow-hidden bg-white">
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse text-xs font-mono">
              <thead>
                <tr className="bg-slate-50 border-b border-slate-200/80 text-slate-600">
                  <th className="p-3 font-semibold uppercase tracking-wider text-[11px]">Audio File</th>
                  <th className="p-3 font-semibold uppercase tracking-wider text-[11px]">Annotation JSON</th>
                  <th className="p-3 font-semibold uppercase tracking-wider text-[11px]">Duration</th>
                  <th className="p-3 font-semibold uppercase tracking-wider text-[11px]">Words</th>
                  <th className="p-3 font-semibold uppercase tracking-wider text-[11px]">Status</th>
                  <th className="p-3 font-semibold uppercase tracking-wider text-[11px]">Details</th>
                </tr>
              </thead>
              <tbody>
                {files.map((f, idx) => {
                  const isItemSuccess = f.status === 'completed'
                  const isItemFailed = f.status === 'failed'
                  const isExpanded = expandedRow === idx

                  return (
                    <tr
                      key={f.input_name ?? idx}
                      className="border-b border-slate-100 transition-colors hover:bg-slate-50/60"
                    >
                      <td className="p-3 font-medium text-slate-900 truncate max-w-[180px]" title={f.input_name}>
                        {f.input_name}
                      </td>
                      <td className="p-3 text-slate-500 truncate max-w-[180px]" title={f.output_name}>
                        {isItemSuccess ? f.output_name : '—'}
                      </td>
                      <td className="p-3 text-slate-500">
                        {f.duration != null ? `${f.duration}s` : '—'}
                      </td>
                      <td className="p-3 text-slate-500">
                        {f.words_count != null ? f.words_count : '—'}
                      </td>
                      <td className="p-3">
                        {isItemSuccess && (
                          <span className="inline-flex items-center px-2 py-0.5 font-semibold text-[11px] rounded-md bg-emerald-50 text-emerald-700 border border-emerald-200">
                            ✓ Completed
                          </span>
                        )}
                        {isItemFailed && (
                          <span className="inline-flex items-center px-2 py-0.5 font-semibold text-[11px] rounded-md bg-red-50 text-red-700 border border-red-200">
                            ✗ Failed
                          </span>
                        )}
                        {!isItemSuccess && !isItemFailed && (
                          <span className="inline-flex items-center px-2 py-0.5 text-[11px] text-slate-500">
                            {f.status}
                          </span>
                        )}
                      </td>
                      <td className="p-3">
                        {isItemFailed && f.error ? (
                          <button
                            type="button"
                            onClick={() => toggleRow(idx)}
                            className="flex items-center gap-1 text-[11px] font-bold text-amber-700 hover:text-amber-800 underline"
                          >
                            <span>{isExpanded ? 'Hide' : 'Error'}</span>
                            {isExpanded ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
                          </button>
                        ) : (
                          <span className="text-slate-300">—</span>
                        )}
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </div>

        {/* Expandable Error Details Drawer if selected */}
        {expandedRow !== null && files[expandedRow]?.error && (
          <div className="p-4 rounded-xl border border-red-200 bg-red-50/70 font-mono text-xs text-red-900 mt-2">
            <span className="font-bold">Error on {files[expandedRow].input_name}: </span>
            {files[expandedRow].error}
          </div>
        )}
      </div>

    </div>
  )
}

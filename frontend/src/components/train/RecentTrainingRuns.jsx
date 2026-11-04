import React, { useEffect, useState } from 'react'
import { ChevronDown, ChevronRight, RotateCw, History } from 'lucide-react'
import { listTrainingJobs } from '../../api'

export default function RecentTrainingRuns({
  onSelectJob,
  currentJobId,
}) {
  const [jobs, setJobs] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [isExpanded, setIsExpanded] = useState(false)

  const fetchJobs = React.useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const data = await listTrainingJobs(10)
      setJobs(Array.isArray(data) ? data : [])
    } catch (err) {
      setError(err.message || 'Failed to fetch job history')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchJobs()
  }, [fetchJobs, currentJobId])

  function formatTimestamp(isoStr) {
    if (!isoStr) return '—'
    try {
      const d = new Date(isoStr)
      return d.toLocaleDateString(undefined, {
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
      })
    } catch {
      return isoStr
    }
  }

  function getStatusBadge(status) {
    switch (status) {
      case 'completed':
        return (
          <span className="inline-flex items-center px-2 py-0.5 rounded-md text-[11px] font-medium bg-emerald-50 text-emerald-700 border border-emerald-200/80">
            completed
          </span>
        )
      case 'training':
      case 'warming_up':
      case 'idec_pretrain':
        return (
          <span className="inline-flex items-center px-2 py-0.5 rounded-md text-[11px] font-medium bg-blue-50 text-blue-700 border border-blue-200/80 animate-pulse">
            {status}
          </span>
        )
      case 'failed':
        return (
          <span className="inline-flex items-center px-2 py-0.5 rounded-md text-[11px] font-medium bg-red-50 text-red-700 border border-red-200/80">
            failed
          </span>
        )
      case 'cancelled':
        return (
          <span className="inline-flex items-center px-2 py-0.5 rounded-md text-[11px] font-medium bg-amber-50 text-amber-800 border border-amber-200/80">
            cancelled
          </span>
        )
      default:
        return (
          <span className="inline-flex items-center px-2 py-0.5 rounded-md text-[11px] font-medium bg-slate-100 text-slate-700 border border-slate-200">
            {status}
          </span>
        )
    }
  }

  return (
    <div className="rounded-2xl border border-slate-200/90 bg-white p-5 shadow-2xs">
      <div className="flex items-center justify-between">
        <button
          type="button"
          onClick={() => setIsExpanded(!isExpanded)}
          className="flex items-center gap-2.5 text-sm font-semibold text-slate-800 hover:text-slate-950 cursor-pointer transition-colors"
        >
          {isExpanded ? (
            <ChevronDown size={16} className="text-slate-500" />
          ) : (
            <ChevronRight size={16} className="text-slate-500" />
          )}
          <History size={16} className="text-slate-500" />
          <span>Recent Training Runs</span>
          <span className="text-xs px-2 py-0.5 rounded-full bg-slate-100 text-slate-600 font-medium">
            {jobs.length}
          </span>
        </button>

        <div className="flex items-center gap-3">
          {loading && (
            <span className="text-xs text-slate-400 font-mono">Refreshing...</span>
          )}
          <button
            type="button"
            onClick={fetchJobs}
            disabled={loading}
            className="btn-white-action flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-600 hover:text-slate-900 disabled:opacity-50"
          >
            <RotateCw size={12} className={loading ? 'animate-spin' : ''} />
            <span>Refresh</span>
          </button>
        </div>
      </div>

      {isExpanded && (
        <div className="mt-4 space-y-3 pt-3 border-t border-slate-100">
          {error && (
            <div className="p-3 rounded-lg border border-red-200 bg-red-50 text-red-800 font-mono text-xs">
              {error}
            </div>
          )}

          {jobs.length === 0 && !loading && (
            <div className="p-6 text-center text-xs text-slate-500 rounded-xl border border-dashed border-slate-200 bg-slate-50/50">
              No recent training runs found on this server instance.
            </div>
          )}

          {jobs.length > 0 && (
            <div className="overflow-hidden rounded-xl border border-slate-200/80">
              <table className="w-full text-xs text-left">
                <thead className="bg-slate-50/80 border-b border-slate-200 text-slate-600 font-semibold">
                  <tr>
                    <th className="px-4 py-2.5">Job ID</th>
                    <th className="px-4 py-2.5">Date</th>
                    <th className="px-4 py-2.5">Language</th>
                    <th className="px-4 py-2.5">Status</th>
                    <th className="px-4 py-2.5 text-right">Action</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100 text-slate-700">
                  {jobs.map((j) => {
                    const isCurrent = j.job_id === currentJobId
                    return (
                      <tr
                        key={j.job_id}
                        className={`hover:bg-slate-50/70 transition-colors ${
                          isCurrent ? 'bg-amber-50/40' : ''
                        }`}
                      >
                        <td className="px-4 py-3 font-mono font-semibold text-slate-900">
                          {j.job_id.slice(0, 8)}...
                          {isCurrent && (
                            <span className="ml-2 text-[10px] text-amber-700 font-medium px-1.5 py-0.5 rounded bg-amber-100/60 border border-amber-200/60">
                              active
                            </span>
                          )}
                        </td>
                        <td className="px-4 py-3 text-slate-500">
                          {formatTimestamp(j.updated_at || j.created_at)}
                        </td>
                        <td className="px-4 py-3 font-bold text-slate-900">{j.language || 'GER'}</td>
                        <td className="px-4 py-3">{getStatusBadge(j.status)}</td>
                        <td className="px-4 py-3 text-right">
                          <button
                            type="button"
                            onClick={() => onSelectJob(j.job_id)}
                            className="btn-white-action px-3 py-1 text-xs font-medium text-slate-700 hover:text-slate-900"
                          >
                            Inspect
                          </button>
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </div>
  )
}


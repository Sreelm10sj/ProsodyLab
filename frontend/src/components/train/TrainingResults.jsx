import React, { useState } from 'react'
import { Download, ChevronDown, CheckCircle2, RotateCcw } from 'lucide-react'
import { downloadTrainingBundle, downloadTrainingArtifact } from '../../api'
import { formatAccuracy } from './formatters'

export default function TrainingResults({
  jobId,
  jobState,
  onResetWorkflow,
}) {
  const [downloadingBundle, setDownloadingBundle] = useState(false)
  const [downloadingArtifact, setDownloadingArtifact] = useState(null)
  const [downloadError, setDownloadError] = useState(null)
  const [showBundleContents, setShowBundleContents] = useState(false)
  const [showHistory, setShowHistory] = useState(false)

  const language = jobState?.language || 'GER'
  const elapsed = jobState?.elapsed_seconds || 0
  const loop = jobState?.loop || 0
  const totalLoops = jobState?.total_loops || 13
  const lossHistory = jobState?.loss_history || []

  // Extract real metrics
  const finalAccuracy = jobState?.training_accuracy ?? jobState?.final_metrics?.training_accuracy ?? null

  let finalIdec = jobState?.final_metrics?.final_idec_loss ?? jobState?.idec_loss ?? null
  if (finalIdec === null) {
    for (let i = lossHistory.length - 1; i >= 0; i--) {
      const item = lossHistory[i]
      if (item.idec_loss !== null && item.idec_loss !== undefined) {
        finalIdec = item.idec_loss
        break
      }
    }
  }

  function formatSeconds(secs) {
    if (!secs || secs < 0) return '0s'
    const m = Math.floor(secs / 60)
    const s = Math.floor(secs % 60)
    if (m === 0) return `${s}s`
    return `${m}m ${s}s`
  }

  async function handleDownloadBundle() {
    if (!jobId || downloadingBundle) return
    setDownloadingBundle(true)
    setDownloadError(null)
    try {
      await downloadTrainingBundle(jobId)
    } catch (err) {
      setDownloadError(err.message || 'Failed to download model bundle.')
    } finally {
      setDownloadingBundle(false)
    }
  }

  async function handleDownloadArtifact(name) {
    if (!jobId || downloadingArtifact) return
    setDownloadingArtifact(name)
    setDownloadError(null)
    try {
      await downloadTrainingArtifact(jobId, name)
    } catch (err) {
      setDownloadError(`Failed to download ${name}: ${err.message}`)
    } finally {
      setDownloadingArtifact(null)
    }
  }

  const benchmark = jobState?.benchmark ?? jobState?.final_metrics?.benchmark ?? null

  const secondaryArtifacts = [
    { name: 'metrics.json', label: 'Metrics JSON' },
    { name: 'evaluation_summary.txt', label: 'Benchmark Summary TXT' },
    { name: 'training_history.csv', label: 'Training History CSV' },
    { name: 'anchors.json', label: 'Anchors JSON' },
    { name: 'config.json', label: 'Config JSON' },
    { name: 'training.log', label: 'Training Log' },
  ]

  return (
    <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-6 md:p-8 space-y-8">
      {/* Header Banner */}
      <div className="flex flex-wrap items-start justify-between gap-4 pb-4 border-b border-slate-200/80">
        <div>
          <div className="flex items-center gap-2.5">
            <span className="w-2.5 h-2.5 rounded-full bg-emerald-600" />
            <h2 className="text-xl font-bold text-slate-900 tracking-tight">
              Training Complete
            </h2>
            <span className="px-2.5 py-0.5 border border-emerald-200 bg-emerald-50 text-emerald-800 rounded-full text-xs font-semibold">
              Production Bundle Verified
            </span>
          </div>
          <p className="font-mono text-xs text-slate-500 mt-1">
            Job ID: <span className="text-slate-900 font-semibold">{jobId}</span> · Language: <span className="font-bold text-slate-900">{language}</span>
          </p>
        </div>

        <button
          type="button"
          onClick={onResetWorkflow}
          className="btn-white-action px-3.5 py-2 text-xs font-semibold shadow-2xs"
        >
          <RotateCcw size={13} className="text-slate-500" />
          <span>Train Another Model</span>
        </button>
      </div>

      {downloadError && (
        <div className="p-4 rounded-xl border border-red-200 bg-red-50/70 text-red-900 text-xs font-mono flex justify-between items-center">
          <span>{downloadError}</span>
          <button
            type="button"
            onClick={() => setDownloadError(null)}
            className="text-red-600 hover:text-red-900 font-bold ml-4 cursor-pointer"
          >
            ✕
          </button>
        </div>
      )}

      {/* Primary Key Metrics Grid - TRAINING METRICS */}
      <div className="space-y-3">
        <div className="text-xs uppercase tracking-wider text-slate-500 font-mono font-semibold">
          Training Metrics (Training Partition)
        </div>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="bg-slate-50/70 rounded-xl border border-slate-200/80 p-4 shadow-2xs">
            <div className="font-mono text-xs font-semibold text-slate-500 uppercase tracking-wider">Final Training Accuracy</div>
            <div className="font-mono font-bold text-2xl text-slate-900 mt-1">
              {formatAccuracy(finalAccuracy)}
            </div>
            <div className="font-sans text-xs text-slate-500 mt-1">Syllable stress classification</div>
          </div>

          <div className="bg-slate-50/70 rounded-xl border border-slate-200/80 p-4 shadow-2xs">
            <div className="font-mono text-xs font-semibold text-slate-500 uppercase tracking-wider">Final IDEC Loss</div>
            <div className="font-mono font-bold text-2xl text-slate-900 mt-1">
              {finalIdec !== null ? Number(finalIdec).toFixed(4) : '—'}
            </div>
            <div className="font-sans text-xs text-slate-500 mt-1">Clustering loss</div>
          </div>

          <div className="bg-slate-50/70 rounded-xl border border-slate-200/80 p-4 shadow-2xs">
            <div className="font-mono text-xs font-semibold text-slate-500 uppercase tracking-wider">Duration</div>
            <div className="font-mono font-bold text-2xl text-slate-900 mt-1">
              {formatSeconds(elapsed)}
            </div>
            <div className="font-sans text-xs text-slate-500 mt-1">Total execution time</div>
          </div>

          <div className="bg-slate-50/70 rounded-xl border border-slate-200/80 p-4 shadow-2xs">
            <div className="font-mono text-xs font-semibold text-slate-500 uppercase tracking-wider">Loops Completed</div>
            <div className="font-mono font-bold text-2xl text-slate-900 mt-1">
              {loop} / {totalLoops}
            </div>
            <div className="font-sans text-xs text-slate-500 mt-1">Self-training iterations</div>
          </div>
        </div>
      </div>

      {/* Held-out Quarantined Benchmark Section */}
      {benchmark && (
        <div className="space-y-3 pt-2">
          <div className="flex items-center justify-between">
            <div className="text-xs uppercase tracking-wider text-amber-900 font-mono font-semibold flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-[#FDB813]" />
              <span>Held-out Quarantined Benchmark (Independent Test Set)</span>
            </div>
            <span className="text-[11px] font-mono text-slate-500">
              {benchmark.total_syllables?.toLocaleString()} syllables · {benchmark.total_words?.toLocaleString()} words
            </span>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
            <div className="bg-amber-50/40 rounded-xl border border-amber-200/80 p-3.5 shadow-2xs font-mono">
              <div className="text-[10px] text-amber-900 uppercase font-semibold">Bi-Syllabic (B)</div>
              <div className="font-bold text-xl text-slate-900 mt-1">
                {benchmark.b_accuracy?.toFixed(2)}%
              </div>
              <div className="text-[10px] text-slate-500 mt-0.5">
                {benchmark.b_words?.toLocaleString()} words
              </div>
            </div>

            <div className="bg-amber-50/40 rounded-xl border border-amber-200/80 p-3.5 shadow-2xs font-mono">
              <div className="text-[10px] text-amber-900 uppercase font-semibold">Bi + Tri (BT)</div>
              <div className="font-bold text-xl text-slate-900 mt-1">
                {benchmark.bt_accuracy?.toFixed(2)}%
              </div>
              <div className="text-[10px] text-slate-500 mt-0.5">
                {benchmark.bt_words?.toLocaleString()} words
              </div>
            </div>

            <div className="bg-amber-50/40 rounded-xl border border-amber-200/80 p-3.5 shadow-2xs font-mono">
              <div className="text-[10px] text-amber-900 uppercase font-semibold">Bi+Tri+Quad (BTQ)</div>
              <div className="font-bold text-xl text-slate-900 mt-1">
                {benchmark.btq_accuracy?.toFixed(2)}%
              </div>
              <div className="text-[10px] text-slate-500 mt-0.5">
                {benchmark.btq_words?.toLocaleString()} words
              </div>
            </div>

            <div className="bg-amber-50/40 rounded-xl border border-amber-200/80 p-3.5 shadow-2xs font-mono">
              <div className="text-[10px] text-amber-900 uppercase font-semibold">Balanced Acc</div>
              <div className="font-bold text-xl text-slate-900 mt-1">
                {benchmark.balanced_accuracy?.toFixed(2)}%
              </div>
              <div className="text-[10px] text-slate-500 mt-0.5">
                BTQ syllables
              </div>
            </div>

            <div className="bg-amber-50/40 rounded-xl border border-amber-200/80 p-3.5 shadow-2xs font-mono">
              <div className="text-[10px] text-amber-900 uppercase font-semibold">Total Syllables</div>
              <div className="font-bold text-xl text-slate-900 mt-1">
                {benchmark.total_syllable_accuracy?.toFixed(2)}%
              </div>
              <div className="text-[10px] text-slate-500 mt-0.5">
                All {benchmark.total_syllables?.toLocaleString()} syls
              </div>
            </div>

            <div className="bg-amber-50/40 rounded-xl border border-amber-200/80 p-3.5 shadow-2xs font-mono">
              <div className="text-[10px] text-amber-900 uppercase font-semibold">Exact Word Acc</div>
              <div className="font-bold text-xl text-slate-900 mt-1">
                {benchmark.word_accuracy?.toFixed(2)}%
              </div>
              <div className="text-[10px] text-slate-500 mt-0.5">
                {benchmark.correct_words?.toLocaleString()}/{benchmark.total_words?.toLocaleString()}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Export Section */}
      <div className="rounded-xl border border-slate-200 bg-slate-50/30 p-6 space-y-6">
        <div>
          <h3 className="text-base font-bold text-slate-900 tracking-tight">
            Model Export & Artifacts
          </h3>
          <p className="text-xs text-slate-500 mt-1">
            Download the verified model bundle ready for production inference, or download individual training artifacts.
          </p>
        </div>

        {/* Primary CTA */}
        <div className="flex flex-col sm:flex-row items-start sm:items-center gap-4 p-5 rounded-xl border border-slate-200 bg-white shadow-2xs">
          <button
            type="button"
            onClick={handleDownloadBundle}
            disabled={downloadingBundle}
            className="btn-gold-action w-full sm:w-auto px-6 py-3.5 text-sm font-bold shadow-gold flex items-center justify-center gap-2"
          >
            {downloadingBundle ? (
              <>
                <span className="w-4 h-4 border-2 border-[#0F172A] border-t-transparent rounded-full animate-spin" />
                <span>Preparing Bundle...</span>
              </>
            ) : (
              <>
                <Download size={16} />
                <span>Download Model Bundle (.zip)</span>
              </>
            )}
          </button>
          <div className="text-xs text-slate-500 leading-relaxed">
            Official format containing PyTorch weights, 10-D prototypes, anchor definitions, and configuration.
          </div>
        </div>

        {/* Secondary Downloads */}
        <div className="space-y-2.5">
          <div className="text-xs uppercase tracking-wider text-slate-500 font-mono font-semibold">
            Individual Artifacts
          </div>
          <div className="flex flex-wrap gap-2">
            {secondaryArtifacts.map((art) => (
              <button
                key={art.name}
                type="button"
                onClick={() => handleDownloadArtifact(art.name)}
                disabled={downloadingArtifact === art.name}
                className="btn-white-action px-3.5 py-2 text-xs font-mono font-semibold shadow-2xs flex items-center gap-1.5"
              >
                {downloadingArtifact === art.name ? (
                  <span className="w-3 h-3 border-2 border-slate-700 border-t-transparent rounded-full animate-spin" />
                ) : (
                  <Download size={13} className="text-slate-500" />
                )}
                <span>{art.label}</span>
              </button>
            ))}
          </div>
        </div>

        {/* Bundle Content Explanation (Collapsible) */}
        <div className="border-t border-slate-200/80 pt-4">
          <button
            type="button"
            onClick={() => setShowBundleContents(!showBundleContents)}
            className="text-xs font-semibold text-slate-600 hover:text-slate-900 flex items-center gap-1.5 cursor-pointer"
          >
            <ChevronDown
              size={14}
              className={`transition-transform duration-150 ${showBundleContents ? 'rotate-180' : ''}`}
            />
            <span>Model bundle contents</span>
          </button>

          {showBundleContents && (
            <div className="mt-3 p-4 bg-white rounded-xl border border-slate-200 space-y-2.5 font-mono text-xs text-slate-800 shadow-2xs">
              <div className="space-y-1.5">
                <div className="flex items-baseline gap-2">
                  <span className="font-bold text-slate-900">config.json</span>
                  <span className="text-slate-500 text-[11px]">— Architecture, hyperparameters, and representative protocol</span>
                </div>
                <div className="flex items-baseline gap-2">
                  <span className="font-bold text-slate-900">anchors.json</span>
                  <span className="text-slate-500 text-[11px]">— Syllable indices, separation score, and acoustic prototype metadata</span>
                </div>
                <div className="flex items-baseline gap-2">
                  <span className="font-bold text-slate-900">model/final_lexirep_model.pt</span>
                  <span className="text-slate-500 text-[11px]">— PyTorch bottleneck encoder weights (768-D → 10-D)</span>
                </div>
                <div className="flex items-baseline gap-2">
                  <span className="font-bold text-slate-900">anchors/prototypes_10d.npy</span>
                  <span className="text-slate-500 text-[11px]">— Serialized 10-D stressed and unstressed prototype vectors</span>
                </div>
              </div>
              <p className="font-sans text-xs text-slate-500 pt-2 border-t border-slate-200/60 leading-relaxed">
                This bundle is verified and can be loaded directly by the ProsodyLab LexiRep inference engine without code modifications.
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Collapsible Training History */}
      {lossHistory.length > 0 && (
        <div className="rounded-xl border border-slate-200 bg-slate-50/30 p-5 shadow-2xs">
          <button
            type="button"
            onClick={() => setShowHistory(!showHistory)}
            className="w-full text-xs text-slate-900 font-bold flex items-center justify-between cursor-pointer font-mono"
          >
            <span className="uppercase tracking-wider">Training History ({lossHistory.length} checkpoints)</span>
            <ChevronDown
              size={14}
              className={`transition-transform duration-150 ${showHistory ? 'rotate-180' : ''}`}
            />
          </button>

          {showHistory && (
            <div className="mt-4 rounded-xl border border-slate-200 overflow-hidden bg-white shadow-2xs max-h-60 overflow-y-auto">
              <table className="w-full font-mono text-xs text-left">
                <thead className="bg-slate-50/70 border-b border-slate-200/80 text-slate-500 uppercase text-[10px] sticky top-0">
                  <tr>
                    <th className="py-2.5 px-3 font-semibold">Loop</th>
                    <th className="py-2.5 px-3 font-semibold">Phase</th>
                    <th className="py-2.5 px-3 font-semibold">IDEC Loss</th>
                    <th className="py-2.5 px-3 font-semibold">Label Changes</th>
                    <th className="py-2.5 px-3 font-semibold">Elapsed</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {lossHistory.map((row, idx) => (
                    <tr key={idx} className="hover:bg-slate-50/60 transition-colors">
                      <td className="py-2 px-3 font-bold text-slate-900">{row.loop ?? '—'}</td>
                      <td className="py-2 px-3 text-slate-600">{row.phase || 'self_training'}</td>
                      <td className="py-2 px-3 text-slate-900 font-semibold">
                        {row.idec_loss !== null && row.idec_loss !== undefined
                          ? Number(row.idec_loss).toFixed(4)
                          : '—'}
                      </td>
                      <td className="py-2 px-3 text-slate-800">{row.label_changes ?? '—'}</td>
                      <td className="py-2 px-3 text-slate-500">{row.elapsed ? `${Math.round(row.elapsed)}s` : '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

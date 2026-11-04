import React from 'react'
import { ArrowLeft, ArrowRight, Compass } from 'lucide-react'
import { formatSeparationScore } from './formatters'

export default function AnchorStep({
  datasetValidation,
  isScanning,
  anchorResults,
  selectedAnchor,
  onScanAnchors,
  onSelectAnchor,
  onContinue,
  onBack,
}) {
  const syllablesCount = datasetValidation?.syllable_count || 0
  const topAnchors = anchorResults?.top_anchors || []

  return (
    <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-6 md:p-8 space-y-6">
      <div>
        <h2 className="text-xl font-bold text-slate-900 tracking-tight">
          Stage 02 — Anchor Discovery
        </h2>
        <p className="text-xs text-slate-500 mt-1 max-w-2xl leading-relaxed">
          LexiRep discovers acoustic primary stress by initializing from a single acoustic anchor pair.
          Search the training set for bisyllabic stressed/unstressed syllable pairs that provide the strongest initial separation signal.
        </p>
      </div>

      {/* Action to trigger scan if not yet executed */}
      {!anchorResults && !isScanning && (
        <div className="rounded-xl border border-slate-200 bg-slate-50/40 p-10 text-center space-y-4">
          <div className="w-12 h-12 rounded-full bg-slate-100 flex items-center justify-center text-slate-700 mx-auto shadow-2xs">
            <Compass size={24} strokeWidth={1.75} />
          </div>
          <p className="text-sm font-semibold text-slate-900">
            Ready to evaluate bisyllabic candidates in the training partition.
          </p>
          <button
            type="button"
            onClick={() => onScanAnchors(20)}
            className="btn-gold-action px-6 py-2.5 text-xs font-bold shadow-gold"
          >
            Scan Training Set
          </button>
        </div>
      )}

      {/* Scanning State */}
      {isScanning && (
        <div className="rounded-xl border border-slate-200 bg-slate-50/40 p-10 flex flex-col items-center justify-center space-y-3 text-center">
          <div className="w-6 h-6 border-2 border-[#FDB813] border-t-transparent rounded-full animate-spin" />
          <div>
            <div className="font-bold text-sm text-slate-900">
              Scanning {syllablesCount.toLocaleString()} syllable representations...
            </div>
            <div className="font-mono text-xs text-slate-500 mt-1">
              Evaluating candidate separation directions across bisyllabic pairs...
            </div>
          </div>
        </div>
      )}

      {/* Anchor Leaderboard & Details */}
      {anchorResults && !isScanning && (
        <div className="space-y-6">
          {/* Selected Anchor Summary Panel */}
          {selectedAnchor && (
            <div className="rounded-xl border border-amber-200 bg-amber-50/50 p-4 space-y-3">
              <div className="text-xs font-bold uppercase tracking-wider text-slate-900 flex items-center justify-between">
                <span>Selected Anchor Pair</span>
                <span className="text-[11px] px-2.5 py-0.5 bg-[#FDB813] text-slate-900 font-bold rounded-full">
                  Rank {selectedAnchor.rank}
                </span>
              </div>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 bg-white p-3.5 rounded-lg border border-slate-200 font-mono text-xs">
                <div>
                  <span className="text-slate-500 text-[10px] uppercase block font-medium">Word ID</span>
                  <span className="font-bold text-slate-900 text-sm">{selectedAnchor.word_id}</span>
                </div>
                <div>
                  <span className="text-slate-500 text-[10px] uppercase block font-medium">Stressed Syllable (s)</span>
                  <span className="font-bold text-slate-900 text-sm">Index {selectedAnchor.stressed_syllable_index}</span>
                </div>
                <div>
                  <span className="text-slate-500 text-[10px] uppercase block font-medium">Unstressed Syllable (u)</span>
                  <span className="font-bold text-slate-900 text-sm">Index {selectedAnchor.unstressed_syllable_index}</span>
                </div>
                <div>
                  <span className="text-slate-500 text-[10px] uppercase block font-medium">S_train Separation</span>
                  <span className="font-bold text-slate-900 text-sm">{formatSeparationScore(selectedAnchor.separation_score)}</span>
                </div>
              </div>
            </div>
          )}

          {/* Leaderboard Table */}
          <div className="rounded-xl border border-slate-200 overflow-hidden bg-white shadow-2xs">
            <div className="px-5 py-3.5 bg-slate-50/70 border-b border-slate-200/80 flex items-center justify-between">
              <span className="text-xs font-bold uppercase tracking-wider text-slate-800 font-mono">
                Candidate Leaderboard ({topAnchors.length} pairs)
              </span>
              <span className="font-mono text-xs text-slate-500">
                Scanned in {anchorResults.scan_time_seconds}s
              </span>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left font-mono text-xs border-collapse">
                <thead>
                  <tr className="border-b border-slate-200/80 bg-slate-50/50 text-slate-500 uppercase text-[10px]">
                    <th className="py-2.5 px-3 font-semibold">Rank</th>
                    <th className="py-2.5 px-3 font-semibold">Word ID</th>
                    <th className="py-2.5 px-3 font-semibold">Stressed (s)</th>
                    <th className="py-2.5 px-3 font-semibold">Unstressed (u)</th>
                    <th className="py-2.5 px-3 text-right font-semibold">S_train</th>
                    <th className="py-2.5 px-3 text-right font-semibold">Action</th>
                  </tr>
                </thead>
                <tbody>
                  {topAnchors.map((cand) => {
                    const isSelected = selectedAnchor?.word_id === cand.word_id &&
                      selectedAnchor?.stressed_syllable_index === cand.stressed_syllable_index

                    return (
                      <tr
                        key={`${cand.word_id}-${cand.stressed_syllable_index}`}
                        onClick={() => onSelectAnchor(cand)}
                        className={`border-b border-slate-100 cursor-pointer transition-colors ${
                          isSelected
                            ? 'bg-amber-50/80 font-semibold'
                            : 'hover:bg-slate-50/70'
                        }`}
                      >
                        <td className="py-2.5 px-3 text-slate-500 font-medium">
                          {String(cand.rank).padStart(2, '0')}
                        </td>
                        <td className="py-2.5 px-3 text-slate-900 font-bold">
                          {cand.word_id}
                          {cand.word_text && <span className="ml-1 text-slate-500 font-normal">({cand.word_text})</span>}
                        </td>
                        <td className="py-2.5 px-3 text-slate-800">{cand.stressed_syllable_index}</td>
                        <td className="py-2.5 px-3 text-slate-800">{cand.unstressed_syllable_index}</td>
                        <td className="py-2.5 px-3 text-right text-slate-900 font-bold">
                          {formatSeparationScore(cand.separation_score)}
                        </td>
                        <td className="py-2.5 px-3 text-right">
                          <button
                            type="button"
                            onClick={(e) => {
                              e.stopPropagation()
                              onSelectAnchor(cand)
                            }}
                            className={`inline-block px-2.5 py-1 text-[10px] font-semibold uppercase rounded-md transition-colors ${
                              isSelected
                                ? 'bg-slate-900 text-white'
                                : 'btn-white-action shadow-2xs'
                            }`}
                          >
                            {isSelected ? 'Selected' : 'Use'}
                          </button>
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* Navigation Buttons */}
      <div className="flex justify-between pt-2">
        <button
          type="button"
          onClick={onBack}
          disabled={isScanning}
          className="btn-white-action px-4 py-2 text-xs font-semibold disabled:opacity-50"
        >
          <ArrowLeft size={14} />
          <span>Back to Dataset</span>
        </button>

        <button
          type="button"
          onClick={onContinue}
          disabled={!selectedAnchor || isScanning}
          className="btn-gold-action px-6 py-3 text-sm font-bold shadow-gold disabled:opacity-40 disabled:cursor-not-allowed"
        >
          <span>Continue to Training</span>
          <ArrowRight size={16} className="arrow-shift" />
        </button>
      </div>
    </div>
  )
}

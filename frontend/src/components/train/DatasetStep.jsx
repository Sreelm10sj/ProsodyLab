import React, { useState, useRef } from 'react'
import { UploadCloud, FileArchive, ChevronDown, ArrowRight } from 'lucide-react'

export default function DatasetStep({
  file,
  validation,
  isValidating,
  onFileSelect,
  onResetFile,
  onContinue,
}) {
  const [isDragOver, setIsDragOver] = useState(false)
  const [showDetails, setShowDetails] = useState(false)
  const inputRef = useRef(null)

  function handleDrop(e) {
    e.preventDefault()
    setIsDragOver(false)
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const dropped = e.dataTransfer.files[0]
      onFileSelect(dropped)
    }
  }

  function handleFileChange(e) {
    if (e.target.files && e.target.files[0]) {
      onFileSelect(e.target.files[0])
    }
  }

  return (
    <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-6 md:p-8 space-y-6">
      <div>
        <h2 className="text-xl font-bold text-slate-900 tracking-tight">
          Stage 01 — Dataset Ingestion
        </h2>
        <p className="text-xs text-slate-500 mt-1">
          Upload a pre-extracted LexiRep syllable feature cache archive (<code className="font-mono text-xs px-1.5 py-0.5 rounded bg-slate-100 text-slate-700">.npz</code>).
        </p>
      </div>

      {/* Hidden File Input */}
      <input
        ref={inputRef}
        type="file"
        accept=".npz"
        onChange={handleFileChange}
        className="sr-only"
        id="dataset-file-input"
        aria-label="Upload LexiRep feature cache file"
      />

      {/* Upload Zone */}
      {!file && (
        <div
          onDragOver={(e) => { e.preventDefault(); setIsDragOver(true) }}
          onDragLeave={() => setIsDragOver(false)}
          onDrop={handleDrop}
          onClick={() => inputRef.current?.click()}
          className={`border-2 border-dashed rounded-xl p-10 flex flex-col items-center justify-center text-center cursor-pointer transition-all duration-150 select-none ${
            isDragOver
              ? 'border-amber-400 bg-amber-50/30'
              : 'border-slate-200/90 hover:border-slate-300 bg-slate-50/30 hover:bg-slate-50/60'
          }`}
          role="button"
          tabIndex={0}
          aria-label="Upload LexiRep feature cache file"
          onKeyDown={(e) => {
            if (e.key === 'Enter' || e.key === ' ') {
              e.preventDefault()
              inputRef.current?.click()
            }
          }}
        >
          <div className="w-12 h-12 rounded-full bg-slate-100 flex items-center justify-center text-slate-700 mb-3 shadow-2xs">
            <UploadCloud size={24} strokeWidth={1.75} />
          </div>
          <p className="text-base font-semibold text-slate-900 mb-1">
            Upload LexiRep feature cache
          </p>
          <p className="text-xs text-slate-500 font-medium mb-1.5">
            .npz · up to 250 MB
          </p>
          <p className="text-xs font-mono text-slate-400 mb-5">
            Expected: <span className="font-semibold text-slate-700">X_tr</span>, <span className="font-semibold text-slate-700">Y_tr</span>, <span className="font-semibold text-slate-700">W_tr</span>
          </p>
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation()
              inputRef.current?.click()
            }}
            className="btn-white-action px-4 py-2 text-xs font-semibold shadow-2xs"
          >
            Choose File
          </button>
        </div>
      )}

      {/* File Selected / Validating State */}
      {file && (
        <div className="border border-slate-200 rounded-xl p-5 bg-slate-50/40 space-y-4">
          <div className="flex items-center justify-between gap-4">
            <div className="flex items-center gap-3 overflow-hidden">
              <div className="w-10 h-10 rounded-lg bg-amber-100/70 border border-amber-200 flex items-center justify-center text-amber-800 flex-shrink-0">
                <FileArchive size={20} />
              </div>
              <div className="overflow-hidden">
                <div className="text-sm font-bold text-slate-900 truncate max-w-md">
                  {file.name}
                </div>
                <div className="text-xs text-slate-500 font-mono">
                  {(file.size / (1024 * 1024)).toFixed(2)} MB
                </div>
              </div>
            </div>

            <button
              type="button"
              onClick={() => {
                onResetFile?.()
                inputRef.current?.click()
              }}
              disabled={isValidating}
              className="btn-white-action px-3 py-1.5 text-xs font-semibold disabled:opacity-50"
            >
              Change file
            </button>
          </div>

          {/* Validation Loading */}
          {isValidating && (
            <div className="pt-4 border-t border-slate-200/80 flex items-center gap-3 text-slate-700">
              <div className="w-4 h-4 border-2 border-slate-700 border-t-transparent rounded-full animate-spin" />
              <span className="font-mono text-xs font-medium">
                Validating representation matrix and linguistic word alignments...
              </span>
            </div>
          )}

          {/* Validation Success Card */}
          {!isValidating && validation && validation.valid && (
            <div className="pt-4 border-t border-slate-200/80 space-y-4">
              <div className="flex items-center gap-2.5 p-3 rounded-xl bg-emerald-50/80 border border-emerald-200">
                <span className="inline-flex items-center justify-center w-5 h-5 rounded-full bg-emerald-600 text-white font-mono text-xs font-bold flex-shrink-0">
                  ✓
                </span>
                <span className="font-sans font-bold text-xs md:text-sm text-emerald-900">
                  Dataset structurally and linguistically verified
                </span>
              </div>

              {/* Primary Highlights Grid */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 bg-white p-4 rounded-xl border border-slate-200 shadow-2xs">
                <div>
                  <div className="font-mono text-[10px] text-slate-500 uppercase font-semibold">Syllables</div>
                  <div className="font-mono font-bold text-lg text-slate-900 mt-0.5">
                    {validation.syllable_count?.toLocaleString()}
                  </div>
                </div>
                <div>
                  <div className="font-mono text-[10px] text-slate-500 uppercase font-semibold">Words</div>
                  <div className="font-mono font-bold text-lg text-slate-900 mt-0.5">
                    {validation.word_count?.toLocaleString()}
                  </div>
                </div>
                <div>
                  <div className="font-mono text-[10px] text-slate-500 uppercase font-semibold">Anchor Candidates</div>
                  <div className="font-mono font-bold text-lg text-slate-900 mt-0.5">
                    {validation.bisyllabic_anchor_candidate_count?.toLocaleString()}
                  </div>
                </div>
                <div>
                  <div className="font-mono text-[10px] text-slate-500 uppercase font-semibold">Feature Dim</div>
                  <div className="font-mono font-bold text-lg text-slate-900 mt-0.5">
                    {validation.feature_shape?.[1] ? `${validation.feature_shape[1]}-D` : '768-D'}
                  </div>
                </div>
              </div>

              {/* Expandable details */}
              <div className="pt-1">
                <button
                  type="button"
                  onClick={() => setShowDetails((p) => !p)}
                  className="text-xs font-semibold text-slate-600 hover:text-slate-900 flex items-center gap-1.5 cursor-pointer"
                >
                  <ChevronDown
                    size={14}
                    className={`transition-transform duration-150 ${showDetails ? 'rotate-180' : ''}`}
                  />
                  <span>{showDetails ? 'Hide Distribution Details' : 'Show Distribution & Class Ratio'}</span>
                </button>

                {showDetails && (
                  <div className="mt-3 p-4 bg-slate-50/70 rounded-xl border border-slate-200 text-xs font-mono space-y-2">
                    <div className="flex justify-between border-b border-slate-200/60 pb-1.5">
                      <span className="text-slate-500">Primary Stressed Syllables:</span>
                      <span className="font-bold text-slate-900">{validation.stressed_count?.toLocaleString()}</span>
                    </div>
                    <div className="flex justify-between border-b border-slate-200/60 pb-1.5">
                      <span className="text-slate-500">Unstressed Syllables:</span>
                      <span className="font-bold text-slate-900">{validation.unstressed_count?.toLocaleString()}</span>
                    </div>
                    <div className="flex justify-between border-b border-slate-200/60 pb-1.5">
                      <span className="text-slate-500">Stress Ratio:</span>
                      <span className="font-bold text-slate-900">{((validation.stress_ratio || 0) * 100).toFixed(1)}%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-500">Rejected Non-Bisyllabic Words:</span>
                      <span className="font-bold text-slate-900">{validation.rejected_word_count?.toLocaleString()}</span>
                    </div>
                    {validation.warnings && validation.warnings.length > 0 && (
                      <div className="mt-2 pt-2 border-t border-slate-200/60 text-amber-800">
                        <span className="font-bold">Warnings:</span> {validation.warnings.join('; ')}
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Action Button */}
      <div className="flex justify-end pt-2">
        <button
          type="button"
          onClick={onContinue}
          disabled={!validation || !validation.valid || isValidating}
          className="btn-gold-action px-6 py-3 text-sm font-bold shadow-gold disabled:opacity-40 disabled:cursor-not-allowed"
        >
          <span>Continue to Anchor Discovery</span>
          <ArrowRight size={16} className="arrow-shift" />
        </button>
      </div>
    </div>
  )
}

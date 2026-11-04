import { useState, useRef, useCallback } from 'react'
import { UploadCloud, FileArchive, X, ArrowRight, FolderArchive } from 'lucide-react'

/**
 * BatchAudioInput — Refined ZIP Archive Upload Screen
 * Matches the scientific, clean ProsodyLab visual system:
 *   - Batch-specific Speaker L1 model selector
 *   - Soft dashed border dropzone
 *   - Cloud archive icon
 *   - Clear limit specs (.wav · .mp3 · .flac · .m4a · .ogg, up to 2,000 files, 250 MB)
 *   - Prominent [ Start Batch Analysis → ] gold action button
 *
 * Props:
 *   onBatchSubmit(file)         — callback when user submits ZIP
 *   isLoading                   — boolean while upload/validation is in flight
 *   batchNativity               — current batch nativity ('GER' | 'ITA' | null)
 *   onBatchNativityChange(val)  — callback when nativity is changed
 */
export default function BatchAudioInput({
  onBatchSubmit,
  isLoading,
  batchNativity = 'GER',
  onBatchNativityChange,
}) {
  const [file, setFile] = useState(null)
  const [dragOver, setDragOver] = useState(false)
  const fileInputRef = useRef(null)

  const handleFile = useCallback((f) => {
    if (!f) return
    setFile(f)
  }, [])

  const handleDrop = useCallback((e) => {
    e.preventDefault()
    setDragOver(false)
    const f = e.dataTransfer.files?.[0]
    if (f) handleFile(f)
  }, [handleFile])

  const handleClear = useCallback(() => {
    setFile(null)
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }, [])

  const handleSubmit = useCallback((e) => {
    e?.preventDefault()
    if (!file || isLoading) return
    onBatchSubmit(file)
  }, [file, isLoading, onBatchSubmit])

  const formatBytes = (bytes) => {
    if (!bytes) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return `${parseFloat((bytes / Math.pow(k, i)).toFixed(2))} ${sizes[i]}`
  }

  return (
    <div className="w-full space-y-6 select-none">
      
      {/* ── Batch Speaker L1 Model Selector ── */}
      <div className="flex items-center justify-between flex-wrap gap-4 p-4 rounded-xl bg-slate-50/70 border border-slate-200/80">
        <div className="flex items-center gap-3">
          <span className="text-xs font-mono font-semibold uppercase tracking-wider text-slate-600">
            Speaker L1 Model:
          </span>
          <div className="inline-flex rounded-lg border border-slate-200 p-0.5 bg-white shadow-2xs">
            <button
              type="button"
              id="btn-batch-nativity-ger"
              onClick={() => onBatchNativityChange?.('GER')}
              disabled={isLoading}
              aria-pressed={batchNativity === 'GER'}
              className={`text-xs font-semibold px-3 py-1 rounded-md transition-all ${
                batchNativity === 'GER'
                  ? 'bg-slate-900 text-white shadow-xs'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              German (GER)
            </button>
            <button
              type="button"
              id="btn-batch-nativity-ita"
              onClick={() => onBatchNativityChange?.('ITA')}
              disabled={isLoading}
              aria-pressed={batchNativity === 'ITA'}
              className={`text-xs font-semibold px-3 py-1 rounded-md transition-all ${
                batchNativity === 'ITA'
                  ? 'bg-slate-900 text-white shadow-xs'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              Italian (ITA)
            </button>
            <button
              type="button"
              id="btn-batch-nativity-none"
              onClick={() => onBatchNativityChange?.(null)}
              disabled={isLoading}
              aria-pressed={batchNativity === null}
              className={`text-xs font-semibold px-3 py-1 rounded-md transition-all ${
                batchNativity === null
                  ? 'bg-slate-900 text-white shadow-xs'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              None
            </button>
          </div>
        </div>
        <span className="text-xs text-slate-400 font-mono">
          Applied uniformly to all audio files in the batch
        </span>
      </div>

      {!file ? (
        /* Dropzone */
        <div
          role="region"
          aria-label="Drop ZIP archive here"
          onClick={() => fileInputRef.current?.click()}
          onDragOver={(e) => {
            e.preventDefault()
            setDragOver(true)
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
          className={`border-2 border-dashed rounded-xl p-10 flex flex-col items-center justify-center text-center cursor-pointer transition-all duration-150 ${
            dragOver
              ? 'border-amber-400 bg-amber-50/30'
              : 'border-slate-200/90 hover:border-slate-300 bg-slate-50/30 hover:bg-slate-50/60'
          }`}
        >
          {/* Cloud Icon Circle */}
          <div className="w-12 h-12 rounded-full bg-slate-100 flex items-center justify-center text-slate-700 mb-3 shadow-2xs">
            <UploadCloud size={24} strokeWidth={1.75} />
          </div>

          <p className="text-base font-semibold text-slate-900 mb-1">
            Drop ZIP archive containing audio files here
          </p>
          <p className="text-xs text-slate-500 font-medium mb-1">
            Supported formats: .wav · .mp3 · .flac · .m4a · .ogg
          </p>
          <p className="text-[11px] text-slate-400 font-mono mb-5">
            Maximum 250 MB · Up to 2,000 audio files per archive
          </p>

          {/* Choose File Button */}
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation()
              fileInputRef.current?.click()
            }}
            className="btn-white-action px-4 py-2 text-xs font-semibold shadow-2xs"
          >
            <FolderArchive size={14} />
            <span>Choose ZIP Archive</span>
          </button>

          <input
            ref={fileInputRef}
            type="file"
            accept=".zip,application/zip,application/x-zip-compressed"
            className="sr-only"
            id="batch-file-input"
            aria-label="Select ZIP archive"
            onChange={(e) => handleFile(e.target.files?.[0])}
          />
        </div>
      ) : (
        /* Archive Selected */
        <div className="space-y-4">
          <div className="flex items-center justify-between gap-4 p-5 rounded-xl border border-slate-200 bg-slate-50/50">
            <div className="flex items-center gap-3.5 overflow-hidden">
              <div className="w-11 h-11 rounded-lg bg-amber-100/70 border border-amber-200 flex items-center justify-center text-amber-800 flex-shrink-0">
                <FileArchive size={22} />
              </div>
              <div className="overflow-hidden">
                <p className="text-sm font-bold text-slate-900 truncate">
                  {file.name}
                </p>
                <p className="text-xs text-slate-500 font-mono mt-0.5">
                  {formatBytes(file.size)} · Ready for batch extraction
                </p>
              </div>
            </div>

            <button
              type="button"
              onClick={handleClear}
              disabled={isLoading}
              aria-label="Remove selected archive"
              className="btn-white-action px-3 py-1.5 text-xs text-slate-600 hover:text-slate-900"
            >
              <X size={13} />
              <span>Remove</span>
            </button>
          </div>

          <button
            type="button"
            id="btn-start-batch"
            onClick={handleSubmit}
            disabled={isLoading}
            className="btn-gold-action w-full py-3.5 text-base font-bold shadow-gold"
          >
            {isLoading ? (
              <>
                <svg
                  className="animate-spin"
                  width="16" height="16"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2.5"
                  strokeLinecap="round"
                  aria-hidden="true"
                >
                  <path d="M21 12a9 9 0 1 1-6.219-8.56" />
                </svg>
                <span>Uploading & Validating Archive…</span>
              </>
            ) : (
              <>
                <span>Start Batch Analysis</span>
                <ArrowRight size={18} className="arrow-shift" />
              </>
            )}
          </button>
        </div>
      )}

    </div>
  )
}

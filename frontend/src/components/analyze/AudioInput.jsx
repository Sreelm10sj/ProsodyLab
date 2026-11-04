import { useState, useCallback, useRef } from 'react'
import { UploadCloud, FileAudio, X, ArrowRight } from 'lucide-react'

/**
 * AudioInput — Clean Dropzone and File Selection View for Single Audio
 *
 * Props:
 *   onAudioSubmit(blob, filename, url) — callback when user submits audio for analysis
 *   onAudioClear()                     — callback to clear audio
 *   isLoading                          — boolean while analysis is in-flight
 */
export default function AudioInput({ onAudioSubmit, onAudioClear, isLoading }) {
  const [file, setFile] = useState(null)
  const [fileUrl, setFileUrl] = useState(null)
  const [dragOver, setDragOver] = useState(false)
  const fileInputRef = useRef(null)

  const handleFile = useCallback((f) => {
    if (!f) return
    const url = URL.createObjectURL(f)
    setFile(f)
    setFileUrl(url)
  }, [])

  const handleClear = useCallback(() => {
    setFile(null)
    if (fileUrl) URL.revokeObjectURL(fileUrl)
    setFileUrl(null)
    if (fileInputRef.current) fileInputRef.current.value = ''
    onAudioClear?.()
  }, [fileUrl, onAudioClear])

  const handleSubmit = useCallback((e) => {
    e?.preventDefault()
    if (file && !isLoading) {
      onAudioSubmit(file, file.name, fileUrl)
    } else if (!file) {
      fileInputRef.current?.click()
    }
  }, [file, fileUrl, isLoading, onAudioSubmit])

  const handleDrop = useCallback((e) => {
    e.preventDefault()
    setDragOver(false)
    const f = e.dataTransfer.files?.[0]
    if (f) handleFile(f)
  }, [handleFile])

  const formatFileSize = (bytes) => {
    if (!bytes) return ''
    const mb = bytes / (1024 * 1024)
    if (mb < 1) return `${(bytes / 1024).toFixed(0)} KB`
    return `${mb.toFixed(1)} MB`
  }

  return (
    <div className="space-y-6">
      {!file ? (
        /* Dropzone */
        <div
          role="region"
          aria-label="Drop audio file here"
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

          <span className="sr-only">Drop audio file here</span>
          <p className="text-base font-semibold text-slate-900 mb-1">
            Drag and drop an audio file here
          </p>
          <p className="text-xs text-slate-500 font-medium mb-5">
            WAV · MP3 · M4A · WebM · OGG (Max 50 MB)
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
            <FileAudio size={14} />
            <span>Choose File</span>
          </button>

          <input
            ref={fileInputRef}
            type="file"
            accept="audio/*,.wav,.mp3,.m4a,.webm,.ogg"
            className="sr-only"
            id="audio-file-input"
            aria-label="Select audio file"
            onChange={(e) => handleFile(e.target.files?.[0])}
          />
        </div>
      ) : (
        /* File Selected Preview */
        <div className="border border-slate-200 rounded-xl p-5 bg-slate-50/40 space-y-4">
          <div className="flex items-center justify-between gap-4">
            <div className="flex items-center gap-3 overflow-hidden">
              <div className="w-10 h-10 rounded-lg bg-amber-100/70 border border-amber-200 flex items-center justify-center text-amber-800 flex-shrink-0">
                <FileAudio size={20} />
              </div>
              <div className="overflow-hidden">
                <p className="text-sm font-bold text-slate-900 truncate">
                  {file.name}
                </p>
                <p className="text-xs text-slate-500 font-mono">
                  {formatFileSize(file.size)}
                </p>
              </div>
            </div>

            <button
              type="button"
              onClick={handleClear}
              disabled={isLoading}
              className="p-1.5 rounded-lg text-slate-400 hover:text-slate-700 hover:bg-slate-200/70 transition-colors"
              aria-label="Remove file"
            >
              <X size={16} />
            </button>
          </div>

          {/* Audio preview player */}
          <audio
            src={fileUrl}
            controls
            className="w-full h-9 rounded-lg"
            aria-label={`Audio preview: ${file.name}`}
          />
        </div>
      )}

      {/* Prominent Analyze → CTA Button */}
      <button
        type="button"
        id="btn-analyze-upload"
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
            <span>Analyzing speech…</span>
          </>
        ) : (
          <>
            <span>Analyze</span>
            <ArrowRight size={18} className="arrow-shift" />
          </>
        )}
      </button>
    </div>
  )
}

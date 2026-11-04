import { useState, useRef, useCallback, useEffect } from 'react'
import { Link } from 'react-router-dom'
import {
  postTranscribe,
  postAnnotate,
  postLexiRepAnalyze,
  createBatchAnnotationJob,
  getBatchAnnotationJob,
  cancelBatchAnnotationJob,
  downloadBatchAnnotations,
} from '../api.js'
import Navigation from '../components/Navigation.jsx'
import AudioInput from '../components/analyze/AudioInput.jsx'
import TranscriptView from '../components/analyze/TranscriptView.jsx'
import AudioWaveformPlayer from '../components/analyze/AudioWaveformPlayer.jsx'
import AnnotationViewer from '../components/analyze/AnnotationViewer.jsx'
import BatchAudioInput from '../components/analyze/BatchAudioInput.jsx'
import BatchProgress from '../components/analyze/BatchProgress.jsx'
import BatchResults from '../components/analyze/BatchResults.jsx'
import Recorder from '../components/analyze/Recorder.jsx'
import {
  ArrowLeft,
  Download,
  RotateCcw,
  Upload,
  Mic,
  Layers,
  ChevronDown,
} from 'lucide-react'

/**
 * AnnotationErrorBoundary — catches render crashes in AnnotationViewer
 * so they show an inline error instead of blanking the whole page.
 */
import { Component } from 'react'
class AnnotationErrorBoundary extends Component {
  constructor(props) { super(props); this.state = { error: null } }
  static getDerivedStateFromError(e) { return { error: e } }
  render() {
    if (this.state.error) {
      return (
        <div
          role="alert"
          className="p-4 rounded-xl border border-red-200 bg-red-50/70 text-red-900 font-mono text-xs"
        >
          <span className="font-bold">Annotation display error: </span>
          {this.state.error.message}
        </div>
      )
    }
    return this.props.children
  }
}

// Feature flag: pitch contour visualization is hidden from UI per user specification
const SHOW_PITCH_CONTOUR = false

export default function Analyze({ showNav = false }) {
  // ── Audio state ────────────────────────────────────────────────────────────
  const [audioBlob, setAudioBlob]  = useState(null)
  const [audioUrl,  setAudioUrl]   = useState(null)
  const [audioName, setAudioName]  = useState(null)
  const [resetKey,  setResetKey]   = useState(0)
  const audioRef = useRef(null)

  // ── Transcription state ───────────────────────────────────────────────────
  const [transcription,     setTranscription]     = useState(null)
  const [words,             setWords]             = useState([])
  const [phrases,           setPhrases]           = useState([])
  const [pitchContour,      setPitchContour]      = useState(null)
  const [pitchStylized,     setPitchStylized]     = useState(null)
  const [utteranceMedianHz, setUtteranceMedianHz] = useState(null)

  // ── Annotation state ───────────────────────────────────────────────────────
  const [annotation,   setAnnotation]   = useState(null)

  // ── LexiRep state ──────────────────────────────────────────────────────────
  const [lexirepNativity, setLexirepNativity] = useState('GER') // 'GER' | 'ITA'
  const [lexirepResult,   setLexirepResult]   = useState(null)
  const [lexirepLoading,  setLexirepLoading]  = useState(false)
  const [lexirepError,    setLexirepError]    = useState(null)
  const activeLexirepRequestIdRef = useRef(0)

  // ── Status ─────────────────────────────────────────────────────────────────
  const [isTranscribing, setIsTranscribing] = useState(false)
  const [isAnnotating,   setIsAnnotating]   = useState(false)
  const [error,          setError]          = useState(null)

  // ── Mode / Tab state (single source of truth for the 3 tabs) ──────────────
  const [activeTab,    setActiveTab]    = useState('upload') // 'upload' | 'record' | 'batch'

  // ── Batch state ────────────────────────────────────────────────────────────
  const [batchJobId, setBatchJobId] = useState(null)
  const [batchJobState, setBatchJobState] = useState(null)
  const [batchNativity, setBatchNativity] = useState('GER') // 'GER' | 'ITA' | null
  const [isBatchUploading, setIsBatchUploading] = useState(false)
  const [isBatchCancelling, setIsBatchCancelling] = useState(false)
  const [isBatchDownloading, setIsBatchDownloading] = useState(false)
  const [batchError, setBatchError] = useState(null)
  const activeBatchPollIdRef = useRef(0)

  // ── Karaoke currentTime + isPlaying ───────────────────────────────────────
  const [currentTime, setCurrentTime] = useState(0)
  const [isPlaying,   setIsPlaying]   = useState(false)

  useEffect(() => {
    const audio = audioRef.current
    if (!audio) return
    const onTimeUpdate = () => setCurrentTime(audio.currentTime)
    const onPlay       = () => setIsPlaying(true)
    const onPause      = () => setIsPlaying(false)
    const onEnded      = () => { setIsPlaying(false); setCurrentTime(0) }
    audio.addEventListener('timeupdate',  onTimeUpdate)
    audio.addEventListener('play',        onPlay)
    audio.addEventListener('pause',       onPause)
    audio.addEventListener('ended',       onEnded)
    return () => {
      audio.removeEventListener('timeupdate',  onTimeUpdate)
      audio.removeEventListener('play',        onPlay)
      audio.removeEventListener('pause',       onPause)
      audio.removeEventListener('ended',       onEnded)
    }
  }, [audioUrl])

  const audioDuration = annotation?.metadata?.duration ??
    annotation?.metadata?.audio_duration_sec ??
    (words.length > 0 ? Math.max(...words.map(w => w.end ?? 0)) : 0)

  // ── Reset All ──────────────────────────────────────────────────────────────
  const resetAll = useCallback(() => {
    activeLexirepRequestIdRef.current++
    if (audioUrl) URL.revokeObjectURL(audioUrl)
    setAudioBlob(null); setAudioUrl(null); setAudioName(null)
    setTranscription(null); setWords([]); setPhrases([])
    setPitchContour(null); setPitchStylized(null); setUtteranceMedianHz(null)
    setAnnotation(null); setError(null)
    setLexirepResult(null); setLexirepError(null); setLexirepLoading(false)
    setIsTranscribing(false); setIsAnnotating(false)
    setLexirepNativity('GER')
    setCurrentTime(0); setIsPlaying(false)
    setResetKey(k => k + 1)
  }, [audioUrl])

  // ── Step 1b: Fetch Lightweight LexiRep Stress (Runs immediately after transcription) ──
  const fetchLexiRepStress = useCallback(async (blob, nativity, filename) => {
    if (!blob) return
    const targetNativity = nativity || lexirepNativity || 'GER'
    const reqId = ++activeLexirepRequestIdRef.current
    setLexirepLoading(true)
    setLexirepError(null)

    try {
      const data = await postLexiRepAnalyze(blob, targetNativity, filename)
      if (activeLexirepRequestIdRef.current === reqId) {
        setLexirepResult({
          nativity: data.nativity ?? targetNativity,
          words: data.words ?? [],
        })
        if (data.words && data.words.length > 0) {
          setWords(prevWords => {
            if (!prevWords || prevWords.length === 0) return data.words
            return prevWords.map((pw, idx) => {
              const lw = data.words[idx]
              if (!lw) return pw
              return {
                ...pw,
                is_polysyllabic: lw.is_polysyllabic,
                syllable_count: lw.syllable_count,
                lexical_stress_index: lw.lexirep_stressed_syllable_index,
                lexirep_stressed_syllable_index: lw.lexirep_stressed_syllable_index,
                syllables: lw.syllables,
                lexirep: {
                  model: data.nativity ?? targetNativity,
                  confidence: lw.syllables?.[0]?.confidence ?? 0.9,
                },
              }
            })
          })
        }
      }
    } catch (err) {
      if (activeLexirepRequestIdRef.current === reqId) {
        setLexirepError('Syllable stress could not be loaded.')
      }
    } finally {
      if (activeLexirepRequestIdRef.current === reqId) {
        setLexirepLoading(false)
      }
    }
  }, [lexirepNativity])

  // ── Step 1: Run Transcription ONLY (Upload/Record -> Transcription) ─────────
  const runTranscription = useCallback(async (blob, filename) => {
    if (!blob) return
    const reqId = ++activeLexirepRequestIdRef.current
    setIsTranscribing(true)
    setError(null)
    setAnnotation(null)
    setLexirepResult(null)
    setLexirepError(null)

    try {
      const data = await postTranscribe(blob, filename)
      if (activeLexirepRequestIdRef.current === reqId) {
        const tx = typeof data.transcription === 'string'
          ? data.transcription
          : data.transcription?.text ?? ''
        setTranscription(tx)
        setWords(data.words ?? [])
        setPhrases(data.phrases ?? [])
        setPitchContour(data.pitch_contour ?? null)
        setPitchStylized(data.pitch_stylized ?? null)
        setUtteranceMedianHz(data.utterance_median_hz ?? null)

        // Immediately request lightweight LexiRep syllable stress without running full annotation
        fetchLexiRepStress(blob, lexirepNativity, filename)
      }
    } catch (err) {
      if (activeLexirepRequestIdRef.current === reqId) {
        setError(err.message)
      }
    } finally {
      if (activeLexirepRequestIdRef.current === reqId) {
        setIsTranscribing(false)
      }
    }
  }, [fetchLexiRepStress, lexirepNativity])

  // ── Step 2: Run Annotation (User clicked "Generate Annotation") ──────────────
  const handleAnnotate = useCallback(async (overrideNativity) => {
    if (!audioBlob) return
    const targetNativity = overrideNativity || lexirepNativity
    if (overrideNativity) {
      setLexirepNativity(overrideNativity)
    }
    const reqId = ++activeLexirepRequestIdRef.current
    setIsAnnotating(true)
    setLexirepLoading(true)
    setError(null)
    setLexirepError(null)

    try {
      const data = await postAnnotate(audioBlob, audioName, targetNativity)
      if (activeLexirepRequestIdRef.current === reqId) {
        setAnnotation(data)
        if (data.words && data.words.length > 0) {
          setWords(data.words)
        }
        if (data.intonation && data.intonation.length > 0) {
          setPhrases(data.intonation)
        }
        setLexirepResult({
          nativity: data.metadata?.speaker_nativity ?? targetNativity,
          words: data.words ?? [],
        })
      }
    } catch (err) {
      if (activeLexirepRequestIdRef.current === reqId) {
        setError(err.message)
      }
    } finally {
      if (activeLexirepRequestIdRef.current === reqId) {
        setIsAnnotating(false)
        setLexirepLoading(false)
      }
    }
  }, [audioBlob, audioName, lexirepNativity])

  const handleNativityChange = useCallback((newNativity) => {
    if (newNativity === lexirepNativity) return
    setLexirepNativity(newNativity)
    if (audioBlob) {
      if (annotation) {
        handleAnnotate(newNativity)
      } else {
        fetchLexiRepStress(audioBlob, newNativity, audioName)
      }
    }
  }, [audioBlob, audioName, annotation, handleAnnotate, lexirepNativity, fetchLexiRepStress])

  const handleAudioSubmit = useCallback(async (blob, filename, url) => {
    setAudioBlob(blob)
    setAudioUrl(url)
    setAudioName(filename)
    setTranscription(null); setWords([]); setPhrases([])
    setPitchContour(null); setPitchStylized(null); setUtteranceMedianHz(null)
    setAnnotation(null); setError(null)
    setLexirepResult(null); setLexirepError(null)
    setCurrentTime(0)

    runTranscription(blob, filename)
  }, [runTranscription])

  const handleRecordingComplete = useCallback(({ blob, url, name }) => {
    handleAudioSubmit(blob, name, url)
  }, [handleAudioSubmit])

  // ── Download Helpers ───────────────────────────────────────────────────────
  const handleDownloadAudio = useCallback(() => {
    if (!audioBlob && !audioUrl) return
    let url = audioUrl
    let revoke = false
    if (audioBlob) {
      url = URL.createObjectURL(audioBlob)
      revoke = true
    }
    const a = document.createElement('a')
    a.href = url
    const defaultExt = audioBlob?.type?.includes('webm') ? '.webm' : '.wav'
    let name = audioName || `recording${defaultExt}`
    if (!/\.[a-zA-Z0-9]+$/.test(name)) {
      name = `${name}${defaultExt}`
    }
    a.download = name
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    if (revoke) {
      setTimeout(() => URL.revokeObjectURL(url), 1000)
    }
  }, [audioBlob, audioUrl, audioName])

  // ── Batch Polling ──────────────────────────────────────────────────────────
  useEffect(() => {
    if (!batchJobId) return
    const status = batchJobState?.status
    if (status && ['completed', 'completed_with_errors', 'failed', 'cancelled'].includes(status)) {
      return
    }

    const pollId = ++activeBatchPollIdRef.current
    const interval = setInterval(async () => {
      try {
        const state = await getBatchAnnotationJob(batchJobId)
        if (activeBatchPollIdRef.current === pollId) {
          setBatchJobState(state)
        }
      } catch (err) {
        if (activeBatchPollIdRef.current === pollId) {
          console.error('Batch polling error:', err)
        }
      }
    }, 1000)

    return () => clearInterval(interval)
  }, [batchJobId, batchJobState?.status])

  const handleBatchSubmit = useCallback(async (file) => {
    setIsBatchUploading(true)
    setBatchError(null)
    activeBatchPollIdRef.current++
    try {
      const resp = await createBatchAnnotationJob(file, batchNativity || null)
      setBatchJobId(resp.batch_id)
      setBatchJobState({
        batch_id: resp.batch_id,
        status: resp.status,
        total_files: resp.total_files,
        completed_files: 0,
        failed_files: 0,
        progress_percent: 0.0,
        files: [],
      })
    } catch (err) {
      setBatchError(err.message)
    } finally {
      setIsBatchUploading(false)
    }
  }, [batchNativity])

  const handleBatchCancel = useCallback(async () => {
    if (!batchJobId || isBatchCancelling) return
    setIsBatchCancelling(true)
    try {
      const state = await cancelBatchAnnotationJob(batchJobId)
      setBatchJobState(state)
    } catch (err) {
      setBatchError(err.message)
    } finally {
      setIsBatchCancelling(false)
    }
  }, [batchJobId, isBatchCancelling])

  const handleBatchDownload = useCallback(async () => {
    if (!batchJobId || isBatchDownloading) return
    setIsBatchDownloading(true)
    try {
      await downloadBatchAnnotations(batchJobId)
    } catch (err) {
      setBatchError(err.message)
    } finally {
      setIsBatchDownloading(false)
    }
  }, [batchJobId, isBatchDownloading])

  const handleBatchReset = useCallback(() => {
    activeBatchPollIdRef.current++
    setBatchJobId(null)
    setBatchJobState(null)
    setBatchError(null)
    setIsBatchUploading(false)
    setIsBatchCancelling(false)
    setIsBatchDownloading(false)
  }, [])

  const isLoading = isTranscribing || isAnnotating
  const hasCompletedResults = transcription !== null || annotation !== null

  return (
    <div className="flex-1 bg-[#F8FAFC] dark:bg-[#0B0F17] text-slate-900 dark:text-slate-100 flex flex-col font-sans">
      
      {/* ── Refined Top Navigation ────────────────────────────────────────── */}
      {showNav && <Navigation />}

      {/* ── Main Container ─────────────────────────────────────────────────── */}
      <main className="max-w-6xl w-full mx-auto px-6 py-10 md:py-12 flex-1 space-y-10">

        {/* ── Global Error Banner ── */}
        {error && (
          <div
            role="alert"
            className="p-4 rounded-xl border border-red-200 bg-red-50/70 text-red-900 text-xs font-mono flex items-center justify-between"
          >
            <span><strong className="text-red-700">Error:</strong> {error}</span>
            <button
              type="button"
              onClick={() => setError(null)}
              className="text-red-600 hover:text-red-900 font-bold ml-4"
              aria-label="Dismiss error"
            >✕</button>
          </div>
        )}

        {/* ─────────────────────────────────────────────────────────────────── */}
        {/* STATE A: IDLE / INPUT WORKBENCH                                     */}
        {/* ─────────────────────────────────────────────────────────────────── */}
        {!hasCompletedResults && (
          <div className="space-y-6">
            
            {/* ── Focused Hero Header (No decorative marketing text or graphic) ── */}
            <div className="pt-2 pb-1">
              <h1 className="text-3xl md:text-4xl font-extrabold text-slate-900 tracking-tight">
                Analyze Speech
              </h1>
            </div>

            {/* ── Main Interactive Workbench Card ── */}
            <div className="bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden">
              
              {/* ── 3 Mutually Exclusive Tabs Bar ── */}
              <div className="px-6 py-4 bg-slate-50/70 border-b border-slate-200/80 flex items-center justify-between flex-wrap gap-4">
                <div
                  role="tablist"
                  aria-label="Audio analysis mode"
                  className="inline-flex flex-wrap sm:flex-nowrap rounded-xl border border-slate-200 p-1 bg-white shadow-2xs gap-1"
                >
                  <button
                    type="button"
                    id="tab-upload"
                    role="tab"
                    aria-selected={activeTab === 'upload'}
                    aria-label="Upload Audio (Single Audio)"
                    onClick={() => setActiveTab('upload')}
                    className={`flex items-center gap-2 text-xs font-semibold px-4 py-2 rounded-lg transition-all ${
                      activeTab === 'upload'
                        ? 'bg-slate-900 text-white shadow-xs'
                        : 'text-slate-600 hover:text-slate-900 hover:bg-slate-50'
                    }`}
                  >
                    <Upload size={14} />
                    <span>Upload Audio</span>
                  </button>
                  <button
                    type="button"
                    id="tab-record"
                    role="tab"
                    aria-selected={activeTab === 'record'}
                    aria-label="Record Audio"
                    onClick={() => setActiveTab('record')}
                    className={`flex items-center gap-2 text-xs font-semibold px-4 py-2 rounded-lg transition-all ${
                      activeTab === 'record'
                        ? 'bg-slate-900 text-white shadow-xs'
                        : 'text-slate-600 hover:text-slate-900 hover:bg-slate-50'
                    }`}
                  >
                    <Mic size={14} />
                    <span>Record Audio</span>
                  </button>
                  <button
                    type="button"
                    id="tab-batch"
                    role="tab"
                    aria-selected={activeTab === 'batch'}
                    aria-label="Upload Audio Batch"
                    onClick={() => setActiveTab('batch')}
                    className={`flex items-center gap-2 text-xs font-semibold px-4 py-2 rounded-lg transition-all ${
                      activeTab === 'batch'
                        ? 'bg-slate-900 text-white shadow-xs'
                        : 'text-slate-600 hover:text-slate-900 hover:bg-slate-50'
                    }`}
                  >
                    <Layers size={14} />
                    <span>Upload Audio Batch</span>
                  </button>
                </div>
              </div>

              {/* ── Tab Content Area ── */}
              <div className="p-6 md:p-8">
                {activeTab === 'upload' && (
                  <AudioInput
                    key={resetKey}
                    onAudioSubmit={handleAudioSubmit}
                    onAudioClear={resetAll}
                    isLoading={isLoading}
                  />
                )}

                {activeTab === 'record' && (
                  <Recorder
                    key={resetKey}
                    onRecordingComplete={handleRecordingComplete}
                    isLoading={isLoading}
                    isTranscribing={isTranscribing}
                    onReset={resetAll}
                  />
                )}

                {activeTab === 'batch' && (
                  <div className="space-y-6">
                    {batchError && (
                      <div
                        role="alert"
                        className="p-4 rounded-xl border border-red-200 bg-red-50/70 text-red-900 text-xs font-mono"
                      >
                        <span className="font-bold">Batch Error: </span>{batchError}
                      </div>
                    )}

                    {/* Batch Upload Screen (with internal Speaker L1 model selector) */}
                    {(!batchJobState || batchJobState.status === 'idle') && (
                      <BatchAudioInput
                        onBatchSubmit={handleBatchSubmit}
                        isLoading={isBatchUploading}
                        batchNativity={batchNativity}
                        onBatchNativityChange={setBatchNativity}
                      />
                    )}

                    {/* Batch Progress Screen */}
                    {batchJobState && (batchJobState.status === 'queued' || batchJobState.status === 'processing') && (
                      <BatchProgress
                        jobState={batchJobState}
                        onCancel={handleBatchCancel}
                        isCancelling={isBatchCancelling}
                      />
                    )}

                    {/* Batch Results Screen */}
                    {batchJobState && ['completed', 'completed_with_errors', 'failed', 'cancelled'].includes(batchJobState.status) && (
                      <BatchResults
                        jobState={batchJobState}
                        onDownload={handleBatchDownload}
                        onReset={handleBatchReset}
                        isDownloading={isBatchDownloading}
                      />
                    )}
                  </div>
                )}
              </div>

            </div>

          </div>
        )}

        {/* ─────────────────────────────────────────────────────────────────── */}
        {/* STATE B: ANALYSIS COMPLETED SCREEN (Screenshot 2)                   */}
        {/* ─────────────────────────────────────────────────────────────────── */}
        {hasCompletedResults && (
          <div className="space-y-6">
            
            {/* ── Top Bar: Back Button, Filename, Duration, Download ── */}
            <div className="flex items-center justify-between gap-4 flex-wrap pb-2">
              
              {/* Back Button & File Title */}
              <div className="flex items-center gap-4">
                <button
                  type="button"
                  onClick={resetAll}
                  className="w-10 h-10 rounded-full bg-white border border-slate-200 flex items-center justify-center text-slate-700 hover:text-slate-900 hover:bg-slate-50 transition-colors shadow-2xs"
                  title="Return to audio input"
                  aria-label="Back to audio input"
                >
                  <ArrowLeft size={18} />
                </button>

                <div>
                  <h2 className="text-xl font-bold text-slate-900 tracking-tight">
                    {audioName ?? 'analyzed_audio.wav'}
                  </h2>
                  <p className="text-xs font-mono text-slate-500 mt-0.5">
                    {audioDuration ? `${audioDuration.toFixed(2)}s` : '—'} · {lexirepNativity === 'GER' ? 'German (GER)' : lexirepNativity === 'ITA' ? 'Italian (ITA)' : 'English'}
                  </p>
                </div>
              </div>

              {/* Action Buttons: Download Audio & Reset */}
              <div className="flex items-center gap-2.5 flex-wrap">
                <button
                  type="button"
                  id="btn-download-audio"
                  onClick={handleDownloadAudio}
                  className="btn-white-action px-3.5 py-2 text-xs font-semibold shadow-2xs flex items-center gap-1.5 focus:outline-none focus-visible:ring-2 focus-visible:ring-slate-900"
                  title="Download audio"
                  aria-label="Download audio"
                >
                  <Download size={14} className="text-slate-600" aria-hidden="true" />
                  <span>Download Audio</span>
                </button>

                {/* Reset analysis button */}
                <button
                  type="button"
                  id="btn-reset-analysis"
                  onClick={resetAll}
                  className="btn-white-action px-3.5 py-2 text-xs font-semibold shadow-2xs flex items-center gap-1.5 focus:outline-none focus-visible:ring-2 focus-visible:ring-slate-900 text-slate-700 hover:text-slate-900"
                  title="Reset analysis"
                  aria-label="Reset analysis"
                >
                  <RotateCcw size={14} className="text-slate-600" aria-hidden="true" />
                  <span>Reset</span>
                </button>
              </div>

            </div>

            {/* ── Single Primary Result View: Transcript & Audio Player ── */}
            <div className="space-y-6">
              {/* 1. Transcript Tokens with Word Pills */}
              <TranscriptView
                transcript={transcription ?? ''}
                words={words}
                phrases={phrases}
                currentTime={currentTime}
                isPlaying={isPlaying}
                lexirepResult={lexirepResult}
                lexirepNativity={lexirepNativity}
                onNativityChange={handleNativityChange}
                lexirepLoading={lexirepLoading || isAnnotating}
                lexirepError={lexirepError}
                onSeek={(t) => {
                  if (audioRef.current) {
                    audioRef.current.currentTime = Math.max(0, Math.min(audioDuration, t))
                  }
                }}
              />

              {/* 2. Custom Audio Waveform & Player */}
              <AudioWaveformPlayer
                audioUrl={audioUrl}
                audioBlob={audioBlob}
                duration={audioDuration}
                currentTime={currentTime}
                isPlaying={isPlaying}
                onSeek={(t) => {
                  if (audioRef.current) audioRef.current.currentTime = t
                }}
                onPlayToggle={(shouldPlay) => {
                  if (!audioRef.current) return
                  if (shouldPlay) audioRef.current.play().catch(() => {})
                  else audioRef.current.pause()
                }}
                audioRef={audioRef}
              />

              {/* 3. Generate Annotation Action Card */}
              <div className="flex items-center justify-between flex-wrap gap-4 p-5 bg-white rounded-2xl border border-slate-200/90 shadow-sm">
                <div>
                  <div className="flex items-center gap-2">
                    <span
                      className={`inline-block w-2.5 h-2.5 rounded-full ${
                        annotation ? 'bg-emerald-500' : 'bg-amber-400 animate-pulse'
                      }`}
                    />
                    <span className="text-xs font-mono font-bold uppercase tracking-wider text-slate-800">
                      {annotation ? 'Annotation Ready' : 'Transcript Ready'}
                    </span>
                  </div>
                  <p className="text-xs text-slate-500 mt-1">
                    {annotation
                      ? 'Detailed lexical stress, syllables, pauses, and intonation are annotated below.'
                      : 'Audio transcription and syllable stress ready. Click Generate Annotation for full prosody analysis.'}
                  </p>
                </div>

                <div className="flex items-center gap-4 flex-wrap">
                  {/* Syllable Stress Model selector */}
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-mono text-slate-500 font-medium">Syllable Stress Model:</span>
                    <div className="inline-flex rounded-lg border border-slate-200 p-0.5 bg-slate-50 shadow-2xs">
                      <button
                        type="button"
                        id="btn-action-nativity-ger"
                        onClick={() => handleNativityChange('GER')}
                        disabled={isAnnotating}
                        aria-pressed={lexirepNativity === 'GER'}
                        className={`text-xs font-semibold px-2.5 py-1 rounded-md transition-all ${
                          lexirepNativity === 'GER'
                            ? 'bg-slate-900 text-white shadow-xs'
                            : 'text-slate-600 hover:text-slate-900 hover:bg-white'
                        }`}
                      >
                        German (GER)
                      </button>
                      <button
                        type="button"
                        id="btn-action-nativity-ita"
                        onClick={() => handleNativityChange('ITA')}
                        disabled={isAnnotating}
                        aria-pressed={lexirepNativity === 'ITA'}
                        className={`text-xs font-semibold px-2.5 py-1 rounded-md transition-all ${
                          lexirepNativity === 'ITA'
                            ? 'bg-slate-900 text-white shadow-xs'
                            : 'text-slate-600 hover:text-slate-900 hover:bg-white'
                        }`}
                      >
                        Italian (ITA)
                      </button>
                    </div>
                  </div>

                  <button
                    type="button"
                    id="btn-generate-annotation"
                    onClick={() => handleAnnotate()}
                    disabled={isAnnotating}
                    className={`flex items-center gap-2.5 px-6 py-3 rounded-xl font-mono text-xs font-bold uppercase tracking-wider transition-all shadow-sm ${
                      isAnnotating
                        ? 'bg-slate-200 text-slate-500 cursor-not-allowed'
                        : 'bg-slate-900 text-white hover:bg-slate-800 hover:shadow-md cursor-pointer'
                    }`}
                  >
                    {isAnnotating ? (
                      <>
                        <span className="inline-block w-3.5 h-3.5 border-2 border-slate-400 border-t-slate-800 rounded-full animate-spin" />
                        <span>Analyzing stress, syllables, pauses and intonation…</span>
                      </>
                    ) : annotation ? (
                      <span>Regenerate Annotation</span>
                    ) : (
                      <span>Generate Annotation</span>
                    )}
                  </button>
                </div>
              </div>

              {/* 4. Structured Annotation Section (Revealed after Generate Annotation) */}
              {annotation && (
                <div id="annotation-results-section" className="space-y-6 pt-2">
                  <AnnotationErrorBoundary>
                    <AnnotationViewer
                      annotation={annotation}
                      lexirepResult={lexirepResult}
                      lexirepNativity={lexirepNativity}
                      onNativityChange={handleNativityChange}
                      lexirepLoading={lexirepLoading || isAnnotating}
                    />
                  </AnnotationErrorBoundary>
                </div>
              )}
            </div>

          </div>
        )}

      </main>

      {/* Embedded Real Audio Element for Playback Control */}
      <audio
        ref={audioRef}
        src={audioUrl || undefined}
        className="sr-only"
        preload="metadata"
        aria-label="Speech playback audio element"
      />

    </div>
  )
}
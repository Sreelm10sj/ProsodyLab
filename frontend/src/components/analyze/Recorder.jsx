import { useState, useRef, useCallback, useEffect } from 'react'
import { Square, Mic, Activity } from 'lucide-react'
import FrequencyVisualizer from './FrequencyVisualizer.jsx'

/**
 * Recorder — Audio-reactive recording workbench with live vocalizer
 *
 * Contract:
 *   onRecordingComplete({ blob, url, name }) — called when recording is finalized
 *   isLoading — boolean
 *   isTranscribing — boolean
 *   onReset — callback
 *
 * States:
 *   - idle: Clean card with Start Recording CTA
 *   - recording: Live frequency vocalizer + status + duration + Stop button
 *   - analyzing: Intermediate "Analyzing Audio" state while transcribing
 */

export default function Recorder({ onRecordingComplete, isLoading, isTranscribing, onReset }) {
  const [recordingState, setRecordingState] = useState('idle') // 'idle' | 'recording' | 'analyzing'
  const [duration, setDuration] = useState(0)
  const [micError, setMicError] = useState(null)
  const [analyserNode, setAnalyserNode] = useState(null)

  const mediaRecorderRef = useRef(null)
  const chunksRef = useRef([])
  const streamRef = useRef(null)
  const audioCtxRef = useRef(null)
  const analyserRef = useRef(null)
  const timerRef = useRef(null)
  const isRecordingSessionRef = useRef(false)

  // Full resource cleanup helper
  const cleanupAudio = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => {
        try { track.stop() } catch (e) {}
      })
      streamRef.current = null
    }
    if (audioCtxRef.current) {
      try { audioCtxRef.current.close() } catch (e) {}
      audioCtxRef.current = null
    }
    analyserRef.current = null
    setAnalyserNode(null)
    if (timerRef.current) {
      clearInterval(timerRef.current)
      timerRef.current = null
    }
  }, [])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      cleanupAudio()
    }
  }, [cleanupAudio])

  // Reset to idle if transcription finishes or fails while in analyzing state
  useEffect(() => {
    if (!isTranscribing && recordingState === 'analyzing') {
      setRecordingState('idle')
      isRecordingSessionRef.current = false
    }
  }, [isTranscribing, recordingState])

  const startRecording = useCallback(async () => {
    setMicError(null)
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      streamRef.current = stream

      // Set up Web Audio API AnalyserNode for live vocalizer
      const AudioCtx = window.AudioContext || window.webkitAudioContext
      const audioCtx = new AudioCtx()
      audioCtxRef.current = audioCtx

      const source = audioCtx.createMediaStreamSource(stream)
      const analyser = audioCtx.createAnalyser()
      analyser.fftSize = 256
      analyser.smoothingTimeConstant = 0.8
      source.connect(analyser)
      analyserRef.current = analyser
      setAnalyserNode(analyser)

      chunksRef.current = []
      const mr = new MediaRecorder(stream)
      mediaRecorderRef.current = mr

      mr.ondataavailable = e => {
        if (e.data && e.data.size > 0) {
          chunksRef.current.push(e.data)
        }
      }

      mr.onstop = () => {
        // Immediately release microphone stream & audio context
        cleanupAudio()

        const blob = new Blob(chunksRef.current, { type: 'audio/webm' })
        const url = URL.createObjectURL(blob)
        const name = `recording-${Date.now()}.webm`
        Object.defineProperty(blob, 'name', { value: name })

        if (typeof onRecordingComplete === 'function') {
          onRecordingComplete({ blob, url, name })
        }
      }

      mr.start(100)
      isRecordingSessionRef.current = true
      setRecordingState('recording')
      setDuration(0)
      timerRef.current = setInterval(() => setDuration(d => d + 1), 1000)
    } catch (err) {
      if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
        setMicError('Microphone access is required to record audio.')
      } else {
        setMicError('Microphone unavailable or not detected.')
      }
      setRecordingState('idle')
    }
  }, [cleanupAudio, onRecordingComplete])

  const stopRecording = useCallback(() => {
    // Immediately transition into analyzing state
    setRecordingState('analyzing')

    if (timerRef.current) {
      clearInterval(timerRef.current)
      timerRef.current = null
    }

    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      try {
        mediaRecorderRef.current.stop()
      } catch (e) {}
    }
  }, [])

  const fmt = (s) => `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`

  const isAnalyzing = recordingState === 'analyzing' || (isTranscribing && isRecordingSessionRef.current)

  return (
    <div className="flex flex-col items-center justify-center py-8 px-4 w-full">
      {micError && (
        <div
          role="alert"
          className="mb-6 p-4 rounded-xl border border-red-200 bg-red-50 text-red-700 text-xs font-mono max-w-md w-full text-center"
        >
          {micError}
        </div>
      )}

      {/* ── STATE 1: LIVE RECORDING ────────────────────────────────────────── */}
      {recordingState === 'recording' && (
        <div className="flex flex-col items-center gap-6 w-full max-w-md animate-in fade-in duration-200">
          {/* Status Badge */}
          <div
            className="inline-flex items-center gap-2.5 px-4 py-1.5 rounded-full bg-red-50 dark:bg-red-950/40 border border-red-200 dark:border-red-900/40 text-red-600 dark:text-red-400"
            aria-live="polite"
          >
            <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse" aria-hidden="true" />
            <span className="font-mono text-xs font-bold uppercase tracking-wider">
              Recording
            </span>
          </div>

          {/* Primary Vocalizer Visualizer */}
          <FrequencyVisualizer analyserNode={analyserNode} isRecording={true} />

          {/* Elapsed Duration Timer */}
          <div
            className="font-mono text-sm font-semibold text-slate-700 dark:text-slate-300 tracking-widest"
            aria-label={`Elapsed time: ${fmt(duration)}`}
          >
            {fmt(duration)}
          </div>

          {/* Stop Recording Action */}
          <button
            type="button"
            id="btn-stop-recording"
            onClick={stopRecording}
            disabled={isLoading || isAnalyzing}
            className="btn-white-action px-6 py-2.5 text-xs font-bold uppercase tracking-wider text-slate-900 dark:text-slate-100 border border-slate-300 dark:border-slate-700 hover:bg-slate-100 dark:hover:bg-slate-800 flex items-center gap-2 shadow-xs transition-transform active:scale-95"
            aria-label="Stop recording"
          >
            <Square size={13} className="fill-current text-red-500" aria-hidden="true" />
            <span>Stop Recording</span>
          </button>
        </div>
      )}

      {/* ── STATE 2: ANALYZING AUDIO (Intermediate Processing) ─────────────── */}
      {isAnalyzing && (
        <div
          className="flex flex-col items-center justify-center py-6 gap-5 max-w-md w-full text-center animate-in fade-in duration-200"
          aria-live="polite"
        >
          <div className="w-14 h-14 rounded-2xl bg-amber-50 dark:bg-amber-950/40 border border-amber-200 dark:border-amber-800/60 flex items-center justify-center text-amber-600 dark:text-amber-400 shadow-xs">
            <Activity size={26} className="animate-pulse" aria-hidden="true" />
          </div>

          <div className="space-y-1.5">
            <h3 className="text-base font-bold text-slate-900 dark:text-slate-100 tracking-tight">
              Analyzing Audio
            </h3>
            <p className="text-xs text-slate-500 dark:text-slate-400 font-normal">
              Preparing your recording for transcription...
            </p>
          </div>

          {/* Subtle indeterminate progress wave */}
          <div className="w-56 h-1.5 bg-slate-100 dark:bg-slate-800 rounded-full overflow-hidden relative">
            <div className="absolute inset-y-0 w-1/3 bg-amber-500 rounded-full animate-indeterminate" />
          </div>

          <p className="text-[11px] font-mono text-slate-400 dark:text-slate-500 mt-1">
            Extracting acoustic tokens and speech prosody
          </p>
        </div>
      )}

      {/* ── STATE 3: IDLE (Before Recording) ───────────────────────────────── */}
      {recordingState === 'idle' && !isAnalyzing && (
        <div className="flex flex-col items-center justify-center py-6 gap-5 max-w-md w-full text-center">
          <div className="w-16 h-16 rounded-2xl bg-slate-100 dark:bg-slate-800 flex items-center justify-center text-slate-500 dark:text-slate-400 shadow-inner">
            <Mic size={28} />
          </div>

          <div className="space-y-1">
            <h3 className="text-sm font-bold text-slate-900 dark:text-slate-100 uppercase tracking-wider font-mono">
              Record Audio
            </h3>
            <p className="text-xs text-slate-500 dark:text-slate-400">
              Click start to record speech directly from your microphone
            </p>
          </div>

          <button
            type="button"
            id="btn-start-recording"
            onClick={startRecording}
            disabled={isLoading}
            className="btn-gold-action px-6 py-2.5 text-xs font-bold uppercase tracking-wider shadow-sm flex items-center gap-2 transition-transform active:scale-95"
            aria-label="Start recording"
          >
            <Mic size={14} aria-hidden="true" />
            <span>Start Recording</span>
          </button>
        </div>
      )}
    </div>
  )
}

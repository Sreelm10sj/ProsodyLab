import React, { useState, useEffect, useRef } from 'react'
import Navigation from '../components/Navigation.jsx'
import TrainingStepper from '../components/train/TrainingStepper.jsx'
import TrainingError from '../components/train/TrainingError.jsx'
import DatasetStep from '../components/train/DatasetStep.jsx'
import AnchorStep from '../components/train/AnchorStep.jsx'
import TrainingConfigStep from '../components/train/TrainingConfigStep.jsx'
import TrainingProgress from '../components/train/TrainingProgress.jsx'
import TrainingResults from '../components/train/TrainingResults.jsx'
import RecentTrainingRuns from '../components/train/RecentTrainingRuns.jsx'
import { formatSeparationScore } from '../components/train/formatters'
import {
  validateTrainingDataset,
  searchTrainingAnchors,
  createTrainingJob,
  getTrainingJob,
  cancelTrainingJob,
} from '../api'

const STORAGE_JOB_KEY = 'lexirep_active_job_id'
const STORAGE_DATASET_KEY = 'lexirep_active_dataset_id'

export default function Train({ showNav = false }) {
  // Stepper state: 1: Dataset, 2: Anchors, 3: Training, 4: Results
  const [currentStep, setCurrentStep] = useState(1)
  const [completedSteps, setCompletedSteps] = useState([])

  // Stage 1: Dataset
  const [selectedFile, setSelectedFile] = useState(null)
  const [isValidatingDataset, setIsValidatingDataset] = useState(false)
  const [datasetId, setDatasetId] = useState(null)
  const [datasetValidation, setDatasetValidation] = useState(null)

  // Stage 2: Anchors
  const [isScanningAnchors, setIsScanningAnchors] = useState(false)
  const [anchorResults, setAnchorResults] = useState(null)
  const [selectedAnchor, setSelectedAnchor] = useState(null)

  // Stage 3 & 4: Job & Training
  const [trainingConfig, setTrainingConfig] = useState({
    language: 'GER',
    device: 'auto',
    seed: 42,
    outer_loops: 13,
    warmup_epochs: 5,
    idec_pretrain_epochs: 10,
    idec_steps_per_loop: 100,
    refinement_epochs: 3,
    supcon_temperature: 0.08,
    learning_rate: 0.001,
    batch_words: 32,
    top_k_rep: 1,
  })
  const [jobId, setJobId] = useState(null)
  const [jobState, setJobState] = useState(null)
  const [isStartingTraining, setIsStartingTraining] = useState(false)
  const [isCancelling, setIsCancelling] = useState(false)

  // General feedback
  const [error, setError] = useState(null)
  const [conflictJobId, setConflictJobId] = useState(null)

  // Refs for stale request protection & lifecycle management
  const requestGenRef = useRef(0)
  const activeJobIdRef = useRef(null)
  const pollingTimerRef = useRef(null)

  // Sync ref with state
  useEffect(() => {
    activeJobIdRef.current = jobId
  }, [jobId])

  // Stop polling helper
  function stopPolling() {
    if (pollingTimerRef.current) {
      clearInterval(pollingTimerRef.current)
      pollingTimerRef.current = null
    }
  }

  // Restore active job from localStorage on initial mount
  useEffect(() => {
    const storedJobId = localStorage.getItem(STORAGE_JOB_KEY)
    if (!storedJobId) return

    let isMounted = true
    async function restoreJob() {
      try {
        const state = await getTrainingJob(storedJobId)
        if (!isMounted) return

        activeJobIdRef.current = storedJobId
        setJobId(storedJobId)
        setJobState(state)

        const status = state.status
        const TERMINAL_STATUSES = ['completed', 'failed', 'cancelled']
        if (status === 'completed') {
          setCurrentStep(4)
          setCompletedSteps([1, 2, 3, 4])
        } else if (!TERMINAL_STATUSES.includes(status)) {
          setCurrentStep(3)
          setCompletedSteps([1, 2])
          startPolling(storedJobId)
        } else {
          // failed or cancelled
          setCurrentStep(3)
          setCompletedSteps([1, 2])
        }
      } catch (err) {
        // Stale or deleted job
        localStorage.removeItem(STORAGE_JOB_KEY)
      }
    }

    restoreJob()

    return () => {
      isMounted = false
      stopPolling()
    }
  }, [])

  // ---------------------------------------------------------------------------
  // Polling loop
  // ---------------------------------------------------------------------------
  function startPolling(targetJobId) {
    stopPolling()
    activeJobIdRef.current = targetJobId

    const poll = async () => {
      // Stale protection: ensure target matches current active job
      if (activeJobIdRef.current !== targetJobId) {
        stopPolling()
        return
      }

      try {
        const state = await getTrainingJob(targetJobId)
        if (activeJobIdRef.current !== targetJobId) return

        setJobState(state)

        if (state.status === 'completed') {
          stopPolling()
          setCurrentStep(4)
          setCompletedSteps((prev) => Array.from(new Set([...prev, 3, 4])))
        } else if (state.status === 'failed' || state.status === 'cancelled') {
          stopPolling()
          if (state.status === 'failed' && state.error) {
            setError(state.error)
          }
        }
      } catch (err) {
        // Keep polling briefly on transient network error, or log
        console.warn('Polling error:', err)
      }
    }

    // Immediate first tick then interval
    poll()
    pollingTimerRef.current = setInterval(poll, 1000)
  }

  // ---------------------------------------------------------------------------
  // Stage 1: Dataset Upload & Validation
  // ---------------------------------------------------------------------------
  async function handleFileSelect(file) {
    setSelectedFile(file)
    setDatasetValidation(null)
    setDatasetId(null)
    setError(null)
    setAnchorResults(null)
    setSelectedAnchor(null)

    if (!file) return

    const gen = ++requestGenRef.current
    setIsValidatingDataset(true)

    try {
      const res = await validateTrainingDataset(file)
      // Stale request protection
      if (gen !== requestGenRef.current) return

      setDatasetId(res.dataset_id)
      setDatasetValidation(res)
      localStorage.setItem(STORAGE_DATASET_KEY, res.dataset_id)
      setCompletedSteps((prev) => Array.from(new Set([...prev, 1])))
    } catch (err) {
      if (gen !== requestGenRef.current) return
      setError(err.message || 'Dataset validation failed.')
    } finally {
      if (gen === requestGenRef.current) {
        setIsValidatingDataset(false)
      }
    }
  }

  function handleResetDataset() {
    requestGenRef.current++
    setSelectedFile(null)
    setDatasetId(null)
    setDatasetValidation(null)
    setAnchorResults(null)
    setSelectedAnchor(null)
    setError(null)
    setCompletedSteps((prev) => prev.filter((s) => s !== 1))
    localStorage.removeItem(STORAGE_DATASET_KEY)
  }

  function handleContinueToAnchors() {
    if (datasetValidation && datasetValidation.valid) {
      setCurrentStep(2)
    }
  }

  // ---------------------------------------------------------------------------
  // Stage 2: Anchor Scanning & Selection
  // ---------------------------------------------------------------------------
  async function handleScanAnchors(maxCandidates = 20) {
    if (!datasetId) return

    const count = typeof maxCandidates === 'number' ? maxCandidates : 20
    const gen = ++requestGenRef.current
    setIsScanningAnchors(true)
    setError(null)

    try {
      const res = await searchTrainingAnchors(datasetId, count)
      if (gen !== requestGenRef.current) return

      setAnchorResults(res)
      // Auto-select rank 1 candidate
      if (res.selected_anchor) {
        setSelectedAnchor(res.selected_anchor)
      } else if (res.top_anchors && res.top_anchors.length > 0) {
        setSelectedAnchor(res.top_anchors[0])
      }
      setCompletedSteps((prev) => Array.from(new Set([...prev, 2])))
    } catch (err) {
      if (gen !== requestGenRef.current) return
      setError(err.message || 'Anchor discovery failed.')
    } finally {
      if (gen === requestGenRef.current) {
        setIsScanningAnchors(false)
      }
    }
  }

  function handleSelectAnchor(anchor) {
    setSelectedAnchor(anchor)
    setCompletedSteps((prev) => Array.from(new Set([...prev, 2])))
  }

  function handleContinueToTraining() {
    if (selectedAnchor) {
      setCurrentStep(3)
    }
  }

  // ---------------------------------------------------------------------------
  // Stage 3: Launch Training
  // ---------------------------------------------------------------------------
  async function handleStartTraining(configPayload) {
    if (!datasetId || !selectedAnchor) return

    setIsStartingTraining(true)
    setError(null)
    setConflictJobId(null)

    const cfg = (configPayload && typeof configPayload === 'object' && configPayload.language)
      ? configPayload
      : trainingConfig

    const payload = {
      dataset_id: datasetId,
      language: cfg.language || 'GER',
      anchor: {
        word_id: selectedAnchor.word_id,
        stressed_syllable_index: selectedAnchor.stressed_syllable_index,
        unstressed_syllable_index: selectedAnchor.unstressed_syllable_index,
      },
      config: {
        device: cfg.device,
        seed: cfg.seed,
        outer_loops: cfg.outer_loops,
        warmup_epochs: cfg.warmup_epochs,
        idec_pretrain_epochs: cfg.idec_pretrain_epochs,
        idec_steps_per_loop: cfg.idec_steps_per_loop,
        refinement_epochs: cfg.refinement_epochs,
        supcon_temperature: cfg.supcon_temperature,
        learning_rate: cfg.learning_rate,
        batch_words: cfg.batch_words,
        top_k_rep: cfg.top_k_rep,
      },
    }

    try {
      const res = await createTrainingJob(payload)
      const newJobId = res.job_id
      setJobId(newJobId)
      localStorage.setItem(STORAGE_JOB_KEY, newJobId)

      // Fetch initial state and start polling
      const initialState = await getTrainingJob(newJobId)
      setJobState(initialState)
      startPolling(newJobId)
    } catch (err) {
      if (err.status === 409) {
        // Concurrency lock conflict: another job is running
        const match = err.message.match(/'([^']+)'/)
        const activeId = match ? match[1] : null
        setConflictJobId(activeId)
        setError('A training session is already running on the server.')
      } else {
        setError(err.message || 'Failed to start training job.')
      }
    } finally {
      setIsStartingTraining(false)
    }
  }

  async function handleCancelJob() {
    if (!jobId || isCancelling) return
    setIsCancelling(true)
    try {
      const res = await cancelTrainingJob(jobId)
      setJobState(res)
      stopPolling()
    } catch (err) {
      setError(err.message || 'Failed to cancel training job.')
    } finally {
      setIsCancelling(false)
    }
  }

  // Attach to existing active conflict job
  async function handleAttachToConflictJob(targetJobId) {
    if (!targetJobId) return
    try {
      const state = await getTrainingJob(targetJobId)
      setJobId(targetJobId)
      setJobState(state)
      localStorage.setItem(STORAGE_JOB_KEY, targetJobId)
      setConflictJobId(null)
      setError(null)
      setCurrentStep(3)
      startPolling(targetJobId)
    } catch (err) {
      setError(`Failed to attach to job ${targetJobId}: ${err.message}`)
    }
  }

  // Load a job clicked from RecentTrainingRuns
  async function handleSelectPastJob(pastJobId) {
    try {
      const state = await getTrainingJob(pastJobId)
      setJobId(pastJobId)
      setJobState(state)
      localStorage.setItem(STORAGE_JOB_KEY, pastJobId)

      const TERMINAL_STATUSES = ['completed', 'failed', 'cancelled']
      if (state.status === 'completed') {
        setCurrentStep(4)
        setCompletedSteps([1, 2, 3, 4])
      } else if (!TERMINAL_STATUSES.includes(state.status)) {
        setCurrentStep(3)
        startPolling(pastJobId)
      } else {
        setCurrentStep(3)
      }
    } catch (err) {
      setError(`Failed to load job ${pastJobId}: ${err.message}`)
    }
  }

  // ---------------------------------------------------------------------------
  // Reset Entire Workflow
  // ---------------------------------------------------------------------------
  function handleResetWorkflow() {
    stopPolling()
    requestGenRef.current++
    localStorage.removeItem(STORAGE_JOB_KEY)
    localStorage.removeItem(STORAGE_DATASET_KEY)

    setCurrentStep(1)
    setCompletedSteps([])
    setSelectedFile(null)
    setDatasetId(null)
    setDatasetValidation(null)
    setAnchorResults(null)
    setSelectedAnchor(null)
    setJobId(null)
    setJobState(null)
    setError(null)
    setConflictJobId(null)
  }

  function handleStepClick(step) {
    if (step === currentStep) return
    // Allow jumping back to earlier steps or to completed steps
    if (step < currentStep || completedSteps.includes(step)) {
      setCurrentStep(step)
    }
  }

  return (
    <div className="flex-1 bg-[#F8FAFC] dark:bg-[#0B0F17] text-slate-900 dark:text-slate-100 flex flex-col font-sans">
      {/* Refined Navigation Header (Analyze | Train Model) */}
      {showNav && <Navigation />}

      {/* Main Container */}
      <main className="max-w-6xl w-full mx-auto px-6 py-10 md:py-12 flex-1 space-y-8">
        {/* Title Area */}
        <div className="space-y-1.5 pt-2 pb-1">
          <h1 className="text-3xl md:text-4xl font-extrabold text-slate-900 tracking-tight">
            Train Model
          </h1>
          <p className="text-sm text-slate-500 max-w-2xl leading-relaxed">
            Train self-supervised bottleneck acoustic representations for discrete syllable-level stress classification using contrastive learning and iterative DEC clustering.
          </p>
        </div>

        {/* 4-Stage Stepper */}
        <TrainingStepper
          currentStep={currentStep}
          completedSteps={completedSteps}
          onStepClick={handleStepClick}
        />

        {/* Global Error & Concurrency Alert */}
        {error && (
          <TrainingError
            error={error}
            onDismiss={() => {
              setError(null)
              setConflictJobId(null)
            }}
          />
        )}

        {conflictJobId && (
          <div className="p-4 rounded-xl border border-amber-200 bg-amber-50/80 text-amber-900 flex flex-wrap items-center justify-between gap-4 shadow-2xs">
            <div className="space-y-1">
              <div className="text-xs font-bold uppercase tracking-wider text-amber-950">
                Active Training Job Detected
              </div>
              <p className="text-xs text-amber-900">
                Server is currently executing job <code className="font-mono font-bold">{conflictJobId}</code>. Only one job is permitted concurrently.
              </p>
            </div>
            <button
              type="button"
              onClick={() => handleAttachToConflictJob(conflictJobId)}
              className="btn-gold-action px-4 py-2 text-xs font-semibold"
            >
              Attach to Running Job
            </button>
          </div>
        )}

        {/* Collapsed Summaries of Completed Prior Stages */}
        {currentStep > 1 && datasetValidation && (
          <div className="rounded-xl border border-slate-200/90 bg-white px-5 py-3 flex flex-wrap items-center justify-between gap-3 text-xs shadow-2xs">
            <div className="flex items-center gap-2.5">
              <span className="w-2 h-2 rounded-full bg-emerald-500 ring-4 ring-emerald-50" />
              <span className="font-semibold text-slate-500 uppercase tracking-wider text-[11px]">01 Dataset:</span>
              <span className="font-bold text-slate-900 font-mono">{datasetValidation.filename}</span>
              <span className="text-slate-500">
                ({datasetValidation.syllable_count.toLocaleString()} syllables, {datasetValidation.word_count.toLocaleString()} words)
              </span>
            </div>
            {currentStep !== 1 && !jobId && (
              <button
                type="button"
                onClick={() => setCurrentStep(1)}
                className="btn-white-action px-3 py-1 text-xs font-medium text-slate-600 hover:text-slate-900"
              >
                Change Dataset
              </button>
            )}
          </div>
        )}

        {currentStep > 2 && selectedAnchor && (
          <div className="rounded-xl border border-slate-200/90 bg-white px-5 py-3 flex flex-wrap items-center justify-between gap-3 text-xs shadow-2xs">
            <div className="flex items-center gap-2.5">
              <span className="w-2 h-2 rounded-full bg-emerald-500 ring-4 ring-emerald-50" />
              <span className="font-semibold text-slate-500 uppercase tracking-wider text-[11px]">02 Anchor:</span>
              <span className="font-bold text-slate-900 font-mono">Word #{selectedAnchor.word_id}</span>
              <span className="text-slate-500 font-mono">
                (syls #{selectedAnchor.stressed_syllable_index} / #{selectedAnchor.unstressed_syllable_index}, S_train: {formatSeparationScore(selectedAnchor.separation_score)})
              </span>
            </div>
            {currentStep !== 2 && !jobId && (
              <button
                type="button"
                onClick={() => setCurrentStep(2)}
                className="btn-white-action px-3 py-1 text-xs font-medium text-slate-600 hover:text-slate-900"
              >
                Change Anchor
              </button>
            )}
          </div>
        )}

        {/* Active Stage Workflow Content */}
        <div className="space-y-6">
          {/* Stage 1: Dataset */}
          {currentStep === 1 && (
            <DatasetStep
              file={selectedFile}
              validation={datasetValidation}
              isValidating={isValidatingDataset}
              onFileSelect={handleFileSelect}
              onResetFile={handleResetDataset}
              onContinue={handleContinueToAnchors}
            />
          )}

          {/* Stage 2: Anchor Discovery */}
          {currentStep === 2 && (
            <AnchorStep
              datasetValidation={datasetValidation}
              anchorResults={anchorResults}
              selectedAnchor={selectedAnchor}
              isScanning={isScanningAnchors}
              onScanAnchors={handleScanAnchors}
              onSelectAnchor={handleSelectAnchor}
              onContinue={handleContinueToTraining}
              onBack={() => setCurrentStep(1)}
            />
          )}

          {/* Stage 3: Training Config OR Active Training Progress */}
          {currentStep === 3 && (
            <>
              {!jobId ? (
                <TrainingConfigStep
                  datasetValidation={datasetValidation}
                  selectedAnchor={selectedAnchor}
                  config={trainingConfig}
                  onChangeConfig={setTrainingConfig}
                  onStartTraining={() => handleStartTraining(trainingConfig)}
                  onBack={() => setCurrentStep(2)}
                  isLaunching={isStartingTraining}
                />
              ) : (
                <TrainingProgress
                  jobId={jobId}
                  jobState={jobState}
                  onCancelJob={handleCancelJob}
                  isCancelling={isCancelling}
                />
              )}
            </>
          )}

          {/* Stage 4: Results & Export */}
          {currentStep === 4 && (
            <TrainingResults
              jobId={jobId}
              jobState={jobState}
              onResetWorkflow={handleResetWorkflow}
            />
          )}
        </div>

        {/* Bottom Section: Recent Training Runs */}
        <div className="pt-8 border-t border-slate-200/80">
          <RecentTrainingRuns
            currentJobId={jobId}
            onSelectJob={handleSelectPastJob}
          />
        </div>
      </main>
    </div>
  )
}

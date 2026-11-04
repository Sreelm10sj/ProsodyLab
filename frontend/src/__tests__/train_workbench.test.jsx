import React from 'react'
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import Train from '../pages/Train'
import * as api from '../api'

// Mock API layer
vi.mock('../api', () => ({
  validateTrainingDataset: vi.fn(),
  searchTrainingAnchors: vi.fn(),
  createTrainingJob: vi.fn(),
  getTrainingJob: vi.fn(),
  listTrainingJobs: vi.fn(),
  cancelTrainingJob: vi.fn(),
  downloadTrainingBundle: vi.fn(),
  downloadTrainingArtifact: vi.fn(),
}))

const MOCK_VALIDATION_GER = {
  dataset_id: 'ds_test_ger_123',
  valid: true,
  filename: 'ger_contextual_cache.npz',
  syllable_count: 7924,
  word_count: 3447,
  stressed_count: 3447,
  unstressed_count: 4477,
  stress_ratio: 0.435,
  bisyllabic_anchor_candidate_count: 2628,
  rejected_word_count: 819,
  feature_shape: [7924, 768],
  feature_dtype: 'float32',
  warnings: [],
  errors: [],
}

const MOCK_ANCHORS_RESPONSE = {
  dataset_id: 'ds_test_ger_123',
  total_training_syllables: 7924,
  total_training_words: 3447,
  bisyllabic_candidates: 2628,
  rejected_words: 819,
  scan_time_seconds: 0.85,
  top_anchors: [
    {
      rank: 1,
      word_id: 1181,
      stressed_syllable_index: 2713,
      unstressed_syllable_index: 2714,
      separation_score: 0.6822,
    },
    {
      rank: 2,
      word_id: 2611,
      stressed_syllable_index: 6006,
      unstressed_syllable_index: 6007,
      separation_score: 0.6764,
    },
  ],
  selected_anchor: {
    rank: 1,
    word_id: 1181,
    stressed_syllable_index: 2713,
    unstressed_syllable_index: 2714,
    separation_score: 0.6822,
  },
}

describe('LexiRep Training Workbench (/train)', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    localStorage.clear()
    api.listTrainingJobs.mockResolvedValue([])
  })

  afterEach(() => {
    vi.useRealTimers()
    vi.restoreAllMocks()
  })

  // 1. Upload UI
  it('1. renders dataset upload dropzone with expected formats', async () => {
    await act(async () => {
      render(
        <MemoryRouter>
          <Train />
        </MemoryRouter>
      )
    })

    expect(screen.getByText(/Upload LexiRep feature cache/i)).toBeDefined()
    expect(screen.getByText(/\.npz · up to 250 MB/i)).toBeDefined()
    expect(screen.getByText('X_tr')).toBeDefined()
    expect(screen.getByText('Y_tr')).toBeDefined()
    expect(screen.getByText('W_tr')).toBeDefined()
    const continueBtn = screen.getByRole('button', { name: /Continue to Anchor Discovery/i })
    expect(continueBtn.getAttribute('disabled')).not.toBeNull()
  })

  // 2. Validation success state
  it('2. displays validation summary on successful upload and enables continue', async () => {
    api.validateTrainingDataset.mockResolvedValue(MOCK_VALIDATION_GER)

    let container
    await act(async () => {
      const res = render(
        <MemoryRouter>
          <Train />
        </MemoryRouter>
      )
      container = res.container
    })

    const fileInput = container.querySelector('input[type="file"]')
    const dummyFile = new File(['fake-npz-content'], 'ger_contextual_cache.npz', { type: 'application/octet-stream' })

    await act(async () => {
      fireEvent.change(fileInput, { target: { files: [dummyFile] } })
    })

    await waitFor(() => {
      expect(screen.getByText('7,924')).toBeDefined()
      expect(screen.getByText('3,447')).toBeDefined()
      expect(screen.getByText('2,628')).toBeDefined()
      expect(screen.getByText('768-D')).toBeDefined()
      expect(screen.getByText(/Dataset structurally and linguistically verified/i)).toBeDefined()
    })

    const continueBtn = screen.getByRole('button', { name: /Continue to Anchor Discovery/i })
    expect(continueBtn.getAttribute('disabled')).toBeNull()
  })

  // 3. Validation error state
  it('3. displays validation error when backend rejects file', async () => {
    api.validateTrainingDataset.mockRejectedValue(new Error('Dataset missing required key: X_tr'))

    let container
    await act(async () => {
      const res = render(
        <MemoryRouter>
          <Train />
        </MemoryRouter>
      )
      container = res.container
    })

    const fileInput = container.querySelector('input[type="file"]')
    const badFile = new File(['invalid'], 'corrupt.npz', { type: 'application/octet-stream' })

    await act(async () => {
      fireEvent.change(fileInput, { target: { files: [badFile] } })
    })

    await waitFor(() => {
      expect(screen.getByText(/Dataset missing required key: X_tr/i)).toBeDefined()
    })
  })

  // 4. Anchor search loading & 5. Anchor results rendering
  it('4 & 5. executes anchor scan and renders candidate leaderboard', async () => {
    api.validateTrainingDataset.mockResolvedValue(MOCK_VALIDATION_GER)
    api.searchTrainingAnchors.mockResolvedValue(MOCK_ANCHORS_RESPONSE)

    let container
    await act(async () => {
      const res = render(
        <MemoryRouter>
          <Train />
        </MemoryRouter>
      )
      container = res.container
    })

    const fileInput = container.querySelector('input[type="file"]')
    await act(async () => {
      fireEvent.change(fileInput, {
        target: { files: [new File([''], 'ger_contextual_cache.npz')] },
      })
    })

    const continueToAnchors = await screen.findByRole('button', { name: /Continue to Anchor Discovery/i })
    await act(async () => {
      fireEvent.click(continueToAnchors)
    })

    expect(screen.getByText(/Stage 02 — Anchor Discovery/i)).toBeDefined()

    const scanBtn = screen.getByRole('button', { name: /Scan Training Set/i })
    await act(async () => {
      fireEvent.click(scanBtn)
    })

    await waitFor(() => {
      expect(screen.getAllByText('1181').length).toBeGreaterThan(0)
      expect(screen.getByText('2611')).toBeDefined()
      expect(screen.getAllByText(/68\.22%/i).length).toBeGreaterThan(0)
      expect(screen.getByText('67.64%')).toBeDefined()
    })

    expect(api.searchTrainingAnchors).toHaveBeenCalledWith('ds_test_ger_123', 20)
  })

  // 6. Selecting a non-rank-1 anchor
  it('6. allows selecting a non-rank-1 candidate anchor', async () => {
    api.validateTrainingDataset.mockResolvedValue(MOCK_VALIDATION_GER)
    api.searchTrainingAnchors.mockResolvedValue(MOCK_ANCHORS_RESPONSE)

    let container
    await act(async () => {
      const res = render(
        <MemoryRouter>
          <Train />
        </MemoryRouter>
      )
      container = res.container
    })

    const fileInput = container.querySelector('input[type="file"]')
    await act(async () => {
      fireEvent.change(fileInput, {
        target: { files: [new File([''], 'ger_contextual_cache.npz')] },
      })
    })

    await act(async () => {
      fireEvent.click(await screen.findByRole('button', { name: /Continue to Anchor Discovery/i }))
    })
    await act(async () => {
      fireEvent.click(await screen.findByRole('button', { name: /Scan Training Set/i }))
    })

    await waitFor(() => {
      expect(screen.getByText('2611')).toBeDefined()
    })

    // Click "Use" button on rank 2
    const useButtons = screen.getAllByRole('button', { name: /Use/i })
    expect(useButtons.length).toBeGreaterThan(0)
    await act(async () => {
      fireEvent.click(useButtons[0])
    })

    // Rank 2 should now be selected
    expect(screen.getByText(/Selected Anchor Pair/i)).toBeDefined()
    expect(screen.getAllByText('2611').length).toBeGreaterThan(0)
  })

  // 7. GER protocol resolution display & 8. ITA protocol resolution display
  it('7 & 8. displays resolved collaborator protocols for GER and ITA correctly', async () => {
    api.validateTrainingDataset.mockResolvedValue(MOCK_VALIDATION_GER)
    api.searchTrainingAnchors.mockResolvedValue(MOCK_ANCHORS_RESPONSE)

    let container
    await act(async () => {
      const res = render(
        <MemoryRouter>
          <Train />
        </MemoryRouter>
      )
      container = res.container
    })

    const fileInput = container.querySelector('input[type="file"]')
    await act(async () => {
      fireEvent.change(fileInput, {
        target: { files: [new File([''], 'ger_contextual_cache.npz')] },
      })
    })

    await act(async () => {
      fireEvent.click(await screen.findByRole('button', { name: /Continue to Anchor Discovery/i }))
    })
    await act(async () => {
      fireEvent.click(await screen.findByRole('button', { name: /Scan Training Set/i }))
    })

    // Wait for anchor scan completion
    await waitFor(() => {
      expect(screen.getAllByText('1181').length).toBeGreaterThan(0)
    })

    const continueToTraining = await screen.findByRole('button', { name: /Continue to Training/i })
    await act(async () => {
      fireEvent.click(continueToTraining)
    })

    // Stage 3 Config should display GER defaults
    await waitFor(() => {
      expect(screen.getByText(/Active Protocol: GER/i)).toBeDefined()
      expect(screen.getByText(/Top 1/i)).toBeDefined()
      expect(screen.getByText(/32 words/i)).toBeDefined()
    })

    // Switch language to ITA
    const itaBtn = screen.getByRole('button', { name: /^ITA$/i })
    await act(async () => {
      fireEvent.click(itaBtn)
    })

    // Should now show ITA defaults
    expect(screen.getByText(/Active Protocol: ITA/i)).toBeDefined()
    expect(screen.getByText(/Top 10/i)).toBeDefined()
    expect(screen.getByText(/48 words/i)).toBeDefined()
  })

  // 9. Training configuration submission & 10. Active progress rendering
  it('9 & 10. submits training configuration and transitions to active progress UI', async () => {
    api.validateTrainingDataset.mockResolvedValue(MOCK_VALIDATION_GER)
    api.searchTrainingAnchors.mockResolvedValue(MOCK_ANCHORS_RESPONSE)
    api.createTrainingJob.mockResolvedValue({ job_id: 'job_active_789', status: 'queued' })
    api.getTrainingJob.mockResolvedValue({
      job_id: 'job_active_789',
      status: 'training',
      current_step: 'Self-training iteration 3 of 13',
      progress_percent: 45.0,
      loop: 3,
      total_loops: 13,
      elapsed_seconds: 42.0,
      pseudo_label_changes: 18,
      loss_history: [{ loop: 1, supcon_loss: 0.245, idec_loss: 0.123, label_changes: 30, elapsed: 14 }],
    })

    let container
    await act(async () => {
      const res = render(
        <MemoryRouter>
          <Train />
        </MemoryRouter>
      )
      container = res.container
    })

    const fileInput = container.querySelector('input[type="file"]')
    await act(async () => {
      fireEvent.change(fileInput, {
        target: { files: [new File([''], 'ger_contextual_cache.npz')] },
      })
    })

    await act(async () => {
      fireEvent.click(await screen.findByRole('button', { name: /Continue to Anchor Discovery/i }))
    })
    await act(async () => {
      fireEvent.click(await screen.findByRole('button', { name: /Scan Training Set/i }))
    })

    await waitFor(() => {
      expect(screen.getAllByText('1181').length).toBeGreaterThan(0)
    })

    await act(async () => {
      fireEvent.click(await screen.findByRole('button', { name: /Continue to Training/i }))
    })

    const startBtn = await screen.findByRole('button', { name: /Start Training/i })
    await act(async () => {
      fireEvent.click(startBtn)
    })

    await waitFor(() => {
      expect(screen.getByText(/Training LexiRep Model/i)).toBeDefined()
      expect(screen.getByText(/Self-training iteration 3 of 13/i)).toBeDefined()
      expect(screen.getByText('45.0%')).toBeDefined()
      expect(screen.getByText('0.1230')).toBeDefined()
    })
  })

  // 11. Polling updates & 12. Polling stops on completion
  it('11 & 12. polls active training job and stops when status is completed, moving to Stage 4', async () => {
    api.validateTrainingDataset.mockResolvedValue(MOCK_VALIDATION_GER)
    api.searchTrainingAnchors.mockResolvedValue(MOCK_ANCHORS_RESPONSE)
    api.createTrainingJob.mockResolvedValue({ job_id: 'job_done_101', status: 'queued' })

    api.getTrainingJob
      .mockResolvedValueOnce({
        job_id: 'job_done_101',
        status: 'training',
        current_step: 'Self-training iteration 12 of 13',
        progress_percent: 92.0,
        loop: 12,
        total_loops: 13,
        elapsed_seconds: 120.0,
      })
      .mockResolvedValue({
        job_id: 'job_done_101',
        status: 'completed',
        current_step: 'Complete',
        progress_percent: 100.0,
        loop: 13,
        total_loops: 13,
        elapsed_seconds: 135.0,
        language: 'GER',
        selected_anchor: { word_id: 1181, stressed_syllable_index: 2713, unstressed_syllable_index: 2714, separation_score: 0.6822 },
        loss_history: [
          { loop: 13, supcon_loss: 0.0512, idec_loss: 0.0194, label_changes: 0, elapsed: 135 },
        ],
      })

    let container
    await act(async () => {
      const res = render(
        <MemoryRouter>
          <Train />
        </MemoryRouter>
      )
      container = res.container
    })

    const fileInput = container.querySelector('input[type="file"]')
    await act(async () => {
      fireEvent.change(fileInput, { target: { files: [new File([''], 'ger.npz')] } })
    })

    await act(async () => {
      fireEvent.click(await screen.findByRole('button', { name: /Continue to Anchor Discovery/i }))
    })
    await act(async () => {
      fireEvent.click(await screen.findByRole('button', { name: /Scan Training Set/i }))
    })

    await waitFor(() => {
      expect(screen.getAllByText('1181').length).toBeGreaterThan(0)
    })

    await act(async () => {
      fireEvent.click(await screen.findByRole('button', { name: /Continue to Training/i }))
    })
    await act(async () => {
      fireEvent.click(await screen.findByRole('button', { name: /Start Training/i }))
    })

    await waitFor(() => {
      expect(screen.getByText(/Training Complete/i)).toBeDefined()
      expect(screen.getByText(/Production Bundle Verified/i)).toBeDefined()
      expect(screen.getByRole('button', { name: /Download Model Bundle/i })).toBeDefined()
    }, { timeout: 3000 })
  })

  // 13. Polling stops on failure
  it('13. stops polling when job fails', async () => {
    api.validateTrainingDataset.mockResolvedValue(MOCK_VALIDATION_GER)
    api.searchTrainingAnchors.mockResolvedValue(MOCK_ANCHORS_RESPONSE)
    api.createTrainingJob.mockResolvedValue({ job_id: 'job_fail_001', status: 'queued' })

    api.getTrainingJob
      .mockResolvedValueOnce({
        job_id: 'job_fail_001',
        status: 'training',
        current_step: 'Pretraining',
        progress_percent: 20.0,
      })
      .mockResolvedValue({
        job_id: 'job_fail_001',
        status: 'failed',
        current_step: 'Failed',
        error: 'CUDA out of memory error in worker',
        progress_percent: 20.0,
      })

    let container
    await act(async () => {
      const res = render(
        <MemoryRouter>
          <Train />
        </MemoryRouter>
      )
      container = res.container
    })

    const fileInput = container.querySelector('input[type="file"]')
    await act(async () => {
      fireEvent.change(fileInput, { target: { files: [new File([''], 'ger.npz')] } })
    })

    await act(async () => {
      fireEvent.click(await screen.findByRole('button', { name: /Continue to Anchor Discovery/i }))
    })
    await act(async () => {
      fireEvent.click(await screen.findByRole('button', { name: /Scan Training Set/i }))
    })

    await waitFor(() => {
      expect(screen.getAllByText('1181').length).toBeGreaterThan(0)
    })

    await act(async () => {
      fireEvent.click(await screen.findByRole('button', { name: /Continue to Training/i }))
    })
    await act(async () => {
      fireEvent.click(await screen.findByRole('button', { name: /Start Training/i }))
    })

    await waitFor(() => {
      expect(screen.getAllByText(/CUDA out of memory error in worker/i).length).toBeGreaterThan(0)
    })
  })

  // 14. Cancellation
  it('14. supports job cancellation with confirmation dialog', async () => {
    api.validateTrainingDataset.mockResolvedValue(MOCK_VALIDATION_GER)
    api.searchTrainingAnchors.mockResolvedValue(MOCK_ANCHORS_RESPONSE)
    api.createTrainingJob.mockResolvedValue({ job_id: 'job_cancel_99', status: 'queued' })
    api.getTrainingJob.mockResolvedValue({
      job_id: 'job_cancel_99',
      status: 'training',
      current_step: 'Self-training loop 4 of 13',
      progress_percent: 30.0,
    })
    api.cancelTrainingJob.mockResolvedValue({
      job_id: 'job_cancel_99',
      status: 'cancelled',
      current_step: 'Cancelled by user',
      progress_percent: 30.0,
    })

    let container
    await act(async () => {
      const res = render(
        <MemoryRouter>
          <Train />
        </MemoryRouter>
      )
      container = res.container
    })

    const fileInput = container.querySelector('input[type="file"]')
    await act(async () => {
      fireEvent.change(fileInput, { target: { files: [new File([''], 'ger.npz')] } })
    })

    await act(async () => {
      fireEvent.click(await screen.findByRole('button', { name: /Continue to Anchor Discovery/i }))
    })
    await act(async () => {
      fireEvent.click(await screen.findByRole('button', { name: /Scan Training Set/i }))
    })

    await waitFor(() => {
      expect(screen.getAllByText('1181').length).toBeGreaterThan(0)
    })

    await act(async () => {
      fireEvent.click(await screen.findByRole('button', { name: /Continue to Training/i }))
    })
    await act(async () => {
      fireEvent.click(await screen.findByRole('button', { name: /Start Training/i }))
    })

    const cancelBtn = await screen.findByRole('button', { name: /Cancel Training/i })
    await act(async () => {
      fireEvent.click(cancelBtn)
    })

    expect(screen.getByText(/Cancel this training run\?/i)).toBeDefined()
    const confirmBtn = screen.getByRole('button', { name: /Confirm Cancel/i })
    await act(async () => {
      fireEvent.click(confirmBtn)
    })

    expect(api.cancelTrainingJob).toHaveBeenCalledWith('job_cancel_99')
  })

  // 15. Stale request protection
  it('15. protects against race conditions when upload A is followed by upload B and response A arrives after B', async () => {
    let resolveA
    const promiseA = new Promise((resolve) => { resolveA = resolve })
    let resolveB
    const promiseB = new Promise((resolve) => { resolveB = resolve })

    api.validateTrainingDataset
      .mockReturnValueOnce(promiseA)
      .mockReturnValueOnce(promiseB)

    let container
    await act(async () => {
      const res = render(
        <MemoryRouter>
          <Train />
        </MemoryRouter>
      )
      container = res.container
    })

    const fileInput = container.querySelector('input[type="file"]')

    // Upload A
    await act(async () => {
      fireEvent.change(fileInput, { target: { files: [new File([''], 'file_A.npz')] } })
    })

    // Upload B
    await act(async () => {
      fireEvent.change(fileInput, { target: { files: [new File([''], 'file_B.npz')] } })
    })

    // Response B resolves FIRST
    await act(async () => {
      resolveB({
        ...MOCK_VALIDATION_GER,
        dataset_id: 'ds_B',
        filename: 'file_B.npz',
        word_count: 500,
      })
    })

    // Response A resolves LATER
    await act(async () => {
      resolveA({
        ...MOCK_VALIDATION_GER,
        dataset_id: 'ds_A',
        filename: 'file_A.npz',
        word_count: 9999,
      })
    })

    // Verify file B's state remains active, NOT overwritten by file A
    await waitFor(() => {
      expect(screen.getByText('file_B.npz')).toBeDefined()
      expect(screen.getByText('500')).toBeDefined()
      expect(screen.queryByText('file_A.npz')).toBeNull()
      expect(screen.queryByText('9,999')).toBeNull()
    })
  })

  // 16 & 17. Bundle download & Artifact download
  it('16 & 17. triggers bundle and individual artifact downloads in completed results state', async () => {
    api.validateTrainingDataset.mockResolvedValue(MOCK_VALIDATION_GER)
    api.searchTrainingAnchors.mockResolvedValue(MOCK_ANCHORS_RESPONSE)
    api.createTrainingJob.mockResolvedValue({ job_id: 'job_export_1', status: 'queued' })
    api.getTrainingJob.mockResolvedValue({
      job_id: 'job_export_1',
      status: 'completed',
      current_step: 'Complete',
      progress_percent: 100.0,
      loop: 13,
      total_loops: 13,
      elapsed_seconds: 88.0,
      language: 'GER',
      selected_anchor: { word_id: 1181, stressed_syllable_index: 2713, unstressed_syllable_index: 2714, separation_score: 0.6822 },
      loss_history: [{ loop: 13, supcon_loss: 0.04, idec_loss: 0.02, label_changes: 0, elapsed: 88 }],
    })
    api.downloadTrainingBundle.mockResolvedValue()
    api.downloadTrainingArtifact.mockResolvedValue()

    let container
    await act(async () => {
      const res = render(
        <MemoryRouter>
          <Train />
        </MemoryRouter>
      )
      container = res.container
    })

    const fileInput = container.querySelector('input[type="file"]')
    await act(async () => {
      fireEvent.change(fileInput, { target: { files: [new File([''], 'ger.npz')] } })
    })

    await act(async () => {
      fireEvent.click(await screen.findByRole('button', { name: /Continue to Anchor Discovery/i }))
    })
    await act(async () => {
      fireEvent.click(await screen.findByRole('button', { name: /Scan Training Set/i }))
    })

    await waitFor(() => {
      expect(screen.getAllByText('1181').length).toBeGreaterThan(0)
    })

    await act(async () => {
      fireEvent.click(await screen.findByRole('button', { name: /Continue to Training/i }))
    })
    await act(async () => {
      fireEvent.click(await screen.findByRole('button', { name: /Start Training/i }))
    })

    const downloadBundleBtn = await screen.findByRole('button', { name: /Download Model Bundle/i })
    await act(async () => {
      fireEvent.click(downloadBundleBtn)
    })
    expect(api.downloadTrainingBundle).toHaveBeenCalledWith('job_export_1')

    const metricsBtn = screen.getByRole('button', { name: /Metrics JSON/i })
    await act(async () => {
      fireEvent.click(metricsBtn)
    })
    expect(api.downloadTrainingArtifact).toHaveBeenCalledWith('job_export_1', 'metrics.json')
  })

  // 18. Job refresh / recovery
  it('18. restores active job from localStorage on initial page load', async () => {
    localStorage.setItem('lexirep_active_job_id', 'job_persisted_555')
    api.getTrainingJob.mockResolvedValue({
      job_id: 'job_persisted_555',
      status: 'training',
      current_step: 'Self-training loop 8 of 13',
      progress_percent: 61.5,
      loop: 8,
      total_loops: 13,
      elapsed_seconds: 65.0,
    })

    await act(async () => {
      render(
        <MemoryRouter>
          <Train />
        </MemoryRouter>
      )
    })

    await waitFor(() => {
      expect(api.getTrainingJob).toHaveBeenCalledWith('job_persisted_555')
      expect(screen.getByText(/Self-training loop 8 of 13/i)).toBeDefined()
    })
  })

  // 19. 409 active-job conflict
  it('19. handles 409 conflict when another training session is active', async () => {
    api.validateTrainingDataset.mockResolvedValue(MOCK_VALIDATION_GER)
    api.searchTrainingAnchors.mockResolvedValue(MOCK_ANCHORS_RESPONSE)

    const conflictErr = new Error("A training job is already active ('job_running_409'). Only one active training job is permitted.")
    conflictErr.status = 409
    api.createTrainingJob.mockRejectedValue(conflictErr)

    let container
    await act(async () => {
      const res = render(
        <MemoryRouter>
          <Train />
        </MemoryRouter>
      )
      container = res.container
    })

    const fileInput = container.querySelector('input[type="file"]')
    await act(async () => {
      fireEvent.change(fileInput, { target: { files: [new File([''], 'ger.npz')] } })
    })

    await act(async () => {
      fireEvent.click(await screen.findByRole('button', { name: /Continue to Anchor Discovery/i }))
    })
    await act(async () => {
      fireEvent.click(await screen.findByRole('button', { name: /Scan Training Set/i }))
    })

    await waitFor(() => {
      expect(screen.getAllByText('1181').length).toBeGreaterThan(0)
    })

    await act(async () => {
      fireEvent.click(await screen.findByRole('button', { name: /Continue to Training/i }))
    })
    await act(async () => {
      fireEvent.click(await screen.findByRole('button', { name: /Start Training/i }))
    })

    await waitFor(() => {
      expect(screen.getByText(/Active Training Job Detected/i)).toBeDefined()
      expect(screen.getByRole('button', { name: /Attach to Running Job/i })).toBeDefined()
    })
  })

  // 20. Completed results state (no fake metrics)
  it('20. renders completed stage without fake accuracy/F1 metrics', async () => {
    localStorage.setItem('lexirep_active_job_id', 'job_completed_999')
    api.getTrainingJob.mockResolvedValue({
      job_id: 'job_completed_999',
      status: 'completed',
      current_step: 'Complete',
      progress_percent: 100.0,
      loop: 13,
      total_loops: 13,
      elapsed_seconds: 142.0,
      language: 'GER',
      training_accuracy: 0.842,
      final_metrics: {
        training_accuracy: 0.842,
        final_idec_loss: 0.0121,
      },
      selected_anchor: {
        word_id: 1181,
        stressed_syllable_index: 2713,
        unstressed_syllable_index: 2714,
        separation_score: 0.6822,
      },
      loss_history: [
        { loop: 13, idec_loss: 0.0121, label_changes: 0, elapsed: 142.0 },
      ],
    })

    await act(async () => {
      render(
        <MemoryRouter>
          <Train />
        </MemoryRouter>
      )
    })

    await waitFor(() => {
      expect(screen.getByText(/Training Complete/i)).toBeDefined()
      expect(screen.getByText('84.2%')).toBeDefined()
      expect(screen.getByText('0.0121')).toBeDefined()
      // Real metrics only: zero fake accuracy/F1 from obsolete mock
      expect(screen.queryByText(/87\.4%/i)).toBeNull()
      expect(screen.queryByText(/0\.86/i)).toBeNull()
    })
  })

  // 21. Live IDEC loss test
  it('21. displays numeric IDEC loss during live polling and fails if rendered as "-"', async () => {
    localStorage.setItem('lexirep_active_job_id', 'job_poll_live')
    api.getTrainingJob.mockResolvedValue({
      job_id: 'job_poll_live',
      status: 'training',
      current_step: 'Self-training loop 2 of 13',
      progress_percent: 35.0,
      loop: 2,
      total_loops: 13,
      elapsed_seconds: 28.0,
      idec_loss: 0.0070,
      pseudo_label_changes: 12,
      loss_history: [
        { loop: 1, idec_loss: 0.0115, label_changes: 35, elapsed: 14.0 },
        { loop: 2, idec_loss: 0.0070, label_changes: 12, elapsed: 28.0 },
      ],
    })

    await act(async () => {
      render(
        <MemoryRouter>
          <Train />
        </MemoryRouter>
      )
    })

    await waitFor(() => {
      expect(screen.getByText('0.0070')).toBeDefined()
      // Ensure IDEC Loss does NOT render as "-" after valid training update
      const idecHeader = screen.getByText('IDEC Loss')
      const card = idecHeader.closest('div')
      expect(card.textContent).toContain('0.0070')
      expect(card.textContent).not.toContain('—')
    })
  })

  // 22. SupCon Loss regression test
  it('22. verifies "SupCon Loss" is NOT present anywhere in the Training Progress UI', async () => {
    localStorage.setItem('lexirep_active_job_id', 'job_live_check_123')
    api.getTrainingJob.mockResolvedValue({
      job_id: 'job_live_check_123',
      status: 'training',
      current_step: 'Self-training loop 1 of 13',
      progress_percent: 25.0,
      loop: 1,
      total_loops: 13,
      elapsed_seconds: 15.0,
      idec_loss: 0.0245,
      loss_history: [
        { loop: 1, idec_loss: 0.0245, label_changes: 40, elapsed: 15.0 },
      ],
    })

    await act(async () => {
      render(
        <MemoryRouter>
          <Train />
        </MemoryRouter>
      )
    })

    await waitFor(() => {
      expect(screen.getByText(/Training LexiRep Model/i)).toBeDefined()
      // Assert "SupCon Loss" is strictly NOT in the document
      expect(screen.queryByText(/SupCon Loss/i)).toBeNull()
    })
  })

  // 23. Results card test: Trained Acoustic Anchor removed
  it('23. verifies "Trained Acoustic Anchor" result card is NOT rendered in Stage 4 Results', async () => {
    localStorage.setItem('lexirep_active_job_id', 'job_anchor_card_check')
    api.getTrainingJob.mockResolvedValue({
      job_id: 'job_anchor_card_check',
      status: 'completed',
      current_step: 'Training and verification complete',
      progress_percent: 100.0,
      loop: 13,
      total_loops: 13,
      elapsed_seconds: 150.0,
      training_accuracy: 0.812,
      selected_anchor: {
        word_id: 1181,
        stressed_syllable_index: 2713,
        unstressed_syllable_index: 2714,
        separation_score: 0.6822,
      },
      loss_history: [
        { loop: 13, idec_loss: 0.0088, label_changes: 0, elapsed: 150.0 },
      ],
    })

    await act(async () => {
      render(
        <MemoryRouter>
          <Train />
        </MemoryRouter>
      )
    })

    await waitFor(() => {
      expect(screen.getByText(/Training Complete/i)).toBeDefined()
      // "Trained Acoustic Anchor" card must NOT be present
      expect(screen.queryByText(/Trained Acoustic Anchor/i)).toBeNull()
      expect(screen.queryByText(/Separation Score \(S_train\)/i)).toBeNull()
      expect(screen.queryByText(/Stressed Syl/i)).toBeNull()
      expect(screen.queryByText(/Unstressed Syl/i)).toBeNull()
    })
  })

  // 24. Final Training Accuracy display test
  it('24. displays "Final Training Accuracy" XX.X% and does NOT use S_train as accuracy', async () => {
    localStorage.setItem('lexirep_active_job_id', 'job_acc_display_check')
    api.getTrainingJob.mockResolvedValue({
      job_id: 'job_acc_display_check',
      status: 'completed',
      current_step: 'Complete',
      progress_percent: 100.0,
      loop: 13,
      total_loops: 13,
      elapsed_seconds: 145.0,
      training_accuracy: 0.842, // decimal fraction
      selected_anchor: {
        word_id: 1181,
        stressed_syllable_index: 2713,
        unstressed_syllable_index: 2714,
        separation_score: 0.6822, // S_train is 68.22%
      },
      loss_history: [
        { loop: 13, idec_loss: 0.0070, label_changes: 0, elapsed: 145.0 },
      ],
    })

    await act(async () => {
      render(
        <MemoryRouter>
          <Train />
        </MemoryRouter>
      )
    })

    await waitFor(() => {
      expect(screen.getByText(/Final Training Accuracy/i)).toBeDefined()
      expect(screen.getByText('84.2%')).toBeDefined()
      expect(screen.getByText('0.0070')).toBeDefined()
      // S_train (68.22% or 0.6822) must NOT be shown as accuracy
      expect(screen.queryByText('68.22%')).toBeNull()
      expect(screen.queryByText('68.2%')).toBeNull()
    })
  })
})

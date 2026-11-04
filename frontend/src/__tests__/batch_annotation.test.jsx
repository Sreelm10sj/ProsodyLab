import React from 'react'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react'
import Analyze from '../pages/Analyze.jsx'
import BatchAudioInput from '../components/analyze/BatchAudioInput.jsx'
import BatchProgress from '../components/analyze/BatchProgress.jsx'
import BatchResults from '../components/analyze/BatchResults.jsx'
import * as api from '../api.js'

// Mock react-router-dom Link
vi.mock('react-router-dom', () => ({
  Link: ({ children, to, ...props }) => <a href={to} {...props}>{children}</a>,
  useLocation: () => ({ pathname: '/analyze' }),
}))

// Mock api.js
vi.mock('../api.js', () => ({
  getHealth: vi.fn().mockResolvedValue({ status: 'ok', device: 'cpu', model_loaded: true }),
  postTranscribe: vi.fn(),
  postAnnotate: vi.fn(),
  postLexiRepAnalyze: vi.fn(),
  createBatchAnnotationJob: vi.fn(),
  getBatchAnnotationJob: vi.fn(),
  cancelBatchAnnotationJob: vi.fn(),
  downloadBatchAnnotations: vi.fn(),
}))

describe('Batch Audio Annotation Frontend Suite', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  // ── 1. Mode Switcher ────────────────────────────────────────────────────────
  it('toggles cleanly between Single Audio and Batch workflows', async () => {
    await act(async () => {
      render(<Analyze />)
    })

    // Initially in Single Audio (Upload Audio) mode
    const uploadTab = screen.getByRole('tab', { name: /single audio/i })
    const batchTab = screen.getByRole('tab', { name: /upload audio batch/i })
    expect(uploadTab.getAttribute('aria-selected')).toBe('true')
    expect(batchTab.getAttribute('aria-selected')).toBe('false')

    // Dropzone for single audio exists
    expect(screen.getByText(/drop audio file here/i)).toBeDefined()

    // Switch to Batch
    await act(async () => {
      fireEvent.click(batchTab)
    })
    expect(batchTab.getAttribute('aria-selected')).toBe('true')
    expect(uploadTab.getAttribute('aria-selected')).toBe('false')

    // Batch upload dropzone is now visible
    expect(screen.getByText(/drop zip archive containing audio files here/i)).toBeDefined()
    expect(screen.getByText(/supported formats: .wav · .mp3 · .flac · .m4a · .ogg/i)).toBeDefined()

    // Switch back to Single Audio
    await act(async () => {
      fireEvent.click(uploadTab)
    })
    expect(uploadTab.getAttribute('aria-selected')).toBe('true')
    expect(screen.getByText(/drop audio file here/i)).toBeDefined()
  })

  // ── 2. BatchAudioInput Component ───────────────────────────────────────────
  it('handles file selection, display, and submit in BatchAudioInput', async () => {
    const mockSubmit = vi.fn()
    await act(async () => {
      render(<BatchAudioInput onBatchSubmit={mockSubmit} isLoading={false} />)
    })

    const file = new File(['fake zip content'], 'test_recordings.zip', { type: 'application/zip' })
    const fileInput = document.getElementById('batch-file-input')

    await act(async () => {
      fireEvent.change(fileInput, { target: { files: [file] } })
    })

    // Displays filename
    expect(screen.getByText('test_recordings.zip')).toBeDefined()

    // Displays Start button
    const startBtn = screen.getByRole('button', { name: /start batch analysis/i })
    expect(startBtn).toBeDefined()

    // Submit trigger
    await act(async () => {
      fireEvent.click(startBtn)
    })
    expect(mockSubmit).toHaveBeenCalledTimes(1)
    expect(mockSubmit).toHaveBeenCalledWith(file)
  })

  it('allows clearing selected archive before starting analysis', async () => {
    const mockSubmit = vi.fn()
    await act(async () => {
      render(<BatchAudioInput onBatchSubmit={mockSubmit} isLoading={false} />)
    })

    const file = new File(['fake zip content'], 'recordings.zip', { type: 'application/zip' })
    const fileInput = document.getElementById('batch-file-input')
    await act(async () => {
      fireEvent.change(fileInput, { target: { files: [file] } })
    })

    expect(screen.getByText('recordings.zip')).toBeDefined()

    const removeBtn = screen.getByRole('button', { name: /remove/i })
    await act(async () => {
      fireEvent.click(removeBtn)
    })

    // Cleared back to dropzone
    expect(screen.queryByText('recordings.zip')).toBeNull()
    expect(screen.getByText(/drop zip archive containing audio files here/i)).toBeDefined()
  })

  // ── 3. BatchProgress Component ─────────────────────────────────────────────
  it('renders progress bar, counters, and handles cancellation confirmation', async () => {
    const mockCancel = vi.fn()
    const jobState = {
      batch_id: '12345',
      status: 'processing',
      total_files: 10,
      completed_files: 6,
      failed_files: 1,
      progress_percent: 70.0,
      current_file: 'speaker07.wav',
    }

    await act(async () => {
      render(<BatchProgress jobState={jobState} onCancel={mockCancel} isCancelling={false} />)
    })

    expect(screen.getByText(/6 of 10 files completed/i)).toBeDefined()
    expect(screen.getByText('70.0%')).toBeDefined()
    expect(screen.getByText('speaker07.wav')).toBeDefined()
    expect(screen.getByText('6')).toBeDefined() // completed
    expect(screen.getByText('1')).toBeDefined() // failed
    expect(screen.getByText('3')).toBeDefined() // remaining (10 - 7)

    // Cancel flow
    const cancelBtn = screen.getByRole('button', { name: /cancel batch/i })
    await act(async () => {
      fireEvent.click(cancelBtn)
    })

    // Confirmation dialog appears
    expect(screen.getByText(/cancel batch\? completed files remain downloadable/i)).toBeDefined()

    const confirmBtn = screen.getByRole('button', { name: /yes, cancel/i })
    await act(async () => {
      fireEvent.click(confirmBtn)
    })
    expect(mockCancel).toHaveBeenCalledTimes(1)
  })

  // ── 4. BatchResults Component ──────────────────────────────────────────────
  it('renders completed results and provides download button and error inspection', async () => {
    const mockDownload = vi.fn()
    const mockReset = vi.fn()
    const jobState = {
      batch_id: 'test-batch-uuid',
      status: 'completed_with_errors',
      total_files: 3,
      completed_files: 2,
      failed_files: 1,
      elapsed_seconds: 14.5,
      files: [
        { input_name: 'audio1.wav', output_name: 'audio1.json', status: 'completed', duration: 3.2, words_count: 5 },
        { input_name: 'audio2.wav', output_name: 'audio2.json', status: 'completed', duration: 4.1, words_count: 8 },
        { input_name: 'bad.wav', output_name: 'bad.json', status: 'failed', duration: null, words_count: null, error: 'Corrupt audio stream' },
      ],
    }

    await act(async () => {
      render(
        <BatchResults
          jobState={jobState}
          onDownload={mockDownload}
          onReset={mockReset}
          isDownloading={false}
        />
      )
    })

    // Header check
    expect(screen.getByText(/batch complete with errors/i)).toBeDefined()
    expect(screen.getByText(/2 annotations generated · 1 failed · 14.5s elapsed/i)).toBeDefined()

    // Download button exists and calls download
    const downloadBtn = screen.getByRole('button', { name: /download annotations \(\.zip\)/i })
    expect(downloadBtn).toBeDefined()
    await act(async () => {
      fireEvent.click(downloadBtn)
    })
    expect(mockDownload).toHaveBeenCalledTimes(1)

    // File table check
    expect(screen.getByText('audio1.wav')).toBeDefined()
    expect(screen.getByText('audio2.wav')).toBeDefined()
    expect(screen.getByText('bad.wav')).toBeDefined()

    // Expand error drawer
    const errorBtn = screen.getByRole('button', { name: /error/i })
    await act(async () => {
      fireEvent.click(errorBtn)
    })
    expect(screen.getByText(/corrupt audio stream/i)).toBeDefined()

    // Reset button
    const resetBtn = screen.getByRole('button', { name: /analyze another batch/i })
    await act(async () => {
      fireEvent.click(resetBtn)
    })
    expect(mockReset).toHaveBeenCalledTimes(1)
  })

  // ── 5. End-to-End Workflow in Analyze Page ─────────────────────────────────
  it('updates nativity model in batch mode and passes it to createBatchAnnotationJob', async () => {
    api.createBatchAnnotationJob.mockResolvedValueOnce({
      batch_id: 'batch-test-uuid',
      status: 'queued',
      total_files: 2,
    })

    await act(async () => {
      render(<Analyze />)
    })

    // Switch to Batch
    await act(async () => {
      fireEvent.click(screen.getByRole('tab', { name: /upload audio batch/i }))
    })

    // Change Speaker L1 Model to Italian (ITA)
    const itaBtn = document.getElementById('btn-batch-nativity-ita')
    await act(async () => {
      fireEvent.click(itaBtn)
    })
    expect(itaBtn.getAttribute('aria-pressed')).toBe('true')

    // Select file and submit
    const file = new File(['zipdata'], 'recordings.zip', { type: 'application/zip' })
    const fileInput = document.getElementById('batch-file-input')
    await act(async () => {
      fireEvent.change(fileInput, { target: { files: [file] } })
    })

    const startBtn = screen.getByRole('button', { name: /start batch analysis/i })
    await act(async () => {
      fireEvent.click(startBtn)
    })

    await waitFor(() => {
      expect(api.createBatchAnnotationJob).toHaveBeenCalledWith(file, 'ITA')
    })
  })
})

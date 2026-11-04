import React from 'react'
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import { render, screen, fireEvent, act, waitFor } from '@testing-library/react'
import { BrowserRouter } from 'react-router-dom'
import Analyze from '../pages/Analyze.jsx'
import { ThemeProvider } from '../ThemeContext.jsx'
import * as api from '../api.js'

// Mock backend API
vi.mock('../api.js', () => ({
  checkBackendHealth: vi.fn().mockResolvedValue({ status: 'ok', device: 'cpu' }),
  postTranscribe: vi.fn(),
  postLexiRepAnalyze: vi.fn(),
  postAnnotate: vi.fn(),
}))

describe('Record Audio Live Vocalizer & Transition Suite', () => {
  let mockTrack
  let mockStream
  let mockAnalyser
  let lastAudioCtxInstance

  beforeEach(() => {
    vi.clearAllMocks()
    localStorage.clear()

    mockTrack = {
      stop: vi.fn(),
      kind: 'audio',
      enabled: true,
    }

    mockStream = {
      getTracks: vi.fn(() => [mockTrack]),
    }

    mockAnalyser = {
      fftSize: 256,
      smoothingTimeConstant: 0.8,
      frequencyBinCount: 128,
      getByteFrequencyData: vi.fn((arr) => {
        arr.fill(45)
      }),
      connect: vi.fn(),
    }

    lastAudioCtxInstance = null
    class MockAudioContext {
      constructor() {
        this.createMediaStreamSource = vi.fn(() => ({ connect: vi.fn() }))
        this.createAnalyser = vi.fn(() => mockAnalyser)
        this.close = vi.fn().mockResolvedValue(undefined)
        lastAudioCtxInstance = this
      }
    }

    global.window.AudioContext = MockAudioContext
    global.window.webkitAudioContext = MockAudioContext

    HTMLCanvasElement.prototype.getContext = vi.fn(() => ({
      scale: vi.fn(),
      clearRect: vi.fn(),
      beginPath: vi.fn(),
      roundRect: vi.fn(),
      rect: vi.fn(),
      fill: vi.fn(),
    }))

    if (!global.navigator.mediaDevices) {
      global.navigator.mediaDevices = {}
    }
    global.navigator.mediaDevices.getUserMedia = vi.fn().mockResolvedValue(mockStream)

    class MockMediaRecorder {
      constructor(stream) {
        this.stream = stream
        this.state = 'inactive'
        this.ondataavailable = null
        this.onstop = null
      }
      start() {
        this.state = 'recording'
      }
      stop() {
        this.state = 'inactive'
        if (this.ondataavailable) {
          this.ondataavailable({ data: new Blob(['audio-content'], { type: 'audio/webm' }) })
        }
        if (this.onstop) {
          this.onstop()
        }
      }
    }
    global.window.MediaRecorder = MockMediaRecorder
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('1-5: opens Record Audio mode, starts recording, displays live visualizer driven by real AnalyserNode', async () => {
    render(
      <ThemeProvider>
        <BrowserRouter>
          <Analyze />
        </BrowserRouter>
      </ThemeProvider>
    )

    // 1. Record Audio mode opens
    const recordTab = screen.getByRole('tab', { name: /record audio/i })
    fireEvent.click(recordTab)

    expect(screen.getByText(/click start to record speech/i)).toBeTruthy()
    const startBtn = screen.getByRole('button', { name: /start recording/i })
    expect(startBtn).toBeTruthy()

    // 2. Recording starts on click
    await act(async () => {
      fireEvent.click(startBtn)
    })

    // 3. Recording state is displayed
    expect(screen.getByText(/^recording$/i)).toBeTruthy()
    expect(screen.getByText(/00:00/)).toBeTruthy()
    expect(screen.getByRole('button', { name: /stop recording/i })).toBeTruthy()

    // 4. Live visualizer component is rendered
    const visualizer = screen.getByRole('img', { name: /live audio frequency visualization/i })
    expect(visualizer).toBeTruthy()

    // 5. Visualizer uses the microphone/analyser stream
    expect(global.navigator.mediaDevices.getUserMedia).toHaveBeenCalledWith({ audio: true })
    expect(lastAudioCtxInstance.createAnalyser).toHaveBeenCalled()
    expect(mockAnalyser.getByteFrequencyData).toHaveBeenCalled()
  })

  it('6-10: Stop Recording stops recorder, releases microphone, immediately enters Analyzing Audio state and persists until API completes', async () => {
    let resolveTranscribe
    const transcribePromise = new Promise((resolve) => {
      resolveTranscribe = resolve
    })

    api.postTranscribe.mockImplementation(() => transcribePromise)
    api.postLexiRepAnalyze.mockResolvedValue({
      nativity: 'GER',
      words: [{ word: 'hello', is_polysyllabic: false, syllable_count: 1 }],
    })

    render(
      <ThemeProvider>
        <BrowserRouter>
          <Analyze />
        </BrowserRouter>
      </ThemeProvider>
    )

    fireEvent.click(screen.getByRole('tab', { name: /record audio/i }))

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /start recording/i }))
    })

    const stopBtn = screen.getByRole('button', { name: /stop recording/i })

    // 6. Stop recording stops the recorder
    await act(async () => {
      fireEvent.click(stopBtn)
    })

    // 7. Microphone tracks are released
    expect(mockTrack.stop).toHaveBeenCalled()
    expect(lastAudioCtxInstance.close).toHaveBeenCalled()

    // 8. Analyzing Audio state appears after Stop
    expect(screen.getByText(/analyzing audio/i)).toBeTruthy()
    expect(screen.getByText(/preparing your recording for transcription/i)).toBeTruthy()

    // 9. Transcription request begins
    expect(api.postTranscribe).toHaveBeenCalled()

    // 10. Analyze state persists until real request completion
    expect(screen.getByText(/analyzing audio/i)).toBeTruthy()

    // Resolve the real request
    await act(async () => {
      resolveTranscribe({
        transcription: 'Hello world',
        words: [
          { text: 'Hello', start: 0, end: 0.5 },
          { text: 'world', start: 0.6, end: 1.0 },
        ],
      })
    })

    // 11. Transcript appears after successful transcription
    await waitFor(() => {
      expect(screen.queryByText(/analyzing audio/i)).toBeNull()
      expect(screen.getByText(/hello/i)).toBeTruthy()
      expect(screen.getByText(/world/i)).toBeTruthy()
    })

    // 12. Full annotation is NOT automatically requested
    expect(api.postAnnotate).not.toHaveBeenCalled()

    // 13. LexiRep stress appears after transcription without full annotation
    expect(api.postLexiRepAnalyze).toHaveBeenCalled()
  })

  it('14-15: Reset cancels/stops recording and prevents stale response from overwriting new session', async () => {
    let resolveStaleTranscribe
    const stalePromise = new Promise((resolve) => {
      resolveStaleTranscribe = resolve
    })

    api.postTranscribe.mockImplementation(() => stalePromise)

    render(
      <ThemeProvider>
        <BrowserRouter>
          <Analyze />
        </BrowserRouter>
      </ThemeProvider>
    )

    fireEvent.click(screen.getByRole('tab', { name: /record audio/i }))

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /start recording/i }))
    })

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /stop recording/i }))
    })

    expect(screen.getByText(/analyzing audio/i)).toBeTruthy()

    // 14. User switches tab or resets
    fireEvent.click(screen.getByRole('tab', { name: /upload audio \(single audio\)/i }))

    // Now stale transcription resolves
    await act(async () => {
      resolveStaleTranscribe({
        transcription: 'Stale transcription',
        words: [{ text: 'Stale', start: 0, end: 1 }],
      })
    })

    // 15. Stale response cannot overwrite the session
    expect(screen.queryByText(/stale transcription/i)).toBeNull()
  })

  it('18: Clean error message when microphone permission is denied', async () => {
    const permErr = new Error('Permission denied')
    permErr.name = 'NotAllowedError'
    global.navigator.mediaDevices.getUserMedia = vi.fn().mockRejectedValue(permErr)

    render(
      <ThemeProvider>
        <BrowserRouter>
          <Analyze />
        </BrowserRouter>
      </ThemeProvider>
    )

    fireEvent.click(screen.getByRole('tab', { name: /record audio/i }))

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /start recording/i }))
    })

    expect(screen.getByText('Microphone access is required to record audio.')).toBeTruthy()
    expect(screen.getByRole('button', { name: /start recording/i })).toBeTruthy()
  })
})

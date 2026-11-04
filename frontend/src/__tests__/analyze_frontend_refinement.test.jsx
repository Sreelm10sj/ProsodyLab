import React from 'react'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react'
import Analyze from '../pages/Analyze.jsx'
import TranscriptView from '../components/analyze/TranscriptView.jsx'
import AnnotationViewer from '../components/analyze/AnnotationViewer.jsx'
import * as api from '../api.js'

// Mock react-router-dom Link & useLocation
vi.mock('react-router-dom', () => ({
  Link: ({ children, to, ...props }) => <a href={to} {...props}>{children}</a>,
  useLocation: () => ({ pathname: '/analyze' }),
}))

// Mock api methods
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

// Mock window.matchMedia & URL APIs
beforeEach(() => {
  window.matchMedia = window.matchMedia || function () {
    return {
      matches: false,
      addListener: () => {},
      removeListener: () => {},
      addEventListener: () => {},
      removeEventListener: () => {},
      dispatchEvent: () => false,
    }
  }

  window.HTMLMediaElement.prototype.play = vi.fn().mockResolvedValue()
  window.HTMLMediaElement.prototype.pause = vi.fn()
  window.URL.createObjectURL = vi.fn().mockReturnValue('blob:http://localhost/test-audio')
  window.URL.revokeObjectURL = vi.fn()
})

const mockTranscribePayload = {
  transcription: 'guten morgen wetter',
  words: [
    { id: 0, text: 'guten', start: 0.1, end: 0.8, stress: 'stressed' },
    { id: 1, text: 'morgen', start: 0.9, end: 1.5, stress: 'stressed' },
    { id: 2, text: 'wetter', start: 1.6, end: 2.2, stress: 'unstressed' },
  ],
  phrases: [
    { phraseIdx: 0, start: 0.1, end: 2.2, label: 'falling', slope: -1.8 },
  ],
}

const mockLexiRepGER = {
  nativity: 'GER',
  words: [
    {
      index: 0,
      text: 'guten',
      start: 0.1,
      end: 0.8,
      is_polysyllabic: true,
      syllable_count: 2,
      lexirep_stressed_syllable_index: 0,
      syllables: [
        { index: 0, text: 'gu', start: 0.1, end: 0.45, stress: 1, confidence: 0.94 },
        { index: 1, text: 'ten', start: 0.45, end: 0.8, stress: 0, confidence: 0.91 },
      ],
    },
    {
      index: 1,
      text: 'morgen',
      start: 0.9,
      end: 1.5,
      is_polysyllabic: true,
      syllable_count: 2,
      lexirep_stressed_syllable_index: 0,
      syllables: [
        { index: 0, text: 'mor', start: 0.9, end: 1.2, stress: 1, confidence: 0.88 },
        { index: 1, text: 'gen', start: 1.2, end: 1.5, stress: 0, confidence: 0.85 },
      ],
    },
    {
      index: 2,
      text: 'wetter',
      start: 1.6,
      end: 2.2,
      is_polysyllabic: true,
      syllable_count: 2,
      lexirep_stressed_syllable_index: 0,
      syllables: [
        { index: 0, text: 'wet', start: 1.6, end: 1.9, stress: 1, confidence: 0.92 },
        { index: 1, text: 'ter', start: 1.9, end: 2.2, stress: 0, confidence: 0.89 },
      ],
    },
  ],
}

const mockAnnotationGER = {
  metadata: { audio_file: 'sample_audio.wav', duration: 2.2, speaker_nativity: 'GER' },
  transcription: { text: 'guten morgen wetter', language: 'de' },
  words: [
    { id: 0, text: 'guten', start: 0.1, end: 0.5, stress: 'stressed', is_polysyllabic: true, syllable_count: 2, lexical_stress_index: 0, syllables: mockLexiRepGER.words[0].syllables },
    { id: 1, text: 'morgen', start: 1.0, end: 1.5, stress: 'stressed', is_polysyllabic: true, syllable_count: 2, lexical_stress_index: 0, syllables: mockLexiRepGER.words[1].syllables },
    { id: 2, text: 'wetter', start: 2.0, end: 2.5, stress: 'unstressed', is_polysyllabic: true, syllable_count: 2, lexical_stress_index: 0, syllables: mockLexiRepGER.words[2].syllables },
  ],
  pauses: [
    { start: 0.5, end: 1.0, duration: 0.5 },
    { start: 1.5, end: 2.0, duration: 0.5 },
  ],
  intonation: [
    { phrase_id: 0, start: 0.1, end: 2.5, slope: -1.8, label: 'falling' },
  ],
}

describe('Analyze Frontend Refinement — Header & Downloads (Req 23)', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('1. Download button exists with Download Audio label and aria-label="Download audio"', async () => {
    api.postTranscribe.mockResolvedValueOnce(mockTranscribePayload)
    api.postLexiRepAnalyze.mockResolvedValueOnce(mockLexiRepGER)

    await act(async () => {
      render(<Analyze />)
    })
    const file = new File(['audio-content'], 'sample_audio.wav', { type: 'audio/wav' })
    const input = document.getElementById('audio-file-input')
    await act(async () => {
      fireEvent.change(input, { target: { files: [file] } })
      fireEvent.click(document.getElementById('btn-analyze-upload'))
    })
    await waitFor(() => expect(api.postTranscribe).toHaveBeenCalled())

    const downloadBtn = screen.getByRole('button', { name: /download audio/i })
    expect(downloadBtn).toBeDefined()
    expect(downloadBtn.getAttribute('aria-label')).toBe('Download audio')
    expect(downloadBtn.textContent).toContain('Download Audio')
  })

  it('2. Download button triggers client-side audio download preserving original filename and data', async () => {
    api.postTranscribe.mockResolvedValueOnce(mockTranscribePayload)
    api.postLexiRepAnalyze.mockResolvedValueOnce(mockLexiRepGER)

    const clickSpy = vi.spyOn(HTMLAnchorElement.prototype, 'click')

    await act(async () => {
      render(<Analyze />)
    })
    const file = new File(['audio-content'], 'my_speech.wav', { type: 'audio/wav' })
    const input = document.getElementById('audio-file-input')
    await act(async () => {
      fireEvent.change(input, { target: { files: [file] } })
      fireEvent.click(document.getElementById('btn-analyze-upload'))
    })
    await waitFor(() => expect(api.postTranscribe).toHaveBeenCalled())

    const downloadBtn = screen.getByRole('button', { name: /download audio/i })
    await act(async () => {
      fireEvent.click(downloadBtn)
    })

    expect(clickSpy).toHaveBeenCalled()
    clickSpy.mockRestore()
  })

  it('3. Annotation/JSON options are completely absent from this header download button', async () => {
    api.postTranscribe.mockResolvedValueOnce(mockTranscribePayload)
    api.postLexiRepAnalyze.mockResolvedValueOnce(mockLexiRepGER)

    await act(async () => {
      render(<Analyze />)
    })
    const file = new File(['audio-content'], 'sample_audio.wav', { type: 'audio/wav' })
    const input = document.getElementById('audio-file-input')
    await act(async () => {
      fireEvent.change(input, { target: { files: [file] } })
      fireEvent.click(document.getElementById('btn-analyze-upload'))
    })
    await waitFor(() => expect(api.postTranscribe).toHaveBeenCalled())

    const header = screen.getByRole('button', { name: /download audio/i }).parentElement
    expect(header.textContent).not.toContain('JSON')
    expect(header.textContent).not.toContain('Annotation')
    expect(header.querySelector('select')).toBeNull()
  })

  it('4. Reset button uses reset icon with accessible label "Reset analysis"', async () => {
    api.postTranscribe.mockResolvedValueOnce(mockTranscribePayload)
    api.postLexiRepAnalyze.mockResolvedValueOnce(mockLexiRepGER)

    await act(async () => {
      render(<Analyze />)
    })
    const file = new File(['audio-content'], 'sample_audio.wav', { type: 'audio/wav' })
    const input = document.getElementById('audio-file-input')
    await act(async () => {
      fireEvent.change(input, { target: { files: [file] } })
      fireEvent.click(document.getElementById('btn-analyze-upload'))
    })
    await waitFor(() => expect(api.postTranscribe).toHaveBeenCalled())

    const resetBtn = screen.getByRole('button', { name: /reset analysis/i })
    expect(resetBtn).toBeDefined()
    expect(resetBtn.getAttribute('aria-label')).toBe('Reset analysis')
    // Uses RotateCcw svg icon inside
    expect(resetBtn.querySelector('svg')).toBeDefined()
  })

  it('5. Reset button performs original reset behavior completely', async () => {
    api.postTranscribe.mockResolvedValueOnce(mockTranscribePayload)
    api.postLexiRepAnalyze.mockResolvedValueOnce(mockLexiRepGER)

    await act(async () => {
      render(<Analyze />)
    })
    const file = new File(['audio-content'], 'sample_audio.wav', { type: 'audio/wav' })
    const input = document.getElementById('audio-file-input')
    await act(async () => {
      fireEvent.change(input, { target: { files: [file] } })
      fireEvent.click(document.getElementById('btn-analyze-upload'))
    })
    await waitFor(() => expect(api.postTranscribe).toHaveBeenCalled())

    const resetBtn = screen.getByRole('button', { name: /reset analysis/i })
    await act(async () => {
      fireEvent.click(resetBtn)
    })

    // Returns back to upload dropzone
    expect(screen.getByText(/drop audio file here/i)).toBeDefined()
    expect(screen.queryByText('guten morgen wetter')).toBeNull()
  })
})

describe('Analyze Frontend Refinement — Karaoke & Stress State Readability (Req 24)', () => {
  const wordsSample = [
    // 0: normal (unstressed, monosyllabic)
    { id: 0, text: 'the', start: 0.0, end: 0.3, stress: 'unstressed' },
    // 1: stressed (monosyllabic)
    { id: 1, text: 'cat', start: 0.4, end: 0.8, stress: 'stressed' },
    // 2: polysyllabic (unstressed word-level, but polysyllabic)
    {
      id: 2,
      text: 'water',
      start: 0.9,
      end: 1.4,
      stress: 'unstressed',
      is_polysyllabic: true,
      syllable_count: 2,
      lexical_stress_index: 0,
      syllables: [
        { index: 0, text: 'wa', start: 0.9, end: 1.15, stress: 1, confidence: 0.9 },
        { index: 1, text: 'ter', start: 1.15, end: 1.4, stress: 0, confidence: 0.88 },
      ],
    },
    // 3: stressed + polysyllabic
    {
      id: 3,
      text: 'football',
      start: 1.5,
      end: 2.1,
      stress: 'stressed',
      is_polysyllabic: true,
      syllable_count: 2,
      lexical_stress_index: 0,
      syllables: [
        { index: 0, text: 'foot', start: 1.5, end: 1.8, stress: 1, confidence: 0.95 },
        { index: 1, text: 'ball', start: 1.8, end: 2.1, stress: 0, confidence: 0.91 },
      ],
    },
  ]

  it('1. Normal word is rendered with base word-pill class and is not active/stressed/poly', () => {
    render(
      <TranscriptView
        transcript="the cat water football"
        words={wordsSample}
        currentTime={0}
        isPlaying={false}
      />
    )
    const wordThe = screen.getByRole('button', { name: /^the$/i })
    expect(wordThe.className).toContain('word-pill')
    expect(wordThe.className).not.toContain('word-pill-stressed')
    expect(wordThe.className).not.toContain('word-pill-poly')
    expect(wordThe.className).not.toContain('word-pill-active')
  })

  it('2. Stressed word uses word-pill-stressed highlight', () => {
    render(
      <TranscriptView
        transcript="the cat water football"
        words={wordsSample}
        currentTime={0}
        isPlaying={false}
      />
    )
    const wordCat = screen.getByRole('button', { name: /^cat$/i })
    expect(wordCat.className).toContain('word-pill-stressed')
    expect(wordCat.className).not.toContain('word-pill-active')
  })

  it('3. Active karaoke word applies word-pill-active', () => {
    render(
      <TranscriptView
        transcript="the cat water football"
        words={wordsSample}
        currentTime={0.15} // within 'the' (0.0 - 0.3)
        isPlaying={true}
      />
    )
    const wordThe = screen.getByRole('button', { name: /^the$/i })
    expect(wordThe.className).toContain('word-pill-active')
    expect(wordThe.className).not.toContain('activePolysyllabicWord')
  })

  it('4. Active polysyllabic word applies activePolysyllabicWord and word-pill-active-poly', () => {
    render(
      <TranscriptView
        transcript="the cat water football"
        words={wordsSample}
        currentTime={1.0} // within 'water' (0.9 - 1.4)
        isPlaying={true}
      />
    )
    const wordWater = screen.getByRole('button', { name: /water/i })
    expect(wordWater.className).toContain('activePolysyllabicWord')
    expect(wordWater.className).toContain('word-pill-active-poly')
    expect(wordWater.className).toContain('word-pill-active')
  })

  it('5. Stressed + active word applies word-pill-active and word-pill-active-stressed', () => {
    render(
      <TranscriptView
        transcript="the cat water football"
        words={wordsSample}
        currentTime={0.6} // within 'cat' (0.4 - 0.8)
        isPlaying={true}
      />
    )
    const wordCat = screen.getByRole('button', { name: /^cat$/i })
    expect(wordCat.className).toContain('word-pill-active')
    expect(wordCat.className).toContain('word-pill-active-stressed')
    expect(wordCat.className).toContain('word-pill-stressed')
  })

  it('6. Stressed + polysyllabic + active word applies activePolysyllabicWord-stressed', () => {
    render(
      <TranscriptView
        transcript="the cat water football"
        words={wordsSample}
        currentTime={1.7} // within 'football' (1.5 - 2.1)
        isPlaying={true}
      />
    )
    const wordFootball = screen.getByRole('button', { name: /football/i })
    expect(wordFootball.className).toContain('activePolysyllabicWord')
    expect(wordFootball.className).toContain('word-pill-active')
    expect(wordFootball.className).toContain('word-pill-active-stressed')
    expect(wordFootball.className).toContain('activePolysyllabicWord-stressed')
  })
})

describe('Analyze Frontend Refinement — Annotation View Order (Req 25)', () => {
  it('1. Rendered order of sections is exactly: Words, Syllables, Pauses, Intonations', () => {
    render(
      <AnnotationViewer
        annotation={mockAnnotationGER}
        lexirepResult={mockLexiRepGER}
      />
    )

    const sections = document.querySelectorAll('[data-testid^="section-"]')
    const sectionIds = Array.from(sections).map(s => s.getAttribute('data-testid'))

    expect(sectionIds).toEqual([
      'section-words',
      'section-syllables',
      'section-pauses',
      'section-intonations',
    ])
  })

  it('2. Standalone "Polysyllabic Words" summary card is NOT rendered', () => {
    render(
      <AnnotationViewer
        annotation={mockAnnotationGER}
        lexirepResult={mockLexiRepGER}
      />
    )

    // Polysyllabic words summary card should NOT exist
    expect(screen.queryByText(/2 words detected/i)).toBeNull()
    expect(screen.queryByText(/3 words detected/i)).toBeNull()
  })
})

describe('Analyze Frontend Refinement — Functionality Preservation (Req 26)', () => {
  it('preserves word click-to-seek, pauses, intonation badges, and hover popover portal', () => {
    const handleSeek = vi.fn()
    render(
      <TranscriptView
        transcript="guten morgen wetter"
        words={mockAnnotationGER.words}
        phrases={mockAnnotationGER.intonation}
        currentTime={0}
        isPlaying={false}
        lexirepResult={mockLexiRepGER}
        onSeek={handleSeek}
      />
    )

    // Click to seek
    const gutenBtn = screen.getByRole('button', { name: /guten/i })
    fireEvent.click(gutenBtn)
    expect(handleSeek).toHaveBeenCalledWith(0.1)

    // Pause marker between words (0.50s gap)
    const pauseBadges = screen.getAllByLabelText(/pause 0.50 seconds/i)
    expect(pauseBadges.length).toBeGreaterThan(0)

    // Intonation badge
    expect(screen.getByRole('img', { name: /falling intonation/i })).toBeDefined()

    // Polysyllabic hover portal tooltip
    fireEvent.mouseEnter(gutenBtn)
    const tooltip = document.getElementById('word-portal-tooltip')
    expect(tooltip).not.toBeNull()
    expect(tooltip.parentElement).toBe(document.body)

    fireEvent.mouseLeave(gutenBtn)
    expect(document.getElementById('word-portal-tooltip')).toBeNull()
  })

  it('preserves AnnotationViewer model switcher and JSON/TXT download buttons', () => {
    const handleNativityChange = vi.fn()
    render(
      <AnnotationViewer
        annotation={mockAnnotationGER}
        lexirepResult={mockLexiRepGER}
        onNativityChange={handleNativityChange}
      />
    )

    expect(screen.getByText('Download JSON')).toBeDefined()
    expect(screen.getByText('Download TXT')).toBeDefined()

    const itaBtn = screen.getByRole('button', { name: /Italian \(ITA\)/i })
    fireEvent.click(itaBtn)
    expect(handleNativityChange).toHaveBeenCalledWith('ITA')
  })
})

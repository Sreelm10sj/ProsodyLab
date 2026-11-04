import React from 'react'
import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import AnnotationViewer from '../components/analyze/AnnotationViewer.jsx'
import TranscriptView from '../components/analyze/TranscriptView.jsx'
import ProsodyPanel from '../components/analyze/ProsodyPanel.jsx'
import PitchContour from '../components/analyze/PitchContour.jsx'
import Analyze from '../pages/Analyze.jsx'

// Mock react-router-dom Link if needed
vi.mock('react-router-dom', () => ({
  Link: ({ children, to, ...props }) => <a href={to} {...props}>{children}</a>,
  useLocation: () => ({ pathname: '/analyze' }),
}))

// Mock API calls
vi.mock('../api.js', () => ({
  checkBackendHealth: vi.fn().mockResolvedValue({ status: 'ok', device: 'cpu' }),
  transcribeAudio: vi.fn(),
  annotateAudio: vi.fn(),
  analyzeLexiRep: vi.fn(),
}))

const MOCK_ANNOTATION = {
  metadata: {
    audio_duration_sec: 2.5,
    sample_rate: 16000,
    speaker_nativity: 'GER',
  },
  transcription: 'Guten Morgen',
  words: [
    {
      id: 1,
      text: 'Guten',
      start: 0.2,
      end: 0.8,
      stress: 'stressed',
      syllable_count: 2,
      is_polysyllabic: true,
      lexical_stress_index: 0,
      syllables: [
        {
          index: 0,
          text: 'Gu',
          start: 0.2,
          end: 0.5,
          duration: 0.3,
          stress: 1,
          contrast_score: 0.75,
          confidence: 0.94,
        },
        {
          index: 1,
          text: 'ten',
          start: 0.5,
          end: 0.8,
          duration: 0.3,
          stress: 0,
          contrast_score: -0.75,
          confidence: 0.91,
        },
      ],
    },
    {
      id: 2,
      text: 'Morgen',
      start: 0.9,
      end: 1.6,
      stress: 'unstressed',
      syllable_count: 2,
      is_polysyllabic: true,
      lexical_stress_index: 0,
      syllables: [
        {
          index: 0,
          text: 'Mor',
          start: 0.9,
          end: 1.25,
          duration: 0.35,
          stress: 1,
          contrast_score: 0.62,
          confidence: 0.88,
        },
        {
          index: 1,
          text: 'gen',
          start: 1.25,
          end: 1.6,
          duration: 0.35,
          stress: 0,
          contrast_score: -0.62,
          confidence: 0.85,
        },
      ],
    },
  ],
  pauses: [
    { start: 0.8, end: 0.9, duration: 0.1 },
  ],
  intonation: [
    {
      phrase_id: 1,
      start: 0.2,
      end: 1.6,
      label: 'falling',
      slope: -1.8,
    },
  ],
  timeline: [
    { type: 'word', text: 'Guten', start: 0.2, end: 0.8 },
    { type: 'pause', text: '', start: 0.8, end: 0.9 },
    { type: 'word', text: 'Morgen', start: 0.9, end: 1.6 },
  ],
}

const MOCK_LEXIREP_RESULT = {
  nativity: 'GER',
  words: MOCK_ANNOTATION.words,
}

describe('Annotation UI and Output Cleanup Verification', () => {
  it('1 & 2: PitchContour and ProsodyPanel components exist in repo but pitch visualization is NOT rendered', () => {
    // 2: Verify component files exist and are callable React components
    expect(typeof ProsodyPanel).toBe('function')
    expect(typeof PitchContour).toBe('function')

    // 1: Render Analyze page and ensure pitch contour panel/canvas is NOT rendered
    render(<Analyze />)
    expect(screen.queryByText(/pitch contour/i)).toBeNull()
    expect(screen.queryByText(/raw f0/i)).toBeNull()
    expect(screen.queryByText(/stylized f0/i)).toBeNull()
    expect(screen.queryByTestId('pitch-canvas')).toBeNull()
  })

  it('3: Intonation information still renders properly', () => {
    render(
      <AnnotationViewer
        annotation={MOCK_ANNOTATION}
        lexirepResult={MOCK_LEXIREP_RESULT}
      />
    )
    expect(screen.getByText(/Intonations? \(1 phrases\)/i)).toBeDefined()
    expect(screen.getByText(/falling/i)).toBeDefined()
    expect(screen.getByText(/-1.80 st\/s/i)).toBeDefined()
  })

  it('4 & 5: Syllable stress and confidence score render in AnnotationViewer', () => {
    render(
      <AnnotationViewer
        annotation={MOCK_ANNOTATION}
        lexirepResult={MOCK_LEXIREP_RESULT}
      />
    )
    // Syllables render
    expect(screen.getAllByText(/Gu ●/i).length).toBeGreaterThan(0)
    expect(screen.getAllByText(/ten/i).length).toBeGreaterThan(0)

    // Syllable stress renders
    expect(screen.getAllByText(/● Stressed \(1\)/i).length).toBe(2)
    expect(screen.getAllByText(/○ Unstressed \(0\)/i).length).toBe(2)

    // Confidence score renders
    expect(screen.getByText('94%')).toBeDefined()
    expect(screen.getByText('91%')).toBeDefined()
    expect(screen.getByText('88%')).toBeDefined()
    expect(screen.getByText('85%')).toBeDefined()
  })

  it('6: Contrast score column and values do NOT render in AnnotationViewer table', () => {
    render(
      <AnnotationViewer
        annotation={MOCK_ANNOTATION}
        lexirepResult={MOCK_LEXIREP_RESULT}
      />
    )
    // Header should NOT have Contrast
    expect(screen.queryByText('Contrast')).toBeNull()
    // Columns should be Word, K, Syllable, Interval, Duration, Lexical Stress, Confidence
    expect(screen.getByText('Lexical Stress')).toBeDefined()
    expect(screen.getByText('Confidence')).toBeDefined()

    // Contrast values like +0.75 or -0.75 should NOT be rendered in the table
    expect(screen.queryByText('+0.75')).toBeNull()
    expect(screen.queryByText('-0.75')).toBeNull()
    expect(screen.queryByText('+0.62')).toBeNull()
    expect(screen.queryByText('-0.62')).toBeNull()
  })

  it('6 (TranscriptView): Contrast score does NOT render in syllable popover, but confidence and stress do', () => {
    render(
      <TranscriptView
        transcript="Guten Morgen"
        words={MOCK_ANNOTATION.words}
        phrases={MOCK_ANNOTATION.intonation}
        currentTime={0}
        isPlaying={false}
        lexirepResult={MOCK_LEXIREP_RESULT}
      />
    )
    // Hover on polysyllabic word 'Guten' to open portal tooltip
    const gutenBtn = screen.getByRole('button', { name: /Guten/i })
    fireEvent.mouseEnter(gutenBtn)

    // Syllable popover elements in portal
    expect(screen.getByText('1st syllable')).toBeDefined()
    expect(screen.getByText('Gu (1)')).toBeDefined()
    expect(screen.getByText('ten (0)')).toBeDefined()
    expect(screen.getByText('94%')).toBeDefined()

    // Contrast badges should NOT appear
    expect(screen.queryByText('+0.75')).toBeNull()
    expect(screen.queryByText('-0.75')).toBeNull()

    // Intonation badge in the transcript MUST be rendered
    expect(screen.getByLabelText(/Falling intonation/i)).toBeDefined()
    expect(screen.getByText('↘')).toBeDefined()
    expect(screen.getByText(/Falling Intonation/i)).toBeDefined()
  })

  it('7: Syllable interaction and click-to-seek still works in TranscriptView', () => {
    const handleSeek = vi.fn()
    render(
      <TranscriptView
        transcript="Guten Morgen"
        words={MOCK_ANNOTATION.words}
        phrases={MOCK_ANNOTATION.intonation}
        currentTime={0}
        isPlaying={false}
        lexirepResult={MOCK_LEXIREP_RESULT}
        onSeek={handleSeek}
      />
    )
    // Click word pill "Guten" to seek to word start
    const gutenBtn = screen.getByRole('button', { name: /Guten/i })
    fireEvent.click(gutenBtn)
    expect(handleSeek).toHaveBeenCalledWith(0.2)

    // Switch view mode to "Syllables"
    const syllablesBtn = screen.getByRole('button', { name: /Syllables/i })
    fireEvent.click(syllablesBtn)
    expect(screen.getByText('GU')).toBeDefined()
  })

  it('8: No unrelated annotation components disappeared', () => {
    render(
      <AnnotationViewer
        annotation={MOCK_ANNOTATION}
        lexirepResult={MOCK_LEXIREP_RESULT}
      />
    )
    // JSON / TXT download buttons present
    expect(screen.getByText('Download JSON')).toBeDefined()
    expect(screen.getByText('Download TXT')).toBeDefined()

    // Words section present
    expect(screen.getByText(/Words \(2\)/i)).toBeDefined()

    // Syllables section present
    expect(screen.getByText(/Syllables/i)).toBeDefined()

    // Pauses section present
    expect(screen.getByText(/Pauses \(1\)/i)).toBeDefined()

    // Intonation section present
    expect(screen.getByText(/Intonations? \(1 phrases\)/i)).toBeDefined()
  })
})

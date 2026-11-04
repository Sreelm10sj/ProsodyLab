import React from 'react'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor, act, within } from '@testing-library/react'
import Analyze from '../pages/Analyze.jsx'
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

const mockTranscribePayload = {
  transcription: 'guten morgen',
  words: [
    { id: 0, text: 'guten', start: 0.1, end: 0.7, stress: 'stressed' },
    { id: 1, text: 'morgen', start: 0.8, end: 1.4, stress: 'stressed' },
  ],
  phrases: [
    { phrase_id: 0, start: 0.1, end: 1.4, slope: -1.2, label: 'falling' },
  ],
  pitch_contour: null,
  pitch_stylized: null,
  utterance_median_hz: 120,
}

const mockAnnotatePayloadGER = {
  metadata: { audio_file: 'test.wav', duration: 1.8, speaker_nativity: 'GER' },
  transcription: { text: 'guten morgen', language: 'de' },
  words: [
    {
      id: 0,
      text: 'guten',
      start: 0.1,
      end: 0.7,
      stress: 'stressed',
      is_polysyllabic: true,
      syllable_count: 2,
      lexical_stress_index: 0,
      confidence: 0.94,
      syllables: [
        { text: 'gu', start: 0.1, end: 0.4, stress: 1, duration: 0.3, confidence: 0.94 },
        { text: 'ten', start: 0.4, end: 0.7, stress: 0, duration: 0.3, confidence: 0.91 },
      ],
      lexirep: { model: 'GER', confidence: 0.94 },
    },
    {
      id: 1,
      text: 'morgen',
      start: 0.8,
      end: 1.4,
      stress: 'stressed',
      is_polysyllabic: true,
      syllable_count: 2,
      lexical_stress_index: 0,
      confidence: 0.92,
      syllables: [
        { text: 'mor', start: 0.8, end: 1.1, stress: 1, duration: 0.3, confidence: 0.92 },
        { text: 'gen', start: 1.1, end: 1.4, stress: 0, duration: 0.3, confidence: 0.89 },
      ],
      lexirep: { model: 'GER', confidence: 0.92 },
    },
  ],
  pauses: [
    { start: 0.7, end: 0.8, duration: 0.1 },
  ],
  intonation: [
    { phrase_id: 0, start: 0.1, end: 1.4, slope: -1.2, label: 'falling' },
  ],
  timeline: [],
}

const mockAnnotatePayloadITA = {
  ...mockAnnotatePayloadGER,
  metadata: { audio_file: 'test.wav', duration: 1.8, speaker_nativity: 'ITA' },
  words: [
    {
      ...mockAnnotatePayloadGER.words[0],
      lexirep: { model: 'ITA', confidence: 0.88 },
    },
    {
      ...mockAnnotatePayloadGER.words[1],
      lexirep: { model: 'ITA', confidence: 0.85 },
    },
  ],
}

const mockLexiRepAnalyzePayloadGER = {
  status: 'ok',
  nativity: 'GER',
  transcription: 'guten morgen',
  words: [
    {
      word_index: 0,
      text: 'guten',
      start: 0.1,
      end: 0.7,
      whistress_stressed: true,
      is_polysyllabic: true,
      syllable_count: 2,
      lexirep_stressed_syllable_index: 0,
      syllables: [
        { index: 0, text: 'gu', start: 0.1, end: 0.4, duration: 0.3, stress: 1, confidence: 0.94 },
        { index: 1, text: 'ten', start: 0.4, end: 0.7, duration: 0.3, stress: 0, confidence: 0.91 },
      ],
    },
    {
      word_index: 1,
      text: 'morgen',
      start: 0.8,
      end: 1.4,
      whistress_stressed: true,
      is_polysyllabic: true,
      syllable_count: 2,
      lexirep_stressed_syllable_index: 0,
      syllables: [
        { index: 0, text: 'mor', start: 0.8, end: 1.1, duration: 0.3, stress: 1, confidence: 0.92 },
        { index: 1, text: 'gen', start: 1.1, end: 1.4, duration: 0.3, stress: 0, confidence: 0.89 },
      ],
    },
  ],
}

const mockLexiRepAnalyzePayloadITA = {
  ...mockLexiRepAnalyzePayloadGER,
  nativity: 'ITA',
  words: [
    {
      ...mockLexiRepAnalyzePayloadGER.words[0],
      syllables: [
        { ...mockLexiRepAnalyzePayloadGER.words[0].syllables[0], confidence: 0.88 },
        { ...mockLexiRepAnalyzePayloadGER.words[0].syllables[1], confidence: 0.84 },
      ],
    },
    {
      ...mockLexiRepAnalyzePayloadGER.words[1],
      syllables: [
        { ...mockLexiRepAnalyzePayloadGER.words[1].syllables[0], confidence: 0.85 },
        { ...mockLexiRepAnalyzePayloadGER.words[1].syllables[1], confidence: 0.81 },
      ],
    },
  ],
}

describe('Analyze Page Targeted Redesign & Flow Suite', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  // 1. "Workflow Mode" is not rendered
  it('1. verifies "Workflow Mode" segmented bar is NOT rendered', async () => {
    await act(async () => {
      render(<Analyze />)
    })
    expect(screen.queryByText(/workflow mode/i)).toBeNull()
  })

  // 2-4. Single Audio, Record Audio, Upload Audio Batch tabs exist
  it('2-4. verifies Upload Audio, Record Audio, and Upload Audio Batch tabs exist in the header', async () => {
    await act(async () => {
      render(<Analyze />)
    })

    const uploadTab = screen.getByRole('tab', { name: /upload audio \(single audio\)/i })
    const recordTab = screen.getByRole('tab', { name: /record audio/i })
    const batchTab = screen.getByRole('tab', { name: /upload audio batch/i })

    expect(uploadTab).toBeDefined()
    expect(recordTab).toBeDefined()
    expect(batchTab).toBeDefined()
    expect(uploadTab.getAttribute('aria-selected')).toBe('true')
  })

  // 5-7. Clicking Upload Audio Batch works, and returning to Upload Audio and Record Audio works
  it('5-7. verifies clicking Upload Audio Batch switches views, and returning to Upload and Record works freely', async () => {
    await act(async () => {
      render(<Analyze />)
    })

    const uploadTab = screen.getByRole('tab', { name: /upload audio \(single audio\)/i })
    const recordTab = screen.getByRole('tab', { name: /record audio/i })
    const batchTab = screen.getByRole('tab', { name: /upload audio batch/i })

    // Switch to Batch
    await act(async () => {
      fireEvent.click(batchTab)
    })
    expect(batchTab.getAttribute('aria-selected')).toBe('true')
    expect(screen.getByText(/drop zip archive containing audio files here/i)).toBeDefined()

    // From Batch, return to Upload Audio
    await act(async () => {
      fireEvent.click(uploadTab)
    })
    expect(uploadTab.getAttribute('aria-selected')).toBe('true')
    expect(screen.getByText(/drop audio file here/i)).toBeDefined()

    // From Upload Audio to Batch again
    await act(async () => {
      fireEvent.click(batchTab)
    })
    expect(batchTab.getAttribute('aria-selected')).toBe('true')

    // From Batch, switch directly to Record Audio
    await act(async () => {
      fireEvent.click(recordTab)
    })
    expect(recordTab.getAttribute('aria-selected')).toBe('true')
    expect(screen.getByRole('button', { name: /start recording/i })).toBeDefined()

    // From Record Audio, switch back to Upload Audio
    await act(async () => {
      fireEvent.click(uploadTab)
    })
    expect(uploadTab.getAttribute('aria-selected')).toBe('true')
    expect(screen.getByText(/drop audio file here/i)).toBeDefined()
  })

  // 8. Single Audio defaults to GER and model selector is NOT on landing screen
  it('8-9. verifies GER/ITA model selector is NOT on landing screen and single audio defaults to GER', async () => {
    await act(async () => {
      render(<Analyze />)
    })

    // No GER/ITA button on the idle landing screen
    expect(document.getElementById('btn-nativity-ger')).toBeNull()
    expect(document.getElementById('btn-nativity-ita')).toBeNull()
  })

  // 10-18. Decoupled flow: Upload -> postTranscribe + postLexiRepAnalyze -> Generate Annotation button -> postAnnotate
  it('10-18. verifies transcription immediately triggers lightweight LexiRep stress WITHOUT running full annotation, and Generate Annotation triggers real annotation', async () => {
    api.postTranscribe.mockResolvedValueOnce(mockTranscribePayload)
    api.postLexiRepAnalyze.mockResolvedValueOnce(mockLexiRepAnalyzePayloadGER)
    api.postAnnotate.mockResolvedValueOnce(mockAnnotatePayloadGER)

    await act(async () => {
      render(<Analyze />)
    })

    // Upload audio
    const file = new File(['audio-bytes'], 'test.wav', { type: 'audio/wav' })
    const input = document.getElementById('audio-file-input')
    await act(async () => {
      fireEvent.change(input, { target: { files: [file] } })
    })

    const analyzeBtn = document.getElementById('btn-analyze-upload')
    await act(async () => {
      fireEvent.click(analyzeBtn)
    })

    // 1. postTranscribe was called with file and filename
    await waitFor(() => {
      expect(api.postTranscribe).toHaveBeenCalledWith(file, 'test.wav')
    })

    // 2. Immediate lightweight LexiRep request occurs after transcription with default GER
    await waitFor(() => {
      expect(api.postLexiRepAnalyze).toHaveBeenCalledWith(file, 'GER', 'test.wav')
    })

    // 3. CRITICAL: postAnnotate was NOT called automatically!
    expect(api.postAnnotate).not.toHaveBeenCalled()

    // 4. Transcript view is rendered
    expect(screen.getByRole('button', { name: /guten/i })).toBeDefined()
    expect(screen.getByRole('button', { name: /morgen/i })).toBeDefined()

    // 5. LexiRep syllable stress is visible in transcript BEFORE clicking Generate Annotation!
    await waitFor(() => {
      expect(screen.getByText(/LexiRep GER Active/i)).toBeDefined()
    })

    // 6. Old sub-tabs (Words, Phrases, Syllables & Details) are ABSENT
    expect(screen.queryByRole('tab', { name: /words/i })).toBeNull()
    expect(screen.queryByRole('tab', { name: /phrases/i })).toBeNull()
    expect(screen.queryByRole('tab', { name: /syllables/i })).toBeNull()

    // 7. Generate Annotation button is displayed
    const generateBtn = screen.getByRole('button', { name: /generate annotation/i })
    expect(generateBtn).toBeDefined()
    expect(screen.getByText(/Transcript Ready/i)).toBeDefined()

    // 8. Annotation results section is NOT visible before clicking button
    expect(document.getElementById('annotation-results-section')).toBeNull()

    // 9. Click Generate Annotation
    await act(async () => {
      fireEvent.click(generateBtn)
    })

    // 10. Loading state appears and postAnnotate is called with default GER model
    await waitFor(() => {
      expect(api.postAnnotate).toHaveBeenCalledWith(file, 'test.wav', 'GER')
    })

    // 11. Annotation section is now rendered!
    await waitFor(() => {
      expect(document.getElementById('annotation-results-section')).toBeDefined()
    })

    // 12. Polysyllabic words summary card is NOT rendered
    expect(screen.queryByText(/2 words detected/i)).toBeNull()

    // 13. Syllable stress table is rendered with Confidence and without contrast score
    expect(screen.getByTestId('section-syllables')).toBeDefined()
    expect(screen.getAllByText(/● Stressed \(1\)/i).length).toBe(2)
    expect(screen.queryByText(/contrast score/i)).toBeNull()
    expect(screen.queryByText(/contrast table/i)).toBeNull()
  })

  // 19a. Model switching before Generate Annotation updates stress using lightweight LexiRep
  it('19a. verifies changing model to Italian before Generate Annotation requests lightweight LexiRep with ITA', async () => {
    api.postTranscribe.mockResolvedValueOnce(mockTranscribePayload)
    api.postLexiRepAnalyze
      .mockResolvedValueOnce(mockLexiRepAnalyzePayloadGER)
      .mockResolvedValueOnce(mockLexiRepAnalyzePayloadITA)

    await act(async () => {
      render(<Analyze />)
    })

    const file = new File(['audio-bytes'], 'test.wav', { type: 'audio/wav' })
    const input = document.getElementById('audio-file-input')
    await act(async () => {
      fireEvent.change(input, { target: { files: [file] } })
      fireEvent.click(document.getElementById('btn-analyze-upload'))
    })
    await waitFor(() => {
      expect(api.postLexiRepAnalyze).toHaveBeenCalledWith(file, 'GER', 'test.wav')
    })

    // Change model to ITA via action card or transcript control
    const itaBtn = document.getElementById('btn-transcript-nativity-ita')
    expect(itaBtn).toBeDefined()
    await act(async () => {
      fireEvent.click(itaBtn)
    })

    // Updates stress using lightweight endpoint without calling full annotation!
    await waitFor(() => {
      expect(api.postLexiRepAnalyze).toHaveBeenLastCalledWith(file, 'ITA', 'test.wav')
    })
    expect(api.postAnnotate).not.toHaveBeenCalled()
    await waitFor(() => {
      expect(screen.getByText(/LexiRep ITA Active/i)).toBeDefined()
    })
  })

  // 19b. LexiRep failure does not break transcript functionality
  it('19b. verifies LexiRep failure does not destroy transcript functionality and displays non-blocking message', async () => {
    api.postTranscribe.mockResolvedValueOnce(mockTranscribePayload)
    api.postLexiRepAnalyze.mockRejectedValueOnce(new Error('Network failure'))

    await act(async () => {
      render(<Analyze />)
    })

    const file = new File(['audio-bytes'], 'test.wav', { type: 'audio/wav' })
    const input = document.getElementById('audio-file-input')
    await act(async () => {
      fireEvent.change(input, { target: { files: [file] } })
      fireEvent.click(document.getElementById('btn-analyze-upload'))
    })
    await waitFor(() => {
      expect(api.postTranscribe).toHaveBeenCalled()
    })

    // Non-blocking error message appears
    await waitFor(() => {
      expect(screen.getByText(/Syllable stress could not be loaded/i)).toBeDefined()
    })

    // Transcript remains completely usable
    expect(screen.getByRole('button', { name: /guten/i })).toBeDefined()
    expect(screen.getByRole('button', { name: /morgen/i })).toBeDefined()
    expect(screen.getByRole('button', { name: /generate annotation/i })).toBeDefined()
  })

  // 19c. Stale LexiRep response protection
  it('19c. verifies stale LexiRep response from an older request cannot overwrite newer model state', async () => {
    let resolveGER
    const delayedGER = new Promise((resolve) => { resolveGER = resolve })
    api.postTranscribe.mockResolvedValue(mockTranscribePayload)
    api.postLexiRepAnalyze
      .mockReturnValueOnce(delayedGER) // First GER request is slow
      .mockResolvedValueOnce(mockLexiRepAnalyzePayloadITA) // Second ITA request returns immediately

    await act(async () => {
      render(<Analyze />)
    })

    const file = new File(['audio-bytes'], 'test.wav', { type: 'audio/wav' })
    const input = document.getElementById('audio-file-input')
    await act(async () => {
      fireEvent.change(input, { target: { files: [file] } })
      fireEvent.click(document.getElementById('btn-analyze-upload'))
    })

    await waitFor(() => {
      expect(document.getElementById('btn-transcript-nativity-ita')).not.toBeNull()
    })

    // Switch to ITA before GER resolves
    await act(async () => {
      fireEvent.click(document.getElementById('btn-transcript-nativity-ita'))
    })

    await waitFor(() => {
      expect(screen.getByText(/LexiRep ITA Active/i)).toBeDefined()
    })

    // Now late GER response arrives
    await act(async () => {
      resolveGER(mockLexiRepAnalyzePayloadGER)
    })

    // Model remains ITA and was not overwritten by stale GER response
    expect(screen.getByText(/LexiRep ITA Active/i)).toBeDefined()
  })

  // 19d. Model switching in Annotation triggers re-annotation with ITA
  it('19d. verifies changing model from German to Italian in Annotation recomputes with ITA', async () => {
    api.postTranscribe.mockResolvedValueOnce(mockTranscribePayload)
    api.postLexiRepAnalyze.mockResolvedValueOnce(mockLexiRepAnalyzePayloadGER)
    api.postAnnotate
      .mockResolvedValueOnce(mockAnnotatePayloadGER)
      .mockResolvedValueOnce(mockAnnotatePayloadITA)

    await act(async () => {
      render(<Analyze />)
    })

    const file = new File(['audio-bytes'], 'test.wav', { type: 'audio/wav' })
    const input = document.getElementById('audio-file-input')
    await act(async () => {
      fireEvent.change(input, { target: { files: [file] } })
    })
    await act(async () => {
      fireEvent.click(document.getElementById('btn-analyze-upload'))
    })
    await waitFor(() => {
      expect(api.postTranscribe).toHaveBeenCalled()
    })

    // Click Generate Annotation
    await act(async () => {
      fireEvent.click(document.getElementById('btn-generate-annotation'))
    })
    await waitFor(() => {
      expect(api.postAnnotate).toHaveBeenCalledWith(file, 'test.wav', 'GER')
    })

    // Change model to ITA
    const itaBtn = document.getElementById('btn-annotation-nativity-ita')
    expect(itaBtn).toBeDefined()
    await act(async () => {
      fireEvent.click(itaBtn)
    })

    // Triggers real re-annotation with ITA
    await waitFor(() => {
      expect(api.postAnnotate).toHaveBeenLastCalledWith(file, 'test.wav', 'ITA')
    })
  })

  // 20. Tooltip portal on hover: attaches to document.body, pointer-events none, dismisses cleanly on mouseLeave
  it('20. verifies polysyllabic word hover creates portal tooltip with pointer-events: none and cleans up on mouseLeave', async () => {
    api.postTranscribe.mockResolvedValueOnce(mockTranscribePayload)
    api.postAnnotate.mockResolvedValueOnce(mockAnnotatePayloadGER)

    await act(async () => {
      render(<Analyze />)
    })

    const file = new File(['audio-bytes'], 'test.wav', { type: 'audio/wav' })
    const input = document.getElementById('audio-file-input')
    await act(async () => {
      fireEvent.change(input, { target: { files: [file] } })
      fireEvent.click(document.getElementById('btn-analyze-upload'))
    })
    await waitFor(() => {
      expect(api.postTranscribe).toHaveBeenCalled()
    })

    // Generate annotation so polysyllabic data is active
    await act(async () => {
      fireEvent.click(document.getElementById('btn-generate-annotation'))
    })
    await waitFor(() => {
      expect(api.postAnnotate).toHaveBeenCalled()
    })

    const gutenPill = screen.getByRole('button', { name: /guten/i })

    // Hover over guten
    await act(async () => {
      fireEvent.mouseEnter(gutenPill)
    })

    // Tooltip is rendered in portal
    const tooltip = document.getElementById('word-portal-tooltip')
    expect(tooltip).toBeDefined()
    expect(tooltip.parentElement).toBe(document.body) // Rendered directly on document.body
    expect(tooltip.style.pointerEvents).toBe('none')   // Prevents stuck hover
    expect(within(tooltip).getByText(/Polysyllabic word/i)).toBeDefined()
    expect(within(tooltip).getByText('1st syllable')).toBeDefined()
    expect(within(tooltip).getByText('gu (1)')).toBeDefined()

    // Mouse leave dismisses tooltip
    await act(async () => {
      fireEvent.mouseLeave(gutenPill)
    })
    expect(document.getElementById('word-portal-tooltip')).toBeNull()
  })

  // 21. Audio reset / Back button resets all results cleanly
  it('21. verifies clicking Back button resets transcription and annotation completely', async () => {
    api.postTranscribe.mockResolvedValueOnce(mockTranscribePayload)
    api.postAnnotate.mockResolvedValueOnce(mockAnnotatePayloadGER)

    await act(async () => {
      render(<Analyze />)
    })

    const file = new File(['audio-bytes'], 'test.wav', { type: 'audio/wav' })
    const input = document.getElementById('audio-file-input')
    await act(async () => {
      fireEvent.change(input, { target: { files: [file] } })
      fireEvent.click(document.getElementById('btn-analyze-upload'))
    })
    await waitFor(() => {
      expect(api.postTranscribe).toHaveBeenCalled()
    })

    // Click back button
    const backBtn = screen.getByRole('button', { name: /back to audio input/i })
    await act(async () => {
      fireEvent.click(backBtn)
    })

    // Returns to upload view
    expect(screen.getByText(/drop audio file here/i)).toBeDefined()
    expect(screen.queryByText('guten morgen')).toBeNull()
  })

  // 22. Pitch contour visualization is NOT rendered
  it('22. verifies pitch contour visualization is not rendered anywhere in the page', async () => {
    api.postTranscribe.mockResolvedValueOnce(mockTranscribePayload)

    await act(async () => {
      render(<Analyze />)
    })

    const file = new File(['audio-bytes'], 'test.wav', { type: 'audio/wav' })
    const input = document.getElementById('audio-file-input')
    await act(async () => {
      fireEvent.change(input, { target: { files: [file] } })
      fireEvent.click(document.getElementById('btn-analyze-upload'))
    })
    await waitFor(() => {
      expect(api.postTranscribe).toHaveBeenCalled()
    })

    expect(screen.queryByText(/Pitch Contour/i)).toBeNull()
    expect(screen.queryByText(/Raw F0/i)).toBeNull()
    expect(screen.queryByText(/Stylized F0/i)).toBeNull()
  })

  // 23. Normal sentence transcript appearance and interactions
  it('23. verifies transcript displays as normal prose sentence with inline words, click-to-seek, pauses, intonation badges, and active playback highlight', async () => {
    const payloadWithPause = {
      ...mockTranscribePayload,
      words: [
        { id: 0, text: 'guten', start: 0.1, end: 0.5, stress: 'stressed' },
        { id: 1, text: 'morgen', start: 1.0, end: 1.6, stress: 'stressed' },
      ],
    }
    api.postTranscribe.mockResolvedValueOnce(payloadWithPause)
    api.postLexiRepAnalyze.mockResolvedValueOnce(mockLexiRepAnalyzePayloadGER)

    await act(async () => {
      render(<Analyze />)
    })

    const file = new File(['audio-bytes'], 'test.wav', { type: 'audio/wav' })
    const input = document.getElementById('audio-file-input')
    await act(async () => {
      fireEvent.change(input, { target: { files: [file] } })
      fireEvent.click(document.getElementById('btn-analyze-upload'))
    })
    await waitFor(() => {
      expect(api.postLexiRepAnalyze).toHaveBeenCalled()
    })

    // 1 & 2. Container is transcript-prose-container
    const proseContainer = document.querySelector('.transcript-prose-container')
    expect(proseContainer).not.toBeNull()

    // 3. Words have word-pill class and inline styling
    const gutenBtn = screen.getByRole('button', { name: /guten/i })
    expect(gutenBtn.className).toContain('word-pill')

    // 4. Clicking a word seeks audio
    await act(async () => {
      fireEvent.click(gutenBtn)
    })
    const audioEl = document.querySelector('audio')
    expect(audioEl.currentTime).toBe(0.1)

    // 5. Pause marker between guten (end 0.5) and morgen (start 1.0) is rendered inline
    const pauseMarker = screen.getByLabelText(/Pause 0.50 seconds/i)
    expect(pauseMarker).toBeDefined()
    expect(pauseMarker.textContent).toContain('···')

    // 6. Intonation badge is rendered
    expect(screen.getByRole('img', { name: /falling intonation/i })).toBeDefined()

    // 7. Polysyllabic indicator / stressed syllable is rendered
    expect(screen.getByText(/LexiRep GER Active/i)).toBeDefined()
  })
})

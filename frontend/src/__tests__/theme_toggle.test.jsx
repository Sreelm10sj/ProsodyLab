import React from 'react'
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import { render, screen, fireEvent, act } from '@testing-library/react'
import { BrowserRouter } from 'react-router-dom'
import Navigation from '../components/Navigation.jsx'
import { ThemeProvider, useTheme } from '../ThemeContext.jsx'
import TranscriptView from '../components/analyze/TranscriptView.jsx'

vi.mock('../api.js', () => ({
  checkBackendHealth: vi.fn().mockResolvedValue({ status: 'ok', device: 'cpu' }),
}))

describe('Theme Toggle & Dark Mode Integration', () => {
  beforeEach(() => {
    localStorage.clear()
    document.documentElement.classList.remove('dark')
  })

  afterEach(() => {
    localStorage.clear()
    document.documentElement.classList.remove('dark')
  })

  it('renders theme toggle button in Navigation with light mode active by default', () => {
    render(
      <ThemeProvider>
        <BrowserRouter>
          <Navigation />
        </BrowserRouter>
      </ThemeProvider>
    )

    const toggleBtn = screen.getByRole('button', { name: /switch to dark mode/i })
    expect(toggleBtn).toBeTruthy()
    expect(toggleBtn.getAttribute('id')).toBe('btn-toggle-theme')
    expect(toggleBtn.getAttribute('title')).toBe('Switch to dark mode')
    expect(document.documentElement.classList.contains('dark')).toBe(false)
  })

  it('toggles to dark mode on click: adds .dark class and persists in localStorage', () => {
    render(
      <ThemeProvider>
        <BrowserRouter>
          <Navigation />
        </BrowserRouter>
      </ThemeProvider>
    )

    const toggleBtn = screen.getByRole('button', { name: /switch to dark mode/i })
    
    // Click toggle to enable dark mode
    act(() => {
      fireEvent.click(toggleBtn)
    })

    expect(document.documentElement.classList.contains('dark')).toBe(true)
    expect(localStorage.getItem('prosodylab_theme')).toBe('dark')
    expect(screen.getByRole('button', { name: /switch to light mode/i })).toBeTruthy()

    // Click toggle again to switch back to light mode
    act(() => {
      fireEvent.click(screen.getByRole('button', { name: /switch to light mode/i }))
    })

    expect(document.documentElement.classList.contains('dark')).toBe(false)
    expect(localStorage.getItem('prosodylab_theme')).toBe('light')
    expect(screen.getByRole('button', { name: /switch to dark mode/i })).toBeTruthy()
  })

  it('initializes in dark mode if localStorage already contains dark', () => {
    localStorage.setItem('prosodylab_theme', 'dark')

    render(
      <ThemeProvider>
        <BrowserRouter>
          <Navigation />
        </BrowserRouter>
      </ThemeProvider>
    )

    expect(document.documentElement.classList.contains('dark')).toBe(true)
    const toggleBtn = screen.getByRole('button', { name: /switch to light mode/i })
    expect(toggleBtn).toBeTruthy()
  })

  it('useTheme provides fallback functions when rendered outside ThemeProvider', () => {
    function FallbackConsumer() {
      const { isDark, toggleTheme } = useTheme()
      return (
        <button type="button" onClick={toggleTheme} data-testid="fallback-btn">
          {isDark ? 'Dark' : 'Light'}
        </button>
      )
    }

    render(<FallbackConsumer />)
    const btn = screen.getByTestId('fallback-btn')
    expect(btn.textContent).toBe('Light')

    act(() => {
      fireEvent.click(btn)
    })
    expect(document.documentElement.classList.contains('dark')).toBe(true)
  })

  it('verifies polysyllabic words in transcript have no word-level yellow line while stressed syllables retain .syl-stressed', () => {
    const mockWords = [
      {
        id: 1,
        text: 'photograph',
        start: 0.1,
        end: 0.7,
        stress: 'stressed',
        is_polysyllabic: true,
        syllables: [
          { text: 'pho', is_stressed: true },
          { text: 'to', is_stressed: false },
          { text: 'graph', is_stressed: false },
        ],
        lexical_stress_index: 0,
      },
    ]

    render(
      <ThemeProvider>
        <TranscriptView
          transcript="photograph"
          words={mockWords}
          currentTime={0}
          isPlaying={false}
        />
      </ThemeProvider>
    )

    const wordBtn = screen.getByRole('button', { name: /photograph/i })
    expect(wordBtn).toBeTruthy()

    // The whole word has word-pill-poly and word-pill-stressed
    expect(wordBtn.classList.contains('word-pill-poly')).toBe(true)
    expect(wordBtn.classList.contains('word-pill-stressed')).toBe(true)

    // The stressed syllable "pho" has syl-stressed
    const stressedSyl = wordBtn.querySelector('.syl-stressed')
    expect(stressedSyl).toBeTruthy()
    expect(stressedSyl.textContent).toBe('pho')

    // Unstressed syllables have syl-unstressed
    const unstressedSyls = wordBtn.querySelectorAll('.syl-unstressed')
    expect(unstressedSyls.length).toBe(2)
  })
})

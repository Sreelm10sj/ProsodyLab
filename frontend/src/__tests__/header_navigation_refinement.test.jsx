import React from 'react'
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import { render, screen, act } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import Navigation from '../components/Navigation.jsx'
import Wordmark from '../components/Wordmark.jsx'
import { ThemeProvider } from '../ThemeContext.jsx'

vi.mock('../api.js', () => ({
  getHealth: vi.fn().mockResolvedValue({ status: 'ok', device: 'cpu', model_loaded: true }),
}))

describe('Header & Navigation Refinements', () => {
  beforeEach(() => {
    localStorage.clear()
    document.documentElement.classList.remove('dark')
  })

  afterEach(() => {
    localStorage.clear()
    document.documentElement.classList.remove('dark')
  })

  it('1. Wordmark renders Prosody text with dark mode classes and no hardcoded dark inline color', () => {
    const { container } = render(<Wordmark variant="refined" />)
    const prosodySpan = screen.getByText('Prosody')
    expect(prosodySpan).toBeTruthy()
    expect(prosodySpan.classList.contains('dark:text-slate-100')).toBe(true)

    // The root span should also have dark:text-slate-100 and no style color: '#0F172A'
    const wordmarkRoot = container.firstChild
    expect(wordmarkRoot.classList.contains('dark:text-slate-100')).toBe(true)
    expect(wordmarkRoot.style.color).toBe('')
  })

  it('2. Navigation does NOT render a Docs link in the header', async () => {
    await act(async () => {
      render(
        <ThemeProvider>
          <MemoryRouter initialEntries={['/analyze']}>
            <Navigation />
          </MemoryRouter>
        </ThemeProvider>
      )
    })

    // Docs should not exist
    expect(screen.queryByRole('link', { name: /docs/i })).toBeNull()
    expect(screen.queryByText(/^docs$/i)).toBeNull()

    // Analyze and Train Model links must exist
    expect(screen.getByRole('link', { name: /analyze/i })).toBeTruthy()
    expect(screen.getByRole('link', { name: /train model/i })).toBeTruthy()
  })

  it('3. Navigation renders active underline on Analyze when on /analyze and on Train Model when on /train', async () => {
    let unmountFn
    await act(async () => {
      const res = render(
        <ThemeProvider>
          <MemoryRouter initialEntries={['/analyze']}>
            <Navigation />
          </MemoryRouter>
        </ThemeProvider>
      )
      unmountFn = res.unmount
    })

    const analyzeLink = screen.getByRole('link', { name: /analyze/i })
    expect(analyzeLink.parentElement.querySelector('[aria-hidden="true"]')).toBeTruthy()
    unmountFn()

    await act(async () => {
      render(
        <ThemeProvider>
          <MemoryRouter initialEntries={['/train']}>
            <Navigation />
          </MemoryRouter>
        </ThemeProvider>
      )
    })

    const trainLink = screen.getByRole('link', { name: /train model/i })
    expect(trainLink.parentElement.querySelector('[aria-hidden="true"]')).toBeTruthy()
  })
})

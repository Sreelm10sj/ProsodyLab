import { useState, useCallback, useEffect } from 'react'
import { createPortal } from 'react-dom'
import PhraseMarker from './PhraseMarker.jsx'

/**
 * TranscriptView — Word-level stress / pause / phrase rendering with LexiRep syllable stress.
 * Features:
 *   - Discrete floating word pill tokens
 *   - Luminous warm yellow background for stressed words
 *   - Soft sky-blue pill badge for pauses
 *   - Interactive click-to-seek
 *   - Zero-clipping React Portal tooltip attached to document.body with collision detection
 *   - Pointer-events: none on tooltip to prevent stuck hover states
 */

const PAUSE_THRESHOLD_S = 0.35

export default function TranscriptView({
  transcript,
  words            = [],
  phrases          = [],
  currentTime      = 0,
  isPlaying        = false,
  lexirepResult    = null,
  onSeek           = null,
  lexirepNativity  = 'GER',
  onNativityChange = null,
  lexirepLoading   = false,
  lexirepError     = null,
}) {
  const [viewMode, setViewMode] = useState('words') // 'words' | 'syllables'
  const [hoveredWord, setHoveredWord] = useState(null)

  // Clear hover state on transcript, words, or nativity updates
  useEffect(() => {
    setHoveredWord(null)
  }, [transcript, words, lexirepResult, lexirepNativity])

  // Clear hover on window scroll or resize to prevent detached/stuck tooltips
  useEffect(() => {
    const clearHover = () => setHoveredWord(null)
    window.addEventListener('scroll', clearHover, true)
    window.addEventListener('resize', clearHover)
    return () => {
      window.removeEventListener('scroll', clearHover, true)
      window.removeEventListener('resize', clearHover)
    }
  }, [])

  // Build phrase index: map lastWordIndex → phrase object
  const phraseEndAt = {}
  for (let pi = 0; pi < phrases.length; pi++) {
    const p = phrases[pi]
    let endIdx = p.end_word_idx ?? (p.word_indices?.at(-1))
    if (endIdx === undefined && words.length > 0 && p.end != null) {
      let bestIdx = -1
      let minDiff = Infinity
      for (let wi = 0; wi < words.length; wi++) {
        const diff = Math.abs(words[wi].end - p.end)
        if (diff < minDiff) {
          minDiff = diff
          bestIdx = wi
        }
      }
      if (bestIdx !== -1 && minDiff < 0.45) {
        endIdx = bestIdx
      } else {
        for (let wi = words.length - 1; wi >= 0; wi--) {
          if (words[wi].start <= p.end) {
            endIdx = wi
            break
          }
        }
      }
    }
    if (endIdx !== undefined && endIdx >= 0 && endIdx < words.length) {
      phraseEndAt[endIdx] = { ...p, phraseIdx: pi }
    }
  }

  const handleWordClick = useCallback((start) => {
    if (onSeek && start != null) {
      onSeek(start)
    }
  }, [onSeek])

  const tokens = []
  for (let i = 0; i < words.length; i++) {
    const w = words[i]
    const wordText = w.text ?? w.word ?? ''
    const isStressed = w.stress === 'stressed' || Boolean(w.stressed) || Boolean(w.is_stressed)

    const lxWord = (w.is_polysyllabic && w.syllables?.length >= 2) ? w : lexirepResult?.words?.[i]
    const hasLexiRep = Boolean(lxWord && lxWord.is_polysyllabic && lxWord.syllables?.length >= 2)
    const speakerNativity = lxWord?.lexirep?.model ?? lexirepResult?.nativity ?? lexirepNativity ?? 'GER'
    const stressedSylIdx = lxWord?.lexical_stress_index ?? lxWord?.lexirep_stressed_syllable_index ?? 0

    // Highlight active playing word
    const isActive = isPlaying && currentTime >= w.start && currentTime < w.end

    // Word display: standard word or segmented syllables
    let wordContent = wordText
    if (viewMode === 'syllables' && hasLexiRep) {
      wordContent = lxWord.syllables.map((s, sIdx) => {
        const isSylStressed = s.stress === 1 || s.is_stressed === true || s.is_stressed === 1 || sIdx === stressedSylIdx
        return (
          <span key={sIdx} className="inline-flex items-center">
            {sIdx > 0 && <span className="syl-bullet opacity-40 mx-0.5 font-normal">·</span>}
            <span className={isSylStressed ? 'syl-stressed' : 'syl-unstressed'}>
              {isSylStressed ? s.text.toUpperCase() : s.text}
            </span>
          </span>
        )
      })
    } else if (hasLexiRep && lxWord.syllables?.length >= 2) {
      const joinedSyl = lxWord.syllables.map(s => s.text).join('')
      if (joinedSyl.toLowerCase() === wordText.toLowerCase()) {
        let cursor = 0
        wordContent = lxWord.syllables.map((s, sIdx) => {
          const part = wordText.slice(cursor, cursor + s.text.length)
          cursor += s.text.length
          const isSylStressed = s.stress === 1 || s.is_stressed === true || s.is_stressed === 1 || sIdx === stressedSylIdx
          return (
            <span
              key={sIdx}
              className={isSylStressed ? 'syl-stressed' : 'syl-unstressed'}
            >
              {part}
            </span>
          )
        })
      }
    }

    tokens.push(
      <span
        key={`w-${i}`}
        className="relative inline-flex items-baseline mx-0.5 my-1"
      >
        <button
          type="button"
          onClick={() => handleWordClick(w.start)}
          onMouseEnter={(e) => {
            const rect = e.currentTarget.getBoundingClientRect()
            setHoveredWord({
              index: i,
              wordText,
              isStressed,
              hasLexiRep,
              lxWord,
              speakerNativity,
              stressedSylIdx,
              rect,
              start: w.start,
              end: w.end,
              confidence: w.confidence,
            })
          }}
          onMouseLeave={() => {
            setHoveredWord(null)
          }}
          className={`word-pill text-lg md:text-xl font-normal tracking-normal select-none transition-all ${
            isStressed ? 'word-pill-stressed' : ''
          } ${hasLexiRep ? 'word-pill-poly' : ''} ${
            isActive ? 'word-pill-active' : ''
          } ${isActive && hasLexiRep ? 'activePolysyllabicWord word-pill-active-poly' : ''} ${
            isActive && isStressed ? 'word-pill-active-stressed' : ''
          } ${isActive && isStressed && hasLexiRep ? 'activePolysyllabicWord-stressed' : ''}`}
          title={`Word: "${wordText}" (${w.start?.toFixed(2)}s - ${w.end?.toFixed(2)}s). Click to seek audio.`}
        >
          {wordContent}
        </button>
      </span>
    )

    // Phrase marker after phrase ends at this word index
    if (phraseEndAt[i]) {
      tokens.push(
        <PhraseMarker
          key={`pm-${i}`}
          phrase={phraseEndAt[i]}
          id={`${i}-${phraseEndAt[i].phraseIdx}`}
        />
      )
    }

    // Pause badge: gap between this word's end and next word's start
    const next = words[i + 1]
    if (next && next.start - w.end >= PAUSE_THRESHOLD_S) {
      const gapS = (next.start - w.end).toFixed(2)
      tokens.push(
        <span
          key={`pause-${i}`}
          className="word-pill word-pill-pause text-xs font-mono font-bold mx-1.5 align-middle"
          title={`Pause duration: ${gapS} seconds`}
          aria-label={`Pause ${gapS} seconds`}
          onClick={() => onSeek && onSeek(w.end)}
        >
          ···
        </span>
      )
    }
  }

  // ── Portal Tooltip Positioning Math with Viewport Collision Handling ───────
  let tooltipPortal = null
  if (hoveredWord && typeof document !== 'undefined') {
    const { rect, hasLexiRep, lxWord, wordText, speakerNativity, stressedSylIdx, isStressed, start, end, confidence } = hoveredWord

    const TOOLTIP_WIDTH = hasLexiRep ? 240 : 180
    const TOOLTIP_HEIGHT = hasLexiRep ? 130 : 64
    const MARGIN = 12

    // Center horizontally relative to word pill
    let left = rect.left + rect.width / 2 - TOOLTIP_WIDTH / 2
    // Clamp horizontally inside viewport
    left = Math.max(MARGIN, Math.min(window.innerWidth - TOOLTIP_WIDTH - MARGIN, left))

    // Vertical placement: place above if space >= height + 16, else below
    const spaceAbove = rect.top
    let top = 0
    if (spaceAbove >= TOOLTIP_HEIGHT + 16) {
      top = rect.top - TOOLTIP_HEIGHT - 8
    } else {
      top = rect.bottom + 8
    }

    tooltipPortal = createPortal(
      <div
        id="word-portal-tooltip"
        role="tooltip"
        style={{
          position: 'fixed',
          left: `${left}px`,
          top: `${top}px`,
          width: `${TOOLTIP_WIDTH}px`,
          zIndex: 99999,
          pointerEvents: 'none', // Prevents intercepting mouse events and getting stuck
          transition: 'opacity 150ms ease-out, transform 150ms ease-out',
        }}
        className="pointer-events-none select-none transition-all duration-150"
      >
        {hasLexiRep ? (
          <div
            className="p-3 bg-white text-slate-800 border border-slate-200/90 rounded-xl shadow-xl space-y-2"
            style={{
              boxShadow: 'var(--shadow-popover)',
              fontFamily: 'var(--font-sans)',
            }}
          >
            <div className="flex items-center justify-between pb-1.5 border-b border-slate-100">
              <span className="font-extrabold text-xs text-slate-900 uppercase tracking-tight">
                {wordText}
              </span>
              <span className="text-[10px] font-mono font-semibold text-amber-700 bg-amber-50 border border-amber-200 px-1.5 py-0.5 rounded">
                LexiRep {speakerNativity}
              </span>
            </div>

            <div className="space-y-1 font-mono text-[11px] text-slate-600">
              <div className="flex justify-between">
                <span className="text-slate-400">Word type:</span>
                <span className="font-semibold text-slate-700">Polysyllabic word</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Syllables:</span>
                <span className="font-bold text-slate-800">
                  {lxWord.syllable_count ?? lxWord.syllables?.length}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Lexical stress:</span>
                <span className="font-bold text-amber-700">
                  {stressedSylIdx === 0
                    ? '1st syllable'
                    : stressedSylIdx === 1
                      ? '2nd syllable'
                      : `${stressedSylIdx + 1}th syllable`}
                </span>
              </div>
              {start != null && end != null && (
                <div className="flex justify-between">
                  <span className="text-slate-400">Duration:</span>
                  <span className="font-bold text-slate-800">
                    {((end - start) * 1000).toFixed(0)}ms ({(end - start).toFixed(2)}s)
                  </span>
                </div>
              )}
              {(lxWord.lexirep?.confidence != null || lxWord.confidence != null || lxWord.syllables?.[0]?.confidence != null) && (
                <div className="flex justify-between">
                  <span className="text-slate-400">Confidence:</span>
                  <span className="font-bold text-slate-800">
                    {(((lxWord.lexirep?.confidence ?? lxWord.confidence ?? lxWord.syllables?.[0]?.confidence)) * 100).toFixed(0)}%
                  </span>
                </div>
              )}
            </div>

            {/* Syllables chips */}
            {lxWord.syllables && lxWord.syllables.length > 0 && (
              <div className="flex flex-wrap gap-1 pt-1.5 border-t border-slate-100">
                {lxWord.syllables.map((s, sIdx) => {
                  const isPrimary = s.stress === 1
                  return (
                    <span
                      key={sIdx}
                      className={`px-1.5 py-0.5 rounded text-[10px] font-mono ${
                        isPrimary
                          ? 'bg-amber-100 text-amber-900 font-bold border border-amber-300'
                          : 'bg-slate-100 text-slate-600'
                      }`}
                    >
                      {s.text} {isPrimary ? '(1)' : '(0)'}
                    </span>
                  )
                })}
              </div>
            )}
          </div>
        ) : (
          <div className="bg-slate-900 text-white rounded-lg shadow-lg px-3 py-2 text-xs font-mono space-y-0.5">
            {isStressed && <span className="block font-bold text-amber-400 text-[11px]">⚡ Stressed word</span>}
            <span className="block text-[10px] text-slate-300">
              {start?.toFixed(2)}s → {end?.toFixed(2)}s
            </span>
            {confidence != null && (
              <span className="block text-[10px] text-slate-400">
                conf: {(confidence * 100).toFixed(0)}%
              </span>
            )}
          </div>
        )}
      </div>,
      document.body
    )
  }

  return (
    <div className="w-full bg-white rounded-2xl border border-slate-200/90 shadow-sm overflow-hidden select-none">
      
      {/* ── Sub-header ── */}
      <div className="px-6 py-3.5 bg-slate-50/70 border-b border-slate-200/80 flex items-center justify-between flex-wrap gap-3">
        <div className="flex items-center gap-3">
          <span className="text-xs font-semibold uppercase tracking-wider text-slate-600 font-mono">
            Transcript
          </span>
          {(lexirepResult || words.some(w => w.is_polysyllabic && w.syllables?.length >= 2)) && (
            <span className="text-[11px] font-mono font-semibold px-2 py-0.5 rounded-md bg-amber-50 text-amber-800 border border-amber-200">
              LexiRep {lexirepResult?.nativity ?? words.find(w => w.lexirep?.model)?.lexirep?.model ?? lexirepNativity ?? 'GER'} Active
            </span>
          )}
          {lexirepLoading && (
            <span className="text-[11px] font-mono text-amber-600 animate-pulse font-medium">
              Detecting syllable stress...
            </span>
          )}
          {lexirepError && (
            <span className="text-[11px] font-mono text-red-600 font-medium">
              {lexirepError}
            </span>
          )}
        </div>

        {/* Right side controls: Syllable Stress Model + Words/Syllables view switcher */}
        <div className="flex items-center gap-4 flex-wrap">
          {/* Speaker L1 Model Selector */}
          <div className="flex items-center gap-2">
            <span className="text-xs font-mono text-slate-500 font-medium">Syllable Stress Model:</span>
            <div className="inline-flex rounded-lg border border-slate-200 p-0.5 bg-white shadow-2xs">
              <button
                type="button"
                id="btn-transcript-nativity-ger"
                onClick={() => onNativityChange?.('GER')}
                aria-pressed={lexirepNativity === 'GER'}
                className={`text-xs font-semibold px-2.5 py-1 rounded-md transition-all ${
                  lexirepNativity === 'GER'
                    ? 'bg-slate-900 text-white shadow-xs'
                    : 'text-slate-600 hover:text-slate-900 hover:bg-slate-50'
                }`}
              >
                German (GER)
              </button>
              <button
                type="button"
                id="btn-transcript-nativity-ita"
                onClick={() => onNativityChange?.('ITA')}
                aria-pressed={lexirepNativity === 'ITA'}
                className={`text-xs font-semibold px-2.5 py-1 rounded-md transition-all ${
                  lexirepNativity === 'ITA'
                    ? 'bg-slate-900 text-white shadow-xs'
                    : 'text-slate-600 hover:text-slate-900 hover:bg-slate-50'
                }`}
              >
                Italian (ITA)
              </button>
            </div>
            {lexirepLoading && (
              <span className="text-[11px] font-mono text-amber-600 animate-pulse font-medium">
                Detecting syllable stress...
              </span>
            )}
          </div>

          {/* Syllables / Words switcher */}
          {(lexirepResult || words.some(w => w.is_polysyllabic && w.syllables?.length >= 2)) && (
            <div className="flex items-center gap-2 pl-2 border-l border-slate-200">
              <span className="text-xs font-mono text-slate-500">View:</span>
              <div className="inline-flex rounded-lg border border-slate-200 p-0.5 bg-white">
                <button
                  type="button"
                  onClick={() => setViewMode('words')}
                  className={`text-xs font-medium px-2.5 py-1 rounded-md transition-colors ${
                    viewMode === 'words' ? 'bg-slate-900 text-white font-semibold' : 'text-slate-600 hover:text-slate-900'
                  }`}
                >
                  Words
                </button>
                <button
                  type="button"
                  onClick={() => setViewMode('syllables')}
                  className={`text-xs font-medium px-2.5 py-1 rounded-md transition-colors ${
                    viewMode === 'syllables' ? 'bg-slate-900 text-white font-semibold' : 'text-slate-600 hover:text-slate-900'
                  }`}
                >
                  Syllables
                </button>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* ── Normal Readable Sentence Transcript Area ── */}
      <div className="p-6 md:p-8">
        <div
          className="transcript-prose-container text-lg md:text-xl text-slate-900 font-normal leading-relaxed"
          style={{ lineHeight: 2.2 }}
          aria-label={transcript}
        >
          {tokens}
        </div>
      </div>

      {tooltipPortal}
    </div>
  )
}

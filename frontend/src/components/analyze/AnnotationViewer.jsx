import { useState, useCallback } from 'react'
import { Download, ChevronDown, ChevronUp } from 'lucide-react'

/**
 * AnnotationViewer — displays structured annotation from /api/annotate.
 *
 * REAL backend shape (from annotation/pipeline.py):
 *   metadata:     { audio_file, duration }
 *   transcription:{ text, language }        ← object, NOT a string
 *   words:        [{ id, text, start, end, stress }]
 *                   stress: "stressed" | "unstressed"
 *   pauses:       [{ start, end, duration }]
 *   intonation:   [{ phrase_id, start, end, slope, label }]
 *                   label: "rising" | "falling" | "level" | "insufficient_voicing"
 *   timeline:     [{ type, id?, start, end, duration? }]
 */

function SectionHeader({ label, count, isOpen = true, onToggle = null }) {
  return (
    <div
      onClick={onToggle}
      className={`px-4 py-2.5 bg-slate-50 border-y border-slate-200/80 flex items-center justify-between select-none ${
        onToggle ? 'cursor-pointer hover:bg-slate-100/70 transition-colors' : ''
      }`}
    >
      <div className="flex items-center gap-2">
        <span className="font-mono font-bold text-[11px] uppercase tracking-wider text-slate-800">
          {label} {count != null ? `(${count})` : ''}
        </span>
      </div>
      {onToggle && (
        <span className="text-slate-400">
          {isOpen ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
        </span>
      )}
    </div>
  )
}

function DownloadButton({ id, label, onClick }) {
  return (
    <button
      type="button"
      id={id}
      onClick={onClick}
      className="btn-white-action px-3 py-1.5 text-xs font-mono font-semibold shadow-2xs flex items-center gap-1.5"
      aria-label={label}
    >
      <Download size={13} className="text-slate-500" />
      <span>{label}</span>
    </button>
  )
}

const INTONATION_META = {
  rising:                { arrow: '↗', color: '#16A34A', bg: '#DCFCE7' },
  falling:               { arrow: '↘', color: '#2563EB', bg: '#DBEAFE' },
  level:                 { arrow: '→', color: '#64748B', bg: '#F1F5F9' },
  insufficient_voicing:  { arrow: '~', color: '#94A3B8', bg: '#F8FAFC' },
}

export default function AnnotationViewer({
  annotation,
  lexirepResult = null,
  lexirepNativity = 'GER',
  onNativityChange = null,
  lexirepLoading = false,
}) {
  const [showWords, setShowWords] = useState(true)
  const [showSyllables, setShowSyllables] = useState(true)
  const [showPauses, setShowPauses] = useState(true)
  const [showIntonation, setShowIntonation] = useState(true)

  // Normalise transcription — may be string (old) or {text,language} (real)
  const txText = typeof annotation?.transcription === 'string'
    ? annotation.transcription
    : annotation?.transcription?.text ?? ''
  const txLang = annotation?.transcription?.language

  const downloadJSON = useCallback(() => {
    const payload = lexirepResult && !annotation.metadata?.speaker_nativity
      ? { ...annotation, lexirep: lexirepResult }
      : annotation
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' })
    const url  = URL.createObjectURL(blob)
    const a    = document.createElement('a')
    a.href = url; a.download = 'annotation.json'; a.click()
    URL.revokeObjectURL(url)
  }, [annotation, lexirepResult])

  const downloadTXT = useCallback(() => {
    const words = annotation.words ?? []
    const polyWords = (annotation.words ?? lexirepResult?.words ?? []).filter(w => w.is_polysyllabic && w.syllables?.length >= 2)
    const speakerNativity = annotation.metadata?.speaker_nativity ?? lexirepResult?.nativity ?? lexirepNativity ?? 'GER'
    const lines = [
      'PROSODYLAB ANNOTATION',
      `Generated: ${new Date().toISOString()}`,
      '',
      'TRANSCRIPTION',
      txText,
      '',
      'WORDS (WHISTRESS PROMINENCE)',
      ...words.map(w => {
        const word = w.text ?? w.word ?? '?'
        const stressed = w.stress === 'stressed' || Boolean(w.stressed) || Boolean(w.is_stressed)
        return `  ${word.padEnd(20)} ${w.start?.toFixed(3) ?? '?'}s → ${w.end?.toFixed(3) ?? '?'}s  whistress_prominence=${stressed ? 'stressed' : 'unstressed'}`
      }),
      '',
      ...(polyWords.length > 0 ? [
        `LEXIREP SYLLABLE LEXICAL STRESS (Speaker L1: ${speakerNativity})`,
        ...polyWords.flatMap(w => [
          `  Word: ${w.text ?? w.word} [${w.start?.toFixed(3)}s → ${w.end?.toFixed(3)}s] (K=${w.syllable_count}, Primary Stressed Syllable=${(w.lexical_stress_index ?? w.lexirep_stressed_syllable_index ?? 0) + 1})`,
          ...w.syllables.map(s =>
            `    Syllable ${s.index + 1}: ${s.text.padEnd(10)} [${s.start?.toFixed(3)}s → ${s.end?.toFixed(3)}s] stress=${s.stress != null ? (s.stress === 1 ? '1 (stressed)' : '0 (unstressed)') : 'null'} conf=${s.confidence != null ? (s.confidence * 100).toFixed(0) + '%' : '—'}`
          )
        ]),
        '',
      ] : []),
      'PAUSES',
      ...(annotation.pauses ?? []).map(p =>
        `  ${p.start?.toFixed(3)}s → ${p.end?.toFixed(3)}s  (${p.duration?.toFixed(3)}s)`
      ),
      '',
      'INTONATION',
      ...(annotation.intonation ?? []).map(i => {
        const label = i.label ?? i.intonation ?? '?'
        const slope = i.slope != null ? ` slope=${i.slope.toFixed(2)}st/s` : ''
        return `  ${label.padEnd(24)} ${i.start?.toFixed(3)}s → ${i.end?.toFixed(3)}s${slope}`
      }),
    ]
    const blob = new Blob([lines.join('\n')], { type: 'text/plain' })
    const url  = URL.createObjectURL(blob)
    const a    = document.createElement('a')
    a.href = url; a.download = 'annotation.txt'; a.click()
    URL.revokeObjectURL(url)
  }, [annotation, txText, lexirepResult, lexirepNativity])

  if (!annotation) return null

  const words      = annotation.words      ?? []
  const pauses     = annotation.pauses     ?? []
  const intonation = annotation.intonation ?? []
  const polyWords  = (annotation.words ?? lexirepResult?.words ?? []).filter(w => w.is_polysyllabic && w.syllables?.length >= 2)
  const currentModel = annotation.metadata?.speaker_nativity ?? lexirepResult?.nativity ?? lexirepNativity ?? 'GER'

  return (
    <div className="bg-white rounded-2xl border border-slate-200/90 shadow-sm overflow-hidden select-none">
      
      {/* ── Header Bar ── */}
      <div className="px-6 py-4 bg-slate-50/70 border-b border-slate-200/80 flex items-center justify-between flex-wrap gap-4">
        <div className="flex items-center gap-3">
          <span className="font-mono font-bold text-xs uppercase tracking-wider text-slate-800">
            Annotation
          </span>
          <span className="text-[11px] font-mono font-semibold px-2 py-0.5 rounded-md bg-amber-50 text-amber-800 border border-amber-200">
            LexiRep {currentModel} Active
          </span>
        </div>

        {/* Action Controls: Model Selector & Downloads */}
        <div className="flex items-center gap-3 flex-wrap">
          {/* Model Selector */}
          <div className="flex items-center gap-2">
            <span className="text-xs font-mono text-slate-500 font-medium">Syllable Stress Model:</span>
            <div className="inline-flex rounded-lg border border-slate-200 p-0.5 bg-white shadow-2xs">
              <button
                type="button"
                id="btn-annotation-nativity-ger"
                onClick={() => onNativityChange?.('GER')}
                disabled={lexirepLoading}
                aria-pressed={currentModel === 'GER'}
                className={`text-xs font-semibold px-2.5 py-1 rounded-md transition-all ${
                  currentModel === 'GER'
                    ? 'bg-slate-900 text-white shadow-xs'
                    : 'text-slate-600 hover:text-slate-900 hover:bg-slate-50'
                }`}
              >
                German (GER)
              </button>
              <button
                type="button"
                id="btn-annotation-nativity-ita"
                onClick={() => onNativityChange?.('ITA')}
                disabled={lexirepLoading}
                aria-pressed={currentModel === 'ITA'}
                className={`text-xs font-semibold px-2.5 py-1 rounded-md transition-all ${
                  currentModel === 'ITA'
                    ? 'bg-slate-900 text-white shadow-xs'
                    : 'text-slate-600 hover:text-slate-900 hover:bg-slate-50'
                }`}
              >
                Italian (ITA)
              </button>
            </div>
            {lexirepLoading && (
              <span className="text-[11px] font-mono text-amber-600 animate-pulse font-medium">
                Updating…
              </span>
            )}
          </div>

          <div className="h-4 w-px bg-slate-200 hidden sm:block" />

          {/* Download buttons */}
          <div className="flex items-center gap-2">
            <DownloadButton id="btn-download-json" label="Download JSON" onClick={downloadJSON} />
            <DownloadButton id="btn-download-txt"  label="Download TXT"  onClick={downloadTXT}  />
          </div>
        </div>
      </div>

      <div className="p-6 md:p-8 space-y-6">

        {/* ── 1. Words Section ── */}
        <div data-testid="section-words" className="border border-slate-200 rounded-xl overflow-hidden shadow-2xs">
          <SectionHeader
            label="Words"
            count={words.length}
            isOpen={showWords}
            onToggle={() => setShowWords(!showWords)}
          />
          {showWords && (
            <div className="overflow-x-auto max-h-72 overflow-y-auto">
              <table className="w-full font-mono text-xs border-collapse">
                <thead className="sticky top-0 bg-slate-50 border-b border-slate-200 text-slate-600">
                  <tr>
                    {['#', 'Word', 'Start', 'End', 'Duration', 'Stress'].map(h => (
                      <th key={h} className="p-3 text-left font-semibold">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {words.map((w, i) => {
                    const word = w.text ?? w.word ?? '?'
                    const stressed = w.stress === 'stressed' || Boolean(w.stressed) || Boolean(w.is_stressed)
                    return (
                      <tr key={w.id ?? i} className="border-b border-slate-100 hover:bg-slate-50/60">
                        <td className="p-3 text-slate-400">{w.id ?? i + 1}</td>
                        <td className="p-3 font-bold text-slate-900">{word}</td>
                        <td className="p-3 text-slate-500">{w.start?.toFixed(3) ?? '—'}s</td>
                        <td className="p-3 text-slate-500">{w.end?.toFixed(3) ?? '—'}s</td>
                        <td className="p-3 text-slate-500">{(w.end - w.start)?.toFixed(3)}s</td>
                        <td className="p-3">
                          {stressed ? (
                            <span className="px-2 py-0.5 rounded font-bold bg-amber-100 text-amber-900 border border-amber-300">
                              ● Stressed
                            </span>
                          ) : (
                            <span className="px-2 py-0.5 rounded text-slate-500 bg-slate-100">
                              ○ Unstressed
                            </span>
                          )}
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {/* ── 2. Syllables Section ── */}
        <div data-testid="section-syllables" className="border border-slate-200 rounded-xl overflow-hidden shadow-2xs">
          <SectionHeader
            label="Syllables"
            count={polyWords.length > 0 ? `${polyWords.reduce((acc, w) => acc + (w.syllables?.length ?? 0), 0)} syllables` : '0'}
            isOpen={showSyllables}
            onToggle={() => setShowSyllables(!showSyllables)}
          />
          {showSyllables && (
            polyWords.length > 0 ? (
              <div className="overflow-x-auto max-h-72 overflow-y-auto">
                <table className="w-full font-mono text-xs border-collapse">
                  <thead className="sticky top-0 bg-slate-50 border-b border-slate-200 text-slate-600">
                    <tr>
                      {['Word', 'K', 'Syllable', 'Interval', 'Duration', 'Lexical Stress', 'Confidence'].map(h => (
                        <th key={h} className="p-3 text-left font-semibold">{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {polyWords.flatMap((w, wIdx) =>
                      w.syllables.map((s, sIdx) => {
                        const isPrimary = s.stress === 1
                        return (
                          <tr
                            key={`${wIdx}-${sIdx}`}
                            className={`border-b border-slate-100 hover:bg-slate-50/60 ${
                              isPrimary ? 'bg-amber-50/30' : ''
                            }`}
                          >
                            <td className="p-3 font-bold text-slate-900">
                              {sIdx === 0 ? (w.text ?? w.word) : ''}
                            </td>
                            <td className="p-3 text-slate-400">
                              {sIdx === 0 ? `K=${w.syllable_count ?? w.syllables?.length}` : ''}
                            </td>
                            <td className={`p-3 font-bold ${isPrimary ? 'text-amber-700' : 'text-slate-700'}`}>
                              {isPrimary ? `${s.text} ●` : s.text}
                            </td>
                            <td className="p-3 text-slate-500">
                              {s.start?.toFixed(3)}s → {s.end?.toFixed(3)}s
                            </td>
                            <td className="p-3 text-slate-500">
                              {s.duration?.toFixed(3) ?? (s.end - s.start)?.toFixed(3)}s
                            </td>
                            <td className="p-3">
                              {isPrimary ? (
                                <span className="px-2 py-0.5 rounded font-bold bg-amber-100 text-amber-900 border border-amber-300">
                                  ● Stressed (1)
                                </span>
                              ) : (
                                <span className="px-2 py-0.5 rounded text-slate-500 bg-slate-100">
                                  ○ Unstressed (0)
                                </span>
                              )}
                            </td>
                            <td className="p-3 font-semibold text-slate-700">
                              {s.confidence != null ? `${(s.confidence * 100).toFixed(0)}%` : '—'}
                            </td>
                          </tr>
                        )
                      })
                    )}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="p-4 text-xs font-mono text-slate-500">
                No polysyllabic words detected.
              </div>
            )
          )}
        </div>

        {/* ── 3. Pauses Section ── */}
        <div data-testid="section-pauses" className="border border-slate-200 rounded-xl overflow-hidden shadow-2xs">
          <SectionHeader
            label="Pauses"
            count={pauses.length}
            isOpen={showPauses}
            onToggle={() => setShowPauses(!showPauses)}
          />
          {showPauses && (
            pauses.length > 0 ? (
              <div className="p-4 grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-2.5 bg-white">
                {pauses.map((p, i) => (
                  <div
                    key={i}
                    className="p-2.5 rounded-lg border border-slate-100 bg-slate-50/60 font-mono text-xs flex items-center justify-between"
                  >
                    <span className="text-slate-600">
                      {p.start?.toFixed(2)}s → {p.end?.toFixed(2)}s
                    </span>
                    <span className="font-bold text-slate-800 bg-white border border-slate-200 px-1.5 py-0.5 rounded text-[11px]">
                      {p.duration?.toFixed(2)}s
                    </span>
                  </div>
                ))}
              </div>
            ) : (
              <div className="p-4 text-xs font-mono text-slate-500">
                No pauses detected.
              </div>
            )
          )}
        </div>

        {/* ── 4. Intonations Section ── */}
        <div data-testid="section-intonations" className="border border-slate-200 rounded-xl overflow-hidden shadow-2xs">
          <SectionHeader
            label="Intonations"
            count={`${intonation.length} phrases`}
            isOpen={showIntonation}
            onToggle={() => setShowIntonation(!showIntonation)}
          />
          {showIntonation && (
            intonation.length > 0 ? (
              <div className="p-4 space-y-2 bg-white">
                {intonation.map((item, i) => {
                  const label = item.label ?? item.intonation ?? 'unknown'
                  const meta  = INTONATION_META[label] ?? INTONATION_META.insufficient_voicing
                  return (
                    <div
                      key={item.phrase_id ?? i}
                      className="p-3 rounded-lg border border-slate-100 hover:border-slate-200 bg-slate-50/50 flex flex-wrap items-center justify-between gap-3 text-xs font-mono"
                    >
                      <div className="flex items-center gap-3">
                        <span
                          className="px-2.5 py-1 rounded-md font-bold uppercase text-[11px] flex items-center gap-1.5"
                          style={{ color: meta.color, backgroundColor: meta.bg }}
                        >
                          <span>{meta.arrow}</span>
                          <span>{label}</span>
                        </span>
                        <span className="text-slate-600 font-medium">Phrase #{i + 1}</span>
                      </div>
                      <div className="flex items-center gap-4 text-slate-500">
                        <span>{item.start?.toFixed(2)}s → {item.end?.toFixed(2)}s</span>
                        {item.slope != null && (
                          <span className="font-semibold text-slate-700">
                            slope: {item.slope >= 0 ? '+' : ''}{item.slope.toFixed(2)} st/s
                          </span>
                        )}
                      </div>
                    </div>
                  )
                })}
              </div>
            ) : (
              <div className="p-4 text-xs font-mono text-slate-500">
                No intonation phrases detected.
              </div>
            )
          )}
        </div>

      </div>
    </div>
  )
}

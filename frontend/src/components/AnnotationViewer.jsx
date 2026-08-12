import React from 'react';

export default function AnnotationViewer({ annotation, filename }) {
  if (!annotation) return null;

  const { metadata, transcription, words, pauses, intonation, timeline } = annotation;

  const downloadFile = (content, name, type) => {
    const blob = new Blob([content], { type });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = name;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const baseName = filename ? filename.replace(/\.[^/.]+$/, '') : 'audio';

  const handleDownloadJSON = () => {
    const jsonStr = JSON.stringify(annotation, null, 2);
    downloadFile(jsonStr, `${baseName}_annotation.json`, 'application/json');
  };

  const handleDownloadTranscript = () => {
    downloadFile(transcription?.text || '', `${baseName}_transcript.txt`, 'text/plain');
  };

  return (
    <div className="w-full bg-paper border border-ink shadow-[4px_4px_0_0_#1E2530] p-6 sm:p-8 flex flex-col gap-8 mt-8">
      
      {/* Header and Controls */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 border-b-2 border-ink pb-4">
        <div>
          <h2 className="font-mono text-xl font-bold uppercase tracking-tight text-ink">Annotation Viewer</h2>
          <p className="font-mono text-xs text-ink-light mt-1">Generated from backend pipeline</p>
        </div>
        <div className="flex gap-4">
          <button
            onClick={handleDownloadTranscript}
            className="px-4 py-2 bg-paper text-ink border-2 border-ink shadow-[2px_2px_0_0_#1E2530] hover:translate-x-[1px] hover:translate-y-[1px] hover:shadow-[1px_1px_0_0_#1E2530] transition-all font-mono text-xs font-bold uppercase tracking-wider"
          >
            Download TXT
          </button>
          <button
            onClick={handleDownloadJSON}
            className="px-4 py-2 bg-ink text-paper shadow-[2px_2px_0_0_#F4C430] hover:translate-x-[1px] hover:translate-y-[1px] hover:shadow-[1px_1px_0_0_#F4C430] transition-all font-mono text-xs font-bold uppercase tracking-wider"
          >
            Download JSON
          </button>
        </div>
      </div>

      {/* Transcription Section */}
      <section>
        <h3 className="font-mono text-sm font-bold uppercase tracking-widest text-ink mb-3 bg-highlight/20 inline-block px-2 py-1">Transcription</h3>
        <div className="bg-ink/5 p-4 border border-ink/10">
          <p className="font-serif text-lg leading-relaxed text-ink">{transcription?.text}</p>
          {transcription?.language && (
            <p className="font-mono text-xs text-ink-light mt-2 uppercase">Lang: {transcription.language}</p>
          )}
        </div>
      </section>

      {/* Words Section */}
      <section>
        <h3 className="font-mono text-sm font-bold uppercase tracking-widest text-ink mb-3 bg-highlight/20 inline-block px-2 py-1">Words</h3>
        <div className="overflow-x-auto border border-ink">
          <table className="w-full text-left border-collapse">
            <thead className="bg-ink text-paper font-mono text-xs uppercase tracking-wider">
              <tr>
                <th className="p-3 border-r border-ink/20">ID</th>
                <th className="p-3 border-r border-ink/20">Word</th>
                <th className="p-3 border-r border-ink/20">Start</th>
                <th className="p-3 border-r border-ink/20">End</th>
                <th className="p-3">Stress</th>
              </tr>
            </thead>
            <tbody className="font-mono text-sm">
              {words?.map((w) => (
                <tr key={w.id} className="border-b border-ink/10 hover:bg-ink/5">
                  <td className="p-3 border-r border-ink/10 text-ink-light">{w.id}</td>
                  <td className="p-3 border-r border-ink/10 font-bold">{w.text}</td>
                  <td className="p-3 border-r border-ink/10">{w.start.toFixed(2)}</td>
                  <td className="p-3 border-r border-ink/10">{w.end.toFixed(2)}</td>
                  <td className="p-3">
                    <span className={w.stress === 'stressed' ? 'text-highlight font-bold px-1 bg-ink' : 'text-ink-light'}>
                      {w.stress}
                    </span>
                  </td>
                </tr>
              ))}
              {(!words || words.length === 0) && (
                <tr><td colSpan="5" className="p-4 text-center text-ink-light">No words found.</td></tr>
              )}
            </tbody>
          </table>
        </div>
      </section>

      {/* Pauses Section */}
      <section>
        <h3 className="font-mono text-sm font-bold uppercase tracking-widest text-ink mb-3 bg-highlight/20 inline-block px-2 py-1">Pauses</h3>
        <div className="overflow-x-auto border border-ink">
          <table className="w-full text-left border-collapse">
            <thead className="bg-ink text-paper font-mono text-xs uppercase tracking-wider">
              <tr>
                <th className="p-3 border-r border-ink/20">Start</th>
                <th className="p-3 border-r border-ink/20">End</th>
                <th className="p-3">Duration</th>
              </tr>
            </thead>
            <tbody className="font-mono text-sm">
              {pauses?.map((p, idx) => (
                <tr key={idx} className="border-b border-ink/10 hover:bg-ink/5">
                  <td className="p-3 border-r border-ink/10">{p.start.toFixed(2)}</td>
                  <td className="p-3 border-r border-ink/10">{p.end.toFixed(2)}</td>
                  <td className="p-3 font-bold text-ink-light">{p.duration.toFixed(2)} s</td>
                </tr>
              ))}
              {(!pauses || pauses.length === 0) && (
                <tr><td colSpan="3" className="p-4 text-center text-ink-light">No pauses found.</td></tr>
              )}
            </tbody>
          </table>
        </div>
      </section>

      {/* Intonation Section */}
      <section>
        <div className="flex items-center gap-4 mb-3">
          <h3 className="font-mono text-sm font-bold uppercase tracking-widest text-ink bg-highlight/20 inline-block px-2 py-1 m-0">Intonation</h3>
          <span className="font-mono text-[10px] uppercase font-bold text-paper bg-ink px-2 py-0.5 rounded-sm">
            Stylized F0 → Overall Slope
          </span>
        </div>
        <div className="overflow-x-auto border border-ink">
          <table className="w-full text-left border-collapse">
            <thead className="bg-ink text-paper font-mono text-xs uppercase tracking-wider">
              <tr>
                <th className="p-3 border-r border-ink/20">Phrase ID</th>
                <th className="p-3 border-r border-ink/20">Start</th>
                <th className="p-3 border-r border-ink/20">End</th>
                <th className="p-3 border-r border-ink/20">Overall Slope</th>
                <th className="p-3">Label</th>
              </tr>
            </thead>
            <tbody className="font-mono text-sm">
              {intonation?.map((p) => (
                <tr key={p.phrase_id} className="border-b border-ink/10 hover:bg-ink/5">
                  <td className="p-3 border-r border-ink/10 text-ink-light">{p.phrase_id}</td>
                  <td className="p-3 border-r border-ink/10">{p.start.toFixed(2)}</td>
                  <td className="p-3 border-r border-ink/10">{p.end.toFixed(2)}</td>
                  <td className="p-3 border-r border-ink/10 font-bold">
                    {p.slope !== null ? `${p.slope.toFixed(3)} st/s` : 'N/A'}
                  </td>
                  <td className="p-3">
                    <span className="capitalize font-bold border-b-2 border-ink/20">{p.label}</span>
                  </td>
                </tr>
              ))}
              {(!intonation || intonation.length === 0) && (
                <tr><td colSpan="5" className="p-4 text-center text-ink-light">No intonation phrases found.</td></tr>
              )}
            </tbody>
          </table>
        </div>
      </section>

      {/* Timeline Section */}
      <section>
        <h3 className="font-mono text-sm font-bold uppercase tracking-widest text-ink mb-3 bg-highlight/20 inline-block px-2 py-1">Chronological Timeline</h3>
        <div className="bg-paper border border-ink max-h-96 overflow-y-auto">
          <div className="flex flex-col">
            {timeline?.map((evt, idx) => (
              <div key={idx} className="flex items-center gap-4 p-3 border-b border-ink/10 hover:bg-ink/5">
                <span className="font-mono text-sm w-16 text-right text-ink-light shrink-0">
                  {evt.start.toFixed(2)} s
                </span>
                <span className={`font-mono text-[10px] uppercase font-bold tracking-wider px-2 py-0.5 w-16 text-center rounded-sm ${
                  evt.type === 'word' ? 'bg-highlight text-ink' : 'bg-ink text-paper'
                }`}>
                  {evt.type}
                </span>
                <span className="font-mono text-sm font-bold truncate">
                  {evt.type === 'word' ? evt.text : `${evt.duration.toFixed(2)} s`}
                </span>
              </div>
            ))}
            {(!timeline || timeline.length === 0) && (
              <div className="p-4 text-center text-ink-light font-mono text-sm">No timeline events found.</div>
            )}
          </div>
        </div>
      </section>

    </div>
  );
}

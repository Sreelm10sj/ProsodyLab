/**
 * PitchContour.jsx
 * ================
 * Interactive two-canvas pitch contour visualization, optimized for research.
 */

import React, { useRef, useEffect, useCallback, useMemo, useState } from 'react';

// ── Geometry ─────────────────────────────────────────────────────────────────
const ML  = 40;   // left margin inside canvas for padding
const MR  = 20;   // right margin
const MT  = 12;   // top margin
const MB  = 24;   // bottom margin — time labels
const SEEK_STEP = 0.5; // seconds per arrow-key press

// ── Color palette ─────────────────────────────────────────────────────────────
const PHRASE_COLORS = {
  rising:               '#059669',
  falling:              '#3b82f6',
  level:                '#94a3b8',
  insufficient_voicing: '#9ca3af',
};

const PHRASE_BG = {
  rising:               'rgba(5,150,105,0.07)',
  falling:              'rgba(59,130,246,0.07)',
  level:                'rgba(148,163,184,0.06)',
  insufficient_voicing: 'rgba(156,163,175,0.03)',
};

const PATTERN_LABEL = {
  rising:               '↗ Rising',
  falling:              '↘ Falling',
  level:                '→ Level',
  insufficient_voicing: '~ No pitch',
};

// ── Pure drawing helpers (Modular Rendering Pipeline) ─────────────────────────

function getYRange(contour, stylized = []) {
  const vals = [];
  if (contour) vals.push(...contour.filter(p => p.semitone !== null).map(p => p.semitone));
  if (stylized) vals.push(...stylized.filter(p => p.semitone !== null).map(p => p.semitone));
  if (!vals.length) return { yMin: -6, yMax: 6 };
  return {
    yMin: Math.min(Math.min(...vals) - 1.5, -5),
    yMax: Math.max(Math.max(...vals) + 1.5,  5),
  };
}

function resizeCanvas(canvas, w, h) {
  const dpr = window.devicePixelRatio || 1;
  canvas.style.width  = w + 'px';
  canvas.style.height = h + 'px';
  canvas.width  = Math.floor(w * dpr);
  canvas.height = Math.floor(h * dpr);
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  return { ctx, w, h };
}

function drawBackground(ctx, w, h) {
  ctx.fillStyle = '#F7F8FA';
  ctx.fillRect(0, 0, w, h);
  ctx.fillStyle = '#ECEFF4';
  ctx.fillRect(ML, MT, w - ML - MR, h - MT - MB);
}

function drawGrid(ctx, w, h, yRange, toY) {
  const { yMin, yMax } = yRange;
  const stRange  = yMax - yMin;
  const gridStep = stRange <= 10 ? 2 : stRange <= 20 ? 4 : 6;
  const firstG   = Math.ceil(yMin / gridStep) * gridStep;

  for (let s = firstG; s <= yMax + 0.01; s += gridStep) {
    const y = toY(s);
    if (y < MT - 1 || y > h - MB + 1) continue;

    const isZero = Math.abs(s) < 0.01;
    ctx.strokeStyle = isZero ? 'rgba(30,37,48,0.22)' : 'rgba(30,37,48,0.07)';
    ctx.lineWidth   = isZero ? 1 : 0.5;
    ctx.setLineDash(isZero ? [5, 4] : []);
    ctx.beginPath();
    ctx.moveTo(ML, y);
    ctx.lineTo(w - MR, y);
    ctx.stroke();
  }
  ctx.setLineDash([]);
}

function drawPhraseBands(ctx, w, h, phrases, toX) {
  for (const phrase of phrases) {
    const x1 = Math.max(toX(phrase.start), ML);
    const x2 = Math.min(toX(phrase.end), w - MR);
    if (x2 <= x1) continue;
    ctx.fillStyle = PHRASE_BG[phrase.pitch_stylized?.pattern] || PHRASE_BG.insufficient_voicing;
    ctx.fillRect(x1, MT, x2 - x1, h - MT - MB);
  }
}

function drawRawContour(ctx, contour, toX, toY) {
  if (!contour || !contour.length) return;
  
  ctx.strokeStyle = 'rgba(30,37,48,0.30)'; // De-emphasized light gray
  ctx.lineWidth   = 1;
  ctx.lineJoin    = 'round';
  ctx.lineCap     = 'round';
  
  let started = false;
  for (const { time, semitone } of contour) {
    if (semitone === null) {
      if (started) { ctx.stroke(); started = false; }
      continue;
    }
    const x = toX(time);
    const y = toY(semitone);
    if (!started) {
      ctx.beginPath();
      ctx.moveTo(x, y);
      started = true;
    } else {
      ctx.lineTo(x, y);
    }
  }
  if (started) ctx.stroke();
}

function drawStylizedContour(ctx, stylized, phrases, toX, toY) {
  if (!stylized || !stylized.length) return;
  const sorted = [...phrases].sort((a, b) => a.start - b.start);

  const colorAt = (t) => {
    for (const p of sorted) {
      if (t >= p.start - 0.005 && t <= p.end + 0.005) {
        return PHRASE_COLORS[p.pitch_stylized?.pattern] || '#94a3b8';
      }
    }
    return '#94a3b8';
  };

  let started = false;
  let prevColor = null;
  let prevX = 0, prevY = 0;

  for (const { time, semitone } of stylized) {
    if (semitone === null) {
      if (started) { ctx.stroke(); started = false; prevColor = null; }
      continue;
    }
    const color = colorAt(time);
    const x     = toX(time);
    const y     = toY(semitone);

    if (!started) {
      ctx.beginPath();
      ctx.moveTo(x, y);
      ctx.strokeStyle = color;
      ctx.lineWidth   = 3; // Emphasized thick line
      ctx.lineJoin    = 'round';
      ctx.lineCap     = 'round';
      started    = true;
      prevColor  = color;
    } else if (color !== prevColor) {
      ctx.lineTo(x, y);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(prevX, prevY);
      ctx.lineTo(x, y);
      ctx.strokeStyle = color;
      ctx.lineWidth   = 3;
      ctx.lineJoin    = 'round';
      ctx.lineCap     = 'round';
      prevColor = color;
    } else {
      ctx.lineTo(x, y);
    }
    prevX = x;
    prevY = y;
  }
  if (started) ctx.stroke();
}

function drawPhraseBoundaries(ctx, w, h, words, toX) {
  for (const word of words) {
    const x = toX(word.start);
    if (x < ML - 0.5 || x > w - MR + 0.5) continue;

    ctx.strokeStyle = 'rgba(30,37,48,0.05)';
    ctx.lineWidth   = 0.5;
    ctx.beginPath();
    ctx.moveTo(x, MT);
    ctx.lineTo(x, h - MB);
    ctx.stroke();

    if (word.stressed) {
      const ds = 3.5;
      ctx.fillStyle = 'rgba(244,196,48,0.88)';
      ctx.beginPath();
      ctx.moveTo(x,      MT + 2);
      ctx.lineTo(x + ds, MT + 2 + ds);
      ctx.lineTo(x,      MT + 2 + ds * 2);
      ctx.lineTo(x - ds, MT + 2 + ds);
      ctx.closePath();
      ctx.fill();
    }
  }
}

// FUTURE EXTENSIBILITY: function drawDPSegmentBoundaries() {}

function drawPlotBorder(ctx, w, h) {
  ctx.strokeStyle = 'rgba(30,37,48,0.15)';
  ctx.lineWidth   = 1;
  ctx.strokeRect(ML, MT, w - ML - MR, h - MT - MB);
}

function drawTimeAxis(ctx, w, h, duration, toX, pixelsPerSec) {
  let tickStep = 1; 
  if (pixelsPerSec >= 400) tickStep = 0.25;
  else if (pixelsPerSec >= 200) tickStep = 0.5;
  else if (pixelsPerSec >= 100) tickStep = 1;
  else tickStep = 2;

  ctx.font         = '10px "IBM Plex Mono", ui-monospace, monospace';
  ctx.fillStyle    = '#4A5568';
  ctx.textAlign    = 'center';
  ctx.textBaseline = 'top';
  ctx.strokeStyle  = 'rgba(30,37,48,0.20)';
  ctx.lineWidth    = 1;

  for (let t = 0; t <= duration + 0.01; t += tickStep) {
    const x = toX(t);
    if (x < ML - 1 || x > w - MR + 1) continue;
    ctx.beginPath();
    ctx.moveTo(x, h - MB);
    ctx.lineTo(x, h - MB + 4);
    ctx.stroke();
    let label = t % 1 === 0 ? `${t} s` : `${t.toFixed(2)} s`;
    ctx.fillText(label, x, h - MB + 6);
  }
}

// ── Component ─────────────────────────────────────────────────────────────────

export default function PitchContour({
  words,
  phrases,
  pitchContour,
  pitchStylized,
  viewMode = 'both',
  pixelsPerSec = 100,
  height = 500, // configurable height
  utteranceMedianHz,
  currentTime,
  duration,
  onSeek,
}) {
  const containerRef  = useRef(null);
  const scrollRef     = useRef(null);
  const staticRef     = useRef(null);
  const dynamicRef    = useRef(null);
  const tooltipRef    = useRef(null);

  const currentTimeRef = useRef(currentTime);
  const durationRef    = useRef(duration);
  const prevZoomRef    = useRef(pixelsPerSec);

  useEffect(() => { currentTimeRef.current = currentTime; }, [currentTime]);
  useEffect(() => { durationRef.current    = duration;    }, [duration]);

  // Handle Zoom Scroll Preservation
  useEffect(() => {
    if (scrollRef.current && prevZoomRef.current !== pixelsPerSec && durationRef.current > 0) {
      const scrollEl = scrollRef.current;
      const centerT = (scrollEl.scrollLeft + scrollEl.clientWidth / 2 - ML) / prevZoomRef.current;
      const newScrollLeft = (centerT * pixelsPerSec) + ML - scrollEl.clientWidth / 2;
      scrollEl.scrollLeft = Math.max(0, newScrollLeft);
      prevZoomRef.current = pixelsPerSec;
    }
  }, [pixelsPerSec]);

  // Memoised properties
  const yRange = useMemo(() => getYRange(pitchContour, pitchStylized), [pitchContour, pitchStylized]);
  const yRangeRef = useRef(yRange);
  useEffect(() => { yRangeRef.current = yRange; }, [yRange]);

  const wordsSorted   = useMemo(() => [...words].sort((a, b) => a.start - b.start), [words]);
  const phrasesSorted = useMemo(() => [...phrases].sort((a, b) => a.start - b.start), [phrases]);

  // Calculate rendering dimensions
  const [containerWidth, setContainerWidth] = useState(0);
  
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const obs = new ResizeObserver((entries) => {
      setContainerWidth(entries[0].contentRect.width);
    });
    obs.observe(el);
    return () => obs.disconnect();
  }, []);

  // Logical dimensions
  const logicalH = Math.max(height, 450);
  const logicalW = Math.max(containerWidth, Math.ceil((duration * pixelsPerSec) + ML + MR));

  // ── Static layer ───────────────────────────────────────────────────────────
  const drawStatic = useCallback(() => {
    const canvas = staticRef.current;
    if (!canvas || logicalW === 0 || duration === 0) return;

    const { ctx, w, h } = resizeCanvas(canvas, logicalW, logicalH);
    const { yMin, yMax } = yRange;
    const toX = (t) => ML + t * pixelsPerSec;
    const toY = (s) => MT + (1 - (s - yMin) / (yMax - yMin)) * (h - MT - MB);

    // Modular Render Pipeline
    drawBackground(ctx, w, h);
    drawGrid(ctx, w, h, yRange, toY);
    drawPhraseBands(ctx, w, h, phrases, toX);
    
    if (viewMode === 'both' || viewMode === 'raw') {
      drawRawContour(ctx, pitchContour, toX, toY);
    }
    if (viewMode === 'both' || viewMode === 'stylized') {
      drawStylizedContour(ctx, pitchStylized, phrases, toX, toY);
    }
    
    drawPhraseBoundaries(ctx, w, h, words, toX);
    // drawDPSegmentBoundaries(...)
    drawPlotBorder(ctx, w, h);
    drawTimeAxis(ctx, w, h, duration, toX, pixelsPerSec);

  }, [logicalW, logicalH, duration, pixelsPerSec, pitchContour, pitchStylized, phrases, words, viewMode, yRange]);

  useEffect(() => { drawStatic(); }, [drawStatic]);

  // ── Dynamic layer: playhead via RAF ───────────────────────────────────────
  useEffect(() => {
    let id;
    const frame = () => {
      const canvas = dynamicRef.current;
      if (!canvas || logicalW === 0) { id = requestAnimationFrame(frame); return; }

      const dpr = window.devicePixelRatio || 1;
      const tw = Math.floor(logicalW * dpr);
      const th = Math.floor(logicalH * dpr);

      if (canvas.width !== tw || canvas.height !== th) {
        canvas.style.width  = logicalW + 'px';
        canvas.style.height = logicalH + 'px';
        canvas.width  = tw;
        canvas.height = th;
      }

      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.save();
      ctx.scale(dpr, dpr);

      const ct  = currentTimeRef.current;
      const dur = durationRef.current;

      if (dur > 0 && ct >= 0) {
        const x = ML + ct * pixelsPerSec;
        if (x >= ML && x <= logicalW - MR) {
            ctx.strokeStyle = 'rgba(30,37,48,0.75)';
            ctx.lineWidth   = 1.5;
            ctx.beginPath();
            ctx.moveTo(x, MT);
            ctx.lineTo(x, logicalH - MB);
            ctx.stroke();

            ctx.fillStyle = 'rgba(30,37,48,0.80)';
            ctx.beginPath();
            ctx.moveTo(x - 4, MT);
            ctx.lineTo(x + 4, MT);
            ctx.lineTo(x,     MT + 8);
            ctx.closePath();
            ctx.fill();
        }
      }

      ctx.restore();
      id = requestAnimationFrame(frame);
    };
    id = requestAnimationFrame(frame);
    return () => cancelAnimationFrame(id);
  }, [logicalW, logicalH, pixelsPerSec]);

  // ── Interaction handlers ───────────────────────────────────────────────────
  const handleClick = useCallback((e) => {
    const scrollEl = scrollRef.current;
    if (!scrollEl || !durationRef.current) return;
    const rect = scrollEl.getBoundingClientRect();
    const lx = e.clientX - rect.left + scrollEl.scrollLeft;
    
    if (lx < ML || lx > logicalW - MR) return;
    const t = Math.max(0, Math.min(durationRef.current, (lx - ML) / pixelsPerSec));
    onSeek(t);
  }, [logicalW, pixelsPerSec, onSeek]);

  const handleKeyDown = useCallback((e) => {
    if (e.key === 'ArrowLeft') {
      e.preventDefault();
      onSeek(Math.max(0, currentTimeRef.current - SEEK_STEP));
    } else if (e.key === 'ArrowRight') {
      e.preventDefault();
      onSeek(Math.min(durationRef.current, currentTimeRef.current + SEEK_STEP));
    }
  }, [onSeek]);

  const handleMouseMove = useCallback((e) => {
    const scrollEl = scrollRef.current;
    const tooltip = tooltipRef.current;
    if (!scrollEl || !tooltip || !durationRef.current) return;

    const rect = scrollEl.getBoundingClientRect();
    const lx = e.clientX - rect.left + scrollEl.scrollLeft;
    const ly = e.clientY - rect.top;

    if (lx < ML || lx > logicalW - MR) {
      tooltip.style.display = 'none';
      return;
    }

    const t = (lx - ML) / pixelsPerSec;

    // Lookup index
    let idx = 0;
    if (pitchContour.length > 1) {
      const t0 = pitchContour[0].time;
      const tN = pitchContour[pitchContour.length - 1].time;
      const stp = (tN - t0) / (pitchContour.length - 1);
      idx = Math.min(Math.max(0, Math.round((t - t0) / stp)), pitchContour.length - 1);
    }

    const rawP = pitchContour[idx];
    const rawSt = rawP?.semitone ?? null;
    const rawHz = (rawSt !== null && utteranceMedianHz) ? (utteranceMedianHz * Math.pow(2, rawSt / 12)).toFixed(0) : null;
    
    const styP = (pitchStylized && pitchStylized.length) ? pitchStylized[idx] : null;
    const stySt = styP?.semitone ?? null;
    
    let nearWord = null;
    let minD = Infinity;
    for (const w of wordsSorted) {
      if (t >= w.start && t <= w.end) { nearWord = w; break; }
      const d = Math.min(Math.abs(t - w.start), Math.abs(t - w.end));
      if (d < minD) { minD = d; nearWord = w; }
    }

    let nearPhrase = null;
    for (const p of phrasesSorted) {
      if (t >= p.start && t <= p.end) { nearPhrase = p; break; }
    }
    
    // Choose what pitch value to show. Since stylized is prominent, we prefer it if available.
    const displaySt = stySt !== null ? stySt : rawSt;
    const displayStr = displaySt !== null ? `${displaySt >= 0 ? '+' : ''}${displaySt.toFixed(2)} st` : '— (unvoiced)';
    const displayHz = (displaySt !== null && utteranceMedianHz) ? (utteranceMedianHz * Math.pow(2, displaySt / 12)).toFixed(0) : null;

    const metaParts = [
      nearWord ? `"${nearWord.word.trim()}"` : null,
      nearPhrase ? PATTERN_LABEL[nearPhrase.pitch_stylized?.pattern] : null,
    ].filter(Boolean).join(' · ');
    
    const pVal = styP?.P;
    const kVal = styP?.K;

    tooltip.innerHTML = `
      <div class="pitch-tooltip-time">${t.toFixed(2)} s</div>
      <div class="pitch-tooltip-st flex justify-between gap-4">
        <span style="opacity: 0.7">Pitch:</span> <span>${displayStr}${displayHz ? ` <span class="pitch-tooltip-hz">(${displayHz} Hz)</span>` : ''}</span>
      </div>
      ${pVal !== undefined && pVal !== null ? `
      <div class="pitch-tooltip-st flex justify-between gap-4">
        <span style="opacity: 0.7">P:</span> <span>${pVal}</span>
      </div>` : ''}
      ${kVal !== undefined && kVal !== null ? `
      <div class="pitch-tooltip-st flex justify-between gap-4">
        <span style="opacity: 0.7">K:</span> <span>${kVal}</span>
      </div>` : ''}
      ${metaParts ? `<div class="pitch-tooltip-meta">${metaParts}</div>` : ''}
    `;
    tooltip.style.display = 'block';

    const flipBelow = ly < 70;
    // Position tooltip globally because it has position: fixed
    tooltip.style.left      = `${e.clientX}px`;
    tooltip.style.top       = flipBelow ? `${e.clientY + 12}px` : `${e.clientY}px`;
    tooltip.style.transform = flipBelow
      ? 'translate(-50%, 0)'
      : 'translate(-50%, calc(-100% - 8px))';
  }, [pitchContour, pitchStylized, logicalW, pixelsPerSec, utteranceMedianHz, wordsSorted, phrasesSorted]);

  const handleMouseLeave = useCallback(() => {
    if (tooltipRef.current) tooltipRef.current.style.display = 'none';
  }, []);

  // Compute fixed Y-axis ticks for the DOM overlay
  const yTicks = useMemo(() => {
    const { yMin, yMax } = yRange;
    const stRange = yMax - yMin;
    const gridStep = stRange <= 10 ? 2 : stRange <= 20 ? 4 : 6;
    const firstG = Math.ceil(yMin / gridStep) * gridStep;
    const ticks = [];
    for (let s = firstG; s <= yMax + 0.01; s += gridStep) {
        ticks.push(s);
    }
    return ticks;
  }, [yRange]);

  if (!pitchContour.length) return null;

  return (
    <div ref={containerRef} className="pitch-contour-container relative w-full border border-ink/10 bg-paper" style={{ height: logicalH }}>
      
      {/* Fixed Y-axis Overlay */}
      <div className="absolute left-0 top-0 bottom-0 w-[40px] bg-paper/90 border-r border-ink/10 z-10 pointer-events-none flex flex-col">
        <span className="absolute text-[9px] font-mono text-ink-lighter" style={{ top: MT - 4, left: 4 }}>st</span>
        {yTicks.map(s => {
          const y = MT + (1 - (s - yRange.yMin) / (yRange.yMax - yRange.yMin)) * (logicalH - MT - MB);
          if (y < MT - 1 || y > logicalH - MB + 1) return null;
          return (
            <span key={s} className="absolute text-[10px] font-mono text-ink text-right w-[30px]" style={{ top: y, left: 4, transform: 'translateY(-50%)' }}>
              {s > 0 ? '+' : ''}{s}
            </span>
          );
        })}
      </div>

      {/* Scrollable Canvas Pane */}
      <div 
        ref={scrollRef}
        className="overflow-x-auto overflow-y-hidden w-full h-full relative"
        tabIndex={0}
        onClick={handleClick}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
        onKeyDown={handleKeyDown}
      >
        <canvas
          ref={staticRef}
          aria-hidden="true"
          style={{ position: 'absolute', top: 0, left: 0, pointerEvents: 'none' }}
        />
        <canvas
          ref={dynamicRef}
          aria-hidden="true"
          style={{ position: 'absolute', top: 0, left: 0, pointerEvents: 'none' }}
        />
        
        {/* Hover tooltip — positioned visually relative to the wrapper */}
        <div ref={tooltipRef} className="pitch-tooltip absolute z-20" style={{ position: 'fixed' }} />
      </div>
    </div>
  );
}

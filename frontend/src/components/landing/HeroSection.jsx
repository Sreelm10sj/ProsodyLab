import { useEffect, useRef } from 'react'
import { gsap } from 'gsap'
import { ScrollTrigger } from 'gsap/ScrollTrigger'
import PitchField from './PitchField.jsx'

gsap.registerPlugin(ScrollTrigger)

export default function HeroSection() {
  const sectionRef = useRef(null)
  const h1Ref = useRef(null)
  const bodyRef = useRef(null)
  const ctaRef = useRef(null)

  useEffect(() => {
    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches

    const h1 = h1Ref.current
    const body = bodyRef.current
    const cta = ctaRef.current
    if (!h1 || !body || !cta) return

    // ── Intro animation ──────────────────────────────────────────────────
    // Both paths run a timeline (no early return that would block ScrollTrigger).
    let tl
    if (reduced) {
      gsap.set([h1, body, cta], { opacity: 0 })
      tl = gsap.timeline({ delay: 0.05 })
      tl.to(h1, { opacity: 1, duration: 0.30, ease: 'power2.out' })
        .to(body, { opacity: 1, duration: 0.25, ease: 'power2.out' }, '-=0.05')
        .to(cta, { opacity: 1, duration: 0.20, ease: 'power2.out' }, '-=0.05')
      // NOTE: no early return here — fall through to ScrollTrigger setup below
    } else {
      gsap.set([h1, body, cta], { opacity: 0, y: 16 })
      tl = gsap.timeline({ delay: 0.08 })
      tl.to(h1, { opacity: 1, y: 0, duration: 0.80, ease: 'power3.out' })
        .to(body, { opacity: 1, y: 0, duration: 0.60, ease: 'power3.out' }, '-=0.4')
        .to(cta, { opacity: 1, y: 0, duration: 0.50, ease: 'power3.out' }, '-=0.35')
    }

    // ── Scroll-driven fade — always runs (no reduced-motion gate) ────────
    // Body copy: fades in as it enters viewport, fades out as it leaves upward.
    // Scrub:1 = one-to-one with scroll speed.
    const stBody = ScrollTrigger.create({
      trigger: body,
      start: 'top 80%',
      end: 'bottom top',
      scrub: 1,
      onUpdate: (self) => {
        const p = self.progress
        // 0→0.15: fade in. 0.15→0.85: fully visible. 0.85→1: fade out.
        let alpha
        if (p < 0.15) alpha = p / 0.15
        else if (p < 0.85) alpha = 1
        else alpha = 1 - (p - 0.85) / 0.15
        gsap.set(body, { opacity: Math.max(0, Math.min(1, alpha)) })
      },
    })

    const stCta = ScrollTrigger.create({
      trigger: cta,
      start: 'top 85%',
      end: 'bottom top',
      scrub: 1,
      onUpdate: (self) => {
        const p = self.progress
        let alpha
        if (p < 0.2) alpha = p / 0.2
        else if (p < 0.8) alpha = 1
        else alpha = 1 - (p - 0.8) / 0.2
        gsap.set(cta, { opacity: Math.max(0, Math.min(1, alpha)) })
      },
    })

    return () => {
      tl?.kill()
      stBody.kill()
      stCta.kill()
    }
  }, [])

  return (
    <section
      ref={sectionRef}
      className="relative min-h-screen flex flex-col items-center justify-center overflow-hidden"
      style={{ backgroundColor: '#10141C' }}
      aria-label="ProsodyLab hero"
    >
      {/* Pitch field canvas */}
      <div className="absolute inset-0" aria-hidden="true">
        <PitchField sectionRef={sectionRef} />
      </div>

      {/* Content — centered */}
      <div className="relative z-10 w-full flex flex-col items-center text-center px-6">

        <h1
          ref={h1Ref}
          className="font-display font-extrabold leading-[1.0] mb-8"
          style={{ fontSize: 'clamp(56px, 10vw, 128px)', color: '#F5F4F0', letterSpacing: '-0.025em' }}
        >
          <span className="block">Prosody</span>
          <span className="inline-flex items-baseline gap-3 flex-wrap justify-center">
            <span>Lab</span>
            <span
              className="font-mono font-bold"
              style={{ color: '#2F9E6F', fontSize: 'clamp(28px, 3.5vw, 60px)' }}
              aria-hidden="true"
            >
              ↗
            </span>
          </span>
        </h1>

        <p
          ref={bodyRef}
          className="font-sans leading-relaxed mb-14 max-w-2xl px-4 text-center"
        >
          <span className="block text-[#F5F4F0] font-normal mb-2 text-base sm:text-lg tracking-tight">
            A research workspace for speech prosody.
          </span>
          <span
            className="block text-sm sm:text-base leading-relaxed font-light"
            style={{ color: 'rgba(245, 244, 240, 0.7)' }}
          >
            Analyze spoken language, explore lexical stress and intonation, generate structured annotations, and train LexiRep models on your own data.
          </span>
        </p>

        {/* CTA two-panel mirrored split */}
        <div
          ref={ctaRef}
          className="w-full flex items-stretch"
          style={{ maxWidth: 760, borderTop: '1px solid rgba(168,176,190,0.2)' }}
          role="group"
          aria-label="Entry points"
        >
          <a
            href="/analyze"
            id="cta-analyze"
            className="cta-panel cta-panel--primary"
            onClick={(e) => {
              e.preventDefault()
              window.dispatchEvent(new CustomEvent('navigate', { detail: { to: '/analyze' } }))
            }}
            aria-label="Go to Analyze Speech"
          >
            <span className="cta-panel__label">Analyze</span>
            <span className="cta-panel__sublabel">Record or upload speech →</span>
          </a>

          <div style={{ width: 1, background: 'rgba(168,176,190,0.25)', flexShrink: 0 }} aria-hidden="true" />

          <a
            href="/train"
            id="cta-train"
            className="cta-panel cta-panel--secondary"
            onClick={(e) => {
              e.preventDefault()
              window.dispatchEvent(new CustomEvent('navigate', { detail: { to: '/train' } }))
            }}
            aria-label="Go to Train Model"
          >
            <span className="cta-panel__label">Train Model</span>
            <span className="cta-panel__sublabel">Upload a dataset →</span>
          </a>
        </div>
      </div>
    </section>
  )
}
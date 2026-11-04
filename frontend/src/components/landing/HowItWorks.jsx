import { useEffect, useRef } from 'react'
import { gsap } from 'gsap'
import { ScrollTrigger } from 'gsap/ScrollTrigger'

gsap.registerPlugin(ScrollTrigger)

const STEPS = [
  {
    title: 'Analyze speech',
    body: 'Upload or record speech and explore transcription, lexical stress, pauses, syllables, and phrase-level intonation with synchronized audio playback.',
  },
  {
    title: 'Generate annotations',
    body: 'Turn speech into structured annotations covering words, syllables, pauses, and intonation, with results available for research workflows.',
  },
  {
    title: 'Train LexiRep',
    body: 'Upload a labeled feature dataset, discover acoustic stress anchors, and train a LexiRep model using the iterative self-training pipeline.',
  },
  {
    title: 'Export and reuse',
    body: 'Download trained model bundles and annotation results for use in further analysis, experiments, and research pipelines.',
  }
]

/**
 * HowItWorks
 *
 * Staggered scroll-driven scrub:
 * As the user scrolls into this section, the heading and each of the 4 columns
 * fade in and glide up smoothly, tracking the scrollbar directly (scrub: 0.8).
 *
 * All 4 columns are guaranteed to animate in and reach 100% completion
 * within 340px of scroll progress, well before the page bottom is reached.
 */
export default function HowItWorks() {
  const sectionRef = useRef(null)
  const headingRef = useRef(null)
  const itemRefs = useRef([])

  useEffect(() => {
    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    const section = sectionRef.current
    const heading = headingRef.current
    const cards = itemRefs.current.filter(Boolean)
    if (!section || cards.length === 0) return

    // Initial state: hidden slightly below resting position
    if (reduced) {
      gsap.set([heading, ...cards], { opacity: 1, y: 0 })
    } else {
      gsap.set(heading, { opacity: 0, y: 16 })
      gsap.set(cards, { opacity: 0, y: 28 })
    }

    const st = ScrollTrigger.create({
      trigger: section,
      start: 'top 88%',
      end: 'bottom 92%',
      scrub: 0.8,
      onUpdate: (self) => {
        const p = self.progress
        // Heading fades & rises first (completes by 30% of scroll)
        const pHead = Math.min(1, Math.max(0, p / 0.30))
        gsap.set(heading, {
          opacity: pHead,
          ...(reduced ? {} : { y: 16 * (1 - pHead) }),
        })

        // All 4 cards stagger into view and are 100% visible by p=0.74
        cards.forEach((card, i) => {
          const startP = 0.05 + i * 0.09
          const endP = startP + 0.42
          const pCard = Math.min(1, Math.max(0, (p - startP) / (endP - startP)))
          gsap.set(card, {
            opacity: pCard,
            ...(reduced ? {} : { y: 28 * (1 - pCard) }),
          })
        })
      },
      onLeaveBack: () => {
        if (!reduced) {
          gsap.set(heading, { opacity: 0, y: 16 })
          gsap.set(cards, { opacity: 0, y: 28 })
        }
      },
    })

    const refreshTimer = setTimeout(() => {
      ScrollTrigger.refresh()
    }, 350)

    return () => {
      clearTimeout(refreshTimer)
      st.kill()
    }
  }, [])

  return (
    <section
      ref={sectionRef}
      className="py-24 md:py-32 px-8 md:px-16 lg:px-24"
      style={{ background: '#F5F4F0' }}
      aria-labelledby="how-it-works-heading"
    >
      <div className="max-w-6xl mx-auto mb-16">
        <h2
          ref={headingRef}
          id="how-it-works-heading"
          className="font-mono font-bold text-2xl md:text-3xl uppercase tracking-wider text-[#1C2128]"
        >
          How it works
        </h2>
      </div>

      <div className="max-w-6xl mx-auto grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-12 lg:gap-8">
        {STEPS.map((step, i) => (
          <div
            key={step.title}
            ref={el => { itemRefs.current[i] = el }}
            className="flex flex-col"
          >
            <h3 className="font-mono font-semibold text-base uppercase tracking-wider mb-3 text-[#1C2128]">
              {step.title}
            </h3>
            <p className="font-sans text-sm md:text-[15px] leading-relaxed font-light text-[#4E5766]">
              {step.body}
            </p>
          </div>
        ))}
      </div>
    </section>
  )
}
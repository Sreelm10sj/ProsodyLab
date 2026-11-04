import { BrowserRouter, Routes, Route, useLocation, useNavigate } from 'react-router-dom'
import { useEffect, useRef } from 'react'
import { AnimatePresence, motion, MotionConfig } from 'motion/react'
import Landing from './pages/Landing.jsx'
import Analyze from './pages/Analyze.jsx'
import Train from './pages/Train.jsx'
import Navigation from './components/Navigation.jsx'
import { ThemeProvider } from './ThemeContext.jsx'

/**
 * Spatial model — three pages in one horizontal space:
 *
 *   /analyze (0)  |  / (1)  |  /train (2)
 *
 * Direction is derived from the navigation event, NOT the destination route.
 * Both exiting and entering variants are functions of `direction` (custom prop).
 *
 *   direction = +1 → moving RIGHT in space (toward higher index)
 *     Exit:  translateX(-100%)   — current page slides left
 *     Enter: translateX(+100%)   — next page arrives from right
 *
 *   direction = -1 → moving LEFT in space (toward lower index)
 *     Exit:  translateX(+100%)   — current page slides right
 *     Enter: translateX(-100%)   — next page arrives from left
 *
 * / → /analyze: direction = -1 (landing slides RIGHT, analyze comes from LEFT) ✓
 * / → /train:   direction = +1 (landing slides LEFT,  train comes from RIGHT) ✓
 * /analyze ↔ /train: header is stationary; page body moves horizontally ✓
 */

const ROUTE_ORDER = { '/analyze': 0, '/': 1, '/train': 2 }

// Variant functions receive the `custom` value (direction: +1 or -1)
const pageVariants = {
  initial: (dir) => ({
    transform: `translateX(${dir >= 0 ? '100%' : '-100%'})`,
  }),
  animate: {
    transform: 'translateX(0%)',
  },
  exit: (dir) => ({
    transform: `translateX(${dir >= 0 ? '-100%' : '100%'})`,
  }),
}

function WorkbenchLayout() {
  const location = useLocation()
  const prevWorkbenchPathRef = useRef(location.pathname)
  const dirRef = useRef(0)

  if (prevWorkbenchPathRef.current !== location.pathname) {
    const fromIdx = ROUTE_ORDER[prevWorkbenchPathRef.current] ?? 0
    const toIdx = ROUTE_ORDER[location.pathname] ?? 0
    dirRef.current = toIdx >= fromIdx ? 1 : -1
    prevWorkbenchPathRef.current = location.pathname
  }

  const direction = dirRef.current

  useEffect(() => {
    window.scrollTo(0, 0)
  }, [location.pathname])

  return (
    <div className="min-h-screen bg-[#F8FAFC] dark:bg-[#0B0F17] text-slate-900 dark:text-slate-100 flex flex-col font-sans">
      <Navigation />
      <div className="flex-1 flex flex-col relative" style={{ overflowX: 'hidden' }}>
        <AnimatePresence custom={direction} mode="wait" initial={false}>
          <motion.div
            key={location.pathname}
            custom={direction}
            variants={pageVariants}
            initial="initial"
            animate="animate"
            exit="exit"
            transition={{
              duration: 0.35,
              ease: [0.23, 1, 0.32, 1], // --ease-out: fast deceleration = feels physical
            }}
            style={{ flex: 1, display: 'flex', flexDirection: 'column', willChange: 'transform' }}
          >
            <Routes location={location}>
              <Route path="/analyze" element={<Analyze />} />
              <Route path="/train" element={<Train />} />
              <Route path="*" element={<Analyze />} />
            </Routes>
          </motion.div>
        </AnimatePresence>
      </div>
    </div>
  )
}

function AnimatedRoutes() {
  const location  = useLocation()
  const navigate  = useNavigate()

  // Track previous pathname to compute direction at render time.
  const prevPathnameRef = useRef(location.pathname)
  const dirRef = useRef(0)

  if (prevPathnameRef.current !== location.pathname) {
    const fromIdx = ROUTE_ORDER[prevPathnameRef.current] ?? 1
    const toIdx   = ROUTE_ORDER[location.pathname]       ?? 1
    // +1 = moving right (to higher index), -1 = moving left (to lower index)
    dirRef.current = toIdx >= fromIdx ? 1 : -1
    prevPathnameRef.current = location.pathname
  }

  const direction = dirRef.current

  // Listen for custom navigate events dispatched by CTA panels in HeroSection
  useEffect(() => {
    function handleNavigate(e) { navigate(e.detail.to) }
    window.addEventListener('navigate', handleNavigate)
    return () => window.removeEventListener('navigate', handleNavigate)
  }, [navigate])

  const isLanding = location.pathname === '/'
  const outerKey = isLanding ? 'landing' : 'workbench'

  return (
    <div style={{ overflowX: 'hidden', minHeight: '100dvh' }}>
      <AnimatePresence
        custom={direction}
        mode="wait"
        initial={false}
      >
        <motion.div
          key={outerKey}
          custom={direction}
          variants={pageVariants}
          initial="initial"
          animate="animate"
          exit="exit"
          transition={{
            duration: 0.35,
            ease: [0.23, 1, 0.32, 1],  // --ease-out: fast deceleration = feels physical
          }}
          style={{ minHeight: '100dvh', willChange: 'transform' }}
        >
          {isLanding ? (
            <Routes location={location}>
              <Route path="/" element={<Landing />} />
              <Route path="*" element={<Landing />} />
            </Routes>
          ) : (
            <WorkbenchLayout />
          )}
        </motion.div>
      </AnimatePresence>
    </div>
  )
}

export default function App() {
  return (
    /*
     * reducedMotion="never" — this app's route slides and scroll scrub
     * are feature interactions, not decorative. OS preference is still
     * respected for CSS @media (prefers-reduced-motion) blocks.
     */
    <ThemeProvider>
      <MotionConfig reducedMotion="never">
        <BrowserRouter>
          <AnimatedRoutes />
        </BrowserRouter>
      </MotionConfig>
    </ThemeProvider>
  )
}
import { Link, useLocation } from 'react-router-dom'
import { motion } from 'motion/react'
import { Sun, Moon, User } from 'lucide-react'
import Wordmark from './Wordmark.jsx'
import BackendStatus from './BackendStatus.jsx'
import { useTheme } from '../ThemeContext.jsx'

export default function Navigation() {
  const location = useLocation()
  const pathname = location.pathname
  const { isDark, toggleTheme } = useTheme()

  const isAnalyze = pathname === '/analyze'
  const isTrain = pathname === '/train'

  return (
    <header className="sticky top-0 z-50 bg-white/95 dark:bg-[#0B0F17]/95 backdrop-blur-sm border-b border-slate-200/80 dark:border-slate-800 transition-colors duration-200">
      <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between gap-6">
        
        {/* Left: Brand Logo */}
        <div className="flex items-center">
          <Link to="/" aria-label="ProsodyLab home" className="focus:outline-none focus-visible:ring-2 focus-visible:ring-amber-500 rounded-md">
            <Wordmark variant="refined" />
          </Link>
        </div>

        {/* Center: Navigation Links with Yellow Active Underline */}
        <nav className="flex items-center gap-8" aria-label="Main Navigation">
          <div className="relative py-2">
            <Link
              to="/analyze"
              className={`text-sm font-semibold tracking-tight transition-colors duration-150 ${
                isAnalyze
                  ? 'text-slate-900 dark:text-slate-100'
                  : 'text-slate-500 hover:text-slate-800 dark:text-slate-400 dark:hover:text-slate-200'
              }`}
            >
              Analyze
            </Link>
            {isAnalyze && (
              <motion.span
                layoutId="nav-active-underline"
                className="absolute bottom-0 left-0 right-0 h-[2.5px] rounded-full bg-[#FDB813]"
                style={{ backgroundColor: '#FDB813' }}
                transition={{ type: 'spring', stiffness: 450, damping: 35 }}
                aria-hidden="true"
              />
            )}
          </div>

          <div className="relative py-2">
            <Link
              to="/train"
              className={`text-sm font-semibold tracking-tight transition-colors duration-150 ${
                isTrain
                  ? 'text-slate-900 dark:text-slate-100'
                  : 'text-slate-500 hover:text-slate-800 dark:text-slate-400 dark:hover:text-slate-200'
              }`}
            >
              Train Model
            </Link>
            {isTrain && (
              <motion.span
                layoutId="nav-active-underline"
                className="absolute bottom-0 left-0 right-0 h-[2.5px] rounded-full bg-[#FDB813]"
                style={{ backgroundColor: '#FDB813' }}
                transition={{ type: 'spring', stiffness: 450, damping: 35 }}
                aria-hidden="true"
              />
            )}
          </div>
        </nav>

        {/* Right: Controls & Account */}
        <div className="flex items-center gap-3">
          <BackendStatus />

          {/* Theme Toggle Button */}
          <button
            id="btn-toggle-theme"
            type="button"
            onClick={toggleTheme}
            className="w-9 h-9 rounded-full flex items-center justify-center text-slate-500 hover:text-slate-800 dark:text-slate-400 dark:hover:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-800/80 transition-colors border border-transparent hover:border-slate-200 dark:hover:border-slate-700 focus:outline-none focus-visible:ring-2 focus-visible:ring-amber-500"
            title={isDark ? "Switch to light mode" : "Switch to dark mode"}
            aria-label={isDark ? "Switch to light mode" : "Switch to dark mode"}
          >
            {isDark ? (
              <Sun size={17} strokeWidth={2} className="text-amber-400" />
            ) : (
              <Moon size={17} strokeWidth={2} />
            )}
          </button>

          {/* Account Profile Avatar */}
          <button
            type="button"
            className="w-9 h-9 rounded-full flex items-center justify-center text-slate-600 dark:text-slate-300 bg-slate-100/90 dark:bg-slate-800 hover:bg-slate-200/80 dark:hover:bg-slate-700 transition-colors border border-slate-200/60 dark:border-slate-700 focus:outline-none focus-visible:ring-2 focus-visible:ring-amber-500"
            title="User profile"
            aria-label="User account"
          >
            <User size={17} strokeWidth={2} />
          </button>
        </div>

      </div>
    </header>
  )
}

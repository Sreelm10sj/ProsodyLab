import React, { createContext, useContext, useState, useEffect, useCallback } from 'react'

const THEME_KEY = 'prosodylab_theme'

export const ThemeContext = createContext(null)

export function ThemeProvider({ children }) {
  const [theme, setThemeState] = useState(() => {
    if (typeof window === 'undefined') return 'light'
    try {
      const stored = localStorage.getItem(THEME_KEY)
      if (stored === 'dark' || stored === 'light') return stored
    } catch (e) {
      // storage unavailable
    }
    return 'light'
  })

  useEffect(() => {
    if (typeof document === 'undefined') return
    const root = document.documentElement
    if (theme === 'dark') {
      root.classList.add('dark')
    } else {
      root.classList.remove('dark')
    }
    try {
      localStorage.setItem(THEME_KEY, theme)
    } catch (e) {
      // ignore
    }
  }, [theme])

  const toggleTheme = useCallback(() => {
    setThemeState(prev => (prev === 'dark' ? 'light' : 'dark'))
  }, [])

  const setTheme = useCallback((newTheme) => {
    if (newTheme === 'dark' || newTheme === 'light') {
      setThemeState(newTheme)
    }
  }, [])

  const value = {
    theme,
    isDark: theme === 'dark',
    toggleTheme,
    setTheme,
  }

  return (
    <ThemeContext.Provider value={value}>
      {children}
    </ThemeContext.Provider>
  )
}

export function useTheme() {
  const ctx = useContext(ThemeContext)
  // Graceful fallback if rendered outside ThemeProvider
  if (!ctx || !ctx.toggleTheme) {
    const isCurrentlyDark = typeof document !== 'undefined' && document.documentElement.classList.contains('dark')
    return {
      theme: isCurrentlyDark ? 'dark' : 'light',
      isDark: isCurrentlyDark,
      toggleTheme: () => {
        if (typeof document !== 'undefined') {
          const nextDark = !document.documentElement.classList.contains('dark')
          if (nextDark) {
            document.documentElement.classList.add('dark')
            try { localStorage.setItem(THEME_KEY, 'dark') } catch (e) {}
          } else {
            document.documentElement.classList.remove('dark')
            try { localStorage.setItem(THEME_KEY, 'light') } catch (e) {}
          }
        }
      },
      setTheme: (t) => {
        if (typeof document !== 'undefined') {
          if (t === 'dark') {
            document.documentElement.classList.add('dark')
            try { localStorage.setItem(THEME_KEY, 'dark') } catch (e) {}
          } else {
            document.documentElement.classList.remove('dark')
            try { localStorage.setItem(THEME_KEY, 'light') } catch (e) {}
          }
        }
      },
    }
  }
  return ctx
}

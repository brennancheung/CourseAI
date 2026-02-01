'use client'

import { ReactNode, useCallback, useEffect, useMemo, useState } from 'react'
import type { Mode, NoteName } from '@/lib/music-theory'
import { ScaleKeyContext } from './ScaleKeyContext'
import { ScaleModeContext } from './ScaleModeContext'
import { ScaleOptionsContext, type ScaleOptions } from './ScaleOptionsContext'
import { ScaleActionsContext, type ScaleActions } from './ScaleActionsContext'

const STORAGE_KEY = 'orchestral-scale-context'

interface StoredState {
  key: NoteName
  mode: Mode
  showHarmonicV: boolean
}

function loadFromStorage(): StoredState | null {
  if (typeof window === 'undefined') return null
  try {
    const stored = localStorage.getItem(STORAGE_KEY)
    if (!stored) return null
    return JSON.parse(stored) as StoredState
  } catch {
    return null
  }
}

function saveToStorage(state: StoredState): void {
  if (typeof window === 'undefined') return
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state))
  } catch {
    // Ignore storage errors
  }
}

interface ScaleProviderProps {
  children: ReactNode
}

export function ScaleProvider({ children }: ScaleProviderProps) {
  const [key, setKeyState] = useState<NoteName>('C')
  const [mode, setModeState] = useState<Mode>('major')
  const [showHarmonicV, setShowHarmonicV] = useState(false)
  const [isHydrated, setIsHydrated] = useState(false)

  // Hydrate from localStorage on mount
  useEffect(() => {
    const stored = loadFromStorage()
    if (stored) {
      setKeyState(stored.key)
      setModeState(stored.mode)
      setShowHarmonicV(stored.showHarmonicV)
    }
    setIsHydrated(true)
  }, [])

  // Persist to localStorage on change (after hydration)
  useEffect(() => {
    if (!isHydrated) return
    saveToStorage({ key, mode, showHarmonicV })
  }, [key, mode, showHarmonicV, isHydrated])

  const setKey = useCallback((newKey: NoteName) => {
    setKeyState(newKey)
  }, [])

  const setMode = useCallback((newMode: Mode) => {
    setModeState(newMode)
  }, [])

  const toggleHarmonicV = useCallback(() => {
    setShowHarmonicV((prev) => !prev)
  }, [])

  const options: ScaleOptions = useMemo(
    () => ({ showHarmonicV }),
    [showHarmonicV]
  )

  const actions: ScaleActions = useMemo(
    () => ({ setKey, setMode, toggleHarmonicV }),
    [setKey, setMode, toggleHarmonicV]
  )

  return (
    <ScaleActionsContext.Provider value={actions}>
      <ScaleKeyContext.Provider value={key}>
        <ScaleModeContext.Provider value={mode}>
          <ScaleOptionsContext.Provider value={options}>
            {children}
          </ScaleOptionsContext.Provider>
        </ScaleModeContext.Provider>
      </ScaleKeyContext.Provider>
    </ScaleActionsContext.Provider>
  )
}

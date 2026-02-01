'use client'

import { createContext, useContext } from 'react'
import type { Mode, NoteName } from '@/lib/music-theory'

export interface ScaleActions {
  setKey: (key: NoteName) => void
  setMode: (mode: Mode) => void
  toggleHarmonicV: () => void
}

export const ScaleActionsContext = createContext<ScaleActions | null>(null)

export function useScaleActions(): ScaleActions {
  const context = useContext(ScaleActionsContext)
  if (!context) {
    throw new Error('useScaleActions must be used within a ScaleProvider')
  }
  return context
}

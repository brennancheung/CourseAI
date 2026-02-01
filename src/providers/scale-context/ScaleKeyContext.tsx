'use client'

import { createContext, useContext } from 'react'
import type { NoteName } from '@/lib/music-theory'

export const ScaleKeyContext = createContext<NoteName>('C')

export function useScaleKey(): NoteName {
  return useContext(ScaleKeyContext)
}

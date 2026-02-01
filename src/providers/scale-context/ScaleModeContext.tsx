'use client'

import { createContext, useContext } from 'react'
import type { Mode } from '@/lib/music-theory'

export const ScaleModeContext = createContext<Mode>('major')

export function useScaleMode(): Mode {
  return useContext(ScaleModeContext)
}

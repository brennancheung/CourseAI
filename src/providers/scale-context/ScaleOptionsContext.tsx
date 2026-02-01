'use client'

import { createContext, useContext } from 'react'

export interface ScaleOptions {
  showHarmonicV: boolean
}

const defaultOptions: ScaleOptions = {
  showHarmonicV: false,
}

export const ScaleOptionsContext = createContext<ScaleOptions>(defaultOptions)

export function useScaleOptions(): ScaleOptions {
  return useContext(ScaleOptionsContext)
}

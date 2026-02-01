'use client'

import React, { createContext, useContext, ReactNode } from 'react'
import { CommandPalette } from './CommandPalette'
import { useTheme } from 'next-themes'

interface CommandPaletteContextType {
  setTheme?: (theme: string) => void
}

const CommandPaletteContext = createContext<CommandPaletteContextType | undefined>(undefined)

export const useCommandPaletteActions = () => {
  const context = useContext(CommandPaletteContext)
  if (!context) {
    throw new Error('useCommandPaletteActions must be used within CommandPaletteProvider')
  }
  return context
}

interface CommandPaletteProviderProps {
  children: ReactNode
}

export const CommandPaletteProvider = ({
  children
}: CommandPaletteProviderProps) => {
  const { setTheme } = useTheme()

  // Actions
  const actions: CommandPaletteContextType = {
    setTheme,
  }

  return (
    <CommandPaletteContext.Provider value={actions}>
      {children}
      <CommandPalette />
    </CommandPaletteContext.Provider>
  )
}

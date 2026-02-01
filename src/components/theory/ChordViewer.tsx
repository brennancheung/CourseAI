'use client'

import { useMemo } from 'react'
import { getDiatonicChords } from '@/lib/music-theory'
import { useScaleKey } from '@/providers/scale-context/ScaleKeyContext'
import { useScaleMode } from '@/providers/scale-context/ScaleModeContext'
import { useScaleOptions } from '@/providers/scale-context/ScaleOptionsContext'
import { ChordCard } from './ChordCard'
import { cn } from '@/lib/utils'

interface ChordViewerProps {
  className?: string
}

export function ChordViewer({ className }: ChordViewerProps) {
  const key = useScaleKey()
  const mode = useScaleMode()
  const { showHarmonicV } = useScaleOptions()

  const chords = useMemo(
    () => getDiatonicChords(key, mode, showHarmonicV),
    [key, mode, showHarmonicV]
  )

  return (
    <div className={cn('space-y-3', className)}>
      <h3 className="text-sm font-medium text-muted-foreground">Diatonic Chords</h3>
      <div className="flex flex-wrap gap-2">
        {chords.map((chord, index) => (
          <ChordCard
            key={`${chord.root}-${chord.quality}-${index}`}
            chord={chord}
            degree={index}
          />
        ))}
      </div>
    </div>
  )
}

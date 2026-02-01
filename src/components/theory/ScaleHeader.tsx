'use client'

import { useMemo } from 'react'
import { getScaleNotes, getModeDisplayName, DEGREE_COLORS } from '@/lib/music-theory'
import { useScaleKey } from '@/providers/scale-context/ScaleKeyContext'
import { useScaleMode } from '@/providers/scale-context/ScaleModeContext'
import { cn } from '@/lib/utils'

interface ScaleHeaderProps {
  className?: string
}

export function ScaleHeader({ className }: ScaleHeaderProps) {
  const key = useScaleKey()
  const mode = useScaleMode()

  const scaleNotes = useMemo(() => getScaleNotes(key, mode), [key, mode])
  const modeDisplayName = getModeDisplayName(mode)

  return (
    <div className={cn('space-y-2', className)}>
      <h2 className="text-2xl font-bold">
        {key} {modeDisplayName}
      </h2>
      <div className="flex items-center gap-2">
        <span className="text-sm text-muted-foreground">Scale:</span>
        <div className="flex gap-1">
          {scaleNotes.map((note, index) => {
            const colors = DEGREE_COLORS[index]
            return (
              <span
                key={`${note}-${index}`}
                className={cn(
                  'inline-flex h-7 w-7 items-center justify-center rounded border text-xs font-medium opacity-60',
                  colors.bg,
                  colors.border,
                  colors.text
                )}
              >
                {note}
              </span>
            )
          })}
        </div>
      </div>
    </div>
  )
}

'use client'

import type { Mode } from '@/lib/music-theory'
import { useScaleMode } from '@/providers/scale-context/ScaleModeContext'
import { useScaleOptions } from '@/providers/scale-context/ScaleOptionsContext'
import { useScaleActions } from '@/providers/scale-context/ScaleActionsContext'
import { cn } from '@/lib/utils'
import { Checkbox } from '@/components/ui/checkbox'
import { Label } from '@/components/ui/label'

interface ModeToggleProps {
  className?: string
}

export function ModeToggle({ className }: ModeToggleProps) {
  const mode = useScaleMode()
  const { showHarmonicV } = useScaleOptions()
  const { setMode, toggleHarmonicV } = useScaleActions()

  const isMinor = mode === 'natural-minor' || mode === 'harmonic-minor'

  const handleModeChange = (newMode: Mode) => {
    setMode(newMode)
  }

  return (
    <div className={cn('space-y-3', className)}>
      {/* Mode selector */}
      <div className="inline-flex rounded-lg border bg-muted p-1">
        <button
          onClick={() => handleModeChange('major')}
          className={cn(
            'rounded-md px-4 py-2 text-sm font-medium transition-colors',
            mode === 'major'
              ? 'bg-background text-foreground shadow-sm'
              : 'text-muted-foreground hover:text-foreground'
          )}
        >
          Major
        </button>
        <button
          onClick={() => handleModeChange('natural-minor')}
          className={cn(
            'rounded-md px-4 py-2 text-sm font-medium transition-colors',
            isMinor
              ? 'bg-background text-foreground shadow-sm'
              : 'text-muted-foreground hover:text-foreground'
          )}
        >
          Minor
        </button>
      </div>

      {/* Harmonic V option (only for minor) */}
      {isMinor && (
        <div className="flex items-center space-x-2">
          <Checkbox
            id="harmonic-v"
            checked={showHarmonicV}
            onCheckedChange={toggleHarmonicV}
          />
          <Label
            htmlFor="harmonic-v"
            className="text-sm text-muted-foreground cursor-pointer"
          >
            Show V major (harmonic minor)
          </Label>
        </div>
      )}
    </div>
  )
}

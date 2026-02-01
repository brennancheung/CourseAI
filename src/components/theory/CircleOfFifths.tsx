'use client'

import { useCallback } from 'react'
import {
  CIRCLE_OF_FIFTHS_MAJOR,
  CIRCLE_OF_FIFTHS_MINOR,
  type NoteName,
} from '@/lib/music-theory'
import { useScaleKey } from '@/providers/scale-context/ScaleKeyContext'
import { useScaleMode } from '@/providers/scale-context/ScaleModeContext'
import { useScaleActions } from '@/providers/scale-context/ScaleActionsContext'
import { cn } from '@/lib/utils'

interface CircleOfFifthsProps {
  className?: string
  size?: number
}

export function CircleOfFifths({ className, size = 320 }: CircleOfFifthsProps) {
  const selectedKey = useScaleKey()
  const selectedMode = useScaleMode()
  const { setKey, setMode } = useScaleActions()

  const isMinorMode = selectedMode === 'natural-minor' || selectedMode === 'harmonic-minor'

  // Find position based on mode - look in the correct array
  const selectedPosition = isMinorMode
    ? CIRCLE_OF_FIFTHS_MINOR.findIndex((k) => k === selectedKey)
    : CIRCLE_OF_FIFTHS_MAJOR.findIndex((k) => k === selectedKey)

  const handleMajorKeyClick = useCallback(
    (key: NoteName) => {
      setKey(key)
      setMode('major')
    },
    [setKey, setMode]
  )

  const handleMinorKeyClick = useCallback(
    (key: NoteName) => {
      setKey(key)
      setMode('natural-minor')
    },
    [setKey, setMode]
  )

  const center = size / 2
  const outerRadius = size * 0.42
  const innerRadius = size * 0.28
  const keyCircleRadius = size * 0.065

  // Calculate position on circle for a given index (0-11)
  // Round to 2 decimal places to avoid hydration mismatches from floating point precision
  const getPosition = (index: number, radius: number) => {
    // Start at 12 o'clock (-90 degrees) and go clockwise
    const angle = (index * 30 - 90) * (Math.PI / 180)
    return {
      x: Math.round((center + radius * Math.cos(angle)) * 100) / 100,
      y: Math.round((center + radius * Math.sin(angle)) * 100) / 100,
    }
  }

  return (
    <div className={cn('inline-block', className)}>
      <svg
        width={size}
        height={size}
        viewBox={`0 0 ${size} ${size}`}
        className="select-none"
      >
        {/* Background circle */}
        <circle
          cx={center}
          cy={center}
          r={size * 0.48}
          className="fill-muted/30 stroke-border"
          strokeWidth={1}
        />

        {/* Inner divider circle */}
        <circle
          cx={center}
          cy={center}
          r={(outerRadius + innerRadius) / 2}
          className="fill-none stroke-border/50"
          strokeWidth={1}
          strokeDasharray="4 4"
        />

        {/* Major keys (outer ring) */}
        {CIRCLE_OF_FIFTHS_MAJOR.map((key, index) => {
          const pos = getPosition(index, outerRadius)
          const isSelected = !isMinorMode && index === selectedPosition

          return (
            <g key={`major-${key}`}>
              <circle
                cx={pos.x}
                cy={pos.y}
                r={keyCircleRadius}
                className={cn(
                  'cursor-pointer transition-colors',
                  isSelected
                    ? 'fill-primary stroke-primary'
                    : 'fill-background stroke-border hover:fill-accent hover:stroke-accent-foreground'
                )}
                strokeWidth={2}
                onClick={() => handleMajorKeyClick(key)}
              />
              <text
                x={pos.x}
                y={pos.y}
                textAnchor="middle"
                dominantBaseline="central"
                className={cn(
                  'pointer-events-none text-xs font-semibold',
                  isSelected ? 'fill-primary-foreground' : 'fill-foreground'
                )}
              >
                {key}
              </text>
            </g>
          )
        })}

        {/* Minor keys (inner ring) */}
        {CIRCLE_OF_FIFTHS_MINOR.map((key, index) => {
          const pos = getPosition(index, innerRadius)
          const isSelected = isMinorMode && index === selectedPosition

          return (
            <g key={`minor-${key}`}>
              <circle
                cx={pos.x}
                cy={pos.y}
                r={keyCircleRadius * 0.85}
                className={cn(
                  'cursor-pointer transition-colors',
                  isSelected
                    ? 'fill-primary/80 stroke-primary'
                    : 'fill-muted stroke-border hover:fill-accent hover:stroke-accent-foreground'
                )}
                strokeWidth={1.5}
                onClick={() => handleMinorKeyClick(key)}
              />
              <text
                x={pos.x}
                y={pos.y}
                textAnchor="middle"
                dominantBaseline="central"
                className={cn(
                  'pointer-events-none text-[10px] font-medium',
                  isSelected ? 'fill-primary-foreground' : 'fill-muted-foreground'
                )}
              >
                {key}m
              </text>
            </g>
          )
        })}

        {/* Center label */}
        <text
          x={center}
          y={center}
          textAnchor="middle"
          dominantBaseline="central"
          className="fill-muted-foreground text-[10px] font-medium"
        >
          Circle of 5ths
        </text>
      </svg>
    </div>
  )
}

'use client'

import { useState, useCallback, useEffect, useRef } from 'react'
import { cn } from '@/lib/utils'
import { Play, Pause, SkipForward, RotateCcw } from 'lucide-react'
import { Button } from '@/components/ui/button'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type GridData = number[][]

type PresetInput = {
  name: string
  grid: GridData
}

type PresetFilter = {
  name: string
  grid: GridData
  description: string
}

type ConvolutionExplorerProps = {
  width?: number
  height?: number
}

// ---------------------------------------------------------------------------
// Data: Preset inputs (7x7)
// ---------------------------------------------------------------------------

const INPUT_SIZE = 7
const FILTER_SIZE = 3

const presetInputs: PresetInput[] = [
  {
    name: 'Vertical Edge',
    grid: [
      [0, 0, 0, 1, 1, 1, 1],
      [0, 0, 0, 1, 1, 1, 1],
      [0, 0, 0, 1, 1, 1, 1],
      [0, 0, 0, 1, 1, 1, 1],
      [0, 0, 0, 1, 1, 1, 1],
      [0, 0, 0, 1, 1, 1, 1],
      [0, 0, 0, 1, 1, 1, 1],
    ],
  },
  {
    name: 'Horizontal Edge',
    grid: [
      [0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0],
      [1, 1, 1, 1, 1, 1, 1],
      [1, 1, 1, 1, 1, 1, 1],
      [1, 1, 1, 1, 1, 1, 1],
      [1, 1, 1, 1, 1, 1, 1],
    ],
  },
  {
    name: 'Diagonal',
    grid: [
      [1, 0, 0, 0, 0, 0, 0],
      [0, 1, 0, 0, 0, 0, 0],
      [0, 0, 1, 0, 0, 0, 0],
      [0, 0, 0, 1, 0, 0, 0],
      [0, 0, 0, 0, 1, 0, 0],
      [0, 0, 0, 0, 0, 1, 0],
      [0, 0, 0, 0, 0, 0, 1],
    ],
  },
  {
    name: 'Uniform',
    grid: [
      [1, 1, 1, 1, 1, 1, 1],
      [1, 1, 1, 1, 1, 1, 1],
      [1, 1, 1, 1, 1, 1, 1],
      [1, 1, 1, 1, 1, 1, 1],
      [1, 1, 1, 1, 1, 1, 1],
      [1, 1, 1, 1, 1, 1, 1],
      [1, 1, 1, 1, 1, 1, 1],
    ],
  },
  {
    name: 'Corner',
    grid: [
      [1, 1, 1, 1, 0, 0, 0],
      [1, 1, 1, 1, 0, 0, 0],
      [1, 1, 1, 1, 0, 0, 0],
      [1, 1, 1, 1, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0],
    ],
  },
]

// ---------------------------------------------------------------------------
// Data: Preset filters (3x3)
// ---------------------------------------------------------------------------

const presetFilters: PresetFilter[] = [
  {
    name: 'Vertical Edge',
    description: 'Detects vertical edges (left-right transitions)',
    grid: [
      [-1, 0, 1],
      [-1, 0, 1],
      [-1, 0, 1],
    ],
  },
  {
    name: 'Horizontal Edge',
    description: 'Detects horizontal edges (top-bottom transitions)',
    grid: [
      [-1, -1, -1],
      [0, 0, 0],
      [1, 1, 1],
    ],
  },
  {
    name: 'Blur',
    description: 'Averages neighboring pixels (smoothing)',
    grid: [
      [1, 1, 1],
      [1, 1, 1],
      [1, 1, 1],
    ],
  },
  {
    name: 'Sharpen',
    description: 'Enhances differences from neighbors',
    grid: [
      [0, -1, 0],
      [-1, 5, -1],
      [0, -1, 0],
    ],
  },
]

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const OUTPUT_SIZE = INPUT_SIZE - FILTER_SIZE + 1

function computeConvolution(input: GridData, filter: GridData): GridData {
  const output: GridData = []
  for (let i = 0; i < OUTPUT_SIZE; i++) {
    const row: number[] = []
    for (let j = 0; j < OUTPUT_SIZE; j++) {
      let sum = 0
      for (let fi = 0; fi < FILTER_SIZE; fi++) {
        for (let fj = 0; fj < FILTER_SIZE; fj++) {
          sum += input[i + fi][j + fj] * filter[fi][fj]
        }
      }
      row.push(sum)
    }
    output.push(row)
  }
  return output
}

function positionToRowCol(pos: number): { row: number; col: number } {
  return {
    row: Math.floor(pos / OUTPUT_SIZE),
    col: pos % OUTPUT_SIZE,
  }
}

function computeStepProducts(
  input: GridData,
  filter: GridData,
  row: number,
  col: number
): number[] {
  const products: number[] = []
  for (let fi = 0; fi < FILTER_SIZE; fi++) {
    for (let fj = 0; fj < FILTER_SIZE; fj++) {
      products.push(input[row + fi][col + fj] * filter[fi][fj])
    }
  }
  return products
}

function getCellColor(value: number, maxAbs: number): string {
  if (maxAbs === 0) return 'bg-zinc-800'
  const normalized = value / maxAbs
  if (normalized > 0.5) return 'bg-blue-400 text-zinc-900'
  if (normalized > 0.2) return 'bg-blue-500/60'
  if (normalized < -0.5) return 'bg-orange-400 text-zinc-900'
  if (normalized < -0.2) return 'bg-orange-500/60'
  return 'bg-zinc-700'
}

function getInputCellColor(value: number): string {
  if (value >= 1) return 'bg-zinc-300 text-zinc-900'
  if (value > 0) return 'bg-zinc-500 text-zinc-100'
  return 'bg-zinc-800 text-zinc-400'
}

function getFilterCellColor(value: number): string {
  if (value > 0) return 'bg-violet-400/70 text-zinc-100'
  if (value < 0) return 'bg-rose-400/70 text-zinc-100'
  return 'bg-zinc-700 text-zinc-400'
}

// ---------------------------------------------------------------------------
// Component: Cell
// ---------------------------------------------------------------------------

function Cell({
  value,
  colorClass,
  highlight,
  size,
  ariaLabel,
}: {
  value: number | string
  colorClass: string
  highlight?: boolean
  size: 'sm' | 'md'
  ariaLabel?: string
}) {
  const sizeClasses = size === 'sm' ? 'w-8 h-8 text-[10px]' : 'w-9 h-9 text-xs'
  return (
    <div
      className={cn(
        sizeClasses,
        'flex items-center justify-center rounded font-mono font-medium transition-all duration-200',
        colorClass,
        highlight && 'ring-2 ring-yellow-400 z-10',
      )}
      aria-label={ariaLabel}
    >
      {typeof value === 'number' ? (Number.isInteger(value) ? value : value.toFixed(1)) : value}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Component: Grid
// ---------------------------------------------------------------------------

function Grid({
  data,
  label,
  highlightRegion,
  colorFn,
  size = 'md',
}: {
  data: GridData
  label: string
  highlightRegion?: { startRow: number; startCol: number; rows: number; cols: number }
  colorFn: (value: number) => string
  size?: 'sm' | 'md'
}) {
  return (
    <div className="flex flex-col items-center gap-1.5">
      <span className="text-xs font-medium text-muted-foreground uppercase tracking-wide">{label}</span>
      <div
        className="grid gap-0.5"
        style={{ gridTemplateColumns: `repeat(${data[0].length}, 1fr)` }}
      >
        {data.flatMap((row, ri) =>
          row.map((val, ci) => {
            const isHighlighted =
              highlightRegion &&
              ri >= highlightRegion.startRow &&
              ri < highlightRegion.startRow + highlightRegion.rows &&
              ci >= highlightRegion.startCol &&
              ci < highlightRegion.startCol + highlightRegion.cols
            return (
              <Cell
                key={`${ri}-${ci}`}
                value={val}
                colorClass={colorFn(val)}
                highlight={isHighlighted}
                size={size}
                ariaLabel={`${label} row ${ri} column ${ci}: ${val}`}
              />
            )
          })
        )}
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Component: OutputGrid (partially filled during stepping)
// ---------------------------------------------------------------------------

function OutputGrid({
  fullOutput,
  revealedUpTo,
  currentPos,
  maxAbs,
}: {
  fullOutput: GridData
  revealedUpTo: number
  currentPos: number
  maxAbs: number
}) {
  return (
    <div className="flex flex-col items-center gap-1.5">
      <span className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
        Feature Map
      </span>
      <div
        className="grid gap-0.5"
        style={{ gridTemplateColumns: `repeat(${OUTPUT_SIZE}, 1fr)` }}
      >
        {fullOutput.flatMap((row, ri) =>
          row.map((val, ci) => {
            const flatIdx = ri * OUTPUT_SIZE + ci
            const isRevealed = flatIdx <= revealedUpTo
            const isCurrent = flatIdx === currentPos
            return (
              <Cell
                key={`${ri}-${ci}`}
                value={isRevealed ? val : ''}
                colorClass={
                  isRevealed
                    ? getCellColor(val, maxAbs)
                    : 'bg-zinc-900 border border-dashed border-zinc-700'
                }
                highlight={isCurrent}
                size="md"
                ariaLabel={`Feature map row ${ri} column ${ci}: ${isRevealed ? val : 'not yet computed'}`}
              />
            )
          })
        )}
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Component: Computation Detail
// ---------------------------------------------------------------------------

function ComputationDetail({
  input,
  filter,
  row,
  col,
}: {
  input: GridData
  filter: GridData
  row: number
  col: number
}) {
  const products = computeStepProducts(input, filter, row, col)
  const sum = products.reduce((a, b) => a + b, 0)

  return (
    <div className="bg-zinc-900/50 rounded-lg p-3 border border-zinc-800">
      <p className="text-xs text-muted-foreground mb-2">
        Position ({row}, {col}): multiply each input by its filter weight, then sum
      </p>
      <div className="flex flex-wrap gap-1 items-center text-xs font-mono">
        {products.map((p, i) => (
          <span key={i} className="flex items-center gap-0.5">
            {i > 0 && <span className="text-zinc-500 mx-0.5">+</span>}
            <span
              className={cn(
                'px-1 py-0.5 rounded',
                p > 0 ? 'text-blue-300' : p < 0 ? 'text-orange-300' : 'text-zinc-500',
              )}
            >
              {Number.isInteger(p) ? p : p.toFixed(1)}
            </span>
          </span>
        ))}
        <span className="text-zinc-500 mx-1">=</span>
        <span className="font-bold text-white px-1.5 py-0.5 bg-zinc-700 rounded">
          {Number.isInteger(sum) ? sum : sum.toFixed(1)}
        </span>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Main Component
// ---------------------------------------------------------------------------

export function ConvolutionExplorer({ width: _width, height: _height }: ConvolutionExplorerProps) {
  const [inputIdx, setInputIdx] = useState(0)
  const [filterIdx, setFilterIdx] = useState(0)
  const [currentStep, setCurrentStep] = useState(-1) // -1 = show full output
  const [isPlaying, setIsPlaying] = useState(false)
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const input = presetInputs[inputIdx].grid
  const filter = presetFilters[filterIdx].grid
  const fullOutput = computeConvolution(input, filter)
  const totalSteps = OUTPUT_SIZE * OUTPUT_SIZE

  // Find max absolute value in output for color scaling
  const maxAbs = fullOutput.reduce(
    (max, row) => Math.max(max, ...row.map(Math.abs)),
    0,
  )

  const isStepping = currentStep >= 0

  const { row: currentRow, col: currentCol } = isStepping
    ? positionToRowCol(currentStep)
    : { row: -1, col: -1 }

  // Cleanup interval on unmount
  useEffect(() => {
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current)
    }
  }, [])

  // Auto-play
  useEffect(() => {
    if (!isPlaying) {
      if (intervalRef.current) {
        clearInterval(intervalRef.current)
        intervalRef.current = null
      }
      return
    }

    intervalRef.current = setInterval(() => {
      setCurrentStep((prev) => {
        const next = prev + 1
        if (next >= totalSteps) {
          setIsPlaying(false)
          return totalSteps - 1
        }
        return next
      })
    }, 600)

    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current)
    }
  }, [isPlaying, totalSteps])

  const handleStepForward = useCallback(() => {
    setIsPlaying(false)
    setCurrentStep((prev) => {
      if (prev < 0) return 0
      if (prev >= totalSteps - 1) return prev
      return prev + 1
    })
  }, [totalSteps])

  const handleReset = useCallback(() => {
    setIsPlaying(false)
    setCurrentStep(-1)
  }, [])

  const handlePlayPause = useCallback(() => {
    if (currentStep < 0) {
      setCurrentStep(0)
      setIsPlaying(true)
      return
    }
    if (currentStep >= totalSteps - 1) {
      setCurrentStep(0)
      setIsPlaying(true)
      return
    }
    setIsPlaying((p) => !p)
  }, [currentStep, totalSteps])

  const handleInputChange = useCallback((idx: number) => {
    setInputIdx(idx)
    setCurrentStep(-1)
    setIsPlaying(false)
  }, [])

  const handleFilterChange = useCallback((idx: number) => {
    setFilterIdx(idx)
    setCurrentStep(-1)
    setIsPlaying(false)
  }, [])

  return (
    <div className="space-y-5">
      {/* Selectors */}
      <div className="flex flex-col sm:flex-row gap-4">
        <div className="space-y-1.5">
          <span className="text-xs font-medium text-muted-foreground">Input Pattern</span>
          <div className="flex flex-wrap gap-1.5">
            {presetInputs.map((preset, i) => (
              <button
                key={preset.name}
                onClick={() => handleInputChange(i)}
                className={cn(
                  'px-2.5 py-1 rounded text-xs font-medium transition-colors cursor-pointer',
                  inputIdx === i
                    ? 'bg-primary text-primary-foreground'
                    : 'bg-muted hover:bg-muted/80 text-muted-foreground',
                )}
              >
                {preset.name}
              </button>
            ))}
          </div>
        </div>
        <div className="space-y-1.5">
          <span className="text-xs font-medium text-muted-foreground">Filter</span>
          <div className="flex flex-wrap gap-1.5">
            {presetFilters.map((preset, i) => (
              <button
                key={preset.name}
                onClick={() => handleFilterChange(i)}
                className={cn(
                  'px-2.5 py-1 rounded text-xs font-medium transition-colors cursor-pointer',
                  filterIdx === i
                    ? 'bg-violet-500 text-white'
                    : 'bg-muted hover:bg-muted/80 text-muted-foreground',
                )}
              >
                {preset.name}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Filter description */}
      <p className="text-xs text-muted-foreground italic">
        {presetFilters[filterIdx].description}
      </p>

      {/* Grids */}
      <div className="flex flex-wrap items-start gap-6 justify-center">
        <Grid
          data={input}
          label="Input"
          colorFn={getInputCellColor}
          highlightRegion={
            isStepping
              ? {
                  startRow: currentRow,
                  startCol: currentCol,
                  rows: FILTER_SIZE,
                  cols: FILTER_SIZE,
                }
              : undefined
          }
        />

        <div className="flex items-center self-center text-zinc-500 text-lg font-bold pt-5">*</div>

        <Grid
          data={filter}
          label="Filter"
          colorFn={getFilterCellColor}
          size="sm"
        />

        <div className="flex items-center self-center text-zinc-500 text-lg font-bold pt-5">=</div>

        <OutputGrid
          fullOutput={fullOutput}
          revealedUpTo={isStepping ? currentStep : totalSteps - 1}
          currentPos={isStepping ? currentStep : -1}
          maxAbs={maxAbs}
        />
      </div>

      {/* Computation detail */}
      {isStepping && (
        <ComputationDetail
          input={input}
          filter={filter}
          row={currentRow}
          col={currentCol}
        />
      )}

      {/* Controls */}
      <div className="flex items-center gap-2 justify-center">
        <Button
          variant="outline"
          size="sm"
          onClick={handlePlayPause}
          className="gap-1.5 cursor-pointer"
        >
          {isPlaying ? (
            <>
              <Pause className="w-3.5 h-3.5" />
              Pause
            </>
          ) : (
            <>
              <Play className="w-3.5 h-3.5" />
              {currentStep < 0 ? 'Animate' : currentStep >= totalSteps - 1 ? 'Replay' : 'Play'}
            </>
          )}
        </Button>
        <Button
          variant="outline"
          size="sm"
          onClick={handleStepForward}
          disabled={isPlaying || currentStep >= totalSteps - 1}
          className="gap-1.5 cursor-pointer"
        >
          <SkipForward className="w-3.5 h-3.5" />
          Step
        </Button>
        <Button
          variant="outline"
          size="sm"
          onClick={handleReset}
          className="gap-1.5 cursor-pointer"
        >
          <RotateCcw className="w-3.5 h-3.5" />
          Reset
        </Button>

        {isStepping && (
          <span className="text-xs text-muted-foreground ml-2">
            Step {currentStep + 1} / {totalSteps}
          </span>
        )}
      </div>
    </div>
  )
}

'use client'

import { useState, useMemo } from 'react'
import { useContainerWidth } from '@/hooks/useContainerWidth'
import { cn } from '@/lib/utils'
import {
  PRECOMPUTED_SAMPLES,
  BOTTLENECK_SIZES,
} from './autoencoder-data'
import type { BottleneckSize } from './autoencoder-data'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type WidgetProps = {
  width?: number
  height?: number
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const IMG_SIZE = 14 // Stored as 14x14
const DISPLAY_SCALE = 4 // Render each stored pixel as 4x4

// ---------------------------------------------------------------------------
// Latent code generation (simulated — exact values matter less than shape)
// ---------------------------------------------------------------------------

function mulberry32(seed: number) {
  return () => {
    let t = (seed += 0x6d2b79f5)
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

function generateLatentCode(
  bottleneckSize: BottleneckSize,
  sampleIndex: number
): number[] {
  const rng = mulberry32(2000 + sampleIndex * 100 + bottleneckSize)
  const code: number[] = []
  // First values tend to be larger (capture more variance)
  for (let i = 0; i < Math.min(bottleneckSize, 32); i++) {
    const scale = Math.exp(-i * 0.08)
    code.push((rng() * 2 - 1) * scale * 3)
  }
  return code
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function PixelGrid({
  pixels,
  size,
  label,
  scale,
}: {
  pixels: number[]
  size: number
  label: string
  scale: number
}) {
  const canvasSize = size * scale
  const cellSize = scale

  return (
    <div className="flex flex-col items-center gap-1.5">
      <span className="text-xs font-medium text-muted-foreground">{label}</span>
      <svg
        width={canvasSize}
        height={canvasSize}
        viewBox={`0 0 ${canvasSize} ${canvasSize}`}
        className="rounded border border-border/50"
        style={{ width: canvasSize, height: canvasSize }}
      >
        <rect width={canvasSize} height={canvasSize} fill="#0a0a0a" />
        {pixels.map((v, i) => {
          const x = (i % size) * cellSize
          const y = Math.floor(i / size) * cellSize
          // Values are 0-255 integers in the data
          const brightness = Math.round(Math.min(255, Math.max(0, v)))
          return (
            <rect
              key={i}
              x={x}
              y={y}
              width={cellSize + 0.5}
              height={cellSize + 0.5}
              fill={`rgb(${brightness}, ${brightness}, ${brightness})`}
            />
          )
        })}
      </svg>
    </div>
  )
}

function LatentCodeBar({
  values,
  bottleneckSize,
  maxDisplay,
}: {
  values: number[]
  bottleneckSize: BottleneckSize
  maxDisplay: number
}) {
  const displayed = values.slice(0, maxDisplay)
  const barHeight = 80
  const maxVal = 3

  return (
    <div className="flex flex-col items-center gap-1.5">
      <span className="text-xs font-medium text-muted-foreground">
        Latent Code ({bottleneckSize}d)
      </span>
      <svg
        width={Math.max(displayed.length * 6, 60)}
        height={barHeight + 16}
        className="rounded border border-border/50"
        style={{ background: '#0a0a0a' }}
      >
        {displayed.map((val, i) => {
          const barW = Math.max(3, Math.min(6, 120 / displayed.length))
          const x = i * (barW + 1) + 2
          const normalized = Math.min(1, Math.max(-1, val / maxVal))
          const h = Math.abs(normalized) * (barHeight / 2)
          const midY = barHeight / 2 + 4

          const isPositive = normalized >= 0
          const y = isPositive ? midY - h : midY
          const color = isPositive
            ? `rgba(99, 102, 241, ${0.4 + Math.abs(normalized) * 0.6})`
            : `rgba(244, 63, 94, ${0.4 + Math.abs(normalized) * 0.6})`

          return (
            <rect
              key={i}
              x={x}
              y={y}
              width={barW}
              height={Math.max(1, h)}
              fill={color}
              rx={1}
            />
          )
        })}
        {/* Zero line */}
        <line
          x1={0}
          y1={barHeight / 2 + 4}
          x2={Math.max(displayed.length * 7, 60)}
          y2={barHeight / 2 + 4}
          stroke="rgba(255,255,255,0.15)"
          strokeWidth={0.5}
        />
        {/* Dimension count label */}
        {bottleneckSize > maxDisplay && (
          <text
            x={Math.max(displayed.length * 6, 60) / 2}
            y={barHeight + 14}
            textAnchor="middle"
            fill="rgba(255,255,255,0.4)"
            fontSize={8}
          >
            showing {maxDisplay} of {bottleneckSize}
          </text>
        )}
      </svg>
    </div>
  )
}

function BottleneckArrow() {
  return (
    <div className="flex flex-col items-center justify-center px-1">
      <svg width={24} height={60} viewBox="0 0 24 60">
        {/* Hourglass shape */}
        <path
          d="M2,5 L22,5 L14,30 L22,55 L2,55 L10,30 Z"
          fill="none"
          stroke="rgba(139,92,246,0.5)"
          strokeWidth={1.5}
        />
        <path
          d="M2,5 L22,5 L14,30 L22,55 L2,55 L10,30 Z"
          fill="rgba(139,92,246,0.08)"
        />
        {/* Arrow down */}
        <line
          x1={12}
          y1={15}
          x2={12}
          y2={45}
          stroke="rgba(139,92,246,0.4)"
          strokeWidth={1}
          strokeDasharray="3,2"
        />
      </svg>
      <span className="text-[9px] text-violet-400/60 font-medium">bottleneck</span>
    </div>
  )
}

function ReconstructionArrow() {
  return (
    <div className="flex items-center justify-center px-1">
      <svg width={20} height={20} viewBox="0 0 20 20">
        <line
          x1={2}
          y1={10}
          x2={14}
          y2={10}
          stroke="rgba(255,255,255,0.3)"
          strokeWidth={1.5}
        />
        <polyline
          points="11,6 16,10 11,14"
          fill="none"
          stroke="rgba(255,255,255,0.3)"
          strokeWidth={1.5}
        />
      </svg>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Insight text for each bottleneck size
// ---------------------------------------------------------------------------

function getInsightText(bottleneckSize: BottleneckSize, label: string): string {
  if (bottleneckSize <= 16) {
    return `The network must capture the essence of a ${label.toLowerCase()} in just ${bottleneckSize} numbers. Only the broad shape survives this extreme compression.`
  }
  if (bottleneckSize === 32) {
    return `With 32 dimensions, the network preserves overall shape and major features. This is roughly 4% of the original pixels.`
  }
  if (bottleneckSize === 64) {
    return `At 64 dimensions, finer details start to appear. The reconstruction captures edges and structural details beyond just the silhouette.`
  }
  if (bottleneckSize === 128) {
    return `With 128 dimensions, the reconstruction is quite faithful. But notice: more dimensions means less compression, and the representation is less forced to learn structure.`
  }
  return `At 256 dimensions (33% of input), reconstruction is nearly perfect. But is the network really learning what matters, or just passing values through?`
}

// ---------------------------------------------------------------------------
// Main widget component
// ---------------------------------------------------------------------------

export function AutoencoderBottleneckWidget({
  width: widthOverride,
}: WidgetProps) {
  const { containerRef, width: measuredWidth } = useContainerWidth(600)
  const width = widthOverride ?? measuredWidth

  const [bottleneckSize, setBottleneckSize] = useState<BottleneckSize>(32)
  const [selectedSample, setSelectedSample] = useState(0)

  const sample = PRECOMPUTED_SAMPLES[selectedSample]
  const reconstruction = sample.reconstructions[bottleneckSize]
  const latentCode = useMemo(
    () => generateLatentCode(bottleneckSize, selectedSample),
    [bottleneckSize, selectedSample]
  )

  // Compute MSE between original and reconstruction (values are 0-255)
  const mse = useMemo(() => {
    let sum = 0
    for (let i = 0; i < sample.original.length; i++) {
      // Normalize to 0-1 for MSE display
      const diff = (sample.original[i] - reconstruction[i]) / 255
      sum += diff * diff
    }
    return sum / sample.original.length
  }, [sample.original, reconstruction])

  const isCompact = width < 500

  return (
    <div ref={containerRef} className="space-y-4">
      {/* Image selector */}
      <div className="flex items-center justify-center gap-2">
        {PRECOMPUTED_SAMPLES.map((s, idx) => (
          <button
            key={s.id}
            onClick={() => setSelectedSample(idx)}
            className={cn(
              'px-3 py-1.5 rounded-lg text-xs font-medium transition-colors cursor-pointer',
              selectedSample === idx
                ? 'bg-violet-500/20 text-violet-300 border border-violet-500/30'
                : 'bg-muted text-muted-foreground hover:bg-muted/80'
            )}
          >
            {s.label}
          </button>
        ))}
      </div>

      {/* Main visualization: Original -> Bottleneck -> Reconstruction */}
      <div className="flex items-center justify-center gap-2 flex-wrap">
        <PixelGrid
          pixels={sample.original}
          size={IMG_SIZE}
          label="Original"
          scale={DISPLAY_SCALE}
        />
        <BottleneckArrow />
        <LatentCodeBar
          values={latentCode}
          bottleneckSize={bottleneckSize}
          maxDisplay={isCompact ? 16 : 32}
        />
        <ReconstructionArrow />
        <PixelGrid
          pixels={reconstruction}
          size={IMG_SIZE}
          label="Reconstruction"
          scale={DISPLAY_SCALE}
        />
      </div>

      {/* Bottleneck size slider */}
      <div className="space-y-2 px-4">
        <div className="flex items-center justify-between">
          <label className="text-xs font-medium text-muted-foreground">
            Bottleneck Size
          </label>
          <span className="text-sm font-mono font-bold text-violet-400">
            {bottleneckSize} dimensions
          </span>
        </div>
        <input
          type="range"
          min={0}
          max={BOTTLENECK_SIZES.length - 1}
          value={BOTTLENECK_SIZES.indexOf(bottleneckSize)}
          onChange={(e) => {
            setBottleneckSize(BOTTLENECK_SIZES[Number(e.target.value)])
          }}
          className="w-full accent-violet-500 cursor-pointer"
        />
        <div className="flex justify-between text-[10px] text-muted-foreground/60">
          {BOTTLENECK_SIZES.map((size) => (
            <span
              key={size}
              className={cn(
                size === bottleneckSize && 'text-violet-400 font-medium'
              )}
            >
              {size}
            </span>
          ))}
        </div>
      </div>

      {/* Stats row */}
      <div className="flex gap-3 flex-wrap">
        <div className="flex-1 min-w-[120px] px-3 py-2 bg-muted/50 rounded-md">
          <p className="text-xs text-muted-foreground">Input Pixels</p>
          <p className="text-sm font-medium text-foreground">
            784 <span className="text-xs text-muted-foreground">(28&times;28)</span>
          </p>
        </div>
        <div className="flex-1 min-w-[120px] px-3 py-2 bg-muted/50 rounded-md">
          <p className="text-xs text-muted-foreground">Bottleneck</p>
          <p className="text-sm font-medium text-foreground">
            <span className="text-violet-400">{bottleneckSize}</span>{' '}
            <span className="text-xs text-muted-foreground">
              ({((bottleneckSize / 784) * 100).toFixed(1)}% of input)
            </span>
          </p>
        </div>
        <div className="flex-1 min-w-[120px] px-3 py-2 bg-muted/50 rounded-md">
          <p className="text-xs text-muted-foreground">Reconstruction MSE</p>
          <p className="text-sm font-medium text-foreground">
            <span
              className={cn(
                mse > 0.02 ? 'text-rose-400' : mse > 0.005 ? 'text-amber-400' : 'text-emerald-400'
              )}
            >
              {mse.toFixed(4)}
            </span>
          </p>
        </div>
      </div>

      {/* Compression ratio insight */}
      <div className="px-3 py-2 bg-muted/30 rounded-lg text-center">
        <p className="text-xs text-muted-foreground">
          {getInsightText(bottleneckSize, sample.label)}
        </p>
      </div>
    </div>
  )
}

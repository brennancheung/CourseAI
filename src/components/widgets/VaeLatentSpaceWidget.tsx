'use client'

import { useState, useMemo, useCallback } from 'react'
import { useContainerWidth } from '@/hooks/useContainerWidth'
import { cn } from '@/lib/utils'
import {
  ENCODED_POINTS,
  CATEGORY_COLORS,
  CATEGORY_LABELS,
  PRESET_SAMPLES,
  getDecodedSample,
} from './vae-latent-space-data'
import type { LatentPoint } from './vae-latent-space-data'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type WidgetProps = {
  width?: number
  height?: number
}

type Mode = 'ae' | 'vae'

type SampledPoint = {
  x: number
  y: number
  pixels: number[]
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const SPACE_RANGE = 3.5 // Latent space goes from -3.5 to 3.5
const POINT_RADIUS = 5
const DISPLAY_SCALE = 4
const IMG_SIZE = 14

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function latentToSvg(
  val: number,
  size: number,
  padding: number,
): number {
  const range = SPACE_RANGE * 2
  return padding + ((val + SPACE_RANGE) / range) * (size - padding * 2)
}

function svgToLatent(
  svgVal: number,
  size: number,
  padding: number,
): number {
  const range = SPACE_RANGE * 2
  return ((svgVal - padding) / (size - padding * 2)) * range - SPACE_RANGE
}

function getModeLabel(mode: Mode): string {
  if (mode === 'ae') return 'Autoencoder'
  return 'VAE'
}

function getInsightText(mode: Mode, beta: number): string {
  if (mode === 'ae') {
    return 'Autoencoder mode: encoded images are scattered points. Sampling from gaps produces garbage\u2014the decoder has never seen these regions.'
  }
  if (beta < 0.2) {
    return 'With \u03B2 near zero, the KL regularizer is off. The VAE behaves like an autoencoder\u2014gaps remain.'
  }
  if (beta > 3.0) {
    return 'With \u03B2 too high, the KL term dominates. All distributions collapse toward N(0,1)\u2014the space is smooth but reconstructions are very blurry.'
  }
  return 'VAE mode: each encoded image is a small cloud (distribution), not a point. The clouds overlap, filling gaps. Sample anywhere and get a plausible image.'
}

function getGaussianOpacity(
  px: number,
  py: number,
  cx: number,
  cy: number,
  sigma: number,
): number {
  const dx = px - cx
  const dy = py - cy
  return Math.exp(-(dx * dx + dy * dy) / (2 * sigma * sigma))
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function PixelPreview({
  pixels,
  label,
  scale = DISPLAY_SCALE,
}: {
  pixels: number[]
  label?: string
  scale?: number
}) {
  const canvasSize = IMG_SIZE * scale

  return (
    <div className="flex flex-col items-center gap-1">
      {label && (
        <span className="text-[10px] font-medium text-muted-foreground truncate max-w-[70px]">
          {label}
        </span>
      )}
      <svg
        width={canvasSize}
        height={canvasSize}
        viewBox={`0 0 ${canvasSize} ${canvasSize}`}
        className="rounded border border-border/50"
        style={{ width: canvasSize, height: canvasSize }}
      >
        <rect width={canvasSize} height={canvasSize} fill="#0a0a0a" />
        {pixels.map((v, i) => {
          const x = (i % IMG_SIZE) * scale
          const y = Math.floor(i / IMG_SIZE) * scale
          const brightness = Math.round(Math.min(255, Math.max(0, v)))
          return (
            <rect
              key={i}
              x={x}
              y={y}
              width={scale + 0.5}
              height={scale + 0.5}
              fill={`rgb(${brightness}, ${brightness}, ${brightness})`}
            />
          )
        })}
      </svg>
    </div>
  )
}

function Legend() {
  const categories = Object.entries(CATEGORY_LABELS) as Array<
    [LatentPoint['category'], string]
  >
  return (
    <div className="flex flex-wrap gap-3 justify-center">
      {categories.map(([cat, label]) => (
        <div key={cat} className="flex items-center gap-1.5">
          <div
            className="w-2.5 h-2.5 rounded-full"
            style={{ backgroundColor: CATEGORY_COLORS[cat] }}
          />
          <span className="text-[10px] text-muted-foreground">{label}</span>
        </div>
      ))}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Main widget component
// ---------------------------------------------------------------------------

export function VaeLatentSpaceWidget({ width: widthOverride }: WidgetProps) {
  const { containerRef, width: measuredWidth } = useContainerWidth(600)
  const width = widthOverride ?? measuredWidth

  const [mode, setMode] = useState<Mode>('ae')
  const [beta, setBeta] = useState(1.0)
  const [sampledPoints, setSampledPoints] = useState<SampledPoint[]>([])
  const [sampleIndex, setSampleIndex] = useState(0)
  const [hoveredPoint, setHoveredPoint] = useState<LatentPoint | null>(null)
  const [selectedSample, setSelectedSample] = useState<SampledPoint | null>(
    null,
  )

  // Canvas dimensions
  const canvasSize = Math.min(width - 16, 450)
  const padding = 30

  // Generate density heatmap for VAE mode
  const densityGrid = useMemo(() => {
    if (mode !== 'vae') return null

    const gridSize = 30
    const grid: number[][] = []
    const sigma = 0.4 + beta * 0.3 // Sigma grows with beta

    for (let gy = 0; gy < gridSize; gy++) {
      const row: number[] = []
      for (let gx = 0; gx < gridSize; gx++) {
        const lx = (gx / (gridSize - 1)) * SPACE_RANGE * 2 - SPACE_RANGE
        const ly = (gy / (gridSize - 1)) * SPACE_RANGE * 2 - SPACE_RANGE
        let density = 0
        for (const p of ENCODED_POINTS) {
          density += getGaussianOpacity(lx, ly, p.x, p.y, sigma)
        }
        row.push(Math.min(1, density * 0.15))
      }
      grid.push(row)
    }
    return grid
  }, [mode, beta])

  const handleSampleRandom = useCallback(() => {
    const preset = PRESET_SAMPLES[sampleIndex % PRESET_SAMPLES.length]
    const pixels = getDecodedSample(
      preset.x,
      preset.y,
      mode,
      beta,
    )
    const newSample: SampledPoint = { x: preset.x, y: preset.y, pixels }
    setSampledPoints((prev) => [...prev.slice(-4), newSample])
    setSelectedSample(newSample)
    setSampleIndex((prev) => prev + 1)
  }, [sampleIndex, mode, beta])

  const handleCanvasClick = useCallback(
    (e: React.MouseEvent<SVGSVGElement>) => {
      const rect = e.currentTarget.getBoundingClientRect()
      const svgX = e.clientX - rect.left
      const svgY = e.clientY - rect.top

      const lx = svgToLatent(svgX, canvasSize, padding)
      const ly = svgToLatent(svgY, canvasSize, padding)

      // Clamp to space range
      const clampedX = Math.max(
        -SPACE_RANGE + 0.1,
        Math.min(SPACE_RANGE - 0.1, lx),
      )
      const clampedY = Math.max(
        -SPACE_RANGE + 0.1,
        Math.min(SPACE_RANGE - 0.1, ly),
      )

      const pixels = getDecodedSample(clampedX, clampedY, mode, beta)
      const newSample: SampledPoint = {
        x: clampedX,
        y: clampedY,
        pixels,
      }
      setSampledPoints((prev) => [...prev.slice(-4), newSample])
      setSelectedSample(newSample)
    },
    [canvasSize, mode, beta],
  )

  const handleReset = useCallback(() => {
    setSampledPoints([])
    setSelectedSample(null)
    setSampleIndex(0)
  }, [])

  return (
    <div ref={containerRef} className="space-y-4">
      {/* Mode toggle */}
      <div className="flex items-center justify-center gap-2">
        <button
          onClick={() => {
            setMode('ae')
            handleReset()
          }}
          className={cn(
            'px-4 py-2 rounded-lg text-sm font-medium transition-colors cursor-pointer',
            mode === 'ae'
              ? 'bg-rose-500/20 text-rose-300 border border-rose-500/30'
              : 'bg-muted text-muted-foreground hover:bg-muted/80',
          )}
        >
          Autoencoder
        </button>
        <button
          onClick={() => {
            setMode('vae')
            handleReset()
          }}
          className={cn(
            'px-4 py-2 rounded-lg text-sm font-medium transition-colors cursor-pointer',
            mode === 'vae'
              ? 'bg-violet-500/20 text-violet-300 border border-violet-500/30'
              : 'bg-muted text-muted-foreground hover:bg-muted/80',
          )}
        >
          VAE
        </button>
      </div>

      <Legend />

      {/* Main visualization */}
      <div className="flex flex-col items-center gap-3">
        <svg
          width={canvasSize}
          height={canvasSize}
          viewBox={`0 0 ${canvasSize} ${canvasSize}`}
          className="rounded-lg border border-border/50 cursor-crosshair"
          style={{ background: '#0a0a12' }}
          onClick={handleCanvasClick}
        >
          {/* Density heatmap in VAE mode */}
          {mode === 'vae' && densityGrid && (() => {
            const gridSize = densityGrid.length
            const cellW = (canvasSize - padding * 2) / gridSize
            const cellH = (canvasSize - padding * 2) / gridSize
            return densityGrid.flatMap((row, gy) =>
              row.map((density, gx) => {
                if (density < 0.01) return null
                return (
                  <rect
                    key={`d-${gy}-${gx}`}
                    x={padding + gx * cellW}
                    y={padding + gy * cellH}
                    width={cellW + 0.5}
                    height={cellH + 0.5}
                    fill={`rgba(139, 92, 246, ${density * 0.5})`}
                  />
                )
              }),
            )
          })()}

          {/* Grid lines */}
          {[-2, -1, 0, 1, 2].map((v) => {
            const pos = latentToSvg(v, canvasSize, padding)
            return (
              <g key={`grid-${v}`}>
                <line
                  x1={pos}
                  y1={padding}
                  x2={pos}
                  y2={canvasSize - padding}
                  stroke="rgba(255,255,255,0.06)"
                  strokeWidth={v === 0 ? 1 : 0.5}
                />
                <line
                  x1={padding}
                  y1={pos}
                  x2={canvasSize - padding}
                  y2={pos}
                  stroke="rgba(255,255,255,0.06)"
                  strokeWidth={v === 0 ? 1 : 0.5}
                />
                {/* Axis labels */}
                <text
                  x={pos}
                  y={canvasSize - padding + 16}
                  textAnchor="middle"
                  fill="rgba(255,255,255,0.3)"
                  fontSize={9}
                >
                  {v}
                </text>
                <text
                  x={padding - 12}
                  y={pos + 3}
                  textAnchor="middle"
                  fill="rgba(255,255,255,0.3)"
                  fontSize={9}
                >
                  {-v}
                </text>
              </g>
            )
          })}

          {/* Gaussian clouds in VAE mode */}
          {mode === 'vae' &&
            ENCODED_POINTS.map((point) => {
              const cx = latentToSvg(point.x, canvasSize, padding)
              const cy = latentToSvg(point.y, canvasSize, padding)
              const cloudRadius =
                ((0.3 + beta * 0.25) / (SPACE_RANGE * 2)) *
                (canvasSize - padding * 2)
              return (
                <circle
                  key={`cloud-${point.id}`}
                  cx={cx}
                  cy={cy}
                  r={cloudRadius}
                  fill={CATEGORY_COLORS[point.category]}
                  opacity={0.08 + beta * 0.03}
                />
              )
            })}

          {/* Encoded data points */}
          {ENCODED_POINTS.map((point) => {
            const cx = latentToSvg(point.x, canvasSize, padding)
            const cy = latentToSvg(point.y, canvasSize, padding)
            const isHovered = hoveredPoint?.id === point.id
            return (
              <circle
                key={point.id}
                cx={cx}
                cy={cy}
                r={isHovered ? POINT_RADIUS + 2 : POINT_RADIUS}
                fill={CATEGORY_COLORS[point.category]}
                opacity={isHovered ? 1 : 0.8}
                stroke={isHovered ? 'white' : 'none'}
                strokeWidth={isHovered ? 1.5 : 0}
                className="cursor-pointer"
                onMouseEnter={() => setHoveredPoint(point)}
                onMouseLeave={() => setHoveredPoint(null)}
              />
            )
          })}

          {/* Sampled points */}
          {sampledPoints.map((sp, idx) => {
            const cx = latentToSvg(sp.x, canvasSize, padding)
            const cy = latentToSvg(sp.y, canvasSize, padding)
            const isSelected = selectedSample === sp
            return (
              <g key={`sample-${idx}`}>
                <circle
                  cx={cx}
                  cy={cy}
                  r={isSelected ? 7 : 5}
                  fill="none"
                  stroke={isSelected ? '#fbbf24' : 'rgba(251, 191, 36, 0.6)'}
                  strokeWidth={isSelected ? 2 : 1.5}
                  strokeDasharray="3,2"
                  className="cursor-pointer"
                  onClick={(e) => {
                    e.stopPropagation()
                    setSelectedSample(sp)
                  }}
                />
                <line
                  x1={cx - 3}
                  y1={cy}
                  x2={cx + 3}
                  y2={cy}
                  stroke={isSelected ? '#fbbf24' : 'rgba(251, 191, 36, 0.6)'}
                  strokeWidth={1}
                />
                <line
                  x1={cx}
                  y1={cy - 3}
                  x2={cx}
                  y2={cy + 3}
                  stroke={isSelected ? '#fbbf24' : 'rgba(251, 191, 36, 0.6)'}
                  strokeWidth={1}
                />
              </g>
            )
          })}

          {/* Axis labels */}
          <text
            x={canvasSize / 2}
            y={canvasSize - 4}
            textAnchor="middle"
            fill="rgba(255,255,255,0.4)"
            fontSize={10}
          >
            z₁
          </text>
          <text
            x={8}
            y={canvasSize / 2}
            textAnchor="middle"
            fill="rgba(255,255,255,0.4)"
            fontSize={10}
            transform={`rotate(-90, 8, ${canvasSize / 2})`}
          >
            z₂
          </text>
        </svg>

        <p className="text-[10px] text-muted-foreground/60">
          Click anywhere in the latent space to sample a point
        </p>
      </div>

      {/* Beta slider (VAE only) */}
      {mode === 'vae' && (
        <div className="space-y-2 px-4">
          <div className="flex items-center justify-between">
            <label className="text-xs font-medium text-muted-foreground">
              KL Weight (&beta;)
            </label>
            <span className="text-sm font-mono font-bold text-violet-400">
              {beta.toFixed(1)}
            </span>
          </div>
          <input
            type="range"
            min={0}
            max={50}
            value={beta * 10}
            onChange={(e) => {
              setBeta(Number(e.target.value) / 10)
              handleReset()
            }}
            className="w-full accent-violet-500 cursor-pointer"
          />
          <div className="flex justify-between text-[10px] text-muted-foreground/60">
            <span>0.0 (no KL)</span>
            <span>1.0</span>
            <span>2.5</span>
            <span>5.0 (strong KL)</span>
          </div>
        </div>
      )}

      {/* Controls */}
      <div className="flex items-center justify-center gap-3">
        <button
          onClick={handleSampleRandom}
          className="px-4 py-2 rounded-lg text-sm font-medium bg-amber-500/20 text-amber-300 border border-amber-500/30 hover:bg-amber-500/30 transition-colors cursor-pointer"
        >
          Sample Random Point
        </button>
        {sampledPoints.length > 0 && (
          <button
            onClick={handleReset}
            className="px-3 py-2 rounded-lg text-xs font-medium bg-muted text-muted-foreground hover:bg-muted/80 transition-colors cursor-pointer"
          >
            Clear Samples
          </button>
        )}
      </div>

      {/* Decoded image preview */}
      {(hoveredPoint || selectedSample) && (
        <div className="flex items-center justify-center gap-4 py-2 px-4 bg-muted/30 rounded-lg">
          {hoveredPoint && (
            <div className="text-center">
              <PixelPreview
                pixels={hoveredPoint.pixels}
                label={hoveredPoint.label}
              />
              <p className="text-[9px] text-muted-foreground/60 mt-1">
                ({hoveredPoint.x.toFixed(1)}, {hoveredPoint.y.toFixed(1)})
              </p>
            </div>
          )}
          {selectedSample && (
            <div className="text-center">
              <PixelPreview
                pixels={selectedSample.pixels}
                label="Sampled"
              />
              <p className="text-[9px] text-muted-foreground/60 mt-1">
                ({selectedSample.x.toFixed(1)}, {selectedSample.y.toFixed(1)})
              </p>
            </div>
          )}
        </div>
      )}

      {/* Stats panel */}
      <div className="flex gap-3 flex-wrap">
        <div className="flex-1 min-w-[100px] px-3 py-2 bg-muted/50 rounded-md">
          <p className="text-xs text-muted-foreground">Mode</p>
          <p
            className={cn(
              'text-sm font-medium',
              mode === 'ae' ? 'text-rose-400' : 'text-violet-400',
            )}
          >
            {getModeLabel(mode)}
          </p>
        </div>
        {mode === 'vae' && (
          <div className="flex-1 min-w-[100px] px-3 py-2 bg-muted/50 rounded-md">
            <p className="text-xs text-muted-foreground">&beta; (KL weight)</p>
            <p className="text-sm font-medium text-violet-400">
              {beta.toFixed(1)}
            </p>
          </div>
        )}
        <div className="flex-1 min-w-[100px] px-3 py-2 bg-muted/50 rounded-md">
          <p className="text-xs text-muted-foreground">Sampled Points</p>
          <p className="text-sm font-medium text-amber-400">
            {sampledPoints.length}
          </p>
        </div>
      </div>

      {/* Insight text */}
      <div className="px-3 py-2 bg-muted/30 rounded-lg text-center">
        <p className="text-xs text-muted-foreground">
          {getInsightText(mode, beta)}
        </p>
      </div>
    </div>
  )
}

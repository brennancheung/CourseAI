'use client'

import { useState, useMemo, useCallback } from 'react'
import { useContainerWidth } from '@/hooks/useContainerWidth'
import { cn } from '@/lib/utils'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type WidgetProps = {
  width?: number
  height?: number
  /** Which noise schedule to display: 'cosine' or 'linear' */
  schedule?: 'cosine' | 'linear'
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const IMG_SIZE = 28
const TOTAL_STEPS = 1000
const CURVE_HEIGHT = 200
const CURVE_PADDING_LEFT = 48
const CURVE_PADDING_RIGHT = 16
const CURVE_PADDING_TOP = 16
const CURVE_PADDING_BOTTOM = 32

// ---------------------------------------------------------------------------
// Noise schedules
// ---------------------------------------------------------------------------

function getAlphaBarCosine(t: number): number {
  const s = 0.008
  const f = Math.cos(((t / TOTAL_STEPS + s) / (1 + s)) * Math.PI * 0.5)
  return f * f
}

function getAlphaBarLinear(t: number): number {
  // Linear beta schedule: beta_1 = 0.0001, beta_T = 0.02
  // alpha_bar_t = product of (1 - beta_i) for i = 1..t
  const betaStart = 0.0001
  const betaEnd = 0.02
  let alphaBar = 1
  for (let i = 1; i <= t; i++) {
    const beta = betaStart + (betaEnd - betaStart) * ((i - 1) / (TOTAL_STEPS - 1))
    alphaBar *= 1 - beta
  }
  return alphaBar
}

function getAlphaBar(t: number, schedule: 'cosine' | 'linear'): number {
  if (t === 0) return 1
  if (schedule === 'cosine') return getAlphaBarCosine(t)
  return getAlphaBarLinear(t)
}

// ---------------------------------------------------------------------------
// Procedural image generation (matches DiffusionNoiseWidget)
// ---------------------------------------------------------------------------

function generateBaseImage(): number[] {
  const pixels = new Array<number>(IMG_SIZE * IMG_SIZE).fill(0)
  for (let y = 0; y < IMG_SIZE; y++) {
    for (let x = 0; x < IMG_SIZE; x++) {
      const idx = y * IMG_SIZE + x
      const inBody = x >= 7 && x <= 20 && y >= 8 && y <= 24
      const inSleeves = x >= 3 && x <= 24 && y >= 8 && y <= 13
      const inNeck =
        x >= 11 && x <= 16 && y >= 7 && y <= 10 && y - 7 < Math.abs(x - 13.5) * 0.8
      const isCollar = x >= 10 && x <= 17 && y === 7
      if (isCollar) {
        pixels[idx] = 220
      } else if ((inBody || inSleeves) && !inNeck) {
        const centerDist = Math.abs(x - 13.5) / 10
        pixels[idx] = Math.max(140, 200 - centerDist * 40)
      }
    }
  }
  return pixels
}

// ---------------------------------------------------------------------------
// Deterministic seeded PRNG
// ---------------------------------------------------------------------------

function mulberry32(seed: number): () => number {
  let a = seed | 0
  return () => {
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

function gaussianNoise(rng: () => number): number {
  const u1 = rng()
  const u2 = rng()
  return Math.sqrt(-2 * Math.log(Math.max(u1, 1e-10))) * Math.cos(2 * Math.PI * u2)
}

function applyNoise(
  basePixels: number[],
  alphaBar: number,
  seed: number,
): number[] {
  if (alphaBar >= 1) return basePixels
  const signalCoeff = Math.sqrt(alphaBar)
  const noiseCoeff = Math.sqrt(1 - alphaBar)
  const rng = mulberry32(seed)
  return basePixels.map((pixel) => {
    const normalized = pixel / 255
    const noise = gaussianNoise(rng)
    const noisy = signalCoeff * normalized + noiseCoeff * noise
    return Math.max(0, Math.min(255, noisy * 255))
  })
}

// ---------------------------------------------------------------------------
// Pixel canvas (small inline SVG image)
// ---------------------------------------------------------------------------

function PixelCanvas({
  pixels,
  scale,
  highlight = false,
}: {
  pixels: number[]
  scale: number
  highlight?: boolean
}) {
  const canvasSize = IMG_SIZE * scale
  return (
    <svg
      width={canvasSize}
      height={canvasSize}
      viewBox={`0 0 ${canvasSize} ${canvasSize}`}
      className={cn(
        'rounded border',
        highlight ? 'border-violet-500/60' : 'border-border/50',
      )}
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
  )
}

// ---------------------------------------------------------------------------
// Curve rendering helpers
// ---------------------------------------------------------------------------

function buildCurvePath(
  schedule: 'cosine' | 'linear',
  plotWidth: number,
  plotHeight: number,
): string {
  const points: string[] = []
  const numSamples = 200
  for (let i = 0; i <= numSamples; i++) {
    const t = (i / numSamples) * TOTAL_STEPS
    const ab = getAlphaBar(Math.round(t), schedule)
    const px = CURVE_PADDING_LEFT + (i / numSamples) * plotWidth
    const py = CURVE_PADDING_TOP + (1 - ab) * plotHeight
    points.push(`${i === 0 ? 'M' : 'L'}${px.toFixed(1)},${py.toFixed(1)}`)
  }
  return points.join(' ')
}

// ---------------------------------------------------------------------------
// Main widget
// ---------------------------------------------------------------------------

export function AlphaBarCurveWidget({
  width: widthOverride,
  schedule: scheduleProp = 'cosine',
}: WidgetProps) {
  const { containerRef, width: measuredWidth } = useContainerWidth(600)
  const width = widthOverride ?? measuredWidth

  const [step, setStep] = useState(0)
  const [schedule, setSchedule] = useState<'cosine' | 'linear'>(scheduleProp)
  const NOISE_SEED = 42

  const basePixels = useMemo(() => generateBaseImage(), [])

  const alphaBar = getAlphaBar(step, schedule)
  const signalCoeff = Math.sqrt(alphaBar)
  const noiseCoeff = Math.sqrt(1 - alphaBar)

  const currentPixels = useMemo(
    () => applyNoise(basePixels, alphaBar, NOISE_SEED),
    [basePixels, alphaBar],
  )

  // SVG curve dimensions
  const svgWidth = Math.max(300, width - 16)
  const plotWidth = svgWidth - CURVE_PADDING_LEFT - CURVE_PADDING_RIGHT
  const plotHeight = CURVE_HEIGHT - CURVE_PADDING_TOP - CURVE_PADDING_BOTTOM

  const curvePath = useMemo(
    () => buildCurvePath(schedule, plotWidth, plotHeight),
    [schedule, plotWidth, plotHeight],
  )

  // Marker position on the curve
  const markerX = CURVE_PADDING_LEFT + (step / TOTAL_STEPS) * plotWidth
  const markerY = CURVE_PADDING_TOP + (1 - alphaBar) * plotHeight

  // Handle click/drag on the SVG to select a timestep
  const handleCurveInteraction = useCallback(
    (e: React.MouseEvent<SVGSVGElement>) => {
      const svg = e.currentTarget
      const rect = svg.getBoundingClientRect()
      const x = e.clientX - rect.left
      const fraction = Math.max(
        0,
        Math.min(1, (x - CURVE_PADDING_LEFT) / plotWidth),
      )
      const t = Math.round(fraction * TOTAL_STEPS)
      setStep(Math.max(0, Math.min(TOTAL_STEPS, t)))
    },
    [plotWidth],
  )

  const [isDragging, setIsDragging] = useState(false)

  const handleMouseDown = useCallback(
    (e: React.MouseEvent<SVGSVGElement>) => {
      setIsDragging(true)
      handleCurveInteraction(e)
    },
    [handleCurveInteraction],
  )

  const handleMouseMove = useCallback(
    (e: React.MouseEvent<SVGSVGElement>) => {
      if (!isDragging) return
      handleCurveInteraction(e)
    },
    [isDragging, handleCurveInteraction],
  )

  const handleMouseUp = useCallback(() => {
    setIsDragging(false)
  }, [])

  const handleSliderChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      setStep(Number(e.target.value))
    },
    [],
  )

  // Image display scale
  const imgScale = width < 500 ? 3 : 4

  return (
    <div ref={containerRef} className="space-y-4">
      {/* Alpha-bar curve */}
      <div className="flex flex-col items-center">
        <svg
          width={svgWidth}
          height={CURVE_HEIGHT}
          className="cursor-crosshair select-none"
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
        >
          {/* Background */}
          <rect width={svgWidth} height={CURVE_HEIGHT} fill="transparent" />

          {/* Grid lines */}
          {[0, 0.25, 0.5, 0.75, 1].map((v) => {
            const y = CURVE_PADDING_TOP + (1 - v) * plotHeight
            return (
              <g key={v}>
                <line
                  x1={CURVE_PADDING_LEFT}
                  y1={y}
                  x2={CURVE_PADDING_LEFT + plotWidth}
                  y2={y}
                  stroke="currentColor"
                  strokeOpacity={0.1}
                  strokeWidth={1}
                />
                <text
                  x={CURVE_PADDING_LEFT - 6}
                  y={y + 3}
                  textAnchor="end"
                  className="fill-muted-foreground"
                  fontSize={10}
                >
                  {v.toFixed(2)}
                </text>
              </g>
            )
          })}

          {/* X-axis labels */}
          {[0, 250, 500, 750, 1000].map((t) => {
            const x = CURVE_PADDING_LEFT + (t / TOTAL_STEPS) * plotWidth
            return (
              <text
                key={t}
                x={x}
                y={CURVE_HEIGHT - 4}
                textAnchor="middle"
                className="fill-muted-foreground"
                fontSize={10}
              >
                {t}
              </text>
            )
          })}

          {/* Axis labels */}
          <text
            x={CURVE_PADDING_LEFT + plotWidth / 2}
            y={CURVE_HEIGHT - 16}
            textAnchor="middle"
            className="fill-muted-foreground"
            fontSize={10}
            opacity={0.6}
          >
            timestep t
          </text>
          <text
            x={12}
            y={CURVE_PADDING_TOP + plotHeight / 2}
            textAnchor="middle"
            className="fill-muted-foreground"
            fontSize={10}
            opacity={0.6}
            transform={`rotate(-90, 12, ${CURVE_PADDING_TOP + plotHeight / 2})`}
          >
            &#x3B1;&#x0304;&#x209C;
          </text>

          {/* Vertical line from marker to x-axis */}
          <line
            x1={markerX}
            y1={markerY}
            x2={markerX}
            y2={CURVE_PADDING_TOP + plotHeight}
            stroke="currentColor"
            strokeOpacity={0.15}
            strokeWidth={1}
            strokeDasharray="3,3"
          />

          {/* Horizontal line from marker to y-axis */}
          <line
            x1={CURVE_PADDING_LEFT}
            y1={markerY}
            x2={markerX}
            y2={markerY}
            stroke="currentColor"
            strokeOpacity={0.15}
            strokeWidth={1}
            strokeDasharray="3,3"
          />

          {/* The alpha_bar curve */}
          <path
            d={curvePath}
            fill="none"
            stroke="#8b5cf6"
            strokeWidth={2.5}
            strokeLinecap="round"
          />

          {/* Marker dot */}
          <circle
            cx={markerX}
            cy={markerY}
            r={7}
            fill="#8b5cf6"
            stroke="white"
            strokeWidth={2}
            className="drop-shadow-md"
          />

          {/* Marker value label */}
          <text
            x={markerX}
            y={markerY - 12}
            textAnchor="middle"
            className="fill-foreground font-mono"
            fontSize={11}
            fontWeight="bold"
          >
            {alphaBar.toFixed(3)}
          </text>
        </svg>
      </div>

      {/* Slider for fine control */}
      <div className="space-y-1 px-2">
        <div className="flex items-center justify-between">
          <label className="text-xs font-medium text-muted-foreground">
            Timestep
          </label>
          <span className="text-sm font-mono font-bold text-violet-400">
            t = {step}
          </span>
        </div>
        <input
          type="range"
          min={0}
          max={TOTAL_STEPS}
          step={1}
          value={step}
          onChange={handleSliderChange}
          className="w-full accent-violet-500 cursor-ew-resize"
        />
      </div>

      {/* Image preview + formula coefficients */}
      <div className="flex items-start gap-6 px-2 flex-wrap justify-center">
        {/* Image at current timestep */}
        <div className="flex flex-col items-center gap-1.5">
          <PixelCanvas pixels={currentPixels} scale={imgScale} highlight />
          <span className="text-[10px] font-mono text-muted-foreground">
            x_{'{'}t{'}'} at t={step}
          </span>
        </div>

        {/* Live coefficients */}
        <div className="flex-1 min-w-[200px] space-y-3">
          <div className="rounded-lg bg-muted/30 p-3 space-y-2 font-mono text-sm">
            <p className="text-muted-foreground text-xs font-sans font-medium mb-2">
              Closed-form formula
            </p>
            <div className="space-y-1.5">
              <div className="flex items-center gap-2">
                <span className="text-emerald-400 font-bold w-[120px] text-right">
                  {signalCoeff.toFixed(4)}
                </span>
                <span className="text-muted-foreground text-xs">
                  &times; x&#x2080;
                </span>
                <span className="text-muted-foreground/50 text-xs ml-auto">
                  &radic;&#x3B1;&#x0304;&#x209C;
                </span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-rose-400 font-bold w-[120px] text-right">
                  {noiseCoeff.toFixed(4)}
                </span>
                <span className="text-muted-foreground text-xs">
                  &times; &epsilon;
                </span>
                <span className="text-muted-foreground/50 text-xs ml-auto">
                  &radic;(1&minus;&#x3B1;&#x0304;&#x209C;)
                </span>
              </div>
            </div>
            <div className="border-t border-border/30 pt-2 mt-2">
              <div className="flex items-center gap-2 text-xs text-muted-foreground">
                <span>&#x3B1;&#x0304;&#x209C; =</span>
                <span className="font-bold text-violet-400">
                  {alphaBar.toFixed(4)}
                </span>
                <span className="text-muted-foreground/50 ml-auto">
                  signal fraction remaining
                </span>
              </div>
            </div>
          </div>

          {/* Signal vs noise bar */}
          <div className="space-y-1">
            <div className="flex items-center justify-between text-[10px] text-muted-foreground">
              <span className="text-emerald-400">
                Signal: {Math.round(alphaBar * 100)}%
              </span>
              <span className="text-rose-400">
                Noise: {Math.round((1 - alphaBar) * 100)}%
              </span>
            </div>
            <div className="h-2 rounded-full bg-muted/50 overflow-hidden flex">
              <div
                className="h-full bg-emerald-500/70 transition-all duration-100"
                style={{ width: `${alphaBar * 100}%` }}
              />
              <div
                className="h-full bg-rose-500/70 transition-all duration-100"
                style={{ width: `${(1 - alphaBar) * 100}%` }}
              />
            </div>
          </div>
        </div>
      </div>

      {/* Schedule toggle */}
      <div className="flex items-center justify-center gap-2 px-2">
        <span className="text-xs text-muted-foreground">Schedule:</span>
        <button
          onClick={() => setSchedule('cosine')}
          className={cn(
            'px-3 py-1 rounded text-xs font-medium transition-colors cursor-pointer',
            schedule === 'cosine'
              ? 'bg-violet-500/20 text-violet-400 ring-1 ring-violet-500/40'
              : 'text-muted-foreground hover:bg-muted/50',
          )}
        >
          Cosine
        </button>
        <button
          onClick={() => setSchedule('linear')}
          className={cn(
            'px-3 py-1 rounded text-xs font-medium transition-colors cursor-pointer',
            schedule === 'linear'
              ? 'bg-violet-500/20 text-violet-400 ring-1 ring-violet-500/40'
              : 'text-muted-foreground hover:bg-muted/50',
          )}
        >
          Linear
        </button>
      </div>

      {/* Helpful description */}
      <div className="px-3 py-2 bg-muted/30 rounded-lg">
        <p className="text-xs text-muted-foreground">
          {getDescription(step, alphaBar)}
        </p>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Descriptions
// ---------------------------------------------------------------------------

function getDescription(t: number, alphaBar: number): string {
  if (t === 0)
    return 'Clean image. Alpha-bar is 1 \u2014 the signal coefficient is 1 and the noise coefficient is 0. The formula gives back the original image unchanged.'
  if (alphaBar > 0.9)
    return 'Very early timestep. Alpha-bar is close to 1 \u2014 nearly all signal, almost no noise. The image is barely changed.'
  if (alphaBar > 0.5)
    return 'Alpha-bar above 0.5 \u2014 the signal term still dominates. You can clearly see the original image through the noise.'
  if (alphaBar > 0.3)
    return 'Alpha-bar between 0.3 and 0.5 \u2014 signal and noise are roughly balanced. The image is recognizable but degraded.'
  if (alphaBar > 0.1)
    return 'Alpha-bar below 0.3 \u2014 the noise term dominates. Hard to make out the original image.'
  if (alphaBar > 0.01)
    return 'Alpha-bar near zero \u2014 almost pure noise. The signal coefficient is tiny. Almost nothing of the original remains.'
  return 'Effectively pure noise. Alpha-bar is essentially zero \u2014 the formula produces pure Gaussian noise regardless of the input image.'
}

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
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const IMG_SIZE = 28
const DISPLAY_SCALE = 4
const TOTAL_STEPS = 1000
const PREVIEW_STEPS = [0, 100, 250, 400, 550, 700, 850, 1000]

// ---------------------------------------------------------------------------
// Procedural "image" generation
// ---------------------------------------------------------------------------

/**
 * Generate a simple procedural image — a recognizable T-shirt-like silhouette
 * on a dark background. Returns an array of 28*28 pixel values in [0, 255].
 */
function generateBaseImage(): number[] {
  const pixels = new Array<number>(IMG_SIZE * IMG_SIZE).fill(0)

  for (let y = 0; y < IMG_SIZE; y++) {
    for (let x = 0; x < IMG_SIZE; x++) {
      const idx = y * IMG_SIZE + x

      // Body: rectangular torso area
      const inBody = x >= 7 && x <= 20 && y >= 8 && y <= 24
      // Sleeves: wider area at the top
      const inSleeves = x >= 3 && x <= 24 && y >= 8 && y <= 13
      // Neckline: cut out a small V at the top center
      const inNeck = x >= 11 && x <= 16 && y >= 7 && y <= 10 && (y - 7) < (Math.abs(x - 13.5) * 0.8)
      // Collar: slight highlight at neckline edge
      const isCollar = x >= 10 && x <= 17 && y === 7

      if (isCollar) {
        pixels[idx] = 220
      } else if ((inBody || inSleeves) && !inNeck) {
        // Add subtle shading for depth
        const centerDist = Math.abs(x - 13.5) / 10
        const brightness = 200 - centerDist * 40
        pixels[idx] = Math.max(140, brightness)
      }
    }
  }

  return pixels
}

// ---------------------------------------------------------------------------
// Noise application (deterministic seeded PRNG for consistency)
// ---------------------------------------------------------------------------

/**
 * Simple mulberry32 PRNG for deterministic noise.
 * Returns a function that produces the next random number in [0, 1).
 */
function mulberry32(seed: number): () => number {
  let a = seed | 0
  return () => {
    a = (a + 0x6D2B79F5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

/**
 * Generate a pair of uniform randoms and convert to Gaussian via Box-Muller.
 */
function gaussianNoise(rng: () => number): number {
  const u1 = rng()
  const u2 = rng()
  return Math.sqrt(-2 * Math.log(Math.max(u1, 1e-10))) * Math.cos(2 * Math.PI * u2)
}

/**
 * Apply forward process noise at a given timestep.
 *
 * In the real forward process, x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps
 * We approximate alpha_bar_t with a simple cosine schedule mapped to [0, 1].
 */
function getAlphaBar(t: number): number {
  // Cosine schedule: starts at 1 (clean), ends at ~0 (pure noise)
  const s = 0.008 // small offset to avoid division by zero
  const f = Math.cos(((t / TOTAL_STEPS + s) / (1 + s)) * Math.PI * 0.5)
  return f * f
}

function applyNoise(basePixels: number[], t: number, seed: number): number[] {
  if (t === 0) return basePixels

  const alphaBar = getAlphaBar(t)
  const signalCoeff = Math.sqrt(alphaBar)
  const noiseCoeff = Math.sqrt(1 - alphaBar)

  const rng = mulberry32(seed)
  return basePixels.map((pixel) => {
    // Normalize pixel to [0, 1] range
    const normalized = pixel / 255
    const noise = gaussianNoise(rng)
    const noisy = signalCoeff * normalized + noiseCoeff * noise
    // Clamp and scale back to [0, 255]
    return Math.max(0, Math.min(255, noisy * 255))
  })
}

// ---------------------------------------------------------------------------
// Helper: noise level description
// ---------------------------------------------------------------------------

function getNoiseDescription(t: number): string {
  if (t === 0) return 'Clean image. No noise added. This is the starting point.'
  if (t <= 100) return 'Very slight noise. The image is almost unchanged — you could easily denoise this.'
  if (t <= 250) return 'Light noise. Global structure is clear, fine details are starting to blur.'
  if (t <= 400) return 'Moderate noise. You can still tell what the image is, but details are lost.'
  if (t <= 550) return 'Heavy noise. The shape is barely visible. Hard to be sure what it was.'
  if (t <= 700) return 'Very heavy noise. Almost impossible to identify the original image.'
  if (t <= 850) return 'Extreme noise. Essentially random static with perhaps a faint ghost of structure.'
  return 'Pure noise. No trace of the original image remains. Any image could be hiding under this.'
}

function getStepLabel(t: number): string {
  if (t === 0) return 'Clean (t=0)'
  if (t === TOTAL_STEPS) return 'Pure noise (t=T)'
  return `t=${t}`
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function PixelCanvas({
  pixels,
  scale = DISPLAY_SCALE,
  highlight = false,
}: {
  pixels: number[]
  scale?: number
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
        highlight ? 'border-amber-500/60' : 'border-border/50',
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
// Main widget
// ---------------------------------------------------------------------------

export function DiffusionNoiseWidget({ width: widthOverride }: WidgetProps) {
  const { containerRef, width: measuredWidth } = useContainerWidth(600)
  const width = widthOverride ?? measuredWidth

  const [step, setStep] = useState(0)
  const NOISE_SEED = 42

  // Generate base image once
  const basePixels = useMemo(() => generateBaseImage(), [])

  // Compute noisy image for the current slider position
  const currentPixels = useMemo(
    () => applyNoise(basePixels, step, NOISE_SEED),
    [basePixels, step],
  )

  // Compute the preview strip at fixed timesteps
  const previewStrip = useMemo(
    () =>
      PREVIEW_STEPS.map((t) => ({
        step: t,
        pixels: applyNoise(basePixels, t, NOISE_SEED),
      })),
    [basePixels],
  )

  const handleSliderChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    setStep(Number(e.target.value))
  }, [])

  // Compute the progress fraction for the signal vs noise indicator
  const alphaBar = getAlphaBar(step)
  const signalPct = Math.round(alphaBar * 100)
  const noisePct = 100 - signalPct

  // Determine small preview scale based on available width
  const previewScale = width < 500 ? 2 : 3

  return (
    <div ref={containerRef} className="space-y-5">
      {/* Main display: current noisy image */}
      <div className="flex flex-col items-center gap-3">
        <PixelCanvas pixels={currentPixels} scale={Math.min(6, Math.floor((width - 32) / IMG_SIZE))} highlight />
        <p className="text-xs text-muted-foreground font-mono">
          {getStepLabel(step)}
        </p>
      </div>

      {/* Slider */}
      <div className="space-y-2 px-2">
        <div className="flex items-center justify-between">
          <label className="text-xs font-medium text-muted-foreground">
            Noise Level (timestep)
          </label>
          <span className="text-sm font-mono font-bold text-amber-400">
            t = {step}
          </span>
        </div>
        <input
          type="range"
          min={0}
          max={TOTAL_STEPS}
          step={10}
          value={step}
          onChange={handleSliderChange}
          className="w-full accent-amber-500 cursor-ew-resize"
        />
        <div className="flex justify-between text-[10px] text-muted-foreground/60">
          <span>Clean (t=0)</span>
          <span>t=250</span>
          <span>t=500</span>
          <span>t=750</span>
          <span>Pure noise (t=1000)</span>
        </div>
      </div>

      {/* Signal vs Noise indicator */}
      <div className="space-y-2 px-2">
        <div className="flex items-center justify-between text-[10px] text-muted-foreground">
          <span>Signal: {signalPct}%</span>
          <span>Noise: {noisePct}%</span>
        </div>
        <div className="h-2 rounded-full bg-muted/50 overflow-hidden flex">
          <div
            className="h-full bg-emerald-500/70 transition-all duration-150"
            style={{ width: `${signalPct}%` }}
          />
          <div
            className="h-full bg-rose-500/70 transition-all duration-150"
            style={{ width: `${noisePct}%` }}
          />
        </div>
      </div>

      {/* Description */}
      <div className="px-3 py-2 bg-muted/30 rounded-lg">
        <p className="text-xs text-muted-foreground">
          {getNoiseDescription(step)}
        </p>
      </div>

      {/* Preview strip: forward process at fixed timesteps */}
      <div className="space-y-2">
        <p className="text-xs font-medium text-muted-foreground px-2">
          The forward process: clean image &rarr; pure noise
        </p>
        <div className="flex justify-center gap-1 overflow-x-auto px-2">
          {previewStrip.map(({ step: t, pixels }) => (
            <button
              key={t}
              onClick={() => setStep(t)}
              className={cn(
                'flex flex-col items-center gap-1 p-1 rounded transition-colors cursor-pointer',
                step === t
                  ? 'bg-amber-500/20 ring-1 ring-amber-500/40'
                  : 'hover:bg-muted/50',
              )}
            >
              <PixelCanvas pixels={pixels} scale={previewScale} />
              <span className="text-[9px] text-muted-foreground/80 font-mono">
                {t === 0 ? 't=0' : t === TOTAL_STEPS ? 't=T' : `t=${t}`}
              </span>
            </button>
          ))}
        </div>
      </div>

      {/* Reverse direction arrow */}
      <div className="flex items-center justify-center gap-2 px-2">
        <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
          <span className="font-mono">t=0 (clean)</span>
          <span className="text-emerald-400 font-medium">&larr; the model learns to go this way &larr;</span>
          <span className="font-mono">t=T (noise)</span>
        </div>
      </div>
    </div>
  )
}

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
const TOTAL_STEPS = 1000

// ---------------------------------------------------------------------------
// Noise schedule (cosine, matching AlphaBarCurveWidget)
// ---------------------------------------------------------------------------

function getAlphaBarCosine(t: number): number {
  const s = 0.008
  const f = Math.cos(((t / TOTAL_STEPS + s) / (1 + s)) * Math.PI * 0.5)
  return f * f
}

function getAlphaBar(t: number): number {
  if (t === 0) return 1
  return getAlphaBarCosine(t)
}

// ---------------------------------------------------------------------------
// Procedural image generation (matches DiffusionNoiseWidget / AlphaBarCurve)
// ---------------------------------------------------------------------------

function generateTshirtImage(): number[] {
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

// Generate a full noise vector for the image
function generateNoiseVector(seed: number): number[] {
  const rng = mulberry32(seed)
  const noise: number[] = []
  for (let i = 0; i < IMG_SIZE * IMG_SIZE; i++) {
    noise.push(gaussianNoise(rng))
  }
  return noise
}

// Apply the closed-form formula: x_t = sqrt(alpha_bar) * x_0 + sqrt(1 - alpha_bar) * epsilon
function createNoisyImage(
  basePixels: number[],
  noiseVector: number[],
  alphaBar: number,
): number[] {
  const signalCoeff = Math.sqrt(alphaBar)
  const noiseCoeff = Math.sqrt(1 - alphaBar)
  return basePixels.map((pixel, i) => {
    const normalized = pixel / 255
    const noisy = signalCoeff * normalized + noiseCoeff * noiseVector[i]
    return Math.max(0, Math.min(255, noisy * 255))
  })
}

// Create a "predicted noise" that's slightly off from the real noise
// This simulates what a partially-trained network might output
function createPredictedNoise(
  realNoise: number[],
  errorLevel: number,
  seed: number,
): number[] {
  const rng = mulberry32(seed)
  return realNoise.map((n) => {
    const perturbation = gaussianNoise(rng) * errorLevel
    return n + perturbation
  })
}

// Compute MSE between two vectors (in normalized space)
function computeMSE(a: number[], b: number[]): number {
  let sum = 0
  for (let i = 0; i < a.length; i++) {
    const diff = a[i] - b[i]
    sum += diff * diff
  }
  return sum / a.length
}

// Render noise as a visual: map values to grayscale (centered at 128)
function noiseToPixels(noiseVector: number[]): number[] {
  return noiseVector.map((n) => {
    // Map noise from roughly [-3, 3] to [0, 255]
    const mapped = 128 + n * 42
    return Math.max(0, Math.min(255, mapped))
  })
}

// ---------------------------------------------------------------------------
// Pixel canvas (small inline SVG image)
// ---------------------------------------------------------------------------

function PixelCanvas({
  pixels,
  scale,
  borderColor = 'border-border/50',
  label,
}: {
  pixels: number[]
  scale: number
  borderColor?: string
  label?: string
}) {
  const canvasSize = IMG_SIZE * scale
  return (
    <div className="flex flex-col items-center gap-1">
      <svg
        width={canvasSize}
        height={canvasSize}
        viewBox={`0 0 ${canvasSize} ${canvasSize}`}
        className={cn('rounded border', borderColor)}
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
      {label && (
        <span className="text-[10px] font-mono text-muted-foreground">
          {label}
        </span>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Description helper
// ---------------------------------------------------------------------------

function getTrainingDescription(t: number, alphaBar: number, mse: number): string {
  if (t === 0) {
    return 'At t=0, there is no noise to predict. Move the slider to see the training algorithm in action.'
  }
  if (alphaBar > 0.9) {
    return `Easy task: at t=${t}, the image is barely noisy. The network only needs to detect subtle perturbations. MSE: ${mse.toFixed(4)}.`
  }
  if (alphaBar > 0.5) {
    return `Moderate task: at t=${t}, there is visible noise but the image structure is clear. The network can use the visible features to guide its prediction. MSE: ${mse.toFixed(4)}.`
  }
  if (alphaBar > 0.1) {
    return `Hard task: at t=${t}, noise dominates. The network must infer what structure remains from limited signal. MSE: ${mse.toFixed(4)}.`
  }
  return `Very hard task: at t=${t}, the image is nearly pure noise. The network must hallucinate plausible structure from almost nothing. MSE: ${mse.toFixed(4)}.`
}

// ---------------------------------------------------------------------------
// Preset timestep buttons
// ---------------------------------------------------------------------------

const PRESETS: { label: string; t: number }[] = [
  { label: 't = 50', t: 50 },
  { label: 't = 200', t: 200 },
  { label: 't = 500', t: 500 },
  { label: 't = 800', t: 800 },
  { label: 't = 950', t: 950 },
]

// ---------------------------------------------------------------------------
// Main widget
// ---------------------------------------------------------------------------

export function TrainingStepSimulator({
  width: widthOverride,
}: WidgetProps) {
  const { containerRef, width: measuredWidth } = useContainerWidth(600)
  const width = widthOverride ?? measuredWidth

  const [step, setStep] = useState(200)
  const NOISE_SEED = 42
  const PREDICTION_SEED = 137

  const basePixels = useMemo(() => generateTshirtImage(), [])
  const noiseVector = useMemo(() => generateNoiseVector(NOISE_SEED), [])

  const alphaBar = getAlphaBar(step)
  const signalCoeff = Math.sqrt(alphaBar)
  const noiseCoeff = Math.sqrt(1 - alphaBar)

  // The noisy image at this timestep
  const noisyPixels = useMemo(
    () => createNoisyImage(basePixels, noiseVector, alphaBar),
    [basePixels, noiseVector, alphaBar],
  )

  // Simulated network prediction (imperfect noise estimate)
  // Error decreases at easy timesteps, increases at hard ones
  const errorLevel = useMemo(() => {
    if (step === 0) return 0
    // Model is best at low noise, worst at high noise
    return 0.15 + (1 - alphaBar) * 0.35
  }, [step, alphaBar])

  const predictedNoise = useMemo(
    () => createPredictedNoise(noiseVector, errorLevel, PREDICTION_SEED + step),
    [noiseVector, errorLevel, step],
  )

  // MSE loss
  const mseLoss = useMemo(
    () => computeMSE(noiseVector, predictedNoise),
    [noiseVector, predictedNoise],
  )

  // Visual representations
  const realNoisePixels = useMemo(() => noiseToPixels(noiseVector), [noiseVector])
  const predictedNoisePixels = useMemo(
    () => noiseToPixels(predictedNoise),
    [predictedNoise],
  )

  const handleSliderChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      setStep(Number(e.target.value))
    },
    [],
  )

  // Responsive image scale
  const imgScale = width < 500 ? 2.5 : 3

  return (
    <div ref={containerRef} className="space-y-4">
      {/* Timestep selector */}
      <div className="space-y-2 px-2">
        <div className="flex items-center justify-between">
          <label className="text-xs font-medium text-muted-foreground">
            Random timestep
          </label>
          <span className="text-sm font-mono font-bold text-violet-400">
            t = {step}
          </span>
        </div>
        <input
          type="range"
          min={1}
          max={TOTAL_STEPS}
          step={1}
          value={step}
          onChange={handleSliderChange}
          className="w-full accent-violet-500 cursor-ew-resize"
        />
        <div className="flex items-center justify-center gap-1.5 flex-wrap">
          {PRESETS.map(({ label, t }) => (
            <button
              key={t}
              onClick={() => setStep(t)}
              className={cn(
                'px-2.5 py-1 rounded text-xs font-medium transition-colors cursor-pointer',
                step === t
                  ? 'bg-violet-500/20 text-violet-400 ring-1 ring-violet-500/40'
                  : 'text-muted-foreground hover:bg-muted/50',
              )}
            >
              {label}
            </button>
          ))}
        </div>
      </div>

      {/* Coefficients display */}
      <div className="rounded-lg bg-muted/30 p-3 mx-2 space-y-2">
        <p className="text-[10px] text-muted-foreground/60 font-medium uppercase tracking-wide">
          Closed-form formula coefficients
        </p>
        <div className="flex items-center gap-3 text-sm font-mono flex-wrap">
          <div className="flex items-center gap-1.5">
            <span className="text-emerald-400 font-bold">{signalCoeff.toFixed(3)}</span>
            <span className="text-muted-foreground text-xs">&middot; x&#x2080;</span>
          </div>
          <span className="text-muted-foreground">+</span>
          <div className="flex items-center gap-1.5">
            <span className="text-rose-400 font-bold">{noiseCoeff.toFixed(3)}</span>
            <span className="text-muted-foreground text-xs">&middot; &epsilon;</span>
          </div>
          <span className="text-muted-foreground text-xs ml-auto">
            &#x3B1;&#x0304; = {alphaBar.toFixed(3)}
          </span>
        </div>
        {/* Signal vs noise bar */}
        <div className="space-y-0.5">
          <div className="flex items-center justify-between text-[10px] text-muted-foreground">
            <span className="text-emerald-400">
              Signal: {Math.round(alphaBar * 100)}%
            </span>
            <span className="text-rose-400">
              Noise: {Math.round((1 - alphaBar) * 100)}%
            </span>
          </div>
          <div className="h-1.5 rounded-full bg-muted/50 overflow-hidden flex">
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

      {/* Image panels: clean -> noisy -> predicted noise vs real noise -> MSE */}
      <div className="flex items-start gap-3 px-2 flex-wrap justify-center">
        <PixelCanvas
          pixels={basePixels}
          scale={imgScale}
          borderColor="border-emerald-500/40"
          label="x\u2080 (clean)"
        />
        <div className="flex flex-col items-center justify-center self-center text-muted-foreground/50 text-lg">
          &rarr;
        </div>
        <PixelCanvas
          pixels={noisyPixels}
          scale={imgScale}
          borderColor="border-violet-500/40"
          label={`x\u209C at t=${step}`}
        />
        <div className="flex flex-col items-center justify-center self-center text-muted-foreground/50 text-lg">
          &rarr;
        </div>
        <PixelCanvas
          pixels={predictedNoisePixels}
          scale={imgScale}
          borderColor="border-sky-500/40"
          label="\u03B5\u0302 (predicted)"
        />
        <div className="flex flex-col items-center justify-center self-center text-muted-foreground/30 text-sm font-mono">
          vs
        </div>
        <PixelCanvas
          pixels={realNoisePixels}
          scale={imgScale}
          borderColor="border-rose-500/40"
          label="\u03B5 (actual)"
        />
      </div>

      {/* MSE loss display */}
      <div className="mx-2 rounded-lg bg-muted/30 p-3 flex items-center justify-between">
        <div className="space-y-0.5">
          <p className="text-xs text-muted-foreground font-medium">
            MSE Loss = ||&epsilon; &minus; &epsilon;&#x0302;||&sup2;
          </p>
          <p className="text-[10px] text-muted-foreground/60">
            Average squared error across {IMG_SIZE * IMG_SIZE} pixels
          </p>
        </div>
        <div className="text-right">
          <span className="text-xl font-mono font-bold text-amber-400">
            {mseLoss.toFixed(4)}
          </span>
        </div>
      </div>

      {/* Description */}
      <div className="px-3 py-2 bg-muted/30 rounded-lg mx-2">
        <p className="text-xs text-muted-foreground">
          {getTrainingDescription(step, alphaBar, mseLoss)}
        </p>
      </div>
    </div>
  )
}

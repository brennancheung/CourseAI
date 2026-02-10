'use client'

import { useState, useMemo, useCallback, useEffect, useRef } from 'react'
import { useContainerWidth } from '@/hooks/useContainerWidth'
import { cn } from '@/lib/utils'
import { Play, Pause, RotateCcw } from 'lucide-react'

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
const NOISE_SEED = 42

/**
 * Key timesteps to display as snapshots on the timeline.
 * From pure noise (1000) down to clean image (0).
 */
const KEY_TIMESTEPS = [1000, 900, 800, 600, 400, 200, 100, 50, 0]

/**
 * Animation timestep frames — sparser at beginning (less change),
 * denser in middle where structure emerges.
 */
const ANIMATION_FRAMES = [
  1000, 975, 950, 925, 900, 875, 850, 825, 800, 775, 750, 725, 700, 675, 650,
  625, 600, 575, 550, 525, 500, 475, 450, 425, 400, 375, 350, 325, 300, 275,
  250, 225, 200, 175, 150, 125, 100, 75, 50, 25, 0,
]

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
// Procedural image generation (matches AlphaBarCurveWidget / DiffusionNoiseWidget)
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
// Deterministic seeded PRNG (matches AlphaBarCurveWidget)
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
  label,
}: {
  pixels: number[]
  scale: number
  highlight?: boolean
  label?: string
}) {
  const canvasSize = IMG_SIZE * scale
  return (
    <div className="flex flex-col items-center gap-1">
      <svg
        width={canvasSize}
        height={canvasSize}
        viewBox={`0 0 ${canvasSize} ${canvasSize}`}
        className={cn(
          'rounded border',
          highlight ? 'border-violet-500/60 ring-1 ring-violet-500/30' : 'border-border/50',
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
      {label && (
        <span className="text-[10px] font-mono text-muted-foreground">{label}</span>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Description helper
// ---------------------------------------------------------------------------

function getStageDescription(t: number): string {
  if (t >= 950)
    return 'Pure static. The model is hallucinating structure from almost nothing. It must decide: is this a T-shirt, a shoe, or something else entirely?'
  if (t >= 750)
    return 'The first hints of structure emerge. A rough shape is forming from the static—the model is making coarse, global decisions.'
  if (t >= 500)
    return 'A shape is clearly emerging. The model has committed to a general form and is starting to define edges and regions.'
  if (t >= 250)
    return 'The object is recognizable. The model is now refining proportions, smoothing edges, and adding internal structure.'
  if (t >= 100)
    return 'Fine details are appearing. The outline is solid, and the model is working on textures and subtle shading.'
  if (t >= 25)
    return 'Nearly done. The model is making minute adjustments—polishing details that are hard to see at this scale.'
  return 'The generated image. Every step from t=1000 to here contributed: early steps created structure, later steps refined details.'
}

// ---------------------------------------------------------------------------
// Main widget
// ---------------------------------------------------------------------------

export function DenoisingTrajectoryWidget({
  width: widthOverride,
}: WidgetProps) {
  const { containerRef, width: measuredWidth } = useContainerWidth(600)
  const width = widthOverride ?? measuredWidth

  const [currentStep, setCurrentStep] = useState(1000)
  const [isPlaying, setIsPlaying] = useState(false)
  const animationFrameIndex = useRef(0)
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const basePixels = useMemo(() => generateBaseImage(), [])

  // Precompute all key snapshots for the timeline strip
  const snapshotPixels = useMemo(() => {
    const map = new Map<number, number[]>()
    for (const t of KEY_TIMESTEPS) {
      const ab = getAlphaBar(t)
      map.set(t, applyNoise(basePixels, ab, NOISE_SEED))
    }
    return map
  }, [basePixels])

  // Current image at the selected timestep
  const alphaBar = getAlphaBar(currentStep)
  const currentPixels = useMemo(
    () => applyNoise(basePixels, alphaBar, NOISE_SEED),
    [basePixels, alphaBar],
  )

  // Animation logic
  const stopAnimation = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current)
      intervalRef.current = null
    }
    setIsPlaying(false)
  }, [])

  const startAnimation = useCallback(() => {
    // Find the closest frame to current position
    let startIdx = 0
    for (let i = 0; i < ANIMATION_FRAMES.length; i++) {
      if (ANIMATION_FRAMES[i] <= currentStep) {
        startIdx = i
        break
      }
    }
    animationFrameIndex.current = startIdx

    setIsPlaying(true)
    intervalRef.current = setInterval(() => {
      animationFrameIndex.current += 1
      if (animationFrameIndex.current >= ANIMATION_FRAMES.length) {
        stopAnimation()
        return
      }
      setCurrentStep(ANIMATION_FRAMES[animationFrameIndex.current])
    }, 150)
  }, [currentStep, stopAnimation])

  const togglePlayback = useCallback(() => {
    if (isPlaying) {
      stopAnimation()
      return
    }
    // If at the end, restart from the beginning
    if (currentStep === 0) {
      setCurrentStep(1000)
      animationFrameIndex.current = 0
      setIsPlaying(true)
      intervalRef.current = setInterval(() => {
        animationFrameIndex.current += 1
        if (animationFrameIndex.current >= ANIMATION_FRAMES.length) {
          stopAnimation()
          return
        }
        setCurrentStep(ANIMATION_FRAMES[animationFrameIndex.current])
      }, 150)
      return
    }
    startAnimation()
  }, [isPlaying, currentStep, startAnimation, stopAnimation])

  const handleReset = useCallback(() => {
    stopAnimation()
    setCurrentStep(1000)
  }, [stopAnimation])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current)
      }
    }
  }, [])

  const handleSliderChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      stopAnimation()
      // Slider goes from 0 (right, clean) to 1000 (left, noisy)
      // We invert so moving right = denoising progress
      setCurrentStep(TOTAL_STEPS - Number(e.target.value))
    },
    [stopAnimation],
  )

  // Image scale based on available width
  const imgScale = width < 400 ? 3 : width < 550 ? 4 : 5
  const snapshotScale = width < 500 ? 2 : 3

  // Signal/noise percentages
  const signalPct = Math.round(alphaBar * 100)
  const noisePct = Math.round((1 - alphaBar) * 100)

  return (
    <div ref={containerRef} className="space-y-4">
      {/* Main image display */}
      <div className="flex flex-col items-center gap-3">
        <div className="flex items-center gap-2">
          <span className="text-sm font-mono font-bold text-violet-400">
            t = {currentStep}
          </span>
          <span className="text-xs text-muted-foreground">
            {currentStep === 1000
              ? '(pure noise)'
              : currentStep === 0
                ? '(generated image)'
                : `(step ${TOTAL_STEPS - currentStep} of ${TOTAL_STEPS})`}
          </span>
        </div>
        <PixelCanvas pixels={currentPixels} scale={imgScale} highlight />
      </div>

      {/* Signal vs noise bar */}
      <div className="space-y-1 px-2">
        <div className="flex items-center justify-between text-[10px] text-muted-foreground">
          <span className="text-emerald-400">Signal: {signalPct}%</span>
          <span className="text-rose-400">Noise: {noisePct}%</span>
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

      {/* Playback controls + slider */}
      <div className="flex items-center gap-3 px-2">
        <button
          onClick={togglePlayback}
          className="flex items-center justify-center w-8 h-8 rounded-full bg-violet-500/20 text-violet-400 hover:bg-violet-500/30 transition-colors cursor-pointer"
          title={isPlaying ? 'Pause' : 'Play'}
        >
          {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4 ml-0.5" />}
        </button>
        <button
          onClick={handleReset}
          className="flex items-center justify-center w-8 h-8 rounded-full bg-muted/50 text-muted-foreground hover:bg-muted/80 transition-colors cursor-pointer"
          title="Reset to t=1000"
        >
          <RotateCcw className="w-3.5 h-3.5" />
        </button>
        <div className="flex-1">
          <input
            type="range"
            min={0}
            max={TOTAL_STEPS}
            step={1}
            value={TOTAL_STEPS - currentStep}
            onChange={handleSliderChange}
            className="w-full accent-violet-500 cursor-ew-resize"
          />
          <div className="flex items-center justify-between text-[10px] text-muted-foreground mt-0.5">
            <span>t=1000 (noise)</span>
            <span>t=0 (clean)</span>
          </div>
        </div>
      </div>

      {/* Key timestep snapshots strip */}
      <div className="space-y-2">
        <p className="text-xs font-medium text-muted-foreground px-2">
          Key snapshots along the denoising trajectory:
        </p>
        <div className="flex items-end gap-1.5 overflow-x-auto pb-1 px-2 justify-center flex-wrap">
          {KEY_TIMESTEPS.map((t) => {
            const pixels = snapshotPixels.get(t)
            if (!pixels) return null
            const isActive = t === currentStep
            return (
              <button
                key={t}
                onClick={() => {
                  stopAnimation()
                  setCurrentStep(t)
                }}
                className={cn(
                  'flex flex-col items-center gap-0.5 rounded p-1 transition-colors cursor-pointer',
                  isActive
                    ? 'bg-violet-500/10'
                    : 'hover:bg-muted/50',
                )}
              >
                <PixelCanvas
                  pixels={pixels}
                  scale={snapshotScale}
                  highlight={isActive}
                />
                <span
                  className={cn(
                    'text-[9px] font-mono',
                    isActive ? 'text-violet-400 font-bold' : 'text-muted-foreground',
                  )}
                >
                  t={t}
                </span>
              </button>
            )
          })}
        </div>
      </div>

      {/* Context-sensitive description */}
      <div className="px-3 py-2 bg-muted/30 rounded-lg">
        <p className="text-xs text-muted-foreground">
          {getStageDescription(currentStep)}
        </p>
      </div>
    </div>
  )
}

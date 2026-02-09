'use client'

import { useState, useCallback } from 'react'
import { cn } from '@/lib/utils'
import { Plus, Trash2, RotateCcw, ChevronDown } from 'lucide-react'
import { Button } from '@/components/ui/button'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type LayerType = 'conv' | 'pool' | 'flatten' | 'linear'

type ConvConfig = {
  type: 'conv'
  outChannels: number
  kernelSize: number
  stride: number
  padding: number
}

type PoolConfig = {
  type: 'pool'
  poolType: 'max' | 'avg'
  kernelSize: number
  stride: number
}

type FlattenConfig = {
  type: 'flatten'
}

type LinearConfig = {
  type: 'linear'
  outFeatures: number
}

type LayerConfig = ConvConfig | PoolConfig | FlattenConfig | LinearConfig

type Layer = {
  id: string
  config: LayerConfig
}

type Shape = {
  h: number
  w: number
  c: number
  flat?: number
}

type ShapeResult = {
  shape: Shape
  valid: boolean
  warning?: string
}

// ---------------------------------------------------------------------------
// Dimension computation
// ---------------------------------------------------------------------------

function computeConvOutput(input: Shape, config: ConvConfig): ShapeResult {
  const h = Math.floor((input.h - config.kernelSize + 2 * config.padding) / config.stride) + 1
  const w = Math.floor((input.w - config.kernelSize + 2 * config.padding) / config.stride) + 1
  const c = config.outChannels

  if (h < 1 || w < 1) {
    return { shape: { h: 0, w: 0, c }, valid: false, warning: 'Spatial dimensions reduced to 0' }
  }

  return { shape: { h, w, c }, valid: true }
}

function computePoolOutput(input: Shape, config: PoolConfig): ShapeResult {
  const stride = config.stride
  const h = Math.floor((input.h - config.kernelSize) / stride) + 1
  const w = Math.floor((input.w - config.kernelSize) / stride) + 1

  if (h < 1 || w < 1) {
    return { shape: { h: 0, w: 0, c: input.c }, valid: false, warning: 'Spatial dimensions reduced to 0' }
  }

  return { shape: { h, w, c: input.c }, valid: true }
}

function computeFlattenOutput(input: Shape): ShapeResult {
  const flat = input.h * input.w * input.c
  const warning = flat > 10000 ? `Large vector: ${flat.toLocaleString()} values` : undefined
  return { shape: { h: 1, w: 1, c: 1, flat }, valid: true, warning }
}

function computeLinearOutput(_input: Shape, config: LinearConfig): ShapeResult {
  return { shape: { h: 1, w: 1, c: 1, flat: config.outFeatures }, valid: true }
}

function computeOutputShape(input: Shape, config: LayerConfig): ShapeResult {
  if (config.type === 'conv') return computeConvOutput(input, config)
  if (config.type === 'pool') return computePoolOutput(input, config)
  if (config.type === 'flatten') return computeFlattenOutput(input)
  return computeLinearOutput(input, config)
}

function computeAllShapes(inputShape: Shape, layers: Layer[]): ShapeResult[] {
  const results: ShapeResult[] = []
  let current = inputShape

  for (const layer of layers) {
    const isFlat = current.flat !== undefined
    const needsSpatial = layer.config.type === 'conv' || layer.config.type === 'pool'

    if (isFlat && needsSpatial) {
      results.push({
        shape: { h: 0, w: 0, c: 0 },
        valid: false,
        warning: 'Cannot apply spatial layer after Flatten',
      })
      continue
    }

    if (!isFlat && layer.config.type === 'linear') {
      results.push({
        shape: { h: 0, w: 0, c: 0 },
        valid: false,
        warning: 'Need Flatten before Linear layer',
      })
      continue
    }

    const result = computeOutputShape(current, layer.config)
    results.push(result)

    if (!result.valid) continue
    current = result.shape
  }

  return results
}

// ---------------------------------------------------------------------------
// Presets
// ---------------------------------------------------------------------------

let idCounter = 0
function nextId(): string {
  idCounter += 1
  return `layer-${idCounter}`
}

function makeLeNetLayers(): Layer[] {
  return [
    { id: nextId(), config: { type: 'conv', outChannels: 32, kernelSize: 3, stride: 1, padding: 1 } },
    { id: nextId(), config: { type: 'pool', poolType: 'max', kernelSize: 2, stride: 2 } },
    { id: nextId(), config: { type: 'conv', outChannels: 64, kernelSize: 3, stride: 1, padding: 1 } },
    { id: nextId(), config: { type: 'pool', poolType: 'max', kernelSize: 2, stride: 2 } },
    { id: nextId(), config: { type: 'flatten' } },
    { id: nextId(), config: { type: 'linear', outFeatures: 128 } },
    { id: nextId(), config: { type: 'linear', outFeatures: 10 } },
  ]
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

const LAYER_TYPE_OPTIONS: { value: LayerType; label: string }[] = [
  { value: 'conv', label: 'Conv2d' },
  { value: 'pool', label: 'MaxPool2d' },
  { value: 'flatten', label: 'Flatten' },
  { value: 'linear', label: 'Linear' },
]

function layerLabel(config: LayerConfig): string {
  if (config.type === 'conv') return `Conv2d(${config.outChannels}, ${config.kernelSize}x${config.kernelSize})`
  if (config.type === 'pool') return `${config.poolType === 'max' ? 'Max' : 'Avg'}Pool2d(${config.kernelSize}x${config.kernelSize})`
  if (config.type === 'flatten') return 'Flatten'
  return `Linear(${config.outFeatures})`
}

function layerColor(type: LayerConfig['type']): string {
  if (type === 'conv') return 'bg-violet-500/20 border-violet-500/40 text-violet-300'
  if (type === 'pool') return 'bg-sky-500/20 border-sky-500/40 text-sky-300'
  if (type === 'flatten') return 'bg-amber-500/20 border-amber-500/40 text-amber-300'
  return 'bg-emerald-500/20 border-emerald-500/40 text-emerald-300'
}

function layerDotColor(type: LayerConfig['type']): string {
  if (type === 'conv') return 'bg-violet-400'
  if (type === 'pool') return 'bg-sky-400'
  if (type === 'flatten') return 'bg-amber-400'
  return 'bg-emerald-400'
}

function shapeLabel(shape: Shape): string {
  if (shape.flat !== undefined) return `${shape.flat}`
  return `${shape.h}x${shape.w}x${shape.c}`
}

function defaultConfig(type: LayerType): LayerConfig {
  if (type === 'conv') return { type: 'conv', outChannels: 32, kernelSize: 3, stride: 1, padding: 1 }
  if (type === 'pool') return { type: 'pool', poolType: 'max', kernelSize: 2, stride: 2 }
  if (type === 'flatten') return { type: 'flatten' }
  return { type: 'linear', outFeatures: 128 }
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function NumberInput({
  label,
  value,
  onChange,
  min = 1,
  max = 512,
}: {
  label: string
  value: number
  onChange: (v: number) => void
  min?: number
  max?: number
}) {
  return (
    <label className="flex items-center gap-2 text-xs text-muted-foreground">
      <span className="w-20 text-right">{label}</span>
      <input
        type="number"
        value={value}
        onChange={(e) => {
          const v = parseInt(e.target.value, 10)
          if (!isNaN(v) && v >= min && v <= max) onChange(v)
        }}
        className="w-16 px-2 py-1 rounded bg-muted border border-border text-foreground text-xs"
        min={min}
        max={max}
      />
    </label>
  )
}

function LayerEditor({
  layer,
  onUpdate,
  onRemove,
  shapeResult,
  inputShape,
}: {
  layer: Layer
  onUpdate: (config: LayerConfig) => void
  onRemove: () => void
  shapeResult: ShapeResult | undefined
  inputShape: Shape
}) {
  const { config } = layer
  const colorClass = layerColor(config.type)
  const dotColor = layerDotColor(config.type)
  const isInvalid = shapeResult && !shapeResult.valid

  return (
    <div
      className={cn(
        'rounded-lg border p-3 space-y-2',
        isInvalid ? 'bg-rose-500/10 border-rose-500/40' : colorClass,
      )}
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className={cn('w-2 h-2 rounded-full', isInvalid ? 'bg-rose-400' : dotColor)} />
          <span className="text-xs font-medium">{layerLabel(config)}</span>
        </div>
        <button
          onClick={onRemove}
          className="text-muted-foreground hover:text-rose-400 transition-colors cursor-pointer"
          aria-label="Remove layer"
        >
          <Trash2 className="w-3.5 h-3.5" />
        </button>
      </div>

      {config.type === 'conv' && (
        <div className="flex flex-wrap gap-x-4 gap-y-1">
          <NumberInput label="Filters" value={config.outChannels} onChange={(v) => onUpdate({ ...config, outChannels: v })} max={512} />
          <NumberInput label="Kernel" value={config.kernelSize} onChange={(v) => onUpdate({ ...config, kernelSize: v })} max={11} />
          <NumberInput label="Stride" value={config.stride} onChange={(v) => onUpdate({ ...config, stride: v })} max={5} />
          <NumberInput label="Padding" value={config.padding} onChange={(v) => onUpdate({ ...config, padding: v })} min={0} max={5} />
        </div>
      )}

      {config.type === 'pool' && (
        <div className="flex flex-wrap gap-x-4 gap-y-1">
          <label className="flex items-center gap-2 text-xs text-muted-foreground">
            <span className="w-20 text-right">Type</span>
            <select
              value={config.poolType}
              onChange={(e) => onUpdate({ ...config, poolType: e.target.value as 'max' | 'avg' })}
              className="px-2 py-1 rounded bg-muted border border-border text-foreground text-xs cursor-pointer"
            >
              <option value="max">Max</option>
              <option value="avg">Average</option>
            </select>
          </label>
          <NumberInput label="Kernel" value={config.kernelSize} onChange={(v) => onUpdate({ ...config, kernelSize: v })} max={5} />
          <NumberInput label="Stride" value={config.stride} onChange={(v) => onUpdate({ ...config, stride: v })} max={5} />
        </div>
      )}

      {config.type === 'linear' && (
        <div className="flex flex-wrap gap-x-4 gap-y-1">
          <NumberInput label="Out features" value={config.outFeatures} onChange={(v) => onUpdate({ ...config, outFeatures: v })} max={4096} />
        </div>
      )}

      {/* Shape result */}
      {shapeResult && (
        <div className="flex items-center gap-2 text-xs pt-1 border-t border-border/50">
          <span className="text-muted-foreground">
            {shapeLabel(inputShape)} {'\u2192'}
          </span>
          <span className={cn('font-mono font-medium', isInvalid ? 'text-rose-400' : 'text-foreground')}>
            {isInvalid ? 'invalid' : shapeLabel(shapeResult.shape)}
          </span>
          {shapeResult.warning && (
            <span className={cn('text-xs', isInvalid ? 'text-rose-400' : 'text-amber-400')}>
              ({shapeResult.warning})
            </span>
          )}
        </div>
      )}
    </div>
  )
}

type PipelineItem = {
  label: string
  shape: Shape
  type: 'input' | LayerConfig['type']
  valid: boolean
  warning?: string
}

// Map a value from [0, maxVal] to [minPx, maxPx] using sqrt scaling for
// perceptual proportionality (linear makes small values too small)
function scaleDimension(value: number, maxVal: number, minPx: number, maxPx: number): number {
  if (maxVal <= 0) return minPx
  const ratio = Math.sqrt(Math.min(value, maxVal) / maxVal)
  return Math.round(minPx + ratio * (maxPx - minPx))
}

function computeBlockDimensions(
  item: PipelineItem,
  maxSpatial: number,
  maxChannels: number,
  maxFlat: number,
): { heightPx: number; widthPx: number } {
  const MIN_H = 32
  const MAX_H = 80
  const MIN_W = 36
  const MAX_W = 72

  if (!item.valid) return { heightPx: MIN_H, widthPx: MIN_W }

  // Flat layers: tall and narrow (vector shape)
  if (item.shape.flat !== undefined) {
    const flatVal = item.shape.flat
    const heightPx = scaleDimension(flatVal, maxFlat, MIN_H, MAX_H)
    return { heightPx, widthPx: MIN_W }
  }

  // Spatial layers: height ~ spatial, width ~ channels
  const spatial = Math.max(item.shape.h, item.shape.w)
  const heightPx = scaleDimension(spatial, maxSpatial, MIN_H, MAX_H)
  const widthPx = scaleDimension(item.shape.c, maxChannels, MIN_W, MAX_W)
  return { heightPx, widthPx }
}

function PipelineVisualization({
  inputShape,
  layers,
  shapeResults,
}: {
  inputShape: Shape
  layers: Layer[]
  shapeResults: ShapeResult[]
}) {
  const allShapes: PipelineItem[] = [
    { label: 'Input', shape: inputShape, type: 'input', valid: true },
  ]

  for (let i = 0; i < layers.length; i++) {
    const result = shapeResults[i]
    if (!result) continue

    allShapes.push({
      label: layerLabel(layers[i].config),
      shape: result.valid ? result.shape : { h: 0, w: 0, c: 0 },
      type: layers[i].config.type,
      valid: result.valid,
      warning: result.warning,
    })
  }

  // Find max values for scaling across all valid shapes
  let maxSpatial = 1
  let maxChannels = 1
  let maxFlat = 1
  for (const item of allShapes) {
    if (!item.valid) continue
    if (item.shape.flat !== undefined) {
      maxFlat = Math.max(maxFlat, item.shape.flat)
      continue
    }
    maxSpatial = Math.max(maxSpatial, item.shape.h, item.shape.w)
    maxChannels = Math.max(maxChannels, item.shape.c)
  }

  return (
    <div className="overflow-x-auto pb-2">
      <div className="flex items-end gap-1 min-w-fit">
        {allShapes.map((item, i) => {
          const pipeColor = pipelineBlockColor(item.type, item.valid)
          const { heightPx, widthPx } = computeBlockDimensions(item, maxSpatial, maxChannels, maxFlat)

          return (
            <div key={i} className="flex items-end gap-1">
              {i > 0 && (
                <div className="text-muted-foreground/40 text-xs px-0.5 pb-4">{'\u2192'}</div>
              )}
              <div className="flex flex-col items-center gap-0.5">
                <div
                  className={cn(
                    'rounded border flex flex-col items-center justify-center transition-all',
                    pipeColor,
                  )}
                  style={{ height: `${heightPx}px`, width: `${widthPx}px` }}
                >
                  <div className={cn(
                    'text-[9px] font-mono font-semibold',
                    !item.valid && 'text-rose-400',
                  )}>
                    {item.valid ? shapeLabel(item.shape) : '\u2716'}
                  </div>
                </div>
                <div className="text-[8px] text-muted-foreground truncate max-w-[80px] text-center">
                  {item.label}
                </div>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

function pipelineBlockColor(type: 'input' | LayerConfig['type'], valid: boolean): string {
  if (!valid) return 'bg-rose-500/10 border-rose-500/30'
  if (type === 'input') return 'bg-zinc-500/20 border-zinc-500/30'
  if (type === 'conv') return 'bg-violet-500/10 border-violet-500/30'
  if (type === 'pool') return 'bg-sky-500/10 border-sky-500/30'
  if (type === 'flatten') return 'bg-amber-500/10 border-amber-500/30'
  return 'bg-emerald-500/10 border-emerald-500/30'
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

type CnnDimensionCalculatorProps = {
  width?: number
  height?: number
}

export function CnnDimensionCalculator({ width: _widthOverride }: CnnDimensionCalculatorProps) {
  const [inputH, setInputH] = useState(28)
  const [inputW, setInputW] = useState(28)
  const [inputC, setInputC] = useState(1)
  const [layers, setLayers] = useState<Layer[]>(makeLeNetLayers)
  const [addMenuOpen, setAddMenuOpen] = useState(false)

  const inputShape: Shape = { h: inputH, w: inputW, c: inputC }
  const shapeResults = computeAllShapes(inputShape, layers)

  const updateLayer = useCallback((id: string, config: LayerConfig) => {
    setLayers((prev) => prev.map((l) => (l.id === id ? { ...l, config } : l)))
  }, [])

  const removeLayer = useCallback((id: string) => {
    setLayers((prev) => prev.filter((l) => l.id !== id))
  }, [])

  const addLayer = useCallback((type: LayerType) => {
    setLayers((prev) => [...prev, { id: nextId(), config: defaultConfig(type) }])
    setAddMenuOpen(false)
  }, [])

  const loadPreset = useCallback(() => {
    setInputH(28)
    setInputW(28)
    setInputC(1)
    setLayers(makeLeNetLayers())
  }, [])

  const clearAll = useCallback(() => {
    setLayers([])
  }, [])

  // Compute input shapes for each layer (for display in editor)
  const layerInputShapes: Shape[] = []
  let currentShape = inputShape
  for (let i = 0; i < layers.length; i++) {
    layerInputShapes.push(currentShape)
    const result = shapeResults[i]
    if (result && result.valid) {
      currentShape = result.shape
    }
  }

  return (
    <div className="space-y-4">
      {/* Pipeline visualization */}
      <div className="bg-muted/30 rounded-lg p-3 border border-border/50">
        <p className="text-[10px] uppercase tracking-wider text-muted-foreground/60 mb-2 font-medium">
          Data Shape Pipeline
        </p>
        <PipelineVisualization
          inputShape={inputShape}
          layers={layers}
          shapeResults={shapeResults}
        />
      </div>

      {/* Input shape config */}
      <div className="flex flex-wrap items-center gap-4">
        <span className="text-xs font-medium text-muted-foreground">Input:</span>
        <NumberInput label="H" value={inputH} onChange={setInputH} min={1} max={512} />
        <NumberInput label="W" value={inputW} onChange={setInputW} min={1} max={512} />
        <NumberInput label="C" value={inputC} onChange={setInputC} min={1} max={64} />
        <div className="flex gap-2 ml-auto">
          <Button variant="outline" size="sm" onClick={loadPreset} className="text-xs gap-1">
            <RotateCcw className="w-3 h-3" />
            LeNet Preset
          </Button>
          <Button variant="outline" size="sm" onClick={clearAll} className="text-xs gap-1 text-rose-400 hover:text-rose-300">
            <Trash2 className="w-3 h-3" />
            Clear
          </Button>
        </div>
      </div>

      {/* Layer stack */}
      <div className="space-y-2">
        {layers.map((layer, i) => (
          <LayerEditor
            key={layer.id}
            layer={layer}
            onUpdate={(config) => updateLayer(layer.id, config)}
            onRemove={() => removeLayer(layer.id)}
            shapeResult={shapeResults[i]}
            inputShape={layerInputShapes[i] ?? inputShape}
          />
        ))}
      </div>

      {/* Add layer button */}
      <div className="relative">
        <Button
          variant="outline"
          size="sm"
          onClick={() => setAddMenuOpen(!addMenuOpen)}
          className="text-xs gap-1 w-full border-dashed cursor-pointer"
        >
          <Plus className="w-3.5 h-3.5" />
          Add Layer
          <ChevronDown className={cn('w-3 h-3 transition-transform', addMenuOpen && 'rotate-180')} />
        </Button>
        {addMenuOpen && (
          <div className="absolute top-full left-0 right-0 mt-1 bg-popover border border-border rounded-md shadow-lg z-10 p-1">
            {LAYER_TYPE_OPTIONS.map((opt) => (
              <button
                key={opt.value}
                onClick={() => addLayer(opt.value)}
                className="w-full text-left px-3 py-1.5 text-xs rounded hover:bg-muted transition-colors cursor-pointer"
              >
                {opt.label}
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Summary stats */}
      <div className="flex flex-wrap gap-3 text-xs text-muted-foreground border-t border-border/50 pt-3">
        <span>
          <strong className="text-foreground">{layers.length}</strong> layers
        </span>
        <span>
          <strong className="text-foreground">{layers.filter((l) => l.config.type === 'conv').length}</strong> conv
        </span>
        <span>
          <strong className="text-foreground">{layers.filter((l) => l.config.type === 'pool').length}</strong> pool
        </span>
        {shapeResults.length > 0 && shapeResults[shapeResults.length - 1]?.valid && (
          <span className="ml-auto">
            Final output: <strong className="text-foreground font-mono">
              {shapeLabel(shapeResults[shapeResults.length - 1].shape)}
            </strong>
          </span>
        )}
      </div>
    </div>
  )
}

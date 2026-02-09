'use client'

import { useState } from 'react'
import { cn } from '@/lib/utils'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type LayerDef = {
  name: string
  type: 'conv' | 'pool' | 'flatten' | 'fc' | 'norm' | 'dropout'
  outputShape: string
  params: number
  receptiveField: number
  note?: string
}

type ArchitectureDef = {
  id: string
  name: string
  year: number
  depth: number
  totalParams: number
  minFilterSize: number
  imagenetAccuracy: string
  keyInnovation: string
  inputShape: string
  layers: LayerDef[]
}

// ---------------------------------------------------------------------------
// Architecture data
// ---------------------------------------------------------------------------

const ARCHITECTURES: ArchitectureDef[] = [
  {
    id: 'lenet',
    name: 'LeNet-5',
    year: 1998,
    depth: 5,
    totalParams: 61_706,
    minFilterSize: 5,
    imagenetAccuracy: 'N/A (MNIST: ~99%)',
    keyInnovation: 'Proved CNNs work',
    inputShape: '32x32x1',
    layers: [
      { name: 'Conv2d(1, 6, 5)', type: 'conv', outputShape: '28x28x6', params: 156, receptiveField: 5, note: '5x5 filters, no padding' },
      { name: 'AvgPool(2x2)', type: 'pool', outputShape: '14x14x6', params: 0, receptiveField: 6 },
      { name: 'Conv2d(6, 16, 5)', type: 'conv', outputShape: '10x10x16', params: 2_416, receptiveField: 14, note: '5x5 filters again' },
      { name: 'AvgPool(2x2)', type: 'pool', outputShape: '5x5x16', params: 0, receptiveField: 16 },
      { name: 'Flatten', type: 'flatten', outputShape: '400', params: 0, receptiveField: 16 },
      { name: 'Linear(400, 120)', type: 'fc', outputShape: '120', params: 48_120, receptiveField: 16 },
      { name: 'Linear(120, 84)', type: 'fc', outputShape: '84', params: 10_164, receptiveField: 16 },
      { name: 'Linear(84, 10)', type: 'fc', outputShape: '10', params: 850, receptiveField: 16 },
    ],
  },
  {
    id: 'alexnet',
    name: 'AlexNet',
    year: 2012,
    depth: 8,
    totalParams: 62_378_344,
    minFilterSize: 3,
    imagenetAccuracy: '~63% top-1',
    keyInnovation: 'ReLU + Dropout + GPU',
    inputShape: '227x227x3',
    layers: [
      { name: 'Conv2d(3, 96, 11, s=4)', type: 'conv', outputShape: '55x55x96', params: 34_944, receptiveField: 11, note: 'Large 11x11 filters, stride 4' },
      { name: 'MaxPool(3x3, s=2)', type: 'pool', outputShape: '27x27x96', params: 0, receptiveField: 19 },
      { name: 'Conv2d(96, 256, 5, p=2)', type: 'conv', outputShape: '27x27x256', params: 614_656, receptiveField: 51, note: '5x5 filters' },
      { name: 'MaxPool(3x3, s=2)', type: 'pool', outputShape: '13x13x256', params: 0, receptiveField: 67 },
      { name: 'Conv2d(256, 384, 3, p=1)', type: 'conv', outputShape: '13x13x384', params: 885_120, receptiveField: 99, note: '3x3 filters start here' },
      { name: 'Conv2d(384, 384, 3, p=1)', type: 'conv', outputShape: '13x13x384', params: 1_327_488, receptiveField: 131 },
      { name: 'Conv2d(384, 256, 3, p=1)', type: 'conv', outputShape: '13x13x256', params: 884_992, receptiveField: 163 },
      { name: 'MaxPool(3x3, s=2)', type: 'pool', outputShape: '6x6x256', params: 0, receptiveField: 195 },
      { name: 'Flatten', type: 'flatten', outputShape: '9216', params: 0, receptiveField: 195 },
      { name: 'Dropout(0.5)', type: 'dropout', outputShape: '9216', params: 0, receptiveField: 195 },
      { name: 'Linear(9216, 4096)', type: 'fc', outputShape: '4096', params: 37_752_832, receptiveField: 195, note: 'Most params are here' },
      { name: 'Dropout(0.5)', type: 'dropout', outputShape: '4096', params: 0, receptiveField: 195 },
      { name: 'Linear(4096, 4096)', type: 'fc', outputShape: '4096', params: 16_781_312, receptiveField: 195 },
      { name: 'Linear(4096, 1000)', type: 'fc', outputShape: '1000', params: 4_097_000, receptiveField: 195 },
    ],
  },
  {
    id: 'vgg16',
    name: 'VGG-16',
    year: 2014,
    depth: 16,
    totalParams: 138_357_544,
    minFilterSize: 3,
    imagenetAccuracy: '~74% top-1',
    keyInnovation: 'All 3x3 filters, deep',
    inputShape: '224x224x3',
    layers: [
      // Block 1
      { name: 'Conv2d(3, 64, 3, p=1)', type: 'conv', outputShape: '224x224x64', params: 1_792, receptiveField: 3, note: 'Block 1: two 3x3 convs' },
      { name: 'Conv2d(64, 64, 3, p=1)', type: 'conv', outputShape: '224x224x64', params: 36_928, receptiveField: 5 },
      { name: 'MaxPool(2x2)', type: 'pool', outputShape: '112x112x64', params: 0, receptiveField: 6 },
      // Block 2
      { name: 'Conv2d(64, 128, 3, p=1)', type: 'conv', outputShape: '112x112x128', params: 73_856, receptiveField: 10, note: 'Block 2: two 3x3 convs' },
      { name: 'Conv2d(128, 128, 3, p=1)', type: 'conv', outputShape: '112x112x128', params: 147_584, receptiveField: 14 },
      { name: 'MaxPool(2x2)', type: 'pool', outputShape: '56x56x128', params: 0, receptiveField: 16 },
      // Block 3
      { name: 'Conv2d(128, 256, 3, p=1)', type: 'conv', outputShape: '56x56x256', params: 295_168, receptiveField: 24, note: 'Block 3: three 3x3 convs' },
      { name: 'Conv2d(256, 256, 3, p=1)', type: 'conv', outputShape: '56x56x256', params: 590_080, receptiveField: 32 },
      { name: 'Conv2d(256, 256, 3, p=1)', type: 'conv', outputShape: '56x56x256', params: 590_080, receptiveField: 40 },
      { name: 'MaxPool(2x2)', type: 'pool', outputShape: '28x28x256', params: 0, receptiveField: 44 },
      // Block 4
      { name: 'Conv2d(256, 512, 3, p=1)', type: 'conv', outputShape: '28x28x512', params: 1_180_160, receptiveField: 60, note: 'Block 4: three 3x3 convs' },
      { name: 'Conv2d(512, 512, 3, p=1)', type: 'conv', outputShape: '28x28x512', params: 2_359_808, receptiveField: 76 },
      { name: 'Conv2d(512, 512, 3, p=1)', type: 'conv', outputShape: '28x28x512', params: 2_359_808, receptiveField: 92 },
      { name: 'MaxPool(2x2)', type: 'pool', outputShape: '14x14x512', params: 0, receptiveField: 100 },
      // Block 5
      { name: 'Conv2d(512, 512, 3, p=1)', type: 'conv', outputShape: '14x14x512', params: 2_359_808, receptiveField: 132, note: 'Block 5: three 3x3 convs' },
      { name: 'Conv2d(512, 512, 3, p=1)', type: 'conv', outputShape: '14x14x512', params: 2_359_808, receptiveField: 164 },
      { name: 'Conv2d(512, 512, 3, p=1)', type: 'conv', outputShape: '14x14x512', params: 2_359_808, receptiveField: 196 },
      { name: 'MaxPool(2x2)', type: 'pool', outputShape: '7x7x512', params: 0, receptiveField: 212 },
      // Classifier
      { name: 'Flatten', type: 'flatten', outputShape: '25088', params: 0, receptiveField: 212 },
      { name: 'Linear(25088, 4096)', type: 'fc', outputShape: '4096', params: 102_764_544, receptiveField: 212, note: 'Most params are in FC layers' },
      { name: 'Linear(4096, 4096)', type: 'fc', outputShape: '4096', params: 16_781_312, receptiveField: 212 },
      { name: 'Linear(4096, 1000)', type: 'fc', outputShape: '1000', params: 4_097_000, receptiveField: 212 },
    ],
  },
]

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function formatParams(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`
  return `${n}`
}

function layerTypeColor(type: LayerDef['type']): string {
  if (type === 'conv') return 'bg-violet-500/20 border-violet-500/40 text-violet-300'
  if (type === 'pool') return 'bg-sky-500/20 border-sky-500/40 text-sky-300'
  if (type === 'flatten') return 'bg-amber-500/20 border-amber-500/40 text-amber-300'
  if (type === 'fc') return 'bg-emerald-500/20 border-emerald-500/40 text-emerald-300'
  if (type === 'norm') return 'bg-orange-500/20 border-orange-500/40 text-orange-300'
  if (type === 'dropout') return 'bg-rose-500/20 border-rose-500/40 text-rose-300'
  return 'bg-zinc-500/20 border-zinc-500/40 text-zinc-300'
}

function layerDotColor(type: LayerDef['type']): string {
  if (type === 'conv') return 'bg-violet-400'
  if (type === 'pool') return 'bg-sky-400'
  if (type === 'flatten') return 'bg-amber-400'
  if (type === 'fc') return 'bg-emerald-400'
  if (type === 'norm') return 'bg-orange-400'
  if (type === 'dropout') return 'bg-rose-400'
  return 'bg-zinc-400'
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function ArchitectureSelector({
  architectures,
  selectedId,
  onSelect,
}: {
  architectures: ArchitectureDef[]
  selectedId: string
  onSelect: (id: string) => void
}) {
  return (
    <div className="flex flex-wrap gap-2">
      {architectures.map((arch) => (
        <button
          key={arch.id}
          onClick={() => onSelect(arch.id)}
          className={cn(
            'px-4 py-2 rounded-lg text-sm font-medium transition-colors cursor-pointer',
            'border',
            selectedId === arch.id
              ? 'bg-primary text-primary-foreground border-primary'
              : 'bg-muted hover:bg-muted/80 text-muted-foreground border-border',
          )}
        >
          <span className="font-semibold">{arch.name}</span>
          <span className="text-xs opacity-75 ml-1.5">({arch.year})</span>
        </button>
      ))}
    </div>
  )
}

function MetricsBar({ arch }: { arch: ArchitectureDef }) {
  const convLayers = arch.layers.filter((l) => l.type === 'conv').length
  const fcLayers = arch.layers.filter((l) => l.type === 'fc').length
  const convParams = arch.layers.filter((l) => l.type === 'conv').reduce((s, l) => s + l.params, 0)
  const fcParams = arch.layers.filter((l) => l.type === 'fc').reduce((s, l) => s + l.params, 0)
  const convPct = arch.totalParams > 0 ? Math.round((convParams / arch.totalParams) * 100) : 0
  const fcPct = arch.totalParams > 0 ? Math.round((fcParams / arch.totalParams) * 100) : 0

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
      <MetricCard label="Total Parameters" value={formatParams(arch.totalParams)} />
      <MetricCard label="Weight Layers" value={`${arch.depth} (${convLayers} conv + ${fcLayers} fc)`} />
      <MetricCard label="Conv Params" value={`${formatParams(convParams)} (${convPct}%)`} />
      <MetricCard label="FC Params" value={`${formatParams(fcParams)} (${fcPct}%)`} />
    </div>
  )
}

function MetricCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-border/50 bg-muted/30 p-2.5">
      <p className="text-[10px] uppercase tracking-wider text-muted-foreground/60 font-medium">{label}</p>
      <p className="text-sm font-semibold text-foreground mt-0.5">{value}</p>
    </div>
  )
}

function LayerTable({ arch }: { arch: ArchitectureDef }) {
  let cumParams = 0

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs">
        <thead>
          <tr className="border-b border-border/50">
            <th className="text-left py-1.5 px-2 text-muted-foreground font-medium">#</th>
            <th className="text-left py-1.5 px-2 text-muted-foreground font-medium">Layer</th>
            <th className="text-left py-1.5 px-2 text-muted-foreground font-medium">Output Shape</th>
            <th className="text-right py-1.5 px-2 text-muted-foreground font-medium">Params</th>
            <th className="text-right py-1.5 px-2 text-muted-foreground font-medium">Cumulative</th>
            <th className="text-right py-1.5 px-2 text-muted-foreground font-medium">RF</th>
            <th className="text-left py-1.5 px-2 text-muted-foreground font-medium hidden md:table-cell">Note</th>
          </tr>
        </thead>
        <tbody>
          {/* Input row */}
          <tr className="border-b border-border/30">
            <td className="py-1.5 px-2 text-muted-foreground/50">0</td>
            <td className="py-1.5 px-2">
              <div className="flex items-center gap-1.5">
                <div className="w-1.5 h-1.5 rounded-full bg-zinc-400" />
                <span className="text-muted-foreground">Input</span>
              </div>
            </td>
            <td className="py-1.5 px-2 font-mono text-foreground">{arch.inputShape}</td>
            <td className="text-right py-1.5 px-2 text-muted-foreground/50">&mdash;</td>
            <td className="text-right py-1.5 px-2 text-muted-foreground/50">&mdash;</td>
            <td className="text-right py-1.5 px-2 text-muted-foreground/50">1</td>
            <td className="py-1.5 px-2 hidden md:table-cell" />
          </tr>
          {arch.layers.map((layer, i) => {
            cumParams += layer.params
            return (
              <tr key={i} className="border-b border-border/30 hover:bg-muted/20">
                <td className="py-1.5 px-2 text-muted-foreground/50">{i + 1}</td>
                <td className="py-1.5 px-2">
                  <div className="flex items-center gap-1.5">
                    <div className={cn('w-1.5 h-1.5 rounded-full flex-shrink-0', layerDotColor(layer.type))} />
                    <span className={cn(
                      'px-1.5 py-0.5 rounded text-[10px] font-medium border',
                      layerTypeColor(layer.type),
                    )}>
                      {layer.name}
                    </span>
                  </div>
                </td>
                <td className="py-1.5 px-2 font-mono text-foreground">{layer.outputShape}</td>
                <td className="text-right py-1.5 px-2 font-mono">
                  {layer.params > 0 ? (
                    <span className="text-foreground">{formatParams(layer.params)}</span>
                  ) : (
                    <span className="text-muted-foreground/50">&mdash;</span>
                  )}
                </td>
                <td className="text-right py-1.5 px-2 font-mono text-muted-foreground">
                  {cumParams > 0 ? formatParams(cumParams) : '\u2014'}
                </td>
                <td className="text-right py-1.5 px-2 font-mono text-muted-foreground">{layer.receptiveField}</td>
                <td className="py-1.5 px-2 text-muted-foreground/70 hidden md:table-cell">{layer.note ?? ''}</td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

function ComparisonSidebar({ architectures }: { architectures: ArchitectureDef[] }) {
  return (
    <div className="rounded-lg border border-border/50 bg-muted/30 p-3 space-y-3">
      <p className="text-[10px] uppercase tracking-wider text-muted-foreground/60 font-medium">
        Quick Comparison
      </p>
      <table className="w-full text-xs">
        <thead>
          <tr className="border-b border-border/30">
            <th className="text-left py-1 text-muted-foreground font-medium" />
            {architectures.map((a) => (
              <th key={a.id} className="text-right py-1 text-muted-foreground font-medium px-1">{a.name}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          <ComparisonTableRow label="Year" values={architectures.map((a) => String(a.year))} />
          <ComparisonTableRow label="Depth" values={architectures.map((a) => String(a.depth))} />
          <ComparisonTableRow label="Params" values={architectures.map((a) => formatParams(a.totalParams))} />
          <ComparisonTableRow label="Min Filter" values={architectures.map((a) => `${a.minFilterSize}x${a.minFilterSize}`)} />
          <ComparisonTableRow label="Accuracy" values={architectures.map((a) => a.imagenetAccuracy)} />
          <ComparisonTableRow label="Innovation" values={architectures.map((a) => a.keyInnovation)} />
        </tbody>
      </table>
    </div>
  )
}

function ComparisonTableRow({ label, values }: { label: string; values: string[] }) {
  return (
    <tr className="border-b border-border/30">
      <td className="py-1 text-muted-foreground font-medium">{label}</td>
      {values.map((v, i) => (
        <td key={i} className="text-right py-1 px-1 font-mono text-foreground text-[11px]">{v}</td>
      ))}
    </tr>
  )
}

// ---------------------------------------------------------------------------
// "What If?" filter swap data
// ---------------------------------------------------------------------------

type VggBlockDef = {
  id: string
  label: string
  numConvs: number
  channels: number
  /** Parameters with 3x3 stacking (actual VGG) */
  params3x3: number
  /** Equivalent single large filter size */
  equivalentFilterSize: number
  /** Parameters with single large filter */
  paramsLarge: number
  receptiveField: number
}

const VGG_BLOCKS: VggBlockDef[] = [
  { id: 'block1', label: 'Block 1', numConvs: 2, channels: 64, params3x3: 38_720, equivalentFilterSize: 5, paramsLarge: 105_664, receptiveField: 5 },
  { id: 'block2', label: 'Block 2', numConvs: 2, channels: 128, params3x3: 221_440, equivalentFilterSize: 5, paramsLarge: 614_528, receptiveField: 5 },
  { id: 'block3', label: 'Block 3', numConvs: 3, channels: 256, params3x3: 1_475_328, equivalentFilterSize: 7, paramsLarge: 3_211_520, receptiveField: 7 },
  { id: 'block4', label: 'Block 4', numConvs: 3, channels: 512, params3x3: 5_899_776, equivalentFilterSize: 7, paramsLarge: 12_845_568, receptiveField: 7 },
  { id: 'block5', label: 'Block 5', numConvs: 3, channels: 512, params3x3: 7_079_424, equivalentFilterSize: 7, paramsLarge: 12_845_568, receptiveField: 7 },
]

function FilterSwapExplorer() {
  const [swappedBlocks, setSwappedBlocks] = useState<Set<string>>(new Set())

  function toggleBlock(blockId: string) {
    setSwappedBlocks((prev) => {
      const next = new Set(prev)
      if (next.has(blockId)) {
        next.delete(blockId)
      } else {
        next.add(blockId)
      }
      return next
    })
  }

  const totalParams3x3 = VGG_BLOCKS.reduce((s, b) => s + b.params3x3, 0)
  const totalParamsActual = VGG_BLOCKS.reduce(
    (s, b) => s + (swappedBlocks.has(b.id) ? b.paramsLarge : b.params3x3),
    0,
  )
  const totalNonlinearities3x3 = VGG_BLOCKS.reduce((s, b) => s + b.numConvs, 0)
  const totalNonlinearitiesActual = VGG_BLOCKS.reduce(
    (s, b) => s + (swappedBlocks.has(b.id) ? 1 : b.numConvs),
    0,
  )
  const paramDiff = totalParamsActual - totalParams3x3
  const hasSwaps = swappedBlocks.size > 0

  return (
    <div className="rounded-lg border border-border/50 bg-muted/20 p-3 space-y-3">
      <div className="flex items-center justify-between">
        <p className="text-[10px] uppercase tracking-wider text-muted-foreground/60 font-medium">
          What If VGG Used Larger Filters?
        </p>
        {hasSwaps && (
          <button
            onClick={() => setSwappedBlocks(new Set())}
            className="text-[10px] text-primary cursor-pointer hover:underline"
          >
            Reset all
          </button>
        )}
      </div>
      <p className="text-xs text-muted-foreground">
        Toggle any block to replace its stacked 3x3 convs with a single
        large filter that covers the same receptive field.
      </p>
      <div className="space-y-1.5">
        {VGG_BLOCKS.map((block) => {
          const isSwapped = swappedBlocks.has(block.id)
          const paramSaving = block.paramsLarge - block.params3x3
          return (
            <button
              key={block.id}
              onClick={() => toggleBlock(block.id)}
              className={cn(
                'w-full flex items-center justify-between gap-2 px-3 py-2 rounded-lg text-xs transition-colors cursor-pointer',
                'border',
                isSwapped
                  ? 'bg-amber-500/10 border-amber-500/30'
                  : 'bg-muted/30 border-border/50 hover:bg-muted/50',
              )}
            >
              <div className="flex items-center gap-2">
                <div className={cn(
                  'w-2 h-2 rounded-full',
                  isSwapped ? 'bg-amber-400' : 'bg-violet-400',
                )} />
                <span className="font-medium text-foreground">{block.label}</span>
                <span className="text-muted-foreground">
                  {isSwapped
                    ? `1x ${block.equivalentFilterSize}x${block.equivalentFilterSize} conv`
                    : `${block.numConvs}x 3x3 conv`}
                </span>
              </div>
              <div className="flex items-center gap-3">
                <span className="font-mono text-muted-foreground">
                  {formatParams(isSwapped ? block.paramsLarge : block.params3x3)}
                </span>
                {isSwapped && (
                  <span className="font-mono text-amber-400 text-[10px]">
                    +{formatParams(paramSaving)}
                  </span>
                )}
                <span className={cn(
                  'text-[10px] px-1.5 py-0.5 rounded',
                  isSwapped
                    ? 'bg-amber-500/20 text-amber-300'
                    : 'bg-violet-500/20 text-violet-300',
                )}>
                  {isSwapped ? '1 ReLU' : `${block.numConvs} ReLUs`}
                </span>
              </div>
            </button>
          )
        })}
      </div>
      {/* Summary bar */}
      <div className={cn(
        'rounded-lg p-2.5 text-xs',
        hasSwaps ? 'bg-amber-500/10 border border-amber-500/20' : 'bg-muted/30 border border-border/30',
      )}>
        <div className="flex justify-between items-center">
          <span className="text-muted-foreground">Conv params total:</span>
          <span className="font-mono font-semibold text-foreground">{formatParams(totalParamsActual)}</span>
        </div>
        <div className="flex justify-between items-center mt-1">
          <span className="text-muted-foreground">Nonlinearities:</span>
          <span className="font-mono font-semibold text-foreground">{totalNonlinearitiesActual}</span>
        </div>
        {hasSwaps && (
          <>
            <div className="flex justify-between items-center mt-1 pt-1 border-t border-border/30">
              <span className="text-amber-300">Extra params vs VGG&apos;s 3x3 stacking:</span>
              <span className="font-mono font-semibold text-amber-300">
                +{formatParams(paramDiff)} ({Math.round((paramDiff / totalParams3x3) * 100)}% more)
              </span>
            </div>
            <div className="flex justify-between items-center mt-1">
              <span className="text-amber-300">Lost nonlinearities:</span>
              <span className="font-mono font-semibold text-amber-300">
                {totalNonlinearitiesActual} of {totalNonlinearities3x3} remain
              </span>
            </div>
          </>
        )}
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

type ArchitectureComparisonExplorerProps = {
  width?: number
  height?: number
}

export function ArchitectureComparisonExplorer({ width: _widthOverride }: ArchitectureComparisonExplorerProps) {
  const [selectedId, setSelectedId] = useState<string>('lenet')
  const selected = ARCHITECTURES.find((a) => a.id === selectedId) ?? ARCHITECTURES[0]

  return (
    <div className="space-y-4">
      {/* Architecture selector */}
      <ArchitectureSelector
        architectures={ARCHITECTURES}
        selectedId={selectedId}
        onSelect={setSelectedId}
      />

      {/* Key metrics */}
      <MetricsBar arch={selected} />

      {/* Layer-by-layer table */}
      <div className="rounded-lg border border-border/50 bg-muted/20 p-3">
        <p className="text-[10px] uppercase tracking-wider text-muted-foreground/60 mb-2 font-medium">
          Layer-by-Layer Pipeline: {selected.name}
        </p>
        <LayerTable arch={selected} />
      </div>

      {/* What if? filter swap — only relevant for VGG */}
      {selectedId === 'vgg16' && <FilterSwapExplorer />}

      {/* Comparison sidebar */}
      <ComparisonSidebar architectures={ARCHITECTURES} />

      {/* Legend */}
      <div className="flex flex-wrap gap-3 text-[10px] text-muted-foreground">
        <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-violet-400 inline-block" /> Conv</span>
        <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-sky-400 inline-block" /> Pool</span>
        <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-amber-400 inline-block" /> Flatten</span>
        <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-emerald-400 inline-block" /> FC</span>
        <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-rose-400 inline-block" /> Dropout</span>
        <span className="ml-auto">RF = Receptive Field (pixels of input each position &ldquo;sees&rdquo;)</span>
      </div>
    </div>
  )
}

'use client'

import { useState, useMemo } from 'react'
import { cn } from '@/lib/utils'
import { useContainerWidth } from '@/hooks/useContainerWidth'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type BlockMode = 'plain' | 'residual'

type SimulationResult = {
  input: number
  convOutput: number
  blockOutput: number
  gradientConvPath: number
  gradientSkipPath: number
  gradientTotal: number
}

// ---------------------------------------------------------------------------
// Simulation helpers
// ---------------------------------------------------------------------------

function computeSimulation(
  input: number,
  convWeight: number,
  mode: BlockMode,
): SimulationResult {
  // Simplified model: F(x) = w * x (single conv weight for demonstration)
  const convOutput = convWeight * input

  // Plain block: output = F(x) = w * x
  // Residual block: output = F(x) + x = w * x + x
  const blockOutput = mode === 'residual' ? convOutput + input : convOutput

  // Gradient through conv path: dOutput/dInput via conv = w
  const gradientConvPath = convWeight

  // Gradient through skip path: dOutput/dInput via skip = 1 (always)
  const gradientSkipPath = mode === 'residual' ? 1 : 0

  // Total gradient: sum of both paths
  const gradientTotal = gradientConvPath + gradientSkipPath

  return {
    input,
    convOutput,
    blockOutput,
    gradientConvPath,
    gradientSkipPath,
    gradientTotal,
  }
}

function formatNum(n: number, decimals = 2): string {
  return n.toFixed(decimals)
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function BlockDiagram({
  mode,
  convWeight,
  sim,
  width,
}: {
  mode: BlockMode
  convWeight: number
  sim: SimulationResult
  width: number
}) {
  const svgWidth = Math.min(width - 16, 520)
  const svgHeight = 200

  // Layout coordinates
  const inputX = 40
  const convStartX = svgWidth * 0.25
  const convEndX = svgWidth * 0.65
  const outputX = svgWidth - 40
  const mainY = svgHeight * 0.55
  const skipY = svgHeight * 0.22
  const addX = svgWidth * 0.72

  return (
    <div className="flex justify-center">
      <svg
        width={svgWidth}
        height={svgHeight}
        viewBox={`0 0 ${svgWidth} ${svgHeight}`}
        className="text-muted-foreground"
      >
        {/* Input node */}
        <circle cx={inputX} cy={mainY} r={16} fill="rgba(99, 102, 241, 0.2)" stroke="rgba(99, 102, 241, 0.6)" strokeWidth={2} />
        <text x={inputX} y={mainY + 1} textAnchor="middle" fontSize={11} fill="currentColor" dominantBaseline="middle">x</text>
        <text x={inputX} y={mainY + 30} textAnchor="middle" fontSize={9} fill="currentColor" opacity={0.6}>{formatNum(sim.input)}</text>

        {/* Main conv path: arrow from input to conv block */}
        <line x1={inputX + 16} y1={mainY} x2={convStartX - 2} y2={mainY} stroke="currentColor" strokeWidth={1.5} markerEnd="url(#arrowhead)" />

        {/* Conv block */}
        <rect x={convStartX} y={mainY - 22} width={convEndX - convStartX} height={44} rx={6} fill="rgba(251, 146, 60, 0.15)" stroke="rgba(251, 146, 60, 0.5)" strokeWidth={1.5} />
        <text x={(convStartX + convEndX) / 2} y={mainY - 5} textAnchor="middle" fontSize={10} fill="rgba(251, 146, 60, 0.9)" fontWeight={600}>Conv layers</text>
        <text x={(convStartX + convEndX) / 2} y={mainY + 10} textAnchor="middle" fontSize={9} fill="currentColor" opacity={0.6}>F(x) = w*x</text>
        <text x={(convStartX + convEndX) / 2} y={mainY + 34} textAnchor="middle" fontSize={9} fill="currentColor" opacity={0.5}>w = {formatNum(convWeight)}, F(x) = {formatNum(sim.convOutput)}</text>

        {/* Skip connection (only for residual mode) */}
        {mode === 'residual' && (
          <>
            {/* Line from input up */}
            <line x1={inputX} y1={mainY - 16} x2={inputX} y2={skipY} stroke="rgba(34, 197, 94, 0.7)" strokeWidth={2} strokeDasharray="4 3" />
            {/* Horizontal skip line */}
            <line x1={inputX} y1={skipY} x2={addX} y2={skipY} stroke="rgba(34, 197, 94, 0.7)" strokeWidth={2} strokeDasharray="4 3" />
            {/* Line down to add node */}
            <line x1={addX} y1={skipY} x2={addX} y2={mainY - 16} stroke="rgba(34, 197, 94, 0.7)" strokeWidth={2} strokeDasharray="4 3" />
            {/* Label */}
            <text x={(inputX + addX) / 2} y={skipY - 6} textAnchor="middle" fontSize={9} fill="rgba(34, 197, 94, 0.8)" fontWeight={600}>Identity shortcut (x)</text>
            <text x={(inputX + addX) / 2} y={skipY + 12} textAnchor="middle" fontSize={9} fill="rgba(34, 197, 94, 0.6)">gradient = 1.0</text>
          </>
        )}

        {/* Arrow from conv to add/output */}
        {mode === 'residual' ? (
          <>
            <line x1={convEndX} y1={mainY} x2={addX - 14} y2={mainY} stroke="currentColor" strokeWidth={1.5} markerEnd="url(#arrowhead)" />
            {/* Add node */}
            <circle cx={addX} cy={mainY} r={14} fill="rgba(34, 197, 94, 0.15)" stroke="rgba(34, 197, 94, 0.5)" strokeWidth={2} />
            <text x={addX} y={mainY + 1} textAnchor="middle" fontSize={14} fill="rgba(34, 197, 94, 0.9)" dominantBaseline="middle" fontWeight={700}>+</text>
            {/* Arrow from add to output */}
            <line x1={addX + 14} y1={mainY} x2={outputX - 18} y2={mainY} stroke="currentColor" strokeWidth={1.5} markerEnd="url(#arrowhead)" />
          </>
        ) : (
          <line x1={convEndX} y1={mainY} x2={outputX - 18} y2={mainY} stroke="currentColor" strokeWidth={1.5} markerEnd="url(#arrowhead)" />
        )}

        {/* Output node */}
        <circle cx={outputX} cy={mainY} r={16} fill="rgba(168, 85, 247, 0.2)" stroke="rgba(168, 85, 247, 0.6)" strokeWidth={2} />
        <text x={outputX} y={mainY + 1} textAnchor="middle" fontSize={9} fill="currentColor" dominantBaseline="middle">
          {mode === 'residual' ? 'F+x' : 'F(x)'}
        </text>
        <text x={outputX} y={mainY + 30} textAnchor="middle" fontSize={9} fill="currentColor" opacity={0.6}>{formatNum(sim.blockOutput)}</text>

        {/* Arrowhead marker */}
        <defs>
          <marker id="arrowhead" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
            <polygon points="0 0, 8 3, 0 6" fill="currentColor" opacity={0.5} />
          </marker>
        </defs>
      </svg>
    </div>
  )
}

function StatBadge({
  label,
  value,
  color,
}: {
  label: string
  value: string
  color: 'violet' | 'emerald' | 'amber' | 'rose' | 'sky'
}) {
  const colorClasses = colorToClasses(color)

  return (
    <div className={cn('px-3 py-1.5 rounded-lg text-xs', colorClasses)}>
      <span className="opacity-70">{label}: </span>
      <span className="font-semibold">{value}</span>
    </div>
  )
}

function colorToClasses(color: 'violet' | 'emerald' | 'amber' | 'rose' | 'sky'): string {
  if (color === 'violet') return 'bg-violet-500/10 text-violet-300'
  if (color === 'emerald') return 'bg-emerald-500/10 text-emerald-300'
  if (color === 'amber') return 'bg-amber-500/10 text-amber-300'
  if (color === 'rose') return 'bg-rose-500/10 text-rose-300'
  return 'bg-sky-500/10 text-sky-300'
}

function GradientBar({
  convGrad,
  skipGrad,
  totalGrad,
  mode,
}: {
  convGrad: number
  skipGrad: number
  totalGrad: number
  mode: BlockMode
}) {
  const maxGrad = Math.max(Math.abs(totalGrad), 2)
  const convWidth = Math.min((Math.abs(convGrad) / maxGrad) * 100, 100)
  const skipWidth = mode === 'residual' ? Math.min((Math.abs(skipGrad) / maxGrad) * 100, 100) : 0

  return (
    <div className="space-y-2">
      <p className="text-xs text-muted-foreground font-medium">Gradient Flow (dOutput/dInput)</p>
      <div className="space-y-1.5">
        <div className="flex items-center gap-2">
          <span className="text-xs text-muted-foreground w-20 text-right">Conv path</span>
          <div className="flex-1 h-4 bg-muted/30 rounded-full overflow-hidden">
            <div
              className="h-full bg-amber-500/60 rounded-full transition-all duration-300"
              style={{ width: `${convWidth}%` }}
            />
          </div>
          <span className="text-xs font-mono w-12 text-muted-foreground">{formatNum(convGrad)}</span>
        </div>
        {mode === 'residual' && (
          <div className="flex items-center gap-2">
            <span className="text-xs text-muted-foreground w-20 text-right">Skip path</span>
            <div className="flex-1 h-4 bg-muted/30 rounded-full overflow-hidden">
              <div
                className="h-full bg-emerald-500/60 rounded-full transition-all duration-300"
                style={{ width: `${skipWidth}%` }}
              />
            </div>
            <span className="text-xs font-mono w-12 text-muted-foreground">{formatNum(skipGrad)}</span>
          </div>
        )}
        <div className="flex items-center gap-2 pt-1 border-t border-muted/20">
          <span className="text-xs text-muted-foreground w-20 text-right font-medium">Total</span>
          <div className="flex-1 h-5 bg-muted/30 rounded-full overflow-hidden">
            <div
              className={cn(
                'h-full rounded-full transition-all duration-300',
                Math.abs(totalGrad) < 0.1 ? 'bg-rose-500/60' : 'bg-violet-500/60',
              )}
              style={{ width: `${Math.min((Math.abs(totalGrad) / maxGrad) * 100, 100)}%` }}
            />
          </div>
          <span className={cn(
            'text-xs font-mono w-12 font-semibold',
            Math.abs(totalGrad) < 0.1 ? 'text-rose-400' : 'text-foreground',
          )}>
            {formatNum(totalGrad)}
          </span>
        </div>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

type ResNetBlockExplorerProps = {
  width?: number
  height?: number
}

export function ResNetBlockExplorer({ width: widthOverride }: ResNetBlockExplorerProps) {
  const { containerRef, width: measuredWidth } = useContainerWidth(560)
  const width = widthOverride ?? measuredWidth

  const [mode, setMode] = useState<BlockMode>('residual')
  const [convWeight, setConvWeight] = useState(0.0)
  const [inputValue] = useState(5.0)

  const sim = useMemo(
    () => computeSimulation(inputValue, convWeight, mode),
    [inputValue, convWeight, mode],
  )

  return (
    <div ref={containerRef} className="space-y-5">
      {/* Mode toggle */}
      <div className="flex items-center gap-2">
        <span className="text-xs text-muted-foreground font-medium">Block type:</span>
        <div className="flex rounded-full bg-muted/30 p-0.5">
          <button
            onClick={() => setMode('plain')}
            className={cn(
              'px-3 py-1 rounded-full text-xs font-medium transition-colors cursor-pointer',
              mode === 'plain'
                ? 'bg-amber-500/20 text-amber-300'
                : 'text-muted-foreground hover:text-foreground',
            )}
          >
            Plain Block
          </button>
          <button
            onClick={() => setMode('residual')}
            className={cn(
              'px-3 py-1 rounded-full text-xs font-medium transition-colors cursor-pointer',
              mode === 'residual'
                ? 'bg-emerald-500/20 text-emerald-300'
                : 'text-muted-foreground hover:text-foreground',
            )}
          >
            Residual Block
          </button>
        </div>
      </div>

      {/* Block diagram */}
      <BlockDiagram mode={mode} convWeight={convWeight} sim={sim} width={width} />

      {/* Stats */}
      <div className="flex flex-wrap gap-2">
        <StatBadge label="Input (x)" value={formatNum(sim.input)} color="sky" />
        <StatBadge label="F(x)" value={formatNum(sim.convOutput)} color="amber" />
        <StatBadge
          label="Output"
          value={formatNum(sim.blockOutput)}
          color={mode === 'residual' ? 'emerald' : 'violet'}
        />
        {mode === 'residual' && Math.abs(convWeight) < 0.05 && (
          <StatBadge label="Identity!" value="output = x" color="emerald" />
        )}
        {mode === 'plain' && Math.abs(convWeight) < 0.05 && (
          <StatBadge label="Near zero!" value="input lost" color="rose" />
        )}
      </div>

      {/* Conv weight slider */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <label className="text-xs text-muted-foreground font-medium">
            Conv weight (w): <span className="font-mono text-foreground">{formatNum(convWeight)}</span>
          </label>
          <div className="flex gap-1.5">
            <button
              onClick={() => setConvWeight(0)}
              className="px-2 py-0.5 text-[10px] rounded bg-muted/30 text-muted-foreground hover:text-foreground transition-colors cursor-pointer"
            >
              w = 0
            </button>
            <button
              onClick={() => setConvWeight(0.5)}
              className="px-2 py-0.5 text-[10px] rounded bg-muted/30 text-muted-foreground hover:text-foreground transition-colors cursor-pointer"
            >
              w = 0.5
            </button>
            <button
              onClick={() => setConvWeight(1.0)}
              className="px-2 py-0.5 text-[10px] rounded bg-muted/30 text-muted-foreground hover:text-foreground transition-colors cursor-pointer"
            >
              w = 1.0
            </button>
          </div>
        </div>
        <input
          type="range"
          min={-1}
          max={2}
          step={0.01}
          value={convWeight}
          onChange={(e) => setConvWeight(parseFloat(e.target.value))}
          className="w-full h-2 bg-muted/30 rounded-lg appearance-none cursor-ew-resize"
        />
        <div className="flex justify-between text-[9px] text-muted-foreground/50 font-mono">
          <span>-1.0</span>
          <span>0.0</span>
          <span>1.0</span>
          <span>2.0</span>
        </div>
      </div>

      {/* Gradient visualization */}
      <GradientBar
        convGrad={sim.gradientConvPath}
        skipGrad={sim.gradientSkipPath}
        totalGrad={sim.gradientTotal}
        mode={mode}
      />

      {/* Explanation */}
      <div className="text-xs text-muted-foreground space-y-1.5 bg-muted/10 rounded-lg p-3">
        {mode === 'residual' ? (
          <>
            <p>
              <strong className="text-emerald-400">Residual block:</strong>{' '}
              Output = F(x) + x. The skip connection adds the input directly to the conv output.
            </p>
            {Math.abs(convWeight) < 0.05 && (
              <p className="text-emerald-400">
                With weights near zero, F(x) is near zero and the output equals x.
                The block acts as an <strong>identity function</strong>&mdash;harmless.
              </p>
            )}
            {Math.abs(convWeight) >= 0.05 && (
              <p>
                The conv layers learn a <strong>correction</strong> (residual) on top of the identity.
                The network edits its input rather than rewriting from scratch.
              </p>
            )}
          </>
        ) : (
          <>
            <p>
              <strong className="text-amber-400">Plain block:</strong>{' '}
              Output = F(x) = w * x. No shortcut. The conv layers must learn the entire mapping.
            </p>
            {Math.abs(convWeight) < 0.05 && (
              <p className="text-rose-400">
                With weights near zero, the output is near zero&mdash;the input is <strong>lost</strong>.
                The block cannot act as identity. This is the degradation problem.
              </p>
            )}
          </>
        )}
      </div>
    </div>
  )
}

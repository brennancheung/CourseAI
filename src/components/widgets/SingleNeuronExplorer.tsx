'use client'

import { useState, useRef, useEffect } from 'react'
import { Circle, Line, Text, Arrow, Rect } from 'react-konva'
import { ZoomableCanvas } from '@/components/canvas/ZoomableCanvas'

/**
 * SingleNeuronExplorer - Interactive widget showing what a single neuron computes
 *
 * Students can adjust:
 * - Two inputs (x₁, x₂)
 * - Two weights (w₁, w₂)
 * - Bias (b)
 *
 * And see the output: y = w₁x₁ + w₂x₂ + b
 *
 * This connects directly to linear regression while introducing neuron terminology.
 */

type SingleNeuronExplorerProps = {
  /** Width override */
  width?: number
  /** Height override */
  height?: number
}

export function SingleNeuronExplorer({
  width: widthOverride,
  height: heightOverride,
}: SingleNeuronExplorerProps) {
  // Inputs
  const [x1, setX1] = useState(1.0)
  const [x2, setX2] = useState(0.5)

  // Weights
  const [w1, setW1] = useState(0.7)
  const [w2, setW2] = useState(-0.3)

  // Bias
  const [bias, setBias] = useState(0.1)

  // Measure container
  const containerRef = useRef<HTMLDivElement>(null)
  const [measuredWidth, setMeasuredWidth] = useState(500)

  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    const updateWidth = () => {
      const rect = container.getBoundingClientRect()
      setMeasuredWidth(rect.width)
    }

    updateWidth()
    const observer = new ResizeObserver(updateWidth)
    observer.observe(container)
    return () => observer.disconnect()
  }, [])

  const width = widthOverride ?? measuredWidth
  const height = heightOverride ?? 280

  // Compute neuron output
  const weightedSum = w1 * x1 + w2 * x2
  const output = weightedSum + bias

  // Layout positions for the diagram
  const inputX = 80
  const neuronX = width / 2
  const outputX = width - 80
  const centerY = height / 2

  const input1Y = centerY - 50
  const input2Y = centerY + 50

  // Color based on output magnitude - smooth gradient from red through gray to green
  const getOutputColor = (val: number) => {
    // Clamp to reasonable range for color mapping
    const clamped = Math.max(-2, Math.min(2, val))
    // Map -2 to +2 onto 0 to 1
    const t = (clamped + 2) / 4

    // Interpolate: red (t=0) -> gray (t=0.5) -> green (t=1)
    if (t < 0.5) {
      // Red to gray
      const ratio = t * 2 // 0 to 1
      const r = Math.round(239 + (107 - 239) * ratio)
      const g = Math.round(68 + (114 - 68) * ratio)
      const b = Math.round(68 + (128 - 68) * ratio)
      return `rgb(${r}, ${g}, ${b})`
    }
    // Gray to green
    const ratio = (t - 0.5) * 2 // 0 to 1
    const r = Math.round(107 + (34 - 107) * ratio)
    const g = Math.round(114 + (197 - 114) * ratio)
    const b = Math.round(128 + (94 - 128) * ratio)
    return `rgb(${r}, ${g}, ${b})`
  }

  return (
    <div ref={containerRef} className="flex flex-col gap-4">
      {/* Diagram */}
      <div className="rounded-lg border bg-card overflow-hidden">
        <ZoomableCanvas width={width} height={height} backgroundColor="#1a1a2e">
          {/* Input nodes */}
          <Circle
            x={inputX}
            y={input1Y}
            radius={25}
            fill="#3b82f6"
            stroke="#60a5fa"
            strokeWidth={2}
          />
          <Text
            x={inputX - 12}
            y={input1Y - 8}
            text={`x₁`}
            fontSize={14}
            fill="white"
            fontStyle="bold"
          />
          <Text
            x={inputX - 20}
            y={input1Y + 30}
            text={x1.toFixed(1)}
            fontSize={12}
            fill="#60a5fa"
          />

          <Circle
            x={inputX}
            y={input2Y}
            radius={25}
            fill="#3b82f6"
            stroke="#60a5fa"
            strokeWidth={2}
          />
          <Text
            x={inputX - 12}
            y={input2Y - 8}
            text={`x₂`}
            fontSize={14}
            fill="white"
            fontStyle="bold"
          />
          <Text
            x={inputX - 20}
            y={input2Y + 30}
            text={x2.toFixed(1)}
            fontSize={12}
            fill="#60a5fa"
          />

          {/* Connection lines with weights */}
          <Line
            points={[inputX + 25, input1Y, neuronX - 35, centerY - 10]}
            stroke="#f97316"
            strokeWidth={2}
            opacity={0.8}
          />
          <Text
            x={(inputX + neuronX) / 2 - 10}
            y={input1Y - 25}
            text={`w₁=${w1.toFixed(1)}`}
            fontSize={11}
            fill="#f97316"
          />

          <Line
            points={[inputX + 25, input2Y, neuronX - 35, centerY + 10]}
            stroke="#f97316"
            strokeWidth={2}
            opacity={0.8}
          />
          <Text
            x={(inputX + neuronX) / 2 - 10}
            y={input2Y + 15}
            text={`w₂=${w2.toFixed(1)}`}
            fontSize={11}
            fill="#f97316"
          />

          {/* Neuron (center) */}
          <Circle
            x={neuronX}
            y={centerY}
            radius={35}
            fill="#1e1e3f"
            stroke="#8b5cf6"
            strokeWidth={3}
          />
          <Text
            x={neuronX - 8}
            y={centerY - 8}
            text="Σ"
            fontSize={20}
            fill="#8b5cf6"
            fontStyle="bold"
          />

          {/* Bias arrow */}
          <Arrow
            points={[neuronX, centerY + 70, neuronX, centerY + 40]}
            stroke="#10b981"
            fill="#10b981"
            strokeWidth={2}
            pointerLength={6}
            pointerWidth={6}
          />
          <Text
            x={neuronX - 25}
            y={centerY + 75}
            text={`b=${bias.toFixed(1)}`}
            fontSize={11}
            fill="#10b981"
          />

          {/* Output connection - consistent color to avoid visual shift */}
          <Arrow
            points={[neuronX + 35, centerY, outputX - 30, centerY]}
            stroke="#9ca3af"
            fill="#9ca3af"
            strokeWidth={2}
            pointerLength={8}
            pointerWidth={8}
          />

          {/* Output node */}
          <Circle
            x={outputX}
            y={centerY}
            radius={30}
            fill={getOutputColor(output)}
            stroke="white"
            strokeWidth={2}
          />
          <Text
            x={outputX - 20}
            y={centerY - 8}
            text={output.toFixed(2)}
            fontSize={14}
            fill="white"
            fontStyle="bold"
          />
          <Text
            x={outputX - 8}
            y={centerY + 35}
            text="y"
            fontSize={12}
            fill="#9ca3af"
          />

          {/* Formula at bottom */}
          <Rect
            x={width / 2 - 140}
            y={height - 40}
            width={280}
            height={30}
            fill="#1e1e3f"
            cornerRadius={4}
          />
          <Text
            x={width / 2 - 130}
            y={height - 32}
            text={`y = (${w1.toFixed(1)}×${x1.toFixed(1)}) + (${w2.toFixed(1)}×${x2.toFixed(1)}) + ${bias.toFixed(1)} = ${output.toFixed(2)}`}
            fontSize={12}
            fill="#9ca3af"
            fontFamily="monospace"
          />
        </ZoomableCanvas>
      </div>

      {/* Output color legend */}
      <div className="flex items-center justify-center gap-2 text-xs text-muted-foreground">
        <span>Output:</span>
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 rounded-full bg-red-500" />
          <span>negative</span>
        </div>
        <div className="w-8 h-1 rounded bg-gradient-to-r from-red-500 via-gray-500 to-green-500" />
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 rounded-full bg-green-500" />
          <span>positive</span>
        </div>
      </div>

      {/* Controls */}
      <div className="grid gap-4 md:grid-cols-2">
        {/* Inputs */}
        <div className="space-y-3 p-3 rounded-lg bg-blue-500/10 border border-blue-500/20">
          <p className="text-sm font-medium text-blue-400">Inputs</p>
          <div className="space-y-2">
            <div className="flex items-center gap-3">
              <label className="text-sm text-muted-foreground w-8">x₁:</label>
              <input
                type="range"
                min={-2}
                max={2}
                step={0.1}
                value={x1}
                onChange={(e) => setX1(parseFloat(e.target.value))}
                className="flex-1"
              />
              <span className="font-mono text-sm w-12 text-right">{x1.toFixed(1)}</span>
            </div>
            <div className="flex items-center gap-3">
              <label className="text-sm text-muted-foreground w-8">x₂:</label>
              <input
                type="range"
                min={-2}
                max={2}
                step={0.1}
                value={x2}
                onChange={(e) => setX2(parseFloat(e.target.value))}
                className="flex-1"
              />
              <span className="font-mono text-sm w-12 text-right">{x2.toFixed(1)}</span>
            </div>
          </div>
        </div>

        {/* Weights */}
        <div className="space-y-3 p-3 rounded-lg bg-orange-500/10 border border-orange-500/20">
          <p className="text-sm font-medium text-orange-400">Weights</p>
          <div className="space-y-2">
            <div className="flex items-center gap-3">
              <label className="text-sm text-muted-foreground w-8">w₁:</label>
              <input
                type="range"
                min={-2}
                max={2}
                step={0.1}
                value={w1}
                onChange={(e) => setW1(parseFloat(e.target.value))}
                className="flex-1"
              />
              <span className="font-mono text-sm w-12 text-right">{w1.toFixed(1)}</span>
            </div>
            <div className="flex items-center gap-3">
              <label className="text-sm text-muted-foreground w-8">w₂:</label>
              <input
                type="range"
                min={-2}
                max={2}
                step={0.1}
                value={w2}
                onChange={(e) => setW2(parseFloat(e.target.value))}
                className="flex-1"
              />
              <span className="font-mono text-sm w-12 text-right">{w2.toFixed(1)}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Bias */}
      <div className="p-3 rounded-lg bg-emerald-500/10 border border-emerald-500/20">
        <div className="flex items-center gap-3">
          <label className="text-sm font-medium text-emerald-400 w-16">Bias (b):</label>
          <input
            type="range"
            min={-2}
            max={2}
            step={0.1}
            value={bias}
            onChange={(e) => setBias(parseFloat(e.target.value))}
            className="flex-1"
          />
          <span className="font-mono text-sm w-12 text-right">{bias.toFixed(1)}</span>
        </div>
      </div>

      {/* Insight */}
      <p className="text-sm text-muted-foreground text-center">
        This is identical to <span className="font-mono">y = w₁x₁ + w₂x₂ + b</span> — multi-input linear regression!
      </p>
    </div>
  )
}

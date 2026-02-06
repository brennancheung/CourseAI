'use client'

import { useState, useRef, useEffect } from 'react'
import { Circle, Line, Text, Arrow } from 'react-konva'
import { ZoomableCanvas } from '@/components/canvas/ZoomableCanvas'

/**
 * NetworkDiagramWidget - Shows a simple neural network with data flowing through
 *
 * Visualizes:
 * - Input layer → Hidden layer → Output layer
 * - Connections between neurons
 * - Values flowing through (animated or static)
 *
 * Helps students see the structure before diving into math.
 */

type NetworkDiagramWidgetProps = {
  /** Width override */
  width?: number
  /** Height override */
  height?: number
  /** Show animated data flow */
  animated?: boolean
}

// Network structure: 2 inputs → 3 hidden → 2 outputs
const LAYER_SIZES = [2, 3, 2]
const LAYER_LABELS = ['Input', 'Hidden', 'Output']
const LAYER_COLORS = ['#3b82f6', '#8b5cf6', '#22c55e']

export function NetworkDiagramWidget({
  width: widthOverride,
  height: heightOverride,
  animated: _animated = false,
}: NetworkDiagramWidgetProps) {
  // Sample input values
  const [inputs, setInputs] = useState([0.8, 0.3])

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
  const height = heightOverride ?? 320

  // Calculate neuron positions
  const layerSpacing = width / (LAYER_SIZES.length + 1)
  const getLayerX = (layerIndex: number) => layerSpacing * (layerIndex + 1)

  const getNeuronPositions = (layerIndex: number) => {
    const layerSize = LAYER_SIZES[layerIndex]
    const layerX = getLayerX(layerIndex)
    const totalHeight = (layerSize - 1) * 70
    const startY = (height - 50) / 2 - totalHeight / 2

    return Array.from({ length: layerSize }, (_, i) => ({
      x: layerX,
      y: startY + i * 70,
    }))
  }

  // Pre-compute all positions
  const allPositions = LAYER_SIZES.map((_, i) => getNeuronPositions(i))

  // Simple forward pass with random weights (for demo)
  const hiddenValues = [
    Math.max(0, inputs[0] * 0.5 + inputs[1] * 0.3 + 0.1),
    Math.max(0, inputs[0] * -0.2 + inputs[1] * 0.8 + 0.2),
    Math.max(0, inputs[0] * 0.4 + inputs[1] * -0.4 + 0.3),
  ]
  const outputValues = [
    hiddenValues[0] * 0.6 + hiddenValues[1] * 0.3 + hiddenValues[2] * -0.2,
    hiddenValues[0] * -0.3 + hiddenValues[1] * 0.5 + hiddenValues[2] * 0.7,
  ]

  const allValues = [inputs, hiddenValues, outputValues]

  return (
    <div ref={containerRef} className="flex flex-col gap-4">
      {/* Diagram */}
      <div className="rounded-lg border bg-card overflow-hidden">
        <ZoomableCanvas width={width} height={height} backgroundColor="#1a1a2e">
          {/* Draw connections first (behind neurons) */}
          {LAYER_SIZES.slice(0, -1).map((_, layerIndex) => {
            const fromPositions = allPositions[layerIndex]
            const toPositions = allPositions[layerIndex + 1]

            return fromPositions.flatMap((from, fromIdx) =>
              toPositions.map((to, toIdx) => (
                <Line
                  key={`conn-${layerIndex}-${fromIdx}-${toIdx}`}
                  points={[from.x + 20, from.y, to.x - 20, to.y]}
                  stroke="#4a4a6a"
                  strokeWidth={1}
                  opacity={0.5}
                />
              ))
            )
          })}

          {/* Draw neurons */}
          {LAYER_SIZES.map((_layerSize, layerIndex) => {
            const positions = allPositions[layerIndex]
            const color = LAYER_COLORS[layerIndex]

            return positions.map((pos, neuronIdx) => (
              <Circle
                key={`neuron-${layerIndex}-${neuronIdx}`}
                x={pos.x}
                y={pos.y}
                radius={20}
                fill="#1e1e3f"
                stroke={color}
                strokeWidth={3}
              />
            ))
          })}

          {/* Draw values inside neurons */}
          {LAYER_SIZES.map((_layerSize, layerIndex) => {
            const positions = allPositions[layerIndex]
            const values = allValues[layerIndex]

            return positions.map((pos, neuronIdx) => (
              <Text
                key={`value-${layerIndex}-${neuronIdx}`}
                x={pos.x - 14}
                y={pos.y - 6}
                text={values[neuronIdx].toFixed(1)}
                fontSize={11}
                fill="#e5e5e5"
                fontFamily="monospace"
              />
            ))
          })}

          {/* Layer labels */}
          {LAYER_LABELS.map((label, layerIndex) => (
            <Text
              key={`label-${layerIndex}`}
              x={getLayerX(layerIndex) - 25}
              y={height - 35}
              text={label}
              fontSize={12}
              fill={LAYER_COLORS[layerIndex]}
              fontStyle="bold"
            />
          ))}

          {/* Arrows showing data flow */}
          <Arrow
            points={[40, height / 2 - 25, 80, height / 2 - 25]}
            stroke="#666"
            fill="#666"
            strokeWidth={2}
            pointerLength={6}
            pointerWidth={6}
          />
          <Text
            x={20}
            y={height / 2 - 45}
            text="data"
            fontSize={10}
            fill="#666"
          />
        </ZoomableCanvas>
      </div>

      {/* Input controls */}
      <div className="grid gap-4 md:grid-cols-2">
        <div className="space-y-2 p-3 rounded-lg bg-blue-500/10 border border-blue-500/20">
          <p className="text-sm font-medium text-blue-400">Input Values</p>
          <div className="flex items-center gap-3">
            <label className="text-sm text-muted-foreground w-8">x₁:</label>
            <input
              type="range"
              min={0}
              max={1}
              step={0.1}
              value={inputs[0]}
              onChange={(e) => setInputs([parseFloat(e.target.value), inputs[1]])}
              className="flex-1"
            />
            <span className="font-mono text-sm w-10">{inputs[0].toFixed(1)}</span>
          </div>
          <div className="flex items-center gap-3">
            <label className="text-sm text-muted-foreground w-8">x₂:</label>
            <input
              type="range"
              min={0}
              max={1}
              step={0.1}
              value={inputs[1]}
              onChange={(e) => setInputs([inputs[0], parseFloat(e.target.value)])}
              className="flex-1"
            />
            <span className="font-mono text-sm w-10">{inputs[1].toFixed(1)}</span>
          </div>
        </div>

        <div className="p-3 rounded-lg bg-muted/30 border">
          <p className="text-sm font-medium mb-2">What&apos;s Happening</p>
          <ul className="text-xs text-muted-foreground space-y-1">
            <li>• Each connection has a weight (not shown)</li>
            <li>• Hidden neurons combine inputs differently</li>
            <li>• Output neurons combine hidden values</li>
            <li>• Change inputs and watch values flow!</li>
          </ul>
        </div>
      </div>
    </div>
  )
}

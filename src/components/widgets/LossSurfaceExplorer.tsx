'use client'

import { useState, useMemo, useRef } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls, Text, Line } from '@react-three/drei'
import * as THREE from 'three'

/**
 * LossSurfaceExplorer - 3D visualization of loss landscape
 *
 * Shows how MSE varies as slope and intercept change.
 * Features:
 * - 3D surface plot of loss
 * - Draggable point on surface
 * - Connected 2D line fitting view
 *
 * Used in:
 * - Lesson 1.1.3: Loss Functions
 * - Lesson 1.1.4: Gradient Descent (with gradient arrows)
 */

interface DataPoint {
  x: number
  y: number
}

// Fixed data points for consistency
const DATA_POINTS: DataPoint[] = [
  { x: -2, y: -0.8 },
  { x: -1.5, y: -0.3 },
  { x: -1, y: 0.2 },
  { x: -0.5, y: 0.8 },
  { x: 0, y: 0.5 },
  { x: 0.5, y: 1.2 },
  { x: 1, y: 1.5 },
  { x: 1.5, y: 1.8 },
  { x: 2, y: 2.3 },
]

// Calculate MSE for given slope and intercept
function calculateMSE(slope: number, intercept: number, points: DataPoint[]): number {
  let sum = 0
  for (const p of points) {
    const predicted = slope * p.x + intercept
    sum += Math.pow(p.y - predicted, 2)
  }
  return sum / points.length
}

// Find optimal parameters analytically (for showing the minimum)
function findOptimalParams(points: DataPoint[]): { slope: number; intercept: number } {
  const n = points.length
  let sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0

  for (const p of points) {
    sumX += p.x
    sumY += p.y
    sumXY += p.x * p.y
    sumX2 += p.x * p.x
  }

  const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX)
  const intercept = (sumY - slope * sumX) / n

  return { slope, intercept }
}

// Generate surface mesh data
function generateSurfaceData(
  slopeRange: [number, number],
  interceptRange: [number, number],
  resolution: number
): { positions: Float32Array; indices: Uint16Array; minLoss: number; maxLoss: number } {
  const [slopeMin, slopeMax] = slopeRange
  const [intMin, intMax] = interceptRange

  const positions: number[] = []
  const indices: number[] = []
  let minLoss = Infinity
  let maxLoss = -Infinity

  // Generate vertices
  for (let i = 0; i <= resolution; i++) {
    for (let j = 0; j <= resolution; j++) {
      const slope = slopeMin + (slopeMax - slopeMin) * (i / resolution)
      const intercept = intMin + (intMax - intMin) * (j / resolution)
      const loss = calculateMSE(slope, intercept, DATA_POINTS)

      minLoss = Math.min(minLoss, loss)
      maxLoss = Math.max(maxLoss, loss)

      // Position: x=slope, z=intercept, y=loss (clamped for visualization)
      positions.push(slope, Math.min(loss, 5), intercept)
    }
  }

  // Generate indices for triangles
  for (let i = 0; i < resolution; i++) {
    for (let j = 0; j < resolution; j++) {
      const a = i * (resolution + 1) + j
      const b = a + 1
      const c = a + resolution + 1
      const d = c + 1

      indices.push(a, b, c)
      indices.push(b, d, c)
    }
  }

  return {
    positions: new Float32Array(positions),
    indices: new Uint16Array(indices),
    minLoss,
    maxLoss,
  }
}

// The 3D surface mesh
function LossSurface() {
  const { positions, indices } = useMemo(
    () => generateSurfaceData([-1, 2], [-1, 2], 30),
    []
  )

  const geometry = useMemo(() => {
    const geo = new THREE.BufferGeometry()
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3))
    geo.setIndex(new THREE.BufferAttribute(indices, 1))
    geo.computeVertexNormals()
    return geo
  }, [positions, indices])

  return (
    <mesh geometry={geometry}>
      <meshStandardMaterial
        color="#4f46e5"
        transparent
        opacity={0.7}
        side={THREE.DoubleSide}
        wireframe={false}
      />
    </mesh>
  )
}

// Wireframe overlay for better depth perception
function LossSurfaceWireframe() {
  const { positions, indices } = useMemo(
    () => generateSurfaceData([-1, 2], [-1, 2], 15),
    []
  )

  const geometry = useMemo(() => {
    const geo = new THREE.BufferGeometry()
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3))
    geo.setIndex(new THREE.BufferAttribute(indices, 1))
    return geo
  }, [positions, indices])

  return (
    <lineSegments geometry={new THREE.WireframeGeometry(geometry)}>
      <lineBasicMaterial color="#818cf8" transparent opacity={0.3} />
    </lineSegments>
  )
}

// Draggable marker on the surface
interface MarkerProps {
  position: [number, number, number]
  onDrag: (slope: number, intercept: number) => void
}

function SurfaceMarker({ position }: MarkerProps) {
  const meshRef = useRef<THREE.Mesh>(null)

  // Animate the marker
  useFrame((state) => {
    if (meshRef.current) {
      meshRef.current.scale.setScalar(1 + Math.sin(state.clock.elapsedTime * 3) * 0.1)
    }
  })

  return (
    <mesh ref={meshRef} position={position}>
      <sphereGeometry args={[0.08, 16, 16]} />
      <meshStandardMaterial color="#f97316" emissive="#f97316" emissiveIntensity={0.5} />
    </mesh>
  )
}

// Minimum point marker
function MinimumMarker({ optimal }: { optimal: { slope: number; intercept: number } }) {
  const loss = calculateMSE(optimal.slope, optimal.intercept, DATA_POINTS)

  return (
    <group position={[optimal.slope, loss, optimal.intercept]}>
      <mesh>
        <sphereGeometry args={[0.06, 16, 16]} />
        <meshStandardMaterial color="#22c55e" emissive="#22c55e" emissiveIntensity={0.3} />
      </mesh>
      <Text
        position={[0, 0.3, 0]}
        fontSize={0.12}
        color="#22c55e"
        anchorX="center"
        anchorY="bottom"
      >
        minimum
      </Text>
    </group>
  )
}

// Axis labels
function AxisLabels() {
  return (
    <>
      <Text position={[2.3, 0, 0]} fontSize={0.15} color="#888">
        slope
      </Text>
      <Text position={[0, 0, 2.3]} fontSize={0.15} color="#888">
        intercept
      </Text>
      <Text position={[-0.3, 3, 0]} fontSize={0.15} color="#888" rotation={[0, Math.PI / 2, 0]}>
        loss
      </Text>
    </>
  )
}

// Grid lines on the base
function BaseGrid() {
  const lines: [THREE.Vector3, THREE.Vector3][] = []

  for (let i = -1; i <= 2; i += 0.5) {
    lines.push([new THREE.Vector3(i, 0, -1), new THREE.Vector3(i, 0, 2)])
    lines.push([new THREE.Vector3(-1, 0, i), new THREE.Vector3(2, 0, i)])
  }

  return (
    <>
      {lines.map((points, i) => (
        <Line key={i} points={points} color="#333" lineWidth={1} />
      ))}
    </>
  )
}

// Main 3D scene
function Scene({ currentSlope, currentIntercept }: { currentSlope: number; currentIntercept: number }) {
  const optimal = useMemo(() => findOptimalParams(DATA_POINTS), [])
  const currentLoss = calculateMSE(currentSlope, currentIntercept, DATA_POINTS)

  return (
    <>
      <ambientLight intensity={0.5} />
      <pointLight position={[5, 10, 5]} intensity={1} />
      <pointLight position={[-5, 5, -5]} intensity={0.5} />

      <LossSurface />
      <LossSurfaceWireframe />
      <BaseGrid />
      <AxisLabels />

      <SurfaceMarker
        position={[currentSlope, Math.min(currentLoss, 5), currentIntercept]}
        onDrag={() => {}}
      />
      <MinimumMarker optimal={optimal} />

      <OrbitControls
        enablePan={false}
        minDistance={3}
        maxDistance={10}
        minPolarAngle={0.2}
        maxPolarAngle={Math.PI / 2.2}
      />
    </>
  )
}

// 2D line preview
function LinePreview({ slope, intercept }: { slope: number; intercept: number }) {
  const width = 300
  const height = 200
  const padding = 30

  const xScale = (x: number) => padding + ((x + 3) / 6) * (width - 2 * padding)
  const yScale = (y: number) => height - padding - ((y + 1) / 4) * (height - 2 * padding)

  const lineY1 = slope * -3 + intercept
  const lineY2 = slope * 3 + intercept

  return (
    <svg width={width} height={height} className="bg-muted/30 rounded-lg">
      {/* Grid lines */}
      {[-2, -1, 0, 1, 2].map((x) => (
        <line
          key={`v${x}`}
          x1={xScale(x)}
          y1={padding}
          x2={xScale(x)}
          y2={height - padding}
          stroke="#333"
          strokeWidth={1}
          opacity={0.3}
        />
      ))}
      {[-1, 0, 1, 2, 3].map((y) => (
        <line
          key={`h${y}`}
          x1={padding}
          y1={yScale(y)}
          x2={width - padding}
          y2={yScale(y)}
          stroke="#333"
          strokeWidth={1}
          opacity={0.3}
        />
      ))}

      {/* Fitted line */}
      <line
        x1={xScale(-3)}
        y1={yScale(lineY1)}
        x2={xScale(3)}
        y2={yScale(lineY2)}
        stroke="#22c55e"
        strokeWidth={2}
      />

      {/* Data points */}
      {DATA_POINTS.map((p, i) => (
        <circle
          key={i}
          cx={xScale(p.x)}
          cy={yScale(p.y)}
          r={4}
          fill="#3b82f6"
        />
      ))}

      {/* Residual lines */}
      {DATA_POINTS.map((p, i) => {
        const predicted = slope * p.x + intercept
        return (
          <line
            key={`r${i}`}
            x1={xScale(p.x)}
            y1={yScale(p.y)}
            x2={xScale(p.x)}
            y2={yScale(predicted)}
            stroke="#ef4444"
            strokeWidth={1}
            opacity={0.5}
          />
        )
      })}
    </svg>
  )
}

export function LossSurfaceExplorer() {
  const [slope, setSlope] = useState(0.5)
  const [intercept, setIntercept] = useState(0.5)

  const mse = calculateMSE(slope, intercept, DATA_POINTS)
  const optimal = useMemo(() => findOptimalParams(DATA_POINTS), [])
  const optimalMSE = calculateMSE(optimal.slope, optimal.intercept, DATA_POINTS)

  return (
    <div className="space-y-4">
      <div className="grid gap-4 lg:grid-cols-2">
        {/* 3D Surface */}
        <div className="rounded-lg border bg-card overflow-hidden" style={{ height: 350 }}>
          <Canvas camera={{ position: [4, 4, 4], fov: 50 }}>
            <Scene currentSlope={slope} currentIntercept={intercept} />
          </Canvas>
        </div>

        {/* 2D Line Preview */}
        <div className="space-y-4">
          <div className="flex justify-center">
            <LinePreview slope={slope} intercept={intercept} />
          </div>

          {/* Sliders */}
          <div className="space-y-3">
            <div className="space-y-1">
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">Slope (w)</span>
                <span className="font-mono">{slope.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="-1"
                max="2"
                step="0.05"
                value={slope}
                onChange={(e) => setSlope(parseFloat(e.target.value))}
                className="w-full"
              />
            </div>

            <div className="space-y-1">
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">Intercept (b)</span>
                <span className="font-mono">{intercept.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="-1"
                max="2"
                step="0.05"
                value={intercept}
                onChange={(e) => setIntercept(parseFloat(e.target.value))}
                className="w-full"
              />
            </div>
          </div>
        </div>
      </div>

      {/* Stats */}
      <div className="flex flex-wrap gap-4 text-sm">
        <div className="px-3 py-2 rounded-md bg-muted">
          <span className="text-muted-foreground">Current MSE: </span>
          <span className="font-mono">{mse.toFixed(3)}</span>
        </div>
        <div className="px-3 py-2 rounded-md bg-emerald-500/10 text-emerald-400">
          <span>Optimal MSE: </span>
          <span className="font-mono">{optimalMSE.toFixed(3)}</span>
          <span className="text-xs ml-2">(w={optimal.slope.toFixed(2)}, b={optimal.intercept.toFixed(2)})</span>
        </div>
      </div>

      <p className="text-xs text-muted-foreground">
        Use the sliders to explore the loss surface. The orange point shows your current position.
        The green point marks the optimal (minimum loss) parameters. Drag to rotate the 3D view.
      </p>
    </div>
  )
}

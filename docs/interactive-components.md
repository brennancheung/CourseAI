# Interactive Components Library Reference

This document covers the visualization and interactive libraries available for building educational components in CourseAI.

## Quick Reference

| Library | Use Case | Import |
|---------|----------|--------|
| **react-konva** | Interactive canvases (primary choice) | `import { Stage, Layer, Circle } from 'react-konva'` |
| **ZoomableCanvas** | Pan/zoom wrapper for react-konva | `import { ZoomableCanvas } from '@/components/canvas/ZoomableCanvas'` |
| **ExpandableWidget** | Fullscreen modal for any widget | `import { ExpandableWidget } from '@/components/widgets/ExpandableWidget'` |
| **Visx** | Custom 2D visualizations, neural net diagrams | `import { Group, Line } from '@visx/visx'` |
| **React Three Fiber** | 3D visualizations | `import { Canvas } from '@react-three/fiber'` |
| **Recharts** | Training curves, metrics, standard charts | `import { LineChart, Line } from 'recharts'` |
| **KaTeX** | Math formula rendering | `import { InlineMath, BlockMath } from 'react-katex'` |

> **Note:** We previously used Mafs but found it unreliable for complex interactions. react-konva provides better control over the canvas and supports pan/zoom natively.

---

## react-konva — Primary Interactive Canvas

Best for: Any interactive visualization with draggable elements, animations, or custom graphics.

### Why react-konva over Mafs?

- **Full control** — Direct pixel manipulation, no viewBox issues
- **Reliable rendering** — Canvas-based, not SVG coordinate transforms
- **Pan/zoom support** — Built-in stage dragging and wheel zoom
- **Touch support** — Pinch-to-zoom works out of the box
- **Performance** — Canvas is faster for many animated elements

### Basic Usage

```tsx
'use client'

import { Stage, Layer, Circle, Line, Text } from 'react-konva'

function SimpleCanvas() {
  return (
    <Stage width={600} height={400} style={{ backgroundColor: '#1a1a2e' }}>
      <Layer>
        <Circle x={100} y={100} radius={20} fill="#f97316" />
        <Line points={[0, 200, 600, 200]} stroke="#666" strokeWidth={2} />
        <Text x={50} y={50} text="Hello" fontSize={16} fill="#fff" />
      </Layer>
    </Stage>
  )
}
```

### Coordinate Transformation Pattern

For math visualizations, create transform functions to convert between math coordinates and pixel coordinates:

```tsx
// Viewport in math coordinates
const VIEW = {
  xMin: -4,
  xMax: 4,
  yMin: -2,
  yMax: 4,
}

// Transform functions
const toPixelX = (x: number) => ((x - VIEW.xMin) / (VIEW.xMax - VIEW.xMin)) * width
const toPixelY = (y: number) => height - ((y - VIEW.yMin) / (VIEW.yMax - VIEW.yMin)) * height

// Usage
<Circle x={toPixelX(mathX)} y={toPixelY(mathY)} radius={10} fill="#f97316" />
```

### Key Konva Components

- `Circle`, `Rect`, `Line`, `Arrow` — Basic shapes
- `Text` — Labels (note: limited font support on canvas)
- `Group` — Group elements together
- `Image` — Render images on canvas

**Docs:** https://konvajs.org/docs/react/

---

## ZoomableCanvas — Pan/Zoom Wrapper

Wraps a react-konva Stage with pan and zoom support.

```tsx
import { ZoomableCanvas } from '@/components/canvas/ZoomableCanvas'
import { Circle, Line } from 'react-konva'

function InteractiveVisualization() {
  return (
    <ZoomableCanvas
      width={600}
      height={400}
      backgroundColor="#1a1a2e"
      minScale={0.25}
      maxScale={4}
    >
      <Circle x={300} y={200} radius={20} fill="#f97316" />
      <Line points={[0, 200, 600, 200]} stroke="#666" />
    </ZoomableCanvas>
  )
}
```

### Features

- **Mouse wheel zoom** — Scroll to zoom, centered on cursor position
- **Trackpad pinch** — Two-finger pinch gesture
- **Touch pinch** — Works on mobile/tablets
- **Drag to pan** — Click and drag to move the canvas
- **Double-click reset** — Returns to initial view

### Props

| Prop | Type | Default | Description |
|------|------|---------|-------------|
| `width` | number | required | Canvas width in pixels |
| `height` | number | required | Canvas height in pixels |
| `backgroundColor` | string | `'#1a1a2e'` | Background color |
| `minScale` | number | `0.25` | Minimum zoom level |
| `maxScale` | number | `4` | Maximum zoom level |
| `initialScale` | number | `1` | Starting zoom level |
| `initialX` | number | `0` | Starting X offset |
| `initialY` | number | `0` | Starting Y offset |

---

## ExpandableWidget — Fullscreen Modal

Wraps any widget to provide a fullscreen expansion option. Essential for complex visualizations that benefit from more space.

```tsx
import { ExpandableWidget } from '@/components/widgets/ExpandableWidget'
import { LinearFitExplorer } from '@/components/widgets/LinearFitExplorer'

function LessonSection() {
  return (
    <ExpandableWidget title="Try Fitting a Line">
      <LinearFitExplorer showResiduals={true} />
    </ExpandableWidget>
  )
}
```

### Features

- **Hover to reveal** — Expand button appears on hover (top-right corner)
- **Fullscreen modal** — Dark backdrop, widget fills available space
- **Keyboard support** — Press ESC to close
- **Click outside** — Click backdrop to close
- **Responsive sizing** — Passes expanded `width` and `height` to child

### How It Works

1. Widget renders inline at its default size
2. User hovers → expand icon appears
3. User clicks expand → modal opens with widget at full size
4. Widget receives new `width` and `height` props automatically

### Requirements for Child Widgets

Child widgets must accept optional `width` and `height` props:

```tsx
type MyWidgetProps = {
  // ... other props
  width?: number
  height?: number
}

export function MyWidget({ width = 600, height = 400 }: MyWidgetProps) {
  return (
    <ZoomableCanvas width={width} height={height}>
      {/* ... */}
    </ZoomableCanvas>
  )
}
```

---

## Custom Widgets Library

Pre-built interactive components for common ML concepts. Import directly from their files (no barrel exports).

### LinearFitExplorer

Interactive line fitting with draggable slope/intercept controls.

```tsx
import { LinearFitExplorer } from '@/components/widgets/LinearFitExplorer'

<ExpandableWidget title="Linear Regression">
  <LinearFitExplorer
    showResiduals={true}      // Show error lines to the fit
    showMSE={true}            // Display MSE calculation
    initialSlope={0.5}
    initialIntercept={0}
    width={600}               // Optional, has default
    height={400}              // Optional, has default
  />
</ExpandableWidget>
```

**Features:**
- Draggable numbers in equation (drag left/right to adjust)
- Sliders for fine control
- MSE display with color feedback (green=optimal, red=poor)
- Shows optimal solution for comparison
- Pan/zoom support via ZoomableCanvas

**Used in:** Linear Regression, Loss Functions lessons

---

### GradientDescentExplorer

Animated gradient descent on a 1D loss curve.

```tsx
import { GradientDescentExplorer } from '@/components/widgets/GradientDescentExplorer'

<ExpandableWidget title="Gradient Descent">
  <GradientDescentExplorer
    showLearningRateSlider={true}
    initialLearningRate={0.15}
    initialPosition={-2.5}
    showGradientArrow={true}
    width={600}
    height={350}
  />
</ExpandableWidget>
```

**Features:**
- Animated ball rolling downhill
- Gradient (red) and update (green) direction arrows
- Step-by-step or continuous animation
- Learning rate slider with sensible default (0.15)
- Position clamping to prevent flying off screen
- Pan/zoom support

**Used in:** Gradient Descent lesson

---

### LearningRateExplorer

Side-by-side comparison of different learning rates.

```tsx
import { LearningRateExplorer } from '@/components/widgets/LearningRateExplorer'

// Comparison mode: shows 4 panels with different LRs
<LearningRateExplorer mode="comparison" />

// Interactive mode: single panel with adjustable LR
<ExpandableWidget title="Find the Sweet Spot">
  <LearningRateExplorer mode="interactive" />
</ExpandableWidget>
```

**Features:**
- Shows too small / just right / too large / diverging
- Clickable preset buttons (0.1, 0.5, 0.9, 1.05)
- Learning rate display next to slider
- Demonstrates oscillation and divergence

**Used in:** Learning Rate Deep Dive lesson

---

### LossSurfaceExplorer

3D visualization of loss landscape with parameter sliders.

```tsx
import { LossSurfaceExplorer } from '@/components/widgets/LossSurfaceExplorer'

<ExpandableWidget title="Loss Landscape">
  <LossSurfaceExplorer />
</ExpandableWidget>
```

**Features:**
- 3D rotatable loss surface
- Sliders for slope/intercept
- Shows optimal minimum point
- Connected 2D line preview

**Used in:** Loss Functions lesson

---

## Canvas Primitives (Atoms)

Reusable building blocks for creating new widgets. Located in `@/components/canvas/primitives/`.

### Grid

```tsx
import { Grid } from '@/components/canvas/primitives/Grid'

<Grid spacing={1} color="#333355" opacity={0.5} />
```

### Axis

```tsx
import { Axis } from '@/components/canvas/primitives/Axis'

<Axis
  showArrows={true}
  color="#666688"
  labelSpacing={1}
  showLabels={true}
  xLabel="θ"
  yLabel="L(θ)"
/>
```

### Curve

```tsx
import { Curve } from '@/components/canvas/primitives/Curve'

<Curve
  fn={(x) => x * x}  // Function to plot
  color="#6366f1"
  strokeWidth={2}
  samples={200}
/>
```

### Ball

```tsx
import { Ball } from '@/components/canvas/primitives/Ball'

<Ball
  x={1}           // Math coordinates
  y={2}
  radius={10}     // Pixels
  color="#f97316"
  label="minimum"
/>
```

---

## Other Libraries

### Visx — Custom 2D SVG Visualizations

Best for: Neural network diagrams, custom data visualizations, anything Recharts can't handle.

```tsx
'use client'
import { Group } from '@visx/group'
import { scaleLinear } from '@visx/scale'

function NeuralNetworkLayer({ nodes, width, height }) {
  const yScale = scaleLinear({
    domain: [0, nodes - 1],
    range: [20, height - 20],
  })

  return (
    <svg width={width} height={height}>
      <Group>
        {Array.from({ length: nodes }).map((_, i) => (
          <circle key={i} cx={width / 2} cy={yScale(i)} r={15} fill="#3b82f6" />
        ))}
      </Group>
    </svg>
  )
}
```

**When to use:** Neural network architecture diagrams, attention visualizations, static diagrams.

**Docs:** https://airbnb.io/visx/

---

### React Three Fiber — 3D Visualizations

Best for: 3D neural networks, high-dimensional data projections, weight space visualizations.

```tsx
'use client'
import { Canvas } from '@react-three/fiber'
import { OrbitControls, Sphere } from '@react-three/drei'

function NeuralNet3D() {
  return (
    <Canvas camera={{ position: [0, 0, 5] }}>
      <ambientLight intensity={0.5} />
      <pointLight position={[10, 10, 10]} />
      <Sphere position={[0, 0, 0]} args={[0.2, 32, 32]}>
        <meshStandardMaterial color="#3b82f6" />
      </Sphere>
      <OrbitControls />
    </Canvas>
  )
}
```

**When to use:** 3D network architectures, loss surface exploration, impressive visualizations.

**Docs:** https://docs.pmnd.rs/react-three-fiber/

---

### Recharts — Standard Charts

Best for: Training curves, loss/accuracy plots, metrics dashboards.

```tsx
'use client'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'

function TrainingCurve({ data }) {
  return (
    <ResponsiveContainer width="100%" height={300}>
      <LineChart data={data}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="epoch" />
        <YAxis />
        <Tooltip />
        <Line type="monotone" dataKey="loss" stroke="#ef4444" />
        <Line type="monotone" dataKey="accuracy" stroke="#22c55e" />
      </LineChart>
    </ResponsiveContainer>
  )
}
```

**Docs:** https://recharts.org/

---

### KaTeX — Math Formula Rendering

```tsx
'use client'
import 'katex/dist/katex.min.css'
import { InlineMath, BlockMath } from 'react-katex'

function GradientFormula() {
  return (
    <div>
      <p>The gradient is <InlineMath math="\nabla L = 2(y - \hat{y})" /></p>
      <BlockMath math={String.raw`\theta_{new} = \theta - \alpha \nabla L`} />
    </div>
  )
}
```

**Docs:** https://katex.org/docs/supported.html

---

## Widget Design Guidelines

When creating new widgets:

1. **Use react-konva** — Primary choice for interactive visualizations
2. **Wrap with ZoomableCanvas** — Users expect pan/zoom on any canvas
3. **Accept width/height props** — Required for ExpandableWidget compatibility
4. **Use ExpandableWidget in lessons** — Always wrap interactive widgets
5. **Create sensible defaults** — Widget should look good without any props
6. **Show values that change** — Display equations/numbers that update with interaction
7. **Consider touch** — Pan/zoom should work on mobile
8. **Add help text** — Brief instruction on how to interact

### Template for New Widgets

```tsx
'use client'

import { useState } from 'react'
import { Circle, Line, Text } from 'react-konva'
import { ZoomableCanvas } from '@/components/canvas/ZoomableCanvas'

type MyWidgetProps = {
  width?: number
  height?: number
  // ... other props
}

export function MyWidget({
  width = 600,
  height = 400,
}: MyWidgetProps) {
  const [value, setValue] = useState(0)

  // Coordinate transforms
  const VIEW = { xMin: -4, xMax: 4, yMin: -2, yMax: 4 }
  const toPixelX = (x: number) => ((x - VIEW.xMin) / (VIEW.xMax - VIEW.xMin)) * width
  const toPixelY = (y: number) => height - ((y - VIEW.yMin) / (VIEW.yMax - VIEW.yMin)) * height

  return (
    <div className="space-y-4">
      <ZoomableCanvas width={width} height={height}>
        {/* Canvas content */}
      </ZoomableCanvas>

      {/* Controls below canvas */}
      <div className="flex gap-4">
        {/* Sliders, buttons, etc. */}
      </div>
    </div>
  )
}
```

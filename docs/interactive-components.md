# Interactive Components Library Reference

This document covers the visualization and interactive libraries available for building educational components in CourseAI.

## Quick Reference

| Library | Use Case | Import |
|---------|----------|--------|
| **react-konva** | Interactive canvases (primary choice) | `import { Stage, Layer, Circle } from 'react-konva'` |
| **ZoomableCanvas** | Pan/zoom wrapper for react-konva | `import { ZoomableCanvas } from '@/components/canvas/ZoomableCanvas'` |
| **ExercisePanel** | Bordered panel with header + fullscreen | `import { ExercisePanel } from '@/components/widgets/ExercisePanel'` |
| **Visx** | Custom 2D visualizations, neural net diagrams | `import { Group, Line } from '@visx/visx'` |
| **React Three Fiber** | 3D visualizations | `import { Canvas } from '@react-three/fiber'` |
| **Recharts** | Training curves, metrics, standard charts | `import { LineChart, Line } from 'recharts'` |
| **KaTeX** | Math formula rendering | `import { InlineMath, BlockMath } from 'react-katex'` |

> **Note:** We previously used Mafs but found it unreliable for complex interactions. react-konva provides better control over the canvas and supports pan/zoom natively.

---

## ExercisePanel — Compound Component for Exercises

**Always wrap interactive widgets in ExercisePanel.** It provides:
- Visible border around the entire exercise
- Header with title and expand button
- Fullscreen modal for larger viewing
- Automatic width/height passing to child widgets

### Basic Usage (Shorthand)

```tsx
import { ExercisePanel } from '@/components/widgets/ExercisePanel'
import { GradientDescentExplorer } from '@/components/widgets/GradientDescentExplorer'

<ExercisePanel title="Watch Gradient Descent in Action">
  <GradientDescentExplorer />
</ExercisePanel>
```

### With Subtitle

```tsx
<ExercisePanel title="Find the Sweet Spot" subtitle="Experiment with different learning rates">
  <LearningRateExplorer mode="interactive" />
</ExercisePanel>
```

### Compound Component Syntax (Advanced)

For custom headers or more control:

```tsx
<ExercisePanel>
  <ExercisePanel.Header title="Custom Title" subtitle="With subtitle" />
  <ExercisePanel.Content>
    <MyWidget />
  </ExercisePanel.Content>
</ExercisePanel>
```

### How It Works

1. **Inline view:** Shows bordered card with header (title + expand icon) and content
2. **User clicks expand:** Modal opens at 90vw width, 90vh max height
3. **ResizeObserver** measures actual container width
4. **Passes dimensions:** Child receives `width` (measured) and `height` (60% of viewport)
5. **Close:** ESC key, click outside, or click minimize icon

### Requirements for Child Widgets

Widgets must accept optional `width` and `height` props with sensible defaults:

```tsx
type MyWidgetProps = {
  width?: number   // Will be ~90vw when expanded
  height?: number  // Will be ~60vh when expanded
}

export function MyWidget({ width = 600, height = 350 }: MyWidgetProps) {
  return (
    <div className="space-y-4">
      <ZoomableCanvas width={width} height={height}>
        {/* Canvas content */}
      </ZoomableCanvas>

      {/* Controls - MUST be centered for wide canvases */}
      <div className="flex flex-wrap gap-4 items-center justify-center">
        {/* buttons, sliders */}
      </div>
    </div>
  )
}
```

### Important: No Duplicate Titles

**Do NOT put a heading above ExercisePanel.** The panel header IS the title:

```tsx
// ❌ WRONG - duplicate titles
<h3>Gradient Descent</h3>
<ExercisePanel title="Gradient Descent">
  ...
</ExercisePanel>

// ✅ CORRECT - panel provides the title
<ExercisePanel title="Gradient Descent">
  ...
</ExercisePanel>
```

---

## Widget Layout Guidelines

### Center Controls When Canvas is Wide

When a canvas expands to fullscreen, left-aligned controls look awkward. **Always center:**

```tsx
{/* Controls */}
<div className="flex flex-wrap gap-4 items-center justify-center">
  <Button>Run</Button>
  <Button>Step</Button>
  <input type="range" />
</div>

{/* Stats */}
<div className="flex flex-wrap gap-4 text-sm justify-center">
  <div className="px-3 py-2 rounded-md bg-muted">Loss: 0.5</div>
</div>

{/* Help text */}
<p className="text-xs text-muted-foreground text-center">
  Click "Step" to move one step at a time.
</p>
```

### Widget Structure Pattern

```tsx
export function MyWidget({ width = 600, height = 350 }: Props) {
  return (
    <div className="space-y-4">
      {/* 1. Canvas - full width */}
      <div className="rounded-lg border bg-card overflow-hidden">
        <ZoomableCanvas width={width} height={height} backgroundColor="#1a1a2e">
          {/* Konva elements */}
        </ZoomableCanvas>
      </div>

      {/* 2. Controls - centered */}
      <div className="flex flex-wrap gap-4 items-center justify-center">
        <Button>...</Button>
        <Slider />
      </div>

      {/* 3. Stats/info - centered */}
      <div className="flex flex-wrap gap-4 text-sm justify-center">
        <div className="px-3 py-2 rounded-md bg-muted">...</div>
      </div>

      {/* 4. Dynamic equation display - centered */}
      <div className="p-3 rounded-md bg-muted/50 font-mono text-sm text-center">
        θ_new = ...
      </div>

      {/* 5. Help text - centered */}
      <p className="text-xs text-muted-foreground text-center">
        Interaction instructions here.
      </p>
    </div>
  )
}
```

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

## Custom Widgets Library

Pre-built interactive components for common ML concepts. Import directly from their files (no barrel exports).

### LinearFitExplorer

Interactive line fitting with draggable slope/intercept controls.

```tsx
import { LinearFitExplorer } from '@/components/widgets/LinearFitExplorer'

<ExercisePanel title="Try Fitting a Line">
  <LinearFitExplorer
    showResiduals={true}      // Show error lines to the fit
    showMSE={true}            // Display MSE calculation
    initialSlope={0.5}
    initialIntercept={0}
  />
</ExercisePanel>
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

<ExercisePanel title="Watch Gradient Descent in Action">
  <GradientDescentExplorer
    showLearningRateSlider={true}
    initialLearningRate={0.15}
    initialPosition={-2.5}
    showGradientArrow={true}
  />
</ExercisePanel>
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
<ExercisePanel title="Find the Sweet Spot" subtitle="Experiment with different learning rates">
  <LearningRateExplorer mode="interactive" />
</ExercisePanel>
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

<ExercisePanel title="Explore the Loss Surface">
  <LossSurfaceExplorer />
</ExercisePanel>
```

**Features:**
- 3D rotatable loss surface
- Sliders for slope/intercept
- Shows optimal minimum point
- Connected 2D line preview

**Used in:** Loss Functions lesson

---

## Lesson Integration Patterns

### Placing Widgets in Lessons

Use the `Row` component for layout. ExercisePanel goes in `Row.Content`:

```tsx
import { Row } from '@/components/layout/Row'
import { ExercisePanel } from '@/components/widgets/ExercisePanel'
import { TryThisBlock } from '@/components/lessons'

<Row>
  <Row.Content>
    <ExercisePanel title="Watch Gradient Descent in Action">
      <GradientDescentExplorer />
    </ExercisePanel>
  </Row.Content>
  <Row.Aside>
    <TryThisBlock title="Experiment">
      <ul className="space-y-2 text-sm">
        <li>Click "Step" to see one update at a time</li>
        <li>Try different learning rates</li>
      </ul>
    </TryThisBlock>
  </Row.Aside>
</Row>
```

### Checklist for Adding a Widget to a Lesson

1. Import the widget and ExercisePanel
2. Wrap widget in ExercisePanel with descriptive title
3. Put ExercisePanel inside `Row.Content`
4. Add a `TryThisBlock` in `Row.Aside` with interaction hints
5. **Do NOT add an h3 or SectionHeader above the panel** — the panel header is the title

---

## Creating New Widgets

### Template

```tsx
'use client'

import { useState, useCallback } from 'react'
import { Circle, Line, Text, Arrow } from 'react-konva'
import { ZoomableCanvas } from '@/components/canvas/ZoomableCanvas'
import { Button } from '@/components/ui/button'

type MyWidgetProps = {
  width?: number
  height?: number
  initialValue?: number
}

export function MyWidget({
  width = 600,
  height = 350,
  initialValue = 0,
}: MyWidgetProps) {
  const [value, setValue] = useState(initialValue)

  // Coordinate transforms for math → pixels
  const VIEW = { xMin: -4, xMax: 4, yMin: -1, yMax: 10 }
  const toPixelX = (x: number) => ((x - VIEW.xMin) / (VIEW.xMax - VIEW.xMin)) * width
  const toPixelY = (y: number) => height - ((y - VIEW.yMin) / (VIEW.yMax - VIEW.yMin)) * height

  const reset = useCallback(() => {
    setValue(initialValue)
  }, [initialValue])

  return (
    <div className="space-y-4">
      {/* Canvas */}
      <div className="rounded-lg border bg-card overflow-hidden">
        <ZoomableCanvas width={width} height={height} backgroundColor="#1a1a2e">
          {/* Grid lines, axes, curve, interactive elements */}
          <Circle x={toPixelX(value)} y={toPixelY(value * value)} radius={12} fill="#f97316" />
        </ZoomableCanvas>
      </div>

      {/* Controls - CENTERED */}
      <div className="flex flex-wrap gap-4 items-center justify-center">
        <Button variant="outline" size="sm" onClick={() => setValue(v => v + 0.1)}>
          Step
        </Button>
        <Button variant="outline" size="sm" onClick={reset}>
          Reset
        </Button>
        <div className="flex items-center gap-2">
          <span className="text-sm text-muted-foreground">Value:</span>
          <input
            type="range"
            min="-3"
            max="3"
            step="0.1"
            value={value}
            onChange={(e) => setValue(parseFloat(e.target.value))}
            className="w-32"
          />
          <span className="font-mono text-sm w-12">{value.toFixed(2)}</span>
        </div>
      </div>

      {/* Stats - CENTERED */}
      <div className="flex flex-wrap gap-4 text-sm justify-center">
        <div className="px-3 py-2 rounded-md bg-muted">
          <span className="text-muted-foreground">Value: </span>
          <span className="font-mono">{value.toFixed(3)}</span>
        </div>
      </div>

      {/* Help text - CENTERED */}
      <p className="text-xs text-muted-foreground text-center">
        Drag the slider or click Step to change the value. The ball shows the current position.
      </p>
    </div>
  )
}
```

### Widget Design Checklist

- [ ] Accept `width` and `height` props with sensible defaults
- [ ] Use `ZoomableCanvas` for pan/zoom support
- [ ] Center all controls with `justify-center`
- [ ] Center all stats with `justify-center`
- [ ] Center help text with `text-center`
- [ ] Wrap canvas in `rounded-lg border bg-card overflow-hidden`
- [ ] Use `space-y-4` for vertical spacing
- [ ] Show updating values (equations, stats) that respond to interaction
- [ ] Add brief help text explaining how to interact
- [ ] Default learning rate around 0.15 (not 0.5 — too high causes divergence)
- [ ] Clamp positions to prevent elements flying off screen

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
</ResponsiveContainer>
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

## Lessons Learned

### Mafs → react-konva Migration

We started with Mafs but found:
- ViewBox issues caused content to render off-screen
- High learning rates made balls "disappear" (flew off viewport)
- No native pan/zoom support
- Coordinate transform bugs

react-konva solved all of these with direct pixel control.

### Default Values Matter

- **Learning rate default: 0.15** (not 0.5) — 0.5 is too high and causes immediate oscillation
- **Initial position: -2 to -2.5** — gives room to show the descent before reaching minimum
- **Canvas height: 350px** inline, **60vh** expanded — good balance of visibility

### Fullscreen Must Show Everything

The fullscreen modal must show the ENTIRE exercise (canvas + controls + stats), not just the canvas. Users need to interact with controls while viewing the larger canvas.

### Avoid Duplicate Titles

When ExercisePanel has a title, don't add an h3 or SectionHeader above it. The panel header IS the title.

### Use Tailwind for Layout, ResizeObserver for Measurement

Don't use `window.innerWidth` directly for layout. Use Tailwind classes (90vw, 90vh) for the modal, then ResizeObserver to measure actual pixel dimensions for the canvas.

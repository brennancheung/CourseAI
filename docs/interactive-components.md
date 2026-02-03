# Interactive Components Library Reference

This document covers the visualization and interactive libraries available for building educational components in CourseAI.

## Quick Reference

| Library | Use Case | Import |
|---------|----------|--------|
| **Mafs** | Interactive math (plots, draggable points) | `import { Mafs, Plot, Point } from 'mafs'` |
| **Visx** | Custom 2D visualizations, neural net diagrams | `import { Group, Line } from '@visx/visx'` |
| **React Three Fiber** | 3D visualizations | `import { Canvas } from '@react-three/fiber'` |
| **Recharts** | Training curves, metrics, standard charts | `import { LineChart, Line } from 'recharts'` |
| **KaTeX** | Math formula rendering | `import { InlineMath, BlockMath } from 'react-katex'` |

---

## Mafs — Interactive Math

Best for: Function plots, coordinate planes, draggable points, geometric visualizations.

```tsx
'use client'
import { Mafs, Coordinates, Plot, Point, useMovablePoint } from 'mafs'
import 'mafs/core.css'

function SigmoidExplorer() {
  const point = useMovablePoint([0, 0.5])

  return (
    <Mafs>
      <Coordinates.Cartesian />
      <Plot.OfX y={(x) => 1 / (1 + Math.exp(-x))} color="#3b82f6" />
      <Point x={point.x} y={point.y} color="#ef4444" />
      {point.element}
    </Mafs>
  )
}
```

**Key features:**
- `useMovablePoint` — Draggable points for exploration
- `Plot.OfX`, `Plot.Parametric` — Function plotting
- `Vector`, `Line`, `Circle`, `Polygon` — Geometric primitives
- `Transform` — Rotate, translate, scale groups of elements

**When to use:** Activation functions, loss landscapes, decision boundaries, linear algebra concepts.

**Docs:** https://mafs.dev/

---

## Visx — Custom 2D Visualizations

Best for: Neural network diagrams, custom data visualizations, anything Recharts can't handle.

```tsx
'use client'
import { Group } from '@visx/group'
import { LinePath } from '@visx/shape'
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
          <circle
            key={i}
            cx={width / 2}
            cy={yScale(i)}
            r={15}
            fill="#3b82f6"
          />
        ))}
      </Group>
    </svg>
  )
}
```

**Key modules:**
- `@visx/shape` — Lines, areas, bars, arcs
- `@visx/scale` — D3 scales for mapping data to pixels
- `@visx/axis` — Axis components
- `@visx/gradient` — SVG gradients
- `@visx/group` — SVG grouping with transforms

**When to use:** Neural network architecture diagrams, attention visualizations, custom animated charts, anything requiring fine SVG control.

**Docs:** https://airbnb.io/visx/

---

## React Three Fiber — 3D Visualizations

Best for: 3D neural networks, high-dimensional data projections, weight space visualizations.

```tsx
'use client'
import { Canvas } from '@react-three/fiber'
import { OrbitControls, Sphere } from '@react-three/drei'

function Neuron3D({ position, color = '#3b82f6' }) {
  return (
    <Sphere position={position} args={[0.2, 32, 32]}>
      <meshStandardMaterial color={color} />
    </Sphere>
  )
}

function NeuralNet3D() {
  return (
    <Canvas camera={{ position: [0, 0, 5] }}>
      <ambientLight intensity={0.5} />
      <pointLight position={[10, 10, 10]} />
      <Neuron3D position={[-1, 0, 0]} />
      <Neuron3D position={[0, 1, 0]} />
      <Neuron3D position={[1, 0, 0]} />
      <OrbitControls />
    </Canvas>
  )
}
```

**Key @react-three/drei helpers:**
- `OrbitControls` — Mouse-controlled camera rotation
- `Sphere`, `Box`, `Plane` — Basic geometry
- `Line` — 3D lines (for connections)
- `Text` — 3D text labels
- `Html` — Embed HTML in 3D scene

**Leva for controls:**
```tsx
import { useControls } from 'leva'

function Scene() {
  const { learningRate } = useControls({ learningRate: { value: 0.01, min: 0, max: 1 } })
  // ... use learningRate
}
```

**When to use:** 3D network architectures, t-SNE/UMAP projections, loss surface exploration, impressive "wow factor" visualizations.

**Docs:** https://docs.pmnd.rs/react-three-fiber/

---

## Recharts — Standard Charts

Best for: Training curves, loss/accuracy plots, metrics dashboards, any standard chart type.

```tsx
'use client'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'

const trainingData = [
  { epoch: 1, loss: 2.5, accuracy: 0.3 },
  { epoch: 2, loss: 1.8, accuracy: 0.5 },
  { epoch: 3, loss: 1.2, accuracy: 0.7 },
  // ...
]

function TrainingCurve() {
  return (
    <ResponsiveContainer width="100%" height={300}>
      <LineChart data={trainingData}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="epoch" />
        <YAxis />
        <Tooltip />
        <Legend />
        <Line type="monotone" dataKey="loss" stroke="#ef4444" />
        <Line type="monotone" dataKey="accuracy" stroke="#22c55e" />
      </LineChart>
    </ResponsiveContainer>
  )
}
```

**Available chart types:** LineChart, AreaChart, BarChart, ScatterChart, PieChart, RadarChart, etc.

**When to use:** Training metrics, performance comparisons, data distributions, any "standard" chart.

**Docs:** https://recharts.org/

---

## KaTeX — Math Formula Rendering

Best for: Inline and block math notation in lessons.

```tsx
'use client'
import 'katex/dist/katex.min.css'
import { InlineMath, BlockMath } from 'react-katex'

function AttentionFormula() {
  return (
    <div>
      <p>
        The attention function is <InlineMath math="softmax(QK^T / \sqrt{d_k})V" />
      </p>

      <BlockMath math={String.raw`
        \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
      `} />
    </div>
  )
}
```

**Tips:**
- Use `String.raw` for complex formulas to avoid escaping backslashes
- Import the CSS once in your layout or component
- `InlineMath` for formulas within text, `BlockMath` for centered display

**When to use:** Any mathematical notation — loss functions, gradients, attention equations, etc.

**Docs:** https://katex.org/docs/supported.html

---

## Decision Guide

### "I want to show a mathematical function"
→ **Mafs** if interactive (user can drag points, explore)
→ **Recharts** if just plotting data points
→ **KaTeX** if just showing the equation

### "I want to visualize a neural network"
→ **Visx** for 2D diagrams (cleaner, simpler)
→ **React Three Fiber** for 3D (more impressive, more complex)

### "I want to show training progress"
→ **Recharts** (LineChart with loss/accuracy)

### "I want to explain attention/transformers"
→ **Mafs** for attention weight heatmaps (interactive)
→ **Visx** for custom attention visualizations
→ **KaTeX** for the math formulas

### "I want something impressive for the landing page"
→ **React Three Fiber** with animated 3D neural network

---

## Common Patterns

### Wrapper Component for Client-Side Only

Many of these libraries need `'use client'` and may have SSR issues:

```tsx
'use client'
import dynamic from 'next/dynamic'

const MafsPlot = dynamic(() => import('./MafsPlot'), { ssr: false })

export function LessonWithPlot() {
  return (
    <div>
      <p>Here's an interactive visualization:</p>
      <MafsPlot />
    </div>
  )
}
```

### Responsive Container

Always wrap visualizations in responsive containers:

```tsx
<div className="w-full aspect-video">
  <Mafs>...</Mafs>
</div>
```

### Loading States

For heavy 3D scenes:

```tsx
import { Suspense } from 'react'

<Canvas>
  <Suspense fallback={null}>
    <HeavyModel />
  </Suspense>
</Canvas>
```

---

## Custom Widgets Library

Pre-built interactive components for common ML concepts. Import from `@/components/widgets`.

### LinearFitExplorer

Interactive line fitting with draggable slope/intercept controls.

```tsx
import { LinearFitExplorer } from '@/components/widgets'

<LinearFitExplorer
  showResiduals={true}      // Show error lines to the fit
  showMSE={true}            // Display MSE calculation
  initialSlope={0.5}
  initialIntercept={0}
  interactive={true}        // Allow dragging
  height={400}
/>
```

**Used in:** Linear Regression, Loss Functions lessons

---

### LossSurfaceExplorer

3D visualization of loss landscape with parameter sliders.

```tsx
import { LossSurfaceExplorer } from '@/components/widgets'

<LossSurfaceExplorer />
```

**Features:**
- 3D rotatable loss surface
- Sliders for slope/intercept
- Shows optimal minimum point
- Connected 2D line preview

**Used in:** Loss Functions lesson

---

### GradientDescentExplorer

Animated gradient descent on a 1D loss curve.

```tsx
import { GradientDescentExplorer } from '@/components/widgets'

<GradientDescentExplorer
  showLearningRateSlider={true}
  initialLearningRate={0.3}
  initialPosition={-2}
  showGradientArrow={true}
  // Optional: custom loss function
  lossFunction={(x) => x * x}
  lossFunctionDerivative={(x) => 2 * x}
/>
```

**Features:**
- Animated ball rolling downhill
- Gradient and update direction arrows
- Step-by-step or continuous animation
- Configurable learning rate

**Used in:** Gradient Descent lesson

---

### LearningRateExplorer

Side-by-side comparison of different learning rates.

```tsx
import { LearningRateExplorer } from '@/components/widgets'

// Comparison mode: shows 4 panels with different LRs
<LearningRateExplorer mode="comparison" />

// Interactive mode: single panel with adjustable LR
<LearningRateExplorer mode="interactive" />
```

**Features:**
- Shows too small / just right / too large / diverging
- Demonstrates oscillation and divergence
- Clear visual comparison

**Used in:** Learning Rate Deep Dive lesson

---

## Widget Design Guidelines

When creating new widgets:

1. **Make them reusable** — Accept props for customization
2. **Document usage** — Add JSDoc comments and update this file
3. **Consider mobile** — Use responsive sizing
4. **Add controls** — Sliders, buttons, toggles increase engagement
5. **Show the math** — Display equations/values that change with interaction
6. **Export from index.ts** — Add to `@/components/widgets/index.ts`

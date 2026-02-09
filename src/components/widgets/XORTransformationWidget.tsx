'use client'

/**
 * XORTransformationWidget - Shows how the hidden layer transforms space
 *
 * The key insight: neural networks don't "draw multiple lines" in input space.
 * They TRANSFORM the input space so that a single line works.
 *
 * Left panel: Original (A, B) space - XOR points NOT linearly separable
 * Right panel: Transformed (h₁, h₂) space - same points, NOW separable
 *
 * Uses plain SVG with viewBox for crisp, responsive rendering.
 */

// XOR data points
const XOR_POINTS = [
  { a: 0, b: 0, label: 0 },
  { a: 0, b: 1, label: 1 },
  { a: 1, b: 0, label: 1 },
  { a: 1, b: 1, label: 0 },
] as const

// Hidden layer transformation:
// h1 = ReLU(A + B - 0.5)  -- fires when at least one input is ~1
// h2 = ReLU(A + B - 1.5)  -- fires only when both inputs are ~1
function transform(a: number, b: number) {
  return {
    h1: Math.max(0, a + b - 0.5),
    h2: Math.max(0, a + b - 1.5),
  }
}

const TRANSFORMED = XOR_POINTS.map((p) => ({ ...p, ...transform(p.a, p.b) }))

const FILL = { 0: '#3b82f6', 1: '#f97316' } as const
const STROKE = { 0: '#93c5fd', 1: '#fdba74' } as const

// SVG viewBox dimensions
const VW = 320
const VH = 300

// Plot area within the viewBox
const PLOT = { left: 50, right: 300, top: 20, bottom: 265 }
const PW = PLOT.right - PLOT.left
const PH = PLOT.bottom - PLOT.top

function DataPoint({ x, y, label }: { x: number; y: number; label: 0 | 1 }) {
  return (
    <g>
      <circle cx={x} cy={y} r={20} fill={FILL[label]} stroke={STROKE[label]} strokeWidth={2} />
      <text x={x} y={y + 6} textAnchor="middle" fill="white" fontSize={16} fontWeight="bold">
        {label}
      </text>
    </g>
  )
}

// Left panel: Original (A, B) space, data range 0-1 with padding
function mapLeftX(v: number) {
  return PLOT.left + ((v + 0.15) / 1.3) * PW
}
function mapLeftY(v: number) {
  return PLOT.bottom - ((v + 0.15) / 1.3) * PH
}

// Right panel: Transformed (h₁, h₂) space
// h1 range: 0-1.75, h2 range: 0-0.5, with padding
function mapRightX(v: number) {
  return PLOT.left + ((v + 0.1) / 2.1) * PW
}
function mapRightY(v: number) {
  return PLOT.bottom - ((v + 0.06) / 0.72) * PH
}

function OriginalSpaceSVG() {
  return (
    <svg viewBox={`0 0 ${VW} ${VH}`} className="w-full rounded-lg border bg-[#1a1a2e]">
      {/* Axes */}
      <line x1={PLOT.left} y1={PLOT.bottom} x2={PLOT.right} y2={PLOT.bottom} stroke="#4a4a6a" strokeWidth={1.5} />
      <line x1={PLOT.left} y1={PLOT.bottom} x2={PLOT.left} y2={PLOT.top} stroke="#4a4a6a" strokeWidth={1.5} />
      <text x={PLOT.right - 2} y={PLOT.bottom + 18} fill="#6b7280" fontSize={14} textAnchor="end">A</text>
      <text x={PLOT.left - 8} y={PLOT.top + 4} fill="#6b7280" fontSize={14} textAnchor="end">B</text>

      {/* Grid lines at 0 and 1 */}
      <line x1={mapLeftX(0)} y1={PLOT.bottom} x2={mapLeftX(0)} y2={PLOT.top} stroke="#4a4a6a" strokeDasharray="3 5" />
      <line x1={mapLeftX(1)} y1={PLOT.bottom} x2={mapLeftX(1)} y2={PLOT.top} stroke="#4a4a6a" strokeDasharray="3 5" />
      <line x1={PLOT.left} y1={mapLeftY(0)} x2={PLOT.right} y2={mapLeftY(0)} stroke="#4a4a6a" strokeDasharray="3 5" />
      <line x1={PLOT.left} y1={mapLeftY(1)} x2={PLOT.right} y2={mapLeftY(1)} stroke="#4a4a6a" strokeDasharray="3 5" />

      {/* Tick labels */}
      <text x={mapLeftX(0)} y={PLOT.bottom + 18} textAnchor="middle" fill="#6b7280" fontSize={13}>0</text>
      <text x={mapLeftX(1)} y={PLOT.bottom + 18} textAnchor="middle" fill="#6b7280" fontSize={13}>1</text>
      <text x={PLOT.left - 10} y={mapLeftY(0) + 5} textAnchor="end" fill="#6b7280" fontSize={13}>0</text>
      <text x={PLOT.left - 10} y={mapLeftY(1) + 5} textAnchor="end" fill="#6b7280" fontSize={13}>1</text>

      {/* Failed separating line */}
      <line
        x1={mapLeftX(-0.1)} y1={mapLeftY(0.5)}
        x2={mapLeftX(1.1)} y2={mapLeftY(0.5)}
        stroke="#ef4444" strokeWidth={2} strokeDasharray="6 4" opacity={0.6}
      />
      <text x={mapLeftX(0.5)} y={mapLeftY(0.72)} textAnchor="middle" fill="#ef4444" fontSize={14}>
        No line works!
      </text>

      {/* XOR Points */}
      {XOR_POINTS.map((p, i) => (
        <DataPoint key={i} x={mapLeftX(p.a)} y={mapLeftY(p.b)} label={p.label} />
      ))}
    </svg>
  )
}

function TransformedSpaceSVG() {
  return (
    <svg viewBox={`0 0 ${VW} ${VH}`} className="w-full rounded-lg border bg-[#1a1a2e]">
      {/* Axes */}
      <line x1={PLOT.left} y1={PLOT.bottom} x2={PLOT.right} y2={PLOT.bottom} stroke="#4a4a6a" strokeWidth={1.5} />
      <line x1={PLOT.left} y1={PLOT.bottom} x2={PLOT.left} y2={PLOT.top} stroke="#4a4a6a" strokeWidth={1.5} />
      <text x={PLOT.right - 2} y={PLOT.bottom + 18} fill="#6b7280" fontSize={14} textAnchor="end">h₁</text>
      <text x={PLOT.left - 8} y={PLOT.top + 4} fill="#6b7280" fontSize={14} textAnchor="end">h₂</text>

      {/* Separating line: h1 - 3*h2 = 0.25 */}
      {/* At h2=0: h1=0.25. At h2=0.5: h1=1.75 */}
      <line
        x1={mapRightX(0.25)} y1={mapRightY(0)}
        x2={mapRightX(1.75)} y2={mapRightY(0.5)}
        stroke="#22c55e" strokeWidth={3}
      />
      <text x={mapRightX(1.1)} y={mapRightY(0.33)} fill="#22c55e" fontSize={14}>
        One line works!
      </text>

      {/* Transformed Points */}
      {TRANSFORMED.map((p, i) => (
        <DataPoint key={i} x={mapRightX(p.h1)} y={mapRightY(p.h2)} label={p.label} />
      ))}

      {/* Coordinate labels showing original inputs */}
      <text x={mapRightX(0)} y={mapRightY(0) + 32} textAnchor="middle" fill="#9ca3af" fontSize={12}>
        (0,0)
      </text>
      <text x={mapRightX(0.5)} y={mapRightY(0) + 32} textAnchor="middle" fill="#9ca3af" fontSize={12}>
        (0,1) &amp; (1,0)
      </text>
      <text x={mapRightX(1.5)} y={mapRightY(0.5) - 28} textAnchor="middle" fill="#9ca3af" fontSize={12}>
        (1,1)
      </text>
    </svg>
  )
}

export function XORTransformationWidget() {
  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-wrap justify-center items-center gap-4">
        {/* Left Panel: Original Space */}
        <div className="flex flex-col items-center flex-1 min-w-[280px] max-w-[440px]">
          <p className="text-sm font-medium text-rose-400 mb-2">
            Original Space (A, B)
          </p>
          <OriginalSpaceSVG />
          <p className="text-sm text-muted-foreground mt-2 text-center">
            Can&apos;t draw one line to separate orange from blue
          </p>
        </div>

        {/* Arrow between panels */}
        <div className="flex flex-col items-center justify-center shrink-0">
          <div className="text-3xl text-emerald-400">→</div>
          <p className="text-sm text-emerald-400 text-center max-w-[100px]">
            Hidden layer transforms
          </p>
        </div>

        {/* Right Panel: Transformed Space */}
        <div className="flex flex-col items-center flex-1 min-w-[280px] max-w-[440px]">
          <p className="text-sm font-medium text-emerald-400 mb-2">
            Transformed Space (h₁, h₂)
          </p>
          <TransformedSpaceSVG />
          <p className="text-sm text-muted-foreground mt-2 text-center">
            Same points, new positions — now separable!
          </p>
        </div>
      </div>

      {/* Transformation formulas */}
      <div className="p-4 rounded-lg bg-muted/30 border">
        <p className="text-sm font-medium mb-3">The Hidden Layer Transformation:</p>
        <div className="grid gap-2 md:grid-cols-2 text-sm">
          <div className="p-2 rounded bg-violet-500/10 border border-violet-500/20">
            <p className="font-mono text-violet-300">h₁ = ReLU(A + B - 0.5)</p>
            <p className="text-xs text-muted-foreground mt-1">
              Fires when A + B {'>'} 0.5 (at least one input is 1)
            </p>
          </div>
          <div className="p-2 rounded bg-violet-500/10 border border-violet-500/20">
            <p className="font-mono text-violet-300">h₂ = ReLU(A + B - 1.5)</p>
            <p className="text-xs text-muted-foreground mt-1">
              Fires only when A + B {'>'} 1.5 (both inputs are 1)
            </p>
          </div>
        </div>
        <div className="mt-3 text-sm text-muted-foreground">
          <p>
            <strong>Notice:</strong> (0,1) and (1,0) land at the same spot — that&apos;s fine,
            they have the same label! The key is that <span className="text-emerald-400">(1,1)
            moved away</span> from where (0,0) is, making them separable.
          </p>
        </div>
      </div>

      {/* The insight */}
      <div className="p-4 rounded-lg bg-emerald-500/10 border border-emerald-500/20">
        <p className="text-sm">
          <strong className="text-emerald-400">The insight:</strong> The hidden layer
          doesn&apos;t &quot;draw multiple lines.&quot; It <em>transforms the geometry</em> of
          the problem. Points that were tangled together get pulled apart. Then the output
          layer draws <strong>one line</strong> in the new space.
        </p>
      </div>
    </div>
  )
}

'use client'

import { useState, useMemo } from 'react'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts'
import { useContainerWidth } from '@/hooks/useContainerWidth'
import 'katex/dist/katex.min.css'
import { InlineMath } from 'react-katex'

// ---------------------------------------------------------------------------
// Data: fixed logits for "The best programming language is ___"
// ---------------------------------------------------------------------------

type TokenLogit = {
  token: string
  logit: number
}

const TOKEN_LOGITS: TokenLogit[] = [
  { token: 'Python', logit: 5.2 },
  { token: 'Java', logit: 4.1 },
  { token: 'JavaScript', logit: 3.8 },
  { token: 'C', logit: 3.3 },
  { token: 'Rust', logit: 2.9 },
  { token: 'Go', logit: 2.5 },
  { token: 'TypeScript', logit: 2.3 },
  { token: 'C++', logit: 2.1 },
  { token: 'Ruby', logit: 1.5 },
  { token: 'Swift', logit: 1.2 },
  { token: 'Kotlin', logit: 0.9 },
  { token: 'PHP', logit: 0.6 },
  { token: 'Haskell', logit: 0.2 },
  { token: 'Perl', logit: -0.3 },
  { token: 'COBOL', logit: -1.1 },
]

// ---------------------------------------------------------------------------
// Softmax with temperature
// ---------------------------------------------------------------------------

function softmaxWithTemperature(
  logits: number[],
  temperature: number,
): number[] {
  const scaled = logits.map((l) => l / temperature)
  const maxScaled = Math.max(...scaled)
  const exps = scaled.map((s) => Math.exp(s - maxScaled))
  const sumExps = exps.reduce((a, b) => a + b, 0)
  return exps.map((e) => e / sumExps)
}

// ---------------------------------------------------------------------------
// Color helpers
// ---------------------------------------------------------------------------

function barFillColor(probability: number, maxProb: number): string {
  const intensity = probability / maxProb
  if (intensity > 0.8) return '#8b5cf6' // violet-500
  if (intensity > 0.5) return '#a78bfa' // violet-400
  if (intensity > 0.2) return '#c4b5fd' // violet-300
  return '#ddd6fe' // violet-200
}

function temperatureLabel(temperature: number): string {
  if (temperature <= 0.3) return 'Very greedy — almost always picks the top token'
  if (temperature <= 0.7) return 'Low temperature — strongly favors likely tokens'
  if (temperature <= 1.1) return 'Standard — balanced between likely and surprising'
  if (temperature <= 2.0) return 'High temperature — more random, surprising choices'
  return 'Very high — nearly uniform, almost random'
}

// ---------------------------------------------------------------------------
// Custom tooltip
// ---------------------------------------------------------------------------

type TooltipPayloadEntry = {
  payload: {
    token: string
    probability: number
    logit: number
  }
}

function CustomTooltip({
  active,
  payload,
}: {
  active?: boolean
  payload?: TooltipPayloadEntry[]
}) {
  if (!active || !payload || payload.length === 0) return null

  const data = payload[0].payload
  return (
    <div className="bg-popover border border-border rounded-md px-3 py-2 text-sm shadow-md">
      <p className="font-medium text-foreground">&ldquo;{data.token}&rdquo;</p>
      <p className="text-muted-foreground">
        Probability: {(data.probability * 100).toFixed(1)}%
      </p>
      <p className="text-muted-foreground/70 text-xs">
        Raw logit: {data.logit.toFixed(1)}
      </p>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export function TemperatureExplorer({
  width: widthOverride,
}: {
  width?: number
  height?: number
}) {
  const { containerRef, width: measuredWidth } = useContainerWidth(600)
  const width = widthOverride ?? measuredWidth

  const [temperature, setTemperature] = useState(1.0)

  const chartData = useMemo(() => {
    const logits = TOKEN_LOGITS.map((t) => t.logit)
    const probs = softmaxWithTemperature(logits, temperature)
    return TOKEN_LOGITS.map((t, i) => ({
      token: t.token,
      logit: t.logit,
      probability: probs[i],
    }))
  }, [temperature])

  const maxProb = useMemo(
    () => Math.max(...chartData.map((d) => d.probability)),
    [chartData],
  )

  const topToken = chartData[0]
  const entropy = useMemo(() => {
    return -chartData.reduce((sum, d) => {
      if (d.probability <= 0) return sum
      return sum + d.probability * Math.log2(d.probability)
    }, 0)
  }, [chartData])

  const chartHeight = 320
  const barSize = Math.max(Math.min(Math.floor((width - 100) / TOKEN_LOGITS.length) - 4, 36), 12)

  return (
    <div ref={containerRef} className="space-y-4">
      {/* Context prompt */}
      <div className="px-4 py-3 bg-muted/50 rounded-lg">
        <p className="text-sm text-muted-foreground">
          <span className="font-medium text-foreground">Context:</span>{' '}
          &ldquo;The best programming language is{' '}
          <span className="text-violet-400 font-medium">___</span>&rdquo;
        </p>
      </div>

      {/* Bar chart */}
      <ResponsiveContainer width="100%" height={chartHeight}>
        <BarChart
          data={chartData}
          margin={{ top: 10, right: 10, left: 10, bottom: 60 }}
          barSize={barSize}
        >
          <XAxis
            dataKey="token"
            tick={{ fontSize: 11, fill: 'var(--muted-foreground)' }}
            angle={-45}
            textAnchor="end"
            interval={0}
            axisLine={{ stroke: 'var(--border)' }}
            tickLine={{ stroke: 'var(--border)' }}
          />
          <YAxis
            tick={{ fontSize: 10, fill: 'var(--muted-foreground)' }}
            tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
            domain={[0, 'auto']}
            axisLine={{ stroke: 'var(--border)' }}
            tickLine={{ stroke: 'var(--border)' }}
            width={45}
          />
          <Tooltip
            content={<CustomTooltip />}
            cursor={{ fill: 'var(--muted)', opacity: 0.3 }}
          />
          <Bar dataKey="probability" radius={[3, 3, 0, 0]}>
            {chartData.map((entry) => (
              <Cell
                key={entry.token}
                fill={barFillColor(entry.probability, maxProb)}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>

      {/* Temperature slider */}
      <div className="space-y-2 px-2">
        <div className="flex items-center justify-between">
          <label className="text-sm font-medium text-foreground">
            Temperature (T)
          </label>
          <span className="text-sm font-mono text-violet-400 font-semibold">
            {temperature.toFixed(1)}
          </span>
        </div>
        <input
          type="range"
          min={0.1}
          max={3.0}
          step={0.1}
          value={temperature}
          onChange={(e) => setTemperature(parseFloat(e.target.value))}
          className="w-full accent-violet-500"
        />
        <div className="flex justify-between text-xs text-muted-foreground/60">
          <span>0.1 (greedy)</span>
          <span>1.0 (standard)</span>
          <span>3.0 (random)</span>
        </div>
        <p className="text-xs text-muted-foreground text-center mt-1">
          {temperatureLabel(temperature)}
        </p>
      </div>

      {/* Stats row */}
      <div className="flex gap-3 flex-wrap">
        <div className="flex-1 min-w-[140px] px-3 py-2 bg-muted/50 rounded-md">
          <p className="text-xs text-muted-foreground">Top token</p>
          <p className="text-sm font-medium text-foreground">
            &ldquo;{topToken.token}&rdquo;{' '}
            <span className="text-violet-400">
              {(topToken.probability * 100).toFixed(1)}%
            </span>
          </p>
        </div>
        <div className="flex-1 min-w-[140px] px-3 py-2 bg-muted/50 rounded-md">
          <p className="text-xs text-muted-foreground">Entropy</p>
          <p className="text-sm font-medium text-foreground">
            {entropy.toFixed(2)} bits{' '}
            <span className="text-xs text-muted-foreground/70">
              (max = {Math.log2(TOKEN_LOGITS.length).toFixed(2)})
            </span>
          </p>
        </div>
      </div>

      {/* Formula */}
      <div className="px-4 py-3 bg-muted/30 rounded-lg text-center">
        <p className="text-xs text-muted-foreground mb-2">
          The formula: divide logits by temperature, then apply softmax
        </p>
        <InlineMath math={`P(\\text{token}_i) = \\text{softmax}\\left(\\frac{\\text{logit}_i}{T}\\right) = \\frac{e^{\\text{logit}_i / T}}{\\sum_j e^{\\text{logit}_j / T}}`} />
      </div>
    </div>
  )
}

'use client'

import { useState, useMemo, useCallback } from 'react'
import { cn } from '@/lib/utils'

// ---------------------------------------------------------------------------
// BPE Algorithm Types
// ---------------------------------------------------------------------------

type MergeRecord = {
  pair: [string, string]
  newToken: string
  count: number
}

type BpeState = {
  tokens: string[]
  merges: MergeRecord[]
  step: number
  vocabSize: number
}

// ---------------------------------------------------------------------------
// Color palette for token boundaries
// ---------------------------------------------------------------------------

const TOKEN_COLORS = [
  'bg-violet-500/20 text-violet-300 border-violet-500/30',
  'bg-sky-500/20 text-sky-300 border-sky-500/30',
  'bg-amber-500/20 text-amber-300 border-amber-500/30',
  'bg-emerald-500/20 text-emerald-300 border-emerald-500/30',
  'bg-rose-500/20 text-rose-300 border-rose-500/30',
  'bg-cyan-500/20 text-cyan-300 border-cyan-500/30',
  'bg-orange-500/20 text-orange-300 border-orange-500/30',
  'bg-pink-500/20 text-pink-300 border-pink-500/30',
  'bg-indigo-500/20 text-indigo-300 border-indigo-500/30',
  'bg-lime-500/20 text-lime-300 border-lime-500/30',
]

function tokenColor(index: number): string {
  return TOKEN_COLORS[index % TOKEN_COLORS.length]
}

// ---------------------------------------------------------------------------
// BPE Algorithm Implementation
// ---------------------------------------------------------------------------

function getInitialTokens(text: string): string[] {
  // Start with character-level tokenization
  // Add word boundary markers (spaces become visible tokens)
  return text.split('')
}

function getPairCounts(tokens: string[]): Map<string, number> {
  const counts = new Map<string, number>()
  for (let i = 0; i < tokens.length - 1; i++) {
    const key = `${tokens[i]}|||${tokens[i + 1]}`
    counts.set(key, (counts.get(key) ?? 0) + 1)
  }
  return counts
}

function getMostFrequentPair(
  counts: Map<string, number>,
): { pair: [string, string]; count: number } | null {
  let bestKey: string | null = null
  let bestCount = 0

  for (const [key, count] of counts) {
    if (count > bestCount) {
      bestKey = key
      bestCount = count
    }
  }

  if (!bestKey || bestCount < 2) return null

  const [first, second] = bestKey.split('|||')
  return { pair: [first, second], count: bestCount }
}

function mergePair(tokens: string[], pair: [string, string]): string[] {
  const result: string[] = []
  let i = 0
  while (i < tokens.length) {
    if (i < tokens.length - 1 && tokens[i] === pair[0] && tokens[i + 1] === pair[1]) {
      result.push(pair[0] + pair[1])
      i += 2
    } else {
      result.push(tokens[i])
      i += 1
    }
  }
  return result
}

function computeAllMerges(text: string, maxMerges: number = 50): BpeState[] {
  const states: BpeState[] = []
  let tokens = getInitialTokens(text)

  // Track unique tokens for vocab size
  const baseVocab = new Set(tokens)

  states.push({
    tokens: [...tokens],
    merges: [],
    step: 0,
    vocabSize: baseVocab.size,
  })

  const allMerges: MergeRecord[] = []
  const vocab = new Set(baseVocab)

  for (let step = 0; step < maxMerges; step++) {
    const counts = getPairCounts(tokens)
    const best = getMostFrequentPair(counts)
    if (!best) break

    const newToken = best.pair[0] + best.pair[1]
    tokens = mergePair(tokens, best.pair)
    vocab.add(newToken)

    const merge: MergeRecord = {
      pair: best.pair,
      newToken,
      count: best.count,
    }
    allMerges.push(merge)

    states.push({
      tokens: [...tokens],
      merges: [...allMerges],
      step: step + 1,
      vocabSize: vocab.size,
    })
  }

  return states
}

// ---------------------------------------------------------------------------
// Display helpers
// ---------------------------------------------------------------------------

function displayToken(token: string): string {
  // Make whitespace visible
  return token.replace(/ /g, '␣') // ␣ symbol for space
}

// ---------------------------------------------------------------------------
// BPE Visualizer Component
// ---------------------------------------------------------------------------

type BpeVisualizerProps = {
  width?: number
  height?: number
}

const DEFAULT_TEXT = 'low low low low low lower lower newest newest'

export function BpeVisualizer({ width: _widthOverride }: BpeVisualizerProps) {
  const [inputText, setInputText] = useState(DEFAULT_TEXT)
  const [currentStep, setCurrentStep] = useState(0)

  // Compute all possible merge states
  const allStates = useMemo(() => {
    if (!inputText.trim()) return []
    return computeAllMerges(inputText)
  }, [inputText])

  const currentState = allStates[currentStep] ?? allStates[0]
  const maxSteps = allStates.length - 1
  const isAtStart = currentStep === 0
  const isAtEnd = currentStep >= maxSteps

  const handleStep = useCallback(() => {
    if (currentStep < maxSteps) {
      setCurrentStep((s) => s + 1)
    }
  }, [currentStep, maxSteps])

  const handleRunAll = useCallback(() => {
    setCurrentStep(maxSteps)
  }, [maxSteps])

  const handleReset = useCallback(() => {
    setCurrentStep(0)
  }, [])

  const handleTextChange = useCallback((text: string) => {
    setInputText(text)
    setCurrentStep(0)
  }, [])

  // The merge that just happened (if any)
  const latestMerge =
    currentStep > 0 ? currentState?.merges[currentStep - 1] : null

  if (!currentState) {
    return (
      <div className="space-y-4">
        <TextInput value={inputText} onChange={handleTextChange} />
        <p className="text-sm text-muted-foreground">Enter some text to tokenize.</p>
      </div>
    )
  }

  return (
    <div className="space-y-5">
      {/* Text input */}
      <TextInput value={inputText} onChange={handleTextChange} />

      {/* Stats bar */}
      <div className="flex flex-wrap gap-3">
        <StatBadge label="Step" value={`${currentStep} / ${maxSteps}`} color="violet" />
        <StatBadge
          label="Tokens"
          value={String(currentState.tokens.length)}
          color="sky"
        />
        <StatBadge
          label="Vocab size"
          value={String(currentState.vocabSize)}
          color="emerald"
        />
        {allStates.length > 1 && (
          <StatBadge
            label="Compression"
            value={`${((1 - currentState.tokens.length / allStates[0].tokens.length) * 100).toFixed(0)}%`}
            color="amber"
          />
        )}
      </div>

      {/* Latest merge info */}
      {latestMerge && (
        <div className="px-4 py-3 bg-violet-500/10 border border-violet-500/20 rounded-lg">
          <p className="text-sm text-muted-foreground">
            <strong className="text-violet-300">Merge #{currentStep}:</strong>{' '}
            <span className="font-mono">
              &ldquo;{displayToken(latestMerge.pair[0])}&rdquo; + &ldquo;{displayToken(latestMerge.pair[1])}&rdquo;
            </span>{' '}
            {'→'}{' '}
            <span className="font-mono font-semibold text-violet-300">
              &ldquo;{displayToken(latestMerge.newToken)}&rdquo;
            </span>{' '}
            <span className="text-muted-foreground/70">
              (appeared {latestMerge.count} times)
            </span>
          </p>
        </div>
      )}

      {isAtStart && (
        <div className="px-4 py-3 bg-muted/50 border border-border rounded-lg">
          <p className="text-sm text-muted-foreground">
            <strong className="text-foreground">Starting state:</strong>{' '}
            Character-level tokenization. Each character is its own token.
            Click <strong>Step</strong> to apply the first BPE merge.
          </p>
        </div>
      )}

      {isAtEnd && currentStep > 0 && (
        <div className="px-4 py-3 bg-emerald-500/10 border border-emerald-500/20 rounded-lg">
          <p className="text-sm text-muted-foreground">
            <strong className="text-emerald-300">Done!</strong>{' '}
            No more pairs appear 2+ times. The merge table has {currentStep} entries.
          </p>
        </div>
      )}

      {/* Token visualization */}
      <div className="min-h-[60px]">
        <p className="text-xs text-muted-foreground/70 mb-2 font-medium">
          Current tokens ({currentState.tokens.length}):
        </p>
        <div className="flex flex-wrap gap-1.5">
          {currentState.tokens.map((token, i) => {
            const isNewlyMerged =
              latestMerge != null && token === latestMerge.newToken
            return (
              <span
                key={`${i}-${token}-${currentStep}`}
                className={cn(
                  'inline-flex items-center px-2 py-1 rounded-md text-sm font-mono border transition-all',
                  isNewlyMerged
                    ? 'bg-violet-500/30 text-violet-200 border-violet-400/50 ring-1 ring-violet-400/30'
                    : tokenColor(i),
                )}
              >
                {displayToken(token)}
              </span>
            )
          })}
        </div>
      </div>

      {/* Controls */}
      <div className="flex items-center gap-3">
        <button
          onClick={handleReset}
          disabled={isAtStart}
          className={cn(
            'px-3 py-1.5 rounded-md text-sm font-medium transition-colors cursor-pointer',
            isAtStart
              ? 'bg-muted text-muted-foreground/40 cursor-not-allowed'
              : 'bg-muted hover:bg-muted/80 text-muted-foreground',
          )}
        >
          Reset
        </button>
        <button
          onClick={handleStep}
          disabled={isAtEnd}
          className={cn(
            'px-4 py-1.5 rounded-md text-sm font-medium transition-colors cursor-pointer',
            isAtEnd
              ? 'bg-primary/30 text-primary-foreground/40 cursor-not-allowed'
              : 'bg-primary text-primary-foreground hover:bg-primary/90',
          )}
        >
          Step
        </button>
        <button
          onClick={handleRunAll}
          disabled={isAtEnd}
          className={cn(
            'px-3 py-1.5 rounded-md text-sm font-medium transition-colors cursor-pointer',
            isAtEnd
              ? 'bg-muted text-muted-foreground/40 cursor-not-allowed'
              : 'bg-muted hover:bg-muted/80 text-muted-foreground',
          )}
        >
          Run All
        </button>
      </div>

      {/* Merge history table */}
      {currentState.merges.length > 0 && (
        <div>
          <p className="text-xs text-muted-foreground/70 mb-2 font-medium">
            Merge table:
          </p>
          <div className="max-h-[180px] overflow-y-auto rounded-lg border border-border">
            <table className="w-full text-sm">
              <thead className="sticky top-0 bg-muted/80 backdrop-blur-sm">
                <tr className="text-left text-muted-foreground/70">
                  <th className="px-3 py-1.5 font-medium">#</th>
                  <th className="px-3 py-1.5 font-medium">Pair</th>
                  <th className="px-3 py-1.5 font-medium">{'→'}</th>
                  <th className="px-3 py-1.5 font-medium text-right">Count</th>
                </tr>
              </thead>
              <tbody>
                {currentState.merges.map((m, i) => (
                  <tr
                    key={i}
                    className={cn(
                      'border-t border-border/50',
                      i === currentStep - 1
                        ? 'bg-violet-500/10'
                        : 'hover:bg-muted/30',
                    )}
                  >
                    <td className="px-3 py-1.5 text-muted-foreground/50 font-mono">
                      {i + 1}
                    </td>
                    <td className="px-3 py-1.5 font-mono">
                      &ldquo;{displayToken(m.pair[0])}&rdquo; + &ldquo;{displayToken(m.pair[1])}&rdquo;
                    </td>
                    <td className="px-3 py-1.5 font-mono font-semibold text-violet-300">
                      &ldquo;{displayToken(m.newToken)}&rdquo;
                    </td>
                    <td className="px-3 py-1.5 text-right text-muted-foreground/70">
                      {m.count}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function TextInput({
  value,
  onChange,
}: {
  value: string
  onChange: (v: string) => void
}) {
  return (
    <div>
      <label className="text-xs text-muted-foreground/70 font-medium block mb-1.5">
        Input text:
      </label>
      <input
        type="text"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder="Type text to tokenize..."
        className="w-full px-3 py-2 rounded-md bg-muted/50 border border-border text-sm text-foreground placeholder:text-muted-foreground/40 focus:outline-none focus:ring-1 focus:ring-primary"
      />
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
  color: 'violet' | 'sky' | 'emerald' | 'amber'
}) {
  const colorClasses = {
    violet: 'bg-violet-500/10 text-violet-300 border-violet-500/20',
    sky: 'bg-sky-500/10 text-sky-300 border-sky-500/20',
    emerald: 'bg-emerald-500/10 text-emerald-300 border-emerald-500/20',
    amber: 'bg-amber-500/10 text-amber-300 border-amber-500/20',
  }

  return (
    <span
      className={cn(
        'inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium border',
        colorClasses[color],
      )}
    >
      <span className="text-muted-foreground/70">{label}:</span>
      {value}
    </span>
  )
}

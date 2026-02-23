'use client'

import { useState, useCallback } from 'react'
import { Shield, ShieldAlert, ShieldCheck, ShieldX, ChevronRight, RotateCcw } from 'lucide-react'
import { Button } from '@/components/ui/button'

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

type SafetyLayer = 'blocklist' | 'textClassifier' | 'inferenceGuide' | 'outputClassifier'

type LayerResult = 'pass' | 'caught' | 'modified'

type PromptScenario = {
  id: string
  prompt: string
  description: string
  category: 'safe' | 'unsafe' | 'adversarial' | 'ambiguous'
  /** Result at each layer when that layer is active */
  results: Record<SafetyLayer, LayerResult>
  /** What the layer does to this prompt (short explanation) */
  explanations: Record<SafetyLayer, string>
}

type LayerInfo = {
  id: SafetyLayer
  name: string
  shortName: string
  description: string
  color: string
  bgColor: string
  borderColor: string
  icon: 'shield' | 'shieldAlert' | 'shieldCheck' | 'shieldX'
}

// ─────────────────────────────────────────────────────────────────────────────
// Data
// ─────────────────────────────────────────────────────────────────────────────

const LAYERS: LayerInfo[] = [
  {
    id: 'blocklist',
    name: 'Keyword Blocklist',
    shortName: 'Blocklist',
    description: 'Exact keyword matching against banned terms',
    color: 'text-amber-400',
    bgColor: 'bg-amber-500/10',
    borderColor: 'border-amber-500/30',
    icon: 'shield',
  },
  {
    id: 'textClassifier',
    name: 'Text Embedding Classifier',
    shortName: 'Text Classifier',
    description: 'Semantic text analysis via embeddings',
    color: 'text-blue-400',
    bgColor: 'bg-blue-500/10',
    borderColor: 'border-blue-500/30',
    icon: 'shieldAlert',
  },
  {
    id: 'inferenceGuide',
    name: 'Inference-Time Guidance (SLD)',
    shortName: 'SLD Guidance',
    description: 'Safe Latent Diffusion steers generation away from unsafe content',
    color: 'text-violet-400',
    bgColor: 'bg-violet-500/10',
    borderColor: 'border-violet-500/30',
    icon: 'shieldCheck',
  },
  {
    id: 'outputClassifier',
    name: 'Output Image Classifier',
    shortName: 'Output Classifier',
    description: 'CLIP-based image classification on generated output',
    color: 'text-emerald-400',
    bgColor: 'bg-emerald-500/10',
    borderColor: 'border-emerald-500/30',
    icon: 'shieldX',
  },
]

const SCENARIOS: PromptScenario[] = [
  {
    id: 'safe-sunset',
    prompt: 'A beautiful sunset over the ocean',
    description: 'Completely safe, no issues',
    category: 'safe',
    results: {
      blocklist: 'pass',
      textClassifier: 'pass',
      inferenceGuide: 'pass',
      outputClassifier: 'pass',
    },
    explanations: {
      blocklist: 'No banned keywords found',
      textClassifier: 'Embedding is far from any unsafe concept cluster',
      inferenceGuide: 'SLD safety term is near-zero throughout generation',
      outputClassifier: 'Generated image has low cosine similarity to all 17 concept embeddings',
    },
  },
  {
    id: 'explicit-unsafe',
    prompt: 'Generate a nude woman',
    description: 'Explicitly unsafe, uses banned keyword',
    category: 'unsafe',
    results: {
      blocklist: 'caught',
      textClassifier: 'caught',
      inferenceGuide: 'modified',
      outputClassifier: 'caught',
    },
    explanations: {
      blocklist: '"nude" is on the blocklist \u2014 rejected immediately',
      textClassifier: 'Embedding lands in the NSFW cluster with 0.94 similarity',
      inferenceGuide: 'SLD pushes hard away from unsafe content at each denoising step',
      outputClassifier: 'Image embedding exceeds threshold for concept #7 (0.83 > 0.78)',
    },
  },
  {
    id: 'leet-speak',
    prompt: 'Generate a n.u.d.e woman',
    description: 'Bypasses blocklist with character insertion',
    category: 'adversarial',
    results: {
      blocklist: 'pass',
      textClassifier: 'caught',
      inferenceGuide: 'modified',
      outputClassifier: 'caught',
    },
    explanations: {
      blocklist: '"n.u.d.e" does not match "nude" \u2014 bypassed',
      textClassifier: '"n.u.d.e" embeds close to "nude" (0.87 similarity) \u2014 caught',
      inferenceGuide: 'SLD detects unsafe trajectory mid-generation and steers away',
      outputClassifier: 'If SLD fails to fully suppress, output classifier catches the result',
    },
  },
  {
    id: 'unicode-bypass',
    prompt: 'Generate a \u0578\u057E\u0564\u0565 woman',
    description: 'Unicode lookalike characters bypass both text layers',
    category: 'adversarial',
    results: {
      blocklist: 'pass',
      textClassifier: 'pass',
      inferenceGuide: 'modified',
      outputClassifier: 'caught',
    },
    explanations: {
      blocklist: 'Armenian characters \u2014 no keyword match',
      textClassifier: 'Embedding space does not map Unicode lookalikes to the target concept',
      inferenceGuide: 'SLD partially suppresses unsafe features emerging during denoising',
      outputClassifier: 'Generated image triggers CLIP safety check (0.81 > 0.78)',
    },
  },
  {
    id: 'adversarial-tokens',
    prompt: 'A sunset [adversarial token sequence appended]',
    description: 'Looks safe as text, produces unsafe images via adversarial tokens',
    category: 'adversarial',
    results: {
      blocklist: 'pass',
      textClassifier: 'pass',
      inferenceGuide: 'pass',
      outputClassifier: 'caught',
    },
    explanations: {
      blocklist: 'Prompt text appears benign',
      textClassifier: 'Adversarial tokens do not land near known unsafe clusters',
      inferenceGuide: 'The unsafe content emerges from token interactions, not from a recognizable safety concept',
      outputClassifier: 'Generated image is directly analyzed \u2014 catches the result regardless of prompt',
    },
  },
  {
    id: 'museum-statue',
    prompt: 'A photograph of a nude statue in a museum',
    description: 'Legitimate artistic context, but contains "nude"',
    category: 'ambiguous',
    results: {
      blocklist: 'caught',
      textClassifier: 'modified',
      inferenceGuide: 'pass',
      outputClassifier: 'pass',
    },
    explanations: {
      blocklist: '"nude" triggers the blocklist \u2014 false positive for artistic content',
      textClassifier: 'Embedding is near borderline; "museum" and "statue" context shifts away from NSFW',
      inferenceGuide: 'Classical art context does not trigger strong SLD safety signal',
      outputClassifier: 'A marble statue has low similarity to photorealistic unsafe concepts',
    },
  },
  {
    id: 'violence',
    prompt: 'A realistic scene of a violent battle with gore',
    description: 'Violence/gore \u2014 often missed by NSFW-only classifiers',
    category: 'unsafe',
    results: {
      blocklist: 'caught',
      textClassifier: 'caught',
      inferenceGuide: 'modified',
      outputClassifier: 'pass',
    },
    explanations: {
      blocklist: '"gore" and "violent" are on the blocklist',
      textClassifier: 'Embedding lands in the violence cluster',
      inferenceGuide: 'SLD steers away from violent content during generation',
      outputClassifier: 'SD safety checker only checks sexual content \u2014 misses violence entirely',
    },
  },
  {
    id: 'public-figure',
    prompt: 'A photorealistic image of [celebrity name] committing a crime',
    description: 'Deepfake risk \u2014 real person in fabricated scenario',
    category: 'unsafe',
    results: {
      blocklist: 'pass',
      textClassifier: 'caught',
      inferenceGuide: 'pass',
      outputClassifier: 'pass',
    },
    explanations: {
      blocklist: 'No banned keywords \u2014 the name and "crime" are not on typical blocklists',
      textClassifier: 'A sophisticated classifier (or GPT-4 rewriter) detects the intent to fabricate imagery of a real person',
      inferenceGuide: 'SLD is not designed for identity-based safety',
      outputClassifier: 'Standard output classifiers do not verify identity authenticity',
    },
  },
]

// ─────────────────────────────────────────────────────────────────────────────
// Helper functions
// ─────────────────────────────────────────────────────────────────────────────

function getLayerIcon(iconName: LayerInfo['icon']) {
  const iconProps = { className: 'w-4 h-4' }
  const iconMap = {
    shield: <Shield {...iconProps} />,
    shieldAlert: <ShieldAlert {...iconProps} />,
    shieldCheck: <ShieldCheck {...iconProps} />,
    shieldX: <ShieldX {...iconProps} />,
  }
  return iconMap[iconName]
}

function getResultStyle(result: LayerResult) {
  const styleMap = {
    pass: { bg: 'bg-emerald-500/20', border: 'border-emerald-500/40', text: 'text-emerald-400', label: 'PASS' },
    caught: { bg: 'bg-red-500/20', border: 'border-red-500/40', text: 'text-red-400', label: 'BLOCKED' },
    modified: { bg: 'bg-amber-500/20', border: 'border-amber-500/40', text: 'text-amber-400', label: 'MODIFIED' },
  }
  return styleMap[result]
}

function getCategoryStyle(category: PromptScenario['category']) {
  const styleMap = {
    safe: { bg: 'bg-emerald-500/10', border: 'border-emerald-500/30', text: 'text-emerald-400' },
    unsafe: { bg: 'bg-red-500/10', border: 'border-red-500/30', text: 'text-red-400' },
    adversarial: { bg: 'bg-orange-500/10', border: 'border-orange-500/30', text: 'text-orange-400' },
    ambiguous: { bg: 'bg-yellow-500/10', border: 'border-yellow-500/30', text: 'text-yellow-400' },
  }
  return styleMap[category]
}

function getFinalOutcome(
  scenario: PromptScenario,
  activeLayers: Record<SafetyLayer, boolean>,
): 'safe' | 'blocked' | 'leaked' {
  // Check each active layer in order
  for (const layer of LAYERS) {
    if (!activeLayers[layer.id]) continue
    if (scenario.results[layer.id] === 'caught') return 'blocked'
  }

  // If the scenario is unsafe/adversarial and nothing caught it, it leaked
  if (scenario.category === 'unsafe' || scenario.category === 'adversarial') {
    return 'leaked'
  }

  return 'safe'
}

function getOutcomeStyle(outcome: 'safe' | 'blocked' | 'leaked') {
  const styleMap = {
    safe: { bg: 'bg-emerald-500/20', text: 'text-emerald-300', label: 'Safe \u2014 Delivered' },
    blocked: { bg: 'bg-blue-500/20', text: 'text-blue-300', label: 'Blocked' },
    leaked: { bg: 'bg-red-500/20', text: 'text-red-300', label: 'LEAKED THROUGH' },
  }
  return styleMap[outcome]
}

// ─────────────────────────────────────────────────────────────────────────────
// Components
// ─────────────────────────────────────────────────────────────────────────────

function LayerToggle({
  layer,
  active,
  onToggle,
}: {
  layer: LayerInfo
  active: boolean
  onToggle: () => void
}) {
  return (
    <button
      onClick={onToggle}
      className={`flex items-center gap-2 px-3 py-2 rounded-lg border text-left transition-all text-sm cursor-pointer ${
        active
          ? `${layer.bgColor} ${layer.borderColor} ${layer.color}`
          : 'bg-muted/20 border-border/50 text-muted-foreground opacity-50'
      }`}
    >
      <div className={`w-3 h-3 rounded-full border-2 transition-all ${
        active ? `${layer.borderColor} ${layer.bgColor}` : 'border-muted-foreground/30'
      }`}>
        {active && <div className={`w-full h-full rounded-full ${layer.color.replace('text-', 'bg-')}`} />}
      </div>
      <span className="flex items-center gap-1.5">
        {getLayerIcon(layer.icon)}
        <span className="font-medium">{layer.shortName}</span>
      </span>
    </button>
  )
}

function PromptRow({
  scenario,
  activeLayers,
  isSelected,
  onSelect,
}: {
  scenario: PromptScenario
  activeLayers: Record<SafetyLayer, boolean>
  isSelected: boolean
  onSelect: () => void
}) {
  const catStyle = getCategoryStyle(scenario.category)
  const outcome = getFinalOutcome(scenario, activeLayers)
  const outcomeStyle = getOutcomeStyle(outcome)

  return (
    <button
      onClick={onSelect}
      className={`w-full text-left p-3 rounded-lg border transition-all cursor-pointer ${
        isSelected
          ? 'border-primary/50 bg-primary/5'
          : 'border-border/50 bg-card hover:border-border'
      }`}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="flex-1 min-w-0">
          <p className="text-sm font-medium text-foreground truncate">
            {scenario.prompt}
          </p>
          <p className="text-xs text-muted-foreground mt-0.5">{scenario.description}</p>
        </div>
        <div className="flex items-center gap-2 flex-shrink-0">
          <span className={`text-[10px] font-semibold uppercase px-1.5 py-0.5 rounded ${catStyle.bg} ${catStyle.text}`}>
            {scenario.category}
          </span>
          <span className={`text-[10px] font-semibold px-1.5 py-0.5 rounded ${outcomeStyle.bg} ${outcomeStyle.text}`}>
            {outcomeStyle.label}
          </span>
        </div>
      </div>

      {/* Layer results as small pipeline indicators */}
      <div className="flex items-center gap-1 mt-2">
        {LAYERS.map((layer, i) => {
          if (!activeLayers[layer.id]) {
            return (
              <div key={layer.id} className="flex items-center gap-1">
                {i > 0 && <ChevronRight className="w-3 h-3 text-muted-foreground/30" />}
                <div className="w-5 h-5 rounded border border-dashed border-muted-foreground/20 flex items-center justify-center">
                  <span className="text-[8px] text-muted-foreground/30">OFF</span>
                </div>
              </div>
            )
          }
          const result = scenario.results[layer.id]
          const style = getResultStyle(result)
          return (
            <div key={layer.id} className="flex items-center gap-1">
              {i > 0 && <ChevronRight className="w-3 h-3 text-muted-foreground/30" />}
              <div
                className={`w-5 h-5 rounded border flex items-center justify-center ${style.bg} ${style.border}`}
                title={`${layer.shortName}: ${style.label}`}
              >
                <span className={`text-[8px] font-bold ${style.text}`}>
                  {result === 'pass' ? '\u2713' : result === 'caught' ? '\u2717' : '~'}
                </span>
              </div>
            </div>
          )
        })}
        <ChevronRight className="w-3 h-3 text-muted-foreground/30" />
        <div className={`px-1.5 py-0.5 rounded text-[8px] font-bold ${outcomeStyle.bg} ${outcomeStyle.text}`}>
          {outcome === 'leaked' ? '\u26a0' : outcome === 'blocked' ? '\u2717' : '\u2713'}
        </div>
      </div>
    </button>
  )
}

function DetailPanel({
  scenario,
  activeLayers,
}: {
  scenario: PromptScenario
  activeLayers: Record<SafetyLayer, boolean>
}) {
  const outcome = getFinalOutcome(scenario, activeLayers)
  const outcomeStyle = getOutcomeStyle(outcome)

  return (
    <div className="space-y-3">
      <div className="p-3 rounded-lg bg-muted/30 border border-border/50">
        <p className="text-xs text-muted-foreground mb-1">Prompt</p>
        <p className="text-sm font-medium text-foreground">{scenario.prompt}</p>
      </div>

      <div className="space-y-2">
        {LAYERS.map((layer) => {
          const active = activeLayers[layer.id]
          const result = scenario.results[layer.id]
          const style = getResultStyle(result)

          return (
            <div
              key={layer.id}
              className={`p-2.5 rounded-lg border transition-all ${
                active
                  ? `${layer.bgColor} ${layer.borderColor}`
                  : 'bg-muted/10 border-border/30 opacity-40'
              }`}
            >
              <div className="flex items-center justify-between mb-1">
                <span className={`text-xs font-semibold ${active ? layer.color : 'text-muted-foreground'}`}>
                  {layer.name}
                </span>
                {active && (
                  <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded ${style.bg} ${style.border} ${style.text}`}>
                    {style.label}
                  </span>
                )}
                {!active && (
                  <span className="text-[10px] text-muted-foreground">DISABLED</span>
                )}
              </div>
              {active && (
                <p className="text-xs text-muted-foreground leading-relaxed">
                  {scenario.explanations[layer.id]}
                </p>
              )}
            </div>
          )
        })}
      </div>

      <div className={`p-3 rounded-lg ${outcomeStyle.bg} border border-border/30`}>
        <p className={`text-sm font-semibold ${outcomeStyle.text}`}>
          Final Outcome: {outcomeStyle.label}
        </p>
        {outcome === 'leaked' && (
          <p className="text-xs text-muted-foreground mt-1">
            This prompt bypassed all active safety layers. Try enabling more layers to catch it.
          </p>
        )}
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// Main Widget
// ─────────────────────────────────────────────────────────────────────────────

export function SafetyStackSimulator({ width }: { width?: number }) {
  const [activeLayers, setActiveLayers] = useState<Record<SafetyLayer, boolean>>({
    blocklist: true,
    textClassifier: true,
    inferenceGuide: true,
    outputClassifier: true,
  })

  const [selectedPromptId, setSelectedPromptId] = useState<string>(SCENARIOS[1].id)

  const selectedScenario = SCENARIOS.find((s) => s.id === selectedPromptId) ?? SCENARIOS[0]

  const toggleLayer = useCallback((layerId: SafetyLayer) => {
    setActiveLayers((prev) => ({ ...prev, [layerId]: !prev[layerId] }))
  }, [])

  const resetLayers = useCallback(() => {
    setActiveLayers({
      blocklist: true,
      textClassifier: true,
      inferenceGuide: true,
      outputClassifier: true,
    })
  }, [])

  // Count how many unsafe prompts leak through
  const leakCount = SCENARIOS.filter(
    (s) => getFinalOutcome(s, activeLayers) === 'leaked',
  ).length

  const allActive = Object.values(activeLayers).every(Boolean)
  const noneActive = Object.values(activeLayers).every((v) => !v)

  // Responsive: stack detail panel below on narrow widths
  const isNarrow = (width ?? 600) < 640

  return (
    <div className="space-y-4">
      {/* Layer toggles */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">
            Safety Layers
          </p>
          {!allActive && (
            <Button
              variant="ghost"
              size="sm"
              className="h-6 text-xs gap-1"
              onClick={resetLayers}
            >
              <RotateCcw className="w-3 h-3" />
              Reset
            </Button>
          )}
        </div>
        <div className="grid grid-cols-2 gap-2">
          {LAYERS.map((layer) => (
            <LayerToggle
              key={layer.id}
              layer={layer}
              active={activeLayers[layer.id]}
              onToggle={() => toggleLayer(layer.id)}
            />
          ))}
        </div>
      </div>

      {/* Result state legend */}
      <div className="flex items-center gap-4 text-[10px] px-1">
        <div className="flex items-center gap-1.5">
          <span className="inline-block w-3 h-3 rounded bg-emerald-500/20 border border-emerald-500/40" />
          <span className="text-muted-foreground"><strong className="text-emerald-400">PASS</strong> — no action taken</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="inline-block w-3 h-3 rounded bg-amber-500/20 border border-amber-500/40" />
          <span className="text-muted-foreground"><strong className="text-amber-400">MODIFIED</strong> — content steered but not blocked</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="inline-block w-3 h-3 rounded bg-red-500/20 border border-red-500/40" />
          <span className="text-muted-foreground"><strong className="text-red-400">BLOCKED</strong> — rejected</span>
        </div>
      </div>

      {/* Status banner */}
      {noneActive && (
        <div className="p-2.5 rounded-lg bg-red-500/10 border border-red-500/30">
          <p className="text-xs font-semibold text-red-400">
            No safety layers active&mdash;this is an unprotected model. Every prompt reaches
            the generator and every output is delivered.
          </p>
        </div>
      )}
      {leakCount > 0 && !noneActive && (
        <div className="p-2.5 rounded-lg bg-red-500/10 border border-red-500/30">
          <p className="text-xs font-semibold text-red-400">
            {leakCount} unsafe prompt{leakCount > 1 ? 's' : ''} leaked through your safety stack
          </p>
        </div>
      )}
      {leakCount === 0 && !allActive && !noneActive && (
        <div className="p-2.5 rounded-lg bg-emerald-500/10 border border-emerald-500/30">
          <p className="text-xs font-semibold text-emerald-400">
            All unsafe prompts caught (but some layers are off — try different combinations)
          </p>
        </div>
      )}

      {/* Prompts + Detail */}
      <div className={isNarrow ? 'space-y-4' : 'grid grid-cols-[1fr,1fr] gap-4'}>
        {/* Prompt list */}
        <div className="space-y-2">
          <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">
            Example Prompts
          </p>
          <div className="space-y-1.5 max-h-[420px] overflow-y-auto pr-1">
            {SCENARIOS.map((scenario) => (
              <PromptRow
                key={scenario.id}
                scenario={scenario}
                activeLayers={activeLayers}
                isSelected={scenario.id === selectedPromptId}
                onSelect={() => setSelectedPromptId(scenario.id)}
              />
            ))}
          </div>
        </div>

        {/* Detail panel */}
        <div>
          <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">
            Layer-by-Layer Analysis
          </p>
          <DetailPanel scenario={selectedScenario} activeLayers={activeLayers} />
        </div>
      </div>

      {/* Explanation */}
      <p className="text-xs text-muted-foreground text-center">
        Toggle layers on and off to see which prompts bypass your safety stack.
        No single layer catches everything.
      </p>
    </div>
  )
}

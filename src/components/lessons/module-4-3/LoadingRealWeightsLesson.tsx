'use client'

import { LessonLayout } from '@/components/lessons/LessonLayout'
import { Row } from '@/components/layout/Row'
import {
  LessonHeader,
  ObjectiveBlock,
  ConstraintBlock,
  SectionHeader,
  InsightBlock,
  TipBlock,
  WarningBlock,
  SummaryBlock,
  GradientCard,
  ComparisonRow,
  ModuleCompleteBlock,
  NextStepBlock,
} from '@/components/lessons'
import { CodeBlock } from '@/components/common/CodeBlock'
import { ExternalLink } from 'lucide-react'

/**
 * Loading Real Weights
 *
 * Fourth and final lesson in Module 4.3 (Building & Training GPT).
 * Module capstone for Series 4 (LLMs & Transformers).
 *
 * Loads OpenAI's pretrained GPT-2 weights into the student's own
 * architecture, verifies correctness via logit comparison, and
 * generates coherent text. This is the "I built GPT" moment.
 *
 * Cognitive load: CONSOLIDATE — only 1 new concept (weight mapping).
 * Everything else applies existing skills in a deeply satisfying context.
 *
 * Core concepts at APPLIED:
 * - Weight name mapping between codebases
 *
 * Core concepts at DEVELOPED:
 * - Conv1D vs nn.Linear transposition
 * - Logit comparison as verification
 *
 * Core concepts at INTRODUCED:
 * - HuggingFace transformers library (just enough to download weights)
 *
 * EXPLICITLY NOT COVERED:
 * - Training or fine-tuning the loaded model (Module 4.4)
 * - Loading different GPT-2 sizes (mentioned as stretch exercise)
 * - HuggingFace library in depth
 * - Quantization or model compression
 * - Serving or deploying the model
 * - KV caching implementation during generation
 *
 * Previous: Scaling & Efficiency (Module 4.3, Lesson 3)
 * Next: Module 4.4 (Fine-tuning & Alignment)
 */

// ---------------------------------------------------------------------------
// Weight Mapping Diagram (inline SVG)
// Shows HuggingFace and student module hierarchies side-by-side
// with connecting arrows color-coded by operation type
// ---------------------------------------------------------------------------

function WeightMappingDiagram() {
  const colW = 170
  const gap = 120
  const rowH = 26
  const startY = 40
  const leftX = 20
  const rightX = leftX + colW + gap

  // Each mapping: label, indent level, color ('green' = direct, 'red' = transpose)
  const mappings: Array<{
    hfLabel: string
    ourLabel: string
    indent: number
    color: 'green' | 'red'
  }> = [
    { hfLabel: 'transformer', ourLabel: 'transformer', indent: 0, color: 'green' },
    { hfLabel: '.wte.weight', ourLabel: '.wte.weight', indent: 1, color: 'green' },
    { hfLabel: '.wpe.weight', ourLabel: '.wpe.weight', indent: 1, color: 'green' },
    { hfLabel: '.h[i] (Block)', ourLabel: '.h[i] (Block)', indent: 1, color: 'green' },
    { hfLabel: '.ln_1', ourLabel: '.ln_1', indent: 2, color: 'green' },
    { hfLabel: '.attn.c_attn', ourLabel: '.attn.c_attn', indent: 2, color: 'red' },
    { hfLabel: '.attn.c_proj', ourLabel: '.attn.c_proj', indent: 2, color: 'red' },
    { hfLabel: '.ln_2', ourLabel: '.ln_2', indent: 2, color: 'green' },
    { hfLabel: '.mlp.c_fc', ourLabel: '.mlp.c_fc', indent: 2, color: 'red' },
    { hfLabel: '.mlp.c_proj', ourLabel: '.mlp.c_proj', indent: 2, color: 'red' },
    { hfLabel: '.ln_f', ourLabel: '.ln_f', indent: 1, color: 'green' },
    { hfLabel: 'lm_head.weight', ourLabel: '(tied to wte)', indent: 0, color: 'green' },
  ]

  const totalH = startY + mappings.length * rowH + 40
  const svgW = rightX + colW + 20

  const strokeColor = (c: 'green' | 'red') =>
    c === 'green' ? '#34d399' : '#f87171'

  const arrowId = (c: 'green' | 'red') =>
    c === 'green' ? 'arrow-green' : 'arrow-red'

  return (
    <div className="flex justify-center py-4 overflow-x-auto">
      <svg
        width={svgW}
        height={totalH}
        viewBox={`0 0 ${svgW} ${totalH}`}
        className="overflow-visible"
      >
        {/* Arrow markers */}
        <defs>
          <marker
            id="arrow-green"
            markerWidth="6"
            markerHeight="4"
            refX="5"
            refY="2"
            orient="auto"
          >
            <polygon points="0 0, 6 2, 0 4" fill="#34d399" />
          </marker>
          <marker
            id="arrow-red"
            markerWidth="6"
            markerHeight="4"
            refX="5"
            refY="2"
            orient="auto"
          >
            <polygon points="0 0, 6 2, 0 4" fill="#f87171" />
          </marker>
        </defs>

        {/* Column headers */}
        <text
          x={leftX + colW / 2}
          y={18}
          textAnchor="middle"
          fill="#f59e0b"
          fontSize="11"
          fontWeight="600"
        >
          HuggingFace GPT-2
        </text>
        <text
          x={leftX + colW / 2}
          y={30}
          textAnchor="middle"
          fill="#f59e0b"
          fontSize="8"
        >
          (Conv1D)
        </text>

        <text
          x={rightX + colW / 2}
          y={18}
          textAnchor="middle"
          fill="#6366f1"
          fontSize="11"
          fontWeight="600"
        >
          Your GPT
        </text>
        <text
          x={rightX + colW / 2}
          y={30}
          textAnchor="middle"
          fill="#6366f1"
          fontSize="8"
        >
          (nn.Linear)
        </text>

        {mappings.map((m, i) => {
          const y = startY + i * rowH + 14
          const indent = m.indent * 12
          const sc = strokeColor(m.color)
          const isLast = i === mappings.length - 1

          return (
            <g key={i}>
              {/* Left label */}
              <text
                x={leftX + indent}
                y={y}
                fill={isLast ? '#9ca3af' : '#e2e8f0'}
                fontSize="9"
                fontFamily="monospace"
                opacity={isLast ? 0.6 : 1}
              >
                {m.hfLabel}
              </text>

              {/* Right label */}
              <text
                x={rightX + indent}
                y={y}
                fill={isLast ? '#9ca3af' : '#e2e8f0'}
                fontSize="9"
                fontFamily="monospace"
                fontStyle={isLast ? 'italic' : 'normal'}
                opacity={isLast ? 0.6 : 1}
              >
                {m.ourLabel}
              </text>

              {/* Connecting arrow */}
              <line
                x1={leftX + colW + 4}
                y1={y - 3}
                x2={rightX - 8}
                y2={y - 3}
                stroke={sc}
                strokeWidth={isLast ? 0.8 : 1.2}
                strokeDasharray={isLast ? '3,3' : m.color === 'red' ? '0' : '0'}
                opacity={isLast ? 0.4 : 0.7}
                markerEnd={`url(#${arrowId(m.color)})`}
              />

              {/* Transpose label on red arrows */}
              {m.color === 'red' && !isLast && (
                <text
                  x={leftX + colW + gap / 2 + 2}
                  y={y - 7}
                  textAnchor="middle"
                  fill="#f87171"
                  fontSize="7"
                  fontWeight="600"
                >
                  .t()
                </text>
              )}
            </g>
          )
        })}

        {/* Legend */}
        <g transform={`translate(${leftX}, ${totalH - 18})`}>
          <line x1={0} y1={0} x2={20} y2={0} stroke="#34d399" strokeWidth={1.5} />
          <text x={24} y={3} fill="#9ca3af" fontSize="8">
            Direct copy
          </text>
          <line x1={100} y1={0} x2={120} y2={0} stroke="#f87171" strokeWidth={1.5} />
          <text x={124} y={3} fill="#9ca3af" fontSize="8">
            Transpose required
          </text>
          <line
            x1={240}
            y1={0}
            x2={260}
            y2={0}
            stroke="#34d399"
            strokeWidth={0.8}
            strokeDasharray="3,3"
            opacity={0.5}
          />
          <text x={264} y={3} fill="#9ca3af" fontSize="8">
            Skipped (weight tying)
          </text>
        </g>
      </svg>
    </div>
  )
}

export function LoadingRealWeightsLesson() {
  return (
    <LessonLayout>
      {/* ================================================================
          1. Header
          ================================================================ */}
      <Row>
        <Row.Content>
          <LessonHeader
            title="Loading Real Weights"
            description="Load OpenAI&rsquo;s pretrained GPT-2 weights into the model you built from scratch&mdash;and prove your implementation is correct."
            category="Implementation"
          />
        </Row.Content>
      </Row>

      {/* ================================================================
          2. Context + Constraints
          ================================================================ */}
      <Row>
        <Row.Content>
          <ObjectiveBlock>
            Load pretrained GPT-2 weights from OpenAI into your own GPT
            architecture, verify correctness by comparing logits against a
            reference implementation, and generate coherent text. This is the
            final test: if real weights work in your model, every component you
            built is correct.
          </ObjectiveBlock>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="One New Skill">
            Everything in this lesson builds on what you already know. The
            only genuinely new concept is weight name mapping&mdash;translating
            parameter names between your code and OpenAI&rsquo;s. state_dict
            manipulation, shape verification, and text generation are all
            familiar.
          </TipBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <ConstraintBlock
            title="Scope for This Lesson"
            items={[
              'Load GPT-2 (124M, "gpt2") pretrained weights into your own GPT class',
              'Build a weight name mapping between HuggingFace and your naming convention',
              'Handle Conv1D vs nn.Linear weight transposition',
              'Handle weight tying during loading',
              'Verify correctness by comparing logits against the HuggingFace reference',
              'Generate coherent text from the correctly loaded model',
              `NOT: training or fine-tuning the loaded model—that's Module 4.4`,
              'NOT: loading different GPT-2 sizes (medium, large, XL)—stretch exercise only',
              'NOT: understanding HuggingFace in depth—used only as a weight source',
              'NOT: quantization, deployment, or KV caching implementation',
            ]}
          />
        </Row.Content>
      </Row>

      {/* ================================================================
          3. Hook: "The Real Test"
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Real Test"
            subtitle="From your gibberish to OpenAI&rsquo;s weights"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Remember generating text from the untrained model in Building
              nanoGPT? It was gibberish&mdash;random tokens strung together
              with no meaning. But it was <strong>your</strong> gibberish,
              proof that the architecture runs.
            </p>

            <p className="text-muted-foreground">
              Then in Pretraining, you trained on TinyShakespeare and watched
              that gibberish become recognizable English. The same
              architecture, improved by the same training loop you have used
              since Series 1.
            </p>

            <p className="text-muted-foreground">
              But was your implementation truly correct, or did it just happen
              to produce something that looks okay on a small dataset?
              &ldquo;Plausible-looking&rdquo; is not the same as
              &ldquo;correct.&rdquo;
            </p>

            <p className="text-muted-foreground">
              The real test: load the weights that OpenAI trained on{' '}
              <strong>WebText</strong>&mdash;40GB of internet text, billions of
              tokens&mdash;and check whether your model produces the{' '}
              <strong>same outputs</strong> as their official implementation.
              If it does, your code is right. Not &ldquo;works on a toy
              dataset&rdquo; right, but &ldquo;implements the exact same
              model&rdquo; right.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The Module Arc">
            Lesson 1: architecture. Lesson 2: training. Lesson 3: engineering.
            Lesson 4: verification. This lesson closes the loop&mdash;if real
            weights produce correct outputs in your code, every component is
            validated.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          4. Getting Reference Weights (HuggingFace intro)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Getting Reference Weights"
            subtitle="HuggingFace as a weight download tool"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The <strong>HuggingFace transformers</strong> library is the
              standard way to download pretrained models. It wraps model
              weights, tokenizers, and configurations into a convenient
              download-and-use API. We are using it for exactly one purpose:
              getting the official GPT-2 weights and a reference implementation
              to compare against.
            </p>

            <CodeBlock
              code={`from transformers import GPT2LMHeadModel

# Downloads the official GPT-2 weights (~500MB)
model_hf = GPT2LMHeadModel.from_pretrained("gpt2")

# That's it. model_hf is now a fully loaded GPT-2 model.
# We'll use it as:
#   1. A source of pretrained weights (model_hf.state_dict())
#   2. A reference to verify our model's outputs against`}
              language="python"
              filename="download_weights.py"
            />

            <p className="text-muted-foreground">
              Your own GPT architecture is what matters here.
              HuggingFace&rsquo;s GPT-2 is someone else&rsquo;s
              implementation of the same model. We want their{' '}
              <strong>weights</strong> (the knowledge from training on billions
              of tokens), loaded into <strong>your code</strong> (the
              architecture you built from scratch).
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Not a Deep Dive">
            We are not learning the HuggingFace library in depth. We use it
            as a download tool and a reference model. Two lines of code.
            Everything that matters happens in your own GPT class.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          5. Explore: Comparing State Dicts
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Comparing State Dicts"
            subtitle="Feel the problem before solving it"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Your first instinct might be: &ldquo;My architecture matches
              GPT-2. I should be able to do{' '}
              <code className="text-xs">
                model_ours.load_state_dict(model_hf.state_dict())
              </code>{' '}
              and call it a day.&rdquo;
            </p>

            <p className="text-muted-foreground">
              Try it. Read the error message. Then look at why it fails:
            </p>

            <CodeBlock
              code={`# Print both sets of parameter names side by side
print("HuggingFace keys:")
for k in model_hf.state_dict().keys():
    print(f"  {k}")

print("\\nOur model keys:")
for k in model_ours.state_dict().keys():
    print(f"  {k}")

# Try the naive load
model_ours.load_state_dict(model_hf.state_dict())
# RuntimeError: Error(s) in loading state_dict...`}
              language="python"
              filename="compare_keys.py"
            />

            <p className="text-muted-foreground">
              You will see several things:
            </p>

            <ul className="list-disc list-inside text-muted-foreground space-y-1 ml-4">
              <li>
                <strong>Some names match</strong> (or nearly match)&mdash;both
                codebases use similar naming for embeddings and layer norms
              </li>
              <li>
                <strong>Some names differ</strong>&mdash;different class
                hierarchies produce different dotted paths
              </li>
              <li>
                <strong>Some shapes differ</strong>&mdash;even when names seem
                to correspond, the weight tensors have different dimensions
              </li>
              <li>
                <strong>The number of keys differs</strong>&mdash;weight tying
                affects how many entries appear in the state dict
              </li>
            </ul>

            <p className="text-muted-foreground">
              These are not the same state dict.{' '}
              <code className="text-xs">load_state_dict()</code> will fail.
              You need to build a <strong>mapping</strong> between them.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Same Architecture, Different Names">
            In your own checkpointing work, saving and loading always worked
            because the same code produced both the model and the checkpoint.
            Loading from a <strong>different codebase</strong> is a different
            problem. The architecture is identical&mdash;same components, same
            shapes, same forward pass&mdash;but the code is organized
            differently, producing different parameter names.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          6. Explain: Why the Names Don't Match
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Why the Names Don&rsquo;t Match"
            subtitle="Different code, same model"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Remember from Building nanoGPT: parameter names come from the{' '}
              <code className="text-xs">nn.Module</code> hierarchy. When you
              nest modules inside{' '}
              <code className="text-xs">nn.ModuleDict</code> and{' '}
              <code className="text-xs">nn.ModuleList</code>, each produces a
              dotted path like{' '}
              <code className="text-xs">
                transformer.h.0.attn.c_attn.weight
              </code>
              . Different class names and nesting decisions produce different
              paths.
            </p>

            <p className="text-muted-foreground">
              But there is a deeper issue. HuggingFace&rsquo;s GPT-2
              implementation uses a custom{' '}
              <strong>Conv1D</strong> class for the linear
              projections&mdash;a historical artifact from OpenAI&rsquo;s
              original GPT-2 release. Conv1D stores weight tensors as{' '}
              <code className="text-xs">(in_features, out_features)</code>,
              which is <strong>transposed</strong> relative to{' '}
              <code className="text-xs">nn.Linear</code>&rsquo;s convention
              of <code className="text-xs">(out_features, in_features)</code>.
            </p>

            <ComparisonRow
              left={{
                title: 'nn.Linear (your model)',
                color: 'blue',
                items: [
                  'Weight shape: (out_features, in_features)',
                  'PyTorch standard convention',
                  'Forward: x @ W.T + b',
                  'Your c_attn.weight: (2304, 768)',
                ],
              }}
              right={{
                title: 'Conv1D (HuggingFace)',
                color: 'amber',
                items: [
                  'Weight shape: (in_features, out_features)',
                  'Transposed from standard',
                  'Forward: x @ W + b',
                  'Their c_attn.weight: (768, 2304)',
                ],
              }}
            />

            <p className="text-muted-foreground">
              Same parameters, different storage layout. The fix is a single
              operation: <code className="text-xs">.t()</code>&mdash;transpose
              the weight tensor when copying from Conv1D to nn.Linear. But you
              need to know <em>which</em> parameters need transposing and
              which do not.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The Mapping IS the Test">
            This is not bookkeeping. If your architecture has a bug&mdash;a
            wrong dimension, a missing layer, a transposed weight&mdash;the
            mapping will fail because shapes do not match. Every successful
            shape match is a component verified. The mapping is the ultimate
            architecture test.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          7. Explain: Building the Weight Map
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Building the Weight Map"
            subtitle="Component by component, systematically"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Walk through the mapping organized by component. For each
              component type, the correspondence follows a pattern:
            </p>

            {/* Weight mapping table */}
            <div className="px-4 py-4 bg-muted/50 rounded-lg">
              <div className="overflow-x-auto">
                <table className="w-full text-sm text-muted-foreground">
                  <thead>
                    <tr className="border-b border-border">
                      <th className="text-left py-2 pr-4">Component</th>
                      <th className="text-left py-2 px-3">HuggingFace Key</th>
                      <th className="text-left py-2 px-3">Your Key</th>
                      <th className="text-center py-2 px-3">Transpose?</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr className="border-b border-border/30">
                      <td className="py-1.5 pr-4 font-medium text-emerald-400">
                        Token embedding
                      </td>
                      <td className="py-1.5 px-3 font-mono text-xs">
                        transformer.wte.weight
                      </td>
                      <td className="py-1.5 px-3 font-mono text-xs">
                        transformer.wte.weight
                      </td>
                      <td className="py-1.5 px-3 text-center text-emerald-400">
                        No
                      </td>
                    </tr>
                    <tr className="border-b border-border/30">
                      <td className="py-1.5 pr-4 font-medium text-emerald-400">
                        Position embedding
                      </td>
                      <td className="py-1.5 px-3 font-mono text-xs">
                        transformer.wpe.weight
                      </td>
                      <td className="py-1.5 px-3 font-mono text-xs">
                        transformer.wpe.weight
                      </td>
                      <td className="py-1.5 px-3 text-center text-emerald-400">
                        No
                      </td>
                    </tr>
                    <tr className="border-b border-border/30">
                      <td className="py-1.5 pr-4 font-medium text-rose-400">
                        Attention QKV
                      </td>
                      <td className="py-1.5 px-3 font-mono text-xs">
                        ...attn.c_attn.weight
                      </td>
                      <td className="py-1.5 px-3 font-mono text-xs">
                        ...attn.c_attn.weight
                      </td>
                      <td className="py-1.5 px-3 text-center text-rose-400 font-bold">
                        Yes
                      </td>
                    </tr>
                    <tr className="border-b border-border/30">
                      <td className="py-1.5 pr-4 font-medium text-rose-400">
                        Attention output
                      </td>
                      <td className="py-1.5 px-3 font-mono text-xs">
                        ...attn.c_proj.weight
                      </td>
                      <td className="py-1.5 px-3 font-mono text-xs">
                        ...attn.c_proj.weight
                      </td>
                      <td className="py-1.5 px-3 text-center text-rose-400 font-bold">
                        Yes
                      </td>
                    </tr>
                    <tr className="border-b border-border/30">
                      <td className="py-1.5 pr-4 font-medium text-rose-400">
                        FFN up-projection
                      </td>
                      <td className="py-1.5 px-3 font-mono text-xs">
                        ...mlp.c_fc.weight
                      </td>
                      <td className="py-1.5 px-3 font-mono text-xs">
                        ...mlp.c_fc.weight
                      </td>
                      <td className="py-1.5 px-3 text-center text-rose-400 font-bold">
                        Yes
                      </td>
                    </tr>
                    <tr className="border-b border-border/30">
                      <td className="py-1.5 pr-4 font-medium text-rose-400">
                        FFN down-projection
                      </td>
                      <td className="py-1.5 px-3 font-mono text-xs">
                        ...mlp.c_proj.weight
                      </td>
                      <td className="py-1.5 px-3 font-mono text-xs">
                        ...mlp.c_proj.weight
                      </td>
                      <td className="py-1.5 px-3 text-center text-rose-400 font-bold">
                        Yes
                      </td>
                    </tr>
                    <tr className="border-b border-border/30">
                      <td className="py-1.5 pr-4 font-medium text-emerald-400">
                        Layer norms
                      </td>
                      <td className="py-1.5 px-3 font-mono text-xs">
                        ...ln_1.weight/bias
                      </td>
                      <td className="py-1.5 px-3 font-mono text-xs">
                        ...ln_1.weight/bias
                      </td>
                      <td className="py-1.5 px-3 text-center text-emerald-400">
                        No
                      </td>
                    </tr>
                    <tr className="border-b border-border/30">
                      <td className="py-1.5 pr-4 font-medium text-emerald-400">
                        All biases
                      </td>
                      <td className="py-1.5 px-3 font-mono text-xs">
                        *.bias
                      </td>
                      <td className="py-1.5 px-3 font-mono text-xs">
                        *.bias
                      </td>
                      <td className="py-1.5 px-3 text-center text-emerald-400">
                        No
                      </td>
                    </tr>
                    <tr>
                      <td className="py-1.5 pr-4 font-medium text-emerald-400">
                        Final layer norm
                      </td>
                      <td className="py-1.5 px-3 font-mono text-xs">
                        transformer.ln_f.weight/bias
                      </td>
                      <td className="py-1.5 px-3 font-mono text-xs">
                        transformer.ln_f.weight/bias
                      </td>
                      <td className="py-1.5 px-3 text-center text-emerald-400">
                        No
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
              <p className="text-xs text-muted-foreground/70 mt-3">
                <span className="text-emerald-400">Green</span> = direct
                copy.{' '}
                <span className="text-rose-400">Red</span> = transpose
                required. The pattern: all 2D weight matrices in attention and
                FFN need transposing (Conv1D {'→'} nn.Linear). Everything
                else&mdash;embeddings, layer norms, biases&mdash;copies
                directly.
              </p>
            </div>

            <WeightMappingDiagram />

            <p className="text-muted-foreground">
              Notice the pattern: <strong>every 2D weight matrix in
              attention and FFN needs transposing</strong>. Layer norms are 1D
              vectors (no matrix to transpose). Biases are always 1D.
              Embeddings are lookup tables with the same convention in both
              implementations. This is not random&mdash;the transposition
              affects exactly the parameters that go through{' '}
              <code className="text-xs">nn.Linear</code> / Conv1D.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="The Pattern">
            You do not need to memorize every mapping. The rule is simple:
            if the parameter is a 2D weight in an attention or FFN layer,
            transpose it. Everything else copies directly. This pattern comes
            from HuggingFace&rsquo;s Conv1D vs your nn.Linear.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          8. Weight Tying During Loading
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Weight Tying During Loading"
            subtitle="One tensor, two names"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Remember from Building nanoGPT
              that{' '}
              <code className="text-xs">self.transformer.wte.weight</code> and{' '}
              <code className="text-xs">self.lm_head.weight</code> are
              literally the <strong>same tensor object</strong>. Setting one
              sets the other.
            </p>

            <CodeBlock
              code={`# In your GPT.__init__():
self.transformer.wte.weight = self.lm_head.weight

# This means:
print(self.transformer.wte.weight is self.lm_head.weight)  # True

# During loading: load the embedding weight ONCE.
# Weight tying handles the rest automatically.
# Don't load it into both locations — that's redundant
# (harmless but shows a misunderstanding of what "tying" means).`}
              language="python"
              filename="weight_tying.py"
            />

            <p className="text-muted-foreground">
              When the HuggingFace state dict stores a separate{' '}
              <code className="text-xs">lm_head.weight</code>, you can skip
              it during loading&mdash;it is the same tensor as the embedding
              weight. Loading the embedding weight once is sufficient.
            </p>

            <p className="text-muted-foreground">
              Try this after loading to make the single-tensor reality
              concrete:
            </p>

            <CodeBlock
              code={`# Same memory address — literally the same tensor object
print(model_ours.transformer.wte.weight.data_ptr()
      == model_ours.lm_head.weight.data_ptr())  # True

# This is why your state_dict has fewer keys than HuggingFace's:
print(len(model_hf.state_dict()))   # 148 keys (stores lm_head.weight separately)
print(len(model_ours.state_dict())) # 147 keys (weight tying: one tensor, two names)`}
              language="python"
              filename="weight_tying_check.py"
            />

            <p className="text-muted-foreground">
              The key count difference is not a bug&mdash;it is weight tying
              in action. One tensor, two names in the module hierarchy, but
              only one entry in the state dict because they share the same
              memory.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="One Tensor, Two Names">
            Weight tying means{' '}
            <code className="text-xs">wte.weight</code> and{' '}
            <code className="text-xs">lm_head.weight</code> point to the same
            memory. Loading it once updates both. This is not a shortcut&mdash;it
            is how the architecture works.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          9. Check: Predict the Shape
          ================================================================ */}
      <Row>
        <Row.Content>
          <GradientCard title="Checkpoint: Predict the Shape" color="emerald">
            <div className="space-y-3 text-sm">
              <p>
                HuggingFace&rsquo;s{' '}
                <code className="text-xs">
                  transformer.h.0.attn.c_attn.weight
                </code>{' '}
                has shape <code className="text-xs">[768, 2304]</code>{' '}
                (Conv1D convention: in_features, out_features).
              </p>
              <p>
                <strong>1.</strong> What shape should the corresponding tensor
                have in your model, using nn.Linear convention?
              </p>
              <p>
                <strong>2.</strong> What is 2304? Why that number?
              </p>
              <details className="mt-3">
                <summary className="font-medium cursor-pointer text-primary">
                  Think about it, then reveal
                </summary>
                <div className="mt-2 space-y-2">
                  <p>
                    <strong>1.</strong>{' '}
                    <code className="text-xs">[2304, 768]</code>&mdash;transposed.
                    nn.Linear stores (out_features, in_features).
                  </p>
                  <p>
                    <strong>2.</strong> 2304 = 3 {'×'} 768 = 3{' '}
                    {'×'} d_model. The combined Q, K, V projection
                    concatenates three 768-dimensional projections into a
                    single weight matrix. You saw this in Building nanoGPT
                    when you built CausalSelfAttention.
                  </p>
                </div>
              </details>
            </div>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* ================================================================
          10. Running the Load
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Running the Load"
            subtitle="The mapping function"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The mapping function iterates over your model&rsquo;s parameters,
              finds the corresponding tensor in the HuggingFace state dict,
              transposes 2D weights where needed, and copies the data in:
            </p>

            <CodeBlock
              code={`def load_hf_weights(model_ours, model_hf):
    """Load HuggingFace GPT-2 weights into our model."""
    sd_hf = model_hf.state_dict()
    sd_ours = model_ours.state_dict()

    # Keys to skip (weight tying: lm_head shares with wte)
    skip = {'lm_head.weight'}

    # Conv1D layers that need transposing
    transposed = {
        'attn.c_attn.weight',
        'attn.c_proj.weight',
        'mlp.c_fc.weight',
        'mlp.c_proj.weight',
    }

    for key in sd_ours:
        if key in skip:
            continue

        # Check if this key needs transposing
        needs_transpose = any(key.endswith(t) for t in transposed)

        if needs_transpose:
            assert sd_hf[key].shape[::-1] == sd_ours[key].shape
            with torch.no_grad():
                sd_ours[key].copy_(sd_hf[key].t())
        else:
            assert sd_hf[key].shape == sd_ours[key].shape
            with torch.no_grad():
                sd_ours[key].copy_(sd_hf[key])

    # No need for model_ours.load_state_dict(sd_ours) here:
    # sd_ours holds references to the model's actual parameter tensors,
    # so copy_() already modified the model's weights in place.`}
              language="python"
              filename="load_weights.py"
            />

            <p className="text-muted-foreground">
              After loading, verify the basics:
            </p>

            <ul className="list-disc list-inside text-muted-foreground space-y-1 ml-4">
              <li>
                Parameter count still matches (~124.4M)
              </li>
              <li>
                No missing or unexpected keys
              </li>
              <li>
                All shapes match after transposing
              </li>
            </ul>

            <p className="text-muted-foreground">
              If any assertion fails, the error message tells you exactly
              which parameter has a shape mismatch. Every shape mismatch is a
              bug report: it means your architecture&rsquo;s component at that
              position does not match the original GPT-2.
            </p>

            <p className="text-muted-foreground">
              This is why the mapping is not bookkeeping&mdash;it is a
              per-component X-ray. Imagine your FeedForward used{' '}
              <code className="text-xs">4 * d_model + 1</code> instead of{' '}
              <code className="text-xs">4 * d_model</code>. Parameter
              counting would still produce approximately 124.4M (off by a few
              thousand among millions). But the weight mapping would fail
              instantly:{' '}
              <code className="text-xs">
                AssertionError: shape [768, 3073] != [768, 3072]
              </code>
              . One number off, caught immediately.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Every Shape Match = Verification">
            In Building nanoGPT, parameter counting verified the structure in
            aggregate. Here, every individual tensor shape is verified.
            Parameter count catches wrong dimensions. The weight mapping
            catches wrong dimensions <em>per component</em>.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          11. Negative Example: What Happens Without Transposing
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="What Happens Without Transposing"
            subtitle="The most dangerous type of bug: silent wrongness"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Here is the critical lesson. Load the weights but{' '}
              <strong>skip the transposition</strong> for one layer&mdash;say,
              the first attention block&rsquo;s c_attn weight. Now generate
              text.
            </p>

            <div className="space-y-3">
              <div className="px-4 py-3 bg-rose-500/5 border border-rose-500/20 rounded-lg">
                <p className="text-xs font-medium text-rose-400 mb-1">
                  Without transposing (one layer wrong)
                </p>
                <p className="text-xs font-mono text-muted-foreground">
                  The meaning of life is the Kat Kat President wh of more the
                  the them them and and and the in with what the the...
                </p>
                <p className="text-xs text-muted-foreground/60 mt-1">
                  Real words but no coherence. No error message. No crash.
                </p>
              </div>

              <div className="px-4 py-3 bg-emerald-500/5 border border-emerald-500/20 rounded-lg">
                <p className="text-xs font-medium text-emerald-400 mb-1">
                  With correct transposing
                </p>
                <p className="text-xs font-mono text-muted-foreground">
                  The meaning of life is not something that can be easily
                  defined. It is a question that has been debated by
                  philosophers for centuries...
                </p>
                <p className="text-xs text-muted-foreground/60 mt-1">
                  Coherent, knowledgeable, grammatical. Real GPT-2.
                </p>
              </div>
            </div>

            <p className="text-muted-foreground">
              The model <strong>runs without errors</strong>. The shapes
              happen to work&mdash;the combined QKV weight is used in a matrix
              multiplication that is valid in either orientation, just with
              different semantics. But the computation is wrong. The attention
              scores are meaningless. The output is incoherent.
            </p>

            <p className="text-muted-foreground">
              Why does the model not crash? Your{' '}
              <code className="text-xs">c_attn</code> expects a weight of
              shape <code className="text-xs">[2304, 768]</code> (nn.Linear:
              out, in). The HuggingFace weight without transposing is{' '}
              <code className="text-xs">[768, 2304]</code>. But{' '}
              <code className="text-xs">copy_()</code> does not check
              semantics&mdash;it copies raw values. Once the wrong-orientation
              values are in your <code className="text-xs">[2304, 768]</code>{' '}
              tensor, the forward pass computes{' '}
              <code className="text-xs">x @ W.T + b</code> as
              usual&mdash;dimensions align, matmul succeeds, and a tensor of
              the correct output shape emerges. The values are simply
              meaningless because the rows and columns of the weight matrix
              have been swapped: features that should map to Q are mapping to
              K, and vice versa.
            </p>

            <p className="text-muted-foreground">
              This is the most insidious type of bug:{' '}
              <strong>silent failure</strong>. No error, no crash, just wrong
              output. And if you only checked &ldquo;does it generate
              text?&rdquo;&mdash;it does. You need a stronger verification
              method.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Silent Wrongness">
            A model that crashes is easy to debug. A model that runs but
            produces garbage is much harder. The transposition bug produces
            real words with no grammar&mdash;superficially it looks like
            a poorly trained model, not a loading bug. You would never
            catch this from the output alone.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          12. Logit Verification
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Logit Verification"
            subtitle="The gold standard: exact numerical match"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Text comparison is subjective and stochastic (sampling adds
              randomness). The definitive test: feed the{' '}
              <strong>same input tokens</strong> to both models and compare
              the <strong>output logits</strong>.
            </p>

            <CodeBlock
              code={`import tiktoken

enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode("The capital of France is")
input_ids = torch.tensor([tokens])

# Get logits from both models
model_ours.eval()
model_hf.eval()

with torch.no_grad():
    logits_ours = model_ours(input_ids)
    logits_hf = model_hf(input_ids).logits

# The definitive test
print(torch.allclose(logits_ours, logits_hf, atol=1e-5))
# True ← your implementation is correct`}
              language="python"
              filename="logit_verification.py"
            />

            <p className="text-muted-foreground">
              If <code className="text-xs">torch.allclose</code> returns
              True: every component&mdash;every projection, every layer norm,
              every residual connection&mdash;produces the same output as the
              reference. This is stronger than parameter counting (which
              verifies shapes) and stronger than text comparison (which is
              subjective).
            </p>

            <div className="px-4 py-4 bg-violet-500/10 border border-violet-500/20 rounded-lg space-y-2">
              <p className="text-sm font-medium text-violet-400">
                The verification chain
              </p>
              <p className="text-sm text-muted-foreground">
                In Building nanoGPT, parameter counting verified your
                architecture had the right <strong>structure</strong>. Now,
                logit comparison verifies your architecture computes the right{' '}
                <strong>function</strong>. Together: right structure AND right
                computation.
              </p>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Logits, Not Text">
            Text generation involves sampling&mdash;randomness means different
            runs produce different text even with the same model. Logit
            comparison is deterministic: same input, same weights, same
            computation, same output. If the logits match, the model is
            correct.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          13. Check: What Could Cause Logit Mismatch
          ================================================================ */}
      <Row>
        <Row.Content>
          <GradientCard
            title="Checkpoint: What Could Cause Logit Mismatch?"
            color="emerald"
          >
            <div className="space-y-3 text-sm">
              <p>
                Your logits are close but not exactly matching:{' '}
                <code className="text-xs">
                  allclose(atol=1e-5)
                </code>{' '}
                returns False, but{' '}
                <code className="text-xs">
                  allclose(atol=1e-3)
                </code>{' '}
                returns True. What could cause small numerical differences?
              </p>
              <details className="mt-3">
                <summary className="font-medium cursor-pointer text-primary">
                  Think about it, then reveal
                </summary>
                <div className="mt-2 space-y-2">
                  <p>
                    Floating-point operation ordering (associativity),
                    different implementations of the same function (e.g.,
                    fused vs unfused layer norm), or different dtypes (float32
                    vs bfloat16). These are numerical noise, not bugs.
                  </p>
                  <p>
                    Would these small differences affect text generation
                    quality? No. The top-k tokens and their relative
                    probabilities would be essentially identical. A difference
                    at the 4th decimal place of a logit does not change which
                    word is most likely.
                  </p>
                </div>
              </details>
            </div>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* ================================================================
          14. Payoff: Generate Real Text
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Generate Real Text"
            subtitle="Your code. OpenAI&rsquo;s knowledge. Real GPT-2."
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              This is the moment the entire module has been building toward.
              Your own GPT class, with OpenAI&rsquo;s pretrained weights,
              generating genuinely coherent English text. Not
              Shakespeare-trained-on-a-laptop. Real GPT-2.
            </p>

            <CodeBlock
              code={`prompts = [
    "The meaning of life is",
    "The capital of France is",
    "Once upon a time in a land far away",
    "The transformer architecture consists of",
]

model_ours.eval()
for prompt in prompts:
    tokens = enc.encode(prompt)
    idx = torch.tensor([tokens])
    generated = model_ours.generate(idx, max_new_tokens=50, temperature=0.8)
    print(f"Prompt: {prompt}")
    print(f"Output: {enc.decode(generated[0].tolist())}")
    print()`}
              language="python"
              filename="generate_text.py"
            />

            <p className="text-muted-foreground">
              Try multiple prompts. Notice the quality: grammatical
              sentences, factual knowledge, coherent reasoning. Compare this
              to the gibberish from Building nanoGPT&rsquo;s untrained
              model, and to the Shakespeare fragments from Pretraining.
            </p>

            <div className="px-4 py-4 bg-muted/50 rounded-lg space-y-3">
              <p className="text-sm font-medium">
                The full arc of Module 4.3
              </p>
              <div className="text-sm text-muted-foreground space-y-2">
                <p>
                  <span className="text-rose-400 font-medium">
                    Random weights
                  </span>{' '}
                  {'→'} gibberish
                </p>
                <p>
                  <span className="text-amber-400 font-medium">
                    Trained on Shakespeare
                  </span>{' '}
                  {'→'} recognizable English
                </p>
                <p>
                  <span className="text-emerald-400 font-medium">
                    Real GPT-2 weights
                  </span>{' '}
                  {'→'} coherent, knowledgeable text
                </p>
              </div>
              <p className="text-xs text-muted-foreground/80">
                Your code did not change. The weights changed. That is what
                pretraining buys.
              </p>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The Vessel and the Knowledge">
            The architecture is the vessel. The weights are the knowledge.
            Your code defines <em>what</em> the model can compute. The
            pretrained weights encode <em>what</em> it has learned about
            language. Same code, different weights, dramatically different
            behavior.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          15. What the Weights Contain
          ================================================================ */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The weights you just loaded encode everything GPT-2 learned from
              WebText: grammar, facts, reasoning patterns, writing styles.
              When you loaded those 124 million parameters into your
              architecture, you transferred all of that knowledge into your
              code.
            </p>

            <p className="text-muted-foreground">
              This connects directly to what comes after this module.
              Fine-tuning starts with these pretrained weights and adapts
              them to specific tasks. The base knowledge is
              general&mdash;the adaptation makes it useful. But that is Module
              4.4.
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* ================================================================
          16. Notebook Link
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Load It Yourself"
            subtitle="Open the notebook and run the mapping"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The lesson showed you the concepts. Now do it yourself. The
              notebook walks you through: downloading the reference model,
              comparing state dicts, building the weight mapping, verifying
              with logit comparison, and generating text from your loaded
              model.
            </p>

            <div className="rounded-lg border-2 border-primary/50 bg-primary/5 p-6">
              <div className="space-y-4">
                <p className="text-muted-foreground">
                  Load real GPT-2 weights into your own architecture.
                </p>
                <a
                  href="https://colab.research.google.com/github/brennancheung/CourseAI/blob/main/notebooks/4-3-3-loading-real-weights.ipynb"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-2 px-4 py-2 rounded-md bg-primary text-primary-foreground font-medium text-sm hover:bg-primary/90 transition-colors"
                >
                  <ExternalLink className="w-4 h-4" />
                  Open in Google Colab
                </a>
                <p className="text-xs text-muted-foreground">
                  The notebook includes: comparing state dict keys,
                  building the weight mapping, the transposition negative
                  example, logit verification against HuggingFace, and text
                  generation with multiple prompts. Stretch exercise:
                  load GPT-2 medium (345M) by updating the config.
                </p>
              </div>
            </div>
          </div>
        </Row.Content>
      </Row>

      {/* ================================================================
          17. Summary
          ================================================================ */}
      <Row>
        <Row.Content>
          <SummaryBlock
            title="What You Built"
            items={[
              {
                headline:
                  'The weight mapping IS the verification.',
                description:
                  'Every shape match is a component verified. Every matching logit is a computation confirmed. If real weights work in your model, your implementation is correct.',
              },
              {
                headline:
                  'Conv1D vs nn.Linear: same parameters, different layout.',
                description:
                  `HuggingFace's GPT-2 uses Conv1D (in, out). Your model uses nn.Linear (out, in). The fix is .t()—one transpose per 2D weight in attention and FFN.`,
              },
              {
                headline:
                  'Logit comparison is the gold standard.',
                description:
                  'Same input, same weights, same computation—if torch.allclose returns True, the models are functionally identical. Stronger than parameter counting. Stronger than text comparison.',
              },
              {
                headline:
                  'The architecture is the vessel. The weights are the knowledge.',
                description:
                  'Your code defines what the model can compute. The pretrained weights encode what it has learned. Same code, different weights, dramatically different behavior.',
              },
            ]}
          />
        </Row.Content>
      </Row>

      {/* Echo the module arc */}
      <Row>
        <Row.Content>
          <div className="px-4 py-4 bg-violet-500/10 border border-violet-500/20 rounded-lg space-y-3">
            <p className="text-sm font-medium text-violet-400">
              Module 4.3: The Complete Arc
            </p>
            <div className="text-sm text-muted-foreground space-y-2">
              <p>
                <strong>Building nanoGPT:</strong> Assembled the architecture.
                Generated gibberish. &ldquo;The architecture works.&rdquo;
              </p>
              <p>
                <strong>Pretraining:</strong> Trained on Shakespeare. Watched
                quality improve. &ldquo;Training works.&rdquo;
              </p>
              <p>
                <strong>Scaling & Efficiency:</strong> Understood the
                engineering that makes it practical at scale.
              </p>
              <p>
                <strong>Loading Real Weights:</strong> Loaded real weights.
                Verified against OpenAI. Generated coherent text.{' '}
                &ldquo;Your implementation is correct.&rdquo;
              </p>
            </div>
            <p className="text-xs text-muted-foreground/70">
              The verification chain: parameter count (right shapes) + logit
              comparison (right computation) + coherent generation (right
              behavior). Three levels of evidence, all confirmed.
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* ================================================================
          18. Module Complete
          ================================================================ */}
      <Row>
        <Row.Content>
          <ModuleCompleteBlock
            module="4.3"
            title="Building & Training GPT"
            achievements={[
              'Complete GPT architecture in PyTorch (Head, CausalSelfAttention, FeedForward, Block, GPT)',
              'Training loop with LR scheduling, gradient clipping, and loss interpretation',
              'Engineering layer: mixed precision, KV caching, flash attention, scaling laws',
              'Weight mapping between codebases with Conv1D transposition handling',
              'Logit verification against a reference implementation',
              'Coherent text generation from pretrained GPT-2 weights',
            ]}
            nextModule="4.4"
            nextTitle="Fine-tuning & Alignment"
          />
        </Row.Content>
      </Row>

      {/* ================================================================
          19. Next Step
          ================================================================ */}
      <Row>
        <Row.Content>
          <div className="px-4 py-3 bg-primary/10 border border-primary/30 rounded-lg space-y-2">
            <p className="text-sm font-medium text-primary">
              What comes next
            </p>
            <p className="text-sm text-muted-foreground">
              You have a working, verified GPT-2 implementation. In the next
              module, you will learn what comes after pretraining: taking
              these pretrained weights and adapting them for specific tasks
              through fine-tuning, instruction tuning, and alignment.
            </p>
          </div>
        </Row.Content>
      </Row>

      <Row>
        <Row.Content>
          <NextStepBlock
            href="/app"
            title="When you're done"
            description="Complete the notebook: load the weights, verify with logit comparison, and generate text from your own GPT-2. Then review your session."
          />
        </Row.Content>
      </Row>
    </LessonLayout>
  )
}

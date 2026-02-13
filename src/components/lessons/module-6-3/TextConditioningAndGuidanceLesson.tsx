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
  ConceptBlock,
  GradientCard,
  ComparisonRow,
  SummaryBlock,
  NextStepBlock,
  ReferencesBlock,
} from '@/components/lessons'
import { CodeBlock } from '@/components/common/CodeBlock'
import { MermaidDiagram } from '@/components/widgets/MermaidDiagram'
import 'katex/dist/katex.min.css'
import { InlineMath, BlockMath } from 'react-katex'

/**
 * Text Conditioning & Guidance
 *
 * Lesson 4 in Module 6.3 (Architecture & Conditioning). Lesson 13 overall in Series 6.
 * Cognitive load: BUILD (cross-attention is a small delta from deep self-attention
 * knowledge; CFG is procedurally simple but conceptually important).
 *
 * Teaches how text descriptions steer the diffusion U-Net's denoising process:
 * - Cross-attention as the mechanism for injecting CLIP text embeddings (Q from spatial, K/V from text)
 * - Where cross-attention layers live in the U-Net (interleaved at 16x16 and 32x32)
 * - Spatially-varying conditioning vs global conditioning contrast
 * - Classifier-free guidance: training trick, inference formula, geometric intuition
 * - Guidance scale tradeoff (w=1, w=7.5, w=20)
 *
 * Core concepts:
 * - Cross-attention mechanism: DEVELOPED
 * - Classifier-free guidance: DEVELOPED
 * - Spatially-varying vs global conditioning: INTRODUCED
 * - Self-attention within the U-Net: INTRODUCED
 * - Classifier guidance (predecessor): MENTIONED
 *
 * Previous: CLIP (module 6.3, lesson 3)
 * Next: From Pixels to Latents (module 6.3, lesson 5)
 */

export function TextConditioningAndGuidanceLesson() {
  return (
    <LessonLayout>
      {/* ================================================================
          Section 1: Header
          ================================================================ */}
      <Row>
        <Row.Content>
          <LessonHeader
            title="Text Conditioning & Guidance"
            description="How cross-attention injects CLIP text embeddings into the U-Net&mdash;and how classifier-free guidance amplifies their influence at inference time."
            category="Architecture & Conditioning"
          />
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 2: Context + Constraints
          ================================================================ */}
      <Row>
        <Row.Content>
          <ObjectiveBlock>
            Understand the two mechanisms that let a text prompt control image
            generation: <strong>cross-attention</strong> (the same QKV formula
            from Module 4.2, applied to inject text embeddings into the U-Net)
            and <strong>classifier-free guidance</strong> (a training trick +
            inference formula that amplifies the text&rsquo;s influence). By the
            end, you can trace the full conditioning pipeline&mdash;timestep via
            adaptive norm (global), text via cross-attention (spatially varying),
            CFG to strengthen text adherence.
          </ObjectiveBlock>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="What You Already Know">
            You built the QKV attention formula across three lessons in{' '}
            <strong>The Problem Attention Solves</strong> through{' '}
            <strong>Values and the Attention Output</strong>. Cross-attention
            uses the <em>exact same formula</em>&mdash;the only change is where
            K and V come from.
          </TipBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <ConstraintBlock
            title="Scope for This Lesson"
            items={[
              'Cross-attention: how CLIP text embeddings enter the U-Net via the QKV formula',
              'Where cross-attention layers sit in the U-Net (interleaved at 16×16 and 32×32)',
              'Classifier-free guidance: training with random text dropout, inference with two forward passes',
              'The guidance scale parameter and its quality/fidelity tradeoff',
              'NOT: implementing cross-attention or CFG from scratch—that is the Module 6.4 capstone',
              'NOT: latent diffusion or the VAE encoder-decoder—next lesson',
              'NOT: negative prompts, prompt engineering, or alternative text encoders (T5 in Imagen)',
              'NOT: CLIP architecture details—covered in the previous lesson',
            ]}
          />
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 3: Recap — Cross-Attention Gap Fill + CLIP Clarification
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Quick Recap: Two Pieces on the Table"
            subtitle="Self-attention review and a CLIP embedding clarification"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              In <strong>Decoder-Only Transformers</strong>, you saw that
              encoder-decoder transformers have &ldquo;cross-attention&rdquo;
              where the decoder attends to the encoder&rsquo;s output. You know
              the concept exists: Q comes from one input, K and V from another.
              This lesson develops the mechanism.
            </p>
            <p className="text-muted-foreground">
              Quick callback: you know the QKV formula deeply from{' '}
              <strong>Values and the Attention Output</strong>. In
              self-attention, Q, K, and V are all projected from the same input.
              In cross-attention, Q comes from one input and K/V from another.{' '}
              <strong>Same formula, different sources.</strong>
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="Self-Attention (Review)">
            <div className="space-y-1 text-sm">
              <p>Q = W<sub>Q</sub> X</p>
              <p>K = W<sub>K</sub> X</p>
              <p>V = W<sub>V</sub> X</p>
              <p className="pt-1 text-xs text-muted-foreground">
                All three come from the same input X.
              </p>
            </div>
          </ConceptBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              One clarification about CLIP embeddings. In the previous lesson,
              you focused on the single summary vector CLIP produces for
              text-image comparison. But CLIP&rsquo;s text encoder actually
              produces a <strong>sequence</strong> of token embeddings&mdash;one
              per text token in the prompt. For the prompt &ldquo;a cat sitting
              in a sunset,&rdquo; that is roughly 7 token embeddings, each
              512-dim. This full sequence is what the U-Net will read from via
              cross-attention. Each word contributes its own key and value
              vector.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Sequence, Not Summary">
            For text-image comparison, CLIP uses a single [CLS] vector. For
            conditioning the U-Net, we use the full sequence of token
            embeddings&mdash;one per word. Different information, same encoder.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 4: Hook — "The U-Net Still Cannot Understand Text"
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The U-Net Still Cannot Understand Text"
            subtitle="Two pieces, no connection"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Recall the end of <strong>Conditioning the U-Net</strong>:
              &ldquo;The timestep tells the network <em>when</em> it is in the
              denoising process. The text will tell it <em>what</em> to
              generate.&rdquo; You now have both pieces&mdash;a U-Net with
              timestep conditioning, and CLIP text embeddings encoding visual
              meaning. But they are disconnected. How do you inject a sequence
              of text embeddings into a convolutional architecture?
            </p>
            <p className="text-muted-foreground">
              Your first instinct might be: average all the text token
              embeddings into one vector and inject it via adaptive
              normalization, just like the timestep. But this loses spatial
              selectivity. &ldquo;A cat sitting in a sunset&rdquo; would push
              the <em>entire</em> image toward cat-ness and sunset-ness
              uniformly. The word &ldquo;cat&rdquo; should influence the cat
              region, not the sky. The word &ldquo;sunset&rdquo; should
              influence the sky, not the cat.
            </p>
            <p className="text-muted-foreground">
              What you need: a mechanism where each spatial location can{' '}
              <strong>selectively extract</strong> the relevant information from
              the text. You already know a mechanism that does exactly
              this.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Why Not Reuse Adaptive Norm?">
            Adaptive normalization injects the{' '}
            <strong>same signal</strong> at every spatial location&mdash;that is
            what makes it &ldquo;global conditioning.&rdquo; Text conditioning
            is inherently spatially varying: different regions need different
            words. A global mechanism is the wrong tool.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 5: Explain — Cross-Attention in the U-Net (DEVELOPED)
          ================================================================ */}

      {/* 5a: The one-line change from self-attention */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Cross-Attention: The One-Line Change"
            subtitle="Same formula, different source for K and V"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              In self-attention, each token asks &ldquo;what in{' '}
              <em>my sequence</em> is relevant to me?&rdquo; In cross-attention,
              each spatial location asks &ldquo;what in <em>the text</em> is
              relevant to me?&rdquo; The formula does not change at all. The
              only difference is where K and V come from:
            </p>
            <ComparisonRow
              left={{
                title: 'Self-Attention',
                color: 'blue',
                items: [
                  'Q = W_Q · X',
                  'K = W_K · X',
                  'V = W_V · X',
                  'All from the same input X',
                ],
              }}
              right={{
                title: 'Cross-Attention',
                color: 'violet',
                items: [
                  'Q = W_Q · X_spatial',
                  'K = W_K · X_text',
                  'V = W_V · X_text',
                  'Q from spatial features, K/V from text',
                ],
              }}
            />
            <div className="py-4 px-6 bg-muted/50 rounded-lg">
              <p className="text-sm font-medium text-foreground mb-2">
                Output formula&mdash;identical in both cases:
              </p>
              <BlockMath math="\text{output} = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V" />
            </div>
            <p className="text-muted-foreground">
              Same three projection matrices. Same dot-product scoring. Same
              softmax. Same weighted average. You built this formula across
              three lessons in Module 4.2. Cross-attention is the same
              mechanism applied to a different question: instead of &ldquo;what
              in my sequence is related?&rdquo; it asks &ldquo;what in the text
              is related to this spatial position?&rdquo;
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Same Formula, New Application">
            This is the exact formula from{' '}
            <strong>Values and the Attention Output</strong>. The only change
            is where K and V come from&mdash;text embeddings instead of the
            same input. Same building blocks, different question.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Cross-attention data flow diagram */}
      <Row>
        <Row.Content>
          <div className="rounded-lg bg-muted/30 p-4 space-y-3">
            <p className="text-sm font-medium text-foreground text-center">
              Cross-Attention Data Flow
            </p>
            <svg viewBox="0 0 520 260" className="w-full max-w-[540px] mx-auto" aria-label="Cross-attention data flow: spatial features on the left produce queries via W_Q, text tokens on the right produce keys and values via W_K and W_V, attention weights connect them with varying thickness">
              {/* Spatial feature grid (4x4 simplified) */}
              {/* Row 1 */}
              <rect x="20" y="30" width="22" height="22" rx="2" fill="#3b82f6" opacity="0.6" />
              <rect x="46" y="30" width="22" height="22" rx="2" fill="#3b82f6" opacity="0.7" />
              <rect x="72" y="30" width="22" height="22" rx="2" fill="#3b82f6" opacity="0.5" />
              <rect x="98" y="30" width="22" height="22" rx="2" fill="#3b82f6" opacity="0.8" />
              {/* Row 2 */}
              <rect x="20" y="56" width="22" height="22" rx="2" fill="#3b82f6" opacity="0.8" />
              <rect x="46" y="56" width="22" height="22" rx="2" fill="#3b82f6" opacity="0.6" />
              <rect x="72" y="56" width="22" height="22" rx="2" fill="#3b82f6" opacity="0.9" />
              <rect x="98" y="56" width="22" height="22" rx="2" fill="#3b82f6" opacity="0.5" />
              {/* Row 3 */}
              <rect x="20" y="82" width="22" height="22" rx="2" fill="#3b82f6" opacity="0.7" />
              <rect x="46" y="82" width="22" height="22" rx="2" fill="#3b82f6" opacity="0.9" />
              <rect x="72" y="82" width="22" height="22" rx="2" fill="#3b82f6" opacity="0.6" />
              <rect x="98" y="82" width="22" height="22" rx="2" fill="#3b82f6" opacity="0.7" />
              {/* Row 4 */}
              <rect x="20" y="108" width="22" height="22" rx="2" fill="#3b82f6" opacity="0.5" />
              <rect x="46" y="108" width="22" height="22" rx="2" fill="#3b82f6" opacity="0.7" />
              <rect x="72" y="108" width="22" height="22" rx="2" fill="#3b82f6" opacity="0.8" />
              <rect x="98" y="108" width="22" height="22" rx="2" fill="#3b82f6" opacity="0.6" />

              <text x="70" y="152" fontSize="10" fill="#94a3b8" textAnchor="middle">Spatial Features</text>
              <text x="70" y="164" fontSize="9" fill="#64748b" textAnchor="middle">(4×4 = 16 locations)</text>

              {/* Text token column */}
              <rect x="410" y="18" width="80" height="24" rx="4" fill="#8b5cf6" opacity="0.7" />
              <text x="450" y="34" fontSize="9" fill="#e2e8f0" textAnchor="middle">a</text>

              <rect x="410" y="48" width="80" height="24" rx="4" fill="#8b5cf6" opacity="0.9" />
              <text x="450" y="64" fontSize="9" fill="#e2e8f0" textAnchor="middle">cat</text>

              <rect x="410" y="78" width="80" height="24" rx="4" fill="#8b5cf6" opacity="0.8" />
              <text x="450" y="94" fontSize="9" fill="#e2e8f0" textAnchor="middle">sitting</text>

              <rect x="410" y="108" width="80" height="24" rx="4" fill="#8b5cf6" opacity="0.6" />
              <text x="450" y="124" fontSize="9" fill="#e2e8f0" textAnchor="middle">in</text>

              <rect x="410" y="138" width="80" height="24" rx="4" fill="#8b5cf6" opacity="0.5" />
              <text x="450" y="154" fontSize="9" fill="#e2e8f0" textAnchor="middle">a</text>

              <rect x="410" y="168" width="80" height="24" rx="4" fill="#8b5cf6" opacity="0.85" />
              <text x="450" y="184" fontSize="9" fill="#e2e8f0" textAnchor="middle">sunset</text>

              <text x="450" y="210" fontSize="10" fill="#94a3b8" textAnchor="middle">CLIP Text Tokens</text>
              <text x="450" y="222" fontSize="9" fill="#64748b" textAnchor="middle">(T = 6 tokens)</text>

              {/* W_Q arrow from spatial features */}
              <line x1="125" y1="67" x2="175" y2="67" stroke="#3b82f6" strokeWidth="2" markerEnd="url(#arrowBlue)" />
              <text x="150" y="60" fontSize="9" fill="#3b82f6" textAnchor="middle" fontWeight="bold">W_Q</text>

              {/* Q label */}
              <text x="195" y="62" fontSize="10" fill="#3b82f6" fontWeight="bold">Q</text>
              <text x="195" y="74" fontSize="8" fill="#64748b">(16, d_k)</text>

              {/* W_K arrow from text tokens */}
              <line x1="405" y1="55" x2="355" y2="55" stroke="#8b5cf6" strokeWidth="2" markerEnd="url(#arrowVioletCross)" />
              <text x="380" y="48" fontSize="9" fill="#8b5cf6" textAnchor="middle" fontWeight="bold">W_K</text>

              {/* K label */}
              <text x="325" y="50" fontSize="10" fill="#8b5cf6" fontWeight="bold">K</text>
              <text x="325" y="62" fontSize="8" fill="#64748b">(6, d_k)</text>

              {/* W_V arrow from text tokens */}
              <line x1="405" y1="145" x2="355" y2="145" stroke="#8b5cf6" strokeWidth="2" markerEnd="url(#arrowVioletCross)" />
              <text x="380" y="138" fontSize="9" fill="#8b5cf6" textAnchor="middle" fontWeight="bold">W_V</text>

              {/* V label */}
              <text x="325" y="140" fontSize="10" fill="#8b5cf6" fontWeight="bold">V</text>
              <text x="325" y="152" fontSize="8" fill="#64748b">(6, d_v)</text>

              {/* Attention weight connections (varying thickness) */}
              {/* From a "cat region" spatial location to text tokens */}
              <line x1="210" y1="67" x2="320" y2="30" stroke="#f59e0b" strokeWidth="0.5" opacity="0.3" />
              <line x1="210" y1="67" x2="320" y2="60" stroke="#f59e0b" strokeWidth="3" opacity="0.8" />
              <line x1="210" y1="67" x2="320" y2="90" stroke="#f59e0b" strokeWidth="1.5" opacity="0.5" />
              <line x1="210" y1="67" x2="320" y2="120" stroke="#f59e0b" strokeWidth="0.5" opacity="0.3" />
              <line x1="210" y1="67" x2="320" y2="150" stroke="#f59e0b" strokeWidth="0.5" opacity="0.2" />
              <line x1="210" y1="67" x2="320" y2="180" stroke="#f59e0b" strokeWidth="1" opacity="0.4" />

              {/* Attention weights label */}
              <text x="260" y="100" fontSize="8" fill="#f59e0b" textAnchor="middle" fontStyle="italic">attention</text>
              <text x="260" y="110" fontSize="8" fill="#f59e0b" textAnchor="middle" fontStyle="italic">weights</text>
              <text x="260" y="120" fontSize="7" fill="#64748b" textAnchor="middle">(16 × 6)</text>

              {/* Legend */}
              <rect x="20" y="235" width="12" height="12" rx="2" fill="#3b82f6" opacity="0.7" />
              <text x="37" y="245" fontSize="8" fill="#94a3b8">spatial feature</text>
              <rect x="120" y="235" width="12" height="12" rx="2" fill="#8b5cf6" opacity="0.7" />
              <text x="137" y="245" fontSize="8" fill="#94a3b8">text token</text>
              <line x1="220" y1="241" x2="240" y2="241" stroke="#f59e0b" strokeWidth="2" opacity="0.7" />
              <text x="245" y="245" fontSize="8" fill="#94a3b8">attention (thicker = stronger)</text>

              {/* Arrow marker definitions */}
              <defs>
                <marker id="arrowBlue" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
                  <path d="M0,0 L6,3 L0,6 Z" fill="#3b82f6" />
                </marker>
                <marker id="arrowVioletCross" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
                  <path d="M0,0 L6,3 L0,6 Z" fill="#8b5cf6" />
                </marker>
              </defs>
            </svg>
            <p className="text-xs text-muted-foreground text-center">
              Each spatial location generates a query (via W<sub>Q</sub>). Each text
              token produces a key and value (via W<sub>K</sub>, W<sub>V</sub>). The
              attention weights are <strong>rectangular</strong> (16 × 6)&mdash;not
              square&mdash;because Q and K come from different inputs.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="Two Inputs, One Output">
            The spatial features provide the <strong>questions</strong> (queries).
            The text tokens provide the <strong>answers</strong> (keys and values).
            The output has the same shape as the spatial input&mdash;each location
            gets enriched by whichever text tokens are most relevant.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* 5b: Per-spatial-location attention */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-sm font-medium text-foreground">
              Per-spatial-location attention
            </p>
            <p className="text-muted-foreground">
              The 16×16 feature map has 256 spatial locations. Each location
              generates its own query vector. Each query independently attends
              over <em>all</em> text tokens. The result: different spatial
              locations can attend to different words.
            </p>
            <p className="text-sm font-medium text-foreground mt-4">
              Shape walkthrough:
            </p>
            <ul className="list-disc list-inside text-muted-foreground space-y-1 ml-4">
              <li>
                <InlineMath math="X_{\text{spatial}}" /> reshaped to{' '}
                <InlineMath math="(H \!\cdot\! W, \, d)" /> = (256, d) for a
                16×16 feature map
              </li>
              <li>
                <InlineMath math="X_{\text{text}}" /> is{' '}
                <InlineMath math="(T, \, d_{\text{text}})" /> where T ≈ 77
                (CLIP&rsquo;s max text tokens)
              </li>
              <li>
                Q is <InlineMath math="(256, \, d_k)" />, K is{' '}
                <InlineMath math="(T, \, d_k)" />, V is{' '}
                <InlineMath math="(T, \, d_v)" />
              </li>
              <li>
                Attention weights:{' '}
                <InlineMath math="(256, \, T)" />&mdash;each of 256 spatial
                locations has its own distribution over T text tokens
              </li>
            </ul>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Not a Square Matrix">
            In self-attention from Module 4.2, the attention matrix was square
            (each token attends to every other token). In cross-attention, it
            is <strong>rectangular</strong>: 256 spatial locations × ~77 text
            tokens. Different dimensions because Q and K come from different
            inputs.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* 5b continued: Concrete example */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-sm font-medium text-foreground">
              Concrete example: &ldquo;a cat sitting in a sunset&rdquo;
            </p>
            <p className="text-muted-foreground">
              Consider three spatial locations in a 16×16 feature map attending
              to the 6 text tokens: [a, cat, sitting, in, a, sunset].
            </p>
            <div className="overflow-x-auto">
              <table className="mx-auto text-xs border-collapse">
                <thead>
                  <tr>
                    <th className="p-2" />
                    <th className="p-2 text-center font-medium text-muted-foreground">a</th>
                    <th className="p-2 text-center font-medium text-muted-foreground">cat</th>
                    <th className="p-2 text-center font-medium text-muted-foreground">sitting</th>
                    <th className="p-2 text-center font-medium text-muted-foreground">in</th>
                    <th className="p-2 text-center font-medium text-muted-foreground">a</th>
                    <th className="p-2 text-center font-medium text-muted-foreground">sunset</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td className="p-2 text-right font-medium text-muted-foreground pr-3">
                      Cat&rsquo;s face region
                    </td>
                    <td className="p-2 text-center rounded text-muted-foreground bg-muted/30">0.03</td>
                    <td className="p-2 text-center rounded font-bold text-white bg-violet-500">0.62</td>
                    <td className="p-2 text-center rounded text-muted-foreground bg-violet-500/30">0.18</td>
                    <td className="p-2 text-center rounded text-muted-foreground bg-muted/30">0.04</td>
                    <td className="p-2 text-center rounded text-muted-foreground bg-muted/30">0.02</td>
                    <td className="p-2 text-center rounded text-muted-foreground bg-muted/30">0.11</td>
                  </tr>
                  <tr>
                    <td className="p-2 text-right font-medium text-muted-foreground pr-3">
                      Sky region
                    </td>
                    <td className="p-2 text-center rounded text-muted-foreground bg-muted/30">0.02</td>
                    <td className="p-2 text-center rounded text-muted-foreground bg-muted/30">0.08</td>
                    <td className="p-2 text-center rounded text-muted-foreground bg-muted/30">0.05</td>
                    <td className="p-2 text-center rounded text-muted-foreground bg-muted/30">0.03</td>
                    <td className="p-2 text-center rounded text-muted-foreground bg-muted/30">0.02</td>
                    <td className="p-2 text-center rounded font-bold text-white bg-violet-500">0.80</td>
                  </tr>
                  <tr>
                    <td className="p-2 text-right font-medium text-muted-foreground pr-3">
                      Cat&rsquo;s body region
                    </td>
                    <td className="p-2 text-center rounded text-muted-foreground bg-muted/30">0.02</td>
                    <td className="p-2 text-center rounded text-muted-foreground bg-violet-500/30">0.31</td>
                    <td className="p-2 text-center rounded font-bold text-white bg-violet-500">0.45</td>
                    <td className="p-2 text-center rounded text-muted-foreground bg-muted/30">0.06</td>
                    <td className="p-2 text-center rounded text-muted-foreground bg-muted/30">0.02</td>
                    <td className="p-2 text-center rounded text-muted-foreground bg-muted/30">0.14</td>
                  </tr>
                </tbody>
              </table>
            </div>
            <p className="text-xs text-muted-foreground text-center mt-1">
              Each row sums to 1.0 (softmax). The cat&rsquo;s face attends
              strongly to &ldquo;cat.&rdquo; The sky attends to
              &ldquo;sunset.&rdquo; The body attends to both &ldquo;cat&rdquo;
              and &ldquo;sitting.&rdquo;
            </p>
            <p className="text-muted-foreground">
              This is what makes text conditioning{' '}
              <strong>spatially varying</strong>. The timestep modulates all
              locations the same way (global). The text gives each location its
              own signal. The cat&rsquo;s ear and the sky get different
              information from the same prompt.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Spatially Varying Conditioning">
            The timestep says &ldquo;remove this much noise everywhere.&rdquo;
            The text says &ldquo;this region should be a cat, that region
            should be a sunset.&rdquo; Different mechanism, different purpose.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* 5c: Connection to the "three lenses" mental model */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-sm font-medium text-foreground">
              The three lenses, revisited
            </p>
            <p className="text-muted-foreground">
              Remember the &ldquo;learned lens&rdquo; pattern from{' '}
              <strong>Queries and Keys</strong>? Same input, different
              projection matrix, different view. In cross-attention, the same
              pattern applies with a twist:
            </p>
            <ul className="list-disc list-inside text-muted-foreground space-y-1 ml-4">
              <li>
                <strong>W<sub>Q</sub></strong> is the spatial features&rsquo;{' '}
                <em>seeking</em> lens: &ldquo;what kind of text information do
                I need here?&rdquo;
              </li>
              <li>
                <strong>W<sub>K</sub></strong> is the text&rsquo;s{' '}
                <em>advertising</em> lens: &ldquo;what information does each
                word offer?&rdquo;
              </li>
              <li>
                <strong>W<sub>V</sub></strong> is the text&rsquo;s{' '}
                <em>contributing</em> lens: &ldquo;what information does each
                word actually deliver when matched?&rdquo;
              </li>
            </ul>
            <p className="text-muted-foreground">
              Same three-lens pattern from{' '}
              <strong>Values and the Attention Output</strong>. The novelty: the
              Q lens looks at spatial features, the K/V lenses look at text.
              Self-attention reads from other spatial features.
              Cross-attention reads from the text. Same &ldquo;reading&rdquo;
              operation, different source document.
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 6: Explain — Where Cross-Attention Lives (INTRODUCED)
          ================================================================ */}

      {/* 6a: Block ordering at attention resolutions */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Where Cross-Attention Lives in the U-Net"
            subtitle="Block ordering at attention resolutions"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              At middle resolutions (16×16 and 32×32), the U-Net interleaves
              three types of processing at each stage:
            </p>
            <ol className="list-decimal list-inside text-muted-foreground space-y-1 ml-4">
              <li>
                <strong>Residual block</strong> with adaptive group
                normalization (timestep conditioning)
              </li>
              <li>
                <strong>Self-attention</strong> (spatial features attend to each
                other)
              </li>
              <li>
                <strong>Cross-attention</strong> (spatial features attend to
                text embeddings)
              </li>
            </ol>
            <p className="text-muted-foreground">
              This pattern repeats at each attention resolution, on both the
              encoder and decoder paths of the U-Net:
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="Three Processing Types">
            <ul className="space-y-1 text-sm">
              <li>
                &bull; <strong>Residual block:</strong> local features +
                timestep
              </li>
              <li>
                &bull; <strong>Self-attention:</strong> spatial coherence
              </li>
              <li>
                &bull; <strong>Cross-attention:</strong> text guidance
              </li>
            </ul>
          </ConceptBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <MermaidDiagram chart={`
graph LR
    RB1["Residual Block\\n(AdaGN + t_emb)"] --> SA1["Self-\\nAttention"]
    SA1 --> CA1["Cross-\\nAttention\\n(text_emb)"]
    CA1 --> RB2["Residual Block\\n(AdaGN + t_emb)"]
    RB2 --> SA2["Self-\\nAttention"]
    SA2 --> CA2["Cross-\\nAttention\\n(text_emb)"]

    style RB1 fill:#1e293b,stroke:#f59e0b,color:#e2e8f0
    style SA1 fill:#1e293b,stroke:#3b82f6,color:#e2e8f0
    style CA1 fill:#1e293b,stroke:#8b5cf6,color:#e2e8f0
    style RB2 fill:#1e293b,stroke:#f59e0b,color:#e2e8f0
    style SA2 fill:#1e293b,stroke:#3b82f6,color:#e2e8f0
    style CA2 fill:#1e293b,stroke:#8b5cf6,color:#e2e8f0
`} />
          <p className="text-sm text-muted-foreground italic mt-2">
            Block ordering at an attention resolution (e.g., 16×16).{' '}
            <span className="text-amber-400">Amber</span> = timestep
            conditioning via AdaGN.{' '}
            <span className="text-blue-400">Blue</span> = self-attention.{' '}
            <span className="text-violet-400">Violet</span> = cross-attention
            with text embeddings.
          </p>
        </Row.Content>
      </Row>

      {/* 6b: Why only at middle resolutions */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-sm font-medium text-foreground">
              Why only at middle resolutions?
            </p>
            <p className="text-muted-foreground">
              Attention is O(n²) in sequence length. At 64×64 = 4,096
              &ldquo;tokens,&rdquo; the self-attention matrix alone would be
              4,096 × 4,096 = 16.7 million entries. Expensive and
              memory-heavy.
            </p>
            <p className="text-muted-foreground">
              At 16×16 = 256 &ldquo;tokens,&rdquo; the self-attention matrix
              is 256 × 256 = 65K entries. The cross-attention matrix is 256 ×
              77 ≈ 20K entries. Feasible. At 8×8 = 64 tokens, the bottleneck
              already has global receptive field, so self-attention adds less
              value.
            </p>
            <p className="text-muted-foreground">
              The sweet spot: <strong>16×16 and 32×32</strong>. Large enough to
              benefit from long-range spatial dependencies, small enough for
              attention to be affordable.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Quadratic Cost Constraint">
            The same O(n²) cost you learned about in{' '}
            <strong>The Problem Attention Solves</strong> constrains where
            attention layers go in the U-Net. Architecture decisions have
            computational consequences.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* 6c: WarningBlock — both conditioning signals coexist */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Both conditioning signals are present simultaneously in the same
              forward pass. They serve different purposes and do not compete:
            </p>
            <ComparisonRow
              left={{
                title: 'Timestep Conditioning',
                color: 'amber',
                items: [
                  'Via adaptive group normalization',
                  'In every residual block, every resolution',
                  'Global: same signal at all spatial locations',
                  'Tells the network WHEN (noise level)',
                ],
              }}
              right={{
                title: 'Text Conditioning',
                color: 'violet',
                items: [
                  'Via cross-attention',
                  'At attention resolutions (16×16, 32×32) only',
                  'Spatially varying: each location gets its own signal',
                  'Tells the network WHAT (semantic content)',
                ],
              }}
            />
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="They Coexist, Not Compete">
            Text conditioning does not replace timestep conditioning. Both are
            present in every forward pass. The residual block handles
            &ldquo;how much noise to remove.&rdquo; Cross-attention handles
            &ldquo;what should be revealed as noise is removed.&rdquo;
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* 6d: Self-attention within the U-Net (INTRODUCED, brief) */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-sm font-medium text-foreground">
              Self-attention in the U-Net (brief)
            </p>
            <p className="text-muted-foreground">
              Self-attention lets different spatial locations communicate
              directly. At 16×16, one part of the image can attend to another
              part 16 pixels away in a single layer. This enables global
              spatial coherence: if the model is generating a face, the left
              eye and right eye can &ldquo;coordinate&rdquo; through
              self-attention to be symmetric.
            </p>
            <p className="text-muted-foreground">
              Self-attention handles <strong>spatial consistency</strong>.
              Cross-attention handles <strong>text guidance</strong>. They sit
              side by side in the same block.
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 7: Explain — Classifier-Free Guidance (DEVELOPED)
          ================================================================ */}

      {/* 7a: The problem CFG solves */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Classifier-Free Guidance"
            subtitle="Amplifying the text signal at inference time"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Cross-attention gives the model <em>access</em> to text
              information, but it does not guarantee the model will{' '}
              <em>use</em> it strongly. During training, the model minimizes
              reconstruction error. If it can achieve low error by mostly
              ignoring the text&mdash;relying on the noisy image itself for
              predictions&mdash;it will. The text signal is one of many inputs,
              and the model may learn to weight it weakly.
            </p>
            <p className="text-muted-foreground">
              The result: text conditioning works but produces images that only
              loosely follow the prompt. &ldquo;A cat&rdquo; generates
              something cat-like but with low fidelity. You need a way to
              turn up the volume on the text signal.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Access ≠ Influence">
            Cross-attention provides the channel for text information. CFG
            controls the volume. Without CFG, the channel exists but the
            signal is often too quiet.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* 7b: The CFG training trick */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-sm font-medium text-foreground">
              The training trick: random text dropout
            </p>
            <p className="text-muted-foreground">
              During training, randomly replace the text embedding with a
              null/empty embedding some fraction of the time (typically
              10&ndash;20%). When the text is dropped, the model must predict
              noise <em>unconditionally</em>&mdash;like the model from{' '}
              <strong>Build a Diffusion Model</strong>. When the text is
              present, the model predicts noise <em>conditionally</em>.
            </p>
            <p className="text-muted-foreground">
              <strong>One model learns both tasks.</strong> No separate
              networks. The same weights handle both conditional and
              unconditional denoising, selected by whether the text embedding
              is present or replaced with the null embedding.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="One Model, Not Two">
            CFG does not use two separate models. The word
            &ldquo;classifier-free&rdquo; and the &ldquo;two forward
            passes&rdquo; at inference might suggest two networks. It is one
            model that was trained with randomly dropped text conditioning.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* 7c: The CFG inference formula */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-sm font-medium text-foreground">
              The inference formula
            </p>
            <p className="text-muted-foreground">
              At inference, run the model <strong>twice</strong> for each
              denoising step:
            </p>
            <ol className="list-decimal list-inside text-muted-foreground space-y-2 ml-4">
              <li>
                <InlineMath math="\epsilon_{\text{uncond}}" /> ={' '}
                model(x<sub>t</sub>, t, ∅)&mdash;&ldquo;what would you predict
                without any text?&rdquo;
              </li>
              <li>
                <InlineMath math="\epsilon_{\text{cond}}" /> ={' '}
                model(x<sub>t</sub>, t, text_emb)&mdash;&ldquo;what would you
                predict with the text?&rdquo;
              </li>
            </ol>
            <p className="text-muted-foreground">
              Combine them:
            </p>
            <div className="py-4 px-6 bg-muted/50 rounded-lg">
              <BlockMath math="\epsilon_{\text{cfg}} = \epsilon_{\text{uncond}} + w \cdot (\epsilon_{\text{cond}} - \epsilon_{\text{uncond}})" />
            </div>
            <p className="text-muted-foreground">
              The difference{' '}
              <InlineMath math="(\epsilon_{\text{cond}} - \epsilon_{\text{uncond}})" />{' '}
              is the <strong>direction of the text&rsquo;s influence</strong> on
              the prediction. It captures &ldquo;how does the text change what
              the model predicts?&rdquo; The guidance scale{' '}
              <InlineMath math="w" /> amplifies this direction.
            </p>
            <ul className="list-disc list-inside text-muted-foreground space-y-1 ml-4">
              <li>
                <InlineMath math="w = 1" />: the conditional prediction
                (no amplification)
              </li>
              <li>
                <InlineMath math="w = 0" />: the unconditional prediction
                (text ignored entirely)
              </li>
              <li>
                <InlineMath math="w = 7.5" />: amplify the text&rsquo;s effect
                by 7.5×&mdash;the typical Stable Diffusion default
              </li>
            </ul>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="Rearranged Form">
            <div className="space-y-2 text-sm">
              <p>The formula is equivalent to:</p>
              <p className="font-mono text-xs">
                ε<sub>cfg</sub> = (1−w)·ε<sub>uncond</sub> + w·ε<sub>cond</sub>
              </p>
              <p>
                At w {'>'} 1, you <strong>extrapolate</strong> beyond the
                conditional prediction, away from the unconditional one.
              </p>
            </div>
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* 7c continued: Concrete example with numbers */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-sm font-medium text-foreground">
              Concrete example with numbers
            </p>
            <p className="text-muted-foreground">
              Consider 3 dimensions of the noise prediction at one spatial
              location:
            </p>
            <CodeBlock
              code={`# Noise predictions at one spatial location (3 dims for illustration)
ε_uncond = [0.30, -0.10,  0.50]   # without text
ε_cond   = [0.50, -0.30,  0.80]   # with text "a cat sitting in a sunset"

# The text's direction of influence:
difference = [0.20, -0.20,  0.30]  # ε_cond - ε_uncond

# CFG at w=7.5:
ε_cfg = ε_uncond + 7.5 * difference
ε_cfg = [0.30 + 1.50, -0.10 + (-1.50), 0.50 + 2.25]
ε_cfg = [1.80, -1.60, 2.75]       # much larger magnitude`}
              language="python"
              filename="cfg_example.py"
            />
            <p className="text-muted-foreground">
              The modest difference between conditional and unconditional
              predictions becomes a large signal after amplification. The text
              does not just nudge&mdash;it <em>shoves</em> the prediction in
              the direction the text implies.
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* 7d: Geometric intuition for CFG */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-sm font-medium text-foreground">
              Geometric intuition
            </p>
            <p className="text-muted-foreground">
              Picture two arrows from a common origin in noise prediction space.
              One arrow is the unconditional prediction (what the model
              generates without text). The other is the conditional prediction
              (what it generates with text). The vector between their tips is
              the &ldquo;text direction&rdquo;&mdash;the effect the text has on
              the prediction.
            </p>
            <p className="text-muted-foreground">
              CFG <em>extends</em> along this text direction by a factor
              of <InlineMath math="w" />. At <InlineMath math="w = 1" />, you
              land at the conditional prediction. At{' '}
              <InlineMath math="w = 7.5" />, you shoot 7.5× further along the
              same direction. You are extrapolating beyond what the model
              learned, amplifying whatever the text signal provides.
            </p>
            {/* SVG diagram: CFG vector geometry */}
            <div className="rounded-lg bg-muted/30 p-4">
              <svg viewBox="0 0 400 200" className="w-full max-w-[500px] mx-auto" aria-label="CFG vector diagram: unconditional prediction, conditional prediction, text direction, and amplified CFG prediction">
                {/* Origin point */}
                <circle cx="60" cy="140" r="4" fill="#94a3b8" />
                <text x="45" y="160" fontSize="10" fill="#94a3b8">origin</text>

                {/* Unconditional prediction arrow */}
                <line x1="60" y1="140" x2="160" y2="120" stroke="#f59e0b" strokeWidth="2" markerEnd="url(#arrowAmber)" />
                <text x="105" y="112" fontSize="9" fill="#f59e0b">ε_uncond</text>

                {/* Conditional prediction arrow */}
                <line x1="60" y1="140" x2="200" y2="95" stroke="#8b5cf6" strokeWidth="2" markerEnd="url(#arrowViolet)" />
                <text x="185" y="82" fontSize="9" fill="#8b5cf6">ε_cond</text>

                {/* Difference vector (text direction) */}
                <line x1="160" y1="120" x2="200" y2="95" stroke="#22c55e" strokeWidth="2" strokeDasharray="4,3" markerEnd="url(#arrowGreen)" />
                <text x="195" y="115" fontSize="8" fill="#22c55e">text direction</text>

                {/* CFG amplified arrow */}
                <line x1="60" y1="140" x2="340" y2="38" stroke="#ef4444" strokeWidth="2.5" markerEnd="url(#arrowRed)" />
                <text x="300" y="30" fontSize="9" fill="#ef4444" fontWeight="bold">ε_cfg (w=7.5)</text>

                {/* Dashed extension from cond to cfg */}
                <line x1="200" y1="95" x2="340" y2="38" stroke="#ef4444" strokeWidth="1" strokeDasharray="4,3" />

                {/* w=1 label at cond point */}
                <text x="210" y="98" fontSize="8" fill="#64748b">w=1</text>

                {/* Legend */}
                <line x1="20" y1="185" x2="40" y2="185" stroke="#f59e0b" strokeWidth="2" />
                <text x="44" y="188" fontSize="8" fill="#94a3b8">unconditional</text>
                <line x1="120" y1="185" x2="140" y2="185" stroke="#8b5cf6" strokeWidth="2" />
                <text x="144" y="188" fontSize="8" fill="#94a3b8">conditional</text>
                <line x1="215" y1="185" x2="235" y2="185" stroke="#22c55e" strokeWidth="2" strokeDasharray="4,3" />
                <text x="239" y="188" fontSize="8" fill="#94a3b8">text direction</text>
                <line x1="315" y1="185" x2="335" y2="185" stroke="#ef4444" strokeWidth="2.5" />
                <text x="339" y="188" fontSize="8" fill="#94a3b8">CFG output</text>

                {/* Arrow markers */}
                <defs>
                  <marker id="arrowAmber" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
                    <path d="M0,0 L6,3 L0,6 Z" fill="#f59e0b" />
                  </marker>
                  <marker id="arrowViolet" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
                    <path d="M0,0 L6,3 L0,6 Z" fill="#8b5cf6" />
                  </marker>
                  <marker id="arrowGreen" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
                    <path d="M0,0 L6,3 L0,6 Z" fill="#22c55e" />
                  </marker>
                  <marker id="arrowRed" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
                    <path d="M0,0 L6,3 L0,6 Z" fill="#ef4444" />
                  </marker>
                </defs>
              </svg>
              <p className="text-xs text-muted-foreground text-center mt-2">
                CFG extrapolates along the text direction. The conditional
                prediction (w=1) is the starting point. Higher guidance scales
                push further in the same direction.
              </p>
            </div>
            <p className="text-muted-foreground">
              Think of it as a &ldquo;contrast slider.&rdquo; CFG turns up the
              contrast between what the model predicts with text versus without.
              Medium contrast (w ≈ 7.5) produces vivid, text-faithful images.
              Too much contrast (w ≈ 20) oversaturates.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Directions in Space">
            In <strong>Exploring Latent Spaces</strong> (Module 6.1), you saw meaningful
            directions in the VAE&rsquo;s latent space. The CFG &ldquo;text
            direction&rdquo; is similar&mdash;a meaningful direction in noise
            prediction space. But it is computed fresh at every denoising
            step, not as a fixed global property.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 8: Elaborate — Guidance Scale Tradeoff + Historical Context
          ================================================================ */}

      {/* 8a: The guidance scale tradeoff */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Guidance Scale Tradeoff"
            subtitle="Not all amplification is improvement"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Higher guidance scale means stronger text adherence. But there is
              a cost:
            </p>
            <div className="grid gap-4 md:grid-cols-3">
              <GradientCard title="w = 1.0 (No Guidance)" color="emerald">
                <ul className="space-y-1">
                  <li>&bull; Model uses text but does not amplify</li>
                  <li>&bull; Diverse, creative outputs</li>
                  <li>&bull; May loosely follow the prompt</li>
                  <li>&bull; Low text fidelity</li>
                </ul>
              </GradientCard>
              <GradientCard title="w = 7.5 (Typical)" color="blue">
                <ul className="space-y-1">
                  <li>&bull; Strong text adherence</li>
                  <li>&bull; Good image quality and coherence</li>
                  <li>&bull; The default in most Stable Diffusion UIs</li>
                  <li>&bull; The sweet spot for most prompts</li>
                </ul>
              </GradientCard>
              <GradientCard title="w = 20 (Extreme)" color="rose">
                <ul className="space-y-1">
                  <li>&bull; Over-committed to text</li>
                  <li>&bull; Oversaturated colors</li>
                  <li>&bull; Distorted anatomy, artifact-heavy</li>
                  <li>&bull; Technically matches the prompt but looks unnatural</li>
                </ul>
              </GradientCard>
            </div>
            <p className="text-muted-foreground">
              The guidance scale is a <strong>user-facing parameter</strong>.
              When you use Stable Diffusion and adjust &ldquo;CFG scale,&rdquo;
              this is the <InlineMath math="w" /> in the formula. Like cranking
              the volume on one instrument until it drowns out the rest of the
              orchestra and distorts&mdash;some amplification helps, too much
              destroys the balance.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Higher ≠ Better">
            It is tempting to think that stronger text signal always means
            better images. It does not. At extreme scales, the model
            over-optimizes for the text at the expense of image coherence.
            The tradeoff is real.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* 8b: Why "classifier-free"? Historical context */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-sm font-medium text-foreground">
              Why &ldquo;classifier-free&rdquo;?
            </p>
            <p className="text-muted-foreground">
              Before CFG, there was <strong>classifier guidance</strong>{' '}
              (Dhariwal &amp; Nichol, 2021): train a separate image classifier
              on noisy images, use its gradients to steer the diffusion process
              toward a class label. It worked, but required a separately
              trained classifier, only supported class labels (not free-form
              text), and was complex.
            </p>
            <p className="text-muted-foreground">
              Classifier-free guidance (Ho &amp; Salimans, 2022) eliminated the
              classifier entirely. Train one model with random text dropout. No
              external classifier needed. Works with any text prompt, not just
              class labels. &ldquo;Classifier-free guidance&rdquo; is the
              standard approach in Stable Diffusion and virtually all modern
              text-to-image diffusion models.
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* 8c: CFG doubles compute */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-sm font-medium text-foreground">
              The cost: doubled inference compute
            </p>
            <p className="text-muted-foreground">
              CFG requires <strong>two forward passes</strong> per denoising
              step (one conditional, one unconditional). This doubles the
              inference compute. At 50 denoising steps, that is 100 forward
              passes through the U-Net. In practice, the two passes can be
              batched together (batch of 2) for efficiency, but the fundamental
              cost is 2×.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Batched for Efficiency">
            Most implementations run both predictions as a single batch of 2,
            so the wall-clock overhead is less than 2×. But the memory cost
            is real.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 9: Check — Predict and Verify
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Check Your Understanding"
            subtitle="Predict, then verify"
          />
          <GradientCard title="Predict and Verify" color="cyan">
            <div className="space-y-3 text-sm">
              <p>
                <strong>Question 1:</strong> In cross-attention, the query comes
                from the U-Net&rsquo;s spatial features and the keys/values come
                from the text embeddings. If you <em>swapped</em> this&mdash;queries
                from text, keys/values from spatial features&mdash;what would
                happen?
              </p>
              <details className="mt-2">
                <summary className="font-medium cursor-pointer text-primary">
                  Reveal
                </summary>
                <div className="mt-2">
                  <p>
                    Each text token would compute a weighted average of spatial
                    features. The output would be a sequence of text-length
                    vectors, each summarizing what the image looks like at
                    locations relevant to that word. This is useful for
                    image understanding (similar to what CLIP does internally),
                    but not for generation&mdash;the U-Net needs spatial features
                    enriched by text, not text features enriched by spatial
                    information.
                  </p>
                </div>
              </details>

              <p className="mt-4">
                <strong>Question 2:</strong> Your colleague says: &ldquo;With
                CFG at w=1, you get the conditional prediction. With w=0, you get
                unconditional. So w=0.5 gives you a blend that is less
                text-dependent.&rdquo; Is this correct?
              </p>
              <details className="mt-2">
                <summary className="font-medium cursor-pointer text-primary">
                  Reveal
                </summary>
                <div className="mt-2">
                  <p>
                    Yes, technically. <InlineMath math="\epsilon_{\text{cfg}} = \epsilon_{\text{uncond}} + 0.5 \cdot (\epsilon_{\text{cond}} - \epsilon_{\text{uncond}}) = 0.5 \cdot \epsilon_{\text{uncond}} + 0.5 \cdot \epsilon_{\text{cond}}" />.
                    This is a 50/50 blend. w {'<'} 1 produces images that
                    are <em>less</em> text-faithful than the base conditional
                    model. In practice, w {'<'} 1 is rarely used&mdash;the whole
                    point of CFG is to amplify the text signal beyond w=1.
                  </p>
                </div>
              </details>

              <p className="mt-4">
                <strong>Question 3:</strong> Why can&rsquo;t you achieve the
                same effect as CFG by just training the model longer or with a
                stronger loss weight on text-conditioned examples?
              </p>
              <details className="mt-2">
                <summary className="font-medium cursor-pointer text-primary">
                  Reveal
                </summary>
                <div className="mt-2">
                  <p>
                    CFG works at <em>inference</em> time by extrapolating in a
                    specific direction. Training longer makes the conditional
                    prediction slightly better but does not extrapolate beyond
                    it. CFG actively amplifies the difference between conditional
                    and unconditional predictions, which training alone cannot do.
                    The model&rsquo;s trained conditional prediction is the
                    starting point; CFG pushes further.
                  </p>
                </div>
              </details>
            </div>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 10: Practice — Notebook Exercises
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Practice: Hands-On Exercises"
            subtitle="Cross-attention and CFG in a notebook"
          />
          <div className="space-y-4">
            <div className="rounded-lg border-2 border-primary/50 bg-primary/5 p-6">
              <div className="space-y-4">
                <p className="text-muted-foreground">
                  Modify self-attention into cross-attention, visualize
                  attention weights, implement the CFG formula, and experiment
                  with guidance scales on a real model.
                </p>
                <a
                  href="https://colab.research.google.com/github/brennancheung/CourseAI/blob/main/notebooks/6-3-4-text-conditioning-and-guidance.ipynb"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-2 px-4 py-2 rounded-md bg-primary text-primary-foreground font-medium text-sm hover:bg-primary/90 transition-colors"
                >
                  Open in Google Colab
                </a>
              </div>
            </div>
            <p className="text-muted-foreground">
              The notebook covers four exercises:
            </p>
            <div className="space-y-3">
              <div className="rounded-lg bg-muted/30 p-4 space-y-2">
                <p className="text-sm font-medium text-foreground">
                  Exercise 1: Cross-Attention from Self-Attention (Guided)
                </p>
                <p className="text-sm text-muted-foreground">
                  Start with a working self-attention implementation. Modify it
                  to perform cross-attention by changing where K and V come
                  from. Predict-before-run: &ldquo;After this change, what
                  shape will the attention weight matrix be?&rdquo; (No longer
                  square&mdash;(H·W) × T instead of (H·W) × (H·W).)
                </p>
              </div>
              <div className="rounded-lg bg-muted/30 p-4 space-y-2">
                <p className="text-sm font-medium text-foreground">
                  Exercise 2: Visualize Cross-Attention Weights (Guided)
                </p>
                <p className="text-sm text-muted-foreground">
                  Compute attention weights for a 4×4 spatial feature map
                  attending to a 6-token text. Visualize as a heatmap: rows
                  are spatial locations (16), columns are text tokens (6).
                  Observe that different spatial locations attend to different
                  tokens.
                </p>
              </div>
              <div className="rounded-lg bg-muted/30 p-4 space-y-2">
                <p className="text-sm font-medium text-foreground">
                  Exercise 3: Implement CFG (Supported)
                </p>
                <p className="text-sm text-muted-foreground">
                  Given a dummy model function, implement the CFG formula. Test
                  at w=0, w=1, w=3, w=7.5, w=15. Plot the L2 norm of the CFG
                  noise prediction as a function of w. Observe that higher w
                  produces larger-magnitude predictions.
                </p>
              </div>
              <div className="rounded-lg bg-muted/30 p-4 space-y-2">
                <p className="text-sm font-medium text-foreground">
                  Exercise 4: CFG with a Real Diffusion Model (Independent)
                </p>
                <p className="text-sm text-muted-foreground">
                  Load a small pretrained text-conditioned diffusion model.
                  Generate images from the same prompt at w=1, 3, 7.5, 12, 20.
                  Observe the quality/fidelity tradeoff. Identify the sweet
                  spot.
                </p>
              </div>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="What to Focus On">
            Exercises 1&ndash;2 verify cross-attention understanding.
            Exercise 3 makes CFG concrete. Exercise 4 is the fun
            one&mdash;seeing the guidance scale tradeoff with real images.
            If time is short, prioritize Exercises 1 and 3.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 11: Summary
          ================================================================ */}
      <Row>
        <Row.Content>
          <SummaryBlock
            title="Key Takeaways"
            items={[
              {
                headline:
                  'Cross-attention injects text into the U-Net\u2014same formula as self-attention, different source for K and V.',
                description:
                  'Each spatial location generates a query; the text tokens provide keys and values. Different locations attend to different words, creating spatially-varying text conditioning. The formula is output = softmax(QK\u1D40/\u221Ad_k) V\u2014identical to Module 4.2.',
              },
              {
                headline:
                  'Classifier-free guidance amplifies the text signal at inference time.',
                description:
                  'Train with randomly dropped text. At inference, run two forward passes (with and without text) and scale the difference: \u03B5_cfg = \u03B5_uncond + w \u00B7 (\u03B5_cond \u2212 \u03B5_uncond). The guidance scale w controls the fidelity/quality tradeoff.',
              },
              {
                headline:
                  'Timestep is global, text is spatially varying\u2014both coexist in every forward pass.',
                description:
                  'Adaptive group normalization handles the timestep (same signal everywhere). Cross-attention handles the text (each location gets its own signal). WHEN vs WHAT, working together.',
              },
            ]}
          />
        </Row.Content>
      </Row>

      {/* Mental model echo */}
      <Row>
        <Row.Content>
          <InsightBlock title="Mental Model">
            <strong>
              The timestep tells the network WHEN (noise level, via adaptive
              norm, global). The text tells it WHAT (semantic content, via
              cross-attention, spatially varying). CFG turns up the volume on
              the WHAT.
            </strong>
          </InsightBlock>
        </Row.Content>
      </Row>

      {/* Connection to next lesson */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The architecture is now complete: U-Net with skip connections,
              timestep conditioning via adaptive normalization, text
              conditioning via cross-attention, and classifier-free guidance to
              amplify the text signal. But there is one remaining problem you
              felt in <strong>Build a Diffusion Model</strong>: pixel-space
              diffusion is painfully slow.
            </p>
            <p className="text-muted-foreground">
              The next lesson addresses this by moving the diffusion process
              from pixel space to latent space. A pretrained VAE compresses
              images before the diffusion process, and the U-Net denoises in
              the smaller latent space. Same algorithm, smaller tensors, much
              faster. After that, you have Stable Diffusion.
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* References */}
      <Row>
        <Row.Content>
          <ReferencesBlock
            references={[
              {
                title: 'High-Resolution Image Synthesis with Latent Diffusion Models',
                authors: 'Rombach et al., 2022',
                url: 'https://arxiv.org/abs/2112.10752',
                note: 'The Stable Diffusion paper. Section 3.3 describes cross-attention conditioning. Figure 3 shows the architecture.',
              },
              {
                title: 'Classifier-Free Diffusion Guidance',
                authors: 'Ho & Salimans, 2022',
                url: 'https://arxiv.org/abs/2207.12598',
                note: 'The CFG paper. Short and readable. Section 2 covers the training trick and inference formula.',
              },
              {
                title: 'Diffusion Models Beat GANs on Image Synthesis',
                authors: 'Dhariwal & Nichol, 2021',
                url: 'https://arxiv.org/abs/2105.05233',
                note: 'Introduces classifier guidance (the predecessor to CFG). Section 4 covers the guidance approach that CFG replaced.',
              },
            ]}
          />
        </Row.Content>
      </Row>

      {/* Next Step */}
      <Row>
        <Row.Content>
          <NextStepBlock
            href="/app"
            title="Up Next: From Pixels to Latents"
            description="Move the diffusion process from pixel space to latent space&mdash;the final piece that makes Stable Diffusion fast enough to use."
          />
        </Row.Content>
      </Row>
    </LessonLayout>
  )
}

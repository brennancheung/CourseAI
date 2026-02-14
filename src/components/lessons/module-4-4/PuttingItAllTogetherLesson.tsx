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

/**
 * Putting It All Together
 *
 * Fifth and final lesson in Module 4.4 (Beyond Pretraining).
 * CONSOLIDATE lesson -- no new concepts, no notebook, no exercises.
 *
 * This is the capstone of Series 4 (LLMs & Transformers). The student
 * has completed 17 lessons and has DEVELOPED or higher depth on every
 * concept referenced here. The goal is synthesis: see the complete
 * LLM pipeline as a coherent whole, trace what each stage adds,
 * and understand why no stage can be skipped.
 *
 * EXPLICITLY NOT COVERED:
 * - Any new concept, technique, or algorithm
 * - Implementation details (no code, no notebook)
 * - Comparing specific models (no Llama vs Mistral benchmarks)
 * - Production deployment, MLOps, or serving infrastructure
 * - Constitutional AI, reasoning models, multimodal (Series 5 previews only)
 *
 * Previous: LoRA, Quantization & Inference (Module 4.4, Lesson 4)
 * Next: Series 5 (Recent LLM Advances)
 */

// ---------------------------------------------------------------------------
// Inline SVG: Full Pipeline Diagram
// The capstone visual for the entire series. Shows every stage from
// raw text to deployed model, labeled with what each stage adds.
// ---------------------------------------------------------------------------

function FullPipelineDiagram() {
  const svgW = 520
  const svgH = 610

  const stageX = 60
  const stageW = 400
  const stageH = 44
  const gap = 22
  const arrowX = svgW / 2

  const stages = [
    { label: 'Tokenization (BPE)', adds: 'Text to integers', color: '#1e3a5f', border: '#38bdf8', module: '4.1' },
    { label: 'Embeddings + Position', adds: 'Integers to vectors', color: '#1e3a5f', border: '#38bdf8', module: '4.1' },
    { label: 'Transformer (N blocks)', adds: 'The architecture', color: '#312e81', border: '#818cf8', module: '4.2' },
    { label: 'Pretraining (next-token)', adds: 'Knowledge', color: '#4c1d95', border: '#a78bfa', module: '4.3' },
    { label: 'SFT (instruction data)', adds: 'Format', color: '#064e3b', border: '#6ee7b7', module: '4.4' },
    { label: 'RLHF / DPO (preferences)', adds: 'Judgment', color: '#7f1d1d', border: '#fca5a5', module: '4.4' },
    { label: 'LoRA (domain adaptation)', adds: 'Specialization', color: '#78350f', border: '#fbbf24', module: '4.4' },
    { label: 'Quantization (int4/int8)', adds: 'Accessibility', color: '#134e4a', border: '#5eead4', module: '4.4' },
  ]

  return (
    <div className="flex justify-center py-4 overflow-x-auto">
      <svg
        width={svgW}
        height={svgH}
        viewBox={`0 0 ${svgW} ${svgH}`}
        className="overflow-visible"
      >
        {/* Starting point — visually distinct, not a pipeline stage */}
        <g>
          <rect
            x={stageX + 80}
            y={4}
            width={stageW - 160}
            height={32}
            rx={16}
            fill="none"
            stroke="#64748b"
            strokeWidth={1}
            strokeDasharray="4 3"
            opacity={0.7}
          />
          <text
            x={arrowX}
            y={22}
            textAnchor="middle"
            dominantBaseline="middle"
            fill="#94a3b8"
            fontSize="10"
            fontStyle="italic"
          >
            Raw Text Corpus
          </text>
          {/* Arrow from starting point to first stage */}
          <line
            x1={arrowX}
            y1={36}
            x2={arrowX}
            y2={52}
            stroke="#475569"
            strokeWidth={1}
            markerEnd="url(#pipelineArrow)"
          />
        </g>

        {stages.map((stage, i) => {
          const y = 56 + i * (stageH + gap)

          return (
            <g key={stage.label}>
              {/* Stage box */}
              <rect
                x={stageX}
                y={y}
                width={stageW}
                height={stageH}
                rx={6}
                fill={stage.color}
                stroke={stage.border}
                strokeWidth={1}
                opacity={0.85}
              />

              {/* Stage label */}
              <text
                x={stageX + 14}
                y={y + stageH / 2 + 1}
                dominantBaseline="middle"
                fill={stage.border}
                fontSize="11"
                fontWeight="600"
              >
                {stage.label}
              </text>

              {/* "Adds" label on right */}
              <text
                x={stageX + stageW - 14}
                y={y + stageH / 2 + 1}
                dominantBaseline="middle"
                textAnchor="end"
                fill={stage.border}
                fontSize="9"
                fontStyle="italic"
                opacity={0.8}
              >
                adds: {stage.adds}
              </text>

              {/* Module label on far right */}
              <text
                x={stageX + stageW + 16}
                y={y + stageH / 2 + 1}
                dominantBaseline="middle"
                fill="#6b7280"
                fontSize="8"
              >
                {stage.module}
              </text>

              {/* Arrow to next stage */}
              {i < stages.length - 1 && (
                <line
                  x1={arrowX}
                  y1={y + stageH}
                  x2={arrowX}
                  y2={y + stageH + gap}
                  stroke="#475569"
                  strokeWidth={1}
                  markerEnd="url(#pipelineArrow)"
                />
              )}
            </g>
          )
        })}

        {/* Output label */}
        <text
          x={arrowX}
          y={56 + stages.length * (stageH + gap) - 6}
          textAnchor="middle"
          fill="#9ca3af"
          fontSize="10"
          fontWeight="500"
        >
          Deployed model on your hardware
        </text>

        <defs>
          <marker
            id="pipelineArrow"
            markerWidth="6"
            markerHeight="4"
            refX="5"
            refY="2"
            orient="auto"
          >
            <polygon points="0 0, 6 2, 0 4" fill="#475569" />
          </marker>
        </defs>
      </svg>
    </div>
  )
}

export function PuttingItAllTogetherLesson() {
  return (
    <LessonLayout>
      {/* ================================================================
          1. Header
          ================================================================ */}
      <Row>
        <Row.Content>
          <LessonHeader
            title="Putting It All Together"
            description="The complete LLM pipeline from raw text to aligned model on your laptop—no new concepts, just synthesis of everything you have learned."
            category="Synthesis"
          />
        </Row.Content>
      </Row>

      {/* ================================================================
          2. Context + Constraints (Section 1 from outline)
          ================================================================ */}
      <Row>
        <Row.Content>
          <ObjectiveBlock>
            This is the final lesson of Series 4. You have spent seventeen
            lessons building up the complete picture of how language models
            work&mdash;from next-token prediction through transformers,
            implementation, finetuning, alignment, and efficient serving.
            This lesson asks nothing new. It asks you to see what you
            already know as a single, coherent pipeline.
          </ObjectiveBlock>
        </Row.Content>
      </Row>

      <Row>
        <Row.Content>
          <ConstraintBlock
            title="What This Lesson Is (and Is Not)"
            items={[
              'A synthesis of everything from Modules 4.1 through 4.4',
              'No new concepts, no notebook, no exercises',
              'Tracing the complete pipeline from raw text to deployed model',
              'Connecting each stage to the lesson where you learned it',
              'A brief preview of Series 5 topics (not lessons)',
              'NOT: any new technique or algorithm',
              'NOT: model comparisons or benchmarks',
              'NOT: production deployment or MLOps',
            ]}
          />
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Victory Lap">
            This is a CONSOLIDATE lesson. After the intensity of LoRA,
            Quantization &amp; Inference, this is your chance to step
            back and see the full picture. No pressure, no exercises&mdash;just
            clarity.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          3. Hook — The Model on Your Laptop (Section 2 from outline)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Model on Your Laptop"
            subtitle="How did it get here?"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Imagine you download a quantized Llama 3 model. You run it
              locally. You type a question, and within seconds, you get a
              coherent, helpful, thoughtful answer. On your own hardware.
              No API key, no cloud.
            </p>

            <p className="text-muted-foreground">
              How did this model get here? What had to happen&mdash;what
              <strong> stages of work</strong>&mdash;to go from a blank
              neural network to this helpful assistant on your laptop?
            </p>

            <p className="text-muted-foreground">
              You already know every stage. You built a tokenizer from
              scratch. You constructed the transformer piece by piece. You
              trained GPT on Shakespeare and loaded real GPT-2 weights. You
              learned classification finetuning, instruction tuning, RLHF,
              LoRA, and quantization. Each lesson added a piece.
            </p>

            <GradientCard title="You Understand Every Stage" color="violet">
              <p className="text-sm">
                This lesson does not teach you anything new. It connects
                what you already know into a single pipeline. When you are
                done, you will be able to trace the full journey from raw
                text to the model on your laptop&mdash;and explain every
                step.
              </p>
            </GradientCard>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Lesson 1 to Lesson 18">
            You started with &ldquo;what is a language model?&rdquo; Now
            you can explain the complete pipeline from pretraining through
            alignment through efficient serving. That is the distance you
            have covered.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          4. The Complete Pipeline (Section 3 from outline)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Complete Pipeline"
            subtitle="From raw text to deployed model"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Here is the full pipeline. Every stage you have studied,
              assembled into one picture. Each stage adds something the
              previous stages cannot provide. Each depends on what came
              before.
            </p>

            <FullPipelineDiagram />

            <p className="text-muted-foreground">
              Start at the top. A raw text corpus&mdash;books, web pages,
              code, conversations&mdash;is the raw material.{' '}
              <strong>Tokenization</strong> (What is a Language Model?,
              Tokenization) converts this text into integer sequences using
              BPE. <strong>Embeddings and positional encoding</strong>{' '}
              (Embeddings &amp; Positional Encoding) turn those integers
              into vectors the model can compute with.
            </p>

            <p className="text-muted-foreground">
              The <strong>transformer architecture</strong> (The Problem
              Attention Solves through Causal Masking &amp; the Full GPT
              Architecture) processes
              these vectors through N blocks of attention and feed-forward
              layers. &ldquo;Assembly, not invention&rdquo;&mdash;each
              component you built had a specific job, and the architecture
              is their composition.
            </p>

            <p className="text-muted-foreground">
              <strong>Pretraining</strong> (Building nanoGPT, Pretraining
              on Real Text) trains this architecture on massive text using
              next-token prediction. The result is a <strong>base
              model</strong>&mdash;one that understands language, has a
              world model, but does not know how to be helpful. It
              completes text; it does not answer questions.
            </p>

            <p className="text-muted-foreground">
              Running alongside the entire pipeline is the{' '}
              <strong>engineering</strong> that makes training and inference
              practical. Mixed precision (bfloat16) for training speed, KV
              caching for efficient autoregressive generation, flash
              attention for memory&mdash;you covered these in Scaling &amp;
              Efficiency. They are not a separate pipeline stage; they are
              the infrastructure that makes every stage feasible at scale.
              The model on your laptop from the hook? KV caching is why it
              generates tokens quickly rather than recomputing everything
              from scratch each step.
            </p>

            <p className="text-muted-foreground">
              <strong>SFT</strong> (Instruction Tuning) teaches the base
              model to follow instructions. &ldquo;Format, not
              knowledge&rdquo;&mdash;the model already knows about the
              world; SFT teaches it how to use that knowledge in a
              conversational format.
            </p>

            <p className="text-muted-foreground">
              <strong>RLHF/DPO</strong> (RLHF &amp; Alignment) adds
              judgment. &ldquo;SFT gives the model a voice; alignment
              gives it judgment.&rdquo; The model learns to prefer helpful,
              harmless, honest responses over harmful or sycophantic ones.
            </p>

            <p className="text-muted-foreground">
              <strong>LoRA</strong> (LoRA, Quantization &amp; Inference)
              enables domain adaptation without full retraining. The
              &ldquo;highway and detour&rdquo;&mdash;freeze the pretrained
              weights, add tiny trainable matrices that learn the
              task-specific adjustment.
            </p>

            <p className="text-muted-foreground">
              <strong>Quantization</strong> (LoRA, Quantization &amp;
              Inference) compresses the model from float16 to int4/int8.
              The precision spectrum continues&mdash;from float32 to
              bfloat16 to int8 to int4. Each step trades precision for
              memory, and neural networks tolerate it.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Where You Learned Each Stage">
            <ul className="space-y-2 text-xs">
              <li>{'Tokenization: Tokenization'}</li>
              <li>{'Embeddings: Embeddings & Positional Encoding'}</li>
              <li>{'Transformer: The Problem Attention Solves through Causal Masking & the Full GPT Architecture'}</li>
              <li>{'Pretraining: Building nanoGPT, Pretraining on Real Text'}</li>
              <li>{'Engineering: Scaling & Efficiency'}</li>
              <li>{'SFT: Instruction Tuning'}</li>
              <li>{'RLHF/DPO: RLHF & Alignment'}</li>
              <li>{'LoRA & Quantization: LoRA, Quantization & Inference'}</li>
            </ul>
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          5. What Each Stage Adds (Section 4 from outline)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="What Each Stage Adds (and Cannot Provide)"
            subtitle="A chain of dependencies, not a menu of options"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Each stage depends on what the previous stage produced.
              The order is not arbitrary&mdash;it is a dependency chain.
              Here is what each stage adds, and what it{' '}
              <strong>cannot</strong> provide.
            </p>

            <GradientCard title="Pretraining" color="purple">
              <div className="space-y-2 text-sm">
                <p>
                  <strong>Adds:</strong> Knowledge, language understanding,
                  world model. The model learns the structure of language
                  from billions of tokens.
                </p>
                <p>
                  <strong>Cannot provide:</strong> Task-specific behavior,
                  instruction following, safety guardrails. A base model
                  completes text&mdash;it does not answer questions.
                </p>
              </div>
            </GradientCard>

            <GradientCard title="SFT (Instruction Tuning)" color="emerald">
              <div className="space-y-2 text-sm">
                <p>
                  <strong>Adds:</strong> Instruction-following format,
                  conversational behavior. The model learns to respond
                  to queries, not just continue text.
                </p>
                <p>
                  <strong>Cannot provide:</strong> Judgment about response
                  quality, safety guardrails. An SFT model follows
                  instructions&mdash;even harmful ones.
                </p>
                <p>
                  <strong>Depends on:</strong> Pretraining. Without the
                  knowledge base, instruction-response pairs are
                  meaningless&mdash;the model has nothing to draw on.
                </p>
              </div>
            </GradientCard>

            <GradientCard title="Alignment (RLHF/DPO)" color="rose">
              <div className="space-y-2 text-sm">
                <p>
                  <strong>Adds:</strong> Judgment, safety, quality
                  preferences. The model learns which responses are better
                  than others.
                </p>
                <p>
                  <strong>Cannot provide:</strong> New knowledge, new
                  capabilities. Alignment refines behavior; it does not
                  teach new facts.
                </p>
                <p>
                  <strong>Depends on:</strong> SFT. Without
                  instruction-following behavior, the model produces
                  document continuations&mdash;you cannot meaningfully
                  compare two completions for &ldquo;helpfulness.&rdquo;
                </p>
              </div>
            </GradientCard>

            <GradientCard title="LoRA (Domain Adaptation)" color="orange">
              <div className="space-y-2 text-sm">
                <p>
                  <strong>Adds:</strong> Domain specialization without full
                  retraining. Medical, legal, or code-specific behavior
                  from a small adapter.
                </p>
                <p>
                  <strong>Cannot provide:</strong> Fundamentally new
                  capabilities. LoRA refines within the model&rsquo;s
                  existing knowledge&mdash;the weight change is low-rank
                  because finetuning is a refinement, not a revolution.
                </p>
                <p>
                  <strong>Depends on:</strong> A pretrained (or
                  SFT/aligned) model to adapt.
                </p>
              </div>
            </GradientCard>

            <GradientCard title="Quantization" color="cyan">
              <div className="space-y-2 text-sm">
                <p>
                  <strong>Adds:</strong> Accessibility. The model runs on
                  real hardware&mdash;consumer GPUs, laptops.
                </p>
                <p>
                  <strong>Cannot provide:</strong> Better quality.
                  Quantization only compresses; it never improves the
                  model.
                </p>
                <p>
                  <strong>Depends on:</strong> A trained model to
                  compress. You quantize after training, not before.
                </p>
              </div>
            </GradientCard>

            <WarningBlock title="More Stages Does Not Always Mean Better">
              Each stage has tradeoffs. Heavy RLHF can make models
              over-cautious&mdash;refusing valid requests with &ldquo;I&rsquo;m
              sorry, I can&rsquo;t help with that.&rdquo; Aggressive
              quantization degrades quality. Each stage adds something
              but also risks something. The pipeline is a series of
              informed choices, not an inevitable escalation.
            </WarningBlock>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Dependency Chain">
            SFT without pretraining: gibberish following instruction
            format. RLHF without SFT: comparing document continuations
            with no concept of &ldquo;helpfulness.&rdquo; Quantization
            without training: compressing random noise. Each stage depends
            on the output of the previous stage.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          6. Check — Name the Missing Stage (Section 5 from outline)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Check: Name the Missing Stage"
            subtitle="What went wrong?"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Each scenario below describes a model that is missing a
              pipeline stage. Can you identify which one?
            </p>

            <GradientCard title="Scenario 1" color="amber">
              <div className="space-y-3 text-sm">
                <p>
                  You download a model and ask it &ldquo;What is the capital
                  of France?&rdquo; It responds: &ldquo;The capital of France
                  is a city that has been the subject of many historical
                  discussions. Throughout the centuries, France has been
                  known for its rich cultural heritage...&rdquo; continuing
                  endlessly in essay style.
                </p>
                <details className="mt-2">
                  <summary className="font-medium cursor-pointer text-primary">
                    What is missing?
                  </summary>
                  <p className="mt-2">
                    <strong>Missing SFT.</strong> This is a base model
                    completing text instead of answering. It has knowledge
                    (it knows about France) but no instruction-following
                    format. You saw this exact behavior in Instruction
                    Tuning&mdash;the &ldquo;capital of France&rdquo;
                    dual-prompt evidence.
                  </p>
                </details>
              </div>
            </GradientCard>

            <GradientCard title="Scenario 2" color="amber">
              <div className="space-y-3 text-sm">
                <p>
                  You ask a model for help debugging code. It provides a
                  confident, well-formatted answer that is completely wrong.
                  It never expresses uncertainty, even when the answer is
                  clearly incorrect. You ask it to help draft a phishing
                  email &ldquo;for a security test,&rdquo; and it happily
                  obliges with a detailed template.
                </p>
                <details className="mt-2">
                  <summary className="font-medium cursor-pointer text-primary">
                    What is missing?
                  </summary>
                  <p className="mt-2">
                    <strong>Missing alignment.</strong> This is an SFT model
                    that follows instructions well (detailed, formatted) but
                    has no quality signal&mdash;no mechanism to prefer
                    accurate responses over confident-but-wrong ones, and no
                    safety guardrails to refuse harmful requests. It is
                    helpfully harmful&mdash;exactly the failure mode you saw
                    in RLHF &amp; Alignment.
                  </p>
                </details>
              </div>
            </GradientCard>

            <GradientCard title="Scenario 3" color="amber">
              <div className="space-y-3 text-sm">
                <p>
                  You have a perfectly aligned, instruction-following model
                  that produces excellent responses. But it requires 80 GB
                  of GPU memory to run.
                </p>
                <details className="mt-2">
                  <summary className="font-medium cursor-pointer text-primary">
                    What is missing?
                  </summary>
                  <p className="mt-2">
                    <strong>Missing quantization.</strong> The model works
                    perfectly but is not practical to deploy. Quantizing to
                    int4 would bring it from ~80 GB to ~20 GB or
                    less&mdash;fitting on consumer hardware. The model is
                    good; it is just not accessible.
                  </p>
                </details>
              </div>
            </GradientCard>
          </div>
        </Row.Content>
      </Row>

      {/* ================================================================
          7. The Open-Source Ecosystem (Section 6 from outline)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Open-Source Ecosystem"
            subtitle="How the pipeline maps to real model artifacts"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The pipeline is not theoretical. It is exactly what happened
              to create the models people actually use. Here is how the
              pipeline maps to real model artifacts you can find on
              HuggingFace:
            </p>

            <div className="grid gap-4 md:grid-cols-2">
              <GradientCard title="Base Models" color="purple">
                <div className="space-y-2 text-sm">
                  <p>
                    Examples: Llama 3 base, Mistral base.
                  </p>
                  <p>
                    Output of pretraining. Used by researchers and
                    practitioners who want to apply their own SFT and
                    alignment. Not useful for direct chat&mdash;they
                    complete text, they do not answer questions.
                  </p>
                </div>
              </GradientCard>

              <GradientCard title="Instruct Models" color="emerald">
                <div className="space-y-2 text-sm">
                  <p>
                    Examples: Llama 3 Instruct, Mistral Instruct.
                  </p>
                  <p>
                    Output of SFT + alignment. Ready for general use. What
                    most people mean when they say &ldquo;an LLM.&rdquo;
                  </p>
                </div>
              </GradientCard>

              <GradientCard title="LoRA Adapters" color="orange">
                <div className="space-y-2 text-sm">
                  <p>
                    Community adapters on HuggingFace.
                  </p>
                  <p>
                    Small add-on weights (~10&ndash;50 MB) for domain
                    specialization. Applied on top of base or instruct
                    models. Shareable because they are tiny.
                  </p>
                </div>
              </GradientCard>

              <GradientCard title="Quantized Models" color="cyan">
                <div className="space-y-2 text-sm">
                  <p>
                    Examples: GGUF 4-bit variants, GPTQ models.
                  </p>
                  <p>
                    Compressed for local inference. Downloaded by people
                    running models on laptops or consumer GPUs.
                  </p>
                </div>
              </GradientCard>
            </div>

            <p className="text-muted-foreground">
              A concrete example: Meta releases <strong>Llama 3
              base</strong> (pretraining output). Meta also releases{' '}
              <strong>Llama 3 Instruct</strong> (SFT + alignment applied).
              The community creates <strong>LoRA adapters</strong> for
              specific domains&mdash;medical, legal, code. Other community
              members quantize these models into{' '}
              <strong>4-bit GGUF variants</strong> that run on consumer
              hardware. The pipeline you learned is the pipeline that
              produced every model you can download.
            </p>

            <ComparisonRow
              left={{
                title: 'Base Models Are Not Useless',
                color: 'blue',
                items: [
                  'Foundation of the open-source ecosystem',
                  'Starting point for domain-specific finetuning',
                  'Classification finetuning (frozen backbone + head) works on base models',
                  'Meta releases base models because the community needs them',
                  'The base model IS the knowledge',
                ],
              }}
              right={{
                title: 'You Do Not Start from Scratch',
                color: 'emerald',
                items: [
                  'Download a pretrained + instruction-tuned + aligned model',
                  'Optionally apply LoRA finetuning for your domain',
                  'Optionally quantize for deployment',
                  'Most practitioners enter at the adaptation stage',
                  'The pipeline explains what already happened to the model you download',
                ],
              }}
            />
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Reading a Model Card">
            When you read &ldquo;Llama 3 70B Instruct, 4-bit GPTQ&rdquo;
            on HuggingFace, you now know exactly what that means: 70
            billion parameter model (architecture), pretrained on text
            (knowledge), instruction-tuned (format), aligned (judgment),
            quantized to 4-bit (accessibility). Every word maps to a
            pipeline stage you understand.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          8. The Adaptation Spectrum (Section 7 from outline)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Adaptation Spectrum"
            subtitle="Every method answers the same question"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              In LoRA, Quantization &amp; Inference, you saw the spectrum:
              frozen backbone, KL penalty, LoRA. Now expand this to cover
              every adaptation method from the module. They are all
              different answers to the same question:{' '}
              <strong>how much should I change to get what I want?</strong>
            </p>

            <div className="space-y-3">
              <GradientCard title="No Adaptation" color="blue">
                <p className="text-sm">
                  Use the base model directly for text completion. No
                  changes at all. Good for embedding extraction,
                  perplexity scoring, research.
                </p>
              </GradientCard>

              <GradientCard title="Classification Head" color="blue">
                <p className="text-sm">
                  Freeze the backbone, add a tiny classification head
                  (~1,536 parameters). From Finetuning for
                  Classification. Minimal change, minimal risk.
                </p>
              </GradientCard>

              <GradientCard title="LoRA / QLoRA" color="orange">
                <div className="space-y-2 text-sm">
                  <p>
                    <strong>LoRA:</strong> Freeze all weights, add small
                    low-rank detours (~2% of each adapted matrix). Surgical
                    adaptation with implicit regularization.
                  </p>
                  <p>
                    <strong>QLoRA:</strong> Quantize the base model to 4-bit,
                    then apply LoRA adapters in full precision on top. This
                    is what makes finetuning a 7B model possible on a single
                    consumer GPU (~4 GB). From LoRA, Quantization &amp;
                    Inference.
                  </p>
                </div>
              </GradientCard>

              <GradientCard title="SFT" color="emerald">
                <p className="text-sm">
                  Same architecture, different data format. From
                  Instruction Tuning. Changes the model&rsquo;s behavior
                  substantially&mdash;from text completer to instruction
                  follower.
                </p>
              </GradientCard>

              <GradientCard title="RLHF / DPO" color="rose">
                <p className="text-sm">
                  Preference-based optimization with KL penalty. From
                  RLHF &amp; Alignment. Refines behavior using human
                  judgment as the training signal.
                </p>
              </GradientCard>

              <GradientCard title="Full Finetuning" color="purple">
                <p className="text-sm">
                  Update all weights. The most powerful but most expensive
                  and most risky (catastrophic forgetting). From
                  Finetuning for Classification, where you saw the
                  tradeoffs.
                </p>
              </GradientCard>
            </div>

            <p className="text-muted-foreground">
              The connecting principle: <strong>every adaptation method is
              a different answer to the same question</strong>&mdash;how
              much should I change, and at what cost? A classification
              head changes almost nothing. LoRA changes very little. SFT
              and RLHF change behavior substantially. Full finetuning
              changes everything.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Spectrum, Not Menu">
            These are not independent options to pick from randomly. They
            form a spectrum from &ldquo;change nothing&rdquo; to
            &ldquo;change everything,&rdquo; and the right choice depends
            on your task, your data, and your hardware.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          9. Check — Match the Method (Section 8 from outline)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Check: Match the Method"
            subtitle="Which adaptation approach fits?"
          />
          <div className="space-y-4">
            <GradientCard title="Scenario 1" color="emerald">
              <div className="space-y-3 text-sm">
                <p>
                  You want to classify customer support tickets into 5
                  categories using a pretrained LLM.
                </p>
                <details className="mt-2">
                  <summary className="font-medium cursor-pointer text-primary">
                    Which method?
                  </summary>
                  <p className="mt-2">
                    <strong>Classification finetuning.</strong> Freeze the
                    backbone, add a 5-class classification head on the
                    last token&rsquo;s hidden state. The pattern from
                    Finetuning for Classification: minimal parameters,
                    minimal risk of overfitting.
                  </p>
                </details>
              </div>
            </GradientCard>

            <GradientCard title="Scenario 2" color="emerald">
              <div className="space-y-3 text-sm">
                <p>
                  You want a general-purpose chatbot that is helpful,
                  harmless, and honest.
                </p>
                <details className="mt-2">
                  <summary className="font-medium cursor-pointer text-primary">
                    Which method?
                  </summary>
                  <p className="mt-2">
                    <strong>SFT + RLHF/DPO.</strong> The full pipeline.
                    Start from a pretrained base model, apply SFT for
                    instruction following, then alignment for judgment and
                    safety. This is exactly how Llama Instruct models are
                    built.
                  </p>
                </details>
              </div>
            </GradientCard>

            <GradientCard title="Scenario 3" color="emerald">
              <div className="space-y-3 text-sm">
                <p>
                  You want to adapt an existing instruct model to write in
                  your company&rsquo;s specific documentation style, using
                  a single consumer GPU.
                </p>
                <details className="mt-2">
                  <summary className="font-medium cursor-pointer text-primary">
                    Which method?
                  </summary>
                  <p className="mt-2">
                    <strong>LoRA or QLoRA.</strong> Efficient adaptation on
                    limited hardware. QLoRA quantizes the base model to
                    4-bit (~4 GB for a 7B model) and adds tiny trainable
                    adapters. Fits on a consumer GPU, captures the
                    style-specific adjustment without rewriting the
                    model&rsquo;s knowledge.
                  </p>
                </details>
              </div>
            </GradientCard>
          </div>
        </Row.Content>
      </Row>

      {/* ================================================================
          10. What You Have Built (Section 9 from outline)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="What You Have Built"
            subtitle="Eighteen lessons, four modules, one complete picture"
          />
          <div className="space-y-4">
            <GradientCard title="Module 4.1: Language Modeling Fundamentals" color="blue">
              <p className="text-sm">
                You learned that a language model predicts the next token,
                built a tokenizer from scratch, and understood how text
                becomes the tensor the model processes. The starting point
                of everything.
              </p>
            </GradientCard>

            <GradientCard title="Module 4.2: Attention & the Transformer" color="violet">
              <p className="text-sm">
                You built the transformer piece by piece&mdash;attention,
                Q/K/V projections, multi-head attention, the residual
                stream, the full architecture. Each mechanism arrived
                because the previous version was insufficient.
              </p>
            </GradientCard>

            <GradientCard title="Module 4.3: Building & Training GPT" color="purple">
              <p className="text-sm">
                You implemented GPT from scratch in PyTorch, trained it
                on Shakespeare, applied engineering optimizations (mixed
                precision, KV caching), and loaded real GPT-2 weights into
                your own code. Your implementation generated coherent
                English.
              </p>
            </GradientCard>

            <GradientCard title="Module 4.4: Beyond Pretraining" color="emerald">
              <p className="text-sm">
                You learned to adapt pretrained models for classification,
                instruction following, and alignment with human
                preferences. You made it all practical with LoRA and
                quantization&mdash;techniques that bring LLMs from
                &ldquo;requires a cluster&rdquo; to &ldquo;runs on your
                laptop.&rdquo;
              </p>
            </GradientCard>

            <div className="space-y-2 mt-4">
              <p className="text-sm font-medium text-primary">
                The mental models you carry forward:
              </p>
              <ul className="list-disc list-inside text-muted-foreground space-y-1 ml-4 text-sm">
                <li>
                  &ldquo;Attention is a weighted average where the input
                  determines the weights.&rdquo;
                </li>
                <li>
                  &ldquo;SFT teaches format, not knowledge.&rdquo;
                </li>
                <li>
                  &ldquo;SFT gives the model a voice; alignment gives it
                  judgment.&rdquo;
                </li>
                <li>
                  &ldquo;Finetuning is a refinement, not a revolution.&rdquo;
                </li>
                <li>
                  &ldquo;The precision spectrum continues.&rdquo;
                </li>
              </ul>
            </div>

            <p className="text-muted-foreground">
              You started Series 4 by learning that a language model
              predicts the next token. Now you can explain the entire
              pipeline from pretraining through alignment through efficient
              serving. That is not a small thing.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Assembly, Not Invention">
            Just as the GPT architecture is &ldquo;assembly, not
            invention&rdquo;&mdash;each component is a known technique,
            and the architecture is their composition&mdash;the full LLM
            pipeline is the same. Every stage is a known technique. The
            pipeline is their careful composition.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          11. What Comes Next (Section 10 from outline)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="What Comes Next"
            subtitle="A preview of Series 5"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Series 4 gave you the foundation. Series 5 is about what
              happened next&mdash;the innovations that turned these models
              from useful tools into the systems that are changing the
              world.
            </p>

            <div className="grid gap-4 md:grid-cols-3">
              <GradientCard title="Constitutional AI" color="blue">
                <p className="text-sm">
                  What if AI provides the preference signal instead of
                  humans? Extends RLHF from RLHF &amp; Alignment.
                </p>
              </GradientCard>

              <GradientCard title="Reasoning Models" color="violet">
                <p className="text-sm">
                  Chain-of-thought, test-time compute, thinking before
                  answering. Extends generation from What is a Language
                  Model?
                </p>
              </GradientCard>

              <GradientCard title="Multimodal Models" color="emerald">
                <p className="text-sm">
                  Vision + language in one transformer. Extends the
                  architecture from Attention &amp; the Transformer.
                </p>
              </GradientCard>
            </div>

            <p className="text-muted-foreground">
              These are previews, not lessons. The foundation you built
              in Series 4 is exactly what you need to understand these
              advances. Each one extends a concept you already know.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="No Obligation">
            Series 5 builds on everything here, but you do not need to
            rush into it. The understanding you have now is
            self-contained and complete. Take a moment to appreciate what
            you have built.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          12. Summary (Section 11 from outline)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SummaryBlock
            title="The Complete Pipeline"
            items={[
              {
                headline: 'The pipeline is sequential and dependent.',
                description:
                  'Each stage builds on the previous one: pretraining produces knowledge, SFT adds format, alignment adds judgment, LoRA adds specialization, quantization adds accessibility. No stage can be skipped or reordered.',
              },
              {
                headline: 'Base models, instruct models, and quantized models are different pipeline stages.',
                description:
                  'When you see "Llama 3 70B Instruct, 4-bit GPTQ" on HuggingFace, every word maps to a stage you understand: parameter count (architecture), pretrained (knowledge), instruction-tuned (format), quantized (accessibility).',
              },
              {
                headline: 'Every adaptation method answers the same question.',
                description:
                  'How much to change, and at what cost? Classification head (almost nothing), LoRA (very little), SFT/RLHF (behavior substantially), full finetuning (everything). The right choice depends on your task, data, and hardware.',
              },
              {
                headline: 'The open-source ecosystem means you start from existing models.',
                description:
                  'You almost never start from pretraining. You download a pretrained, instruction-tuned, aligned model and adapt from there. The pipeline explains what already happened to create the model you download.',
              },
              {
                headline: 'From "what is a language model?" to "here is the complete pipeline" in 18 lessons.',
                description:
                  'A modern LLM is the result of pretraining (knowledge), instruction tuning (format), alignment (judgment), and engineering (accessibility)—and you understand every stage.',
              },
            ]}
          />
        </Row.Content>
      </Row>

      {/* ================================================================
          13. Module + Series Complete
          ================================================================ */}
      <Row>
        <Row.Content>
          <ModuleCompleteBlock
            module="4.4"
            title="Beyond Pretraining"
            achievements={[
              'Classification finetuning: frozen backbone + task head on last-token hidden state',
              'SFT: instruction-response pairs teach format, not knowledge',
              'RLHF/DPO: human preferences become the training signal for judgment and safety',
              'LoRA: low-rank trainable detours alongside frozen weights (~2% of parameters)',
              'Quantization: float to int4/int8 with minimal quality loss',
              'QLoRA: quantized base + LoRA adapters for 7B models on consumer GPUs',
              'The complete pipeline from raw text to deployed model',
            ]}
            nextModule="5.1"
            nextTitle="Recent LLM Advances"
          />
        </Row.Content>
      </Row>

      <Row>
        <Row.Content>
          <GradientCard title="Series 4 Complete: LLMs & Transformers" color="emerald">
            <p className="text-sm">
              Eighteen lessons. Four modules. One complete picture&mdash;from
              &ldquo;what is a language model?&rdquo; to the full pipeline
              on your laptop.
            </p>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* ================================================================
          14. Next Step
          ================================================================ */}
      <Row>
        <Row.Content>
          <NextStepBlock
            href="/app"
            title="Series 4 Complete"
            description="You can now trace the full LLM pipeline from raw text to deployed model and explain what each stage adds. When you read about a new model release, every word in the description maps to a concept you understand. Series 5 awaits when you are ready."
          />
        </Row.Content>
      </Row>
    </LessonLayout>
  )
}

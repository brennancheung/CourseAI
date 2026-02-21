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
  NextStepBlock,
  ReferencesBlock,
} from '@/components/lessons'
import { CodeBlock } from '@/components/common/CodeBlock'

/**
 * Prompt Engineering
 *
 * Second lesson in Module 5.2 (Reasoning & In-Context Learning).
 * BUILD lesson — lower cognitive load, practical application of ICL insights.
 * No custom interactive widget needed.
 *
 * Cognitive load: 2-3 new concepts:
 *   1. Systematic prompt construction as structured engineering (the organizing concept)
 *   2. RAG as retrieval-augmented context for attention
 *   3. Individual techniques (role, format, examples) are facets of #1, not new paradigms
 *
 * Core concepts at DEVELOPED:
 * - Prompt engineering as structured programming
 * - Format specification and output constraints
 * - Role/system prompting
 * - Few-shot example selection principles
 *
 * Concepts at INTRODUCED:
 * - RAG as retrieval-augmented context
 *
 * EXPLICITLY NOT COVERED:
 * - Chain-of-thought prompting or step-by-step reasoning (Lesson 3)
 * - Reasoning models or test-time compute (Lesson 4)
 * - Building a RAG pipeline (vector databases, embedding models)
 * - Prompt optimization or automated prompt search (DSPy, etc.)
 * - Specific model-dependent prompt formatting (ChatML, special tokens)
 * - Agentic patterns or tool use
 *
 * Previous: In-Context Learning (Module 5.2, Lesson 1)
 * Next: Chain-of-Thought (Module 5.2, Lesson 3)
 */

// ---------------------------------------------------------------------------
// Inline SVG: Prompt Anatomy Diagram
// Shows the composable sections of a well-designed prompt, color-coded
// with annotations connecting each section to its function in attention.
// ---------------------------------------------------------------------------

function PromptAnatomyDiagram() {
  const svgW = 600
  const svgH = 420

  const sections = [
    {
      label: 'System / Role',
      color: '#a78bfa',
      y: 55,
      example: '"You are a senior data engineer..."',
      annotation: 'Biases attention toward domain-relevant features',
      analogy: '= import statement',
    },
    {
      label: 'Task Instruction',
      color: '#60a5fa',
      y: 115,
      example: '"Extract all entities from the following text"',
      annotation: 'Defines the task for this forward pass',
      analogy: '= function name',
    },
    {
      label: 'Format Specification',
      color: '#34d399',
      y: 175,
      example: '"Return JSON: {name, date, amount}"',
      annotation: 'Narrows the output distribution',
      analogy: '= type signature',
    },
    {
      label: 'Few-Shot Examples',
      color: '#fbbf24',
      y: 235,
      example: '"Input: ... -> Output: {name: ...}"',
      annotation: 'Creates retrieval patterns in attention',
      analogy: '= unit tests',
    },
    {
      label: 'Context / Retrieved Docs',
      color: '#f97316',
      y: 295,
      example: '[Retrieved support article text...]',
      annotation: 'Adds K/V entries for attention to attend to',
      analogy: '= dependency injection',
    },
    {
      label: 'User Input',
      color: '#f472b6',
      y: 355,
      example: '"Invoice from Acme Corp, dated Jan 15..."',
      annotation: 'Q vectors attend to everything above',
      analogy: '= function argument',
    },
  ]

  const boxX = 20
  const boxW = 310
  const boxH = 44
  const annotX = 350

  return (
    <div className="flex justify-center py-4 overflow-x-auto">
      <svg
        width={svgW}
        height={svgH}
        viewBox={`0 0 ${svgW} ${svgH}`}
        className="overflow-visible"
      >
        {/* Title */}
        <text
          x={svgW / 2}
          y={20}
          textAnchor="middle"
          fill="#e2e8f0"
          fontSize="13"
          fontWeight="600"
        >
          Anatomy of a Structured Prompt
        </text>
        <text
          x={svgW / 2}
          y={36}
          textAnchor="middle"
          fill="#94a3b8"
          fontSize="9"
          fontStyle="italic"
        >
          Each section shapes the attention pattern. The software analogy is in parentheses.
        </text>

        {sections.map((sec, i) => (
          <g key={i}>
            {/* Colored box */}
            <rect
              x={boxX}
              y={sec.y}
              width={boxW}
              height={boxH}
              rx={6}
              fill={sec.color}
              opacity={0.12}
              stroke={sec.color}
              strokeWidth={1.5}
            />

            {/* Section label + analogy */}
            <text
              x={boxX + 10}
              y={sec.y + 16}
              fill={sec.color}
              fontSize="11"
              fontWeight="600"
            >
              {sec.label}
              <tspan fill="#94a3b8" fontSize="9" fontWeight="400">
                {' '}({sec.analogy})
              </tspan>
            </text>

            {/* Example text */}
            <text
              x={boxX + 10}
              y={sec.y + 34}
              fill="#94a3b8"
              fontSize="8"
              fontFamily="monospace"
            >
              {sec.example}
            </text>

            {/* Connecting line to annotation */}
            <line
              x1={boxX + boxW}
              y1={sec.y + boxH / 2}
              x2={annotX - 4}
              y2={sec.y + boxH / 2}
              stroke={sec.color}
              strokeWidth={1}
              opacity={0.4}
              strokeDasharray="3 2"
            />

            {/* Annotation */}
            <text
              x={annotX}
              y={sec.y + boxH / 2 + 4}
              fill={sec.color}
              fontSize="9"
              opacity={0.85}
            >
              {sec.annotation}
            </text>
          </g>
        ))}

        {/* Flow arrow along the left side */}
        <line
          x1={12}
          y1={60}
          x2={12}
          y2={390}
          stroke="#64748b"
          strokeWidth={1}
          opacity={0.4}
        />
        <polygon
          points="12,395 8,385 16,385"
          fill="#64748b"
          opacity={0.4}
        />
        <text
          x={12}
          y={410}
          textAnchor="middle"
          fill="#64748b"
          fontSize="8"
        >
          token order
        </text>
      </svg>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Main Lesson Component
// ---------------------------------------------------------------------------

export function PromptEngineeringLesson() {
  return (
    <LessonLayout>
      {/* ================================================================
          1. Header
          ================================================================ */}
      <Row>
        <Row.Content>
          <LessonHeader
            title="Prompt Engineering"
            description="If the prompt is a program, then prompt engineering is programming&mdash;with all the rigor, structure, and deliberate design that implies. Not finding magic phrases. Designing reliable systems."
            category="Reasoning & ICL"
          />
        </Row.Content>
      </Row>

      {/* ================================================================
          2. Context + Constraints (Section 1)
          ================================================================ */}
      <Row>
        <Row.Content>
          <ObjectiveBlock>
            This lesson teaches you to treat prompt construction as structured
            programming&mdash;selecting and combining specific techniques (format
            specification, role framing, example selection, context augmentation)
            to reliably control model behavior, grounded in the attention
            mechanism you already understand.
          </ObjectiveBlock>
        </Row.Content>
      </Row>

      <Row>
        <Row.Content>
          <ConstraintBlock
            title="Scope for This Lesson"
            items={[
              'Prompt engineering as a systematic discipline (not tricks or magic phrases)',
              'Format specification and output constraints',
              'Role and system prompts&mdash;what they do mechanistically',
              'Few-shot example selection principles',
              'RAG overview: retrieval as context augmentation, grounded in attention',
              'NOT: chain-of-thought prompting or step-by-step reasoning (next lesson)',
              'NOT: building a RAG pipeline (vector databases, embedding models)',
              'NOT: prompt optimization tools (DSPy, automated search)',
              'NOT: agentic patterns or tool use',
            ]}
          />
        </Row.Content>
        <Row.Aside>
          <TipBlock title="BUILD Lesson">
            In-Context Learning was a paradigm shift&mdash;surprising,
            conceptually demanding. This lesson is different: practical,
            systematic, lower activation energy. You already understand
            why prompting works. Now you learn how to control it.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          3. Recap (Section 2)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Where You Left Off"
            subtitle="The insight that makes this lesson possible"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              In In-Context Learning, you learned that the prompt is a
              program and attention is the interpreter. Examples in the prompt
              steer the model&rsquo;s behavior through attention
              patterns&mdash;Q vectors from the test input attend to K/V entries
              from the examples, and the model performs the task without any
              weight update. But you also learned that ICL is
              fragile: ordering matters, format matters, and the wrong examples
              can hurt more than help. That fragility is exactly why prompt
              engineering exists.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="From Insight to Discipline">
            You have the &ldquo;why&rdquo; (attention). This lesson teaches
            the &ldquo;how&rdquo;: which structural elements of a prompt
            shape attention patterns, and how to combine them deliberately.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          4. Hook: Before/After Contrast (Section 3)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Difference Structure Makes"
            subtitle="Same task, same model, different reliability"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Here is a real task: extract structured data from an invoice
              paragraph. Two developers write prompts for the same model.
            </p>

            <ComparisonRow
              left={{
                title: 'Prompt A (Conversational)',
                color: 'rose',
                items: [
                  '"Please extract the key information from this invoice and organize it nicely."',
                  'Run 1: bullet points',
                  'Run 2: a paragraph',
                  'Run 3: a markdown table',
                  'Different format every time',
                ],
              }}
              right={{
                title: 'Prompt B (Structured)',
                color: 'emerald',
                items: [
                  'System role + explicit JSON schema + one example + invoice text',
                  'Run 1: {"vendor": "Acme", "date": "2024-01-15", "amount": 2340.00}',
                  'Run 2: {"vendor": "Acme", "date": "2024-01-15", "amount": 2340.00}',
                  'Run 3: {"vendor": "Acme", "date": "2024-01-15", "amount": 2340.00}',
                  'Consistent, parseable output',
                ],
              }}
            />

            <p className="text-muted-foreground">
              Both prompts ask for the same thing. Prompt A is perfectly clear
              English. The difference is not politeness or
              specificity&mdash;it is <strong>structure</strong>. Prompt B
              constrains the output space. Prompt B is a program. Prompt A is
              a wish.
            </p>

            <GradientCard title="The Core Reframe" color="orange">
              <p className="text-sm">
                Prompt engineering is not about finding the right words. It is
                about designing the right structure. Each structural element
                shapes the attention pattern in a specific, predictable way.
              </p>
            </GradientCard>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Not About Phrasing">
            The popular framing of prompting is conversational&mdash;&ldquo;ask
            nicely,&rdquo; &ldquo;be specific,&rdquo; &ldquo;say please.&rdquo;
            That misses the point entirely. Phrasing is about natural language.
            Engineering is about controlling computation.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          5. Explain Part 1: The Prompt as a Composable Program (Section 4)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Prompt as a Composable Program"
            subtitle="Identifiable components, each with a specific function"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              A well-designed prompt has identifiable components, each serving a
              specific function&mdash;like a well-designed function where every
              parameter is there for a reason. Here is the anatomy:
            </p>

            <PromptAnatomyDiagram />

            <p className="text-muted-foreground">
              Every section you add to the prompt changes the K and V matrices.
              The model&rsquo;s query vectors for the user input attend
              to <strong>all</strong> of these sections. The structure of the
              prompt determines the structure of the attention pattern.
            </p>

            <p className="text-muted-foreground">
              The ordering is not arbitrary. Because of causal masking, each
              token can only attend to tokens <em>before</em> it. System and
              role tokens placed first are visible to every subsequent
              token&mdash;the task instruction, the examples, the user input,
              and all generated output tokens can attend to them. User input
              tokens placed last can attend to everything above. This is why
              prompt order matters: it is not convention, it is a consequence
              of the attention mechanism you learned in Decoder-Only
              Transformers.
            </p>

            <div className="py-4 px-6 bg-muted/50 rounded-lg">
              <p className="text-muted-foreground text-sm mb-3">
                As pseudocode, a structured prompt is a function call:
              </p>
              <CodeBlock
                code={`prompt(role, task, format, examples, context, input) -> output

# Each parameter is a design choice.
# Prompt engineering is choosing these parameters deliberately.`}
                language="python"
                filename="prompt-as-function.py"
              />
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Software Engineering Mapping">
            <ul className="space-y-2 text-sm">
              <li><strong>Format spec</strong> = type signature (constrains the return type)</li>
              <li><strong>Role prompt</strong> = import (brings context into scope)</li>
              <li><strong>Few-shot examples</strong> = unit tests (show the expected contract)</li>
              <li><strong>System prompt</strong> = config file (global behavior settings)</li>
              <li><strong>Context / RAG</strong> = dependency injection (data at runtime)</li>
            </ul>
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          6. Explain Part 2A: Format Specification (Section 5A)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Technique 1: Format Specification"
            subtitle="Constraining the output distribution"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The hook already showed format specification in action. Now
              deepen the understanding: <em>why</em> does specifying an output
              format work so reliably?
            </p>

            <p className="text-muted-foreground">
              Two mechanisms work together. First, format tokens in the
              prompt create <strong>structural anchors</strong> for attention.
              When the model sees tokens
              like <code className="text-sm bg-muted px-1 rounded">{'{"name": "...", "date": "..."}'}</code> in
              the prompt, its attention mechanism picks up these format tokens as
              a strong signal for what the output should look like.
            </p>

            <p className="text-muted-foreground">
              Second, <strong>autoregressive generation</strong> makes each token
              consistent with previous tokens. Once the model generates
              an opening <code className="text-sm bg-muted px-1 rounded">{'{'}</code>,
              the distribution over next tokens heavily favors JSON-valid
              continuations. That first curly brace constrains all subsequent
              tokens. Remember &ldquo;autoregressive generation is a feedback
              loop&rdquo;? Format specification exploits that loop&mdash;the
              first format token constrains the rest.
            </p>

            <CodeBlock
              code={`# Without format specification:
"Extract info from: Acme Corp invoice, Jan 15, $2340"
# Output varies: bullets, paragraphs, tables...

# With format specification:
"""
Extract info from the text below.
Return ONLY valid JSON matching this schema:
{"vendor": string, "date": string, "amount": number}

Text: Acme Corp invoice, Jan 15, $2340
"""
# Output: {"vendor": "Acme Corp", "date": "Jan 15", "amount": 2340}`}
              language="python"
              filename="format-specification.py"
            />
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Two Mechanisms">
            Format specification works because: (1) attention anchors on format
            tokens in the prompt, and (2) autoregressive generation makes each
            token consistent with the previous ones. The first curly brace
            constrains everything after it.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          7. Explain Part 2B: Role and System Prompts (Section 5B)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Technique 2: Role and System Prompts"
            subtitle="Biasing attention toward relevant features"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Here is a concrete demonstration. A developer asks for a code
              review with two different prompts:
            </p>

            <ComparisonRow
              left={{
                title: 'Generic Prompt',
                color: 'amber',
                items: [
                  '"Review this code and suggest improvements."',
                  'Response: style nits, naming suggestions, general cleanup',
                  'Finds: variable naming, missing docstring',
                  'Misses: SQL injection vulnerability',
                ],
              }}
              right={{
                title: 'Role-Framed Prompt',
                color: 'blue',
                items: [
                  '"You are a senior engineer who prioritizes security vulnerabilities. Review this code."',
                  'Response: security-focused analysis',
                  'Finds: SQL injection, unvalidated input, missing auth check',
                  'Different issues surfaced entirely',
                ],
              }}
            />

            <p className="text-muted-foreground">
              The role did not just change the <em>style</em> of the
              response&mdash;it changed <em>what the model attended to</em> in
              the code. The role text adds tokens to the context that bias
              attention. When the model processes the code, its Q vectors
              attend to both the code tokens and the role tokens.
              The word &ldquo;security&rdquo; in the role makes
              security-related code patterns more salient to the attention
              mechanism.
            </p>

            <GradientCard title="What Role Prompts Do NOT Do" color="rose">
              <div className="space-y-3 text-sm">
                <p>
                  &ldquo;You are a world-class expert in quantum computing&rdquo;
                  does not make the model know more about quantum computing.
                  Watch what happens with a factual question outside the
                  model&rsquo;s training data:
                </p>
                <ComparisonRow
                  left={{
                    title: 'No Role',
                    color: 'blue',
                    items: [
                      '"What were the exact revenue figures in Acme Corp\'s Q3 2024 earnings report?"',
                      'Model: "I don\'t have access to Acme Corp\'s specific financial data."',
                      'Honest about its limitation',
                    ],
                  }}
                  right={{
                    title: 'With "Expert Financial Analyst" Role',
                    color: 'rose',
                    items: [
                      'Same question, but system prompt says "You are an expert financial analyst"',
                      'Model: "Acme Corp reported $847M in Q3 revenue, reflecting 12% YoY growth..."',
                      'Confident, specific, and entirely fabricated',
                    ],
                  }}
                />
                <p className="text-muted-foreground">
                  The role made the model more confidently wrong, not more
                  correct. Remember &ldquo;SFT teaches format, not
                  knowledge&rdquo; from Instruction Tuning? Role prompts are
                  the same principle at inference time. They shape <em>how</em> the
                  model presents what it already knows, not <em>what</em> it knows.
                </p>
              </div>
            </GradientCard>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Confident ≠ Correct">
            A role prompt can make the model more confidently wrong, not more
            correct. Roles shape focus and style. They do not add knowledge.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          8. Explain Part 2C: Few-Shot Example Selection (Section 5C)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Technique 3: Few-Shot Example Selection"
            subtitle="Diversity over quantity, format over labels"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              You already know few-shot prompting works&mdash;you saw it in
              In-Context Learning. The question now is: <em>how do you choose
              examples systematically?</em> Three principles:
            </p>

            <div className="space-y-3">
              <GradientCard title="1. Diversity Over Quantity" color="blue">
                <p className="text-sm">
                  Examples should cover the space of expected inputs, not repeat
                  one pattern. For sentiment classification, include one clearly
                  positive, one clearly negative, and one nuanced/mixed example.
                  Three diverse examples outperform ten that all look the same.
                  Connect to the ordering sensitivity finding: if ordering
                  matters, then <em>which</em> examples you include matters even
                  more.
                </p>
              </GradientCard>

              <GradientCard title="2. Format Consistency" color="blue">
                <p className="text-sm">
                  All examples should follow the exact output format you want.
                  The label-flipping finding from In-Context Learning showed that
                  format matters more than labels for many tasks. Use this to your
                  advantage: consistent format is a stronger signal than many
                  examples. If your examples use different output formats, you are
                  sending the model mixed structural signals.
                </p>
              </GradientCard>

              <GradientCard title="3. Difficulty Calibration" color="blue">
                <p className="text-sm">
                  Examples should be representative of the real task difficulty.
                  Too-easy examples set the wrong baseline&mdash;the model may
                  generate oversimplified outputs. Too-hard examples may confuse.
                  Match the difficulty of what you will actually send.
                </p>
              </GradientCard>
            </div>

            <div className="py-4 px-6 bg-muted/50 rounded-lg">
              <p className="text-muted-foreground text-sm mb-2 font-medium">
                Concrete example&mdash;sentiment classification:
              </p>
              <ComparisonRow
                left={{
                  title: '3 Same-Polarity Examples',
                  color: 'amber',
                  items: [
                    '"Amazing product!" → Positive',
                    '"Absolutely loved it!" → Positive',
                    '"Best purchase ever!" → Positive',
                    'Test: "The battery is decent but the screen cracks easily"',
                    'Model: "Positive" (wrong — mixed sentiment, leans negative)',
                  ],
                }}
                right={{
                  title: '3 Diverse Examples',
                  color: 'emerald',
                  items: [
                    '"Amazing product!" → Positive',
                    '"Broke after one week" → Negative',
                    '"Good value but slow shipping" → Mixed',
                    'Test: "The battery is decent but the screen cracks easily"',
                    'Model: "Mixed" or "Negative" (correct — nuance captured)',
                  ],
                }}
              />
              <p className="text-muted-foreground text-sm mt-2">
                The diverse set includes a mixed example, so the model&rsquo;s
                attention has K/V patterns for nuanced inputs. The
                same-polarity set never showed the model what a negative or
                mixed case looks like&mdash;so it defaults to the only pattern
                it has seen.
              </p>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="The ICL Connection">
            Every principle here follows from the attention mechanism. Diverse
            examples create richer K/V patterns. Consistent format strengthens
            format anchoring. Calibrated difficulty sets appropriate attention
            weights. You can reason about example selection from the mechanism.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          9. Check 1: Predict-and-Verify (Section 6)
          ================================================================ */}
      <Row>
        <Row.Content>
          <GradientCard title="Checkpoint: Which Components Matter Most?" color="emerald">
            <div className="space-y-4 text-sm">
              <p>
                Task: summarize a technical document for a non-technical audience.
                Three prompt variants:
              </p>

              <div className="space-y-2">
                <p>
                  <strong>Variant A:</strong> Task instruction + document only.
                </p>
                <p>
                  <strong>Variant B:</strong> Task instruction + format
                  specification (&ldquo;3 bullet points, no jargon, each under
                  20 words&rdquo;) + document.
                </p>
                <p>
                  <strong>Variant C:</strong> Task instruction + role
                  (&ldquo;technical writer for a general audience&rdquo;) +
                  format specification + one example summary + document.
                </p>
              </div>

              <p>
                <strong>Rank them from least to most reliable output. Which
                component provides the biggest single improvement?</strong>
              </p>

              <details className="mt-3">
                <summary className="font-medium cursor-pointer text-primary">
                  Think about it, then reveal
                </summary>
                <div className="mt-3 space-y-3">
                  <p>
                    <strong>C &gt; B &gt; A.</strong> But the gap between A and B
                    is much larger than the gap between B and C.
                  </p>
                  <p className="text-muted-foreground">
                    Format specification provides the biggest single improvement
                    for this task. &ldquo;3 bullet points, no jargon&rdquo;
                    constrains the output distribution dramatically. The role and
                    the example add incremental value&mdash;the role biases the
                    tone, the example shows the expected length and style.
                  </p>
                  <p className="text-muted-foreground">
                    Not every prompt needs every component.{' '}
                    <strong>
                      Knowing which components to include for a given task is
                      the engineering judgment.
                    </strong>
                  </p>
                </div>
              </details>
            </div>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* ================================================================
          10. Elaborate: RAG and Context Augmentation (Section 7)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Beyond Parameters: Retrieval-Augmented Generation"
            subtitle="When the answer is not in the model's weights"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Every technique so far works with information already in the
              model&rsquo;s parameters&mdash;knowledge from pretraining, behavior
              from SFT. But what if the answer requires information the model
              does not have?
            </p>

            <p className="text-muted-foreground">
              The model&rsquo;s knowledge is frozen at its training cutoff. It
              cannot answer questions about recent events, private documents, or
              domain-specific data not in its training corpus. The
              solution: <strong>put the relevant information in the
              prompt.</strong>
            </p>

            <ComparisonRow
              left={{
                title: 'Closed-Book (No Context)',
                color: 'rose',
                items: [
                  '"What were the key decisions from our Q3 planning meeting?"',
                  'Model has no access to meeting notes',
                  'Output: confabulates plausible-sounding decisions',
                  'Or: "I don\'t have access to that information"',
                ],
              }}
              right={{
                title: 'Open-Book (RAG)',
                color: 'emerald',
                items: [
                  'Same question + retrieved meeting notes in the prompt',
                  'Model attends to the actual document',
                  'Output: accurate summary citing specific decisions',
                  'Q vectors from the question attend to K/V in the document',
                ],
              }}
            />

            <p className="text-muted-foreground">
              If the prompt is a program, retrieved documents are the data the
              program operates on. The model&rsquo;s attention attends to the
              retrieved text the same way it attends to few-shot
              examples&mdash;same mechanism, different content. RAG is not a model
              feature. It is a prompt engineering pattern: augment the prompt
              with retrieved context so that attention has relevant tokens to
              attend to.
            </p>

            <div className="py-4 px-6 bg-muted/50 rounded-lg">
              <p className="text-muted-foreground text-sm mb-2 font-medium">
                RAG as a two-step process:
              </p>
              <ol className="list-decimal list-inside text-muted-foreground space-y-1 text-sm">
                <li><strong>Retrieve</strong> relevant documents (search step&mdash;outside the model)</li>
                <li><strong>Place them in the prompt</strong> as context (attention step&mdash;inside the model)</li>
              </ol>
              <p className="text-muted-foreground text-sm mt-2">
                Software engineering analogy: dependency injection. Instead of
                hardcoding data into the model (pretraining), you provide it at
                runtime (in the prompt). The model&rsquo;s forward pass operates
                on whatever data you inject.
              </p>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Same Mechanism, Different Content">
            RAG works because of the same attention mechanism that enables
            in-context learning. Retrieved documents add K/V entries to
            the context. The model&rsquo;s Q vectors attend to them.
            No new mechanism&mdash;just longer, more useful context.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          11. Negative Example: Context Stuffing (Section 7 continued)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="When More Context Hurts"
            subtitle="Retrieval quality matters more than retrieval quantity"
          />
          <div className="space-y-4">
            <ComparisonRow
              left={{
                title: '5 Relevant Documents Only',
                color: 'emerald',
                items: [
                  'Tightly focused context',
                  'Attention concentrates on relevant tokens',
                  'Model answers accurately',
                  'Strong signal, low noise',
                ],
              }}
              right={{
                title: '5 Relevant + 15 Irrelevant',
                color: 'rose',
                items: [
                  'Padded with tangentially related text',
                  'Attention distributes across all tokens',
                  'Accuracy drops measurably',
                  'Relevant signal diluted by noise',
                ],
              }}
            />

            <p className="text-muted-foreground">
              You can reason about why this happens from the attention
              mechanism. Softmax distributes weight across <em>all</em> tokens.
              Irrelevant tokens &ldquo;steal&rdquo; attention weight from
              relevant ones. Remember from In-Context Learning that 50 examples
              in a 4096-token window caused problems? Same principle.
              The best RAG system retrieves few, highly relevant
              documents&mdash;not everything tangentially related.
            </p>

            <GradientCard title="RAG Does Not Solve Hallucination" color="amber">
              <div className="space-y-3 text-sm">
                <p>
                  Even when the retrieved document contains the correct answer,
                  the model can hallucinate. Here is what that looks like:
                </p>
                <ComparisonRow
                  left={{
                    title: 'Retrieved Document Says',
                    color: 'emerald',
                    items: [
                      '"Acme Corp reported Q3 revenue of $4.2M, a 3% decline from Q2."',
                    ],
                  }}
                  right={{
                    title: 'Model Outputs',
                    color: 'rose',
                    items: [
                      '"Acme Corp\'s Q3 revenue exceeded $5M, reflecting strong growth driven by new product launches."',
                    ],
                  }}
                />
                <p className="text-muted-foreground">
                  The document is right there in the context, but the model
                  blended the retrieved figure with parametric priors about what
                  earnings reports typically say. The result contradicts the
                  source. This happens because attention is not guaranteed to
                  attend to the retrieved passage more strongly than to
                  parametric knowledge from pretraining.
                </p>
                <p className="text-muted-foreground">
                  RAG <em>reduces</em> hallucination by making relevant
                  information available to attention. It does
                  not <em>eliminate</em> hallucination. The model still runs the
                  same forward pass.
                </p>
              </div>
            </GradientCard>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Context Stuffing">
            More context is not always better. Irrelevant documents dilute
            attention, reducing accuracy. Quality of retrieval matters more
            than quantity.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          12. No "One Best Prompt" (Section 7 continued)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="No Universal Template"
            subtitle="Principles transfer; specific prompts do not"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Now that you have seen multiple techniques, it is tempting to
              combine them all into one &ldquo;ultimate prompt template.&rdquo;
              Resist that temptation.
            </p>

            <p className="text-muted-foreground">
              Different models and different tasks benefit from different
              technique combinations. A prompt optimized for one model version
              may perform worse on the next. A format specification that helps
              for data extraction may be unnecessary for creative writing. The
              prompt interacts with the model&rsquo;s learned
              representations&mdash;which differ across models, versions, and
              fine-tuning runs.
            </p>

            <GradientCard title="Tools, Not Templates" color="violet">
              <p className="text-sm">
                The techniques are tools. Prompt engineering is knowing which
                tools to use for which job. The principles (structure,
                specificity, format consistency) transfer across models. Specific
                prompt templates do not. There is no universal
                prompt&mdash;there is understanding of what each component does
                and when it helps.
              </p>
            </GradientCard>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Engineering Judgment">
            The real skill is not memorizing prompt patterns. It is reasoning
            about which components will help for a given task, based on what
            you know about how attention processes the prompt.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          13. Check 2: Transfer Question (Section 8)
          ================================================================ */}
      <Row>
        <Row.Content>
          <GradientCard title="Checkpoint: Design a Prompt System" color="emerald">
            <div className="space-y-4 text-sm">
              <p>
                A developer is building a customer support chatbot. They have:
              </p>
              <ul className="list-disc list-inside space-y-1 ml-2">
                <li>A database of 10,000 support articles</li>
                <li>A requirement to respond in a specific JSON format (status, answer, sources, confidence)</li>
                <li>Users who ask questions about products the model was not trained on</li>
              </ul>

              <p>
                <strong>Design the prompt structure. Which components do you
                need? Why? What are the risks?</strong>
              </p>

              <details className="mt-3">
                <summary className="font-medium cursor-pointer text-primary">
                  Design your answer, then reveal
                </summary>
                <div className="mt-3 space-y-3">
                  <p><strong>Components needed:</strong></p>
                  <ul className="space-y-2 ml-2">
                    <li>
                      &bull; <strong>System/role prompt:</strong> Set customer
                      support behavior and tone. Biases attention toward
                      helpful, product-relevant responses.
                    </li>
                    <li>
                      &bull; <strong>Format specification:</strong> The JSON
                      schema with status, answer, sources, and confidence
                      fields. Ideally with one example showing the exact format.
                    </li>
                    <li>
                      &bull; <strong>RAG:</strong> Retrieve relevant support
                      articles based on the user&rsquo;s question. The model
                      was not trained on these products&mdash;the information
                      must come from the context.
                    </li>
                    <li>
                      &bull; <strong>User question:</strong> The actual customer
                      query as input.
                    </li>
                  </ul>

                  <p><strong>Risks:</strong></p>
                  <ul className="space-y-2 ml-2">
                    <li>
                      &bull; Retrieval may return irrelevant articles (the
                      context stuffing problem&mdash;noise dilutes signal).
                    </li>
                    <li>
                      &bull; The model may hallucinate beyond what the articles
                      say (RAG does not eliminate hallucination).
                    </li>
                    <li>
                      &bull; The JSON format may break for unusual questions
                      (format constraints are probabilistic, not guaranteed).
                    </li>
                  </ul>

                  <p className="text-muted-foreground">
                    This is a <em>composition</em> of the techniques you just
                    learned, not a new concept. Each component serves a specific
                    function in shaping the attention pattern.
                  </p>
                </div>
              </details>
            </div>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* ================================================================
          14. Practice: Notebook Exercises (Section 9)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Practice: Notebook Exercises"
            subtitle="Systematic prompt construction, hands-on"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Four exercises that build systematic prompt construction skills.
              Each exercise can be completed independently, but they share a
              theme: using the techniques from this lesson deliberately, not
              through trial and error.
            </p>

            <div className="space-y-3">
              <GradientCard title="Exercise 1: Format Specification (Guided)" color="blue">
                <p className="text-sm">
                  Given a paragraph, construct prompts that extract structured
                  data in three formats: bullet points, JSON, and markdown table.
                  Start with a conversational prompt, observe inconsistency, then
                  add format specification progressively. Predict the output
                  format before running each variant.
                </p>
              </GradientCard>
              <GradientCard title="Exercise 2: Role Prompting Effects (Supported)" color="blue">
                <p className="text-sm">
                  Take a code snippet with multiple issues (security, performance,
                  style). Write three prompts with different roles: security
                  auditor, performance engineer, code style reviewer. Compare
                  which issues each role surfaces. Then try a combined role and
                  see what happens.
                </p>
              </GradientCard>
              <GradientCard title="Exercise 3: Few-Shot Example Selection (Supported)" color="blue">
                <p className="text-sm">
                  Compare classification accuracy with: 3 random examples,
                  3 diverse examples, 3 same-category examples, and 5 random
                  examples. Run 5 trials each and plot accuracy. Discover that
                  diversity matters more than quantity.
                </p>
              </GradientCard>
              <GradientCard title="Exercise 4: Build a Structured Prompt (Independent)" color="blue">
                <p className="text-sm">
                  Design a complete structured prompt for a real task (e.g.,
                  generate meeting summaries from raw notes). Use at least 3
                  techniques from this lesson. Test on 3 different inputs and
                  evaluate consistency.
                </p>
              </GradientCard>
            </div>

            <div className="rounded-lg border-2 border-primary/50 bg-primary/5 p-6">
              <div className="space-y-4">
                <p className="text-muted-foreground">
                  Open the notebook to work through the exercises hands-on.
                  Each exercise builds a different prompt engineering skill
                  with immediate, visible feedback.
                </p>
                <a
                  href="https://colab.research.google.com/github/brennancheung/CourseAI/blob/main/notebooks/5-2-2-prompt-engineering.ipynb"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-2 px-4 py-2 rounded-md bg-primary text-primary-foreground font-medium text-sm hover:bg-primary/90 transition-colors"
                >
                  Open in Google Colab
                  <svg
                    width="16"
                    height="16"
                    viewBox="0 0 16 16"
                    fill="none"
                    xmlns="http://www.w3.org/2000/svg"
                  >
                    <path
                      d="M6 3H3a1 1 0 0 0-1 1v9a1 1 0 0 0 1 1h9a1 1 0 0 0 1-1v-3M9 2h5v5M8 8l5-5"
                      stroke="currentColor"
                      strokeWidth="1.5"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                  </svg>
                </a>
              </div>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Exercise Progression">
            Exercise 1 establishes format control. Exercise 2 explores
            attention bias. Exercise 3 tests example selection empirically.
            Exercise 4 composes everything into a real system.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          15. Summary (Section 10)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SummaryBlock
            title="What You Learned"
            items={[
              {
                headline:
                  'Prompt engineering is programming, not conversation.',
                description:
                  'The prompt is a structured program with identifiable components, each shaping the attention pattern in a specific way. Structure produces reliability. Phrasing does not.',
              },
              {
                headline:
                  'Format specification constrains the output distribution.',
                description:
                  'Explicit output schemas (JSON, markdown, structured text) produce consistent, parseable results. Format tokens anchor attention, and autoregressive generation maintains format consistency.',
              },
              {
                headline:
                  'Role prompts bias attention, not knowledge.',
                description:
                  'Role and system prompts shift what the model attends to in the input. They shape focus and style, not the model\'s actual knowledge. "SFT teaches format, not knowledge" applies at inference time too.',
              },
              {
                headline:
                  'Example selection: diversity and format consistency over quantity.',
                description:
                  'Diverse examples covering the input space outperform many similar examples. Consistent output format in examples is a stronger signal than more examples with mixed formats.',
              },
              {
                headline:
                  'RAG extends the prompt with retrieved context.',
                description:
                  'Retrieved documents add K/V entries for attention. Same mechanism as few-shot examples, different content. Retrieval quality matters more than quantity\u2014irrelevant context dilutes attention.',
              },
              {
                headline:
                  'The prompt is a program; attention is the interpreter. Prompt engineering is writing better programs.',
                description:
                  'Every technique works because of attention. Understanding the mechanism lets you reason about which techniques will help for a given task, rather than memorizing templates.',
              },
            ]}
          />
        </Row.Content>
      </Row>

      {/* ================================================================
          16. References
          ================================================================ */}
      <Row>
        <Row.Content>
          <ReferencesBlock
            references={[
              {
                title: 'Language Models are Few-Shot Learners',
                authors: 'Brown et al., 2020',
                url: 'https://arxiv.org/abs/2005.14165',
                note: 'The GPT-3 paper that demonstrated in-context learning. Section 3 covers few-shot evaluation. Relevant here as the foundation for prompt engineering.',
              },
              {
                title: 'Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks',
                authors: 'Lewis et al., 2020',
                url: 'https://arxiv.org/abs/2005.11401',
                note: 'The original RAG paper. Introduces the retrieve-then-generate pattern. Read the abstract and Section 1 for the core idea.',
              },
              {
                title: 'Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?',
                authors: 'Min et al., 2022',
                url: 'https://arxiv.org/abs/2202.12837',
                note: 'Shows that format and input distribution matter more than label correctness. Directly relevant to the example selection principles in this lesson.',
              },
              {
                title: 'Calibrate Before Use: Improving Few-Shot Performance of Language Models',
                authors: 'Zhao et al., 2021',
                url: 'https://arxiv.org/abs/2102.09690',
                note: 'Demonstrates ordering sensitivity and recency bias in few-shot prompting. Proposes calibration methods to reduce variance.',
              },
            ]}
          />
        </Row.Content>
      </Row>

      {/* ================================================================
          17. Next Step (Section 11)
          ================================================================ */}
      <Row>
        <Row.Content>
          <NextStepBlock
            href="/app"
            title="When you're done"
            description="You can now construct structured prompts that reliably control model behavior. But there is a class of problems where even the best-structured prompt fails: problems that require more computation than a single forward pass provides. A sentiment classification fits in one pass. A multi-step math problem does not. What happens when you ask the model to &ldquo;think step by step&rdquo;? The next lesson explains why chain-of-thought works&mdash;and it is not because the model &ldquo;decides&rdquo; to think harder."
          />
        </Row.Content>
      </Row>
    </LessonLayout>
  )
}

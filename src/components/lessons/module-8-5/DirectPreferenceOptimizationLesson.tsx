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
  PhaseCard,
  SummaryBlock,
  NextStepBlock,
  ReferencesBlock,
  LessonLink,
} from '@/components/lessons'
import { CodeBlock } from '@/components/common/CodeBlock'
import 'katex/dist/katex.min.css'
import { InlineMath, BlockMath } from 'react-katex'

/**
 * Direct Preference Optimization — Deriving the DPO Loss from First Principles
 *
 * Lesson 1 in Module 8.5 (Preference Optimization Deep Dives). Series 8 Special Topics.
 * Cognitive load: STRETCH (first multi-step mathematical derivation in the course).
 *
 * Teaches the student to derive the DPO loss function from the RLHF objective:
 * - The Bradley-Terry preference model (formalizing pairwise comparisons)
 * - The RLHF objective and its closed-form optimal solution
 * - Substituting the optimal policy into Bradley-Terry to obtain the DPO loss
 * - What each term in the DPO loss does (log-ratios, implicit KL, gradient behavior)
 * - Why the reference model is structurally essential (not optional regularization)
 * - The "implicit reward model" insight
 *
 * Core concepts:
 * - Bradley-Terry preference model: DEVELOPED
 * - Closed-form optimal policy for the RLHF objective: DEVELOPED
 * - The DPO loss function: DEVELOPED
 * - The implicit reward model: INTRODUCED
 * - DPO implementation: APPLIED (notebook)
 *
 * Previous: Various (Series 8 standalone)
 * Next: TBD (other preference optimization deep dives)
 */

// ---------------------------------------------------------------------------
// Derivation Roadmap SVG
// ---------------------------------------------------------------------------
function DerivationRoadmap() {
  return (
    <svg
      viewBox="0 0 680 120"
      className="w-full"
      aria-label="Derivation roadmap: four boxes connected by arrows showing the path from RLHF objective to DPO loss"
    >
      {/* Box 1: RLHF Objective */}
      <rect x="4" y="20" width="140" height="80" rx="8" fill="#6366f1" opacity="0.15" stroke="#6366f1" strokeWidth="1.5" />
      <text x="74" y="50" fontSize="11" fill="#a5b4fc" textAnchor="middle" fontWeight="600">Start: RLHF</text>
      <text x="74" y="66" fontSize="10" fill="#94a3b8" textAnchor="middle">Maximize reward,</text>
      <text x="74" y="80" fontSize="10" fill="#94a3b8" textAnchor="middle">stay close to ref</text>

      {/* Arrow 1 */}
      <line x1="148" y1="60" x2="170" y2="60" stroke="#6366f1" strokeWidth="1.5" />
      <polygon points="170,55 180,60 170,65" fill="#6366f1" />

      {/* Box 2: Optimal Policy */}
      <rect x="184" y="20" width="140" height="80" rx="8" fill="#8b5cf6" opacity="0.15" stroke="#8b5cf6" strokeWidth="1.5" />
      <text x="254" y="50" fontSize="11" fill="#c4b5fd" textAnchor="middle" fontWeight="600">Optimal Policy</text>
      <text x="254" y="66" fontSize="10" fill="#94a3b8" textAnchor="middle">Solve constrained</text>
      <text x="254" y="80" fontSize="10" fill="#94a3b8" textAnchor="middle">optimization</text>

      {/* Arrow 2 */}
      <line x1="328" y1="60" x2="350" y2="60" stroke="#8b5cf6" strokeWidth="1.5" />
      <polygon points="350,55 360,60 350,65" fill="#8b5cf6" />

      {/* Box 3: Implicit Reward */}
      <rect x="364" y="20" width="140" height="80" rx="8" fill="#a78bfa" opacity="0.15" stroke="#a78bfa" strokeWidth="1.5" />
      <text x="434" y="50" fontSize="11" fill="#c4b5fd" textAnchor="middle" fontWeight="600">Implicit Reward</text>
      <text x="434" y="66" fontSize="10" fill="#94a3b8" textAnchor="middle">Rearrange: reward</text>
      <text x="434" y="80" fontSize="10" fill="#94a3b8" textAnchor="middle">= f(policy, ref)</text>

      {/* Arrow 3 */}
      <line x1="508" y1="60" x2="530" y2="60" stroke="#a78bfa" strokeWidth="1.5" />
      <polygon points="530,55 540,60 530,65" fill="#a78bfa" />

      {/* Box 4: DPO Loss */}
      <rect x="544" y="20" width="130" height="80" rx="8" fill="#22c55e" opacity="0.15" stroke="#22c55e" strokeWidth="1.5" />
      <text x="609" y="50" fontSize="11" fill="#86efac" textAnchor="middle" fontWeight="600">DPO Loss</text>
      <text x="609" y="66" fontSize="10" fill="#94a3b8" textAnchor="middle">Substitute into</text>
      <text x="609" y="80" fontSize="10" fill="#94a3b8" textAnchor="middle">Bradley-Terry</text>
    </svg>
  )
}

// ---------------------------------------------------------------------------
// DPO Loss Landscape SVG
// ---------------------------------------------------------------------------
function DpoLossLandscape() {
  // The DPO loss is -log(sigma(x)) where x = beta * (log_ratio_w - log_ratio_l).
  // sigma(x) = 1/(1+exp(-x)), so -log(sigma(x)) = log(1+exp(-x)).
  // We plot loss vs x.
  const points: Array<{ x: number; y: number }> = []
  for (let px = -60; px <= 60; px++) {
    const x = px / 10 // range [-6, 6]
    const loss = Math.log(1 + Math.exp(-x))
    points.push({ x, y: loss })
  }

  // Map to SVG coordinates
  const svgW = 500
  const svgH = 240
  const padL = 60
  const padR = 20
  const padT = 20
  const padB = 50
  const plotW = svgW - padL - padR
  const plotH = svgH - padT - padB

  const xMin = -6
  const xMax = 6
  const yMin = 0
  const yMax = 6.5

  const toSvgX = (v: number) => padL + ((v - xMin) / (xMax - xMin)) * plotW
  const toSvgY = (v: number) => padT + ((yMax - v) / (yMax - yMin)) * plotH

  const pathD = points
    .map((p, i) => `${i === 0 ? 'M' : 'L'} ${toSvgX(p.x).toFixed(1)} ${toSvgY(p.y).toFixed(1)}`)
    .join(' ')

  return (
    <svg
      viewBox={`0 0 ${svgW} ${svgH}`}
      className="w-full"
      aria-label="DPO loss landscape: sigmoid-shaped curve showing high loss when model disagrees with preferences (left) and low loss when model agrees (right)"
    >
      {/* Axes */}
      <line x1={padL} y1={toSvgY(0)} x2={svgW - padR} y2={toSvgY(0)} stroke="#475569" strokeWidth="1" />
      <line x1={padL} y1={padT} x2={padL} y2={toSvgY(0)} stroke="#475569" strokeWidth="1" />

      {/* Y-axis label */}
      <text x="14" y={toSvgY(yMax / 2)} fontSize="11" fill="#94a3b8" textAnchor="middle" transform={`rotate(-90, 14, ${toSvgY(yMax / 2)})`}>
        Loss
      </text>

      {/* X-axis label */}
      <text x={toSvgX(0)} y={svgH - 4} fontSize="11" fill="#94a3b8" textAnchor="middle">
        log-ratio difference (preferred - dispreferred)
      </text>

      {/* Y ticks */}
      {[0, 2, 4, 6].map((v) => (
        <g key={v}>
          <line x1={padL - 4} y1={toSvgY(v)} x2={padL} y2={toSvgY(v)} stroke="#475569" strokeWidth="1" />
          <text x={padL - 8} y={toSvgY(v) + 4} fontSize="10" fill="#94a3b8" textAnchor="end">{v}</text>
        </g>
      ))}

      {/* X ticks */}
      {[-4, -2, 0, 2, 4].map((v) => (
        <g key={v}>
          <line x1={toSvgX(v)} y1={toSvgY(0)} x2={toSvgX(v)} y2={toSvgY(0) + 4} stroke="#475569" strokeWidth="1" />
          <text x={toSvgX(v)} y={toSvgY(0) + 16} fontSize="10" fill="#94a3b8" textAnchor="middle">{v}</text>
        </g>
      ))}

      {/* Shaded regions */}
      <rect x={padL} y={padT} width={toSvgX(0) - padL} height={plotH} fill="#ef4444" opacity="0.06" />
      <rect x={toSvgX(0)} y={padT} width={svgW - padR - toSvgX(0)} height={plotH} fill="#22c55e" opacity="0.06" />

      {/* Region labels */}
      <text x={toSvgX(-3)} y={padT + 16} fontSize="10" fill="#f87171" textAnchor="middle" fontWeight="500">
        Model disagrees
      </text>
      <text x={toSvgX(-3)} y={padT + 28} fontSize="9" fill="#f87171" textAnchor="middle">
        (steep gradient, strong push)
      </text>
      <text x={toSvgX(3)} y={padT + 16} fontSize="10" fill="#4ade80" textAnchor="middle" fontWeight="500">
        Model agrees
      </text>
      <text x={toSvgX(3)} y={padT + 28} fontSize="9" fill="#4ade80" textAnchor="middle">
        (flat gradient, near zero)
      </text>

      {/* The curve */}
      <path d={pathD} fill="none" stroke="#a78bfa" strokeWidth="2.5" />
    </svg>
  )
}

// ---------------------------------------------------------------------------
// Main lesson component
// ---------------------------------------------------------------------------
export function DirectPreferenceOptimizationLesson() {
  return (
    <LessonLayout>
      {/* ================================================================
          Section 1: Header
          ================================================================ */}
      <Row>
        <Row.Content>
          <LessonHeader
            title="Direct Preference Optimization"
            description="Deriving the DPO loss from first principles&mdash;why every term exists and what it does."
            category="Preference Optimization Deep Dives"
          />
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 2: Context + Constraints
          ================================================================ */}
      <Row>
        <Row.Content>
          <ObjectiveBlock>
            Derive the DPO loss function from the RLHF objective, explain what each
            term does and why the reference model is structurally essential, and
            understand DPO well enough to implement it from scratch.
          </ObjectiveBlock>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="What You Already Know">
            From <LessonLink slug="rlhf-and-alignment">RLHF &amp; Alignment</LessonLink>, you know
            DPO as &ldquo;the simpler alternative to PPO.&rdquo;
            From <LessonLink slug="alignment-techniques-landscape">Alignment Techniques Landscape</LessonLink>, you
            can place DPO on the four-axis design space. You know{' '}
            <em>what</em> DPO does. This lesson explains <em>why</em> the
            loss function looks the way it does.
          </TipBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <ConstraintBlock
            title="Scope for This Lesson"
            items={[
              'The Bradley-Terry preference model and its connection to pairwise comparisons',
              'Deriving the closed-form optimal policy from the RLHF objective',
              'Substituting the optimal policy into Bradley-Terry to obtain the DPO loss',
              'What each term in the DPO loss does\u2014log-ratios, sigmoid, implicit KL',
              'A worked numerical example tracing one training step',
              'NOT: PPO algorithm details (clipping, advantage estimation, value functions)',
              'NOT: DPO variants (IPO, KTO, ORPO)\u2014those were introduced in Alignment Techniques Landscape',
              'NOT: reinforcement learning formalism\u2014despite starting from the RLHF objective, the derivation uses calculus and algebra, not RL',
            ]}
          />
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="No RL Required">
            Despite the name &ldquo;RLHF,&rdquo; the derivation involves zero
            reinforcement learning concepts. No value functions, no policy
            gradients, no episodes. It is constrained optimization&mdash;calculus
            and algebra.
          </InsightBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <GradientCard title="The Bridge" color="violet">
            <p>
              You accepted on faith that DPO &ldquo;directly adjusts
              probabilities&rdquo; and that the implicit KL penalty &ldquo;is
              built into the formulation.&rdquo; Today we open the black box.
              The question is not &ldquo;what does DPO do?&rdquo; but{' '}
              <strong>
                &ldquo;why does the DPO loss function look the way it does, and
                why does it work?&rdquo;
              </strong>
            </p>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 3: Recap -- Alignment Foundations
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Recap: Alignment Foundations"
            subtitle="From human preferences to policy optimization"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              <strong>Beat 1: The alignment problem.</strong> SFT gives the model a voice;
              alignment gives it judgment. You cannot write a loss function for
              &ldquo;be helpful&rdquo;&mdash;but you <em>can</em> compare two
              responses and say &ldquo;this one is better.&rdquo; Human
              preference data takes this form: a prompt, two responses, and a
              label indicating which is preferred. Relative signal (A &gt; B) is
              more reliable than absolute scoring.
            </p>
            <p className="text-muted-foreground">
              <strong>Beat 2: The RLHF approach.</strong> Train a reward model on
              preferences&mdash;an &ldquo;experienced editor&rdquo; that scores
              response quality. Then optimize the policy (language model) to
              maximize reward while staying close to the SFT reference via a KL
              penalty&mdash;the &ldquo;continuous version of freeze the
              backbone.&rdquo; Two models, complex training loop.
            </p>
            <p className="text-muted-foreground">
              <strong>Beat 3: DPO overview.</strong> Same goal, no separate reward
              model. &ldquo;Directly adjusts probabilities.&rdquo; Comparable
              results. Widely adopted (Zephyr, Mistral instruct variants, many
              open models). But how? The loss function was never shown.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="Mental Models Carry Forward">
            <ul className="space-y-1 text-sm">
              <li>&bull; &ldquo;SFT gives the model a voice; alignment gives it judgment&rdquo;</li>
              <li>&bull; &ldquo;The reward model is an experienced editor&rdquo;</li>
              <li>&bull; &ldquo;KL penalty = continuous freeze the backbone&rdquo;</li>
              <li>&bull; &ldquo;Tradeoffs, not upgrades&rdquo;</li>
            </ul>
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 4: Hook -- The Derivation Roadmap
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Derivation Roadmap"
            subtitle="Where we are going before we start walking"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Before diving into math, here is the entire derivation path. Four
              steps, each building on the last. You will not get lost because you
              know the destination.
            </p>
            <div className="py-4 px-2 bg-muted/30 rounded-lg overflow-x-auto">
              <DerivationRoadmap />
            </div>
            <p className="text-muted-foreground">
              <strong>The punchline:</strong> the reward model was always inside the
              policy. DPO makes this explicit. By finding the optimal policy in
              closed form and substituting it back into the preference model, we
              eliminate the reward model entirely. The result is a loss function
              that requires only the policy, a reference policy, and preference
              pairs.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Why Preview the Plan?">
            The derivation has multiple steps. If you do not see the destination,
            each step feels arbitrary. This roadmap is your scaffold&mdash;hang
            each step on it as we go.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 5: Explain -- The Bradley-Terry Preference Model
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Bradley-Terry Preference Model"
            subtitle="Formalizing pairwise preferences"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Start with what you already know: preference pairs. A prompt, two
              responses, a human label saying which is better. From{' '}
              <LessonLink slug="rlhf-and-alignment">RLHF &amp; Alignment</LessonLink>, you know
              the reward model assigns scores, and a higher score means a better
              response.
            </p>
            <p className="text-muted-foreground">
              The question: if the reward model gives response A a score
              of <InlineMath math="r_A" /> and response B a score
              of <InlineMath math="r_B" />, what is the <em>probability</em> that
              B is preferred? We need a function that converts score differences
              into probabilities.
            </p>
          </div>
        </Row.Content>
      </Row>

      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The <strong>Bradley-Terry model</strong> answers this:
            </p>
            <div className="py-4 px-6 bg-muted/50 rounded-lg space-y-3">
              <BlockMath math="P(B \succ A) = \sigma(r_B - r_A)" />
              <p className="text-sm text-muted-foreground text-center">
                where <InlineMath math="\sigma(x) = \frac{1}{1 + e^{-x}}" /> is
                the logistic (sigmoid) function
              </p>
            </div>
            <p className="text-muted-foreground">
              The sigmoid function maps any real number to a probability between 0
              and 1. When <InlineMath math="x" /> is large and
              positive, <InlineMath math="\sigma(x) \approx 1" />. When large and
              negative, <InlineMath math="\sigma(x) \approx 0" />. When
              zero, <InlineMath math="\sigma(0) = 0.5" />. You have seen sigmoid
              activations in{' '}
              <LessonLink slug="activation-functions">Activation Functions</LessonLink>&mdash;this
              is the same function, now used for preference modeling.
            </p>
            <p className="text-muted-foreground">
              Why this model? It is the simplest function that converts score
              differences to probabilities with the right properties:
            </p>
            <ul className="list-disc list-inside text-muted-foreground space-y-1 ml-4">
              <li>
                If <InlineMath math="r_B \gg r_A" />,
                then <InlineMath math="P(B \succ A) \approx 1" />&mdash;much
                higher score means near-certain preference
              </li>
              <li>
                If <InlineMath math="r_B = r_A" />,
                then <InlineMath math="P(B \succ A) = 0.5" />&mdash;equal scores
                mean a coin flip
              </li>
              <li>
                If <InlineMath math="r_A \gg r_B" />,
                then <InlineMath math="P(B \succ A) \approx 0" />&mdash;much
                lower score means near-certain rejection
              </li>
            </ul>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="Bradley-Terry Is Not DPO-Specific">
            The Bradley-Terry model dates to 1952 (psychometrics). It is the
            mathematical foundation for chess Elo ratings, Chatbot Arena
            rankings (from{' '}
            <LessonLink slug="evaluating-llms">Evaluating LLMs</LessonLink>),
            recommendation systems, and anywhere you model pairwise preferences.
            DPO adopts it from existing literature.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-sm font-medium text-foreground">
              Example: chess Elo ratings
            </p>
            <p className="text-muted-foreground">
              Player A has Elo 1600, Player B has Elo 1400. The idea behind Elo is
              Bradley-Terry: the probability of winning depends on the score
              difference. If we feed a scaled difference into our sigma function:
            </p>
            <div className="py-3 px-5 bg-muted/50 rounded-lg">
              <BlockMath math="P(A \text{ wins}) = \sigma(R_A - R_B) = \sigma(200)" />
            </div>
            <p className="text-muted-foreground">
              With a raw difference of 200 and natural exponents, sigma saturates
              near 1. The actual Elo system uses a rescaled formula with base-10
              exponents and a divisor of 400, giving roughly 0.76. The details
              differ, but the principle is the same: map a score difference through
              a sigmoid-shaped function to get a win probability. Chatbot Arena
              uses the same model for LLM rankings.
            </p>
            <p className="text-sm font-medium text-foreground mt-4">
              Concrete trace with reward model scores
            </p>
            <p className="text-muted-foreground">
              Recall the reward model training from{' '}
              <LessonLink slug="rlhf-and-alignment">RLHF &amp; Alignment</LessonLink>: the reward
              model learned to assign score 0.7 to the preferred response and 0.3
              to the dispreferred response. Plugging into Bradley-Terry:
            </p>
            <div className="py-3 px-5 bg-muted/50 rounded-lg">
              <BlockMath math="P(\text{preferred}) = \sigma(0.7 - 0.3) = \sigma(0.4) \approx 0.60" />
            </div>
            <p className="text-muted-foreground">
              The reward model&rsquo;s training pushed this probability toward 1.
              The Bradley-Terry model formalizes the relationship you already
              understood intuitively.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Elo = Bradley-Terry">
            If you have encountered Elo ratings (chess, competitive games,
            Chatbot Arena), you have already used the Bradley-Terry model. The
            same math underlies both.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 6: Explain -- The RLHF Objective and Its Optimal Solution
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The RLHF Objective and Its Optimal Solution"
            subtitle="Formalizing 'maximize reward, stay close to the reference'"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Now formalize what you know informally. The RLHF objective:
            </p>
            <div className="py-4 px-6 bg-muted/50 rounded-lg">
              <BlockMath math="\max_\pi \; \mathbb{E}_{x, y \sim \pi}\!\big[r(x, y)\big] - \beta \cdot \text{KL}\!\big(\pi \,\|\, \pi_\text{ref}\big)" />
            </div>
            <p className="text-muted-foreground">
              In words: generate responses that get high reward, but do not drift
              too far from the reference policy.
            </p>
          </div>
        </Row.Content>
      </Row>

      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-sm font-medium text-foreground">
              What each term means
            </p>
            <ul className="list-disc list-inside text-muted-foreground space-y-2 ml-4">
              <li>
                <InlineMath math="\mathbb{E}[r(x, y)]" />: expected reward when
                following policy <InlineMath math="\pi" />. Higher is better.
              </li>
              <li>
                <InlineMath math="\text{KL}(\pi \| \pi_\text{ref})" />: how
                different the current policy is from the reference. Lower means
                closer to the SFT model. Formally:{' '}
                <InlineMath math="\sum_y \pi(y|x) \log \frac{\pi(y|x)}{\pi_\text{ref}(y|x)}" />.
                It measures how different two distributions are&mdash;zero if they
                are identical, always non-negative.
              </li>
              <li>
                <InlineMath math="\beta" />: controls the tradeoff. Large{' '}
                <InlineMath math="\beta" /> = strict, stay very close to the
                reference. Small <InlineMath math="\beta" /> = loose, allow more
                drift for higher reward.
              </li>
            </ul>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="KL Divergence">
            KL divergence measures the &ldquo;distance&rdquo; between two
            probability distributions. Think of it as: &ldquo;how surprised
            would I be by samples from <InlineMath math="\pi" /> if I expected
            samples from <InlineMath math="\pi_\text{ref}" />?&rdquo; The
            larger the KL, the more the policy has drifted.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-sm font-medium text-foreground">
              The key step: this has a closed-form solution
            </p>
            <p className="text-muted-foreground">
              This constrained optimization problem has been studied extensively.
              The optimal policy is:
            </p>
            <div className="py-4 px-6 bg-muted/50 rounded-lg space-y-3">
              <BlockMath math="\pi^*(y \mid x) = \frac{1}{Z(x)} \; \pi_\text{ref}(y \mid x) \; \exp\!\left(\frac{r(y, x)}{\beta}\right)" />
              <p className="text-sm text-muted-foreground text-center">
                where <InlineMath math="Z(x)" /> is a normalizing constant (partition
                function) ensuring probabilities sum to 1
              </p>
            </div>
            <p className="text-muted-foreground">
              <strong>What this says in words:</strong> the optimal policy is the
              reference policy <em>reweighted</em> by the exponentiated reward.
              Responses with high reward get more probability mass, responses with
              low reward get less, all anchored to the reference distribution.
            </p>
            <p className="text-muted-foreground">
              <strong>Why the reference model appears:</strong> the KL constraint
              literally puts <InlineMath math="\pi_\text{ref}" /> into the
              solution. Remove the KL constraint (set <InlineMath math="\beta" /> to
              zero) and <InlineMath math="\pi^*" /> just maximizes reward with
              no anchor&mdash;reward hacking. The reference model is not an
              engineering choice; it falls out of the math.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Roadmap: Step 1 Complete">
            We found the optimal policy. It is the reference policy reweighted by
            exponentiated reward. Now comes the key insight: rearranging to
            express the reward in terms of the policy.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 7: Explain -- From Optimal Policy to DPO Loss
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="From Optimal Policy to DPO Loss"
            subtitle="The implicit reward insight"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Now rearrange the optimal policy equation to solve for the reward.
              Start with what we just derived:
            </p>
            <div className="py-4 px-6 bg-muted/50 rounded-lg space-y-3">
              <p className="text-xs text-muted-foreground">Start with the optimal policy:</p>
              <BlockMath math="\pi^*(y \mid x) = \frac{1}{Z(x)} \; \pi_\text{ref}(y \mid x) \; \exp\!\left(\frac{r(y,x)}{\beta}\right)" />
              <p className="text-xs text-muted-foreground">Multiply both sides by <InlineMath math="Z(x)" /> and divide by <InlineMath math="\pi_\text{ref}(y \mid x)" />:</p>
              <BlockMath math="\frac{\pi^*(y \mid x)}{\pi_\text{ref}(y \mid x)} = \frac{1}{Z(x)} \; \exp\!\left(\frac{r(y,x)}{\beta}\right)" />
              <p className="text-xs text-muted-foreground">Take the log of both sides:</p>
              <BlockMath math="\log \frac{\pi^*(y \mid x)}{\pi_\text{ref}(y \mid x)} = \frac{r(y,x)}{\beta} - \log Z(x)" />
              <p className="text-xs text-muted-foreground">Multiply both sides by <InlineMath math="\beta" /> and rearrange:</p>
              <BlockMath math="r(y, x) = \beta \log \frac{\pi^*(y \mid x)}{\pi_\text{ref}(y \mid x)} + \beta \log Z(x)" />
            </div>
            <p className="text-muted-foreground">
              &ldquo;The reward is determined by the log-ratio of the optimal
              policy to the reference policy (plus a constant that depends only on
              the prompt, not the response).&rdquo;
            </p>
            <p className="text-muted-foreground">
              <strong>What this means:</strong> if you have the optimal policy, you
              do not need a separate reward model. The reward is implicit in the
              policy&mdash;it is the log-ratio of how much more likely the policy
              makes a response compared to the reference.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The Reward Was Always Inside">
            Extend the editor analogy: you do not need a separate editor to score
            the drafts. The writer who has already learned to write well
            implicitly embodies the editorial standards. You can extract the
            reward by measuring how much the policy has shifted from the
            reference.
          </InsightBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-sm font-medium text-foreground">
              Substituting into the Bradley-Terry model
            </p>
            <p className="text-muted-foreground">
              Now substitute the implicit reward into the preference model. For a
              preferred response <InlineMath math="y_w" /> and
              dispreferred <InlineMath math="y_l" />:
            </p>
            <div className="py-4 px-6 bg-muted/50 rounded-lg space-y-3">
              <BlockMath math="P(y_w \succ y_l \mid x) = \sigma\!\big(r(y_w, x) - r(y_l, x)\big)" />
            </div>
            <p className="text-muted-foreground">
              Expanding the rewards:
            </p>
            <div className="py-4 px-6 bg-muted/50 rounded-lg space-y-3">
              <BlockMath math="= \sigma\!\left(\beta \log \frac{\pi(y_w \mid x)}{\pi_\text{ref}(y_w \mid x)} - \beta \log \frac{\pi(y_l \mid x)}{\pi_\text{ref}(y_l \mid x)}\right)" />
              <p className="text-sm text-muted-foreground text-center">
                The <InlineMath math="\beta \log Z(x)" /> terms cancel! Same
                prompt, same partition function, subtraction eliminates it.
              </p>
            </div>
            <p className="text-muted-foreground">
              The DPO loss is the negative log-likelihood of this preference
              probability:
            </p>
            <div className="py-5 px-6 bg-primary/5 border border-primary/20 rounded-lg space-y-3">
              <BlockMath math="\mathcal{L}_\text{DPO} = -\mathbb{E}\!\left[\log \sigma\!\left(\beta \left(\log \frac{\pi(y_w \mid x)}{\pi_\text{ref}(y_w \mid x)} - \log \frac{\pi(y_l \mid x)}{\pi_\text{ref}(y_l \mid x)}\right)\right)\right]" />
            </div>
            <p className="text-muted-foreground">
              Notice that the implicit reward was derived in terms
              of <InlineMath math="\pi^*" />&mdash;the <em>optimal</em> policy.
              But the DPO loss above uses <InlineMath math="\pi" />&mdash;the
              current policy we are training, which is not optimal yet. Why is this
              valid? Because the DPO loss is set up so that minimizing it
              drives <InlineMath math="\pi" /> toward <InlineMath math="\pi^*" />.
              The loss equals zero only when the policy perfectly matches human
              preferences under the Bradley-Terry model&mdash;exactly the
              condition satisfied by <InlineMath math="\pi^*" />. So we substitute{' '}
              <InlineMath math="\pi" /> for <InlineMath math="\pi^*" /> in the
              preference model and use the resulting expression as a training
              objective. Gradient descent does the rest, pushing the policy toward
              the optimum.
            </p>
          </div>
        </Row.Content>
      </Row>

      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              <strong>This is it.</strong> The entire RLHF objective&mdash;reward
              model, KL penalty, policy optimization&mdash;collapsed into a single
              loss function that requires only:
            </p>
            <ol className="list-decimal list-inside text-muted-foreground space-y-1 ml-4">
              <li>
                The current policy <InlineMath math="\pi" />
              </li>
              <li>
                The reference policy <InlineMath math="\pi_\text{ref}" /> (frozen
                copy of the initial model)
              </li>
              <li>
                Preference pairs <InlineMath math="(x, y_w, y_l)" />
              </li>
            </ol>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Not an Approximation">
            This is not an approximation. We solved for the reward model exactly
            and substituted. The optimal policy of this loss is identical to the
            optimal policy of the RLHF objective under the Bradley-Terry model.
          </WarningBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <GradientCard title="Roadmap: Step 3 Complete" color="emerald">
            <p>
              The reward model is gone. The loss depends only on the policy and the
              reference. Every property of DPO from the design space&mdash;paired
              data (Bradley-Terry compares pairs), reference model (KL constraint),
              offline (loss only needs existing data), no reward model (solved for
              analytically)&mdash;is now <strong>derivable</strong>, not just
              descriptive.
            </p>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 8: Check 1 -- Predict the Gradient
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Check: Predict the Gradient"
            subtitle="When is the loss large? When is it small?"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Look at the DPO loss. The argument to <InlineMath math="\sigma" /> is
              the difference in log-ratios. Reason about what happens in two cases:
            </p>
            <GradientCard title="Predict Before Reading" color="cyan">
              <div className="space-y-3 text-sm">
                <p>
                  <strong>Case 1:</strong> The model already prefers the
                  winner (log-ratio for <InlineMath math="y_w" /> is much larger
                  than for <InlineMath math="y_l" />). What is the loss?
                </p>
                <details className="mt-2">
                  <summary className="font-medium cursor-pointer text-primary">
                    Reveal
                  </summary>
                  <div className="mt-2">
                    <p>
                      The argument to <InlineMath math="\sigma" /> is large and
                      positive. <InlineMath math="\sigma \approx 1" />,
                      so <InlineMath math="\log \sigma \approx 0" />, and the loss
                      is near zero. <strong>The gradient is weak.</strong>
                    </p>
                  </div>
                </details>

                <p className="mt-4">
                  <strong>Case 2:</strong> The model prefers the loser
                  (log-ratio for <InlineMath math="y_l" /> is larger than
                  for <InlineMath math="y_w" />). What is the loss?
                </p>
                <details className="mt-2">
                  <summary className="font-medium cursor-pointer text-primary">
                    Reveal
                  </summary>
                  <div className="mt-2">
                    <p>
                      The argument to <InlineMath math="\sigma" /> is negative.{' '}
                      <InlineMath math="\sigma \approx 0" />,
                      so <InlineMath math="\log \sigma" /> is very negative, and the
                      loss is large. <strong>The gradient is strong</strong>, pushing
                      the model to correct itself.
                    </p>
                  </div>
                </details>
              </div>
            </GradientCard>
          </div>
        </Row.Content>
      </Row>

      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              <strong>Key insight:</strong> DPO&rsquo;s gradient is strongest for
              training examples where the model <em>disagrees</em> with human
              preferences. When the model already agrees, the gradient is near
              zero. DPO naturally focuses learning on the hard cases.
            </p>
            <div className="py-4 px-2 bg-muted/30 rounded-lg overflow-x-auto">
              <DpoLossLandscape />
            </div>
            <p className="text-sm text-muted-foreground text-center">
              The DPO loss as a function of the log-ratio difference. Left: model
              disagrees with the preference (steep gradient, strong learning
              signal). Right: model agrees (flat, near-zero gradient).
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Not 'Increase Preferred Probability'">
            The intuitive description from{' '}
            <LessonLink slug="rlhf-and-alignment">RLHF &amp; Alignment</LessonLink> was
            &ldquo;increase preferred, decrease dispreferred.&rdquo; That captures
            the <em>effect</em> but misses the <em>mechanism</em>. DPO&rsquo;s
            gradient is context-dependent&mdash;it focuses on the cases where the
            model is most wrong.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 9: Explain -- What Each Term Does (Numerical Example)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="What Each Term Does"
            subtitle="A worked numerical example"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Walk through one DPO training step with concrete numbers. The prompt
              is the quantum computing example from{' '}
              <LessonLink slug="rlhf-and-alignment">RLHF &amp; Alignment</LessonLink>:
              &ldquo;Explain quantum computing to a 10-year-old.&rdquo;
            </p>
            <ul className="list-disc list-inside text-muted-foreground space-y-1 ml-4">
              <li>
                <InlineMath math="y_w" /> = age-appropriate analogy response
                (preferred)
              </li>
              <li>
                <InlineMath math="y_l" /> = jargon-heavy response (dispreferred)
              </li>
            </ul>
          </div>
        </Row.Content>
      </Row>

      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-sm font-medium text-foreground">
              Step 1: Log-probabilities under each model
            </p>
            <div className="overflow-x-auto">
              <table className="w-full text-sm border-collapse">
                <thead>
                  <tr className="border-b border-muted">
                    <th className="p-2 text-left text-muted-foreground font-medium" />
                    <th className="p-2 text-center text-muted-foreground font-medium">
                      Policy <InlineMath math="\log \pi" />
                    </th>
                    <th className="p-2 text-center text-muted-foreground font-medium">
                      Reference <InlineMath math="\log \pi_\text{ref}" />
                    </th>
                  </tr>
                </thead>
                <tbody>
                  <tr className="border-b border-muted/50">
                    <td className="p-2 text-muted-foreground">
                      Preferred <InlineMath math="y_w" />
                    </td>
                    <td className="p-2 text-center font-mono text-sm">&minus;45.2</td>
                    <td className="p-2 text-center font-mono text-sm">&minus;48.1</td>
                  </tr>
                  <tr>
                    <td className="p-2 text-muted-foreground">
                      Dispreferred <InlineMath math="y_l" />
                    </td>
                    <td className="p-2 text-center font-mono text-sm">&minus;42.8</td>
                    <td className="p-2 text-center font-mono text-sm">&minus;43.0</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </Row.Content>
      </Row>

      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-sm font-medium text-foreground">
              Step 2: Compute log-ratios
            </p>
            <div className="py-3 px-5 bg-muted/50 rounded-lg space-y-3">
              <p className="text-sm text-muted-foreground">
                <strong>Preferred:</strong>{' '}
                <InlineMath math="\log \frac{\pi(y_w|x)}{\pi_\text{ref}(y_w|x)} = -45.2 - (-48.1) = 2.9" />
              </p>
              <p className="text-xs text-muted-foreground ml-4">
                The policy is <em>more</em> likely to produce the preferred
                response than the reference is. Good.
              </p>
              <p className="text-sm text-muted-foreground">
                <strong>Dispreferred:</strong>{' '}
                <InlineMath math="\log \frac{\pi(y_l|x)}{\pi_\text{ref}(y_l|x)} = -42.8 - (-43.0) = 0.2" />
              </p>
              <p className="text-xs text-muted-foreground ml-4">
                The policy is only <em>slightly</em> more likely to produce the
                dispreferred response.
              </p>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="Log-Ratios, Not Absolutes">
            The DPO loss operates on log-ratios: how much has the policy shifted
            relative to the reference? This is fundamentally different from
            absolute log-probabilities. A response the policy finds very likely
            gets no credit unless it is <em>more</em> likely than the reference
            expected.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-sm font-medium text-foreground">
              Step 3: Compute the loss
            </p>
            <div className="py-3 px-5 bg-muted/50 rounded-lg space-y-3">
              <p className="text-sm text-muted-foreground">
                Difference: <InlineMath math="2.9 - 0.2 = 2.7" />
              </p>
              <p className="text-sm text-muted-foreground">
                Multiply by <InlineMath math="\beta = 0.1" />:{' '}
                <InlineMath math="0.1 \times 2.7 = 0.27" />
              </p>
              <p className="text-sm text-muted-foreground">
                Apply sigma: <InlineMath math="\sigma(0.27) \approx 0.57" />
              </p>
              <p className="text-sm text-muted-foreground">
                Loss: <InlineMath math="-\log(0.57) \approx 0.56" />
              </p>
            </div>
            <p className="text-muted-foreground">
              <strong>Interpretation:</strong> the model slightly prefers the
              preferred response (log-ratio difference is positive), but not
              strongly. The loss is moderate, pushing the model to increase this
              preference.
            </p>
          </div>
        </Row.Content>
      </Row>

      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-sm font-medium text-foreground">
              Step 4: What convergence looks like
            </p>
            <p className="text-muted-foreground">
              After training, imagine the log-ratio difference grows to 10.0:
            </p>
            <div className="py-3 px-5 bg-muted/50 rounded-lg space-y-2">
              <p className="text-sm text-muted-foreground">
                <InlineMath math="\beta \times 10.0 = 1.0" />, <InlineMath math="\sigma(1.0) \approx 0.73" />, Loss <InlineMath math="= -\log(0.73) \approx 0.31" />
              </p>
            </div>
            <p className="text-muted-foreground">
              Smaller loss&mdash;the model already agrees strongly with the
              preference. The gradient is weaker, and the model spends less effort
              on this pair.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The Implicit KL Penalty">
            The reference model defines the coordinate system. If the policy had
            just memorized &ldquo;always say the preferred response&rdquo;
            (absolute log-prob very high), but the reference also found it
            likely, the log-ratio would be small. DPO measures how much the
            policy has <em>changed</em>, not the absolute probability. This IS
            the implicit KL penalty.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 10: Negative Example -- DPO Without the Reference Model
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="What Breaks Without the Reference Model"
            subtitle="A negative example"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              What happens if you remove the reference model from the DPO loss?
              The loss simplifies to:
            </p>
            <div className="py-4 px-6 bg-muted/50 rounded-lg">
              <BlockMath math="\mathcal{L} = -\log \sigma\!\left(\beta \big(\log \pi(y_w \mid x) - \log \pi(y_l \mid x)\big)\right)" />
            </div>
            <p className="text-muted-foreground">
              This says: &ldquo;just make the preferred response more probable than
              the dispreferred response.&rdquo; Sounds reasonable. What goes wrong?
            </p>
            <p className="text-muted-foreground">
              The model pushes <InlineMath math="\log \pi(y_w|x)" /> as high as
              possible and <InlineMath math="\log \pi(y_l|x)" /> as low as
              possible. But probability mass is finite. Pushing{' '}
              <InlineMath math="y_l" /> down pushes probability mass
              elsewhere&mdash;potentially to bizarre outputs. The model ends up
              assigning near-zero probability to most text, including text that is
              perfectly good but did not appear as <InlineMath math="y_w" /> in the
              training data.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Probability Mass Collapse">
            Without the reference model anchor, the model over-optimizes for the
            specific preferences in the training set at the cost of general
            coherence. This is the preference optimization version of reward
            hacking.
          </WarningBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <ComparisonRow
            left={{
              title: 'Without Reference',
              color: 'rose',
              items: [
                'Loss measures absolute log-probabilities',
                'Model pushes y_w up, y_l down without limit',
                'Probability mass collapses onto few responses',
                'Good text not in training set gets near-zero probability',
              ],
            }}
            right={{
              title: 'With Reference (DPO)',
              color: 'emerald',
              items: [
                'Loss measures log-RATIOS relative to reference',
                'Model adjusts relative to a stable anchor',
                'Reference prevents probability mass collapse',
                'General coherence preserved via implicit KL',
              ],
            }}
          />
        </Row.Content>
      </Row>

      <Row>
        <Row.Content>
          <GradientCard title="The Reference Model Is Structural" color="orange">
            <p>
              The reference model is not a regularizer you could remove for
              stronger optimization. It defines the coordinate system. The loss
              does not reward making <InlineMath math="y_w" /> likely in absolute
              terms&mdash;only making <InlineMath math="y_w" /> more
              likely <em>relative to the reference</em>. Remove the anchor and the
              optimization has no stable foundation.
            </p>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 11: Explore -- Implementation
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Implementation"
            subtitle="The DPO loss in code"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The DPO training loop is remarkably similar to supervised
              learning&mdash;the &ldquo;familiar training loop shape&rdquo; from{' '}
              <LessonLink slug="rlhf-and-alignment">RLHF &amp; Alignment</LessonLink>. The
              complexity lives in the loss function, not the loop.
            </p>
          </div>
        </Row.Content>
      </Row>

      <Row>
        <Row.Content>
          <div className="space-y-4">
            <PhaseCard number={1} title="Compute log-probabilities" subtitle="Four forward passes, two models" color="cyan">
              <p className="text-sm text-muted-foreground">
                For each preference pair, compute log-probs of both responses
                under both models. The reference model is frozen (no gradient).
              </p>
            </PhaseCard>
            <PhaseCard number={2} title="Compute log-ratios" subtitle="Policy minus reference" color="blue">
              <p className="text-sm text-muted-foreground">
                For each response: <InlineMath math="\log \pi(y|x) - \log \pi_\text{ref}(y|x)" />.
                This gives the log-ratio for preferred and dispreferred.
              </p>
            </PhaseCard>
            <PhaseCard number={3} title="Compute the DPO loss" subtitle="Sigmoid of scaled difference" color="violet">
              <p className="text-sm text-muted-foreground">
                <InlineMath math="-\log \sigma(\beta \cdot (\text{log\_ratio}_w - \text{log\_ratio}_l))" />.
                Average over the batch.
              </p>
            </PhaseCard>
            <PhaseCard number={4} title="Backpropagate and update" subtitle="Standard optimizer step" color="emerald">
              <p className="text-sm text-muted-foreground">
                Gradients flow through the policy model only. The reference model
                is frozen&mdash;load it once and never update it.
              </p>
            </PhaseCard>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="The Reference Model">
            The reference model is the same model as the initial policy (before
            training), kept as a frozen snapshot. You load it once. It never
            receives gradients. It is there so the loss can measure how much the
            policy has changed.
          </TipBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-sm font-medium text-foreground">
              Core implementation (~15 lines)
            </p>
            <CodeBlock
              code={`import torch
import torch.nn.functional as F

def dpo_loss(
    policy_logps_w: torch.Tensor,   # log P(y_w|x) under policy
    policy_logps_l: torch.Tensor,   # log P(y_l|x) under policy
    ref_logps_w: torch.Tensor,      # log P(y_w|x) under reference
    ref_logps_l: torch.Tensor,      # log P(y_l|x) under reference
    beta: float = 0.1,
) -> torch.Tensor:
    # Log-ratios: how much has the policy shifted from the reference?
    log_ratio_w = policy_logps_w - ref_logps_w
    log_ratio_l = policy_logps_l - ref_logps_l

    # DPO loss: -log sigma(beta * (log_ratio_w - log_ratio_l))
    logits = beta * (log_ratio_w - log_ratio_l)
    loss = -F.logsigmoid(logits).mean()
    return loss`}
              language="python"
              filename="dpo_loss.py"
            />
            <p className="text-muted-foreground">
              Each line maps to a step in the derivation. The log-probabilities of
              a full response are the sum of per-token log-probabilities&mdash;the
              same quantity you have computed in prior notebooks.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="~15 Lines">
            The entire DPO loss function is ~15 lines of PyTorch. The
            complexity is in the derivation that justifies these lines, not in
            the code itself.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 12: Elaborate -- The Implicit Reward Model
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Implicit Reward Model"
            subtitle="Any policy paired with a reference defines a reward"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The derivation showed:
            </p>
            <div className="py-4 px-6 bg-muted/50 rounded-lg">
              <BlockMath math="r(y, x) = \beta \log \frac{\pi(y \mid x)}{\pi_\text{ref}(y \mid x)} + \beta \log Z(x)" />
            </div>
            <p className="text-muted-foreground">
              <strong>Any policy paired with a reference policy implicitly defines
              a reward function.</strong> You can extract a reward for any response
              by computing the log-ratio of the policy to the reference.
            </p>
            <p className="text-muted-foreground">
              After DPO training, you can compute the &ldquo;implicit reward&rdquo;
              for any response by measuring how much more likely the trained model
              makes it compared to the reference. Responses the model strongly
              prefers (high log-ratio) have high implicit reward.
            </p>
            <p className="text-muted-foreground">
              Connection to the design space: DPO is plotted as &ldquo;no reward
              model&rdquo; in{' '}
              <LessonLink slug="alignment-techniques-landscape">Alignment Techniques Landscape</LessonLink>.
              More precisely, it has no <em>explicit</em> reward model that is
              separately trained. But it implicitly defines one. The distinction is
              between a separately trained reward model (PPO) and an implicit one
              extracted from the policy (DPO).
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The Deepest Insight">
            The reward model and the policy are not independent. Given one, you
            can derive the other. This is DPO&rsquo;s key contribution:
            showing that the reward model was never a separate entity&mdash;it
            was always embedded in the optimal policy.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 13: Check 2 -- Transfer Question
          ================================================================ */}
      <Row>
        <Row.Content>
          <GradientCard title="Check Your Understanding" color="cyan">
            <div className="space-y-3 text-sm">
              <p>
                A colleague says: &ldquo;DPO is just supervised learning on
                preference pairs. There is nothing interesting about the
                loss.&rdquo; How would you respond?
              </p>
              <details className="mt-2">
                <summary className="font-medium cursor-pointer text-primary">
                  Reveal
                </summary>
                <div className="mt-2 space-y-2">
                  <p>
                    DPO&rsquo;s loss is <em>not</em> naive supervised learning.
                    Four key differences:
                  </p>
                  <ol className="list-decimal list-inside space-y-1 ml-2">
                    <li>
                      It operates on <strong>pairs</strong>, not individual
                      examples
                    </li>
                    <li>
                      It uses log-<strong>ratios</strong> relative to a reference,
                      not absolute probabilities
                    </li>
                    <li>
                      The sigmoid structure means the gradient{' '}
                      <strong>focuses on hard cases</strong> where the model
                      disagrees
                    </li>
                    <li>
                      It is mathematically <strong>equivalent to the RLHF
                      objective</strong>, not a heuristic
                    </li>
                  </ol>
                  <p>
                    The &ldquo;just supervised learning&rdquo; framing captures the
                    simplicity of the training loop (good!), but misses the
                    derivation that guarantees the same optimal policy as RLHF.
                  </p>
                </div>
              </details>
            </div>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 14: Practice -- Notebook Exercises
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Practice: Notebook Exercises"
            subtitle="From hand calculation to implementation"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The exercises are cumulative&mdash;each builds on the previous.
            </p>
            <div className="space-y-3">
              <GradientCard title="Exercise 1: Verify by Hand (Guided)" color="cyan">
                <p>
                  Given pre-computed log-probabilities for 5 preference pairs,
                  compute the DPO loss for each pair and the total loss. Vary{' '}
                  <InlineMath math="\beta" /> and observe: higher{' '}
                  <InlineMath math="\beta" /> = more conservative, lower{' '}
                  <InlineMath math="\beta" /> = more aggressive.
                </p>
              </GradientCard>
              <GradientCard title="Exercise 2: Implement the Loss (Supported)" color="blue">
                <p>
                  Write the DPO loss function in PyTorch. Verify against Exercise
                  1&rsquo;s hand calculations. Compute the gradient via autograd
                  and inspect its direction: confirm it pushes toward the preferred
                  response, with magnitude proportional to disagreement.
                </p>
              </GradientCard>
              <GradientCard title="Exercise 3: Train a Small Model (Supported)" color="violet">
                <p>
                  Use GPT-2 small with preference pairs. Implement the full loop:
                  load model as policy, frozen copy as reference, iterate, compute
                  loss, backpropagate, update. Compare outputs before and after.
                </p>
              </GradientCard>
              <GradientCard title="Exercise 4: Explore Implicit Reward (Independent)" color="emerald">
                <p>
                  After training, compute the implicit reward for several
                  responses (log-ratio of policy to reference). Verify that
                  preferred responses have higher implicit reward. Generate a new
                  response not in the training data and compute its implicit
                  reward&mdash;does it generalize?
                </p>
              </GradientCard>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Exercise Progression">
            <ul className="space-y-1 text-sm">
              <li>&bull; Ex 1: pure arithmetic (verify the formula)</li>
              <li>&bull; Ex 2: implementation (code the loss)</li>
              <li>&bull; Ex 3: training loop (end-to-end DPO)</li>
              <li>&bull; Ex 4: interpretation (implicit reward)</li>
            </ul>
            Each exercise uses the output of the previous one.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 15: Summary
          ================================================================ */}
      <Row>
        <Row.Content>
          <SummaryBlock
            title="Key Takeaways"
            items={[
              {
                headline: 'DPO is not a heuristic or approximation\u2014it is derived from the RLHF objective.',
                description:
                  'Start with KL-constrained reward maximization, find the closed-form optimal policy, rearrange to express the reward in terms of the policy, substitute into the Bradley-Terry preference model. The resulting loss produces the same optimal policy as RLHF.',
              },
              {
                headline: 'The DPO loss requires only the policy, the reference policy, and preference pairs.',
                description:
                  'No reward model. No RL algorithm. The loss operates on log-ratios of how much the policy has shifted from the reference for each response in a preference pair.',
              },
              {
                headline: 'The reference model is structurally essential\u2014not optional regularization.',
                description:
                  'It defines the coordinate system. Without it, the loss measures absolute probabilities, leading to probability mass collapse. The reference model prevents this by anchoring the optimization to log-ratios.',
              },
              {
                headline: 'DPO\u2019s gradient naturally focuses on the hardest examples.',
                description:
                  'The sigmoid structure means the gradient is strongest when the model disagrees with human preferences and near zero when it already agrees. Learning effort concentrates where it matters most.',
              },
              {
                headline: 'The reward model was always inside the policy.',
                description:
                  'Any policy paired with a reference implicitly defines a reward function: the log-ratio of policy to reference. The reward model is not missing from DPO\u2014it is absorbed into the policy itself.',
              },
            ]}
          />
        </Row.Content>
      </Row>

      <Row>
        <Row.Content>
          <InsightBlock title="Mental Model">
            <strong>
              The reward model was always inside the policy. DPO makes this
              explicit&mdash;it derives the exact optimal policy from the RLHF
              objective, discovers that the reward is a function of the policy and
              reference, and substitutes it out. No approximation. No RL. Just
              algebra.
            </strong>
          </InsightBlock>
        </Row.Content>
      </Row>

      {/* ================================================================
          References
          ================================================================ */}
      <Row>
        <Row.Content>
          <ReferencesBlock
            references={[
              {
                title: 'Direct Preference Optimization: Your Language Model is Secretly a Reward Model',
                authors: 'Rafailov et al., 2023 (Stanford)',
                url: 'https://arxiv.org/abs/2305.18290',
                note: 'The original DPO paper. Section 4 contains the derivation. Section 5 has the experiments.',
              },
              {
                title: 'Training Language Models to Follow Instructions with Human Feedback',
                authors: 'Ouyang et al., 2022 (OpenAI)',
                url: 'https://arxiv.org/abs/2203.02155',
                note: 'The InstructGPT paper that established the RLHF pipeline DPO derives from.',
              },
              {
                title: 'Zephyr: Direct Distillation of LM Alignment',
                authors: 'Tunstall et al., 2023 (Hugging Face)',
                url: 'https://arxiv.org/abs/2310.16944',
                note: 'Practical DPO at scale. Shows DPO training a 7B model to outperform much larger aligned models.',
              },
            ]}
          />
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 16: Next Step
          ================================================================ */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              You now understand one preference optimization method at the
              mathematical level. The design space from{' '}
              <LessonLink slug="alignment-techniques-landscape">Alignment Techniques Landscape</LessonLink>{' '}
              has other points: IPO changes the preference model (bounded instead
              of Bradley-Terry), KTO changes the data format (single responses
              instead of pairs), ORPO removes the reference model entirely. Each
              variation changes an assumption in the derivation. The mathematical
              framework you now have is the foundation for understanding any of
              them.
            </p>
          </div>
        </Row.Content>
      </Row>

      <Row>
        <Row.Content>
          <NextStepBlock
            href="/app"
            title="Return to Dashboard"
            description="Review your progress and choose what to explore next."
          />
        </Row.Content>
      </Row>
    </LessonLayout>
  )
}

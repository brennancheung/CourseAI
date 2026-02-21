'use client'

import { useMemo } from 'react'
import { LessonLayout } from '@/components/lessons/LessonLayout'
import { Row } from '@/components/layout/Row'
import {
  LessonHeader,
  ObjectiveBlock,
  ConstraintBlock,
  SectionHeader,
  InsightBlock,
  TipBlock,
  TryThisBlock,
  WarningBlock,
  SummaryBlock,
  NextStepBlock,
  GradientCard,
  ComparisonRow,
  PhaseCard,
  ReferencesBlock,
  LessonLink,
} from '@/components/lessons'
import { ExercisePanel } from '@/components/widgets/ExercisePanel'
import { DiffusionNoiseWidget } from '@/components/widgets/DiffusionNoiseWidget'

// ---------------------------------------------------------------------------
// Inline noise progression strip (static, no interactivity)
// ---------------------------------------------------------------------------

const STRIP_IMG_SIZE = 28
const STRIP_SCALE = 3
const STRIP_STEPS = [0, 250, 500, 750, 1000]
const STRIP_LABELS = ['t=0', 't=250', 't=500', 't=750', 't=T']

function stripMulberry32(seed: number): () => number {
  let a = seed | 0
  return () => {
    a = (a + 0x6D2B79F5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

function stripGaussian(rng: () => number): number {
  const u1 = rng()
  const u2 = rng()
  return Math.sqrt(-2 * Math.log(Math.max(u1, 1e-10))) * Math.cos(2 * Math.PI * u2)
}

function stripGetAlphaBar(t: number): number {
  const s = 0.008
  const f = Math.cos(((t / 1000 + s) / (1 + s)) * Math.PI * 0.5)
  return f * f
}

function stripGenerateImage(): number[] {
  const pixels = new Array<number>(STRIP_IMG_SIZE * STRIP_IMG_SIZE).fill(0)
  for (let y = 0; y < STRIP_IMG_SIZE; y++) {
    for (let x = 0; x < STRIP_IMG_SIZE; x++) {
      const idx = y * STRIP_IMG_SIZE + x
      const inBody = x >= 7 && x <= 20 && y >= 8 && y <= 24
      const inSleeves = x >= 3 && x <= 24 && y >= 8 && y <= 13
      const inNeck = x >= 11 && x <= 16 && y >= 7 && y <= 10 && (y - 7) < (Math.abs(x - 13.5) * 0.8)
      const isCollar = x >= 10 && x <= 17 && y === 7
      if (isCollar) {
        pixels[idx] = 220
      } else if ((inBody || inSleeves) && !inNeck) {
        const centerDist = Math.abs(x - 13.5) / 10
        pixels[idx] = Math.max(140, 200 - centerDist * 40)
      }
    }
  }
  return pixels
}

function stripApplyNoise(base: number[], t: number, seed: number): number[] {
  if (t === 0) return base
  const alphaBar = stripGetAlphaBar(t)
  const sc = Math.sqrt(alphaBar)
  const nc = Math.sqrt(1 - alphaBar)
  const rng = stripMulberry32(seed)
  return base.map((px) => {
    const n = px / 255
    const noisy = sc * n + nc * stripGaussian(rng)
    return Math.max(0, Math.min(255, noisy * 255))
  })
}

function NoiseProgressionStrip() {
  const base = useMemo(() => stripGenerateImage(), [])
  const frames = useMemo(
    () => STRIP_STEPS.map((t, i) => stripApplyNoise(base, t, 42 + i * 1000)),
    [base],
  )
  const size = STRIP_IMG_SIZE * STRIP_SCALE

  return (
    <div className="my-4 flex flex-col items-center gap-1.5">
      <div className="flex items-center gap-2">
        {frames.map((pixels, i) => (
          <div key={STRIP_STEPS[i]} className="flex flex-col items-center gap-0.5">
            <svg width={size} height={size} className="rounded border border-border/30">
              {pixels.map((v, pi) => {
                const x = (pi % STRIP_IMG_SIZE) * STRIP_SCALE
                const y = Math.floor(pi / STRIP_IMG_SIZE) * STRIP_SCALE
                const c = Math.round(v)
                return (
                  <rect
                    key={pi}
                    x={x}
                    y={y}
                    width={STRIP_SCALE}
                    height={STRIP_SCALE}
                    fill={`rgb(${c},${c},${c})`}
                  />
                )
              })}
            </svg>
            <span className="text-[9px] font-mono text-muted-foreground/70">
              {STRIP_LABELS[i]}
            </span>
          </div>
        ))}
      </div>
      <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
        <span>clean</span>
        <span className="text-muted-foreground/50">&rarr; &rarr; &rarr;</span>
        <span>pure noise</span>
      </div>
    </div>
  )
}

/**
 * The Diffusion Idea
 *
 * Lesson 1 in Module 6.2 (Diffusion).
 * First lesson in the module. Cognitive load: BUILD.
 *
 * Introduces the core diffusion insight through intuition and analogy:
 * breaking image generation into many small denoising steps makes the
 * problem learnable. Distinguishes the forward process (adding noise)
 * from the reverse process (learned denoising).
 *
 * Core concepts at INTRODUCED:
 * - Forward process (gradual noise destruction)
 * - Reverse process (learned iterative denoising)
 *
 * Core insight at DEVELOPED:
 * - "Small steps make it learnable"
 *
 * Previous: Exploring Latent Spaces (module 6.1, lesson 4)
 * Next: The Forward Process (module 6.2, lesson 2)
 */

export function TheDiffusionIdeaLesson() {
  return (
    <LessonLayout>
      {/* ================================================================
          Header
          ================================================================ */}
      <Row>
        <Row.Content>
          <LessonHeader
            title="The Diffusion Idea"
            description="Destruction is easy. Creation from scratch is impossibly hard. But undoing one small step of destruction? That's learnable."
            category="Diffusion"
          />
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 1: Context + Constraints (Outline item 1)
          ================================================================ */}
      <Row>
        <Row.Content>
          <ObjectiveBlock>
            Understand why breaking image generation into many small denoising
            steps makes the problem learnable. Distinguish the forward process
            (adding noise) from the reverse process (learned denoising). No
            formulas, no code&mdash;just the idea.
          </ObjectiveBlock>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="What You Already Know">
            You built a VAE that generates images by sampling from a smooth
            latent space. The images are recognizable but blurry&mdash;a
            fundamental consequence of the reconstruction-vs-KL tradeoff. This
            lesson introduces the approach that overcomes that quality ceiling.
          </TipBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <ConstraintBlock
            title="Scope for This Lesson"
            items={[
              'Building intuition for WHY diffusion works (small steps make creation learnable)',
              'The forward process conceptually (gradual noise addition)',
              'The reverse process conceptually (learned denoising, step by step)',
              'Motivating the approach from the VAE quality ceiling',
              'NOT: mathematical formulation, noise schedules, or alpha_bar—that is the next lesson',
              'NOT: the training objective or loss function',
              'NOT: the sampling algorithm or code',
              'NOT: U-Net or any specific architecture',
              'NOT: score matching, SDEs, or continuous-time formulations',
            ]}
          />
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 2: Hook — Before/After (Outline item 2)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Quality Gap"
            subtitle="Same goal, radically different results"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              In <LessonLink slug="exploring-latent-spaces">Exploring Latent Spaces</LessonLink>, you experienced
              generation firsthand&mdash;sampling random vectors from a trained
              VAE and watching novel images appear. You also saw the quality
              ceiling: blurry, soft, recognizable but never sharp. That
              blurriness is not a training failure. It is the fundamental price
              of a smooth, sampleable latent space.
            </p>
          </div>
          <ComparisonRow
            left={{
              title: 'Your VAE',
              color: 'amber',
              items: [
                '28×28 grayscale',
                'Blurry, soft edges',
                'Recognizable but not impressive',
                'One-shot generation from latent space',
              ],
            }}
            right={{
              title: 'Stable Diffusion',
              color: 'emerald',
              items: [
                '512×512+ full color',
                'Sharp, detailed, photorealistic',
                'Text-guided, arbitrary scenes',
                'Iterative refinement from noise',
              ],
            }}
          />
          <div className="space-y-4 mt-4">
            <p className="text-muted-foreground">
              Same goal&mdash;sample from P(x) to generate images. Radically
              different approach. The VAE compresses to a latent space and
              decodes in one shot. Diffusion does something completely
              different: it starts from <strong>pure noise</strong> and
              refines it step by step, removing a little noise at each stage
              until a clean image emerges.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The Bridge">
            In <LessonLink slug="exploring-latent-spaces">Exploring Latent Spaces</LessonLink>, you were told:
            &ldquo;The VAE proved the concept. Diffusion delivers the
            quality.&rdquo; This lesson explains <em>how</em>.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 3: Destruction Is Easy (Outline item 3)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Destruction Is Easy"
            subtitle="The forward process"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Drop a bead of ink into a glass of water. Watch what happens.
              The ink spreads&mdash;molecules bouncing randomly, drifting outward,
              diffusing through the water. After enough time, the water is a
              uniform pale color. The original drop is gone.
            </p>
            <p className="text-muted-foreground">
              This physical process is called <strong>diffusion</strong>&mdash;and
              it is where diffusion models get their name. The key properties:
            </p>
            <ul className="list-disc list-inside text-muted-foreground space-y-1 ml-4">
              <li>
                Each small step is <strong>predictable</strong>&mdash;molecules
                spread randomly by a tiny amount
              </li>
              <li>
                The process is <strong>purely mechanical</strong>&mdash;no
                intelligence or learning required
              </li>
              <li>
                The final state (uniform color) contains{' '}
                <strong>no trace of the original</strong>&mdash;you cannot look
                at pale water and know where the ink was dropped
              </li>
            </ul>
            <p className="text-muted-foreground">
              Now replace &ldquo;ink in water&rdquo; with &ldquo;image under
              noise.&rdquo; Take a photograph and add a tiny amount of random
              static at each step. The image gradually dissolves. After enough
              steps, only pure noise remains&mdash;no trace of the original
              image, just random pixels.
            </p>
            <p className="text-muted-foreground">
              This is the <strong>forward process</strong>. It takes any clean
              image and destroys it, step by step, by adding Gaussian
              noise&mdash;random values drawn from N(0,1), the same
              distribution you sampled &epsilon; from in the reparameterization
              trick. Each pixel gets an independent random nudge at every step.
              You already know how to do this.
            </p>
            <NoiseProgressionStrip />
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The Name">
            &ldquo;Diffusion&rdquo; is not a metaphor. The mathematical process
            of adding Gaussian noise to an image over many steps is directly
            analogous to physical diffusion&mdash;molecules spreading randomly
            through a medium. Same math, different domain.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 4: Check — Predict and Verify (Outline item 4)
          ================================================================ */}
      <Row>
        <Row.Content>
          <GradientCard title="Predict and Verify" color="cyan">
            <div className="space-y-3 text-sm">
              <p>
                What happens if you add noise to two completely different
                images&mdash;a cat and a car&mdash;for enough steps?
              </p>
              <details className="mt-3">
                <summary className="font-medium cursor-pointer text-primary">
                  Reveal
                </summary>
                <div className="mt-2 space-y-2">
                  <p>
                    Both converge to the same thing: pure random static. After
                    enough noise is added, the cat and the car are
                    indistinguishable&mdash;both are just Gaussian noise. All
                    roads lead to noise. The identity of the original image is
                    completely erased.
                  </p>
                </div>
              </details>
            </div>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 5: Why One-Shot Reversal Fails (Outline item 5)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Why One-Shot Reversal Fails"
            subtitle="The impossibility of creating from pure noise"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Destruction was easy. Can we just reverse it? Take the pure noise
              at the end, feed it to a neural network, and ask it to produce the
              original image in a single step?
            </p>
            <p className="text-muted-foreground">
              No. And the reason is fundamental, not technical.
            </p>
            <p className="text-muted-foreground">
              Look at pure static&mdash;random pixels with no structure. Which
              image is hiding underneath? A cat? A car? A landscape? A face?{' '}
              <strong>There are infinitely many images</strong> that are
              consistent with any given patch of random noise. The problem is
              massively underdetermined. Asking a neural network to produce
              &ldquo;the&rdquo; image from pure noise is asking it to choose
              one answer out of an infinite set&mdash;with no information to
              guide the choice.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="This Is What VAEs Do">
            Notice the parallel: the VAE generates images by decoding a single
            random vector in one step. The result is blurry precisely because the
            decoder must hedge its bets across many possible images. One-shot
            generation from randomness produces averaged-out, soft results.
          </WarningBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Think of it like a jigsaw puzzle in a tornado. Throw all the
              pieces into the air and hope they land assembled. The odds are
              zero. One-shot creation from maximal disorder does not work,
              whether you are assembling puzzle pieces or generating pixels.
            </p>
          </div>
          <ComparisonRow
            left={{
              title: 'One-Shot (Impossible)',
              color: 'rose',
              items: [
                'Pure noise → clean image in one step',
                'Infinitely many valid answers',
                'No information to guide the choice',
                'Like assembling a puzzle in a tornado',
              ],
            }}
            right={{
              title: 'Small Steps (Learnable)',
              color: 'emerald',
              items: [
                'Remove a little noise at each step',
                'Most of the image is still there',
                'The task is constrained and well-defined',
                'Like sculpting: rough shape → details → polish',
              ],
            }}
          />
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 6: The Key Insight — Small Steps (Outline item 6)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Key Insight: Small Steps"
            subtitle="Why decomposing generation makes it learnable"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Here is the question that changes everything: could <em>you</em>{' '}
              denoise a slightly noisy photograph?
            </p>
            <p className="text-muted-foreground">
              Yes. If someone showed you a photo with a tiny bit of static, you
              could point at the noisy pixels and say &ldquo;that is noise,
              not signal.&rdquo; The image is <strong>mostly still
              there</strong>. The task is easy because the gap between the noisy
              image and the clean image is small.
            </p>
            <p className="text-muted-foreground">
              Now, could you denoise pure static? No. You have no idea what
              image was buried under it. The gap between noise and the clean
              image is enormous&mdash;the entire image must be invented.
            </p>
            <p className="text-muted-foreground">
              <strong>This is the entire insight of diffusion models.</strong>{' '}
              Creation from scratch is impossibly hard. But undoing a single
              small step of destruction is learnable. A neural network that can
              remove a little bit of noise is solving a tractable problem. Chain
              1,000 of those small steps together&mdash;each one removing a
              little noise&mdash;and you get creation.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The Core Idea">
            Denoising a slightly noisy image is easy. Denoising pure static is
            impossible. Diffusion decomposes the impossible task into 1,000
            easy tasks chained together. Each step is a small refinement, not a
            creative leap.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Misconception: generating from nothing — placed immediately after
          "chain 1,000 steps" so the student does not carry a wrong model
          through the widget and later sections */}
      <Row>
        <Row.Content>
          <GradientCard title="Misconception: Generating from Nothing" color="amber">
            <div className="space-y-3 text-sm">
              <p>
                It is tempting to say diffusion models &ldquo;generate images
                from nothing&rdquo; because they start from pure noise, which
                looks like nothing. But the model{' '}
                <strong>never generates from nothing</strong>. It always starts
                from something noisy and makes it <em>slightly less
                noisy</em>.
              </p>
              <p>
                If you skip the first 900 denoising steps and start from step
                100, you get a slightly noisy image, not nothing. Each step is
                a small refinement, not a creative act from the void. The
                &ldquo;creation&rdquo; emerges from the accumulation of a
                thousand small corrections.
              </p>
            </div>
          </GradientCard>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Not From Scratch">
            Each denoising step is: &ldquo;take what I have and make it
            slightly better.&rdquo; No single step creates an image. The image
            emerges from the <em>sequence</em> of small improvements.
          </WarningBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Think of a sculptor working a block of marble. You do not carve a
              nostril into a rough block. First you establish the rough
              shape&mdash;head, shoulders, torso. Then the face. Then the
              features. Then the fine details. Each step is simple; the result is
              complex.
            </p>
            <p className="text-muted-foreground">
              Diffusion denoising works the same way. At early steps (heavy
              noise), the model makes big structural decisions: is this a face or
              a landscape? At later steps (light noise), it adds fine
              details&mdash;eyelashes, textures, sharp edges. This mirrors the
              CNN feature hierarchy you already know from earlier lessons: global
              structure first, local details last.
            </p>
            <p className="text-muted-foreground">
              Here is another way to picture it. Think of all possible images as
              a vast high-dimensional space. Real, natural images occupy a tiny,
              thin surface&mdash;a manifold&mdash;within that space, much like
              the latent space surface you explored in{' '}
              <strong>Exploring Latent Spaces</strong>. Adding noise pushes an
              image off the manifold into the surrounding void. Denoising is
              learning to walk back toward that surface, one step at a time.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Feature Hierarchy Connection">
            Remember CNN feature visualizations? Early layers detect edges and
            shapes. Later layers detect complex patterns. Diffusion denoising
            follows the same progression&mdash;at heavy noise, the model decides
            edges and outlines (like the first convolutional layer). At light
            noise, it adds textures and fine patterns (like the deeper layers).
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 7: Interactive Widget (Outline item 7)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="See It: The Forward Process"
            subtitle="Watch an image dissolve into noise"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Drag the slider to add noise to an image, step by step. Watch the
              image dissolve from clean to pure static. Then think about the
              reverse: the model learns to go from right to left, one small step
              at a time.
            </p>
          </div>
          <ExercisePanel title="Explore the Forward Process" subtitle="Drag the slider to add noise">
            <DiffusionNoiseWidget />
          </ExercisePanel>
        </Row.Content>
        <Row.Aside>
          <TryThisBlock title="Experiment">
            <ul className="space-y-2 text-sm">
              <li>&bull; Find the noise level where you can no longer tell what the image is.</li>
              <li>&bull; Go one step back&mdash;could you guess the original from here?</li>
              <li>&bull; Notice how early noise preserves global shape but loses fine detail. Late noise destroys everything.</li>
              <li>&bull; At each level, ask: &ldquo;Could a neural network learn to remove <em>just this much</em> noise?&rdquo;</li>
            </ul>
          </TryThisBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 8: What the Model Learns (Outline item 8)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="What the Model Actually Learns"
            subtitle="Different noise levels, different tasks"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              At each noise level, the neural network sees a noisy image and
              predicts what to remove. But the <em>nature</em> of the task
              changes dramatically depending on how much noise is present:
            </p>
          </div>
          <div className="space-y-3">
            <PhaseCard
              number={1}
              title="High Noise (early denoising steps)"
              subtitle="Decide the composition"
              color="violet"
            >
              <p className="text-sm">
                The image is mostly static. The model must decide the global
                structure&mdash;is this a face? A landscape? A building? These
                are big, coarse decisions. Like the sculptor roughing out the
                basic shape from a block of marble.
              </p>
            </PhaseCard>
            <PhaseCard
              number={2}
              title="Medium Noise (middle steps)"
              subtitle="Refine the structure"
              color="blue"
            >
              <p className="text-sm">
                Global structure is established. Now the model adds
                medium-scale features&mdash;the shape of a nose, the curve of a
                horizon, the outline of a window. The image is recognizable but
                lacks sharpness.
              </p>
            </PhaseCard>
            <PhaseCard
              number={3}
              title="Low Noise (final denoising steps)"
              subtitle="Paint the details"
              color="cyan"
            >
              <p className="text-sm">
                The image is nearly clean. The model adds fine
                details&mdash;eyelashes, fabric textures, sharp edges, subtle
                color gradients. Like the sculptor polishing the finished piece.
              </p>
            </PhaseCard>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Multi-Scale Generation">
            High noise = decide the composition. Medium noise = refine the
            structure. Low noise = paint the details. Diffusion generates images
            from coarse to fine&mdash;the same progression as the CNN feature
            hierarchy you studied in earlier lessons.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Misconception: denoising is trivially easy */}
      <Row>
        <Row.Content>
          <GradientCard title="Is Denoising Trivially Easy?" color="amber">
            <div className="space-y-3 text-sm">
              <p>
                Removing a little noise sounds simple&mdash;like a Photoshop
                filter. If each step is easy, why is diffusion impressive?
              </p>
              <p>
                The answer: denoising is <strong>not</strong> trivially easy at
                every noise level. At high noise, the image is almost pure
                static. The model must <em>hallucinate</em> plausible image
                content from very little signal&mdash;that is genuine
                generation, not just cleanup. At low noise, the model must
                reproduce precise fine details. The key insight is that at any{' '}
                <strong>single</strong> noise level, the task is learnable
                (the increment is small enough), even though creating the whole
                image at once is not.
              </p>
            </div>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 9: Check — Transfer (Outline item 9)
          ================================================================ */}
      <Row>
        <Row.Content>
          <GradientCard title="Transfer Check" color="cyan">
            <div className="space-y-3 text-sm">
              <p>
                Your friend says: &ldquo;Diffusion models are just fancy
                Photoshop noise reduction filters.&rdquo; What is wrong with
                this claim?
              </p>
              <details className="mt-3">
                <summary className="font-medium cursor-pointer text-primary">
                  Reveal
                </summary>
                <div className="mt-2 space-y-2">
                  <p>
                    Photoshop denoising removes noise from a real
                    photograph&mdash;the signal is present but corrupted. The
                    filter recovers existing content.
                  </p>
                  <p>
                    Diffusion denoising at high noise levels must{' '}
                    <strong>invent</strong> plausible content, not recover
                    existing content. At step 999 of reverse diffusion, the model
                    must decide what the image <em>is</em>&mdash;that is
                    generation, not restoration. A Photoshop filter never has to
                    decide whether a photo contains a face or a landscape.
                  </p>
                </div>
              </details>
            </div>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 10: Connecting to What You Know (Outline item 10)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Same Building Blocks, Different Question"
            subtitle="Diffusion is not a new universe"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The language of diffusion&mdash;forward process, reverse process,
              noise schedules&mdash;can make it feel like an entirely new
              paradigm unrelated to anything you have learned. It is not.
            </p>
            <p className="text-muted-foreground">
              The denoising network uses <strong>convolutional
              layers</strong>&mdash;the same conv layers from your CNN lessons.
              The loss function is <strong>MSE</strong>&mdash;the same loss you
              used in your very first lesson on linear regression. The training
              loop is <strong>sample data, compute loss, backpropagate, update
              weights</strong>&mdash;the same loop you have written dozens of
              times. The building blocks are identical.
            </p>
            <p className="text-muted-foreground">
              What is new is the <strong>question</strong>. Instead of
              &ldquo;what class is this image?&rdquo; (classification) or
              &ldquo;can you reconstruct this image?&rdquo; (autoencoder), the
              question is: <strong>&ldquo;what noise was added to this
              image?&rdquo;</strong>
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The Pattern Continues">
            In <LessonLink slug="from-classification-to-generation">From Classification to Generation</LessonLink> you learned:
            &ldquo;Same building blocks, different question.&rdquo; That mental
            model applies again. Conv layers, MSE loss, backprop&mdash;the
            revolution is in what you ask the network to do, not in how you
            build it.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Address the "no bottleneck" difference */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              One notable difference from autoencoders: diffusion has{' '}
              <strong>no bottleneck</strong>. The denoising network works in
              the full image space, not through a compressed
              representation. The learning mechanism is the noise schedule, not
              information compression. You should not expect to find a latent
              space with mu and sigma parameters here.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Latent Diffusion (Later)">
            Stable Diffusion <em>does</em> combine diffusion with a latent
            space&mdash;but that is latent diffusion, which you will study in
            Module 6.3. For now, think of diffusion as operating directly on
            images.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Brief practical note about training (misconception about sequential training) */}
      <Row>
        <Row.Content>
          <GradientCard title="A Practical Note" color="blue">
            <div className="space-y-3 text-sm">
              <p>
                You might wonder: does training iterate through all 1,000 steps
                for every image? No&mdash;each training step picks{' '}
                <strong>one random noise level</strong>. How this works is the
                subject of the next lesson.
              </p>
            </div>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 11: Summary (Outline item 11)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SummaryBlock
            title="Key Takeaways"
            items={[
              {
                headline:
                  'Destruction is easy; creation from scratch is impossibly hard; but undoing a small step of destruction is learnable.',
                description:
                  'Adding noise to an image is trivial (forward process). Going from pure noise to a clean image in one step is underdetermined—infinitely many images are consistent with noise. But removing a small amount of noise is a tractable task a neural network can learn.',
              },
              {
                headline:
                  'The forward process gradually adds Gaussian noise until the image becomes pure static. The reverse process undoes this, one step at a time.',
                description:
                  'The forward process is mechanical—no learning required. The reverse process is where the neural network lives, learning to denoise at every noise level. Chain enough reverse steps together and you get generation.',
              },
              {
                headline:
                  'This is still “same building blocks, different question.”',
                description:
                  `Conv layers from your CNN lessons. MSE loss from your first regression. The same training loop you have written many times. What’s new is the question: “What noise was added to this image?” The decomposition of generation into many small denoising steps is the revolutionary idea.`,
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
              &ldquo;A neural network that can remove a little noise is solving
              an easy problem. Chain a thousand of those together, and you get
              creation.&rdquo;
            </strong>
          </InsightBlock>
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 12: Next Step (Outline item 12)
          ================================================================ */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              You understand <em>why</em> diffusion works&mdash;small steps
              make creation learnable. Next: the math of the forward process.
              Noise schedules, Gaussian properties, and an elegant shortcut
              that makes training practical.
            </p>
          </div>
        </Row.Content>
      </Row>

      <Row>
        <Row.Content>
          <ReferencesBlock
            references={[
              {
                title: 'Denoising Diffusion Probabilistic Models',
                authors: 'Ho, Jain & Abbeel, 2020',
                url: 'https://arxiv.org/abs/2006.11239',
                note: 'The foundational DDPM paper that established gradual noise addition and learned iterative denoising for generation.',
              },
            ]}
          />
        </Row.Content>
      </Row>

      <Row>
        <Row.Content>
          <NextStepBlock
            href="/app"
            title="Up Next: The Forward Process"
            description="The mathematical formulation of noise addition—noise schedules, alpha_bar, and a closed-form formula that lets you jump to any timestep directly."
          />
        </Row.Content>
      </Row>
    </LessonLayout>
  )
}

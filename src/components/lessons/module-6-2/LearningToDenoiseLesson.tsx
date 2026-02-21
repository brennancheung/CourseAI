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
import { MermaidDiagram } from '@/components/widgets/MermaidDiagram'
import { ExercisePanel } from '@/components/widgets/ExercisePanel'
import { TrainingStepSimulator } from '@/components/widgets/TrainingStepSimulator'
import { ExternalLink } from 'lucide-react'
import 'katex/dist/katex.min.css'
import { InlineMath, BlockMath } from 'react-katex'

/**
 * Learning to Denoise (DDPM Training Objective)
 *
 * Lesson 3 in Module 6.2 (Diffusion). Lesson 7 overall in Series 6.
 * Cognitive load: BUILD (relief after the STRETCH of the-forward-process).
 *
 * Teaches the DDPM training objective and algorithm:
 * - The training objective: predict epsilon (the noise), not x_0
 * - Why noise prediction is preferred (consistent target, algebraic equivalence)
 * - The complete training algorithm (pseudocode level)
 * - Three faces of MSE (regression, reconstruction, noise prediction)
 *
 * Core concepts:
 * - DDPM training objective (noise prediction with MSE): DEVELOPED
 * - DDPM training algorithm (data prep + standard loop): DEVELOPED
 *
 * Previous: The Forward Process (module 6.2, lesson 2)
 * Next: Sampling and Generation (module 6.2, lesson 4)
 */

export function LearningToDenoiseLesson() {
  return (
    <LessonLayout>
      {/* ================================================================
          Section 1: Header
          ================================================================ */}
      <Row>
        <Row.Content>
          <LessonHeader
            title="Learning to Denoise"
            description="The DDPM training objective&mdash;predict the noise, compare with MSE loss, update weights. The same training loop you already know."
            category="Diffusion"
          />
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 2: Context + Constraints (Outline item 1)
          ================================================================ */}
      <Row>
        <Row.Content>
          <ObjectiveBlock>
            Understand the DDPM training objective&mdash;predict the noise that
            was added&mdash;and trace the complete training algorithm step by
            step. Recognize that the loss function is the same MSE you have used
            since your very first lessons.
          </ObjectiveBlock>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="What You Already Know">
            In <LessonLink slug="the-forward-process">The Forward Process</LessonLink>, you derived the closed-form
            formula that jumps to any noise level in one step. This lesson uses
            that formula as a tool inside the training algorithm.
          </TipBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <ConstraintBlock
            title="Scope for This Lesson"
            items={[
              'The DDPM training objective: predict noise, compute MSE loss',
              'Why predicting noise is preferred over predicting the clean image',
              'The complete training algorithm at pseudocode level',
              'Connecting MSE loss from Series 1 to the diffusion context',
              'NOT: the reverse/sampling process (how to use the trained model to generate)—that is next',
              'NOT: the U-Net architecture or how the network receives the timestep',
              'NOT: code implementation—that comes in the capstone lesson',
              'NOT: score matching, SDEs, or the full variational lower bound derivation',
            ]}
          />
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 3: Brief Recap (Outline item 2)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Where We Left Off"
            subtitle="The formula you derived is about to become useful"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              In <LessonLink slug="the-forward-process">The Forward Process</LessonLink>, you derived a closed-form
              formula that jumps from a clean image to any noise level in one
              step:
            </p>
            <div className="py-3 px-4 bg-violet-500/5 border border-violet-500/20 rounded-lg">
              <BlockMath math="x_t = \sqrt{\bar{\alpha}_t} \cdot x_0 + \sqrt{1 - \bar{\alpha}_t} \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)" />
            </div>
            <p className="text-muted-foreground">
              That formula was not just elegant math. It is the data preparation
              step for every iteration of training. Today you will see exactly
              how.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="From Math to Tool">
            The closed-form formula shifts from being something you derived to
            something you <em>use</em>. It is the engine inside the training
            loop.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 4: Hook (Outline item 3)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Surprisingly Simple Objective"
            subtitle="After all that math, it comes down to this"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              You have just survived the hardest lesson in this module. You
              derived the forward process, proved the closed-form shortcut, and
              verified it with arithmetic. The reverse process&mdash;learning to
              undo noise&mdash;must be at least as complex, right?
            </p>
            <p className="text-muted-foreground">
              Here is the DDPM training loss:
            </p>
            <div className="py-4 px-6 bg-violet-500/10 border-2 border-violet-500/30 rounded-lg">
              <BlockMath math="L = \| \epsilon - \epsilon_\theta(x_t, t) \|^2" />
            </div>
            <p className="text-muted-foreground">
              That is MSE loss. The same loss you have been computing since{' '}
              <strong>Loss Functions</strong> in your very first module.{' '}
              <InlineMath math="\epsilon" /> is the noise that was actually
              added.{' '}
              <InlineMath math="\epsilon_\theta(x_t, t)" /> is the
              network&rsquo;s prediction of that noise. Subtract, square, average.
              Done.
            </p>
            <p className="text-muted-foreground">
              The training objective is an open-book exam: the network receives a
              noisy image (the question), guesses which noise was added (its
              answer), and we check against the actual noise (the answer key).
              The wrongness score? MSE&mdash;the same &ldquo;wrongness
              score&rdquo; from your third lesson.
            </p>

            <p className="text-muted-foreground">
              Write them side by side and see for yourself:
            </p>
            <div className="space-y-2">
              <div className="py-2 px-4 bg-muted/50 rounded-lg">
                <BlockMath math="\text{Series 1: } L = \frac{1}{n}\sum(\hat{y}_i - y_i)^2" />
              </div>
              <div className="py-2 px-4 bg-violet-500/5 border border-violet-500/20 rounded-lg">
                <BlockMath math="\text{DDPM: } L = \frac{1}{n}\sum(\epsilon_{\theta,i} - \epsilon_i)^2" />
              </div>
            </div>
            <p className="text-muted-foreground text-sm">
              Same formula. Different letters. The same{' '}
              <code className="text-xs bg-muted px-1 py-0.5 rounded">nn.MSELoss()</code>{' '}
              in PyTorch.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Cognitive Relief">
            After the mathematical intensity of the forward process, the
            training objective is almost embarrassingly simple. That is
            intentional&mdash;the hard math already happened.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 5: Why Predict Noise (Outline item 4)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Why Predict Noise, Not the Clean Image?"
            subtitle="The choice that seems backwards but is not"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Given a noisy image{' '}
              <InlineMath math="x_t" />, the network could try to predict two
              things:
            </p>
          </div>
          <ComparisonRow
            left={{
              title: 'Option A: Predict x₀',
              color: 'amber',
              items: [
                'Predict the clean image directly',
                'Seems intuitive—denoising means recovering the original',
                'The target varies wildly: cats, dogs, landscapes, shoes...',
              ],
            }}
            right={{
              title: 'Option B: Predict ε',
              color: 'emerald',
              items: [
                'Predict the noise that was added',
                'Seems backwards—why predict what you want to remove?',
                'The target is always from the same distribution: N(0, 1)',
              ],
            }}
          />
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="The #1 Misconception">
            Most people assume the model predicts the clean image. It does not.
            It predicts the noise. This feels unintuitive until you see why.
          </WarningBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Think about what each target looks like during training. If the
              model predicts <InlineMath math="x_0" />, its target changes with
              every image&mdash;a T-shirt, a sneaker, a handbag. Each is a
              completely different pattern of pixel values. If the model predicts{' '}
              <InlineMath math="\epsilon" />, its target is <em>always</em>{' '}
              standard Gaussian noise. The distribution of the target is the
              same regardless of whether the original image was a cat or a
              mountain.
            </p>
            <p className="text-muted-foreground">
              A consistent target makes optimization easier. The gradients are
              better behaved. The loss landscape is smoother. The model does not
              need to simultaneously learn &ldquo;what images look like&rdquo;
              and &ldquo;how to remove noise&rdquo;&mdash;it only needs to learn
              the second task.
            </p>
            <p className="text-muted-foreground">
              But wait&mdash;if the model only predicts noise, have we lost the
              ability to recover the clean image? Not at all. Rearrange the
              closed-form formula:
            </p>

            <div className="py-3 px-4 bg-muted/50 rounded-lg">
              <BlockMath math="x_0 = \frac{x_t - \sqrt{1 - \bar{\alpha}_t} \cdot \epsilon_\theta}{\sqrt{\bar{\alpha}_t}}" />
            </div>

            <p className="text-muted-foreground">
              Given the noisy image <InlineMath math="x_t" /> and the predicted
              noise <InlineMath math="\epsilon_\theta" />, you recover{' '}
              <InlineMath math="x_0" /> with basic algebra. Predicting noise and
              predicting the clean image are <strong>algebraically equivalent</strong>&mdash;you
              can always compute one from the other. Nothing is lost.
            </p>
            <p className="text-muted-foreground">
              Ho et al. (2020) confirmed this empirically: noise prediction
              produces better sample quality than direct image prediction. A
              consistent target is not just theoretically nicer&mdash;it works
              better in practice.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Nothing Is Lost">
            Predicting <InlineMath math="\epsilon" /> and predicting{' '}
            <InlineMath math="x_0" /> are mathematically equivalent. You can
            always recover one from the other using the closed-form formula.
            The network just has an easier time learning one over the other.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 6: Check (Outline item 5)
          ================================================================ */}
      <Row>
        <Row.Content>
          <GradientCard title="Predict and Verify" color="cyan">
            <div className="space-y-3 text-sm">
              <p>
                Your colleague says: &ldquo;Predicting noise is wasteful because
                you have to do extra math to recover the clean image. Why not
                predict the clean image directly?&rdquo;
              </p>
              <details className="mt-3">
                <summary className="font-medium cursor-pointer text-primary">
                  Reveal
                </summary>
                <div className="mt-2 space-y-2">
                  <p>
                    The &ldquo;extra math&rdquo; is one line of algebra. The
                    benefit is a <strong>consistent target</strong>: the noise is
                    always from{' '}
                    <InlineMath math="\mathcal{N}(0, I)" /> regardless of the
                    image content. Direct image prediction forces the network to
                    hit wildly varying targets (every possible image), which
                    makes optimization harder. The tiny algebraic recovery step
                    is a small price for much better gradients during training.
                  </p>
                </div>
              </details>
            </div>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 7: Training Algorithm (Outline item 6)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Training Algorithm"
            subtitle="Seven steps. Most of them you already know."
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Here is the complete DDPM training algorithm. Read each step and
              notice how many pieces are familiar.
            </p>
          </div>
        </Row.Content>
      </Row>

      <Row>
        <Row.Content>
          <div className="space-y-3">
            <PhaseCard
              number={1}
              title="Sample a training image"
              subtitle="Data loading"
              color="blue"
            >
              <p className="text-sm">
                Pick an image <InlineMath math="x_0" /> from the dataset. A
                T-shirt, a sneaker&mdash;whatever the batch serves up. This is
                the standard first step of any training loop.
              </p>
            </PhaseCard>

            <PhaseCard
              number={2}
              title="Sample a random timestep"
              subtitle="Diffusion-specific"
              color="violet"
            >
              <p className="text-sm">
                Pick <InlineMath math="t \sim \text{Uniform}(1, T)" />. Not a
                sequence. Not in order. One random number. In one batch, the
                model might see{' '}
                <InlineMath math="t = 723" /> for one image,{' '}
                <InlineMath math="t = 42" /> for another, and{' '}
                <InlineMath math="t = 891" /> for a third.
              </p>
            </PhaseCard>

            <PhaseCard
              number={3}
              title="Sample noise"
              subtitle="Diffusion-specific"
              color="violet"
            >
              <p className="text-sm">
                Draw <InlineMath math="\epsilon \sim \mathcal{N}(0, I)" />.
                Fresh Gaussian noise, same shape as the image. This is the
                answer key&mdash;the noise we are about to add and will ask the
                network to predict.
              </p>
            </PhaseCard>

            <PhaseCard
              number={4}
              title="Create the noisy image"
              subtitle="The closed-form formula"
              color="violet"
            >
              <p className="text-sm">
                Compute{' '}
                <InlineMath math="x_t = \sqrt{\bar{\alpha}_t} \cdot x_0 + \sqrt{1 - \bar{\alpha}_t} \cdot \epsilon" />.
                This is the formula you derived in{' '}
                <strong>The Forward Process</strong>. Teleport directly to noise
                level <InlineMath math="t" />. No iterating through intermediate
                steps.
              </p>
            </PhaseCard>

            <PhaseCard
              number={5}
              title="Predict the noise"
              subtitle="Forward pass"
              color="blue"
            >
              <p className="text-sm">
                Feed the noisy image and timestep to the network:{' '}
                <InlineMath math="\epsilon_\theta = \text{network}(x_t, t)" />.
                The network takes the exam&mdash;it outputs its best guess of the
                noise that was added. The timestep embedding is the open-book
                answer: the network always knows what noise level it is dealing
                with.
              </p>
            </PhaseCard>

            <PhaseCard
              number={6}
              title="Compute MSE loss"
              subtitle="Same loss from Series 1"
              color="blue"
            >
              <p className="text-sm">
                <InlineMath math="L = \| \epsilon - \epsilon_\theta \|^2" />.
                Grade the exam: how far off was the prediction? The same MSE
                formula. The same{' '}
                <code className="text-xs bg-muted px-1 py-0.5 rounded">nn.MSELoss</code>{' '}
                in PyTorch.
              </p>
            </PhaseCard>

            <PhaseCard
              number={7}
              title="Backpropagate and update"
              subtitle="Standard training loop"
              color="blue"
            >
              <p className="text-sm">
                <code className="text-xs bg-muted px-1 py-0.5 rounded">loss.backward()</code>,{' '}
                <code className="text-xs bg-muted px-1 py-0.5 rounded">optimizer.step()</code>.
                Learn from mistakes and try again. The heartbeat of training,
                unchanged since <strong>Training Loop</strong> in Series 2.
              </p>
            </PhaseCard>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="What Is New vs Familiar">
            Steps 2, 3, and 4 (highlighted in purple) are
            diffusion-specific&mdash;choosing a noise level and creating the
            noisy image. Steps 1, 5, 6, and 7 (in blue) are the standard
            training loop you have used dozens of times. The heartbeat has not
            changed.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 8: Flow Diagram (Visual modality)
          ================================================================ */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Here is the same algorithm as a data flow. Trace the path from the
              training image to the loss:
            </p>
            <MermaidDiagram chart={`
              graph LR
                X0["x₀ (clean image)"] --> CF["Closed-form formula"]
                T["t ~ Uniform(1,T)"] --> CF
                E["ε ~ N(0,I)"] --> CF
                CF --> XT["x_t (noisy image)"]
                XT --> NN["Neural network"]
                T --> NN
                NN --> EP["predicted noise ε_θ"]
                EP --> MSE["MSE Loss"]
                E --> MSE
                MSE --> BP["Backprop + Update"]
            `} />
            <p className="text-muted-foreground text-sm">
              Three inputs (image, timestep, noise) flow through the
              closed-form formula to create the noisy image. The network
              predicts the noise. MSE compares prediction to truth. Gradients
              flow back. Repeat.
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 9: Misconceptions (Outline items addressing #2 and #3)
          ================================================================ */}
      <Row>
        <Row.Content>
          <GradientCard title="One Random Timestep, Not a Sequence" color="amber">
            <div className="space-y-3 text-sm">
              <p>
                A common mistake: &ldquo;The model trains on timestep 1, then
                timestep 2, then timestep 3...&rdquo; <strong>No.</strong> Each
                training iteration picks one random timestep. The model might
                see <InlineMath math="t = 500" /> on this iteration and{' '}
                <InlineMath math="t = 37" /> on the next. Different images in
                the same batch have different timesteps.
              </p>
              <p>
                This is why the closed-form formula matters. If training had to
                iterate through all previous timesteps to reach{' '}
                <InlineMath math="t = 500" />, it would be unbearably slow.
                Instead, the formula teleports to any{' '}
                <InlineMath math="t" /> in one step. Random timestep sampling is
                critical for efficiency and ensures the model learns all noise
                levels equally.
              </p>
            </div>
          </GradientCard>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Not Sequential">
            The forward process was <em>defined</em> as step 1, step 2, ...
            step T. But training does not follow that sequence. Each iteration
            is a random jump to a single timestep.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* Mini-batch example: batch-level timestep randomness */}
      <Row>
        <Row.Content>
          <div className="rounded-lg bg-muted/30 p-4 space-y-3">
            <p className="text-sm font-medium text-foreground">
              What a real training batch looks like
            </p>
            <p className="text-sm text-muted-foreground">
              In one batch with three images, each gets its own random timestep:
            </p>
            <div className="overflow-x-auto">
              <table className="w-full text-sm border-collapse">
                <thead>
                  <tr className="border-b border-border">
                    <th className="text-left py-1.5 pr-3 text-foreground font-medium text-xs">Image</th>
                    <th className="text-left py-1.5 pr-3 text-foreground font-medium text-xs">Timestep</th>
                    <th className="text-left py-1.5 text-foreground font-medium text-xs">Noise Level</th>
                  </tr>
                </thead>
                <tbody className="text-muted-foreground text-xs">
                  <tr className="border-b border-border/50">
                    <td className="py-1.5 pr-3">Cat photo</td>
                    <td className="py-1.5 pr-3 font-mono">t = 800</td>
                    <td className="py-1.5">Heavy noise&mdash;mostly static</td>
                  </tr>
                  <tr className="border-b border-border/50">
                    <td className="py-1.5 pr-3">Shoe photo</td>
                    <td className="py-1.5 pr-3 font-mono">t = 50</td>
                    <td className="py-1.5">Light noise&mdash;barely visible</td>
                  </tr>
                  <tr>
                    <td className="py-1.5 pr-3">Landscape</td>
                    <td className="py-1.5 pr-3 font-mono">t = 400</td>
                    <td className="py-1.5">Moderate noise&mdash;details fading</td>
                  </tr>
                </tbody>
              </table>
            </div>
            <p className="text-sm text-muted-foreground">
              Each gets its own <InlineMath math="\epsilon" />, its own{' '}
              <InlineMath math="x_t" />, its own{' '}
              <InlineMath math="\epsilon_\theta" />. MSE is averaged across the
              batch. Different noise levels, different images, one gradient
              update.
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 10: Concrete Example (Outline item 6, continued)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Tracing a Training Step"
            subtitle="Concrete values through the algorithm"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Let us walk through one training iteration with specific values.
              Imagine a 28&times;28 T-shirt image from Fashion-MNIST.
            </p>
            <div className="rounded-lg bg-muted/30 p-4 space-y-3">
              <p className="text-sm font-medium text-foreground">
                Setup
              </p>
              <ul className="list-disc list-inside text-muted-foreground space-y-1 ml-4 text-sm">
                <li>
                  <InlineMath math="x_0" />: T-shirt image (784 pixel values,
                  normalized to [-1, 1])
                </li>
                <li>
                  Random timestep: <InlineMath math="t = 500" /> (midpoint of
                  the schedule)
                </li>
                <li>
                  <InlineMath math="\bar{\alpha}_{500} \approx 0.05" /> (using a
                  cosine schedule&mdash;only 5% signal remains at the midpoint)
                </li>
                <li>
                  <InlineMath math="\epsilon" />: a fresh sample from{' '}
                  <InlineMath math="\mathcal{N}(0, I)" /> (784 random values)
                </li>
              </ul>
            </div>

            <div className="rounded-lg bg-muted/30 p-4 space-y-3">
              <p className="text-sm font-medium text-foreground">
                Step 4: Create the noisy image
              </p>
              <div className="py-2 px-4 bg-muted/50 rounded-lg">
                <BlockMath math="x_{500} = \sqrt{0.05} \cdot x_0 + \sqrt{0.95} \cdot \epsilon = 0.224 \cdot x_0 + 0.975 \cdot \epsilon" />
              </div>
              <p className="text-sm text-muted-foreground">
                At <InlineMath math="t = 500" />, the noisy image is mostly
                noise (97.5% noise, 22.4% signal). You can barely see the
                T-shirt.
              </p>
            </div>

            <div className="rounded-lg bg-muted/30 p-4 space-y-3">
              <p className="text-sm font-medium text-foreground">
                Steps 5&ndash;6: Predict and compute loss
              </p>
              <p className="text-sm text-muted-foreground">
                The network receives <InlineMath math="x_{500}" /> and{' '}
                <InlineMath math="t = 500" />. It outputs{' '}
                <InlineMath math="\epsilon_\theta" />&mdash;its guess of the
                784-dimensional noise vector. We compute:
              </p>
              <div className="py-2 px-4 bg-muted/50 rounded-lg">
                <BlockMath math="L = \frac{1}{784} \sum_{i=1}^{784} (\epsilon_i - \epsilon_{\theta,i})^2" />
              </div>
              <p className="text-sm text-muted-foreground">
                Average squared error across all 784 pixels. The same formula,
                the same code, as every MSE computation you have done before.
              </p>
            </div>

            <div className="rounded-lg bg-violet-500/5 border border-violet-500/20 p-4 space-y-3">
              <p className="text-sm font-medium text-foreground">
                What if <InlineMath math="t = 50" /> instead?
              </p>
              <p className="text-sm text-muted-foreground">
                At <InlineMath math="t = 50" />,{' '}
                <InlineMath math="\bar{\alpha}_{50} \approx 0.95" />&mdash;the
                mirror image of{' '}
                <InlineMath math="t = 500" />. The coefficients swap roles: now
                the signal coefficient is 0.975 and the noise coefficient is
                only 0.224. The T-shirt is clearly visible with a light dusting
                of static. The network&rsquo;s job is easy&mdash;detect subtle
                perturbations.
              </p>
              <p className="text-sm text-muted-foreground">
                At <InlineMath math="t = 950" />,{' '}
                <InlineMath math="\bar{\alpha}_{950} \approx 0.001" />. The
                image is essentially pure noise. The network must hallucinate
                plausible structure from almost nothing&mdash;a much harder task.
              </p>
              <p className="text-sm text-muted-foreground">
                Same algorithm, same loss, same code. The difficulty varies
                entirely with the timestep. This connects to the multi-scale
                denoising from <LessonLink slug="the-diffusion-idea">The Diffusion Idea</LessonLink>: high noise
                levels require structural decisions, low noise levels require
                fine detail.
              </p>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="The Signal-to-Noise Dial">
            Remember the alpha-bar curve from The Forward Process? The coefficients 0.224 and 0.975 are
            exactly what the widget showed at{' '}
            <InlineMath math="t = 500" />. The training algorithm uses the
            same formula at every iteration&mdash;just with a different random
            dial position each time.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 11: Interactive Widget — Training Step Simulator
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Explore: One Training Step"
            subtitle="Pick a timestep and trace the algorithm"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Drag the slider to choose a timestep. Watch the noisy image change,
              see what the network would predict, and observe how MSE loss varies
              with noise level.
            </p>
          </div>
          <ExercisePanel
            title="Training Step Simulator"
            subtitle="Pick a timestep and see the complete training step"
          >
            <TrainingStepSimulator />
          </ExercisePanel>
        </Row.Content>
        <Row.Aside>
          <TryThisBlock title="Experiment">
            <ul className="space-y-2 text-sm">
              <li>
                &bull; Try <strong>t&nbsp;=&nbsp;50</strong>: the image is barely noisy.
                How easy is it for the network to predict the noise?
              </li>
              <li>
                &bull; Try <strong>t&nbsp;=&nbsp;950</strong>: nearly pure static.
                The network must hallucinate structure.
              </li>
              <li>
                &bull; Watch the MSE loss: it rises as the task gets harder.
              </li>
              <li>
                &bull; Compare the predicted vs actual noise images&mdash;at low
                timesteps they look nearly identical.
              </li>
            </ul>
          </TryThisBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 12: Check — Trace a Training Step (Outline item 7)
          ================================================================ */}
      <Row>
        <Row.Content>
          <GradientCard title="Trace a Training Step" color="cyan">
            <div className="space-y-3 text-sm">
              <p>
                <strong>Scenario:</strong> Your training image is a sneaker.
                The random timestep is <InlineMath math="t = 200" />, where{' '}
                <InlineMath math="\bar{\alpha}_{200} = 0.50" />.
              </p>
              <p>
                Describe each step of the algorithm. Then: what would change
                if <InlineMath math="t = 900" /> instead?
              </p>
              <details className="mt-3">
                <summary className="font-medium cursor-pointer text-primary">
                  Reveal
                </summary>
                <div className="mt-2 space-y-2">
                  <p>
                    <strong>At <InlineMath math="t = 200" />:</strong> Sample{' '}
                    <InlineMath math="\epsilon \sim \mathcal{N}(0, I)" />.
                    Compute{' '}
                    <InlineMath math="x_{200} = \sqrt{0.5} \cdot x_0 + \sqrt{0.5} \cdot \epsilon = 0.707 \cdot x_0 + 0.707 \cdot \epsilon" />.
                    Equal parts signal and noise&mdash;the sneaker is visible
                    but noisy. Network predicts{' '}
                    <InlineMath math="\epsilon_\theta" />, compute MSE,
                    backprop, update.
                  </p>
                  <p>
                    <strong>At <InlineMath math="t = 900" />:</strong> The noisy
                    image is almost pure static. The network must infer that
                    there was a sneaker there at all. The task is vastly harder.
                    But the algorithm is identical&mdash;same seven steps, same
                    loss formula.
                  </p>
                </div>
              </details>
            </div>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 13: Three Faces of MSE (Outline item 8)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Three Faces of MSE"
            subtitle="Same formula, three different questions"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              You have now seen MSE loss in three completely different contexts.
              The formula has not changed. The code has not changed. Only the
              question has changed.
            </p>

            <div className="overflow-x-auto">
              <table className="w-full text-sm border-collapse">
                <thead>
                  <tr className="border-b border-border">
                    <th className="text-left py-2 pr-4 text-foreground font-medium">Context</th>
                    <th className="text-left py-2 pr-4 text-foreground font-medium">Prediction</th>
                    <th className="text-left py-2 pr-4 text-foreground font-medium">Target</th>
                    <th className="text-left py-2 text-foreground font-medium">Source</th>
                  </tr>
                </thead>
                <tbody className="text-muted-foreground">
                  <tr className="border-b border-border/50">
                    <td className="py-2 pr-4">Linear regression</td>
                    <td className="py-2 pr-4">
                      <InlineMath math="\hat{y}" /> (predicted price)
                    </td>
                    <td className="py-2 pr-4">
                      <InlineMath math="y" /> (actual price)
                    </td>
                    <td className="py-2">Series 1</td>
                  </tr>
                  <tr className="border-b border-border/50">
                    <td className="py-2 pr-4">Autoencoder</td>
                    <td className="py-2 pr-4">
                      <InlineMath math="\hat{x}" /> (reconstructed image)
                    </td>
                    <td className="py-2 pr-4">
                      <InlineMath math="x" /> (input image)
                    </td>
                    <td className="py-2">Module 6.1</td>
                  </tr>
                  <tr>
                    <td className="py-2 pr-4">DDPM</td>
                    <td className="py-2 pr-4">
                      <InlineMath math="\epsilon_\theta" /> (predicted noise)
                    </td>
                    <td className="py-2 pr-4">
                      <InlineMath math="\epsilon" /> (actual noise)
                    </td>
                    <td className="py-2">This lesson</td>
                  </tr>
                </tbody>
              </table>
            </div>

            <p className="text-muted-foreground">
              Write them side by side. The formulas are identical:
            </p>

            <div className="space-y-2">
              <div className="py-2 px-4 bg-muted/50 rounded-lg">
                <BlockMath math="\text{Regression: } L = \frac{1}{n}\sum(\hat{y}_i - y_i)^2" />
              </div>
              <div className="py-2 px-4 bg-muted/50 rounded-lg">
                <BlockMath math="\text{Autoencoder: } L = \frac{1}{n}\sum(\hat{x}_i - x_i)^2" />
              </div>
              <div className="py-2 px-4 bg-muted/50 rounded-lg">
                <BlockMath math="\text{DDPM: } L = \frac{1}{n}\sum(\epsilon_{\theta,i} - \epsilon_i)^2" />
              </div>
            </div>

            <p className="text-muted-foreground">
              Same formula. Same gradients. Same PyTorch code:{' '}
              <code className="text-xs bg-muted px-1 py-0.5 rounded">nn.MSELoss()</code>.
              The only difference is what you plug in for &ldquo;prediction&rdquo;
              and &ldquo;target.&rdquo;
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Same Building Blocks, Different Question">
            In <LessonLink slug="the-diffusion-idea">The Diffusion Idea</LessonLink>, you learned that diffusion
            uses the same building blocks as everything else&mdash;conv layers,
            MSE loss, backprop. Here is the payoff: the training objective IS
            MSE loss. The question is &ldquo;what noise was added?&rdquo;
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 14: One Network, All Timesteps (Outline item 8 cont.)
          ================================================================ */}
      <Row>
        <Row.Content>
          <GradientCard title="One Network for All Timesteps" color="blue">
            <div className="space-y-3 text-sm">
              <p>
                You might wonder: with 1,000 different noise levels, do you
                need 1,000 different networks? <strong>No.</strong> One network
                handles all timesteps. It receives the timestep{' '}
                <InlineMath math="t" /> as an additional input alongside the
                noisy image.
              </p>
              <p>
                At <InlineMath math="t = 50" />, it has learned to detect
                subtle noise in nearly clean images. At{' '}
                <InlineMath math="t = 900" />, it has learned to hallucinate
                plausible structure from near-pure static. The same network,
                conditioned on the timestep.
              </p>
              <p className="text-muted-foreground/80 italic">
                How the network uses the timestep (the conditioning mechanism)
                is an architecture question. You will see the details in the
                U-Net lesson. For now, accept that it works&mdash;the network
                receives <InlineMath math="t" /> and adjusts its behavior
                accordingly.
              </p>
            </div>
          </GradientCard>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Architecture Is Later">
            The neural network is a black box in this lesson. You know its
            inputs (<InlineMath math="x_t" /> and{' '}
            <InlineMath math="t" />) and its output (
            <InlineMath math="\epsilon_\theta" />). How it works internally is
            Module 6.3.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 15: Simplified vs Full Loss (Outline item 8 cont.)
          ================================================================ */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              One more detail worth mentioning. The loss we wrote&mdash;
              <InlineMath math="\| \epsilon - \epsilon_\theta \|^2" />&mdash;is
              the <strong>simplified loss</strong> from the DDPM paper. The
              paper derives it from a variational lower bound (the same ELBO
              technique from the VAE). The full derivation produces a loss with
              per-timestep weighting terms. Ho et al. found that dropping those
              weights and using the simple, unweighted MSE works better
              empirically.
            </p>
            <p className="text-muted-foreground">
              You do not need the derivation. The simplified loss is what
              practitioners use. The important point: even the &ldquo;theoretical&rdquo;
              version of the loss is MSE at its core. The simplification just
              makes it cleaner.
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 16: Transfer Check (Outline item 9)
          ================================================================ */}
      <Row>
        <Row.Content>
          <GradientCard title="Transfer Check" color="cyan">
            <div className="space-y-3 text-sm">
              <p>
                You are explaining DDPM to a friend who knows the VAE from{' '}
                <strong>Variational Autoencoders</strong>. They ask: &ldquo;So
                the VAE reconstructs images using MSE, and DDPM predicts noise
                using MSE&mdash;what is the fundamental difference?&rdquo;
              </p>
              <details className="mt-3">
                <summary className="font-medium cursor-pointer text-primary">
                  Reveal
                </summary>
                <div className="mt-2 space-y-2">
                  <p>
                    The VAE compresses the input through a bottleneck and tries
                    to reconstruct it. Its target is the input image itself.
                    DDPM adds noise at a specific level and tries to predict what
                    was added. Its target is the noise, not the image.
                  </p>
                  <p>
                    The VAE learns a latent space (a compressed representation).
                    DDPM learns a denoising function (how to remove noise at any
                    level). Same loss formula, completely different learning
                    objectives. The VAE generates by sampling a latent code;
                    DDPM generates by iteratively denoising pure noise.
                  </p>
                </div>
              </details>
            </div>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 17: Practice — Notebook Exercises (Outline item 10)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Practice"
            subtitle="Hands-on exercises in a Colab notebook"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Reinforce the concepts by working through guided exercises: create
              noisy images using the closed-form formula, compute MSE loss on
              noise predictions, and write the training pseudocode.
            </p>
            <div className="rounded-lg border-2 border-primary/50 bg-primary/5 p-6">
              <div className="space-y-4">
                <p className="text-muted-foreground">
                  Open the notebook and work through the exercises at your own
                  pace. Each exercise builds on the patterns before it, but you
                  can skip ahead if you are comfortable with the code patterns.
                </p>
                <a
                  href="https://colab.research.google.com/github/brennancheung/CourseAI/blob/main/notebooks/6-2-3-learning-to-denoise.ipynb"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-2 px-4 py-2 rounded-md bg-primary text-primary-foreground font-medium text-sm hover:bg-primary/90 transition-colors"
                >
                  <ExternalLink className="w-4 h-4" />
                  Open in Google Colab
                </a>
                <p className="text-xs text-muted-foreground">
                  Exercises: create noisy images at various timesteps, compute
                  MSE loss by hand, write training pseudocode, and predict the
                  loss landscape.
                </p>
              </div>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="No Implementation Yet">
            These exercises reinforce the concepts from this lesson. Full model
            implementation comes in the capstone lesson.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 18: Summary (Outline item 11)

          ================================================================ */}
      <Row>
        <Row.Content>
          <SummaryBlock
            title="Key Takeaways"
            items={[
              {
                headline:
                  'The DDPM training objective is MSE loss between predicted and actual noise.',
                description:
                  'The same loss formula from Series 1, the same nn.MSELoss in PyTorch. No new math was invented for the loss function.',
              },
              {
                headline:
                  'The model predicts noise (ε), not the clean image.',
                description:
                  'Noise is a consistent target (always from N(0,1)) regardless of image content. You can recover x₀ algebraically—nothing is lost.',
              },
              {
                headline:
                  'Training samples one random timestep per image.',
                description:
                  'The closed-form formula teleports to that timestep. No iterating through all 1,000 steps. Each batch mixes different noise levels.',
              },
              {
                headline:
                  'The training loop is the standard loop with diffusion-specific data preparation.',
                description:
                  'Steps 1–4 prepare the data (sample image, timestep, noise, create noisy image). Steps 5–7 are the familiar forward-loss-backward-update cycle.',
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
              Same building blocks, different question. The building blocks: MSE
              loss, backprop, gradient descent. The question: &ldquo;what noise
              was added to this image?&rdquo;
            </strong>
          </InsightBlock>
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 19: Next Step (Outline item 12)
          ================================================================ */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              You now know how to train the model. But how do you use it?
              Starting from pure noise, how does the trained network iteratively
              denoise to create an image? That is the reverse process&mdash;the
              sampling algorithm&mdash;and it is the next lesson.
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
                note: 'Sections 3-4 derive the simplified noise-prediction objective, showing it produces better results than predicting x₀ directly.',
              },
            ]}
          />
        </Row.Content>
      </Row>

      <Row>
        <Row.Content>
          <NextStepBlock
            href="/app"
            title="Up Next: Sampling and Generation"
            description="Starting from pure noise, the trained network iteratively denoises to create an image. Trace the reverse process step by step."
          />
        </Row.Content>
      </Row>
    </LessonLayout>
  )
}

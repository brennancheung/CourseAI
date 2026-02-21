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
import { ExternalLink } from 'lucide-react'
import 'katex/dist/katex.min.css'
import { InlineMath, BlockMath } from 'react-katex'

const NOTEBOOK_URL =
  'https://colab.research.google.com/github/brennancheung/CourseAI/blob/main/notebooks/7-2-1-score-functions-and-sdes.ipynb'

/**
 * Score Functions & SDEs
 *
 * Lesson 1 in Module 7.2 (The Score-Based Perspective). Lesson 4 of 11 in Series 7.
 * Cognitive load: STRETCH (2-3 new concepts).
 *
 * Core concepts:
 * 1. Score function as gradient of log probability (NEW)
 * 2. SDE forward process as continuous-time generalization of DDPM (NEW)
 * 3. Probability flow ODE formalized (DEEPENED from MENTIONED to INTRODUCED)
 *
 * Builds on: gradients from 1.1.4, DDPM forward/reverse from 6.2.2-6.2.4,
 * ODE perspective from 6.4.2, log probabilities from Series 4.
 *
 * Previous: IP-Adapter (Module 7.1, Lesson 3 / BUILD)
 * Next: Flow Matching (Module 7.2, Lesson 2)
 */

export function ScoreFunctionsAndSDEsLesson() {
  return (
    <LessonLayout>
      {/* Section 1: Header */}
      <Row>
        <Row.Content>
          <LessonHeader
            title="Score Functions & SDEs"
            description="What your noise prediction network has been secretly learning all along—the score function—and how it connects DDPM to a continuous-time SDE/ODE framework."
            category="The Score-Based Perspective"
          />
        </Row.Content>
      </Row>

      {/* Section 2: Objective + Constraints */}
      <Row>
        <Row.Content>
          <ObjectiveBlock>
            Understand the score function (gradient of log probability) as the
            concept that unifies DDPM&rsquo;s noise prediction with a
            continuous-time SDE/ODE framework. See that the noise prediction
            network you already know IS a scaled version of the score function,
            that the DDPM forward process generalizes to a continuous SDE, and
            that the probability flow ODE from the sampler lesson now has a
            proper name and derivation.
          </ObjectiveBlock>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Theory Lesson">
            This is the most theoretical lesson in Series 7. No new tools or
            models to run. Every concept connects to things you already have:
            gradients, DDPM, ODE solvers. By the end, you will see what was
            hiding inside DDPM all along.
          </TipBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <ConstraintBlock
            title="Scope for This Lesson"
            items={[
              'The score function: what it is, geometric intuition, concrete examples',
              'The equivalence between noise prediction and the score function',
              'The SDE forward process as a continuous generalization of DDPM',
              'The reverse SDE and probability flow ODE (at intuition level)',
              'Why this perspective matters for understanding modern diffusion models',
              'NOT: Ito calculus, stochastic integration, or Fokker-Planck equations',
              'NOT: Score matching training objective (DDPM training already does this implicitly)',
              'NOT: Flow matching (next lesson)',
              'NOT: Any implementation or coding—this is a conceptual/theoretical lesson',
            ]}
          />
        </Row.Content>
      </Row>

      {/* Section 3: Recap */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Quick Recap"
            subtitle="Three concepts you will need in the next fifteen minutes"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              <strong>Gradients.</strong> In{' '}
              <strong>Gradient Descent</strong>, you learned that the
              gradient points in the direction of steepest increase of a
              function. You used{' '}
              <InlineMath math="\nabla_\theta L" /> to find the direction
              that increases loss the most, then went the opposite way.
              The gradient is a direction at every point in space.
            </p>
            <p className="text-muted-foreground">
              <strong>The ODE perspective.</strong> In{' '}
              <strong>Samplers and Efficiency</strong>, you learned that
              DDIM is approximately Euler&rsquo;s method on the diffusion ODE.
              The model&rsquo;s noise predictions define a trajectory from noise
              to data. Different samplers follow that trajectory with different
              step sizes.
            </p>
            <p className="text-muted-foreground">
              <strong>DDPM noise prediction.</strong> In{' '}
              <strong>Learning to Denoise</strong>, you learned that the
              model takes{' '}
              <InlineMath math="x_t" /> and <InlineMath math="t" />, and
              predicts the noise <InlineMath math="\epsilon" /> that was
              added. The training loss is{' '}
              <InlineMath math="\|\epsilon - \epsilon_\theta(x_t, t)\|^2" />.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="Key Bridge">
            These three concepts are about to converge. The gradient tells you
            a direction. The noise prediction tells you a direction. The ODE
            defines a trajectory from those directions. They are all the same
            thing, seen from different angles.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* Section 4: Hook */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="What Does the Network Really Know?"
            subtitle="A question that changes how you see diffusion"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              We said the model predicts noise. That is the training
              objective. But consider what it actually does: at any point{' '}
              <InlineMath math="x_t" /> in noisy image space, the network
              tells you <strong>which direction to move</strong> to get
              closer to a clean image. It does this for{' '}
              <strong>every possible noisy image</strong>, at{' '}
              <strong>every noise level</strong>.
            </p>
            <p className="text-muted-foreground">
              That is not just noise removal. That is a map of the entire
              noisy data distribution. A map that says, at every point:{' '}
              <em>&ldquo;go this way for higher probability.&rdquo;</em>
            </p>
            <GradientCard title="The Hidden Identity" color="violet">
              <p className="text-sm">
                The noise prediction{' '}
                <InlineMath math="\epsilon_\theta" /> was always a disguised
                version of something more fundamental. Let&rsquo;s find out
                what.
              </p>
            </GradientCard>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Reframing">
            You have been thinking of the model as answering: &ldquo;what
            noise was added?&rdquo; The deeper answer is: &ldquo;which
            direction is the data?&rdquo;
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Section 5: Explain — Part A: What is the score function? */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Score Function"
            subtitle="A compass toward likely data"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Imagine you have a probability distribution{' '}
              <InlineMath math="p(x)" />—say, the distribution of all
              natural images. You are at some point <InlineMath math="x" />{' '}
              in this space (maybe a noisy image, maybe a random point). You
              want to know: <strong>which direction should I move to get to
              a more probable point?</strong> Not &ldquo;how probable is this
              point?&rdquo; but &ldquo;which way is up?&rdquo;
            </p>
            <p className="text-muted-foreground">
              The answer is the <strong>score function</strong>:
            </p>
            <div className="py-4 px-6 bg-muted/50 rounded-lg">
              <BlockMath math="\text{score}(x) = \nabla_x \log p(x)" />
            </div>
            <p className="text-muted-foreground">
              The gradient of the log probability density with respect
              to <InlineMath math="x" />. At every point in data space, it
              gives you a direction vector pointing toward higher
              probability.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Same Operation, Different Space">
            You have been taking gradients since{' '}
            <strong>Gradient Descent</strong>. The score function is the
            same mathematical operation—just applied to data space instead
            of parameter space.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Warning: Not the gradient you are used to */}
      <Row>
        <Row.Content>
          <WarningBlock title="Not the Gradient You Are Used To">
            <div className="space-y-3">
              <p>
                The score function is{' '}
                <InlineMath math="\nabla_x \log p(x)" />, <strong>not</strong>{' '}
                <InlineMath math="\nabla_\theta L(\theta)" />. This is the
                gradient with respect to <strong>data</strong>, not{' '}
                <strong>parameters</strong>.
              </p>
              <div className="overflow-x-auto">
                <table className="w-full text-sm border-collapse">
                  <thead>
                    <tr className="border-b border-muted">
                      <th className="text-left py-2 pr-4 text-muted-foreground font-medium" />
                      <th className="text-left py-2 pr-4 text-muted-foreground font-medium">
                        Optimization
                      </th>
                      <th className="text-left py-2 text-muted-foreground font-medium">
                        Score Function
                      </th>
                    </tr>
                  </thead>
                  <tbody className="text-muted-foreground">
                    <tr className="border-b border-muted/50">
                      <td className="py-2 pr-4 font-medium">What changes</td>
                      <td className="py-2 pr-4">
                        Parameters <InlineMath math="\theta" />
                      </td>
                      <td className="py-2">
                        Data point <InlineMath math="x" />
                      </td>
                    </tr>
                    <tr className="border-b border-muted/50">
                      <td className="py-2 pr-4 font-medium">Function</td>
                      <td className="py-2 pr-4">
                        Loss <InlineMath math="L(\theta)" />
                      </td>
                      <td className="py-2">
                        Log probability <InlineMath math="\log p(x)" />
                      </td>
                    </tr>
                    <tr>
                      <td className="py-2 pr-4 font-medium">Direction</td>
                      <td className="py-2 pr-4">Toward lower loss</td>
                      <td className="py-2">Toward higher probability</td>
                    </tr>
                  </tbody>
                </table>
              </div>
              <p>
                In optimization, you ask &ldquo;which way should I move the{' '}
                <strong>weights</strong>?&rdquo; The score function asks
                &ldquo;which way should I move the{' '}
                <strong>image</strong>?&rdquo;
              </p>
            </div>
          </WarningBlock>
        </Row.Content>
      </Row>

      {/* Section 5 Part B: 1D Gaussian concrete example */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Concrete Example: 1D Gaussian"
            subtitle="Computing the score function by hand"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Take the simplest distribution:{' '}
              <InlineMath math="p(x) = \mathcal{N}(0, 1)" />. Write out the
              log probability:
            </p>
            <div className="py-4 px-6 bg-muted/50 rounded-lg">
              <BlockMath math="\log p(x) = -\frac{x^2}{2} + \text{const}" />
            </div>
            <p className="text-muted-foreground">
              Take the derivative with respect to <InlineMath math="x" />:
            </p>
            <div className="py-4 px-6 bg-muted/50 rounded-lg">
              <BlockMath math="\text{score}(x) = \frac{d}{dx}\log p(x) = -x" />
            </div>
            <p className="text-muted-foreground">
              That is it. For a standard Gaussian, the score function is just{' '}
              <InlineMath math="-x" />. Now work through specific values:
            </p>
            <ul className="list-disc list-inside text-muted-foreground space-y-2 ml-4">
              <li>
                At <InlineMath math="x = 3" /> (far from mean): score ={' '}
                <InlineMath math="-3" />. <strong>Strong pull toward center.</strong>
              </li>
              <li>
                At <InlineMath math="x = 1" /> (near mean): score ={' '}
                <InlineMath math="-1" />. Gentle pull toward center.
              </li>
              <li>
                At <InlineMath math="x = 0" /> (at mean): score ={' '}
                <InlineMath math="0" />. Already at the peak—no direction
                preferred.
              </li>
              <li>
                At <InlineMath math="x = -2" /> (other side): score ={' '}
                <InlineMath math="+2" />. Pull toward center from the left.
              </li>
            </ul>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Compass Toward Data">
            The score function is always pointing toward the peak of the
            distribution. It is a compass toward likely data. Further from
            the peak, the pull is stronger. At the peak, the compass reads
            zero—you have arrived.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Negative example: score at the peak */}
      <Row>
        <Row.Content>
          <GradientCard title="Boundary: Score at the Peak" color="rose">
            <div className="space-y-3 text-sm">
              <p>
                At <InlineMath math="x = 0" />, the score is zero—even though
                the probability is <strong>highest</strong> there. The score
                tells you <strong>direction</strong>, not{' '}
                <strong>value</strong>.
              </p>
              <p>
                A score of zero means &ldquo;you are at a peak,&rdquo; not
                &ldquo;the probability is zero.&rdquo;
              </p>
              <p className="text-muted-foreground">
                This parallels gradient = 0 at a minimum in optimization. Zero
                gradient does not mean &ldquo;the loss is zero.&rdquo; It means
                &ldquo;you are at an extremum—no better direction to go.&rdquo;
              </p>
            </div>
          </GradientCard>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Direction, Not Value">
            The score function tells you WHERE to go, not HOW LIKELY you are.
            To compare probabilities of two points, you would need the density
            itself—the score only gives the slope.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* Section 5 Part C: 2D vector field */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Score as a Vector Field"
            subtitle="Extending to 2D: arrows everywhere pointing toward data"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              In 1D, the score is a number (positive or negative). In 2D,
              the score at every point is a <strong>2D vector</strong>—an
              arrow with both direction and magnitude. For a distribution
              with two peaks, the score field is a sea of arrows, each
              pointing toward the nearest region of high probability.
            </p>
            <p className="text-muted-foreground">
              Picture a 2D Gaussian mixture with two peaks—one at{' '}
              <InlineMath math="(-3, 0)" /> and one at{' '}
              <InlineMath math="(3, 0)" />. Here is what the score field
              looks like at three different noise levels:
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Arrows Everywhere">
            The score field assigns a direction vector to every point in
            space. It is the complete map of &ldquo;which way is higher
            probability?&rdquo;—at every location simultaneously.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Score field visualization: 3 panels at different noise levels */}
      <Row>
        <Row.Content>
          <div className="grid gap-4 md:grid-cols-3">
            <GradientCard title="No Noise (Clean Data)" color="violet">
              <div className="space-y-2 text-sm">
                <p className="font-medium text-violet-300/90">
                  Two sharp peaks, two basins
                </p>
                <p className="text-muted-foreground">
                  Arrows near <InlineMath math="(-3, 0)" /> all converge
                  tightly on that peak. Arrows near{' '}
                  <InlineMath math="(3, 0)" /> converge on the other. At
                  each peak center, the arrows vanish—score is zero.
                </p>
                <p className="text-muted-foreground">
                  Between the peaks (around the origin), the field
                  splits: arrows on the left side point left, arrows on
                  the right point right. A sharp boundary divides the
                  two basins.
                </p>
                <p className="text-muted-foreground">
                  Far from both peaks, arrows are long and
                  strong—the score magnitude is large. This is a complex,
                  detailed field.
                </p>
              </div>
            </GradientCard>
            <GradientCard title="Moderate Noise" color="blue">
              <div className="space-y-2 text-sm">
                <p className="font-medium text-blue-300/90">
                  Softer peaks, blurred boundary
                </p>
                <p className="text-muted-foreground">
                  The two basins are still visible, but their edges blur.
                  Arrows near each peak still converge inward, but less
                  sharply—the peaks have spread out.
                </p>
                <p className="text-muted-foreground">
                  The boundary between basins is no longer a crisp
                  dividing line. Near the origin, arrows are shorter
                  and less decisive—the field is &ldquo;confused&rdquo;
                  about which peak to point toward.
                </p>
                <p className="text-muted-foreground">
                  Overall magnitude is lower. The landscape has been
                  smoothed by noise.
                </p>
              </div>
            </GradientCard>
            <GradientCard title="High Noise (Nearly Pure Noise)" color="sky">
              <div className="space-y-2 text-sm">
                <p className="font-medium text-sky-300/90">
                  One broad basin, nearly uniform
                </p>
                <p className="text-muted-foreground">
                  The two peaks have merged into a single broad
                  mound centered at the origin. Every arrow points
                  roughly toward{' '}
                  <InlineMath math="(0, 0)" />—the combined center
                  of mass.
                </p>
                <p className="text-muted-foreground">
                  The field looks almost like a single Gaussian&rsquo;s
                  score: <InlineMath math="\text{score}(x) \approx -x" />,
                  arrows radiating inward from all directions.
                </p>
                <p className="text-muted-foreground">
                  Simple, smooth, nearly linear. This is what the model
                  sees at high noise levels—an easy target.
                </p>
              </div>
            </GradientCard>
          </div>
          <p className="text-sm text-muted-foreground mt-4">
            The notebook (Exercise 2) generates the actual quiver plots for
            this exact scenario. The progression above is what you will
            see: noise smooths the score field from complex to simple.
          </p>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="This Is What the Model Sees">
            At high noise, the model&rsquo;s job is easy—the noisy
            distribution is nearly Gaussian, and the score field is simple.
            At low noise, the model must capture the fine structure of the
            data distribution. This is why denoising starts from easy tasks
            (large timesteps, high noise) and progresses to hard tasks
            (small timesteps, low noise).
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Connect to alpha-bar */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Connect this to the noise schedule you already know. When{' '}
              <InlineMath math="\bar\alpha" /> is near 0 (high noise), the
              noisy distribution{' '}
              <InlineMath math="q(x_t)" /> is close to{' '}
              <InlineMath math="\mathcal{N}(0, I)" /> and the score field
              is nearly linear. When{' '}
              <InlineMath math="\bar\alpha" /> is near 1 (low noise),{' '}
              <InlineMath math="q(x_t)" /> is close to the data distribution
              and the score field is complex.
            </p>
            <p className="text-muted-foreground">
              This is exactly{' '}
              <strong>why the noise schedule matters</strong>. The model
              learns the score at every noise level. The schedule determines
              how the difficulty ramps up.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="Alpha-Bar Revisited">
            <InlineMath math="\bar\alpha \approx 0" />: nearly pure noise,
            simple score field.{' '}
            <InlineMath math="\bar\alpha \approx 1" />: nearly clean data,
            complex score field. Same{' '}
            <InlineMath math="\bar\alpha" /> curve from{' '}
            <strong>The Forward Process</strong>, new interpretation.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* Section 6: Check #1 */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Check Your Understanding"
            subtitle="Predict before you read the answers"
          />
          <div className="space-y-4">
            <GradientCard title="Question 1" color="cyan">
              <div className="space-y-2 text-sm">
                <p>
                  For a 1D Gaussian{' '}
                  <InlineMath math="\mathcal{N}(5, 1)" />, what is the score
                  function? What is its value at{' '}
                  <InlineMath math="x = 7" />?
                </p>
                <details className="mt-2">
                  <summary className="font-medium cursor-pointer text-primary">
                    Reveal
                  </summary>
                  <p className="mt-2 text-muted-foreground">
                    <InlineMath math="\text{score}(x) = -(x - 5)" />, so{' '}
                    <InlineMath math="\text{score}(7) = -2" />. The score
                    points toward the mean at{' '}
                    <InlineMath math="x = 5" /> with magnitude 2, pulling
                    the point back.
                  </p>
                </details>
              </div>
            </GradientCard>

            <GradientCard title="Question 2" color="cyan">
              <div className="space-y-2 text-sm">
                <p>
                  Two points:{' '}
                  <InlineMath math="x_A" /> has score = <InlineMath math="-5" />{' '}
                  and <InlineMath math="x_B" /> has score ={' '}
                  <InlineMath math="-0.1" />. Which is further from the peak?
                </p>
                <details className="mt-2">
                  <summary className="font-medium cursor-pointer text-primary">
                    Reveal
                  </summary>
                  <p className="mt-2 text-muted-foreground">
                    <InlineMath math="x_A" />. Larger score magnitude means
                    steeper probability gradient—the point is further from a
                    peak. <InlineMath math="x_B" /> with score{' '}
                    <InlineMath math="-0.1" /> is nearly at a peak.
                  </p>
                </details>
              </div>
            </GradientCard>

            <GradientCard title="Question 3" color="cyan">
              <div className="space-y-2 text-sm">
                <p>
                  Can the score function tell you whether{' '}
                  <InlineMath math="p(x_A) > p(x_B)" />?
                </p>
                <details className="mt-2">
                  <summary className="font-medium cursor-pointer text-primary">
                    Reveal
                  </summary>
                  <p className="mt-2 text-muted-foreground">
                    No. The score tells you the direction at each point, not the
                    absolute value of the density. You would need to integrate
                    to compare densities. The score is a derivative—it says
                    &ldquo;which way is up&rdquo; at each location, not
                    &ldquo;how high am I.&rdquo;
                  </p>
                </details>
              </div>
            </GradientCard>
          </div>
        </Row.Content>
      </Row>

      {/* Section 7: The Score-Noise Equivalence (The Reveal) */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Score-Noise Equivalence"
            subtitle="The key insight of this entire lesson"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Here is the connection that ties everything together. The DDPM
              noise prediction{' '}
              <InlineMath math="\epsilon_\theta(x_t, t)" /> has this
              relationship to the score function:
            </p>
            <div className="py-4 px-6 bg-muted/50 rounded-lg space-y-2">
              <BlockMath math="\epsilon_\theta(x_t, t) \approx -\sqrt{1 - \bar\alpha_t} \;\nabla_{x_t} \log q(x_t)" />
            </div>
            <p className="text-muted-foreground">
              Or equivalently, solving for the score:
            </p>
            <div className="py-4 px-6 bg-muted/50 rounded-lg">
              <BlockMath math="\text{score}(x_t, t) = \nabla_{x_t} \log q(x_t) \approx -\frac{\epsilon_\theta(x_t, t)}{\sqrt{1 - \bar\alpha_t}}" />
            </div>
            <p className="text-muted-foreground">
              The noise prediction <strong>IS</strong> a scaled version of the
              score function.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The Reveal">
            The noise prediction network has been learning the score function
            all along. Every DDPM model you have ever seen is a score-based
            model in disguise.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Walk through the intuition */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Walk through the intuition:
            </p>
            <ul className="list-disc list-inside text-muted-foreground space-y-2 ml-4">
              <li>
                The <strong>score</strong> says: &ldquo;to increase the
                probability of{' '}
                <InlineMath math="x_t" />, move in this direction.&rdquo;
              </li>
              <li>
                The <strong>noise prediction</strong> says: &ldquo;the noise
                that was added to reach{' '}
                <InlineMath math="x_t" /> was in this direction.&rdquo;
              </li>
              <li>
                These are the <strong>same direction</strong> (up to a sign
                flip and scaling). Moving toward higher probability means
                undoing the noise. The noise points{' '}
                <strong>away from</strong> the data; the score points{' '}
                <strong>toward</strong> the data.
              </li>
            </ul>
            <p className="text-muted-foreground">
              Let&rsquo;s see this concretely. The DDPM reverse step removes
              noise with this term (from <strong>Sampling and
              Generation</strong>):
            </p>
            <div className="py-3 px-6 bg-muted/50 rounded-lg space-y-2">
              <p className="text-xs text-muted-foreground mb-1">
                Start: the noise removal term from DDPM&rsquo;s reverse step
              </p>
              <BlockMath math="\frac{\beta_t}{\sqrt{1 - \bar\alpha_t}}\;\epsilon_\theta(x_t, t)" />
              <p className="text-xs text-muted-foreground mb-1">
                Substitute{' '}
                <InlineMath math="\epsilon_\theta \approx -\sqrt{1 - \bar\alpha_t}\;\nabla_{x_t}\!\log q(x_t)" />:
              </p>
              <BlockMath math="= \frac{\beta_t}{\sqrt{1 - \bar\alpha_t}} \cdot \bigl(-\sqrt{1 - \bar\alpha_t}\bigr)\;\nabla_{x_t}\!\log q(x_t)" />
              <p className="text-xs text-muted-foreground mb-1">
                The <InlineMath math="\sqrt{1 - \bar\alpha_t}" /> cancels:
              </p>
              <BlockMath math="= -\beta_t\;\nabla_{x_t}\!\log q(x_t) \;=\; -\beta_t\;\text{score}(x_t, t)" />
            </div>
            <p className="text-muted-foreground">
              The noise removal term was always a score-guided step. The{' '}
              <InlineMath math="\sqrt{1-\bar\alpha_t}" /> factors cancel,
              leaving the score function multiplied by the noise schedule.
              The model was always telling you the direction toward higher
              probability—just wearing a different hat.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="Opposite Arrows">
            Noise points away from data. The score points toward data.{' '}
            <InlineMath math="\epsilon_\theta \approx -\sqrt{1 - \bar\alpha_t} \cdot \text{score}" />.
            The negative sign is the direction flip. The{' '}
            <InlineMath math="\sqrt{1 - \bar\alpha_t}" /> is the scale.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* Address misconception #1 */}
      <Row>
        <Row.Content>
          <GradientCard title="Same Framework, Different Lens" color="emerald">
            <div className="space-y-2 text-sm">
              <p>
                This is <strong>not</strong> a different framework.
                Score-based diffusion and DDPM are two descriptions of the
                same process. The score was hiding inside DDPM all along.
              </p>
              <p className="text-muted-foreground">
                You do not need to retrain anything. You do not need to change
                your model. If you have a trained DDPM, you already have a
                score-based model. The equivalence is mathematical—not a new
                architecture.
              </p>
            </div>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* Section 8: Check #2 */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Check: The Equivalence"
            subtitle="Verify your understanding of the score-noise connection"
          />
          <div className="space-y-4">
            <GradientCard title="Question 1" color="cyan">
              <div className="space-y-2 text-sm">
                <p>
                  If{' '}
                  <InlineMath math="\epsilon_\theta(x_t, t)" /> is a vector
                  pointing to the upper-right, what direction does the score
                  point?
                </p>
                <details className="mt-2">
                  <summary className="font-medium cursor-pointer text-primary">
                    Reveal
                  </summary>
                  <p className="mt-2 text-muted-foreground">
                    Lower-left, with magnitude scaled by{' '}
                    <InlineMath math="1/\sqrt{1-\bar\alpha_t}" />. The score
                    is the negative of the noise prediction, scaled. The noise
                    points away from data; the score points toward data.
                  </p>
                </details>
              </div>
            </GradientCard>

            <GradientCard title="Question 2" color="cyan">
              <div className="space-y-2 text-sm">
                <p>
                  At <InlineMath math="t=1000" /> (near pure noise,{' '}
                  <InlineMath math="\bar\alpha" /> very small), how does the
                  scaling factor{' '}
                  <InlineMath math="1/\sqrt{1-\bar\alpha_t}" /> behave? What
                  about at{' '}
                  <InlineMath math="t=50" /> (nearly clean)?
                </p>
                <details className="mt-2">
                  <summary className="font-medium cursor-pointer text-primary">
                    Reveal
                  </summary>
                  <p className="mt-2 text-muted-foreground">
                    At <InlineMath math="t=1000" />,{' '}
                    <InlineMath math="1 - \bar\alpha" /> is near 1, so the
                    factor is near 1—noise prediction and score are almost
                    the same thing. At <InlineMath math="t=50" />,{' '}
                    <InlineMath math="1 - \bar\alpha" /> is small, so the
                    factor is large—small noise predictions correspond to
                    large scores. This makes sense: near-clean data has
                    sharply peaked probability, so the score is steep.
                  </p>
                </details>
              </div>
            </GradientCard>
          </div>
        </Row.Content>
      </Row>

      {/* Section 9: From Discrete Steps to Continuous Time (SDEs) */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="From Discrete Steps to Continuous Time"
            subtitle="The staircase becomes a ramp"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              DDPM adds noise in <InlineMath math="T=1000" /> discrete steps.
              What happens if we make <InlineMath math="T" /> infinitely large
              and the step size infinitely small?
            </p>
            <p className="text-muted-foreground">
              Picture the discrete forward process as a staircase—the signal
              drops in jumps at each step. As the number of steps increases,
              the staircase becomes smoother. In the limit, you get a
              continuous ramp:
            </p>
            <div className="grid gap-4 md:grid-cols-3">
              <GradientCard title="T = 10 Steps" color="amber">
                <div className="space-y-2 text-sm">
                  <p className="font-medium text-amber-300/90">
                    Clear staircase
                  </p>
                  <p className="text-muted-foreground">
                    Signal strength drops in 10 visible jumps. Each step
                    is a big chunk of noise addition. You can count the
                    individual steps—the trajectory is jagged and blocky.
                  </p>
                </div>
              </GradientCard>
              <GradientCard title="T = 100 Steps" color="blue">
                <div className="space-y-2 text-sm">
                  <p className="font-medium text-blue-300/90">
                    Finer steps, starting to smooth
                  </p>
                  <p className="text-muted-foreground">
                    100 smaller jumps. The staircase is still there, but
                    the steps are much finer. From a distance, it starts
                    to look like a curve. Each step adds a smaller amount
                    of noise.
                  </p>
                </div>
              </GradientCard>
              <GradientCard title="T → ∞ (SDE)" color="emerald">
                <div className="space-y-2 text-sm">
                  <p className="font-medium text-emerald-300/90">
                    Perfectly smooth ramp
                  </p>
                  <p className="text-muted-foreground">
                    Infinitely many infinitely small steps. The staircase
                    has become a smooth curve. No visible jumps—just a
                    continuous slide from data to noise. This is the SDE.
                  </p>
                </div>
              </GradientCard>
            </div>
            <p className="text-sm text-muted-foreground">
              Same starting point (clean data), same ending point (pure
              noise), same overall trajectory. The SDE is just the
              infinitely refined version of the DDPM forward process.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Staircase to Ramp">
            DDPM = staircase (discrete jumps). SDE = ramp (continuous curve).
            Same direction, same destination, smoother path.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* The forward SDE */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The continuous-time forward process is described by a stochastic
              differential equation (SDE):
            </p>
            <div className="py-4 px-6 bg-muted/50 rounded-lg">
              <BlockMath math="dx = -\tfrac{1}{2}\beta(t)\, x\, dt + \sqrt{\beta(t)}\, dw" />
            </div>
            <p className="text-muted-foreground">
              In plain language: <em>shrink the signal a tiny bit, add a tiny
              bit of random noise, repeat continuously.</em> This is the DDPM
              step{' '}
              <InlineMath math="x_t = \sqrt{1-\beta_t}\, x_{t-1} + \sqrt{\beta_t}\, \epsilon" />{' '}
              taken to the continuous limit.
            </p>
            <GradientCard title="Numerical Example: One Forward SDE Step" color="cyan">
              <div className="space-y-2 text-sm">
                <p>
                  Suppose you are at <InlineMath math="x = 3.0" /> at time{' '}
                  <InlineMath math="t" /> where{' '}
                  <InlineMath math="\beta(t) = 0.02" />, and the step size
                  is <InlineMath math="dt = 0.001" />:
                </p>
                <ul className="space-y-1 ml-2">
                  <li>
                    <strong>Drift</strong> (shrink the signal):{' '}
                    <InlineMath math="-\tfrac{1}{2}(0.02)(3.0)(0.001) = -0.00003" />
                  </li>
                  <li>
                    <strong>Diffusion</strong> (random nudge):{' '}
                    <InlineMath math="\sqrt{0.02} \cdot \sqrt{0.001} \cdot z \approx 0.0045 \cdot z" />{' '}
                    where <InlineMath math="z \sim \mathcal{N}(0,1)" />
                  </li>
                  <li>
                    <strong>Result:</strong>{' '}
                    <InlineMath math="x \approx 3.0 - 0.00003 + 0.0045z" />
                  </li>
                </ul>
                <p className="text-muted-foreground">
                  A tiny shrink, a tiny random nudge. Now repeat this a million
                  times. That is the forward SDE—the same thing as DDPM&rsquo;s
                  noise addition, just with infinitely small steps.
                </p>
              </div>
            </GradientCard>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="Reading the SDE">
            <ul className="space-y-1 text-sm">
              <li>
                <InlineMath math="-\tfrac{1}{2}\beta(t)\, x\, dt" /> = shrink
                the signal
              </li>
              <li>
                <InlineMath math="\sqrt{\beta(t)}\, dw" /> = add random noise
              </li>
              <li>
                <InlineMath math="\beta(t)" /> = noise schedule (same concept
                as DDPM&rsquo;s <InlineMath math="\beta_t" />)
              </li>
            </ul>
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* SDE warning */}
      <Row>
        <Row.Content>
          <WarningBlock title="We Will NOT Do Ito Calculus">
            The <InlineMath math="dx" /> notation means &ldquo;an infinitely
            small version of the DDPM step you already know.&rdquo; The{' '}
            <InlineMath math="dw" /> means &ldquo;an infinitely small random
            nudge.&rdquo; If you want to think of this as &ldquo;DDPM with
            infinitely many infinitely small steps,&rdquo; that is perfectly
            correct. We will never write an Ito integral in this course.
          </WarningBlock>
        </Row.Content>
      </Row>

      {/* Address misconception #2 */}
      <Row>
        <Row.Content>
          <GradientCard title="Same Process, Smoother Description" color="emerald">
            <p className="text-sm">
              The SDE <strong>IS</strong> the DDPM forward process, just with
              the step size taken to zero. Same process, smoother description.
              DDPM&rsquo;s variance-preserving step is a discrete
              approximation of this SDE. You already know the process—the
              SDE is just the continuous-time name for it.
            </p>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* Part B: The reverse SDE */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Reverse SDE"
            subtitle="Generation guided by the score"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Anderson showed in 1982 that if the forward process is an SDE,
              the reverse process is <strong>also</strong> an SDE—and it
              depends on the score function:
            </p>
            <div className="py-4 px-6 bg-muted/50 rounded-lg">
              <BlockMath math="dx = \left[-\tfrac{1}{2}\beta(t)\, x - \beta(t)\,\nabla_{x}\log q_t(x)\right] dt + \sqrt{\beta(t)}\, d\bar{w}" />
            </div>
            <p className="text-muted-foreground">
              In plain language: <em>undo the shrinking, follow the score
              toward higher probability, add a bit of random exploration.
              Repeat continuously.</em>
            </p>
            <p className="text-muted-foreground">
              Connect this to the DDPM reverse step you already know.
              The{' '}
              <InlineMath math="\epsilon_\theta" /> term (noise removal) IS
              the score term. The{' '}
              <InlineMath math="\sigma_t z" /> injection IS the stochastic{' '}
              <InlineMath math="d\bar{w}" /> term. DDPM reverse sampling was
              always approximately solving this reverse SDE.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="DDPM Was Always This">
            DDPM reverse sampling was always approximately solving the reverse
            SDE. The score function was always the guide. The{' '}
            <InlineMath math="\sigma_t z" /> noise injection at each step was
            always the stochastic term.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Part C: The probability flow ODE */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Probability Flow ODE"
            subtitle="Remove the randomness, keep the trajectory"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              What if we remove the stochastic term from the reverse SDE?
            </p>
            <div className="py-4 px-6 bg-muted/50 rounded-lg">
              <BlockMath math="dx = \left[-\tfrac{1}{2}\beta(t)\, x - \tfrac{1}{2}\beta(t)\,\nabla_{x}\log q_t(x)\right] dt" />
            </div>
            <p className="text-muted-foreground">
              No randomness. Fully deterministic. Notice the score coefficient
              changed from <InlineMath math="\beta(t)" /> in the reverse SDE
              to <InlineMath math="\tfrac{1}{2}\beta(t)" /> here—removing the
              stochastic term requires adjusting the drift to preserve the
              same marginal distributions. The derivation involves
              Fokker-Planck (which we are skipping), but the result is
              clean: halve the score coefficient, drop the noise.
            </p>
            <p className="text-muted-foreground">
              And here is the remarkable fact: this ODE produces the{' '}
              <strong>same marginal distributions</strong> at every time{' '}
              <InlineMath math="t" /> as the reverse SDE.
            </p>
            <p className="text-muted-foreground">
              This <strong>IS</strong> the ODE from{' '}
              <strong>Samplers and Efficiency</strong>. &ldquo;DDIM is
              approximately Euler&rsquo;s method on the probability flow
              ODE&rdquo;—now you know exactly what that ODE is.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Naming What You Already Know">
            You have been using the probability flow ODE since DDIM. It was
            always the &ldquo;deterministic sampling trajectory.&rdquo; Now
            it has a proper name and a derivation: it is the reverse SDE
            with the noise term removed.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Comparison: Reverse SDE vs Probability Flow ODE */}
      <Row>
        <Row.Content>
          <ComparisonRow
            left={{
              title: 'Reverse SDE',
              color: 'amber',
              items: [
                'Has randomness (stochastic dw term)',
                'DDPM analog: DDPM sampling (σ_t > 0)',
                'Different path each time',
                'Diverse outputs from same starting noise',
              ],
            }}
            right={{
              title: 'Probability Flow ODE',
              color: 'blue',
              items: [
                'No randomness (fully deterministic)',
                'DDPM analog: DDIM sampling (σ = 0)',
                'Same path every time',
                'Identical output from same starting noise',
              ],
            }}
          />
          <div className="mt-4 space-y-2">
            <p className="text-sm text-muted-foreground">
              <strong>What they share:</strong> Same score function. Same model
              weights. Same marginal distributions at every timestep. The
              reverse SDE and probability flow ODE describe the same generative
              process. One is stochastic (explores), the other is deterministic
              (commits).
            </p>
            <p className="text-sm text-muted-foreground">
              The student already knows both: DDPM vs DDIM. Same landscape,
              different lens—the callback from{' '}
              <strong>Samplers and Efficiency</strong>.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Same Landscape, Different Lens">
            This is the &ldquo;same vehicle, different route&rdquo; analogy
            from <strong>Samplers and Efficiency</strong>, formalized. DDPM
            (stochastic) and DDIM (deterministic) were always the reverse
            SDE and probability flow ODE. Now you know their proper names.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Section 10: Check #3 (Transfer) */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Transfer Questions"
            subtitle="Apply the score-based perspective to what you already know"
          />
          <div className="space-y-4">
            <GradientCard title="Scenario 1: Vocabulary" color="cyan">
              <div className="space-y-2 text-sm">
                <p>
                  A colleague says they are using a &ldquo;score-based
                  model&rdquo; and you are using &ldquo;DDPM.&rdquo; Are you
                  using fundamentally different generative models?
                </p>
                <details className="mt-2">
                  <summary className="font-medium cursor-pointer text-primary">
                    Reveal
                  </summary>
                  <p className="mt-2 text-muted-foreground">
                    No. DDPM IS a score-based model. The noise prediction
                    network learns a scaled version of the score function. The
                    DDPM forward/reverse process is a discrete approximation of
                    the SDE framework. Different vocabulary for the same thing.
                  </p>
                </details>
              </div>
            </GradientCard>

            <GradientCard title="Scenario 2: DPM-Solver++" color="cyan">
              <div className="space-y-2 text-sm">
                <p>
                  DPM-Solver++ is described as a &ldquo;higher-order ODE solver
                  for the probability flow ODE.&rdquo; Based on what you know
                  from <strong>Samplers and Efficiency</strong>, what
                  does &ldquo;higher-order&rdquo; mean here, and why does it
                  help?
                </p>
                <details className="mt-2">
                  <summary className="font-medium cursor-pointer text-primary">
                    Reveal
                  </summary>
                  <p className="mt-2 text-muted-foreground">
                    Higher-order means evaluating the score/model at multiple
                    nearby timesteps to estimate how the trajectory curves,
                    enabling larger steps with fewer evaluations. This is the
                    same insight from{' '}
                    <strong>Samplers and Efficiency</strong>—DPM-Solver was
                    solving the probability flow ODE all along. You just did
                    not have the vocabulary for it yet.
                  </p>
                </details>
              </div>
            </GradientCard>
          </div>
        </Row.Content>
      </Row>

      {/* Section 11: Elaborate — Why This Perspective Matters */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Why This Perspective Matters"
            subtitle="Three reasons this changes how you read papers"
          />
          <div className="space-y-4">
            <GradientCard title="1. Unified Vocabulary" color="blue">
              <p className="text-sm">
                &ldquo;Score-based,&rdquo; &ldquo;SDE,&rdquo;
                &ldquo;probability flow ODE&rdquo; appear in every modern
                diffusion paper. You can now parse these terms and connect them
                to the DDPM mechanics you already understand deeply.
              </p>
            </GradientCard>
            <GradientCard title="2. Flow Matching Depends on This" color="violet">
              <p className="text-sm">
                The next lesson replaces the SDE&rsquo;s curved trajectories
                with straight lines. That only makes sense if you see
                generation as following a trajectory defined by a vector
                field—the score. Without this lesson, flow matching is just
                another technique. With this lesson, it is a natural
                simplification.
              </p>
            </GradientCard>
            <GradientCard title="3. Consistency Models Depend on This" color="emerald">
              <p className="text-sm">
                Later in the series: &ldquo;any point on the same ODE
                trajectory maps to the same endpoint.&rdquo; That statement
                requires the probability flow ODE. Without it, consistency
                models are unmotivated.
              </p>
            </GradientCard>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The Unifying Thread">
            The score function is the thread that connects everything. DDPM,
            DDIM, DPM-Solver, flow matching, consistency models—they all work
            because a trained network can estimate the score of the noisy data
            distribution.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Section 12: Practice — Notebook Link */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Practice: Score Functions Hands-On"
            subtitle="Four exercises from guided to independent"
          />
          <div className="rounded-lg border-2 border-primary/50 bg-primary/5 p-6">
            <div className="space-y-4">
              <p className="text-muted-foreground">
                The notebook makes the score function concrete—compute it by
                hand, visualize it as a vector field, verify the score-noise
                equivalence with a real model, and compare SDE vs ODE
                trajectories.
              </p>
              <a
                href={NOTEBOOK_URL}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 px-4 py-2 rounded-md bg-primary text-primary-foreground font-medium text-sm hover:bg-primary/90 transition-colors"
              >
                <ExternalLink className="w-4 h-4" />
                Open in Google Colab
              </a>
              <ul className="text-sm text-muted-foreground space-y-3">
                <li>
                  <strong>Exercise 1 (Guided): Compute Score Functions by
                  Hand.</strong>{' '}
                  Given 1D Gaussians ({' '}
                  <InlineMath math="\mathcal{N}(0,1)" />,{' '}
                  <InlineMath math="\mathcal{N}(3,2)" />, mixture of two
                  Gaussians), compute the score function analytically. Plot
                  score alongside PDF. Verify that the score is zero at peaks.
                </li>
                <li>
                  <strong>Exercise 2 (Guided): 2D Score Vector Field.</strong>{' '}
                  Create a 2D Gaussian mixture. Plot the score field as arrows
                  (quiver plot). Add increasing noise and re-plot—observe how
                  high noise simplifies the field.
                </li>
                <li>
                  <strong>Exercise 3 (Supported): Verify the
                  Score-Noise Equivalence.</strong>{' '}
                  Load a pre-trained diffusion model. For a given noisy image,
                  get the noise prediction, compute the implied score, and
                  verify the direction makes sense.
                </li>
                <li>
                  <strong>Exercise 4 (Independent): SDE vs ODE
                  Trajectories.</strong>{' '}
                  Generate samples by solving the reverse SDE (stochastic) and
                  the probability flow ODE (deterministic). Compare: SDE paths
                  are wiggly and diverse, ODE paths are smooth and
                  deterministic.
                </li>
              </ul>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Exercise Progression">
            <ol className="space-y-1 text-sm list-decimal list-inside">
              <li>Guided: compute scores analytically</li>
              <li>Guided: visualize score fields</li>
              <li>Supported: verify equivalence with real model</li>
              <li>Independent: compare SDE vs ODE generation</li>
            </ol>
            <p className="text-sm mt-2">
              Exercises mirror the lesson arc: define the score, see it
              visually, connect to DDPM, explore the SDE/ODE duality.
            </p>
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Section 13: Summary */}
      <Row>
        <Row.Content>
          <SummaryBlock
            title="Key Takeaways"
            items={[
              {
                headline:
                  'The score function is the gradient of log probability.',
                description:
                  'score(x) = nabla_x log p(x). At every point in data space, it gives you a direction toward higher probability. It is a compass toward likely data.',
              },
              {
                headline:
                  'DDPM\'s noise prediction IS the score function (up to scaling).',
                description:
                  'epsilon_theta approx -sqrt(1 - alpha_bar_t) * score. The model was always learning to point toward data. Nothing new was trained—the score was hiding inside DDPM all along.',
              },
              {
                headline:
                  'The SDE framework generalizes DDPM to continuous time.',
                description:
                  'Forward SDE = continuous noise addition (DDPM with infinitely small steps). Reverse SDE = score-guided generation with stochastic exploration.',
              },
              {
                headline:
                  'The probability flow ODE is the deterministic version.',
                description:
                  'Remove the noise term from the reverse SDE and you get a deterministic ODE. DDIM was already approximately solving it. DPM-Solver is a better solver for it.',
              },
              {
                headline:
                  'Same model, multiple perspectives.',
                description:
                  'Score-based, diffusion, DDPM, DDIM—different vocabulary for the same underlying process. A trained noise prediction network is a score estimator. The reverse process is an SDE or ODE guided by the score.',
              },
            ]}
          />
        </Row.Content>
      </Row>

      {/* References */}
      <Row>
        <Row.Content>
          <ReferencesBlock
            references={[
              {
                title: 'Score-Based Generative Modeling through Stochastic Differential Equations',
                authors: 'Song, Sohl-Dickstein, Kingma, Kumar, Ermon & Poole, 2021',
                url: 'https://arxiv.org/abs/2011.13456',
                note: 'The paper that unified score-based and diffusion models under the SDE framework. Sections 2-3 cover the forward/reverse SDE and probability flow ODE.',
              },
              {
                title: 'Generative Modeling by Estimating Gradients of the Data Distribution',
                authors: 'Song & Ermon, 2019',
                url: 'https://arxiv.org/abs/1907.05600',
                note: 'The original score matching paper. Introduces the idea of learning the score function for generation. Historical context for the field.',
              },
              {
                title: 'Denoising Diffusion Probabilistic Models',
                authors: 'Ho, Jain & Abbeel, 2020',
                url: 'https://arxiv.org/abs/2006.11239',
                note: 'The DDPM paper you already know. Read Section 3.2 with fresh eyes—the noise prediction objective is implicitly learning the score.',
              },
            ]}
          />
        </Row.Content>
      </Row>

      {/* Section 14: Next Step */}
      <Row>
        <Row.Content>
          <NextStepBlock
            href="/app"
            title="Next: Flow Matching"
            description="Now you know that generation follows a trajectory—either the stochastic SDE or the deterministic ODE—defined by the score function. But look at these trajectories: they curve through high-dimensional space. What if we could straighten them? A straight path from noise to data would be simpler, need fewer steps, and be easier to train. That is flow matching."
          />
        </Row.Content>
      </Row>
    </LessonLayout>
  )
}

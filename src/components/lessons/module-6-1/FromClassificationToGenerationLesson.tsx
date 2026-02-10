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
} from '@/components/lessons'
import { GenerativeVsDiscriminativeWidget } from '@/components/widgets/GenerativeVsDiscriminativeWidget'
import { ExercisePanel } from '@/components/widgets/ExercisePanel'
import 'katex/dist/katex.min.css'
import { InlineMath } from 'react-katex'

/**
 * From Classification to Generation
 *
 * First lesson in Module 6.1 (Generative Foundations).
 * Teaches the conceptual distinction between discriminative
 * models (learning decision boundaries) and generative models
 * (learning the data distribution), and that generation means
 * sampling from a learned probability distribution.
 *
 * Core concepts at INTRODUCED:
 * - Generative vs discriminative framing
 * - Probability distributions over data
 * - Sampling as generation
 *
 * Previous: Decoder-Only Transformers (module 4.2, Series 4 capstone)
 * Next: Autoencoders (module 6.1, lesson 2)
 */

export function FromClassificationToGenerationLesson() {
  return (
    <LessonLayout>
      {/* ================================================================
          Header
          ================================================================ */}
      <Row>
        <Row.Content>
          <LessonHeader
            title="From Classification to Generation"
            description="Every model you&rsquo;ve built answers &ldquo;what is this?&rdquo; This lesson asks a different question: &ldquo;what could exist?&rdquo;"
            category="Generative Foundations"
          />
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 1: Context + Constraints
          ================================================================ */}
      <Row>
        <Row.Content>
          <ObjectiveBlock>
            Understand the distinction between discriminative models (learning
            decision boundaries between categories) and generative models
            (learning the distribution of the data itself), and see that
            generation means sampling from a learned probability distribution.
          </ObjectiveBlock>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="What You Already Know">
            You&apos;ve built classifiers for MNIST, trained CNNs on real images,
            and studied how language models predict the next token by sampling
            from a distribution. This lesson reframes what you already know.
          </TipBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <ConstraintBlock
            title="Scope for This Lesson"
            items={[
              'The conceptual distinction between discriminative and generative models',
              'What it means to "learn a distribution" at an intuitive level',
              'Why sampling from a distribution produces novel instances',
              'NOT: any specific generative architecture (autoencoders, GANs, diffusion)—those start next lesson',
              'NOT: probability density functions, likelihood, or formal probability theory',
              'NOT: how to train a generative model—this is the "what and why," not the "how"',
              'NOT: code or implementation—this is conceptual only',
            ]}
          />
        </Row.Content>
      </Row>

      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              This is the first lesson in a new series. We&apos;re headed toward
              Stable Diffusion&mdash;understanding how neural networks can generate
              photorealistic images from text descriptions. But before we get there, we
              need to answer a foundational question: what does it even mean for a
              neural network to <em>create</em> something new?
            </p>
            <p className="text-muted-foreground">
              This lesson is a change in perspective, not a new technique. You won&apos;t
              write code or learn new math. Instead, you&apos;ll see that everything
              you&apos;ve built so far has been answering one kind of question&mdash;and
              there&apos;s a fundamentally different kind of question a neural network can
              learn to answer.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Series 6 Roadmap">
            This module (Generative Foundations) builds the concepts you need:
            discriminative vs generative, autoencoders, latent spaces, then
            variational autoencoders. After this, we&apos;ll study diffusion models
            and finally Stable Diffusion itself.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 2: Hook — You've already seen a generative model
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="You&rsquo;ve Already Seen a Generative Model"
            subtitle="You just didn&rsquo;t call it that"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Every model you&apos;ve built answers &ldquo;what is this?&rdquo; Your MNIST
              classifier looks at pixels and says &ldquo;that&apos;s a 7.&rdquo; Your CNN
              looks at a photo and says &ldquo;that&apos;s a cat.&rdquo; But you&apos;ve
              already seen a model that answers a different question: &ldquo;what could come
              next?&rdquo;
            </p>
            <p className="text-muted-foreground">
              In <strong>What is a Language Model?</strong>, you watched a language model
              assign probabilities to every possible next token, then <em>sample</em> one. You
              dragged the temperature slider and watched the probability distribution reshape.
              When the model sampled &ldquo;mat&rdquo; after &ldquo;The cat sat on the,&rdquo;
              it was <strong>generating</strong>&mdash;producing something new by sampling from a
              learned distribution.
            </p>
            <p className="text-muted-foreground">
              That was generation. The language model learned the distribution of text well
              enough to produce new text that it had never seen in training. The sentences it
              generated were not copied from the training data&mdash;they were <em>sampled</em>{' '}
              from a learned distribution over possible continuations.
            </p>
            <p className="text-muted-foreground font-medium text-foreground">
              This lesson asks: can we do the same thing with images?
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The Bridge">
            You already understand generation intuitively. When a language model
            samples the next token, it&apos;s generating. When we say
            &ldquo;generative model,&rdquo; we mean the same idea extended to any
            kind of data&mdash;including images.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 3: The Discriminative Paradigm (recap + reframe)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Discriminative Paradigm"
            subtitle="What every model you&rsquo;ve built has in common"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Let&apos;s name what you&apos;ve been doing. Every model from Series 1 through
              Series 4 has been <strong>discriminative</strong>: given an input, produce a
              judgment about which category it belongs to.
            </p>
            <ul className="list-disc list-inside text-muted-foreground space-y-1 ml-4">
              <li>
                <strong>MNIST classifier:</strong> 784 pixels {'→'} 10 probabilities (which digit?)
              </li>
              <li>
                <strong>CNN on images:</strong> spatial features {'→'} class label (cat or dog?)
              </li>
              <li>
                <strong>Sentiment classifier:</strong> text {'→'} positive or negative (what&apos;s
                the tone?)
              </li>
            </ul>
            <p className="text-muted-foreground text-sm italic">
              Notice the language model isn&apos;t on this list. It predicts the next
              token given context&mdash;discriminative in form&mdash;but the
              autoregressive loop turns that prediction into generation. It lives at the
              boundary between paradigms, which is why it was such a natural bridge a
              moment ago.
            </p>
            <p className="text-muted-foreground">
              The defining characteristic: all of these learn{' '}
              <InlineMath math="P(y \mid x)" />&mdash;the probability of a <em>label</em>{' '}
              given an <em>input</em>. The model doesn&apos;t need to understand what the input
              looks like in any deep sense. It only needs to figure out which side of a boundary
              it falls on.
            </p>
            <p className="text-muted-foreground">
              Think of a 2D scatter plot with two classes of points and a line separating
              them (you&apos;ll interact with exactly this in a moment). The discriminative
              model&apos;s entire job is drawing that line. It says nothing about where the
              points cluster, how dense they are, or what a typical point from each class
              looks like. It only cares about the boundary.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="The Notation">
            <InlineMath math="P(y \mid x)" /> reads: &ldquo;probability of label{' '}
            <InlineMath math="y" /> given input <InlineMath math="x" />.&rdquo; This is
            the objective of every classifier. The vertical bar means &ldquo;given&rdquo;&mdash;the
            same notation from language modeling where you saw{' '}
            <InlineMath math="P(x_t \mid x_1, \ldots, x_{t-1})" />.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 4: The Generative Question
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Generative Question"
            subtitle="From &ldquo;what is this?&rdquo; to &ldquo;what could exist?&rdquo;"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              What if, instead of learning where the boundary is, we learned what the data
              looks like?
            </p>
            <p className="text-muted-foreground font-medium text-foreground">
              The negative example: can you invert the classifier?
            </p>
            <p className="text-muted-foreground">
              Here&apos;s a natural first idea. Your MNIST classifier maps images to labels:
              all images of 7s map to the label &ldquo;7.&rdquo; To generate, just run it
              backward: give it &ldquo;7&rdquo; and get back an image of a 7. Simple, right?
            </p>
            <p className="text-muted-foreground">
              But which 7 do you get back? There are millions of possible handwritten 7s&mdash;thin
              ones, thick ones, slanted ones, ones with a crossbar, ones without. Classification
              is <strong>many-to-one</strong>: all those different 7s map to the same label.
              Inverting a many-to-one function is <strong>one-to-many</strong>&mdash;you need to
              somehow specify <em>which</em> 7 you want. The label alone doesn&apos;t contain
              enough information.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Generation &ne; Reverse Classification">
            Classification squashes infinite variety into a single label.
            You can&apos;t unsquash it. The label &ldquo;7&rdquo; tells you
            nothing about stroke width, slant, or size. This is why generation
            requires a fundamentally different approach.
          </WarningBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground font-medium text-foreground">
              The key shift: learn the distribution of the data itself
            </p>
            <p className="text-muted-foreground">
              Instead of <InlineMath math="P(y \mid x)" /> (probability of a label given an
              input), learn <InlineMath math="P(x)" />&mdash;the probability of the data
              itself. But what does &ldquo;a distribution over data&rdquo; actually look like?
            </p>
            <p className="text-muted-foreground">
              Start simple. Imagine measuring the stroke width of every 7 in MNIST. You&apos;d
              get a histogram: most 7s have medium stroke width, some are thin, a few are very
              thick. That histogram <em>is</em> a distribution over one feature of 7s. Sampling
              from it gives you a stroke width for a new 7. Now imagine doing this for every
              feature simultaneously&mdash;stroke angle, crossbar presence, overall slant. A
              distribution over all these features together is a distribution over 7s. That&apos;s
              what <InlineMath math="P(x)" /> captures.
            </p>
            <p className="text-muted-foreground">
              An important clarification: <InlineMath math="P(x)" /> does <strong>not</strong>{' '}
              mean the model stores a separate probability for every possible image. With 784
              pixels each taking 256 values, there are roughly <InlineMath math="10^{1888}" />{' '}
              possible images&mdash;more than the atoms in the universe. No model could
              enumerate them. Instead, the model learns a compact representation of which regions
              of image space are likely. It learns <em>structure</em>, not an explicit lookup
              table. We&apos;ll see the staggering scale of this space in detail shortly.
            </p>
            <p className="text-muted-foreground">
              &ldquo;What does a 7 look like?&rdquo; has an answer, but it&apos;s not a
              single image. It&apos;s a distribution over possible 7s. Some 7s are
              more typical (upright, standard stroke width) and some are unusual (very slanted,
              extra thick). Once you have a model of that distribution, generation is sampling
              from it. Each sample is a different plausible 7, just as each sample from the
              language model was a different plausible next token.
            </p>

            {/* Art critic vs artist analogy */}
            <ComparisonRow
              left={{
                title: 'The Art Critic (Discriminative)',
                color: 'violet',
                items: [
                  'Studies paintings and judges: "This is impressionist, not cubist"',
                  'Draws boundaries between styles',
                  'Cannot create a new painting',
                  'Learns P(y|x)—the style given a painting',
                ],
              }}
              right={{
                title: 'The Artist (Generative)',
                color: 'emerald',
                items: [
                  'Studies paintings and internalizes: "This is what impressionism looks like"',
                  'Models the space of possible paintings within a style',
                  'Can paint new impressionist works',
                  'Learns P(x)—the distribution of paintings',
                ],
              }}
            />
            <p className="text-muted-foreground text-sm">
              Both the critic and the artist studied the same paintings. They have the same
              domain knowledge. The difference is in the <em>question</em> they learned to
              answer: the critic classifies, the artist generates.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Two Objectives">
            <p className="text-sm">
              <InlineMath math="P(y \mid x)" /> = discriminative.
              <br />
              <InlineMath math="P(x)" /> = generative.
            </p>
            <p className="text-sm mt-2">
              Same neural network building blocks (linear layers, convolutions,
              activations, backprop). Different question. Different loss function.
            </p>
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 5: Check — Predict-and-verify
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Check: Sampling from a Distribution"
            subtitle="Test your mental model"
          />
          <GradientCard title="Predict and Verify" color="emerald">
            <div className="space-y-3 text-sm">
              <p>
                A model has learned the distribution of handwritten 5s. You sample
                from it twice. <strong>Do you get the same image both times? Why or
                why not?</strong>
              </p>
              <details className="mt-3">
                <summary className="font-medium cursor-pointer text-primary">
                  Reveal
                </summary>
                <div className="mt-2 space-y-2">
                  <p>
                    No&mdash;sampling from a distribution is stochastic. Each sample is a
                    different plausible 5, just as each sample from the language model was
                    a different plausible next token. The distribution describes <em>which</em>{' '}
                    5s are more or less likely, but each draw from it is random.
                  </p>
                  <p>
                    If you got the exact same image both times, the model would be memorizing
                    rather than learning a distribution. The <em>variety</em> of samples is
                    evidence that the model learned structure, not specific instances.
                  </p>
                </div>
              </details>
            </div>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 6: Interactive widget
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Explore: Discriminative vs Generative"
            subtitle="The same data, two different lenses"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Below are 80 training points from two classes (blue and orange). Toggle between
              the two paradigms to see what each model learns from the same data.
            </p>
          </div>
        </Row.Content>
      </Row>

      <Row>
        <Row.Content>
          <ExercisePanel
            title="Generative vs Discriminative"
            subtitle="Toggle modes and sample new points"
          >
            <GenerativeVsDiscriminativeWidget />
          </ExercisePanel>
        </Row.Content>
        <Row.Aside>
          <TryThisBlock title="Experiments">
            <ul className="space-y-2 text-sm">
              <li>{'•'} Start in <strong>Discriminative</strong> mode. Notice the dashed
                boundary&mdash;everything on one side is &ldquo;Class A,&rdquo; everything on
                the other is &ldquo;Class B.&rdquo; The model says nothing about where the
                points cluster.</li>
              <li>{'•'} Switch to <strong>Generative</strong> mode. The boundary
                disappears. Instead you see a heatmap showing where data is likely to appear.
                Darker = more likely.</li>
              <li>{'•'} Click <strong>Sample 5 Points</strong> a few times. Are the new
                points identical? Do they fall exactly on training points? Notice how they
                cluster in high-density regions but are never exact copies.</li>
              <li>{'•'} Sample 20+ points. The sampled distribution starts to resemble
                the training data&mdash;because both come from the same underlying
                distribution.</li>
            </ul>
          </TryThisBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 7: Elaborate — Why this is hard / not memorization
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Why Generation Is Not Memorization"
            subtitle="The scale argument"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              A natural reaction: &ldquo;Isn&apos;t the model just memorizing training images
              and replaying them?&rdquo; Let&apos;s do the math and see why that can&apos;t
              be the full story.
            </p>
            <p className="text-muted-foreground">
              A 28&times;28 grayscale image (like MNIST) has 784 pixels, each taking a value
              from 0 to 255. The number of possible pixel configurations is{' '}
              <InlineMath math="256^{784}" />&mdash;approximately{' '}
              <InlineMath math="10^{1888}" />. That is incomprehensibly larger than the number
              of atoms in the observable universe (roughly <InlineMath math="10^{80}" />).
            </p>
            <p className="text-muted-foreground">
              The MNIST training set has 60,000 images. A model that memorized every single
              training image would have 60,000 points in a space of{' '}
              <InlineMath math="10^{1888}" /> possibilities. If the model can generate{' '}
              <em>novel</em> digits that look realistic but don&apos;t match any training
              image, it cannot be relying on memorization. It must have learned something
              about the <strong>structure</strong> of what makes a plausible digit.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Structure, Not Instances">
            The model learns that digits have strokes. Strokes have thickness and
            curvature. 7s have a horizontal bar and a diagonal. The model learns
            these <em>structural regularities</em>&mdash;the rules of what makes a
            plausible digit&mdash;not a catalog of specific digits.
          </InsightBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Remember how CNN layers learned features in a hierarchy&mdash;edges, then
              textures, then parts, then objects? A generative model learns similar
              structure, but uses it to <strong>create</strong> rather than
              <strong> classify</strong>. The feature hierarchy becomes a recipe for
              building new instances rather than a checklist for recognizing them.
            </p>
            <p className="text-muted-foreground">
              We are not yet talking about <em>how</em> a model learns this distribution.
              That starts next lesson with autoencoders. For now, the concept is clear:
              generation means sampling from a learned distribution, and the model must
              learn the structure of the data to do it well.
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 8: Check — Transfer question
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Check: The Memorization Argument"
            subtitle="Apply what you&rsquo;ve learned"
          />
          <GradientCard title="Transfer Question" color="emerald">
            <div className="space-y-3 text-sm">
              <p>
                Your colleague says: &ldquo;Generative AI just memorizes images from the
                internet and remixes them.&rdquo; Using what you learned about the
                dimensionality of image space and the size of training sets, explain why
                this can&apos;t be the full story.
              </p>
              <details className="mt-3">
                <summary className="font-medium cursor-pointer text-primary">
                  Reveal
                </summary>
                <div className="mt-2 space-y-2">
                  <p>
                    The space of possible images is astronomically larger than any training
                    set. Even MNIST has <InlineMath math="10^{1888}" /> possible images but
                    only 60,000 training examples. Real image models work with much higher
                    resolution, making the gap even more extreme.
                  </p>
                  <p>
                    If the model only memorized, it could only reproduce exact training images.
                    But generative models produce novel images&mdash;combinations of features
                    never seen together in training. The model must have learned <strong>structure</strong>{' '}
                    (stroke patterns, spatial relationships, textures), not a lookup table.
                  </p>
                  <p>
                    This doesn&apos;t mean memorization <em>never</em> happens&mdash;some
                    models do memorize specific training examples, especially unusual ones. But
                    memorization alone cannot explain the breadth and novelty of generated
                    output.
                  </p>
                </div>
              </details>
            </div>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 9: Summary
          ================================================================ */}
      <Row>
        <Row.Content>
          <SummaryBlock
            title="Key Takeaways"
            items={[
              {
                headline:
                  'Two paradigms: "what is this?" (discriminative) vs "what could exist?" (generative).',
                description:
                  'Discriminative models learn P(y|x)—decision boundaries between classes. Generative models learn P(x)—the distribution of the data itself.',
              },
              {
                headline: 'Generation = sampling from a learned distribution.',
                description:
                  `The model doesn't memorize specific instances. It learns the structure of the data well enough to produce plausible new instances by sampling.`,
              },
              {
                headline: 'You already know a generative model: the language model.',
                description:
                  'When you watched a language model sample the next token and adjusted the temperature slider, that was generation from a learned distribution. This module extends the idea to images.',
              },
              {
                headline: 'Same building blocks, different question.',
                description:
                  'Generative models use the same neural network operations (linear layers, convolutions, activations, backprop). The difference is the objective: what the loss function measures.',
              },
            ]}
          />
        </Row.Content>
      </Row>

      {/* Mental model echo */}
      <Row>
        <Row.Content>
          <InsightBlock title="Mental Models">
            <strong>&ldquo;Discriminative models draw boundaries; generative models
            learn density.&rdquo;</strong> The discriminative model asks &ldquo;which
            side of the line?&rdquo; The generative model asks &ldquo;how dense is this
            region?&rdquo;
            <br /><br />
            <strong>&ldquo;ML is function approximation&rdquo;</strong> extends to:
            a generative model approximates the data distribution{' '}
            <InlineMath math="P(x)" /> instead of a function{' '}
            <InlineMath math="f(x) \rightarrow y" />.
          </InsightBlock>
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 10: Next step
          ================================================================ */}
      <Row>
        <Row.Content>
          <NextStepBlock
            href="/app/lesson/autoencoders"
            title="Autoencoders"
            description="We know WHAT generation means&mdash;sample from a learned distribution. But HOW does a neural network learn a distribution over images? Next lesson: autoencoders. The idea is simple and powerful: force a network to compress an image into a tiny representation, then reconstruct it. If the reconstruction is good, the tiny representation captured the essential structure. And that tiny representation is our first step toward a space we can sample from."
            buttonText="Continue to Autoencoders"
          />
        </Row.Content>
      </Row>
    </LessonLayout>
  )
}

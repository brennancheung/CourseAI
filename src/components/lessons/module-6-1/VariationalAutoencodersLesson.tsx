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
  ReferencesBlock,
  LessonLink,
} from '@/components/lessons'
import { VaeLatentSpaceWidget } from '@/components/widgets/VaeLatentSpaceWidget'
import { ExercisePanel } from '@/components/widgets/ExercisePanel'
import { CodeBlock } from '@/components/common/CodeBlock'
import { ExternalLink } from 'lucide-react'
import 'katex/dist/katex.min.css'
import { InlineMath, BlockMath } from 'react-katex'

/**
 * Variational Autoencoders
 *
 * Lesson 3 in Module 6.1 (Generative Foundations).
 * Teaches why autoencoder latent spaces fail at generation (gaps),
 * how encoding to a distribution instead of a point fills those gaps,
 * and how KL divergence acts as a regularizer that keeps the latent
 * space organized and sampleable.
 *
 * Core concepts at DEVELOPED:
 * - Encoding to a distribution (mean + variance)
 * - KL divergence as latent space regularizer
 * - VAE loss function (reconstruction + KL)
 *
 * Core concepts at INTRODUCED:
 * - The reparameterization trick
 *
 * Core concepts at MENTIONED:
 * - ELBO
 *
 * Previous: Autoencoders (module 6.1, lesson 2)
 * Next: Exploring Latent Spaces (module 6.1, lesson 4)
 */

export function VariationalAutoencodersLesson() {
  return (
    <LessonLayout>
      {/* ================================================================
          Header
          ================================================================ */}
      <Row>
        <Row.Content>
          <LessonHeader
            title="Variational Autoencoders"
            description="Encode to a distribution, not a point. Add a regularizer to keep the latent space organized. The autoencoder becomes generative."
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
            Understand why autoencoder latent spaces have gaps that prevent
            generation, how encoding to a distribution (mean + variance) fills
            those gaps, and how KL divergence regularizes the latent space to
            keep it organized and sampleable. By the end, you will convert your
            autoencoder into a VAE and generate your first novel images.
          </ObjectiveBlock>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="What You Already Know">
            You built an autoencoder on Fashion-MNIST and saw that random latent
            codes produce garbage. You know the encoder-decoder architecture,
            reconstruction loss, and the bottleneck as a learning mechanism. The
            VAE modifies two things: what the encoder outputs and the loss
            function.
          </TipBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <ConstraintBlock
            title="Scope for This Lesson"
            items={[
              `Why the autoencoder’s latent space has gaps (the problem)`,
              'Encoding to a distribution (mean + variance) instead of a point (the fix, part 1)',
              'KL divergence as a regularizer that keeps the latent space organized (the fix, part 2)',
              'The reparameterization trick at intuition level only',
              'The VAE loss function: reconstruction + KL, and the tradeoff between them',
              'ELBO at intuition level only (one paragraph, named but not derived)',
              'NOT: full ELBO derivation or variational inference theory',
              'NOT: conditional VAEs, beta-VAE theory, or disentangled representations',
              'NOT: comparing VAEs to GANs or other generative architectures',
              'NOT: latent space interpolation, arithmetic, or generation experiments—that is the next lesson',
            ]}
          />
        </Row.Content>
      </Row>

      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              In <LessonLink slug="autoencoders">Autoencoders</LessonLink>, you built a network that
              compresses images through a bottleneck and reconstructs them. You
              saw that the bottleneck forces the network to learn what
              matters&mdash;but you also saw the critical failure: random latent
              codes fed to the decoder produce garbage.
            </p>
            <p className="text-muted-foreground">
              This lesson fixes that failure. We are adding two things to the
              autoencoder: (1) encode to a <strong>distribution</strong> instead
              of a point, and (2) a <strong>regularizer</strong> (KL divergence)
              that keeps the latent space organized. The result is a smooth
              latent space you can sample from&mdash;your first true generative
              model.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The Road So Far">
            Lesson 1: what generation <em>is</em> (sampling from P(x)).
            <br />
            Lesson 2: an architecture that creates a compressed
            representation&mdash;but cannot generate.
            <br />
            This lesson: make that architecture generative.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 2: Hook — Before/After
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Same Decoder, Different Result"
            subtitle="The autoencoder’s failure, fixed"
          />
          <ComparisonRow
            left={{
              title: 'Autoencoder',
              color: 'rose',
              items: [
                'Sample random z from the latent space',
                'Feed it to the decoder',
                'Result: unrecognizable garbage',
                'The decoder has never seen this region',
              ],
            }}
            right={{
              title: 'VAE',
              color: 'emerald',
              items: [
                'Sample random z from N(0, 1)',
                'Feed it to the same decoder architecture',
                'Result: a recognizable Fashion-MNIST item',
                'The entire latent space is meaningful',
              ],
            }}
          />
          <div className="space-y-4 mt-4">
            <p className="text-muted-foreground">
              Same decoder architecture. Same bottleneck size. The difference is
              how the encoder was trained. By the end of this lesson, you will
              understand exactly what changed.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The Payoff">
            The VAE does not require a fundamentally different network. It
            requires a different <em>training objective</em>&mdash;one that
            organizes the latent space, not just compresses into it.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 3: The Gap Problem
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Gap Problem"
            subtitle="Why random latent codes produce garbage"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Recall the autoencoder&apos;s latent space from{' '}
              <strong>Autoencoders</strong>. Each of the 60,000 training images
              maps to a specific point in 32-dimensional space. The decoder
              learns to reconstruct images <em>at</em> those points. But what
              about the space <strong>between</strong> those points?
            </p>
            <p className="text-muted-foreground">
              Nothing. The decoder has never seen anything from those regions.
              When you sample a random vector, it almost certainly lands in a
              gap&mdash;a region where no real image was ever encoded. The
              decoder produces garbage because it has no training signal for
              those locations.
            </p>
            <p className="text-muted-foreground">
              Think of it like a city map with buildings but no roads. You can
              visit buildings you already know (reconstruct training images), but
              you cannot walk between them. There is no path through the empty
              space. To generate, we need roads connecting everything&mdash;we
              need the entire map filled in.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="The Gap Problem">
            The autoencoder&apos;s latent space is a scattered collection of
            points with no structure between them. A map with dots but no
            terrain. Generation requires terrain everywhere&mdash;every point
            should decode to a plausible image.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 4: Encoding to a Distribution
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Encoding to a Distribution"
            subtitle="The fix, part 1: clouds instead of points"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The autoencoder encodes each image to a single point{' '}
              <InlineMath math="z" />. The VAE encodes each image to a{' '}
              <strong>distribution</strong>&mdash;described by two vectors: a
              mean <InlineMath math="\mu" /> and a log-variance{' '}
              <InlineMath math="\log \sigma^2" /> (we work in log-space
              because variance must be positive, but a network output can be
              any real number&mdash;exponentiating recovers the variance).
              Instead of saying
              &ldquo;this T-shirt is at position [-1.2, 0.8]&rdquo;, the
              encoder says &ldquo;this T-shirt is a cloud centered at [-1.2,
              0.8] with spread [0.3, 0.4]&rdquo;.
            </p>
            <p className="text-muted-foreground">
              During training, we <strong>sample</strong> a{' '}
              <InlineMath math="z" /> from this cloud. Each forward pass picks
              a slightly different point from the cloud. This means the decoder
              must handle a <em>range</em> of <InlineMath math="z" /> values
              for each image, not just one precise point. (How exactly we
              sample in a way that lets gradients flow is a clever trick we
              will see shortly.)
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Clouds, Not Points">
            Each image becomes a small cloud in latent space. Nearby images have
            overlapping clouds. The overlap fills the gaps&mdash;every region
            of the space gets training signal.
          </InsightBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground font-medium text-foreground">
              Concrete example: two Fashion-MNIST items
            </p>
            <div className="grid gap-4 md:grid-cols-2">
              <GradientCard title="T-Shirt Encoding" color="blue">
                <div className="space-y-1 text-sm">
                  <p>
                    <InlineMath math="\mu = [-1.2, \; 0.8]" />
                  </p>
                  <p>
                    <InlineMath math="\sigma = [0.3, \; 0.4]" />
                  </p>
                  <p className="text-muted-foreground mt-2">
                    A cloud centered at (-1.2, 0.8) with modest spread. Each
                    forward pass samples a slightly different z from this cloud.
                  </p>
                </div>
              </GradientCard>
              <GradientCard title="Sneaker Encoding" color="emerald">
                <div className="space-y-1 text-sm">
                  <p>
                    <InlineMath math="\mu = [0.5, \; -0.9]" />
                  </p>
                  <p>
                    <InlineMath math="\sigma = [0.5, \; 0.3]" />
                  </p>
                  <p className="text-muted-foreground mt-2">
                    A different cloud, in a different region. But if the spreads
                    are large enough, the edges of these clouds overlap.
                  </p>
                </div>
              </GradientCard>
            </div>
            <p className="text-muted-foreground">
              Each image gets its <strong>own</strong> mean and variance. The
              encoder is a function from image to distribution parameters. This
              is not one global distribution&mdash;it is a separate cloud for
              every image in the training set.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Per-Image, Not Global">
            A common misconception: &ldquo;the encoder outputs one distribution
            for the entire dataset.&rdquo; No. Each image gets its own{' '}
            <InlineMath math="\mu" /> and{' '}
            <InlineMath math="\sigma" />. The encoder <em>maps</em> each image
            to its own cloud.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 5: Check — Predict-and-verify
          ================================================================ */}
      <Row>
        <Row.Content>
          <GradientCard title="Predict and Verify" color="cyan">
            <div className="space-y-3 text-sm">
              <p>
                If the encoder outputs a cloud instead of a point for each
                image, and nearby images have overlapping clouds, what happens to
                the gaps in the latent space?
              </p>
              <details className="mt-3">
                <summary className="font-medium cursor-pointer text-primary">
                  Reveal
                </summary>
                <div className="mt-2 space-y-2">
                  <p>
                    They fill in. The overlapping clouds create continuous
                    coverage. Any point in the latent space is now within reach
                    of at least one cloud, so the decoder has training signal
                    for that region. No more gaps, no more garbage.
                  </p>
                </div>
              </details>
            </div>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 6: The Reparameterization Trick (brief)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Reparameterization Trick"
            subtitle="How gradients flow through sampling"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              We just said we sample <InlineMath math="z" /> during training.
              But sampling is a random operation&mdash;how does the gradient
              flow backward through randomness?
            </p>
            <p className="text-muted-foreground">
              The trick: instead of sampling <InlineMath math="z" /> directly
              from <InlineMath math="\mathcal{N}(\mu, \sigma^2)" />, we sample{' '}
              <InlineMath math="\epsilon" /> from{' '}
              <InlineMath math="\mathcal{N}(0, 1)" /> (which has no learnable
              parameters) and compute:
            </p>
            <div className="py-4 px-6 bg-muted/50 rounded-lg">
              <BlockMath math="z = \mu + \sigma \cdot \epsilon" />
            </div>
            <p className="text-muted-foreground">
              Now <InlineMath math="z" /> is a deterministic function of{' '}
              <InlineMath math="\mu" />, <InlineMath math="\sigma" />, and{' '}
              <InlineMath math="\epsilon" />. The randomness is isolated in{' '}
              <InlineMath math="\epsilon" />, which is just input noise.
              Gradients flow through <InlineMath math="\mu" /> and{' '}
              <InlineMath math="\sigma" /> normally.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="What You Need to Know">
            The reparameterization trick is a clever engineering solution. You
            need to know <strong>what</strong> it does (lets gradients flow
            through sampling) and <strong>how</strong> it works (sample noise
            separately, combine deterministically). You do not need to derive
            why it is valid.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 7: KL Divergence as Regularizer
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="KL Divergence: Keeping the Latent Space Organized"
            subtitle="The fix, part 2: a regularizer on the shape of the latent space"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              If we only use reconstruction loss with distributional encoding,
              the encoder will cheat. It will make{' '}
              <InlineMath math="\sigma" /> very small (approaching zero),
              collapsing the clouds back to points and recovering the original
              autoencoder. The gaps return. No improvement.
            </p>
            <p className="text-muted-foreground">
              We need a second loss term that prevents this collapse. That term
              is <strong>KL divergence</strong>, and it acts as a regularizer
              on the latent space shape. Two simple intuitions capture what it
              does:
            </p>
            <div className="grid gap-4 md:grid-cols-2 mt-2">
              <GradientCard title="Don’t hide in a corner" color="violet">
                <p className="text-sm">
                  KL penalizes means far from zero. If the encoder pushes all
                  T-shirts to <InlineMath math="\mu = [50, 80]" />, far from
                  the center, KL pulls them back. This keeps all distributions
                  near the origin, promoting overlap.
                </p>
              </GradientCard>
              <GradientCard
                title="Don’t make clouds too small"
                color="orange"
              >
                <p className="text-sm">
                  KL penalizes very small variance. If the encoder shrinks{' '}
                  <InlineMath math="\sigma \to 0" />, the cloud becomes a
                  point&mdash;back to the autoencoder. KL keeps the clouds
                  spread out so they overlap.
                </p>
              </GradientCard>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="KL = Regularizer">
            KL divergence is to the latent space what L2 regularization is to
            weights. L2 prevents weights from growing too large. KL prevents
            latent distributions from collapsing to points or hiding in corners.
            Same principle: constraints force better representations.
          </InsightBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              You do not need to derive KL divergence. You need to understand
              what it does (keeps the latent space organized) and how to compute
              it (a one-line formula for Gaussians):
            </p>
            <div className="py-4 px-6 bg-muted/50 rounded-lg">
              <BlockMath math="\text{KL} = -\frac{1}{2}\sum_{j=1}^{d}\left(1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2\right)" />
            </div>
            <p className="text-muted-foreground">
              Where <InlineMath math="d" /> is the latent dimension,{' '}
              <InlineMath math="\mu_j" /> and{' '}
              <InlineMath math="\sigma_j^2" /> are the encoder&apos;s output
              for dimension <InlineMath math="j" />. This formula measures how
              far each dimension&apos;s distribution is from a standard normal{' '}
              <InlineMath math="\mathcal{N}(0, 1)" />.
            </p>
            <p className="text-muted-foreground font-medium text-foreground">
              Worked example: the T-shirt encoding
            </p>
            <p className="text-muted-foreground">
              Take our T-shirt with{' '}
              <InlineMath math="\mu = [-1.2,\; 0.8]" /> and{' '}
              <InlineMath math="\sigma = [0.3,\; 0.4]" />. Compute KL for each
              dimension:
            </p>
            <div className="py-3 px-5 bg-muted/50 rounded-lg space-y-2 text-sm text-muted-foreground font-mono">
              <p>
                Dim 1: <InlineMath math="-\tfrac{1}{2}(1 + \log(0.3^2) - (-1.2)^2 - 0.3^2)" />{' '}
                = <InlineMath math="-\tfrac{1}{2}(1 - 2.41 - 1.44 - 0.09)" />{' '}
                = <strong>1.47</strong>
              </p>
              <p>
                Dim 2: <InlineMath math="-\tfrac{1}{2}(1 + \log(0.4^2) - 0.8^2 - 0.4^2)" />{' '}
                = <InlineMath math="-\tfrac{1}{2}(1 - 1.83 - 0.64 - 0.16)" />{' '}
                = <strong>0.82</strong>
              </p>
              <p>
                Total KL = 1.47 + 0.82 = <strong>2.29</strong>
              </p>
            </div>
            <p className="text-muted-foreground">
              Now imagine the encoder cheats:{' '}
              <InlineMath math="\sigma = [0.01,\; 0.01]" /> (clouds collapsed
              to near-points). The KL jumps to{' '}
              <strong>9.25</strong>&mdash;the regularizer catches it.
              Dimension 1 alone contributes 4.83 because{' '}
              <InlineMath math="\log(0.01^2) = -9.21" /> is a huge penalty for
              tiny variance.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="KL &ne; Reconstruction">
            KL divergence does <em>not</em> measure reconstruction quality. It
            measures how well-organized the latent space is. Setting KL weight
            too high makes reconstructions blurry because the model prioritizes
            latent space structure over pixel accuracy. The two losses
            <em> compete</em>.
          </WarningBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              In PyTorch, the KL term is a single line:
            </p>
          </div>
          <CodeBlock
            language="python"
            filename="vae_loss.py"
            code={`# KL divergence for Gaussian encoder
# mu, logvar are the encoder outputs (batch_size x latent_dim)
kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())`}
          />
        </Row.Content>
        <Row.Aside>
          <TipBlock title="logvar, Not sigma">
            The encoder outputs <code>logvar</code> (log variance), not{' '}
            <code>sigma</code>. This avoids numerical issues: log variance can
            be any real number, while variance must be positive.{' '}
            <InlineMath math="\sigma^2 = e^{\log \sigma^2}" />.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 7b: Negative example — noise alone does not fix it
          ================================================================ */}
      <Row>
        <Row.Content>
          <GradientCard title="Misconception Check: “VAE = Autoencoder + Noise”" color="rose">
            <div className="space-y-3 text-sm">
              <p>
                It is tempting to think the VAE works because we add noise to
                the latent codes during training. But{' '}
                <strong>noise alone does not fix the gaps</strong>.
              </p>
              <p>
                If you just added random noise to an autoencoder&apos;s latent
                codes, the network would learn to make codes far from zero so
                the noise is relatively tiny. The latent space structure would be
                unchanged&mdash;still scattered points, just with jitter.
              </p>
              <p className="font-medium text-foreground">
                The KL term is what forces organization. It prevents the encoder
                from cheating by (a) pushing codes far away or (b) making
                variances so small the noise disappears. Without KL, adding
                noise changes nothing.
              </p>
            </div>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 8: Check — Spot-the-difference
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Check: The Two Extremes"
            subtitle="What happens at each end of the tradeoff?"
          />
          <GradientCard title="Spot the Difference" color="emerald">
            <div className="space-y-3 text-sm">
              <p>
                <strong>Question 1:</strong> What happens if you set the KL
                weight to zero?
              </p>
              <details className="mt-2">
                <summary className="font-medium cursor-pointer text-primary">
                  Reveal
                </summary>
                <div className="mt-2">
                  <p>
                    You get the autoencoder back. The encoder collapses{' '}
                    <InlineMath math="\sigma \to 0" />, the clouds become
                    precise points, and the gaps return. Reconstructions are
                    sharp, but random sampling produces garbage.
                  </p>
                </div>
              </details>
              <p className="mt-4">
                <strong>Question 2:</strong> What happens if you set the
                reconstruction weight to zero (KL only)?
              </p>
              <details className="mt-2">
                <summary className="font-medium cursor-pointer text-primary">
                  Reveal
                </summary>
                <div className="mt-2">
                  <p>
                    Every image encodes to the same{' '}
                    <InlineMath math="\mathcal{N}(0, 1)" />. The latent space is
                    perfectly smooth, but the decoder gets no useful information.
                    Everything it produces looks the same&mdash;a blurry average
                    of all training images.
                  </p>
                </div>
              </details>
            </div>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 9: Interactive Widget
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Explore: Autoencoder vs VAE Latent Space"
            subtitle="See the gap problem and the VAE fix, side by side"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Toggle between Autoencoder and VAE mode. In Autoencoder mode,
              encoded items are scattered points with large empty gaps&mdash;
              sampling from gaps produces garbage. In VAE mode, each point
              becomes a cloud, the clouds overlap, and the entire space is
              meaningful. Use the &beta; slider to see the reconstruction-vs-
              regularization tradeoff.
            </p>
          </div>
        </Row.Content>
      </Row>

      <Row>
        <Row.Content>
          <ExercisePanel
            title="VAE Latent Space Explorer"
            subtitle="Compare autoencoder and VAE latent spaces"
          >
            <VaeLatentSpaceWidget />
          </ExercisePanel>
        </Row.Content>
        <Row.Aside>
          <TryThisBlock title="Experiments">
            <ul className="space-y-2 text-sm">
              <li>
                {'•'} In <strong>Autoencoder</strong> mode, click on the
                gaps between clusters. See the garbage output.
              </li>
              <li>
                {'•'} Switch to <strong>VAE</strong> mode. Click the same
                regions. What changed?
              </li>
              <li>
                {'•'} Set <strong>&beta; to 0</strong>. Does VAE mode still
                look different from Autoencoder mode?
              </li>
              <li>
                {'•'} Set <strong>&beta; to 5.0</strong>. What happens to
                the decoded images? Are they sharper or blurrier?
              </li>
              <li>
                {'•'} Find the <strong>sweet spot</strong> for &beta;
                &mdash;smooth enough to generate, sharp enough to recognize.
              </li>
            </ul>
          </TryThisBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 10: The Tradeoff
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Tradeoff: Reconstruction vs. Regularization"
            subtitle="Two losses pulling in opposite directions"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The VAE loss function is:
            </p>
            <div className="py-4 px-6 bg-muted/50 rounded-lg">
              <BlockMath math="\mathcal{L}_{\text{VAE}} = \underbrace{\mathcal{L}_{\text{recon}}}_{\text{reconstruction}} + \beta \cdot \underbrace{\text{KL}(\,q(z|x)\;\|\;\mathcal{N}(0,1)\,)}_{\text{regularization}}" />
            </div>
            <p className="text-muted-foreground">
              These two terms <strong>compete</strong>:
            </p>
            <ComparisonRow
              left={{
                title: 'Reconstruction Loss Wants...',
                color: 'blue',
                items: [
                  'Precise, specialized latent codes',
                  'Each image at a unique, exact point',
                  'Small variance (sharper clouds = sharper images)',
                  'Sharp reconstructions',
                ],
              }}
              right={{
                title: 'KL Divergence Wants...',
                color: 'violet',
                items: [
                  'Organized, overlapping distributions',
                  'All codes near the center',
                  'Moderate variance (spread out = smooth space)',
                  'Smooth, sampleable latent space',
                ],
              }}
            />
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The Fundamental Tension">
            The VAE tradeoff is not a flaw&mdash;it is the fundamental tension
            of generative modeling. You want the model to be both{' '}
            <em>precise</em> (reconstruct well) and <em>general</em> (generate
            well). Perfect precision prevents generalization; perfect
            generalization loses detail.
          </InsightBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              This is why VAE reconstructions are typically{' '}
              <strong>blurrier</strong> than autoencoder reconstructions at the
              same bottleneck size. The KL term forces overlapping distributions,
              which means the decoder must handle a range of{' '}
              <InlineMath math="z" /> values for each image, not just one
              precise point. The blurriness is the price of a smooth, sampleable
              latent space.
            </p>
            <p className="text-muted-foreground">
              The <InlineMath math="\beta" /> parameter controls the balance.
              Too much reconstruction emphasis: sharp images but gaps in the
              latent space. Too much KL emphasis: smooth space but blurry
              images. The standard VAE uses{' '}
              <InlineMath math="\beta = 1" />, but you can adjust it.
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 11: ELBO (one paragraph, MENTIONED)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="A Formal Name: The ELBO"
            subtitle="Connecting the loss to learning P(x)"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The VAE loss function has a formal name: the{' '}
              <strong>Evidence Lower Bound (ELBO)</strong>. It is called this
              because maximizing it is equivalent to maximizing a lower bound on
              the log-probability of the data&mdash;
              <InlineMath math="\log P(x)" />. In other words, the VAE loss is
              directly connected to learning{' '}
              <InlineMath math="P(x)" />, which is the goal we established in{' '}
              <strong>From Classification to Generation</strong>. You do not
              need to understand the derivation. What matters is: the
              reconstruction term measures how well the model explains
              individual data points, and the KL term ensures the latent space
              is organized enough to be a proper distribution.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="ELBO = Evidence Lower BOund">
            &ldquo;Evidence&rdquo; is a statistics term for
            the data itself. The ELBO is a lower bound on how well the model
            explains the evidence. Maximizing this bound trains a better
            generative model. The derivation is elegant but not
            needed for using VAEs.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 12: Full PyTorch Code
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The VAE in PyTorch"
            subtitle="Three changes from the autoencoder"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Compare this to the autoencoder from the previous lesson. There
              are exactly three changes: (1) the encoder outputs{' '}
              <code>mu</code> and <code>logvar</code> instead of{' '}
              <code>z</code>, (2) the reparameterization trick samples{' '}
              <code>z</code>, and (3) the loss function adds KL divergence.
            </p>
          </div>
          <CodeBlock
            language="python"
            filename="vae.py"
            code={`import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()

        # Encoder: same CNN as the autoencoder
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # CHANGE 1: Two output heads instead of one
        self.fc_mu = nn.Linear(32 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(32 * 7 * 7, latent_dim)

        # Decoder: same as the autoencoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32 * 7 * 7),
            nn.Unflatten(1, (32, 7, 7)),
            nn.ConvTranspose2d(32, 16, 3, stride=2,
                               padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2,
                               padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    # CHANGE 2: Reparameterization trick
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)    # sigma
        eps = torch.randn_like(std)       # epsilon ~ N(0,1)
        return mu + std * eps             # z = mu + sigma * eps

    def forward(self, x):
        h = self.encoder_conv(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar`}
          />
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Three Changes">
            1. Encoder outputs <code>mu</code> + <code>logvar</code> (not z)
            <br />
            2. <code>reparameterize()</code> samples z from the distribution
            <br />
            3. Forward returns <code>mu</code>, <code>logvar</code> for the KL
            loss
            <br />
            <br />
            Everything else&mdash;the conv layers, the decoder, the
            training loop&mdash;is identical.
          </InsightBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              And the training loop adds one line&mdash;the KL loss:
            </p>
          </div>
          <CodeBlock
            language="python"
            filename="train_vae.py"
            code={`criterion = nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(num_epochs):
    for images, _ in train_loader:
        recon, mu, logvar = model(images)

        # Reconstruction loss (same as autoencoder)
        recon_loss = criterion(recon, images)

        # CHANGE 3: KL divergence
        kl_loss = -0.5 * torch.sum(
            1 + logvar - mu.pow(2) - logvar.exp()
        )

        loss = recon_loss + kl_loss  # beta=1 by default

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()`}
          />
          <p className="text-muted-foreground mt-4">
            The labels are still ignored. The reconstruction loss still compares
            output to input. The only addition is the KL term that regularizes
            the latent space.
          </p>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="reduction=&apos;sum&apos;">
            We use <code>reduction=&apos;sum&apos;</code> instead of{' '}
            <code>&apos;mean&apos;</code> for reconstruction loss to match the
            scale of the KL term, which is also a sum. This is a common
            convention in VAE implementations.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 13: Colab Notebook
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Practice: Convert Your Autoencoder to a VAE"
            subtitle="Hands-on in a Colab notebook"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Time to build it yourself. You will convert the autoencoder from{' '}
              <strong>Autoencoders</strong> into a VAE:
            </p>
            <ul className="list-disc list-inside text-muted-foreground space-y-1 ml-4">
              <li>
                <strong>Part 1 (guided):</strong> Modify the encoder to output{' '}
                <code>mu</code> and <code>logvar</code> instead of{' '}
                <code>z</code>. Add the reparameterization trick.
              </li>
              <li>
                <strong>Part 2 (guided):</strong> Implement the VAE loss
                (reconstruction + KL). Train on Fashion-MNIST.
              </li>
              <li>
                <strong>Part 3 (supported):</strong> Compare autoencoder vs VAE
                reconstructions. Note the blurriness tradeoff.
              </li>
              <li>
                <strong>Part 4 (supported):</strong> Sample random z vectors
                from{' '}
                <InlineMath math="\mathcal{N}(0, 1)" />, decode them. Compare
                to the garbage from the autoencoder notebook. This is the
                generative payoff.
              </li>
            </ul>
          </div>

          <div className="rounded-lg border-2 border-primary/50 bg-primary/5 p-6 mt-4">
            <div className="space-y-4">
              <p className="text-muted-foreground">
                Open the notebook and convert your autoencoder into a VAE.
              </p>
              <a
                href="https://colab.research.google.com/github/brennancheung/CourseAI/blob/main/notebooks/6-1-3-variational-autoencoders.ipynb"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 px-4 py-2 rounded-md bg-primary text-primary-foreground font-medium text-sm hover:bg-primary/90 transition-colors"
              >
                <ExternalLink className="w-4 h-4" />
                Open in Google Colab
              </a>
              <p className="text-xs text-muted-foreground">
                Start from the autoencoder code in the previous notebook. The
                three changes are highlighted in the cells.
              </p>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Notebook Tips">
            The only new code is <code>fc_mu</code>, <code>fc_logvar</code>,
            the reparameterize function, and the KL loss line. Everything
            else&mdash;the conv layers, the DataLoader, the training
            loop&mdash;is identical to your autoencoder notebook.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 14: Summary
          ================================================================ */}
      <Row>
        <Row.Content>
          <SummaryBlock
            title="Key Takeaways"
            items={[
              {
                headline:
                  `The autoencoder’s latent space has gaps because each image maps to one point.`,
                description:
                  'Random points in the gaps produce garbage—the decoder has no training signal for those regions. Generation requires the entire space to be meaningful.',
              },
              {
                headline:
                  'The VAE encodes to a distribution (mu + sigma), filling the gaps with overlapping clouds.',
                description:
                  'Each image is a small cloud, not a point. Nearby images have overlapping clouds, creating continuous coverage across the latent space.',
              },
              {
                headline:
                  'KL divergence is a regularizer that keeps the distributions organized—near the center, not collapsed.',
                description:
                  `Two intuitions: don’t hide your codes in a corner (penalizes large means), and don’t make your clouds so tiny they’re basically points (penalizes small variance).`,
              },
              {
                headline:
                  'The VAE loss = reconstruction + KL, and the two terms compete.',
                description:
                  'Reconstruction wants sharp, specialized codes. KL wants organized, overlapping distributions. The balance is a tradeoff—VAE reconstructions are blurrier than autoencoder reconstructions, but the latent space is smooth and sampleable.',
              },
              {
                headline:
                  `The result: a smooth latent space you can sample from. The autoencoder’s failure is fixed.`,
                description:
                  'Sample any point from N(0,1), decode it, and you get a plausible image. You have built your first true generative model.',
              },
            ]}
          />
        </Row.Content>
      </Row>

      {/* Mental model echo */}
      <Row>
        <Row.Content>
          <InsightBlock title="Mental Models">
            <strong>
              &ldquo;Clouds, not points.&rdquo;
            </strong>{' '}
            The autoencoder gives each image a precise location. The VAE gives
            each image a neighborhood. The overlap between neighborhoods is
            what makes the space continuous and sampleable.
            <br />
            <br />
            <strong>
              &ldquo;KL is a regularizer on the latent space shape.&rdquo;
            </strong>{' '}
            Same principle as L2 on weights or dropout on activations. The
            constraint is the learning mechanism.
          </InsightBlock>
        </Row.Content>
      </Row>

      <Row>
        <Row.Content>
          <ReferencesBlock
            references={[
              {
                title: 'Auto-Encoding Variational Bayes',
                authors: 'Kingma & Welling, 2013',
                url: 'https://arxiv.org/abs/1312.6114',
                note: 'The original VAE paper introducing the reparameterization trick, KL divergence regularization, and ELBO objective.',
              },
            ]}
          />
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 15: Next Step
          ================================================================ */}
      <Row>
        <Row.Content>
          <NextStepBlock
            href="/app/lesson/exploring-latent-spaces"
            title="Exploring Latent Spaces"
            description="You have a smooth latent space and you can generate images by sampling from it. But you have only sampled randomly. In the next lesson, you will explore what this space actually looks like—interpolate between two images, discover that similar items cluster together, and do latent space arithmetic. The real fun starts now."
            buttonText="Continue to Exploring Latent Spaces"
          />
        </Row.Content>
      </Row>
    </LessonLayout>
  )
}

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
  ModuleCompleteBlock,
  GradientCard,
  ComparisonRow,
  PhaseCard,
  LessonLink,
} from '@/components/lessons'
import { CodeBlock } from '@/components/common/CodeBlock'
import { ExternalLink } from 'lucide-react'
import 'katex/dist/katex.min.css'
import { InlineMath, BlockMath } from 'react-katex'

/**
 * Exploring Latent Spaces
 *
 * Lesson 4 in Module 6.1 (Generative Foundations).
 * FINAL lesson in the module. Cognitive load: CONSOLIDATE.
 *
 * This is the reward after three lessons of building up the machinery.
 * The student explores their trained VAE's latent space through
 * sampling, interpolation, arithmetic, and visualization. No new
 * theory---just experiential payoff and recognition of VAE quality
 * limitations that motivate diffusion models.
 *
 * Core techniques at APPLIED:
 * - Latent space interpolation
 *
 * Core techniques at INTRODUCED:
 * - Latent arithmetic (vector operations on encoded representations)
 *
 * Core techniques at MENTIONED:
 * - t-SNE/UMAP for latent space visualization
 *
 * Previous: Variational Autoencoders (module 6.1, lesson 3)
 * Next: Module 6.2 (Diffusion Models)
 */

export function ExploringLatentSpacesLesson() {
  return (
    <LessonLayout>
      {/* ================================================================
          Header
          ================================================================ */}
      <Row>
        <Row.Content>
          <LessonHeader
            title="Exploring Latent Spaces"
            description="Sample, interpolate, and do arithmetic in the latent space. Create images that have never existed. Then see why you need diffusion models."
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
            Explore your trained VAE&apos;s latent space through sampling,
            interpolation, and arithmetic. Experience generation firsthand,
            recognize the quality ceiling inherent to VAEs, and understand why
            diffusion models are needed. This is a hands-on, notebook-centric
            lesson&mdash;no new theory.
          </ObjectiveBlock>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="What You Already Know">
            You built and trained a VAE on Fashion-MNIST. You know the latent
            space is smooth (KL regularization), that sampling from{' '}
            <InlineMath math="\mathcal{N}(0,1)" /> produces recognizable images,
            and that blurriness comes from the reconstruction-vs-KL tradeoff.
            This lesson lets you <em>play</em> with what you built.
          </TipBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <ConstraintBlock
            title="Scope for This Lesson"
            items={[
              'Sampling novel images from a trained VAE',
              'Linear interpolation between two images in latent space',
              'Latent arithmetic (vector addition and subtraction)',
              'Visualizing latent space structure with t-SNE/UMAP (as a tool, not a deep dive)',
              'Recognizing VAE quality limitations and connecting them to the tradeoff you already know',
              'NOT: training or modifying the VAE (that was the previous lesson)',
              'NOT: any new mathematical theory',
              'NOT: GANs, diffusion, or other generative architectures (those come next)',
              'NOT: t-SNE/UMAP algorithmic details',
              'NOT: disentangled representations or beta-VAE theory',
            ]}
          />
        </Row.Content>
      </Row>

      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              You have spent three lessons building the machinery: the generative
              framing that says &ldquo;learn P(x) and sample from it,&rdquo; the
              autoencoder that compresses images through a bottleneck, and the
              VAE that organizes that bottleneck into a smooth, sampleable space.
              Along the way, you briefly saw that sampling from the VAE produces
              recognizable images&mdash;but you rushed past it.
            </p>
            <p className="text-muted-foreground">
              This lesson slows down and lets you play. You will sample random
              images into existence, walk smoothly between two images through
              latent interpolation, discover that the latent space encodes
              meaningful relationships you can do arithmetic with, and visualize
              the structure of the space itself. This is the first time in the
              entire course that you <strong>create something that has never
              existed before</strong>.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The Road So Far">
            Lesson 1: what generation <em>is</em> (sampling from P(x)).
            <br />
            Lesson 2: an architecture that compresses but cannot generate.
            <br />
            Lesson 3: make it generative (VAE).
            <br />
            <strong>This lesson: experience generation.</strong>
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 2: Hook — "Create Something That Has Never Existed"
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Create Something That Has Never Existed"
            subtitle="The generative payoff"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              In <LessonLink slug="from-classification-to-generation">From Classification to Generation</LessonLink>, you asked
              the question: &ldquo;What does it mean to create?&rdquo; Now you
              have an answer. Sample a random vector{' '}
              <InlineMath math="z" /> from{' '}
              <InlineMath math="\mathcal{N}(0,1)" />, feed it to the decoder,
              and out comes an image that <strong>does not exist in the
              training set</strong>. The network learned the distribution of
              Fashion-MNIST items and can now produce new ones on demand.
            </p>
            <p className="text-muted-foreground">
              In this lesson, you will do exactly that. You will sample 25
              random vectors, decode them into a 5x5 grid, and see T-shirts,
              trousers, sneakers, dresses&mdash;all recognizable, all slightly
              different, <strong>none of them from the training set</strong>.
              Then you will walk smoothly between two images through latent
              interpolation, do vector arithmetic on encoded representations,
              and visualize the structure of the space itself. By the end, you
              will also see why these images are blurry&mdash;and why that
              matters.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Not Memorization">
            These images are not stored anywhere. The decoder does not have a
            lookup table. It learned a <em>function</em> from latent space to
            image space. A random input produces a novel output&mdash;structured
            by what the network learned about clothing.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 3: Sampling Deep Dive
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Sampling: How Generation Works"
            subtitle="From random noise to recognizable images"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The mechanics are simple. You already know them from{' '}
              <strong>Variational Autoencoders</strong>, but now you get to see
              them in action:
            </p>
          </div>
          <CodeBlock
            language="python"
            filename="sample.py"
            code={`# Sample 25 random latent vectors
z = torch.randn(25, latent_dim)   # 25 samples from N(0,1)

# Decode each one into an image
with torch.no_grad():
    images = model.decoder(z)     # (25, 1, 28, 28)

# Display as a 5x5 grid
show_grid(images, nrow=5)`}
          />
          <div className="space-y-4 mt-4">
            <p className="text-muted-foreground">
              Why <InlineMath math="\mathcal{N}(0,1)" />? Because the KL term
              in training organized the entire latent space around{' '}
              <InlineMath math="\mathcal{N}(0,1)" />. Every point near the
              origin is a region the decoder has been trained on. Sampling from
              this distribution guarantees you land in meaningful territory.
            </p>
            <p className="text-muted-foreground">
              Run it with different random seeds and you get different images.
              Similar items cluster in similar regions of z-space&mdash;all
              T-shirts live in one neighborhood, all sneakers in another. Two
              different z vectors from the same region produce <em>variations</em>{' '}
              of the same category.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Try in the Notebook">
            Run the sampling code multiple times with different seeds. Notice
            how some regions consistently produce T-shirts and others produce
            sneakers. The latent space has structure.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Predict-and-verify check */}
      <Row>
        <Row.Content>
          <GradientCard title="Predict and Verify" color="cyan">
            <div className="space-y-3 text-sm">
              <p>
                If you sample{' '}
                <InlineMath math="z = [0, 0, \ldots, 0]" />&mdash;the exact
                mean of <InlineMath math="\mathcal{N}(0,1)" />&mdash;what kind
                of image do you get?
              </p>
              <details className="mt-3">
                <summary className="font-medium cursor-pointer text-primary">
                  Reveal
                </summary>
                <div className="mt-2 space-y-2">
                  <p>
                    Something average-looking. Possibly a blend of the most
                    common categories, or a generic-looking garment. Not
                    garbage&mdash;because the center of the space is the
                    most well-populated region. The KL term pushed all
                    distributions toward this center, so the decoder has
                    seen many training samples near <InlineMath math="z = 0" />.
                  </p>
                </div>
              </details>
            </div>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 4: Interpolation — Walking Between Images
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Interpolation: Walking Between Images"
            subtitle="The latent space is smooth enough to walk through"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              You have a smooth latent space with roads connecting everything
              (the city map analogy from <LessonLink slug="variational-autoencoders">Variational
              Autoencoders</LessonLink>). What happens when you <em>walk</em> along
              those roads? You get <strong>interpolation</strong>&mdash;a
              smooth transition between two images that passes through coherent
              intermediate forms.
            </p>
            <p className="text-muted-foreground">
              The idea: encode image A to{' '}
              <InlineMath math="z_A" />, encode image B to{' '}
              <InlineMath math="z_B" />, and create a path between them:
            </p>
          </div>
          <div className="py-4 px-6 bg-muted/50 rounded-lg mt-2">
            <BlockMath math="z_t = (1-t) \cdot z_A + t \cdot z_B \quad \text{for } t \in [0, 1]" />
          </div>
          <div className="space-y-4 mt-4">
            <p className="text-muted-foreground">
              At <InlineMath math="t=0" /> you get image A. At{' '}
              <InlineMath math="t=1" /> you get image B. At every point in
              between, you get a <em>coherent intermediate image</em>&mdash;not
              garbage, not a ghostly double exposure, but a plausible garment
              that smoothly morphs from one to the other.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Walking the Roads">
            Remember the city map analogy? The VAE built roads between all the
            buildings. Interpolation is literally walking those roads. Every
            location along the path is a real place&mdash;a plausible image.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Negative example: pixel vs latent interpolation */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Pixel Interpolation vs. Latent Interpolation"
            subtitle="Why the latent space is special"
          />
          <ComparisonRow
            left={{
              title: 'Pixel-Space Interpolation',
              color: 'rose',
              items: [
                'Formula: 0.5 * image_A + 0.5 * image_B',
                'T-shirt + trousers = faint T-shirt ghosted on top of faint trousers',
                'Both shapes visible at once, transparent and overlapping',
                'Edges from both objects compete—nothing looks solid',
                'Like a double-exposed photograph, not a real garment',
              ],
            }}
            right={{
              title: 'Latent-Space Interpolation',
              color: 'emerald',
              items: [
                'Formula: decode(0.5 * z_A + 0.5 * z_B)',
                'T-shirt + trousers = a long shirt morphing toward trousers',
                'One coherent shape at every step—no ghosting',
                'Intermediate forms look like actual clothing items',
                'Like watching one garment transform into another',
              ],
            }}
          />
          <div className="space-y-4 mt-4">
            <p className="text-muted-foreground">
              Take a T-shirt and a pair of trousers. Average their pixels and
              you get a faint T-shirt ghosted on top of faint trousers. Average
              their <em>latent codes</em> and you get something like a long
              shirt morphing into trousers&mdash;a coherent transition through
              plausible garments that actually look like clothing.
            </p>
            <p className="text-muted-foreground">
              Why does latent interpolation work? Because the VAE&apos;s latent
              space is <strong>smooth</strong>. The KL regularization filled all
              the gaps. Every point along the path from{' '}
              <InlineMath math="z_A" /> to <InlineMath math="z_B" /> is in a
              region the decoder has seen during training. The decoder knows
              what to do with every intermediate code.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Not Just Blending">
            Interpolation is NOT blending two images. It is asking the decoder:
            &ldquo;What image lives at this intermediate point in the space you
            organized?&rdquo; The answer is a coherent image because the space
            is organized.
          </WarningBlock>
          <TryThisBlock title="See It Yourself">
            In the notebook, try pixel interpolation <em>first</em>:{' '}
            <code>0.5 * image_A + 0.5 * image_B</code>. See the ghostly double
            exposure. <em>Then</em> try latent interpolation:{' '}
            <code>decode(0.5 * z_A + 0.5 * z_B)</code>. The difference is
            visceral.
          </TryThisBlock>
        </Row.Aside>
      </Row>

      {/* Concrete walkthrough */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground font-medium text-foreground">
              Concrete walkthrough
            </p>
            <p className="text-muted-foreground">
              Suppose you encode a T-shirt and a sneaker. Here are the first 4
              dimensions of their latent codes (the full codes are
              32-dimensional):
            </p>
            <div className="grid gap-4 md:grid-cols-2">
              <GradientCard title="T-Shirt" color="blue">
                <div className="space-y-1 text-sm">
                  <p>
                    <InlineMath math="z_A = [-0.8, \; 1.2, \; 0.3, \; -0.5, \; \ldots]" />
                  </p>
                </div>
              </GradientCard>
              <GradientCard title="Sneaker" color="emerald">
                <div className="space-y-1 text-sm">
                  <p>
                    <InlineMath math="z_B = [0.6, \; -0.4, \; 1.1, \; 0.9, \; \ldots]" />
                  </p>
                </div>
              </GradientCard>
            </div>
            <p className="text-muted-foreground">
              At <InlineMath math="t = 0.5" />, the midpoint:
            </p>
            <div className="py-3 px-5 bg-muted/50 rounded-lg text-sm text-muted-foreground font-mono">
              <InlineMath math="z_{0.5} = 0.5 \cdot [-0.8, 1.2, 0.3, -0.5, \ldots] + 0.5 \cdot [0.6, -0.4, 1.1, 0.9, \ldots]" />
              <br />
              <InlineMath math="= [-0.1, \; 0.4, \; 0.7, \; 0.2, \; \ldots]" />
            </div>
            <p className="text-muted-foreground">
              Decode <InlineMath math="z_{0.5}" /> and you get something that
              looks like neither a pure T-shirt nor a pure sneaker, but a
              coherent intermediate form. Create a strip with{' '}
              <InlineMath math="t = 0, 0.25, 0.5, 0.75, 1.0" /> and you see a
              smooth morphing sequence.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="In the Notebook">
            Pick two test images from different categories. Encode both. Create
            an interpolation strip with 8 steps. Try multiple pairs&mdash;some
            transitions are more dramatic than others.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 5: Check — Predict the Interpolation
          ================================================================ */}
      <Row>
        <Row.Content>
          <GradientCard title="Predict and Verify" color="cyan">
            <div className="space-y-3 text-sm">
              <p>
                <strong>Question 1:</strong> You interpolate between a sneaker
                and an ankle boot. At <InlineMath math="t=0.5" />, what does the
                image look like?
              </p>
              <details className="mt-2">
                <summary className="font-medium cursor-pointer text-primary">
                  Reveal
                </summary>
                <div className="mt-2">
                  <p>
                    Something boot-like that shares features of both&mdash;the
                    sole of a sneaker with the height of a boot. A plausible
                    hybrid shoe, not a ghostly overlay of two shoes.
                  </p>
                </div>
              </details>
              <p className="mt-4">
                <strong>Question 2:</strong> Would this smooth interpolation
                work with an autoencoder (not a VAE)?
              </p>
              <details className="mt-2">
                <summary className="font-medium cursor-pointer text-primary">
                  Reveal
                </summary>
                <div className="mt-2">
                  <p>
                    No. The autoencoder&apos;s latent space has gaps between
                    encoded points. The midpoint{' '}
                    <InlineMath math="z_{0.5}" /> lands in uncharted territory
                    where the decoder has no training signal. The result is
                    garbage&mdash;not a coherent intermediate. This is exactly
                    the gap problem you solved with the VAE.
                  </p>
                </div>
              </details>
            </div>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 6: Latent Arithmetic — "Relationships Are Directions"
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Latent Arithmetic: Relationships Are Directions"
            subtitle="Vector operations on encoded representations"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              If the latent space captures meaningful structure, then the{' '}
              <strong>direction</strong> between two encoded items captures the{' '}
              <strong>difference</strong> between them. You can extract that
              direction and apply it to something else.
            </p>
            <p className="text-muted-foreground">
              The classic setup: encode an ankle boot and a sneaker. The vector{' '}
              <InlineMath math="z(\text{ankle boot}) - z(\text{sneaker})" />{' '}
              captures roughly &ldquo;what makes a boot different from a
              sneaker&rdquo;&mdash;something about height or ankle coverage. Now
              add that direction to a sandal:
            </p>
          </div>
          <div className="py-4 px-6 bg-muted/50 rounded-lg mt-2">
            <BlockMath math="z(\text{sandal}) + \big[z(\text{ankle boot}) - z(\text{sneaker})\big] = \; ?" />
          </div>
          <div className="space-y-4 mt-4">
            <p className="text-muted-foreground">
              If the space is well-organized, the result should decode to
              something like a higher sandal or a boot-like sandal. The
              &ldquo;height&rdquo; concept was transferred via vector
              arithmetic.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Directions = Concepts">
            The latent space is not just organized by category&mdash;it encodes{' '}
            <em>relationships</em> as directions. Subtracting two codes gives
            you a direction that represents the difference between them. Adding
            that direction to a third code transfers the difference.
          </InsightBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              In code, the arithmetic is trivial:
            </p>
          </div>
          <CodeBlock
            language="python"
            filename="arithmetic.py"
            code={`# Encode three items (encode returns mu, logvar — use mu)
z_boot, _ = model.encode(ankle_boot_image)
z_sneaker, _ = model.encode(sneaker_image)
z_sandal, _ = model.encode(sandal_image)

# Compute the "boot-ness" direction
boot_direction = z_boot - z_sneaker

# Apply it to the sandal
z_result = z_sandal + boot_direction

# Decode the result
result_image = model.decoder(z_result)`}
          />
          <div className="space-y-4 mt-4">
            <p className="text-muted-foreground">
              On Fashion-MNIST, the result will be <strong>noisy but
              directionally correct</strong>. You might see a sandal-like shape
              that is taller or more boot-like. It will not be as clean as the
              famous &ldquo;smile vector&rdquo; examples from CelebA face
              datasets.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Famous Example">
            The most well-known latent arithmetic example is on face images
            (CelebA): the direction from &ldquo;face without
            glasses&rdquo; to &ldquo;face with glasses&rdquo; transfers glasses
            onto a new face. It works cleanly because face datasets have
            consistent, continuous attribute variation.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Tempering expectations */}
      <Row>
        <Row.Content>
          <GradientCard title="Tempering Expectations" color="amber">
            <div className="space-y-3 text-sm">
              <p>
                Latent arithmetic is compelling, but do not overgeneralize.{' '}
                <strong>Most random directions in latent space do not
                correspond to interpretable features.</strong> Only specific
                learned directions encode meaningful attributes like
                &ldquo;height&rdquo; or &ldquo;glasses.&rdquo;
              </p>
              <p>
                On Fashion-MNIST, the results are noisier than on face datasets.
                Fashion-MNIST has discrete categories with less smooth attribute
                variation than faces. The concept is real, but clean results
                require data with consistent, continuous variation in the
                attribute you want to manipulate.
              </p>
            </div>
          </GradientCard>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Not Every Direction Is Meaningful">
            If you pick a random direction in the latent space and move along
            it, you will probably get nonsensical changes&mdash;brightness
            shifts, texture noise, nothing interpretable. Meaningful directions
            exist, but finding them requires work.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 7: Visualizing the Latent Space
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Visualizing the Latent Space"
            subtitle="Seeing structure with t-SNE"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The latent space is 32-dimensional. You cannot see 32 dimensions.
              But you can project it down to 2D using a technique called{' '}
              <strong>t-SNE</strong> (t-distributed Stochastic Neighbor
              Embedding). t-SNE tries to preserve neighborhood
              relationships&mdash;points that are close in 32D should be close
              in the 2D projection.
            </p>
            <p className="text-muted-foreground">
              Encode all 10,000 test set images, project the latent codes to 2D
              with t-SNE, and color each point by its category label. The result
              is a scatter plot that reveals the structure of the latent space:
            </p>
            <ul className="list-disc list-inside text-muted-foreground space-y-1 ml-4">
              <li>
                <strong>Clusters</strong>&mdash;T-shirts near T-shirts, sneakers
                near sneakers
              </li>
              <li>
                <strong>Smooth transitions</strong> between related
                categories&mdash;shoes blur into boots, shirts blur into coats
              </li>
              <li>
                <strong>Overlap</strong> where categories share
                features&mdash;pullover and coat regions may blend together
              </li>
            </ul>
            <p className="text-muted-foreground">
              This is the structure that the KL term created. Without it (pure
              autoencoder), the scatter plot would show scattered points with no
              organization. With it, the space is organized by similarity.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="KL Created This Structure">
            The clusters and smooth transitions you see in the t-SNE plot are a
            direct consequence of the KL regularizer. It pushed all
            distributions toward <InlineMath math="\mathcal{N}(0,1)" />,
            centering and organizing the space. Similar items were forced into
            overlapping regions.
          </InsightBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <CodeBlock
            language="python"
            filename="visualize.py"
            code={`from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Encode all test images
all_z = []
all_labels = []
for images, labels in test_loader:
    mu, _ = model.encode(images)
    all_z.append(mu.detach())
    all_labels.append(labels)

all_z = torch.cat(all_z).numpy()
all_labels = torch.cat(all_labels).numpy()

# Project to 2D
tsne = TSNE(n_components=2, random_state=42)
z_2d = tsne.fit_transform(all_z)

# Plot colored by category
plt.scatter(z_2d[:, 0], z_2d[:, 1],
            c=all_labels, cmap='tab10', s=1, alpha=0.5)
plt.colorbar()
plt.title("VAE Latent Space (t-SNE)")
plt.show()`}
          />
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="t-SNE Caveats">
            t-SNE is a <em>visualization</em> tool, not ground truth. It
            distorts distances and can show phantom clusters. If you see a
            cluster, those points are nearby in the real space. If you see a
            gap, it might be real or t-SNE exaggerating. Two different runs can
            give different layouts. The <code>perplexity</code> parameter
            (default 30) controls how many neighbors each point considers&mdash;
            try different values and notice how the plot changes.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 8: The Quality Ceiling
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Quality Ceiling"
            subtitle="Why VAE images are blurry—and why that matters"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              You have now generated, interpolated, and explored. Time for
              honest assessment. Look closely at your generated
              images&mdash;they are recognizable, but <strong>blurry</strong>.
              Compare them:
            </p>
          </div>
        </Row.Content>
      </Row>

      <Row>
        <Row.Content>
          <div className="space-y-3">
            <PhaseCard
              number={1}
              title="Original Training Images"
              subtitle="The ground truth"
              color="blue"
            >
              <p className="text-sm">
                Sharp, clean, well-defined edges. This is what Fashion-MNIST
                images actually look like at 28x28 resolution.
              </p>
            </PhaseCard>
            <PhaseCard
              number={2}
              title="Autoencoder Reconstructions"
              subtitle="Compressed and decompressed"
              color="cyan"
            >
              <p className="text-sm">
                Pretty sharp. Some fine detail lost from the bottleneck
                compression, but edges are clear and items are easily
                identifiable. Not generative, though&mdash;these are
                reconstructions of existing images.
              </p>
            </PhaseCard>
            <PhaseCard
              number={3}
              title="VAE Reconstructions"
              subtitle="The tradeoff in action"
              color="violet"
            >
              <p className="text-sm">
                Slightly blurrier than autoencoder reconstructions. The KL term
                forced overlapping distributions, so the decoder must hedge its
                bets across a range of z values. Sharper edges are sacrificed
                for a smooth, sampleable space.
              </p>
            </PhaseCard>
            <PhaseCard
              number={4}
              title="VAE Samples (Novel Images)"
              subtitle="Generated from random z"
              color="orange"
            >
              <p className="text-sm">
                Blurrier still. These are new images, not reconstructions. The
                decoder is working with z vectors it has never seen
                exactly&mdash;only nearby ones during training. Recognizable, but
                soft and lacking crisp detail.
              </p>
            </PhaseCard>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Blurriness Is Not a Bug">
            The blurriness is not a training failure you can fix with more
            epochs. It is a <em>fundamental consequence</em> of the
            reconstruction-vs-KL tradeoff you studied in{' '}
            <strong>Variational Autoencoders</strong>. A smoother latent space
            means the decoder must average over more possible inputs, producing
            softer outputs.
          </InsightBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              This quality progression&mdash;from sharp originals to increasingly
              blurry reconstructions and samples&mdash;is not a limitation of
              <em> your</em> VAE implementation. It is inherent to how VAEs
              work. The reconstruction-vs-KL tradeoff from{' '}
              <strong>Variational Autoencoders</strong> means you cannot have
              both a perfectly smooth latent space and perfectly sharp outputs.
            </p>
            <p className="text-muted-foreground">
              So: can neural networks do better? <strong>Much</strong> better.
              Look at any recent text-to-image model&mdash;photorealistic faces,
              detailed landscapes, complex scenes. The quality gap between a VAE
              on Fashion-MNIST and Stable Diffusion is enormous.
            </p>
          </div>
          <ComparisonRow
            left={{
              title: 'Your VAE',
              color: 'amber',
              items: [
                '28x28 grayscale',
                'Blurry, soft edges',
                'Recognizable but not impressive',
                'Simple clothing items',
              ],
            }}
            right={{
              title: 'Stable Diffusion',
              color: 'emerald',
              items: [
                '512x512+ full color',
                'Sharp, detailed, photorealistic',
                'Stunning quality, text-guided',
                'Arbitrary scenes and concepts',
              ],
            }}
          />
          <div className="space-y-4 mt-4">
            <p className="text-muted-foreground">
              The VAE proved the concept: you <strong>can</strong> generate
              images by sampling from a learned latent space. The quality is the
              problem&mdash;and solving that problem is what the rest of Series 6
              is about.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Concept Proven, Quality Not">
            You should feel two things right now: (1) the thrill of generation
            (&ldquo;I can create images from noise!&rdquo;) and (2) the itch
            to do better (&ldquo;but these are blurry&mdash;how do real
            generators work?&rdquo;). That itch is what the rest of Series 6
            answers.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 9: Practice — Notebook
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Practice: Explore Your Trained VAE"
            subtitle="Hands-on in a Colab notebook"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Use the VAE you trained in the{' '}
              <strong>Variational Autoencoders</strong> notebook. This notebook
              has four parts, progressing from fully guided to independent:
            </p>
            <ul className="list-disc list-inside text-muted-foreground space-y-2 ml-4">
              <li>
                <strong>Part 1 (guided):</strong> Sample 25 random z vectors
                from <InlineMath math="\mathcal{N}(0,1)" />, decode them,
                display as a 5x5 grid. Observe: recognizable items, variety,
                some oddities.
              </li>
              <li>
                <strong>Part 2 (supported):</strong> Pick two test images from
                different categories. Encode both. Create an interpolation strip
                with 8 steps. Display the smooth transition. Try multiple pairs.
              </li>
              <li>
                <strong>Part 3 (supported):</strong> Latent arithmetic. Encode
                items from different categories. Try vector subtraction and
                addition to transfer attributes. Observe: sometimes works,
                sometimes noisy.
              </li>
              <li>
                <strong>Part 4 (independent):</strong> Visualize the latent
                space with t-SNE. Encode the full test set. Plot with category
                colors. Interpret the structure you see.
              </li>
            </ul>
          </div>

          <div className="rounded-lg border-2 border-primary/50 bg-primary/5 p-6 mt-4">
            <div className="space-y-4">
              <p className="text-muted-foreground">
                Open the notebook and explore what your VAE learned.
              </p>
              <a
                href="https://colab.research.google.com/github/brennancheung/CourseAI/blob/main/notebooks/6-1-4-exploring-latent-spaces.ipynb"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 px-4 py-2 rounded-md bg-primary text-primary-foreground font-medium text-sm hover:bg-primary/90 transition-colors"
              >
                <ExternalLink className="w-4 h-4" />
                Open in Google Colab
              </a>
              <p className="text-xs text-muted-foreground">
                You will need the trained VAE from the previous notebook. If you
                saved a checkpoint, load it. Otherwise, retrain (it only takes a
                few minutes on a GPU).
              </p>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Notebook Tips">
            Part 1 is the emotional payoff&mdash;your first real generated
            images. Take a moment to appreciate that these did not exist before
            you ran the cell. Parts 2&ndash;4 deepen your understanding of the
            space your model created.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 10: Summary
          ================================================================ */}
      <Row>
        <Row.Content>
          <SummaryBlock
            title="Key Takeaways"
            items={[
              {
                headline:
                  `A trained VAE’s latent space is a continuous, organized space where you can sample, interpolate, and do arithmetic.`,
                description:
                  'Sample from N(0,1), decode, and you get novel images. The KL regularizer organized the space so every region is meaningful.',
              },
              {
                headline:
                  'Interpolation in latent space produces coherent transitions; pixel-space interpolation does not.',
                description:
                  'Pixel averaging creates ghostly overlays. Latent averaging creates plausible intermediate images because the decoder understands the space between encoded points.',
              },
              {
                headline:
                  'The structure in the latent space reflects what the network learned about the data.',
                description:
                  'Similar items cluster together. Directions between items capture relationships. The t-SNE visualization makes this structure visible.',
              },
              {
                headline:
                  'VAE generation works but has a fundamental quality ceiling from the reconstruction-vs-KL tradeoff.',
                description:
                  'The blurriness is not a training failure—it is the price of a smooth, sampleable latent space. You cannot fix it with more epochs.',
              },
              {
                headline:
                  'Diffusion models overcome this quality limitation.',
                description:
                  `The VAE proved the concept: generation by sampling from a learned latent space works. Diffusion models deliver the quality. That’s Module 6.2.`,
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
              &ldquo;You learned to sample from a distribution and create things
              that have never existed. That is the core of generative AI.
              Everything from here forward is about doing it better.&rdquo;
            </strong>
          </InsightBlock>
        </Row.Content>
      </Row>

      {/* ================================================================
          What's Next — Bridge to Diffusion
          ================================================================ */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground font-medium text-foreground">
              What comes next
            </p>
            <p className="text-muted-foreground">
              You have experienced generation: sampling from a learned latent
              space produces novel images. But those images are blurry, and you
              now know why&mdash;the reconstruction-vs-KL tradeoff is
              fundamental to how VAEs work. In Module 6.2, you will learn a
              completely different approach: instead of compressing images into
              a latent space and sampling, you will learn to{' '}
              <strong>destroy images with noise</strong> and then{' '}
              <strong>train a network to undo the destruction, step by
              step</strong>. This is diffusion&mdash;and it is how modern
              image generators achieve stunning quality.
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* ================================================================
          Module Complete
          ================================================================ */}
      <Row>
        <Row.Content>
          <ModuleCompleteBlock
            module="6.1"
            title="Generative Foundations"
            achievements={[
              'Discriminative vs generative framing: from P(y|x) to learning and sampling from P(x)',
              'Built and trained an autoencoder on Fashion-MNIST: encoder-decoder, bottleneck, reconstruction loss',
              'Converted it to a VAE: distributional encoding (mu + logvar), KL regularizer, reparameterization trick',
              'Explored the latent space hands-on: sampled novel images, interpolated between items, performed vector arithmetic',
              'Recognized the VAE quality ceiling (blurriness from reconstruction-vs-KL tradeoff) and why diffusion takes a different approach',
            ]}
            nextModule="6.2"
            nextTitle="Diffusion Models"
          />
        </Row.Content>
      </Row>
    </LessonLayout>
  )
}

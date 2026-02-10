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
import { AutoencoderBottleneckWidget } from '@/components/widgets/AutoencoderBottleneckWidget'
import { ExercisePanel } from '@/components/widgets/ExercisePanel'
import { MermaidDiagram } from '@/components/widgets/MermaidDiagram'
import { CodeBlock } from '@/components/common/CodeBlock'
import { ExternalLink } from 'lucide-react'
import 'katex/dist/katex.min.css'
import { InlineMath, BlockMath } from 'react-katex'

/**
 * Autoencoders
 *
 * Lesson 2 in Module 6.1 (Generative Foundations).
 * Teaches the encoder-decoder architecture as a way to force
 * a neural network to learn a compressed representation through
 * a bottleneck, and that reconstruction loss measures how well
 * the compressed representation preserves what matters.
 *
 * Core concepts at DEVELOPED:
 * - Encoder-decoder architecture (hourglass)
 * - Bottleneck / latent representation
 * - Reconstruction loss (MSE on pixels, target IS the input)
 *
 * Previous: From Classification to Generation (module 6.1, lesson 1)
 * Next: Variational Autoencoders (module 6.1, lesson 3)
 */

export function AutoencodersLesson() {
  return (
    <LessonLayout>
      {/* ================================================================
          Header
          ================================================================ */}
      <Row>
        <Row.Content>
          <LessonHeader
            title="Autoencoders"
            description="Force a neural network through a tiny bottleneck. What it learns to keep is what matters."
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
            Understand the encoder-decoder architecture as a way to force a neural
            network to learn a compressed representation of its input through a
            bottleneck, and recognize that reconstruction loss measures how well the
            compressed representation preserves what matters.
          </ObjectiveBlock>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="What You Already Know">
            You&apos;ve built CNNs (the encoder is one), used MSE loss (that&apos;s
            the reconstruction objective), and trained models end-to-end in PyTorch.
            The autoencoder reuses all of this&mdash;the new idea is the architecture
            and what it forces the network to learn.
          </TipBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <ConstraintBlock
            title="Scope for This Lesson"
            items={[
              'The encoder-decoder architecture as an hourglass that compresses and reconstructs',
              'The bottleneck / latent representation as a learned compression of what matters',
              'Reconstruction loss (MSE on pixels) as the training objective',
              'What the bottleneck forces the network to learn (and lose)',
              'Why the autoencoder is NOT a generative model (yet)',
              'NOT: variational autoencoders, KL divergence, or probabilistic encoding\u2014that is the next lesson',
              'NOT: sampling from the latent space to generate novel images (the autoencoder cannot do this)',
              'NOT: denoising autoencoders, sparse autoencoders, or other variants',
              'NOT: latent space interpolation or arithmetic\u2014that comes later',
            ]}
          />
        </Row.Content>
      </Row>

      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              In <strong>From Classification to Generation</strong>, you learned what
              generation means: sampling from a learned distribution of data. You
              understand the goal, but you have no idea how a neural network could
              actually learn a distribution over 784-dimensional images. This lesson
              introduces the simplest architecture that gets us part of the way there.
            </p>
            <p className="text-muted-foreground">
              By the end, you&apos;ll have built an autoencoder on Fashion-MNIST in a
              Colab notebook and seen how it compresses and reconstructs images. You&apos;ll
              also see why it <em>fails</em> at generation&mdash;and that failure is
              exactly what motivates the next lesson.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The Road So Far">
            Lesson 1: what generation <em>is</em> (sampling from P(x)).
            This lesson: the first architecture that creates a compressed
            representation we could <em>eventually</em> sample from. But not yet.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 2: Hook — "Describe a shoe in 32 numbers"
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Describe a Shoe in 32 Words"
            subtitle="Compression is a choice: what do you keep?"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Imagine someone shows you a sneaker and says: &ldquo;Describe this shoe
              using only 32 words so that I can redraw it.&rdquo; Which features
              would you mention? Overall shape? Sole thickness? The angle of the toe?
              How dark the fabric is?
            </p>
            <p className="text-muted-foreground">
              You cannot transmit all 784 pixel values&mdash;you only get 32 words.
              So you have to decide <strong>what matters</strong>. The overall silhouette
              is probably more important than the exact brightness of pixel (14, 7). The
              shape of the sole matters more than subtle shading in the middle.
            </p>
            <p className="text-muted-foreground font-medium text-foreground">
              A neural network can learn to make this choice automatically.
            </p>
            <p className="text-muted-foreground">
              A neural network does the same thing, but instead of 32 words, it uses
              32 numbers. Force an image through a 32-number bottleneck, train the
              network to reconstruct the original, and it learns which 32 numbers
              matter most. The encoder is the describer; the decoder is the artist;
              the 32-number limit is the bottleneck.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Compression as Learning">
            The bottleneck is not an obstacle&mdash;it is the learning mechanism.
            Without it, the network could just copy every pixel. The constraint
            forces it to discover what matters about the data.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 3: The Encoder — A CNN You Already Know
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Encoder: A CNN You Already Know"
            subtitle="Same building blocks, different endpoint"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The encoder is a CNN. The exact same pattern from your MNIST and
              Fashion-MNIST projects: <InlineMath math="\text{Conv2d} \rightarrow \text{ReLU} \rightarrow \text{Conv2d} \rightarrow \text{ReLU} \rightarrow \text{Flatten} \rightarrow \text{Linear}" />.
              Spatial dimensions shrink, channels grow&mdash;exactly what you&apos;ve done before.
            </p>
            <p className="text-muted-foreground">
              The difference: instead of ending at 10 class probabilities, the encoder
              ends at a small vector&mdash;the bottleneck. In a classifier, the final
              layer maps features to 10 class probabilities. In an autoencoder, the final
              encoder layer maps features to <InlineMath math="N" /> latent dimensions.
              Same building blocks, different endpoint.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Same Pattern, New Goal">
            &ldquo;Spatial shrinks, channels grow&rdquo;&mdash;you know
            this from building CNNs. The encoder does the same thing but
            aims for a tiny vector instead of class logits.
          </TipBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground font-medium text-foreground">
              Dimension walkthrough for Fashion-MNIST:
            </p>
            <div className="py-3 px-4 bg-muted/50 rounded-lg font-mono text-sm text-muted-foreground">
              <div className="space-y-1">
                <p>1 &times; 28 &times; 28 <span className="text-muted-foreground/60">(input image: 784 values)</span></p>
                <p>&darr; Conv2d(1, 16, 3, stride=2, padding=1) + ReLU</p>
                <p>16 &times; 14 &times; 14 <span className="text-muted-foreground/60">(3,136 values)</span></p>
                <p>&darr; Conv2d(16, 32, 3, stride=2, padding=1) + ReLU</p>
                <p>32 &times; 7 &times; 7 <span className="text-muted-foreground/60">(1,568 values)</span></p>
                <p>&darr; Flatten</p>
                <p>1,568 <span className="text-muted-foreground/60">(flat vector)</span></p>
                <p>&darr; Linear(1568, 32)</p>
                <p className="text-violet-400 font-bold">32 <span className="text-violet-400/60">(bottleneck!)</span></p>
              </div>
            </div>
            <p className="text-muted-foreground">
              From 784 pixel values down to 32 numbers. The network must decide what
              to keep.
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 4: The Bottleneck — What 32 Numbers Can Capture
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Bottleneck: What 32 Numbers Can Capture"
            subtitle="Learned compression, not fixed rules"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The bottleneck is the key architectural choice. 784 pixels compressed to
              32 numbers. If the network can reconstruct a recognizable image from those
              32 numbers, then those numbers must capture the essential structure of the
              input&mdash;the stroke angle of a 7, the collar shape of a shirt, the sole
              thickness of a sneaker.
            </p>
            <p className="text-muted-foreground">
              You might think: &ldquo;This is like JPEG compression.&rdquo; That&apos;s a
              useful starting point, but there&apos;s a crucial difference. JPEG applies the
              same fixed transform (the DCT) to every image regardless of content. An
              autoencoder trained on shoes learns to preserve <em>shoe-relevant</em>{' '}
              features. One trained on faces learns to preserve <em>face-relevant</em>{' '}
              features. The compression rules are <strong>learned from the data</strong>,
              not hand-designed.
            </p>
            <p className="text-muted-foreground">
              These 32 numbers are not a lookup table of &ldquo;the 32 most common pixel
              patterns.&rdquo; They are a continuous representation&mdash;each number
              captures a learned feature of the input. Together, they form a{' '}
              <strong>latent code</strong>: a compact description of what matters about
              this specific image.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Not Like JPEG">
            JPEG uses fixed, hand-designed rules for every image. The
            autoencoder learns <em>data-specific</em> compression. An
            autoencoder trained on Fashion-MNIST would produce terrible
            results on face photos&mdash;because it learned clothing features,
            not face features.
          </WarningBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              This connects directly to <strong>From Classification to Generation</strong>:
              the bottleneck forces structure learning. The network cannot memorize 60,000
              training images in 32 numbers. It must learn what shoes have in common&mdash;shape,
              sole, opening&mdash;and represent each shoe as a combination of those learned
              features.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Architecture Encodes Assumptions">
            The bottleneck encodes the assumption that the real variety of
            images is much smaller than the space of all possible pixel
            combinations. 784 pixels could produce any random noise pattern,
            but actual clothing images cluster in a tiny region. The
            network&apos;s job is to find that region.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 5: The Decoder — Going Back Up
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Decoder: Going Back Up"
            subtitle="From 32 numbers back to an image"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The decoder reverses the encoder. It takes the 32-number latent code and
              reconstructs a 28&times;28 image. But the encoder <em>shrank</em> spatial
              dimensions (28&times;28 &rarr; 14&times;14 &rarr; 7&times;7). How does the
              decoder <em>grow</em> them back?
            </p>
            <p className="text-muted-foreground">
              This is the one genuinely new piece:{' '}
              <strong>ConvTranspose2d</strong>&mdash;a learned upsampling operation. Where
              Conv2d asks &ldquo;what pattern is here?&rdquo; and produces a smaller output,
              ConvTranspose2d asks &ldquo;what should this region look like?&rdquo; and
              produces a <em>larger</em> output. Think of it as convolution in reverse:
              it takes a small feature map and expands it into a larger one.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="ConvTranspose2d">
            You don&apos;t need to understand the math of transposed convolution
            in detail. The key idea: it takes a small spatial feature map and
            produces a larger one. The opposite of what Conv2d + stride does.
          </TipBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground font-medium text-foreground">
              Dimension walkthrough (decoder):
            </p>
            <div className="py-3 px-4 bg-muted/50 rounded-lg font-mono text-sm text-muted-foreground">
              <div className="space-y-1">
                <p className="text-violet-400 font-bold">32 <span className="text-violet-400/60">(bottleneck)</span></p>
                <p>&darr; Linear(32, 1568)</p>
                <p>1,568 <span className="text-muted-foreground/60">(flat vector)</span></p>
                <p>&darr; Unflatten &rarr; 32 &times; 7 &times; 7 <span className="text-muted-foreground/60">(reshape back to spatial&mdash;reverse of Flatten)</span></p>
                <p>32 &times; 7 &times; 7</p>
                <p>&darr; ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1) + ReLU</p>
                <p>16 &times; 14 &times; 14</p>
                <p>&darr; ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1) + Sigmoid</p>
                <p>1 &times; 28 &times; 28 <span className="text-muted-foreground/60">(reconstructed image)</span></p>
              </div>
            </div>
            <p className="text-muted-foreground">
              Notice the decoder ends with <strong>Sigmoid</strong>, not ReLU. Pixel values
              must be in [0, 1], and Sigmoid maps everything to that range. The encoder and
              decoder are roughly symmetric in shape, but they are not constrained to be
              perfect mirrors&mdash;the decoder figures out the best way to reconstruct given
              the latent code.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Not Perfect Mirrors">
            Our autoencoder is symmetric by design&mdash;but that is a simplicity
            choice, not a requirement. You could use three conv layers in the
            encoder and only two ConvTranspose2d layers in the decoder, and the
            network would still learn to reconstruct. Different activation functions,
            different layer counts&mdash;the decoder&apos;s job is to reconstruct,
            not to literally reverse each encoder operation.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Architecture Diagram
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Full Architecture"
            subtitle="The hourglass shape"
          />
          <MermaidDiagram chart={`
            graph LR
              A["Input<br/>1×28×28"] --> B["Conv+ReLU<br/>16×14×14"]
              B --> C["Conv+ReLU<br/>32×7×7"]
              C --> D["Flatten<br/>1568"]
              D --> E["<b>Bottleneck</b><br/>32"]
              E --> F["Linear<br/>1568"]
              F --> G["Unflatten<br/>32×7×7"]
              G --> H["ConvT+ReLU<br/>16×14×14"]
              H --> I["ConvT+Sigmoid<br/>1×28×28"]

              style E fill:#7c3aed,stroke:#a78bfa,color:#fff
              style A fill:#1e1e2e,stroke:#6366f1,color:#c4b5fd
              style I fill:#1e1e2e,stroke:#6366f1,color:#c4b5fd
          `} />
          <p className="text-muted-foreground text-sm mt-3">
            The hourglass shape: wide input, narrow bottleneck, wide output.
            The encoder (left half) is a CNN you already know. The decoder (right
            half) reverses the spatial dimensions using ConvTranspose2d.
          </p>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Same Building Blocks">
            Conv2d, ReLU, Linear, Flatten&mdash;you&apos;ve used all of these.
            The new pieces are ConvTranspose2d (learned upsampling) and the
            architecture itself (hourglass, not funnel).
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 6: Reconstruction Loss — The Target IS the Input
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Reconstruction Loss: The Target IS the Input"
            subtitle="No labels needed"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Every loss function you have used so far compares a prediction to a{' '}
              <strong>label</strong>. Cross-entropy compared class predictions to the
              correct class. MSE compared predicted values to target values. The
              autoencoder&apos;s loss is different in a subtle but important way:
            </p>
            <div className="py-4 px-6 bg-muted/50 rounded-lg">
              <BlockMath math="\mathcal{L} = \frac{1}{n}\sum_{i=1}^{n}(x_i - \hat{x}_i)^2" />
            </div>
            <p className="text-muted-foreground">
              Where <InlineMath math="x" /> is the input image and{' '}
              <InlineMath math="\hat{x}" /> is the reconstruction. The
              &ldquo;correct answer&rdquo; is the thing the network was given.
              No labels needed&mdash;the data is its own target. This is
              sometimes called <strong>self-supervised learning</strong>&mdash;the
              labels come from the data itself, not from human annotation.
            </p>
            <p className="text-muted-foreground">
              The loss measures what the bottleneck fails to preserve. High loss means the
              latent code did not capture enough. Low loss means the 32 numbers preserved
              what matters.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="A Subtle Shift">
            Classification: compare prediction to a <em>label</em>.
            <br />
            Autoencoder: compare prediction to the <em>input</em>.
            <br /><br />
            Same MSE formula, fundamentally different target. The autoencoder
            does not need anyone to label the data.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 7: Check — Predict-and-verify
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Check: What Does the Bottleneck Preserve?"
            subtitle="Test your mental model before exploring"
          />
          <GradientCard title="Predict and Verify" color="emerald">
            <div className="space-y-3 text-sm">
              <p>
                You train an autoencoder with a <strong>4-dimensional bottleneck</strong>{' '}
                on Fashion-MNIST. What do you expect the reconstructions to look like?
              </p>
              <details className="mt-3">
                <summary className="font-medium cursor-pointer text-primary">
                  Reveal
                </summary>
                <div className="mt-2 space-y-2">
                  <p>
                    Very blurry. The shapes should be recognizable&mdash;you can tell a
                    trouser from a sneaker&mdash;but all fine detail is lost. Only 4
                    numbers to capture everything about a 784-pixel image. The network
                    keeps the broadest features (overall shape, size) and discards
                    everything else.
                  </p>
                </div>
              </details>
              <p className="mt-4">
                Now the bottleneck is <strong>256 dimensions</strong>. Better or worse
                reconstruction? What is the tradeoff?
              </p>
              <details className="mt-3">
                <summary className="font-medium cursor-pointer text-primary">
                  Reveal
                </summary>
                <div className="mt-2 space-y-2">
                  <p>
                    Much better reconstruction&mdash;the image is sharper and more
                    detailed. But the representation is less compressed. With 256
                    dimensions (33% of the input), the network is under less pressure
                    to learn what truly matters. The tradeoff: quality vs. meaningful
                    compression.
                  </p>
                </div>
              </details>
            </div>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 8: Interactive Widget
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Explore: The Bottleneck Tradeoff"
            subtitle="See what different bottleneck sizes preserve"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Drag the slider to change the bottleneck size. Watch how
              reconstruction quality changes as you give the network more or fewer
              dimensions to work with. Switch between different clothing items to
              see how the same bottleneck handles different shapes.
            </p>
          </div>
        </Row.Content>
      </Row>

      <Row>
        <Row.Content>
          <ExercisePanel
            title="Autoencoder Bottleneck Visualizer"
            subtitle="Adjust the bottleneck size and observe reconstruction quality"
          >
            <AutoencoderBottleneckWidget />
          </ExercisePanel>
        </Row.Content>
        <Row.Aside>
          <TryThisBlock title="Experiments">
            <ul className="space-y-2 text-sm">
              <li>{'\u2022'} Start at <strong>4 dimensions</strong>. What survives the
                compression? Can you still tell which category the item belongs to?</li>
              <li>{'\u2022'} Slide to <strong>32</strong>. What details appear that
                were missing at 4?</li>
              <li>{'\u2022'} Try <strong>256</strong>. Is there a point where more
                dimensions stop helping noticeably?</li>
              <li>{'\u2022'} Switch between items. Does the T-shirt compress
                differently than the sneaker? Why might that be?</li>
            </ul>
          </TryThisBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 9: PyTorch Code
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Autoencoder in PyTorch"
            subtitle="Every piece is familiar except ConvTranspose2d"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Here is the complete autoencoder as an <code>nn.Module</code>. Read
              through it&mdash;you should recognize every piece except{' '}
              <code>ConvTranspose2d</code>.
            </p>
          </div>
          <CodeBlock
            language="python"
            filename="autoencoder.py"
            code={`import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, bottleneck_size=32):
        super().__init__()

        # Encoder: same CNN pattern you already know
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  # 28x28 -> 14x14
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), # 14x14 -> 7x7
            nn.ReLU(),
            nn.Flatten(),                                # 32*7*7 = 1568
            nn.Linear(32 * 7 * 7, bottleneck_size),     # -> bottleneck
        )

        # Decoder: reverse the encoder
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_size, 32 * 7 * 7),     # bottleneck ->
            nn.Unflatten(1, (32, 7, 7)),                 # reshape to spatial
            nn.ConvTranspose2d(32, 16, 3, stride=2,      # 7x7 -> 14x14
                               padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2,       # 14x14 -> 28x28
                               padding=1, output_padding=1),
            nn.Sigmoid(),  # pixel values in [0, 1]
        )

    def forward(self, x):
        latent = self.encoder(x)    # compress
        recon = self.decoder(latent) # reconstruct
        return recon`}
          />
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Reading the Code">
            The encoder is a CNN ending at a linear layer&mdash;exactly like your
            MNIST classifier, but targeting a bottleneck vector instead of class
            logits. The decoder mirrors it with ConvTranspose2d and ends with
            Sigmoid.
          </TipBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              And the training loop is identical to every model you have trained:
            </p>
          </div>
          <CodeBlock
            language="python"
            filename="train.py"
            code={`criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(num_epochs):
    for images, _ in train_loader:  # labels ignored!
        recon = model(images)
        loss = criterion(recon, images)  # target IS the input

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()`}
          />
          <p className="text-muted-foreground mt-4">
            Notice: the labels from the DataLoader are ignored (the underscore{' '}
            <code>_</code>). The loss compares the reconstruction to the{' '}
            <strong>input</strong>, not to any label. The data is its own target.
          </p>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Labels Ignored">
            <code>for images, _ in train_loader</code>&mdash;the underscore
            is telling. Classification needs labels. The autoencoder does not.
            The input IS the target.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 10: Why the Autoencoder is NOT Generative (Yet)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Why the Autoencoder Is NOT Generative"
            subtitle="The critical limitation that motivates the next lesson"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The autoencoder compresses and reconstructs. It does <strong>not</strong>{' '}
              generate. Here is the test: take a random vector of 32 numbers and feed it
              to the decoder. What comes out?
            </p>
            <p className="text-muted-foreground font-medium text-foreground">
              Garbage. Not a recognizable image.
            </p>
            <p className="text-muted-foreground">
              Why? Because the latent space only has meaningful values where real images
              were encoded. The 60,000 training images each map to a specific point in
              32-dimensional space. But the spaces <em>between</em> those points are
              uncharted territory. A random vector almost certainly lands in a gap where
              no real image was ever encoded.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Autoencoder &ne; Generative Model">
            The autoencoder can reconstruct images it has seen (or close
            variations). It cannot generate novel images from scratch. Random
            latent codes produce noise, not clothing.
          </WarningBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <ComparisonRow
            left={{
              title: 'Encode Real Image',
              color: 'emerald',
              items: [
                'Feed a T-shirt through the encoder',
                'Get a 32-number latent code',
                'Decode it \u2192 recognizable T-shirt',
                'The latent code is meaningful',
              ],
            }}
            right={{
              title: 'Random Latent Code',
              color: 'rose',
              items: [
                'Generate 32 random numbers',
                'Feed them to the decoder',
                'Decode it \u2192 unrecognizable noise',
                'The random point is in a gap',
              ],
            }}
          />
          <div className="space-y-4 mt-4">
            <p className="text-muted-foreground">
              Remember from <strong>From Classification to Generation</strong>:
              generation means sampling from a learned distribution. The
              autoencoder&apos;s latent space is not a distribution you can sample
              from. It is a scattered collection of points with gaps between them.
            </p>
            <p className="text-muted-foreground font-medium text-foreground">
              What if we could organize the latent space so that random points DO
              produce good images?
            </p>
            <p className="text-muted-foreground">
              That is exactly what a Variational Autoencoder does&mdash;and that is the
              next lesson.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The Gap Problem">
            The autoencoder&apos;s latent space is like a map with dots but
            no terrain between them. To generate, we need the entire map
            filled in&mdash;every point should correspond to a plausible
            image.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 11: The Overcomplete Trap
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Overcomplete Trap"
            subtitle="If bigger is better, why not use 784 dimensions?"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Natural intuition: if 32 dimensions gives okay reconstruction and 128
              gives great reconstruction, why not use 784 dimensions&mdash;the same
              size as the input?
            </p>
            <p className="text-muted-foreground">
              Because an <strong>overcomplete</strong> autoencoder (bottleneck &ge; input
              size) can learn the identity function. It copies every pixel through, learning{' '}
              <em>nothing</em> about structure. The reconstruction is perfect but the
              representation is useless. The bottleneck is not a limitation&mdash;it{' '}
              <strong>is</strong> the entire learning mechanism.
            </p>
            <p className="text-muted-foreground">
              This is the same principle as regularization from earlier in the course.
              Dropout and weight decay constrain the model to prevent memorization.
              The bottleneck constrains the autoencoder to prevent copying. Constraints
              force the model to learn generalizable representations instead of memorizing.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Bigger Is Not Better">
            An overcomplete autoencoder (bottleneck &ge; input) can learn the
            identity function&mdash;perfect reconstruction, zero learning.
            The constraint IS the point.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 12: Colab Notebook
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Practice: Build an Autoencoder"
            subtitle="Hands-on in a Colab notebook"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Time to build it yourself. The notebook walks you through three parts:
            </p>
            <ul className="list-disc list-inside text-muted-foreground space-y-1 ml-4">
              <li>
                <strong>Part 1 (guided):</strong> Build the encoder and decoder, define
                MSE reconstruction loss, train for a few epochs on Fashion-MNIST.
              </li>
              <li>
                <strong>Part 2 (supported):</strong> Visualize input vs reconstruction
                for several test images. Compare bottleneck sizes (8, 32, 128) and observe
                the reconstruction quality tradeoff.
              </li>
              <li>
                <strong>Part 3 (supported):</strong> Feed random noise vectors to the
                decoder. Observe the garbage output. Confirm the autoencoder is not
                generative.
              </li>
            </ul>
          </div>

          <div className="rounded-lg border-2 border-primary/50 bg-primary/5 p-6 mt-4">
            <div className="space-y-4">
              <p className="text-muted-foreground">
                Open the notebook and train an autoencoder on Fashion-MNIST.
              </p>
              <a
                href="https://colab.research.google.com/github/brennancheung/CourseAI/blob/main/notebooks/6-1-2-autoencoders.ipynb"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 px-4 py-2 rounded-md bg-primary text-primary-foreground font-medium text-sm hover:bg-primary/90 transition-colors"
              >
                <ExternalLink className="w-4 h-4" />
                Open in Google Colab
              </a>
              <p className="text-xs text-muted-foreground">
                The notebook uses the same PyTorch patterns from every project:
                Dataset, DataLoader, training loop, model.eval() for testing.
              </p>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Notebook Tips">
            The training loop is identical to your previous projects. The
            only new API is <code>nn.ConvTranspose2d</code> and{' '}
            <code>nn.Unflatten</code>. Everything else is familiar.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 13: Summary
          ================================================================ */}
      <Row>
        <Row.Content>
          <SummaryBlock
            title="Key Takeaways"
            items={[
              {
                headline:
                  'The autoencoder is an hourglass: compress through a bottleneck, then reconstruct.',
                description:
                  'The encoder is a CNN you already know. The decoder reverses it with ConvTranspose2d. The bottleneck forces the network to learn what matters about the input.',
              },
              {
                headline:
                  'Reconstruction loss = MSE between input and output. The target IS the input.',
                description:
                  'No labels needed. The data supervises itself. The loss measures what the bottleneck fails to preserve.',
              },
              {
                headline:
                  'The bottleneck creates a latent representation\u2014a compressed code of what matters.',
                description:
                  'The smaller the bottleneck, the more the network must discover what is essential. Too large, and it can just copy pixels (the overcomplete trap).',
              },
              {
                headline:
                  'The autoencoder is NOT a generative model.',
                description:
                  'Random latent codes produce garbage. The latent space has gaps\u2014only points near real encoded images are meaningful. We need to organize the latent space to enable generation.',
              },
            ]}
          />
        </Row.Content>
      </Row>

      {/* Mental model echo */}
      <Row>
        <Row.Content>
          <InsightBlock title="Mental Models">
            <strong>&ldquo;Force it through a bottleneck; it learns what
            matters.&rdquo;</strong> The bottleneck is not an obstacle. It is the
            learning mechanism. Without compression, there is nothing to learn.
            <br /><br />
            <strong>&ldquo;Same building blocks, different question.&rdquo;</strong>{' '}
            Same conv layers, same ReLU, same MSE loss. Different question: not
            &ldquo;what class?&rdquo; but &ldquo;can you reconstruct this?&rdquo;
          </InsightBlock>
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 14: Next step
          ================================================================ */}
      <Row>
        <Row.Content>
          <NextStepBlock
            href="/app/lesson/variational-autoencoders"
            title="Variational Autoencoders"
            description="The autoencoder gives us a latent representation, but we cannot generate from it\u2014random points in latent space produce garbage. What if we could make the latent space smooth and organized, so that every point corresponds to a plausible image? The next lesson makes the autoencoder generative by encoding not to a single point, but to a distribution."
            buttonText="Continue to Variational Autoencoders"
          />
        </Row.Content>
      </Row>
    </LessonLayout>
  )
}

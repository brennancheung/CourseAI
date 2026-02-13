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
  GradientCard,
  ComparisonRow,
  SummaryBlock,
  NextStepBlock,
  ReferencesBlock,
  ModuleCompleteBlock,
} from '@/components/lessons'
import { CodeBlock } from '@/components/common/CodeBlock'
import { MermaidDiagram } from '@/components/widgets/MermaidDiagram'
import 'katex/dist/katex.min.css'
import { InlineMath } from 'react-katex'

/**
 * From Pixels to Latents
 *
 * Lesson 5 in Module 6.3 (Architecture & Conditioning). Lesson 14 overall in Series 6.
 * Cognitive load: CONSOLIDATE (zero new algorithms — synthesizes VAE from 6.1,
 * diffusion from 6.2, and architecture/conditioning from 6.3 into latent diffusion).
 *
 * Core insight: the diffusion algorithm is IDENTICAL in latent space. The only change
 * is running it on 64×64×4 latent tensors instead of 512×512×3 pixel tensors, with a
 * VAE encoder/decoder wrapping the process.
 *
 * Core concepts:
 * - Latent diffusion as an architectural pattern: DEVELOPED
 * - Frozen-VAE pattern: INTRODUCED
 * - SD VAE improvements (perceptual + adversarial loss): MENTIONED
 * - Computational cost reduction: DEVELOPED
 *
 * Previous: Text Conditioning & Guidance (module 6.3, lesson 4)
 * Next: Module 6.4 (Full Stable Diffusion Pipeline Assembly)
 */

export function FromPixelsToLatentsLesson() {
  return (
    <LessonLayout>
      {/* ================================================================
          Section 1: Header
          ================================================================ */}
      <Row>
        <Row.Content>
          <LessonHeader
            title="From Pixels to Latents"
            description="The diffusion algorithm you built is identical in latent space&mdash;the VAE compresses, diffusion denoises, and Stable Diffusion is born."
            category="Architecture & Conditioning"
          />
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 2: Context + Constraints
          ================================================================ */}
      <Row>
        <Row.Content>
          <ObjectiveBlock>
            Understand that latent diffusion runs the <strong>same diffusion
            algorithm</strong> you implemented in Module 6.2, operating on the
            VAE&rsquo;s compressed 64×64×4 latent tensors instead of 512×512×3
            pixel images. By the end, you can trace the complete Stable Diffusion
            pipeline from text prompt to generated image&mdash;and explain why
            every component exists.
          </ObjectiveBlock>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="This Is a CONSOLIDATE Lesson">
            No new algorithms. No new math. The diffusion algorithm, the VAE,
            the U-Net, cross-attention, CFG&mdash;you already know all of them.
            This lesson connects the pieces.
          </TipBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <ConstraintBlock
            title="Scope for This Lesson"
            items={[
              'Combining the VAE (Module 6.1) with the diffusion algorithm (Module 6.2) in latent space',
              'The frozen-VAE pattern: VAE trained separately, frozen during diffusion training',
              'Computational cost savings: 512×512×3 → 64×64×4 (48× compression)',
              'The complete Stable Diffusion pipeline overview',
              'NOT: implementing latent diffusion from scratch—that is the Module 6.4 capstone',
              'NOT: DDIM or accelerated samplers—Module 6.4',
              'NOT: perceptual loss or adversarial training in depth—mentioned only',
              'NOT: SD v1 vs v2 vs XL differences, LoRA, or fine-tuning',
            ]}
          />
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 3: VAE Reactivation (Reinforcement Rule)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Quick Recap: The VAE You Built"
            subtitle="Reactivating Module 6.1 concepts (~10 lessons ago)"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              In <strong>Variational Autoencoders</strong>, you built a VAE that
              learned to compress Fashion-MNIST images into a 32-number latent
              code and reconstruct them. Remember the analogy: describing a shoe
              in 32 words instead of listing 784 pixel values. The encoder
              outputs a <em>distribution</em> (mean + variance), not a fixed
              point&mdash;&ldquo;clouds, not points.&rdquo;
            </p>
            <p className="text-muted-foreground">
              The pipeline you built: image → encoder → (μ, log σ²) →
              reparameterize → z → decoder → reconstructed image. That
              encoder-decoder pipeline is what we are about to repurpose.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Clouds, Not Points">
            The VAE encodes to a <em>distribution</em>, not a single point.
            KL regularization organized the latent space so that every point
            decodes to something meaningful&mdash;no gaps, no garbage.
          </InsightBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The key property you proved in <strong>Exploring Latent
              Spaces</strong>: you interpolated between a T-shirt and a sneaker
              in latent space and got coherent intermediate images at every
              step. You sampled random points and got plausible Fashion-MNIST
              items. The KL regularization organized the space so that{' '}
              <strong>every point decodes to something meaningful</strong>.
            </p>
            <p className="text-muted-foreground">
              But there was a quality ceiling. VAE reconstructions were blurry
              compared to originals. You saw a comparison: your VAE versus
              Stable Diffusion. We said: <em>&ldquo;The VAE proves the concept;
              diffusion delivers the quality.&rdquo;</em> This lesson fulfills
              that promise.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Scale Transition">
            Your VAE compressed 28×28×1 (784 values) to 32 numbers. Stable
            Diffusion&rsquo;s VAE compresses 512×512×3 (786,432 values) to
            64×64×4 (16,384 values). Same idea, much bigger scale. The
            compression ratio is about 48×.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 4: Hook — "Remember How Slow That Was?"
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title='Remember How Slow That Was?'
            subtitle="The pain point from Module 6.2, revisited"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              In <strong>Build a Diffusion Model</strong>, you generated 64
              tiny 28×28 MNIST images. It took minutes. You calculated what
              512×512 would cost and the numbers were terrifying: 512×512 has
              roughly 335× more pixels than 28×28. If your model took T minutes
              for 28×28, pixel-space diffusion at 512×512 would take hundreds of
              times longer <em>per step</em>. At 1,000 steps, generation becomes
              impractical for everyday use.
            </p>
            <p className="text-muted-foreground">
              Every lesson since then has made the architecture more
              capable&mdash;U-Net architecture, timestep conditioning, CLIP,
              cross-attention, classifier-free guidance. But none has addressed
              the speed problem.
            </p>
            <p className="text-muted-foreground">
              What if you did not have to run the U-Net on 512×512 pixel
              tensors at all?
            </p>
            <p className="text-muted-foreground">
              What if you could run it on something <strong>much
              smaller</strong>, while keeping the same image quality?
            </p>
            <p className="text-muted-foreground">
              You already have the tool for compression. You built it in
              Module 6.1.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The Unresolved Pain">
            You felt the slowness. You timed it. You calculated the scaling
            problem. Every lesson since has added capability without addressing
            speed. This lesson finally resolves it.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 5: Explain — The Latent Diffusion Insight (DEVELOPED)
          ================================================================ */}

      {/* 5a: The combination */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Latent Diffusion Insight"
            subtitle="Two tools you already have, combined"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Latent diffusion combines two things you already know: the
              VAE&rsquo;s compression (Module 6.1) and the diffusion algorithm
              (Module 6.2). The pipeline has three stages:
            </p>
            <ol className="list-decimal list-inside text-muted-foreground space-y-2 ml-4">
              <li>
                A pre-trained <strong>VAE encoder</strong> compresses images from
                pixel space to latent space.
              </li>
              <li>
                The <strong>diffusion model</strong> runs entirely in latent
                space&mdash;noising, denoising, everything.
              </li>
              <li>
                A pre-trained <strong>VAE decoder</strong> translates the final
                denoised latent back to pixels.
              </li>
            </ol>
            <p className="text-muted-foreground">
              The VAE is a <strong>translator between two
              languages</strong>: pixel language (what you see) and latent
              language (what the diffusion model speaks). The encoder translates
              pixel → latent. The decoder translates latent → pixel. The
              diffusion model only speaks latent.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Same Orchestra, Smaller Hall">
            The U-Net, conditioning mechanisms, and diffusion algorithm are the
            same orchestra performing the same piece. The latent space is a
            smaller, more acoustically efficient venue. Same music, less energy
            to fill the room.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* 5b: The "of course" chain — intuitive modality */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Think about what you have already proved. In Module 6.1, you
              demonstrated that the VAE&rsquo;s latent space preserves
              everything perceptually meaningful about images&mdash;you
              interpolated between a T-shirt and a sneaker and got coherent
              results at every step. In Module 6.2, you proved that diffusion
              works by iteratively denoising tensors&mdash;any tensors, using
              the same noise schedule and the same reverse-step formula. If the
              latent space preserves what matters, and diffusion only needs to
              denoise tensors, then <strong>of course</strong> you can run
              diffusion on the latent tensors. The latent space contains
              everything the diffusion model needs. The only question was
              whether anyone would think to combine them. Rombach et al. (2022)
              did.
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* 5c: The frozen-VAE pattern */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <WarningBlock title="The Frozen-VAE Pattern">
              The VAE is trained <strong>first</strong>, completely separately,
              on plain image reconstruction. Then it is <strong>frozen</strong>.
              The diffusion model trains on the frozen VAE&rsquo;s latent
              representations. The VAE does not know it will be used for
              diffusion. The diffusion model does not know it is operating on
              VAE latents&mdash;it just sees 64×64×4 tensors.
            </WarningBlock>
            <p className="text-muted-foreground">
              This is a <strong>modular pipeline</strong>, not an end-to-end
              system. The VAE and diffusion model are independent components
              that happen to work beautifully together. When you trained your
              VAE on Fashion-MNIST, you did not know you would later use latent
              codes for anything. The same is true here&mdash;the VAE was
              trained for compression, not for diffusion.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Why Not End-to-End?">
            If diffusion training gradients modified the VAE encoder, the latent
            space geometry would shift, invalidating all the U-Net&rsquo;s
            learned denoising. The VAE and diffusion model would fight each
            other. Freezing the VAE gives the diffusion model a stable target.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* 5c: Pipeline comparison diagram */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-sm font-medium text-foreground">
              Pipeline comparison: pixel-space vs latent-space diffusion
            </p>
            <MermaidDiagram chart={`
graph LR
    subgraph Pixel["Pixel-Space Diffusion"]
        direction LR
        P1["Image\\n512×512×3"] --> P2["Add noise +\\nU-Net denoise\\n×50 steps"] --> P3["Generated\\nimage\\n512×512×3"]
    end

    subgraph Latent["Latent-Space Diffusion"]
        direction LR
        L1["Image\\n512×512×3"] --> L2["VAE\\nEncode"] --> L3["Latent\\n64×64×4"] --> L4["Add noise +\\nU-Net denoise\\n×50 steps"] --> L5["Denoised\\nlatent\\n64×64×4"] --> L6["VAE\\nDecode"] --> L7["Generated\\nimage\\n512×512×3"]
    end

    style P1 fill:#1e293b,stroke:#f59e0b,color:#e2e8f0
    style P2 fill:#1e293b,stroke:#f59e0b,color:#e2e8f0
    style P3 fill:#1e293b,stroke:#f59e0b,color:#e2e8f0
    style L1 fill:#1e293b,stroke:#8b5cf6,color:#e2e8f0
    style L2 fill:#1e293b,stroke:#22c55e,color:#e2e8f0
    style L3 fill:#1e293b,stroke:#8b5cf6,color:#e2e8f0
    style L4 fill:#1e293b,stroke:#8b5cf6,color:#e2e8f0
    style L5 fill:#1e293b,stroke:#8b5cf6,color:#e2e8f0
    style L6 fill:#1e293b,stroke:#22c55e,color:#e2e8f0
    style L7 fill:#1e293b,stroke:#8b5cf6,color:#e2e8f0
`} />
            <p className="text-sm text-muted-foreground italic mt-2">
              The denoising loop (middle) is <strong>identical</strong> in both
              pipelines. The only additions in latent diffusion are{' '}
              <span className="text-emerald-400">VAE encode</span> before and{' '}
              <span className="text-emerald-400">VAE decode</span> after. The
              U-Net processes 64×64×4 tensors instead of 512×512×3.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Same Middle, Different Bookends">
            The entire core of diffusion&mdash;noising, denoising, the training
            loop, the sampling loop&mdash;is unchanged. The VAE just bookends
            the process with encode (before) and decode (after).
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 6: Explain — The Algorithm Does Not Change (core reveal)
          ================================================================ */}

      {/* 6a: Side-by-side training algorithm */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Algorithm Does Not Change"
            subtitle="Side-by-side: the core reveal"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Here is the DDPM training algorithm from{' '}
              <strong>Learning to Denoise</strong>, next to the latent diffusion
              training algorithm. Read them carefully and count the differences:
            </p>
            <ComparisonRow
              left={{
                title: 'Pixel-Space DDPM',
                color: 'amber',
                items: [
                  '1. Sample image x₀ from dataset',
                  '2. Sample t ~ Uniform(1, T)',
                  '3. Sample ε ~ N(0, I)',
                  '4. xₜ = √ᾱₜ · x₀ + √(1−ᾱₜ) · ε',
                  '5. ε_θ = U-Net(xₜ, t)',
                  '6. L = ‖ε − ε_θ‖²',
                  '7. Backprop, update weights',
                ],
              }}
              right={{
                title: 'Latent Diffusion',
                color: 'violet',
                items: [
                  '1. Sample image x₀; z₀ = VAE.encode(x₀) ← NEW',
                  '2. Sample t ~ Uniform(1, T)',
                  '3. Sample ε ~ N(0, I)',
                  '4. zₜ = √ᾱₜ · z₀ + √(1−ᾱₜ) · ε',
                  '5. ε_θ = U-Net(zₜ, t, text_emb)',
                  '6. L = ‖ε − ε_θ‖²',
                  '7. Backprop, update U-Net (NOT VAE) ← NEW',
                ],
              }}
            />
            <p className="text-muted-foreground">
              Steps 2, 3, 4, and 6 are <strong>identical</strong>. The noise
              schedule is the same. The loss function is the same. The
              closed-form forward process formula is the same. The only changes:
              step 1 adds an encode, step 5 uses text conditioning (which you
              learned in the previous lesson), and step 7 does not update the
              frozen VAE.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Same Algorithm, Different Tensors">
            Every formula from Module 6.2 still applies. The closed-form
            shortcut, the noise schedule, the reverse step formula&mdash;all
            unchanged. The only thing that changed is the <strong>size</strong>{' '}
            of the tensors being noised and denoised.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* 6b: Side-by-side sampling algorithm */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-sm font-medium text-foreground">
              Sampling: same story
            </p>
            <ComparisonRow
              left={{
                title: 'Pixel-Space Sampling',
                color: 'amber',
                items: [
                  'Sample x_T ~ N(0, I) shape (3, 512, 512)',
                  'Loop T reverse steps in pixel space',
                  'Return x₀ as the generated image',
                ],
              }}
              right={{
                title: 'Latent-Space Sampling',
                color: 'violet',
                items: [
                  'Sample z_T ~ N(0, I) shape (4, 64, 64) ← SMALLER',
                  'Loop T reverse steps in latent space',
                  'x₀ = VAE.decode(z₀) ← DECODE AT THE END',
                ],
              }}
            />
            <p className="text-muted-foreground">
              Sampling starts from random noise in <strong>latent
              space</strong>, not pixel space. The U-Net denoises in latent
              space. Only the very last step&mdash;after all denoising is
              complete&mdash;touches pixels. The VAE decoder runs{' '}
              <strong>once</strong>, at the end.
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* 6c: The "same building blocks" callback */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              This is the recurring pattern of this entire course. The building
              blocks&mdash;conv layers, MSE loss, attention, the diffusion
              algorithm itself&mdash;do not change. The <em>question</em>{' '}
              changes. In Module 6.2, the question was &ldquo;can you denoise
              pixels?&rdquo; In latent diffusion, the question is &ldquo;can you
              denoise latents?&rdquo; The answer is the same.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Same Building Blocks">
            Same algorithm. Same formulas. Different input space. &ldquo;Same
            building blocks, different question&rdquo;&mdash;the pattern that
            runs through the entire course.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 7: Dimension Walkthrough + Computational Cost
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Numbers: Why Latent Space Is Fast"
            subtitle="Concrete dimensions and computational savings"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Trace the data flow with specific tensor dimensions:
            </p>
            <CodeBlock
              code={`# Encode: pixel space → latent space
image = torch.randn(1, 3, 512, 512)    # 786,432 values
latent = vae.encode(image)              # shape: (1, 4, 64, 64) → 16,384 values
# Compression ratio: 786,432 / 16,384 = 48x

# Diffuse: entirely in latent space
# At timestep t=500 with alpha_bar=0.05 (same formula from Module 6.2):
z_t = sqrt(0.05) * latent + sqrt(0.95) * noise   # shape: (1, 4, 64, 64)
predicted_noise = unet(z_t, t=500, text_emb)      # shape: (1, 4, 64, 64)

# After 50 denoising steps...
denoised_latent = ...                   # shape: (1, 4, 64, 64)

# Decode: latent space → pixel space (runs ONCE at the end)
generated_image = vae.decode(denoised_latent)  # shape: (1, 3, 512, 512)`}
              language="python"
              filename="latent_diffusion_pipeline.py"
            />
            <p className="text-muted-foreground">
              The U-Net processes 64×64 spatial tensors at every denoising
              step. Convolution cost scales roughly with spatial area (H × W):
            </p>
          </div>
        </Row.Content>
      </Row>

      <Row>
        <Row.Content>
          <ComparisonRow
            left={{
              title: 'Pixel-Space U-Net',
              color: 'rose',
              items: [
                'Input: 512 × 512 = 262,144 spatial positions',
                '× 3 channels = 786,432 values per step',
                '× 50 denoising steps',
                'Impractical for everyday use',
              ],
            }}
            right={{
              title: 'Latent-Space U-Net',
              color: 'emerald',
              items: [
                'Input: 64 × 64 = 4,096 spatial positions',
                '× 4 channels = 16,384 values per step',
                '× 50 denoising steps',
                '~48× less data per step',
              ],
            }}
          />
          <div className="space-y-4 mt-4">
            <p className="text-muted-foreground">
              In your capstone, you generated 28×28 images and it took minutes.
              At 512×512 in pixel space, the spatial area is{' '}
              <InlineMath math="(512/28)^2 \approx 335" />× larger. At 64×64
              in latent space, the spatial area is only{' '}
              <InlineMath math="(64/28)^2 \approx 5" />× larger than your
              capstone. The difference between &ldquo;impractical&rdquo; and
              &ldquo;runs on a consumer GPU&rdquo; is the VAE compression.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="The VAE Runs Once">
            The encode and decode are each a <strong>single</strong> forward
            pass&mdash;cheap compared to 50 U-Net denoising steps. The
            computational bottleneck is the denoising loop, and that loop
            operates on 48× fewer values.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 8: Elaborate — Why It Actually Works (quality concerns)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Why It Actually Works"
            subtitle="Addressing the quality concern"
          />
          <div className="space-y-4">
            <p className="text-sm font-medium text-foreground">
              Why the latent space is a good place for diffusion
            </p>
            <p className="text-muted-foreground">
              The VAE&rsquo;s latent space is designed to preserve perceptual
              content while discarding imperceptible high-frequency noise. This
              is exactly what diffusion needs&mdash;the U-Net can focus on the
              semantically meaningful aspects of images rather than reproducing
              individual pixel noise patterns.
            </p>
            <p className="text-muted-foreground">
              You proved in <strong>Exploring Latent Spaces</strong> that
              interpolation in latent space produces coherent intermediate
              images. That continuity means the denoising trajectory stays in
              semantically meaningful territory at every step. The VAE built
              the roads. Diffusion walks them.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The Roads Were Already Built">
            The VAE&rsquo;s organized latent space&mdash;continuous,
            gap-free, perceptually meaningful&mdash;is exactly the kind of
            space where iterative denoising works well. Every point along
            the trajectory maps to something coherent.
          </InsightBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-sm font-medium text-foreground">
              SD&rsquo;s VAE is sharper than your toy VAE
            </p>
            <p className="text-muted-foreground">
              Your VAE used MSE reconstruction loss, which optimizes pixel-level
              accuracy and produces blurry results (averaging over uncertainty).
              Stable Diffusion&rsquo;s VAE uses <strong>perceptual
              loss</strong> (comparing features extracted by a pre-trained
              network, not raw pixels) plus <strong>adversarial
              training</strong> (a discriminator that penalizes
              &ldquo;fake-looking&rdquo; reconstructions). The result: much
              sharper reconstructions that faithfully preserve edges, textures,
              and fine detail.
            </p>
            <p className="text-muted-foreground">
              You do not need the details of these losses. The key point: the
              VAE decoder is a <strong>high-fidelity translator</strong> from
              latent space to pixel space. The creative generation happens in
              the diffusion model. The decoder just faithfully renders what
              diffusion produces.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Modularity Is the Point">
            The VAE was not designed &ldquo;for&rdquo; diffusion. It was
            designed to compress images into a good latent representation. Any
            VAE with sufficient reconstruction quality could work. The VAE does
            not know about diffusion; the diffusion model does not know about
            the VAE.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Negative example: skipping the decoder */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-sm font-medium text-foreground">
              What the raw latent output looks like (negative example)
            </p>
            <p className="text-muted-foreground">
              The 64×64×4 latent output of the diffusion process is{' '}
              <strong>not</strong> a small image. It is a 4-channel tensor in an
              abstract learned representation that has no direct visual
              interpretation. You cannot display it as an image. The latent
              space is not a downsampled version of the pixel image&mdash;it
              is a qualitatively different representation. This connects to{' '}
              <strong>Autoencoders</strong>: the bottleneck representation is
              a learned compression, not a thumbnail.
            </p>
            <p className="text-muted-foreground">
              The decoder is essential. It translates from the abstract latent
              representation back to human-interpretable pixels. Without it,
              you have a 64×64×4 tensor of meaningless (to humans) numbers.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Not a Small Image">
            Latent diffusion is <strong>not</strong> &ldquo;just small-resolution
            diffusion.&rdquo; The 64×64×4 latent is a learned abstract
            representation, not a downscaled 64×64 RGB image. The decoder
            is essential, not optional.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 9: Check — Predict and Verify
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Check Your Understanding"
            subtitle="Predict, then verify"
          />
          <GradientCard title="Predict and Verify" color="cyan">
            <div className="space-y-3 text-sm">
              <p>
                <strong>Question 1:</strong> Your colleague says:
                &ldquo;Latent diffusion must produce worse images than
                pixel-space diffusion because you lose information in the VAE
                compression.&rdquo; Is this right?
              </p>
              <details className="mt-2">
                <summary className="font-medium cursor-pointer text-primary">
                  Reveal
                </summary>
                <div className="mt-2">
                  <p>
                    No. Stable Diffusion produces <strong>better</strong> images
                    than pixel-space diffusion at the same compute budget. The
                    latent space discards imperceptible high-frequency detail
                    and concentrates on perceptually meaningful structure. The
                    U-Net spends its capacity on what matters, not pixel noise.
                    The result is both faster <em>and</em> often better, not a
                    quality sacrifice.
                  </p>
                </div>
              </details>

              <p className="mt-4">
                <strong>Question 2:</strong> In the latent diffusion training
                algorithm, step 7 says &ldquo;update U-Net weights (NOT VAE
                weights).&rdquo; Why? What would go wrong if gradients flowed
                back through the VAE encoder?
              </p>
              <details className="mt-2">
                <summary className="font-medium cursor-pointer text-primary">
                  Reveal
                </summary>
                <div className="mt-2">
                  <p>
                    The VAE was pre-trained on a reconstruction objective. Its
                    latent space is organized and continuous. If diffusion
                    training gradients modified the VAE encoder, the latent
                    space geometry would shift, invalidating all the
                    U-Net&rsquo;s learned denoising. The VAE and diffusion
                    model would fight each other. Keeping the VAE frozen gives
                    the diffusion model a stable target to learn in.
                  </p>
                </div>
              </details>

              <p className="mt-4">
                <strong>Question 3:</strong> What would happen if you replaced
                the VAE with a standard autoencoder (no KL regularization) for
                latent diffusion?
              </p>
              <details className="mt-2">
                <summary className="font-medium cursor-pointer text-primary">
                  Reveal
                </summary>
                <div className="mt-2">
                  <p>
                    The autoencoder&rsquo;s latent space has gaps&mdash;regions
                    between encoded training images that are unmapped territory.
                    During the denoising trajectory, the latent vectors would
                    pass through these gaps, producing garbage when decoded. The
                    VAE&rsquo;s KL regularization ensures every point in latent
                    space decodes to something meaningful&mdash;exactly what
                    the denoising trajectory needs. You saw this in{' '}
                    <strong>Autoencoders</strong>: feeding random noise to the
                    autoencoder&rsquo;s decoder produced garbage.
                  </p>
                </div>
              </details>
            </div>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 10: The Complete Picture
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Complete Picture"
            subtitle="Every component of Stable Diffusion, assembled"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              After 14 lessons across 3 modules, you now know every component
              of Stable Diffusion:
            </p>
            <div className="grid gap-3 md:grid-cols-2">
              <GradientCard title="VAE (Module 6.1)" color="emerald">
                <p>
                  Compresses 512×512×3 images to 64×64×4 latents and decodes
                  back. Trained separately, frozen during diffusion.
                </p>
              </GradientCard>
              <GradientCard title="Diffusion Algorithm (Module 6.2)" color="blue">
                <p>
                  Noises and denoises in latent space. Same algorithm you
                  implemented on MNIST.
                </p>
              </GradientCard>
              <GradientCard title="U-Net (Module 6.3, Lesson 1)" color="violet">
                <p>
                  Encoder-decoder with skip connections. Processes 64×64×4
                  latent tensors. Bottleneck for WHAT, skips for WHERE.
                </p>
              </GradientCard>
              <GradientCard title="Timestep Conditioning (Lesson 2)" color="amber">
                <p>
                  Sinusoidal embedding + adaptive group normalization.
                  Tells the U-Net WHEN in the denoising process.
                </p>
              </GradientCard>
              <GradientCard title="CLIP (Lesson 3)" color="cyan">
                <p>
                  Turns text into embeddings that encode visual meaning.
                  Trained contrastively on text-image pairs.
                </p>
              </GradientCard>
              <GradientCard title="Cross-Attention + CFG (Lesson 4)" color="purple">
                <p>
                  Injects text embeddings into the U-Net. CFG amplifies the
                  text signal. WHEN vs WHAT.
                </p>
              </GradientCard>
            </div>
            <GradientCard title="Latent Space (This Lesson)" color="orange">
              <p>
                The stage where it all happens. Fast because small. Works
                because the VAE organized it. 48× fewer values per denoising
                step.
              </p>
            </GradientCard>
            <p className="text-muted-foreground">
              <strong>That is Stable Diffusion.</strong> Every piece exists
              because it solves a specific problem. You understand all of them.
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 11: Check — Transfer Question
          ================================================================ */}
      <Row>
        <Row.Content>
          <GradientCard title="Transfer Question" color="cyan">
            <div className="space-y-3 text-sm">
              <p>
                <strong>Question:</strong> Imagine you want to build a latent
                diffusion model for <strong>audio</strong> instead of images.
                You have an audio VAE that compresses spectrograms to a latent
                representation. What changes in the diffusion pipeline? What
                stays the same?
              </p>
              <details className="mt-2">
                <summary className="font-medium cursor-pointer text-primary">
                  Reveal
                </summary>
                <div className="mt-2">
                  <p>
                    Almost everything stays the same. The diffusion algorithm,
                    noise schedule, training loop, and sampling loop are
                    identical. The U-Net architecture might change shape (1D
                    convolutions instead of 2D), and the VAE is different, but
                    the core diffusion framework is unchanged. Same building
                    blocks, different question.
                  </p>
                </div>
              </details>
            </div>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 12: Practice — Notebook Exercises
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Practice: Hands-On Exercises"
            subtitle="Explore the encode-diffuse-decode pipeline"
          />
          <div className="space-y-4">
            <div className="rounded-lg border-2 border-primary/50 bg-primary/5 p-6">
              <div className="space-y-4">
                <p className="text-muted-foreground">
                  Explore SD&rsquo;s VAE encoder and decoder, visualize the
                  latent space, compute the compression ratio, and trace the
                  full pipeline step by step.
                </p>
                <a
                  href="https://colab.research.google.com/github/brennancheung/CourseAI/blob/main/notebooks/6-3-5-from-pixels-to-latents.ipynb"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-2 px-4 py-2 rounded-md bg-primary text-primary-foreground font-medium text-sm hover:bg-primary/90 transition-colors"
                >
                  Open in Google Colab
                </a>
              </div>
            </div>
            <p className="text-muted-foreground">
              The notebook covers four exercises:
            </p>
            <div className="space-y-3">
              <div className="rounded-lg bg-muted/30 p-4 space-y-2">
                <p className="text-sm font-medium text-foreground">
                  Exercise 1: Explore SD&rsquo;s VAE Encoder and Decoder (Guided)
                </p>
                <p className="text-sm text-muted-foreground">
                  Load a pre-trained Stable Diffusion VAE. Encode a 512×512
                  image. Inspect the latent tensor shape (predict before
                  running!). Decode back. Compare reconstruction to original.
                  Notice the quality is much higher than your toy VAE.
                </p>
              </div>
              <div className="rounded-lg bg-muted/30 p-4 space-y-2">
                <p className="text-sm font-medium text-foreground">
                  Exercise 2: Visualize the Latent Space (Guided)
                </p>
                <p className="text-sm text-muted-foreground">
                  Encode 3&ndash;4 different images. Visualize the 4 channels
                  of each latent tensor as separate heatmaps. Observe that
                  different channels capture different aspects. Interpolate
                  between two encoded images in latent space and decode the
                  intermediates&mdash;a callback to your Module 6.1 experience.
                </p>
              </div>
              <div className="rounded-lg bg-muted/30 p-4 space-y-2">
                <p className="text-sm font-medium text-foreground">
                  Exercise 3: Compute the Compression Ratio (Supported)
                </p>
                <p className="text-sm text-muted-foreground">
                  Calculate 512×512×3 vs 64×64×4. Estimate the FLOP reduction
                  for a U-Net forward pass at each resolution. Connect to your
                  capstone timing: if 28×28 took T seconds per step, estimate
                  512×512 pixel-space and 64×64 latent-space cost.
                </p>
              </div>
              <div className="rounded-lg bg-muted/30 p-4 space-y-2">
                <p className="text-sm font-medium text-foreground">
                  Exercise 4: Trace the Full Pipeline (Independent)
                </p>
                <p className="text-sm text-muted-foreground">
                  Given a pre-trained SD pipeline, manually execute the
                  steps: encode a text prompt with CLIP, encode an image with
                  the VAE, add noise at a specific timestep, run one denoising
                  step with the U-Net, decode the result. Identify which
                  parts are identical to Module 6.2.
                </p>
              </div>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="What to Focus On">
            Exercise 1 proves SD&rsquo;s VAE quality. Exercise 2 connects
            to your Module 6.1 interpolation experience. Exercise 3 grounds
            the 48× compression in real numbers. Exercise 4 ties everything
            together. If time is short, prioritize Exercises 1 and 4.
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
                  'Latent diffusion runs the SAME diffusion algorithm in the VAE\u2019s compressed latent space.',
                description:
                  'Encode: VAE encoder compresses 512\u00d7512\u00d73 images to 64\u00d764\u00d74 latents (48\u00d7 compression). Diffuse: the U-Net noises and denoises in latent space\u2014same formulas, smaller tensors. Decode: VAE decoder translates back to pixels.',
              },
              {
                headline:
                  'The VAE is frozen\u2014trained separately and never modified by the diffusion model.',
                description:
                  'The VAE provides a stable, organized latent space for the diffusion model to learn in. The two components are modular\u2014they don\u2019t know about each other.',
              },
              {
                headline:
                  'Every component of Stable Diffusion exists because it solves a specific problem.',
                description:
                  'VAE for compression, diffusion for generation, U-Net for multi-scale denoising, sinusoidal embeddings for timestep, cross-attention for text, CFG for text amplification. You understand all of them.',
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
              The VAE proves the concept; diffusion delivers the quality.
              Together, they are Stable Diffusion.
            </strong>{' '}
            Same algorithm. Same formulas. Smaller tensors. Faster generation.
            Every component you learned across Modules 6.1, 6.2, and 6.3 is
            now in place.
          </InsightBlock>
        </Row.Content>
      </Row>

      {/* References */}
      <Row>
        <Row.Content>
          <ReferencesBlock
            references={[
              {
                title: 'High-Resolution Image Synthesis with Latent Diffusion Models',
                authors: 'Rombach et al., 2022',
                url: 'https://arxiv.org/abs/2112.10752',
                note: 'The Stable Diffusion paper. Sections 3.1\u20133.2 describe latent diffusion. Section 3.3 covers conditioning. Figure 3 shows the full architecture.',
              },
              {
                title: 'Auto-Encoding Variational Bayes',
                authors: 'Kingma & Welling, 2014',
                url: 'https://arxiv.org/abs/1312.6114',
                note: 'The original VAE paper. The VAE concept underpinning the compression stage of latent diffusion.',
              },
              {
                title: 'Denoising Diffusion Probabilistic Models',
                authors: 'Ho, Jain & Abbeel, 2020',
                url: 'https://arxiv.org/abs/2006.11239',
                note: 'The DDPM algorithm that transfers unchanged to latent space. The training and sampling algorithms you implemented in Module 6.2.',
              },
            ]}
          />
        </Row.Content>
      </Row>

      {/* Module completion */}
      <Row>
        <Row.Content>
          <ModuleCompleteBlock
            module="6.3"
            title="Architecture & Conditioning"
            achievements={[
              'U-Net encoder-decoder with skip connections for multi-scale denoising',
              'Timestep conditioning via sinusoidal embedding + adaptive group normalization',
              'CLIP: contrastive learning for text-image shared embedding space',
              'Cross-attention for spatially-varying text conditioning',
              'Classifier-free guidance for text signal amplification',
              'Latent diffusion: same algorithm, compressed space, 48\u00d7 faster',
            ]}
            nextModule="6.4"
            nextTitle="Full Pipeline Assembly"
          />
        </Row.Content>
      </Row>

      {/* Next Step */}
      <Row>
        <Row.Content>
          <NextStepBlock
            href="/app"
            title="Up Next: Module 6.4—Full Pipeline Assembly"
            description="You understand WHY each piece exists and HOW they connect. Module 6.4 puts them together and lets you use them."
          />
        </Row.Content>
      </Row>
    </LessonLayout>
  )
}

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
import { MermaidDiagram } from '@/components/widgets/MermaidDiagram'
import 'katex/dist/katex.min.css'
import { InlineMath, BlockMath } from 'react-katex'

/**
 * SAM 3 -- The Segment Anything Model
 *
 * Lesson 2 in Module 8.1 (Vision & Vision-Language Models). Series 8 Special Topics.
 * Cognitive load: BUILD (new composition of known components applied to a new task domain).
 *
 * Teaches how SAM brings the "foundation model" approach to image segmentation:
 * - What image segmentation is (gap resolution: MENTIONED -> INTRODUCED)
 * - SAM 1's three-component architecture (ViT encoder + prompt encoder + mask decoder)
 * - Prompt encoding math: Fourier positional encoding, learned type embeddings
 * - Mask decoder internals: three attention operations per layer, token composition, upsampling
 * - Loss function: focal loss, dice loss, combined loss, IoU prediction, multi-mask min-loss
 * - The SA-1B data engine (1B masks via human-AI annotation partnership)
 * - SAM 2's video extension with memory mechanism (memory encoder, memory bank, memory attention)
 * - SAM 3's concept-level segmentation (text prompts, open-vocabulary)
 *
 * Core concepts:
 * - Promptable segmentation: DEVELOPED
 * - SAM's three-component architecture: DEVELOPED
 * - Image segmentation as a task: INTRODUCED
 * - SA-1B data engine: INTRODUCED
 * - SAM 2 video extension: INTRODUCED
 * - SAM 3 concept-level segmentation: INTRODUCED
 *
 * Previous: SigLIP 2 (module 8.1, lesson 1)
 * Next: TBD (Special Topics continues)
 */

export function Sam3Lesson() {
  return (
    <LessonLayout>
      {/* ================================================================
          Section 1: Header
          ================================================================ */}
      <Row>
        <Row.Content>
          <LessonHeader
            title="SAM 3"
            description="How the Segment Anything Model brings the &ldquo;foundation model&rdquo; approach to image segmentation&mdash;one model that segments any object in any image, guided by a prompt."
            category="Vision & Vision-Language Models"
          />
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 2: Context + Constraints
          ================================================================ */}
      <Row>
        <Row.Content>
          <ObjectiveBlock>
            Understand how SAM&rsquo;s promptable segmentation architecture (ViT
            image encoder + prompt encoder + lightweight mask decoder) enables a
            single model to segment any object in any image given a spatial or
            text prompt. Trace the prompt encoding math, the mask decoder&rsquo;s
            three attention operations per layer, and the loss function that
            trains it all.
          </ObjectiveBlock>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="What You Already Know">
            From <LessonLink slug="diffusion-transformers">Diffusion Transformers</LessonLink>, you know ViT and
            &ldquo;tokenize the image.&rdquo; From{' '}
            <LessonLink slug="text-conditioning-and-guidance">Text Conditioning</LessonLink>, you know cross-attention
            (&ldquo;same formula, different source for K and V&rdquo;). From{' '}
            <LessonLink slug="unet-architecture">U-Net Architecture</LessonLink>, you know encoder-decoder
            patterns. All of these carry over directly.
          </TipBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <ConstraintBlock
            title="Scope for This Lesson"
            items={[
              'What image segmentation is and how it differs from classification and detection',
              'SAM 1\'s three-component architecture and why each piece exists',
              'Prompt encoding math: Fourier positional encoding for spatial coordinates, learned type embeddings',
              'Mask decoder internals: the three attention operations per layer, token composition, upsampling',
              'Loss function: focal loss, dice loss, combined loss, IoU prediction, multi-mask minimum-loss training',
              'PyTorch pseudocode for prompt encoding, mask decoder forward pass, and loss computation',
              'SAM 2\'s memory mechanism with technical detail: memory encoder, memory bank, memory attention',
              'SAM 3\'s concept-level segmentation (text prompts, open-vocabulary)',
              'NOT: implementing SAM from scratch or training end-to-end',
              'NOT: DETR or transformer-based detection in depth',
              'NOT: efficiency variants (EfficientSAM, MobileSAM)',
            ]}
          />
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 3: Recap -- Building Blocks
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Recap: The Building Blocks SAM Uses"
            subtitle="ViT, cross-attention, encoder-decoder -- all familiar"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              SAM&rsquo;s architecture is built entirely from components you
              already know. Before diving in, a quick reactivation:
            </p>
            <p className="text-muted-foreground">
              <strong>ViT image encoder:</strong> From{' '}
              <LessonLink slug="diffusion-transformers">Diffusion Transformers</LessonLink>, you know the pattern:
              split the image into patches, treat each patch as a token, process
              with a standard transformer. The output is a spatial grid of
              feature vectors&mdash;one per patch. SAM uses this as its image
              encoder. Same &ldquo;tokenize the image&rdquo; idea.
            </p>
            <p className="text-muted-foreground">
              <strong>Cross-attention:</strong> From{' '}
              <LessonLink slug="text-conditioning-and-guidance">Text Conditioning and Guidance</LessonLink>, you know
              cross-attention as &ldquo;same formula, different source for K and
              V.&rdquo; Q comes from one modality, K and V come from another.
              SAM uses this to fuse prompt information with image
              features&mdash;Q from the prompt tokens, K/V from the image
              embedding.
            </p>
            <p className="text-muted-foreground">
              <strong>Encoder-decoder pattern:</strong> From{' '}
              <LessonLink slug="unet-architecture">U-Net Architecture</LessonLink>, you know the pattern of
              encoding spatial input into a compressed representation, then
              decoding back to spatial output. SAM&rsquo;s mask decoder produces
              spatial masks from encoded features&mdash;but it is much lighter
              than a U-Net (we will see why shortly).
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="Familiar Building Blocks">
            <ul className="space-y-1 text-sm">
              <li>&bull; ViT: &ldquo;tokenize the image&rdquo;</li>
              <li>&bull; Cross-attention: Q from one source, K/V from another</li>
              <li>&bull; Encoder-decoder: spatial input &rarr; compressed &rarr; spatial output</li>
            </ul>
            Every piece of SAM&rsquo;s architecture is something you have built
            or traced before.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 4: The Segmentation Problem (gap resolution)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Segmentation Problem"
            subtitle="What does it mean to segment an image?"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Throughout this course, you have worked with models that look at
              images and produce labels (classification), generate images from
              noise (diffusion), or bridge images and text (CLIP/SigLIP). But
              there is a fundamental computer vision task you have not formally
              confronted: given an image, <strong>which exact pixels belong to a
              specific object?</strong>
            </p>
            <p className="text-muted-foreground">
              This is <strong>image segmentation</strong>, and it is harder than
              it sounds. Consider a photograph of a dog in a park. Three
              different vision tasks give three different outputs:
            </p>
          </div>
        </Row.Content>
      </Row>

      <Row>
        <Row.Content>
          <div className="grid gap-4 md:grid-cols-3">
            <GradientCard title="Classification" color="emerald">
              <p>&ldquo;This image contains a dog.&rdquo;</p>
              <p className="mt-2 text-sm opacity-75">
                One label for the whole image. The simplest output&mdash;but
                says nothing about <em>where</em> the dog is.
              </p>
            </GradientCard>
            <GradientCard title="Detection" color="blue">
              <p>&ldquo;There is a dog here.&rdquo; A bounding box (rectangle) around the dog.</p>
              <p className="mt-2 text-sm opacity-75">
                Locates the object, but the rectangle includes background
                pixels&mdash;the grass behind the dog&rsquo;s legs, the sky
                above its ears.
              </p>
            </GradientCard>
            <GradientCard title="Segmentation" color="violet">
              <p>&ldquo;These exact pixels are the dog.&rdquo; A binary mask&mdash;every pixel labeled dog or not-dog.</p>
              <p className="mt-2 text-sm opacity-75">
                The richest output: the irregular ear shape, the tail curving
                behind the leg, the nose against the grass. Pixel-precise.
              </p>
            </GradientCard>
          </div>
        </Row.Content>
      </Row>

      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Segmentation produces a <strong>mask</strong>&mdash;a grid of 0s
              and 1s the same size as the image, where 1 means &ldquo;this pixel
              belongs to the object.&rdquo; Classification says &ldquo;this is a
              dog.&rdquo; Segmentation says &ldquo;these exact pixels are the
              dog.&rdquo;
            </p>
            <p className="text-muted-foreground">
              This distinction matters: SAM is <strong>not</strong> doing
              classification. It does not output a class label like &ldquo;dog,
              97% confidence.&rdquo; It outputs a binary mask&mdash;which pixels
              belong to the object&mdash;without needing to know{' '}
              <em>what</em> the object is. SAM knows boundaries, not
              categories.
            </p>
            <p className="text-muted-foreground">
              You already know an architecture that produces spatial outputs at
              full resolution&mdash;the <LessonLink slug="unet-architecture">U-Net</LessonLink> produces a grid
              at the original image dimensions. Segmentation is the task that
              needs this kind of spatial, per-pixel output.
            </p>
            <p className="text-muted-foreground">
              There are several variants of segmentation: semantic segmentation
              labels every pixel with a class (&ldquo;sky,&rdquo;
              &ldquo;grass,&rdquo; &ldquo;dog&rdquo;). Instance segmentation
              distinguishes individual objects (dog 1 vs dog 2). Panoptic
              segmentation combines both. SAM does instance-level
              segmentation&mdash;each object gets its own mask.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Segmentation = Per-Pixel Labeling">
            Classification: one label per image. Detection: one box per object.
            Segmentation: one label per pixel. Each step up gives richer spatial
            information about <em>where</em> the object is.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 5: Hook -- The Segmentation Bottleneck
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Segmentation Bottleneck"
            subtitle="Before SAM, segmentation required a specialist for every object"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Before SAM, segmentation required one of two approaches&mdash;both
              painful:
            </p>
            <ul className="list-disc list-inside text-muted-foreground space-y-1 ml-4">
              <li>
                A <strong>specialized model per object category</strong>&mdash;train
                one model for &ldquo;car segmentation,&rdquo; another for
                &ldquo;person segmentation,&rdquo; another for &ldquo;cell
                segmentation&rdquo;
              </li>
              <li>
                <strong>Expensive per-pixel annotation</strong> for every new
                category, followed by training from scratch
              </li>
            </ul>
            <p className="text-muted-foreground">
              No general-purpose segmentation model existed. Draw the parallel to
              language: before GPT-3, NLP had task-specific models&mdash;one for
              sentiment analysis, one for translation, one for summarization.
              GPT-3 showed that one large pretrained model could handle any text
              task via prompting.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="The Pattern You Know">
            <ul className="space-y-1 text-sm">
              <li>&bull; <strong>Before GPT-3:</strong> one model per NLP task</li>
              <li>&bull; <strong>After GPT-3:</strong> one model, many tasks, via prompting</li>
              <li>&bull; <strong>Before SAM:</strong> one model per object category</li>
              <li>&bull; <strong>After SAM:</strong> &hellip;</li>
            </ul>
          </ConceptBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Think of it this way: traditional segmentation models are like
              having <strong>one cookie cutter per shape</strong>. A star cutter,
              a circle cutter, a tree cutter. If you need to cut out a new
              shape, you have to buy (train) a new cutter. Need a flamingo?
              Train a flamingo model. Need a cell under a microscope? Train a
              cell model.
            </p>
            <p className="text-muted-foreground">
              SAM is a <strong>universal cookie cutter</strong>&mdash;a
              programmable one that adjusts its shape based on what you point
              at. You choose the shape (your prompt), press it on the dough
              (the image), and it cuts out the piece (the mask). You do not
              need a different cutter for every object. And with SAM 3, you
              do not even need to point&mdash;you can <em>describe</em> what
              you want (&ldquo;cut out all the stars&rdquo;) and it finds
              them.
            </p>
          </div>

          <GradientCard title="The Question" color="orange">
            <p>
              What if one model could segment <strong>anything</strong>&mdash;a
              dog, a car, a cell in a microscope image, an object it has never
              seen&mdash;guided only by a click or a text description?
            </p>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 6: SAM 1 Architecture Overview
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="SAM 1: The Architecture"
            subtitle="Three familiar components, one new composition"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              SAM&rsquo;s core architecture has three components. Each one is
              familiar&mdash;the innovation is how they are composed:
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* Image Encoder */}
      <Row>
        <Row.Content>
          <GradientCard title="1. Image Encoder (ViT-H)" color="violet">
            <div className="space-y-2">
              <p>
                A MAE (Masked Autoencoder)-pretrained ViT-H (Huge) processes the image{' '}
                <strong>once</strong>.
              </p>
              <ul className="space-y-1 ml-4">
                <li>&bull; Input: 1024&times;1024&times;3 image</li>
                <li>&bull; Output: 64&times;64&times;256 image embedding (a spatial grid of 256-dim feature vectors)</li>
                <li>&bull; This is the heavy computation (~150ms on GPU)</li>
                <li>&bull; Runs <strong>once per image</strong>, regardless of how many prompts follow</li>
              </ul>
              <p className="text-sm opacity-75 mt-2">
                Same &ldquo;tokenize the image&rdquo; idea from{' '}
                <LessonLink slug="diffusion-transformers">Diffusion Transformers</LessonLink>: patches become tokens,
                transformer processes them, output is a grid of features.
              </p>
            </div>
          </GradientCard>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Familiar Architecture">
            The image encoder is a standard ViT. You have traced this exact
            pattern before: patchify, add positional embeddings, process with
            transformer blocks. The output is spatial features&mdash;a grid, not
            a single vector.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Prompt Encoder */}
      <Row>
        <Row.Content>
          <GradientCard title="2. Prompt Encoder" color="blue">
            <div className="space-y-2">
              <p>
                Converts user prompts into tokens the mask decoder can process:
              </p>
              <ul className="space-y-1 ml-4">
                <li>&bull; <strong>Point prompts:</strong> Fourier positional encoding + learned type embedding (foreground vs background)</li>
                <li>&bull; <strong>Box prompts:</strong> two corner points (top-left + bottom-right) with positional encodings</li>
                <li>&bull; <strong>Mask prompts:</strong> downsampled and embedded with lightweight convolutions</li>
                <li>&bull; <strong>Text prompts (SAM 3):</strong> encoded via a language-vision encoder</li>
              </ul>
              <p className="text-sm opacity-75 mt-2">
                Output: a small set of prompt tokens (typically 1&ndash;4 tokens). Lightweight&mdash;runs in milliseconds.
                We will trace the exact encoding math in the next section.
              </p>
            </div>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* Mask Decoder */}
      <Row>
        <Row.Content>
          <GradientCard title="3. Mask Decoder" color="emerald">
            <div className="space-y-2">
              <p>
                A lightweight transformer decoder with{' '}
                <strong>only 2 decoder layers</strong>:
              </p>
              <ul className="space-y-1 ml-4">
                <li>&bull; Uses cross-attention: <strong>Q from prompt tokens, K/V from image embedding tokens</strong></li>
                <li>&bull; Produces a low-resolution mask (256&times;256), upsampled to original image resolution</li>
                <li>&bull; Predicts a confidence score (IoU&mdash;Intersection over Union&mdash;estimating how well the mask fits the true object) for each mask</li>
                <li>&bull; Outputs <strong>multiple candidate masks</strong> (typically 3) to handle ambiguity</li>
                <li>&bull; ~50ms per prompt</li>
              </ul>
              <p className="text-sm opacity-75 mt-2">
                We will trace every operation inside these 2 layers shortly.
              </p>
            </div>
          </GradientCard>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Not a U-Net">
            SAM&rsquo;s mask decoder is <strong>not</strong> a U-Net. It is a
            lightweight 2-layer transformer decoder. No encoder-decoder with
            skip connections. The decoder is deliberately lightweight to enable
            real-time interaction. A full U-Net would be too slow.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* Architecture Diagram */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-sm font-medium text-foreground">
              Architecture overview: three components, asymmetric compute
            </p>
            <MermaidDiagram chart={`
              graph LR
                A["Image (1024x1024x3)"] --> B["ViT-H Image Encoder (~150ms, ONCE)"]
                B --> C["Image Embedding (64x64x256)"]
                D["User Prompt (point/box/text)"] --> E["Prompt Encoder (< 1ms)"]
                E --> F["Prompt Tokens (1-4 tokens)"]
                C --> G["Mask Decoder (~50ms, PER PROMPT)"]
                F --> G
                G --> H["3 Candidate Masks + Confidence Scores"]
            `} />
            <p className="text-muted-foreground text-sm">
              The image encoder is the expensive part and runs <strong>once</strong>.
              The prompt encoder and mask decoder are lightweight and run{' '}
              <strong>per prompt</strong>.
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* The Amortization Insight */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-sm font-medium text-foreground">
              The amortization insight: why SAM is interactive
            </p>
            <p className="text-muted-foreground">
              This asymmetric design is the key to interactive use. Imagine
              exploring a photograph of a desk with a laptop, coffee mug, and
              notebook:
            </p>
            <ul className="list-disc list-inside text-muted-foreground space-y-1 ml-4">
              <li>Image encoding: ~150ms (paid once)</li>
              <li>Click on the coffee mug: ~50ms</li>
              <li>Click on the laptop: ~50ms</li>
              <li>Click on the notebook: ~50ms</li>
              <li><strong>Total for 3 prompts: ~300ms</strong> (not 3 &times; 200ms = 600ms)</li>
            </ul>
            <p className="text-muted-foreground">
              Every subsequent interaction costs only the lightweight decoder
              pass. This is what makes SAM feel instantaneous during interactive
              use.
            </p>
            <p className="text-muted-foreground">
              <em>Of course</em> you would separate the heavy encoder from the
              light decoder&mdash;you want to click around exploring, not wait
              150ms per click.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Architecture Encodes Assumptions">
            Remember &ldquo;architecture encodes assumptions about data&rdquo;
            from the CNN module? SAM&rsquo;s asymmetric design encodes the
            assumption that <strong>you will query the same image multiple times
            with different prompts</strong>. The heavy work happens once; each
            prompt is cheap.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 7: Prompt Encoding -- From Clicks to Tokens (NEW)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Prompt Encoding: From Clicks to Tokens"
            subtitle="How a pixel coordinate becomes a 256-dimensional vector"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              When you click a point on an image, that click needs to become a
              token the mask decoder can process. The question is: how does a
              pixel coordinate like (342, 517) become a 256-dimensional vector?
            </p>
            <p className="text-muted-foreground">
              The answer uses an idea you already know:{' '}
              <strong>sinusoidal positional encoding</strong>. In{' '}
              <strong>The Transformer Block</strong>, you saw how sequence
              position is encoded using sine and cosine functions at different
              frequencies. SAM uses the same idea, but encodes{' '}
              <strong>spatial position</strong> (x, y coordinates) instead of
              sequence position.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="Same Idea, New Domain">
            Transformer sequence position encoding: position &rarr; sin/cos at
            multiple frequencies &rarr; high-dim vector. SAM prompt encoding:
            (x, y) coordinate &rarr; sin/cos at multiple frequencies &rarr;
            high-dim vector. Same mechanism, applied to 2D space instead of 1D
            sequence.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* Fourier positional encoding formula */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-sm font-medium text-foreground">
              Fourier positional encoding for spatial coordinates
            </p>
            <p className="text-muted-foreground">
              First, normalize the pixel coordinates to [0, 1]:
            </p>
            <div className="py-3 px-6 bg-muted/50 rounded-lg">
              <BlockMath math="p_x = \frac{x}{\text{image\_size}}, \quad p_y = \frac{y}{\text{image\_size}}" />
            </div>
            <p className="text-muted-foreground">
              Then encode each normalized coordinate using <InlineMath math="L" /> frequency
              bands (SAM uses <InlineMath math="L = 128" />):
            </p>
            <div className="py-3 px-6 bg-muted/50 rounded-lg">
              <BlockMath math="\text{PE}(p) = \Big[\sin(2^0 \pi p_x),\; \cos(2^0 \pi p_x),\; \sin(2^0 \pi p_y),\; \cos(2^0 \pi p_y),\; \ldots,\; \sin(2^{L-1} \pi p_x),\; \cos(2^{L-1} \pi p_x),\; \ldots \Big]" />
            </div>
            <p className="text-muted-foreground">
              Each coordinate produces <InlineMath math="2L = 256" /> values
              (sin and cos at each of 128 frequencies). For both{' '}
              <InlineMath math="p_x" /> and <InlineMath math="p_y" />, that
              gives <InlineMath math="512" /> values total. A learned linear
              projection maps this 512-dim vector down to 256 dimensions to
              match the prompt token dimension.
            </p>
            <p className="text-muted-foreground">
              Finally, a <strong>learned type embedding</strong> is
              added&mdash;a different 256-dim vector for foreground points vs
              background points:
            </p>
            <div className="py-3 px-6 bg-muted/50 rounded-lg">
              <BlockMath math="\text{prompt\_token} = \text{Linear}_{512 \to 256}\!\big(\text{PE}(p_x, p_y)\big) + \text{type\_embedding}" />
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Why Multiple Frequencies?">
            Low frequencies capture coarse spatial position (&ldquo;left side
            vs right side&rdquo;). High frequencies capture fine position
            (&ldquo;this pixel vs the one next to it&rdquo;). Together, they
            give the model a precise spatial fingerprint for any click
            location.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Concrete traced computation */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-sm font-medium text-foreground">
              Tracing a specific click: pixel (342, 517) on a 1024&times;1024 image
            </p>
            <div className="space-y-3">
              <div className="rounded-lg bg-muted/30 p-3">
                <p className="text-sm">
                  <strong>Step 1 &mdash; Normalize:</strong>{' '}
                  <InlineMath math="p_x = 342 / 1024 = 0.3340" />,{' '}
                  <InlineMath math="p_y = 517 / 1024 = 0.5049" />
                </p>
              </div>
              <div className="rounded-lg bg-muted/30 p-3">
                <p className="text-sm">
                  <strong>Step 2 &mdash; First frequency band</strong> (<InlineMath math="\sigma = 2^0 = 1" />):
                </p>
                <ul className="text-sm ml-4 mt-1 space-y-0.5">
                  <li>&bull; <InlineMath math="\sin(\pi \times 0.3340) = \sin(1.0493) \approx 0.864" /></li>
                  <li>&bull; <InlineMath math="\cos(\pi \times 0.3340) = \cos(1.0493) \approx 0.498" /></li>
                  <li>&bull; <InlineMath math="\sin(\pi \times 0.5049) = \sin(1.5864) \approx 1.000" /></li>
                  <li>&bull; <InlineMath math="\cos(\pi \times 0.5049) = \cos(1.5864) \approx -0.016" /></li>
                </ul>
                <p className="text-sm text-muted-foreground mt-2">
                  Notice the <strong>negative</strong> cosine for y: <InlineMath math="\pi \times 0.5049 = 1.586" /> is
                  just past <InlineMath math="\pi/2 = 1.571" />, where cosine crosses zero and goes
                  negative. This sign flip encodes that the y-coordinate is just
                  past the midpoint&mdash;exactly the kind of fine spatial
                  information these encodings capture.
                </p>
              </div>
              <div className="rounded-lg bg-muted/30 p-3">
                <p className="text-sm">
                  <strong>Step 3 &mdash; Second frequency band</strong> (<InlineMath math="\sigma = 2^1 = 2" />):
                </p>
                <ul className="text-sm ml-4 mt-1 space-y-0.5">
                  <li>&bull; <InlineMath math="\sin(2\pi \times 0.3340) = \sin(2.0987) \approx 0.864" /></li>
                  <li>&bull; <InlineMath math="\cos(2\pi \times 0.3340) = \cos(2.0987) \approx -0.504" /></li>
                  <li>&bull; <InlineMath math="\sin(2\pi \times 0.5049) = \sin(3.1729) \approx -0.031" /></li>
                  <li>&bull; <InlineMath math="\cos(2\pi \times 0.5049) = \cos(3.1729) \approx -1.000" /></li>
                </ul>
                <p className="text-sm text-muted-foreground mt-2">
                  At the second frequency band, all four y-values are negative&mdash;the
                  doubled frequency pushes <InlineMath math="2\pi \times 0.5049 = 3.173" /> past{' '}
                  <InlineMath math="\pi" />, where both sin and cos are negative. Negative values
                  are normal and important: the mix of positive and negative values
                  across frequency bands is what gives each spatial position a unique
                  fingerprint.
                </p>
              </div>
              <div className="rounded-lg bg-muted/30 p-3">
                <p className="text-sm">
                  <strong>Steps 4&ndash;128:</strong> Continue through all 128
                  frequency bands. Higher frequencies oscillate faster, creating
                  finer-grained spatial distinctions. The result is a{' '}
                  <strong>512-dim vector</strong> (4 values per band &times; 128
                  bands).
                </p>
              </div>
              <div className="rounded-lg bg-muted/30 p-3">
                <p className="text-sm">
                  <strong>Step 129 &mdash; Project and add type:</strong> Linear
                  projection maps [512] &rarr; [256]. Add the learned foreground
                  embedding [256]. Final result: <strong>one 256-dim prompt
                  token</strong>.
                </p>
              </div>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Notice the Pattern">
            At the first frequency band, <InlineMath math="\sin(\pi \times 0.5049) \approx 1.0" /> tells
            us the y-coordinate is near the middle, while{' '}
            <InlineMath math="\cos(\pi \times 0.5049) \approx -0.016" /> tells us
            it is <em>just past</em> the midpoint (cosine just crossed zero).
            Higher frequencies will pinpoint the exact position. This is exactly
            how sinusoidal encodings work in transformers&mdash;low frequencies
            for coarse position, high frequencies for fine position.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Box and mask prompt encoding */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-sm font-medium text-foreground">
              Other prompt types
            </p>
            <div className="grid gap-4 md:grid-cols-2">
              <div className="rounded-lg bg-muted/30 p-4">
                <p className="text-sm font-medium">Box prompts: 2 tokens</p>
                <p className="text-sm text-muted-foreground mt-1">
                  A bounding box is encoded as <strong>two corner
                  points</strong> (top-left and bottom-right). Each corner gets
                  its own Fourier positional encoding + a learned type
                  embedding (top-left type vs bottom-right type). Result:{' '}
                  <strong>2 prompt tokens</strong>, each 256-dim.
                </p>
              </div>
              <div className="rounded-lg bg-muted/30 p-4">
                <p className="text-sm font-medium">Mask prompts: spatial embedding</p>
                <p className="text-sm text-muted-foreground mt-1">
                  A coarse mask input is downsampled to 256&times;256 and
                  processed through lightweight convolutions (two 2&times;2
                  stride-2 convs + GELU + one 1&times;1 conv). Output:
                  64&times;64&times;256 spatial embedding, added{' '}
                  <strong>element-wise</strong> to the image embedding (not as
                  a separate token).
                </p>
              </div>
            </div>
          </div>
        </Row.Content>
      </Row>

      {/* PyTorch pseudocode for prompt encoding */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-sm font-medium text-foreground">
              PyTorch pseudocode: encoding a point prompt
            </p>
            <CodeBlock
              code={`def encode_point(point_xy, image_size, foreground=True):
    # Normalize to [0, 1]
    normalized = point_xy / image_size              # [2]

    # Fourier positional encoding (128 frequency bands)
    freqs = 2.0 ** torch.arange(128)               # [128]
    # For each coordinate, compute sin and cos at each frequency
    angles_x = normalized[0] * freqs * math.pi     # [128]
    angles_y = normalized[1] * freqs * math.pi     # [128]
    pe = torch.cat([
        angles_x.sin(), angles_x.cos(),            # [256]
        angles_y.sin(), angles_y.cos(),             # [256]
    ])                                              # [512]

    # Project to prompt token dimension
    pe = linear_projection(pe)                      # [256]

    # Add learned type embedding
    type_emb = fg_embedding if foreground else bg_embedding  # [256]
    return pe + type_emb                            # [256]`}
              language="python"
              filename="prompt_encoder.py"
            />
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Dimension Summary">
            <ul className="space-y-1 text-sm">
              <li>&bull; Point prompt: 1 token [256]</li>
              <li>&bull; Box prompt: 2 tokens [2, 256]</li>
              <li>&bull; Mask prompt: spatial [64, 64, 256] (added to image embedding)</li>
            </ul>
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 8: Mask Decoder Deep Dive (EXPANDED)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Mask Decoder: The Two-Layer Transformer"
            subtitle="Three attention operations per layer, traced with tensor shapes"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The mask decoder is where prompt information meets image
              features. It has only 2 transformer layers, but each layer
              performs <strong>three distinct attention operations</strong>.
              Let us trace exactly what happens.
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* Token composition */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-sm font-medium text-foreground">
              Token composition entering the decoder
            </p>
            <p className="text-muted-foreground">
              Before the first decoder layer, the model assembles a token set
              from two sources:
            </p>
            <div className="grid gap-4 md:grid-cols-2">
              <div className="rounded-lg bg-muted/30 p-4">
                <p className="text-sm font-medium">Prompt tokens (from prompt encoder)</p>
                <p className="text-sm text-muted-foreground mt-1">
                  1&ndash;4 tokens depending on the prompt type. For a single
                  point click: 1 token.
                </p>
              </div>
              <div className="rounded-lg bg-muted/30 p-4">
                <p className="text-sm font-medium">Learned output tokens (fixed)</p>
                <p className="text-sm text-muted-foreground mt-1">
                  3 mask tokens (one per candidate mask) + 1 IoU prediction
                  token = <strong>4 learned tokens</strong>. These are
                  model parameters, not computed from the input.
                </p>
              </div>
            </div>
            <p className="text-muted-foreground">
              For a single point prompt: <InlineMath math="1 + 4 = 5" /> tokens
              total, each 256-dim. These 5 tokens will interact with the 4,096
              image embedding positions (64&times;64 flattened) through
              cross-attention.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Why Learned Mask Tokens?">
            The 3 mask tokens start as learned parameters and{' '}
            <strong>specialize</strong> during training: one learns to produce
            small-region masks, another produces object-level masks, another
            produces object-plus-context masks. Self-attention between them is
            how they &ldquo;discuss&rdquo; which granularity each handles.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Three attention operations per layer */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-sm font-medium text-foreground">
              Each decoder layer: three attention operations
            </p>
            <PhaseCard number={1} title="Self-Attention Among Tokens" subtitle="Tokens communicate with each other" color="cyan">
              <div className="space-y-2">
                <p>
                  Q, K, V all come from the token set (prompt tokens + output
                  tokens). Standard self-attention: the 5 tokens communicate
                  with each other.
                </p>
                <p className="text-sm opacity-75">
                  This is where the mask tokens &ldquo;discuss&rdquo; which
                  granularity each will handle. The prompt token broadcasts its
                  spatial information to all output tokens.
                </p>
              </div>
            </PhaseCard>
            <PhaseCard number={2} title="Token-to-Image Cross-Attention" subtitle="Tokens query the image" color="blue">
              <div className="space-y-2">
                <p>
                  <strong>Q = token embeddings</strong> (5 tokens, each 256-dim).{' '}
                  <strong>K, V = image embedding</strong> (4,096 spatial positions,
                  each 256-dim).
                </p>
                <p className="text-sm opacity-75">
                  Each token queries the image: &ldquo;which spatial locations
                  are relevant to me?&rdquo; This is the core fusion
                  operation&mdash;prompt information meets image features.
                </p>
              </div>
            </PhaseCard>
            <PhaseCard number={3} title="Image-to-Token Cross-Attention" subtitle="Image queries the tokens" color="violet">
              <div className="space-y-2">
                <p>
                  <strong>Q = image embedding tokens</strong> (4,096 positions).{' '}
                  <strong>K, V = updated token embeddings</strong> (from Step 2).
                </p>
                <p className="text-sm opacity-75">
                  The image features attend BACK to the prompt/output tokens.
                  This updates the image embedding with prompt-specific
                  information: &ldquo;which parts of the image are most relevant
                  given what the prompt is asking?&rdquo;
                </p>
              </div>
            </PhaseCard>
          </div>
        </Row.Content>
      </Row>

      {/* Cross-attention formula and tensor dimension trace */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-sm font-medium text-foreground">
              Token-to-image cross-attention: the formula and tensor shapes
            </p>
            <p className="text-muted-foreground">
              The token-to-image cross-attention (Step 2) is where the click
              &ldquo;reads&rdquo; the image. The formula is the same scaled
              dot-product attention you know from{' '}
              <strong>Queries and Keys</strong>:
            </p>
            <div className="py-3 px-6 bg-muted/50 rounded-lg">
              <BlockMath math="\text{Attention}(Q_\text{tokens}, K_\text{image}, V_\text{image}) = \text{softmax}\!\left(\frac{Q_\text{tokens} \cdot K_\text{image}^T}{\sqrt{d}}\right) V_\text{image}" />
            </div>
            <p className="text-muted-foreground">
              Tracing the dimensions with 5 tokens and 4,096 image positions:
            </p>
            <div className="space-y-2">
              <div className="rounded-lg bg-muted/30 p-3">
                <p className="text-sm font-mono">
                  Q = W_Q &middot; tokens: &nbsp;&nbsp;[5, 256] &rarr; [5, 256]
                </p>
              </div>
              <div className="rounded-lg bg-muted/30 p-3">
                <p className="text-sm font-mono">
                  K = W_K &middot; image: &nbsp;&nbsp;[4096, 256] &rarr; [4096, 256]
                </p>
              </div>
              <div className="rounded-lg bg-muted/30 p-3">
                <p className="text-sm font-mono">
                  V = W_V &middot; image: &nbsp;&nbsp;[4096, 256] &rarr; [4096, 256]
                </p>
              </div>
              <div className="rounded-lg bg-primary/10 border border-primary/30 p-3">
                <p className="text-sm font-mono">
                  Scores = Q &middot; K<sup>T</sup> / &radic;256: &nbsp;&nbsp;[5, 256] &times; [256, 4096] = <strong>[5, 4096]</strong>
                </p>
              </div>
              <div className="rounded-lg bg-muted/30 p-3">
                <p className="text-sm font-mono">
                  Weights = softmax(Scores, dim=-1): &nbsp;&nbsp;<strong>[5, 4096]</strong>
                </p>
              </div>
              <div className="rounded-lg bg-muted/30 p-3">
                <p className="text-sm font-mono">
                  Output = Weights &middot; V: &nbsp;&nbsp;[5, 4096] &times; [4096, 256] = <strong>[5, 256]</strong>
                </p>
              </div>
            </div>
            <p className="text-muted-foreground text-sm">
              The prompt token (row 0) produces a <strong>4,096-element attention
              distribution</strong> over the 64&times;64 image grid. High weights
              correspond to image positions near the clicked point. The output is
              a 256-dim vector that aggregates image features from the attended
              positions&mdash;this is how the click location &ldquo;reads&rdquo;
              the image.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Same Formula, Familiar Shapes">
            This is exactly the cross-attention formula from{' '}
            <LessonLink slug="text-conditioning-and-guidance">Text Conditioning</LessonLink>. Q from one source (tokens), K/V
            from another (image). The only difference: instead of text tokens
            querying spatial features, <strong>prompt tokens</strong> query
            spatial features.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* After 2 decoder layers: upsampling */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-sm font-medium text-foreground">
              After 2 decoder layers: producing the final masks
            </p>
            <p className="text-muted-foreground">
              After both decoder layers, the model extracts the output tokens
              and produces masks through a sequence of operations:
            </p>
            <div className="space-y-3">
              <div className="rounded-lg bg-muted/30 p-3">
                <p className="text-sm">
                  <strong>1. Extract output tokens:</strong> 3 mask tokens
                  [3, 256] and 1 IoU token [1, 256].
                </p>
              </div>
              <div className="rounded-lg bg-muted/30 p-3">
                <p className="text-sm">
                  <strong>2. Upsample image features:</strong> Two transposed
                  convolution layers upsample the 64&times;64 image features:
                  64&times;64 &rarr; 128&times;128 &rarr; 256&times;256.
                  Result: [256, 256, 256] (256 channels at 256&times;256
                  resolution).
                </p>
              </div>
              <div className="rounded-lg bg-muted/30 p-3">
                <p className="text-sm">
                  <strong>3. Generate masks via dot product:</strong> Each mask
                  token (256-dim) is passed through a small MLP, then dotted
                  with the upsampled features. Each token produces one
                  256&times;256 mask prediction. Result: [3, 256, 256].
                </p>
              </div>
              <div className="rounded-lg bg-muted/30 p-3">
                <p className="text-sm">
                  <strong>4. Final upsample:</strong> Bilinear interpolation
                  from 256&times;256 to the original image resolution (e.g.,
                  1024&times;1024).
                </p>
              </div>
              <div className="rounded-lg bg-muted/30 p-3">
                <p className="text-sm">
                  <strong>5. IoU prediction:</strong> The IoU token goes through
                  a small MLP to predict 3 scores (one per mask candidate).
                  Each score estimates how well that mask overlaps with the
                  true object. Highest score = best mask.
                </p>
              </div>
            </div>
          </div>
        </Row.Content>
      </Row>

      {/* Mask decoder PyTorch pseudocode */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-sm font-medium text-foreground">
              PyTorch pseudocode: full mask decoder forward pass
            </p>
            <CodeBlock
              code={`class MaskDecoder(nn.Module):
    def forward(self, image_embedding, prompt_tokens):
        # image_embedding: [1, 256, 64, 64]
        # prompt_tokens: [1, N_prompt, 256]  (N_prompt=1 for point, 2 for box)

        # Concatenate prompt tokens with learned output tokens
        output_tokens = self.mask_tokens.weight     # [4, 256] (3 mask + 1 IoU)
        tokens = torch.cat([prompt_tokens, output_tokens], dim=1)  # [1, N+4, 256]

        # Flatten image embedding for attention
        image_tokens = image_embedding.flatten(2).permute(0, 2, 1)  # [1, 4096, 256]

        # Two decoder layers
        for layer in self.layers:
            # Step 1: Self-attention among tokens
            tokens = layer.self_attn(q=tokens, k=tokens, v=tokens)

            # Step 2: Token-to-image cross-attention
            tokens = layer.cross_attn_token_to_image(
                q=tokens, k=image_tokens, v=image_tokens
            )

            # Step 3: Image-to-token cross-attention
            image_tokens = layer.cross_attn_image_to_token(
                q=image_tokens, k=tokens, v=tokens
            )

        # Extract output tokens
        mask_tokens = tokens[:, -4:-1, :]     # [1, 3, 256] -- 3 mask tokens
        iou_token = tokens[:, -1:, :]         # [1, 1, 256] -- IoU token

        # Upsample image features: 64x64 -> 128x128 -> 256x256
        upsampled = self.upsample(image_tokens)   # [1, 256, 256, 256]

        # Each mask token produces one mask via dot product
        masks = torch.einsum('bmc,bcwh->bmwh',
            self.mask_mlp(mask_tokens),           # [1, 3, 256]
            upsampled                             # [1, 256, 256, 256]
        )                                         # [1, 3, 256, 256]

        # Predict IoU scores
        iou_scores = self.iou_head(iou_token)     # [1, 3]

        return masks, iou_scores`}
              language="python"
              filename="mask_decoder.py"
            />
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="The Key Insight">
            The mask is generated by a <strong>dot product</strong> between a
            learned mask token and the upsampled image features. Each mask token
            acts as a &ldquo;query&rdquo; that selects which spatial positions
            belong to the object. Different mask tokens select different
            granularities.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Putting it all together: end-to-end trace */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-sm font-medium text-foreground">
              Putting it all together: click to mask
            </p>
            <p className="text-muted-foreground">
              You have now traced every component in isolation. Here is the
              complete pipeline for a single click at pixel (342, 517):
            </p>
            <div className="space-y-2">
              <div className="rounded-lg bg-muted/30 p-3">
                <p className="text-sm">
                  <strong>Step 1 &mdash; Image encoder (already done):</strong>{' '}
                  Image [3, 1024, 1024] &rarr; ViT-H &rarr; image embedding
                  [256, 64, 64] = 4,096 spatial tokens, each 256-dim.{' '}
                  <strong>~150ms, paid once.</strong>
                </p>
              </div>
              <div className="rounded-lg bg-muted/30 p-3">
                <p className="text-sm">
                  <strong>Step 2 &mdash; Prompt encoding:</strong>{' '}
                  Click (342, 517) &rarr; normalize to (0.334, 0.505) &rarr;
                  Fourier PE at 128 bands &rarr; [512] &rarr; linear &rarr; [256]
                  + foreground type embedding [256] ={' '}
                  <strong>1 prompt token [256]</strong>. &lt;1ms.
                </p>
              </div>
              <div className="rounded-lg bg-muted/30 p-3">
                <p className="text-sm">
                  <strong>Step 3 &mdash; Assemble tokens:</strong>{' '}
                  1 prompt token + 3 learned mask tokens + 1 IoU token ={' '}
                  <strong>5 tokens [5, 256]</strong>.
                </p>
              </div>
              <div className="rounded-lg bg-primary/10 border border-primary/30 p-3">
                <p className="text-sm">
                  <strong>Step 4 &mdash; Mask decoder (2 layers):</strong>{' '}
                  Each layer: self-attention among 5 tokens &rarr; token-to-image
                  cross-attention (5 tokens query 4,096 image positions) &rarr;
                  image-to-token cross-attention (4,096 positions attend back to
                  5 tokens). After 2 layers: extract 3 mask tokens [3, 256] +
                  1 IoU token [1, 256].
                </p>
              </div>
              <div className="rounded-lg bg-muted/30 p-3">
                <p className="text-sm">
                  <strong>Step 5 &mdash; Generate masks:</strong>{' '}
                  Upsample image features: 64&times;64 &rarr; 256&times;256.
                  Dot each mask token with upsampled features &rarr;{' '}
                  <strong>3 masks [3, 256, 256]</strong>. Bilinear upsample to
                  1024&times;1024. IoU head predicts 3 confidence scores.{' '}
                  <strong>~50ms total for decoder + upsampling.</strong>
                </p>
              </div>
            </div>
            <p className="text-muted-foreground text-sm">
              <strong>Total: ~200ms for the first prompt, ~50ms for each
              additional prompt on the same image.</strong> Every operation in
              this pipeline uses building blocks you have traced
              before&mdash;sinusoidal encoding, cross-attention, dot-product
              masks. SAM&rsquo;s contribution is composing them into a
              promptable segmentation pipeline.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Composition, Not Invention">
            Nothing in this pipeline is architecturally novel on its own.
            Fourier encoding, cross-attention, transposed convolutions, dot
            product decoding&mdash;all existed before SAM. The innovation is the
            composition: how these pieces fit together to make segmentation
            promptable and interactive.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 9: The Promptable Interface
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Promptable Interface"
            subtitle="What makes SAM a foundation model, not just a segmentation network"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              SAM is not just another segmentation model&mdash;it is a{' '}
              <strong>promptable</strong> segmentation model. The prompt tells
              SAM <em>where</em> to look (or, with SAM 3, <em>what</em> to look
              for). The model figures out the precise boundary.
            </p>
          </div>
        </Row.Content>
      </Row>

      <Row>
        <Row.Content>
          <div className="grid gap-4 md:grid-cols-2">
            <GradientCard title="Point Prompts" color="cyan">
              <p>
                Click a single point on the object. SAM infers the full mask.
                The simplest interaction&mdash;but inherently ambiguous. A click
                on a shirt button could mean the button, the shirt, or the whole
                person.
              </p>
            </GradientCard>
            <GradientCard title="Box Prompts" color="blue">
              <p>
                Draw a rectangle around the object. More constrained than a
                point&mdash;reduces ambiguity about which object you mean.
                A bounding box is just the rough location and extent; SAM fills
                in the precise boundary.
              </p>
            </GradientCard>
            <GradientCard title="Mask Prompts" color="violet">
              <p>
                Provide an initial rough mask, SAM refines it. Enables iterative
                refinement: start with a coarse outline, SAM snaps it to the
                true object boundary.
              </p>
            </GradientCard>
            <GradientCard title="Text Prompts (SAM 3)" color="emerald">
              <p>
                Describe what to segment: &ldquo;red car,&rdquo; &ldquo;shipping
                container,&rdquo; &ldquo;striped umbrella.&rdquo; SAM 3 finds{' '}
                <strong>all instances</strong> and segments each one.
              </p>
            </GradientCard>
          </div>
        </Row.Content>
      </Row>

      {/* Multi-mask output -- uses the shirt button example per review fix */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-sm font-medium text-foreground">
              Multi-mask output: handling ambiguity
            </p>
            <p className="text-muted-foreground">
              A single point click is inherently ambiguous. Click on a
              person&rsquo;s shirt button&mdash;did you mean the button, the
              shirt, or the whole person? SAM handles this by predicting{' '}
              <strong>3 masks at different granularities</strong> (part, object,
              scene-level) with confidence scores. The user or downstream system
              picks the right one.
            </p>
            <p className="text-muted-foreground">
              Concrete example: you click on the button of a person&rsquo;s
              shirt. SAM returns three masks:
            </p>
            <div className="grid gap-3 md:grid-cols-3">
              <div className="rounded-lg bg-muted/30 p-3 text-center">
                <p className="text-sm font-medium">Mask 1: Button only</p>
                <p className="text-xs text-muted-foreground mt-1">Score: 0.82</p>
              </div>
              <div className="rounded-lg bg-primary/10 border border-primary/30 p-3 text-center">
                <p className="text-sm font-medium">Mask 2: The shirt</p>
                <p className="text-xs text-muted-foreground mt-1">Score: 0.96</p>
              </div>
              <div className="rounded-lg bg-muted/30 p-3 text-center">
                <p className="text-sm font-medium">Mask 3: Whole person</p>
                <p className="text-xs text-muted-foreground mt-1">Score: 0.68</p>
              </div>
            </div>
            <p className="text-muted-foreground text-sm">
              The highest-confidence mask (the shirt at 0.96) is the
              default, but all three are available. This is a critical design
              choice: SAM does not force a single interpretation of an ambiguous
              click. <em>Of course</em> you would produce multiple
              masks&mdash;a point on a shirt button could mean the button, the
              shirt, or the person.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Ambiguity Is a Feature">
            Rather than guessing which granularity you want, SAM gives you
            three options and lets you (or your application) choose. This
            turns an unsolvable problem (one click, multiple valid
            interpretations) into a simple selection.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* SAM doesn't need retraining */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-sm font-medium text-foreground">
              SAM does not need to know what the object is
            </p>
            <p className="text-muted-foreground">
              From <LessonLink slug="transfer-learning">Transfer Learning</LessonLink>, you know the pattern:
              pretrain on general data, then fine-tune for specific tasks. SAM
              breaks this expectation. Give it a point on a flamingo, and it
              segments the flamingo&mdash;even if no flamingo was in the training
              data.
            </p>
            <p className="text-muted-foreground">
              This works because SAM does not learn &ldquo;what is a
              flamingo.&rdquo; It learns &ldquo;given a point inside an object,{' '}
              <strong>which nearby pixels belong to the same thing</strong>.&rdquo;
              The prompt tells SAM WHERE to look; SAM&rsquo;s learned ability is
              figuring out where the <strong>boundaries</strong> are. Any object
              with coherent boundaries can be segmented.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Boundaries, Not Categories">
            SAM&rsquo;s learned ability is finding <strong>where boundaries
            are</strong>, not what the object is. Any object with coherent
            boundaries can be segmented&mdash;even objects SAM never saw
            during training.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Negative example: where SAM struggles */}
      <Row>
        <Row.Content>
          <GradientCard title="Where SAM Struggles" color="amber">
            <div className="space-y-2">
              <p>
                SAM is not magic. It learned boundary detection from 1 billion
                masks of objects with clear edges. When an image has{' '}
                <strong>no coherent boundaries</strong>&mdash;gradual cloud
                gradients in a blue sky, abstract textures, smooth color
                blends&mdash;the prompt does not map to a meaningful
                segmentation.
              </p>
              <p>
                &ldquo;Segment Anything&rdquo; really means &ldquo;segment any{' '}
                <strong>object with discernible boundaries</strong>.&rdquo; This
                is not a failure&mdash;it defines the scope of what boundary-based
                segmentation can do.
              </p>
            </div>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 10: Loss Function -- Training SAM (NEW)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Loss Function: Training SAM"
            subtitle="Focal loss, dice loss, and minimum-loss assignment"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              How do you train a model to produce accurate segmentation masks?
              The challenge starts with a severe <strong>class
              imbalance</strong>: in a typical segmentation mask, the vast
              majority of pixels are background (label 0), and only a small
              fraction are foreground (label 1). A standard cross-entropy loss
              would be dominated by the easy background pixels&mdash;the model
              could achieve low loss by simply predicting &ldquo;background
              everywhere&rdquo; and ignoring the object entirely.
            </p>
            <p className="text-muted-foreground">
              SAM uses two complementary losses that address this from different
              angles, plus a clever training strategy for its multi-mask output.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="The Imbalance Problem">
            If an object covers 5% of the image, 95% of pixels are background.
            Standard cross-entropy gives equal weight to every pixel. The model
            can get 95% accuracy by predicting all background&mdash;and
            completely miss the object.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* Focal loss */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-sm font-medium text-foreground">
              Focal loss: focus on the hard pixels
            </p>
            <p className="text-muted-foreground">
              Focal loss is a modification of cross-entropy that{' '}
              <strong>down-weights well-classified pixels</strong>, letting the
              model focus its learning on the hard cases. The formula:
            </p>
            <div className="py-3 px-6 bg-muted/50 rounded-lg">
              <BlockMath math="\mathcal{L}_\text{focal} = -\alpha \cdot (1 - p_t)^\gamma \cdot \log(p_t)" />
            </div>
            <p className="text-muted-foreground">
              where <InlineMath math="p_t" /> is the model&rsquo;s predicted
              probability for the <em>correct</em> class:
            </p>
            <div className="py-2 px-6 bg-muted/50 rounded-lg">
              <BlockMath math="p_t = \begin{cases} p & \text{if } y = 1 \text{ (foreground)} \\ 1 - p & \text{if } y = 0 \text{ (background)} \end{cases}" />
            </div>
            <p className="text-muted-foreground">
              In plain terms: <InlineMath math="p_t" /> is the model&rsquo;s{' '}
              <strong>confidence in the correct answer</strong>&mdash;
              <InlineMath math="p" /> for foreground pixels,{' '}
              <InlineMath math="1-p" /> for background pixels. Higher{' '}
              <InlineMath math="p_t" /> means the model is more confident
              about the right class.
            </p>
            <p className="text-muted-foreground">
              The key is the <InlineMath math="(1 - p_t)^\gamma" /> factor.
              SAM uses <InlineMath math="\gamma = 2" />. When{' '}
              <InlineMath math="p_t" /> is high (easy pixel, correctly
              classified), this factor shrinks toward zero, suppressing the
              loss. When <InlineMath math="p_t" /> is low (hard pixel,
              misclassified), this factor stays large.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Focal Loss in One Sentence">
            Standard cross-entropy treats every pixel equally. Focal loss{' '}
            <strong>automatically ignores easy pixels</strong> and focuses
            the gradient signal on the hard ones. The{' '}
            <InlineMath math="\gamma" /> parameter controls how aggressively
            easy pixels are down-weighted.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Focal loss traced computation */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-sm font-medium text-foreground">
              Tracing focal loss on two pixels
            </p>
            <p className="text-muted-foreground text-sm">
              Recall that <InlineMath math="-\log(p_t)" /> is the standard
              cross-entropy term you know from CLIP&mdash;it is always positive
              because <InlineMath math="0 < p_t < 1" /> makes{' '}
              <InlineMath math="\log(p_t)" /> negative, and the leading minus
              flips it. We can rewrite the focal loss as:
            </p>
            <div className="py-2 px-6 bg-muted/50 rounded-lg">
              <BlockMath math="\mathcal{L}_\text{focal} = \alpha \cdot (1 - p_t)^\gamma \cdot \big[-\log(p_t)\big]" />
            </div>
            <p className="text-muted-foreground text-sm">
              We trace the <strong>focusing factor</strong>{' '}
              <InlineMath math="(1 - p_t)^\gamma \cdot [-\log(p_t)]" /> below,
              setting <InlineMath math="\alpha" /> aside since SAM
              uses <InlineMath math="\alpha = 0.25" /> as a constant multiplier
              on all pixels and it does not affect the ratio between easy and
              hard.
            </p>
            <div className="grid gap-4 md:grid-cols-2">
              <div className="rounded-lg bg-emerald-500/10 border border-emerald-500/30 p-4">
                <p className="text-sm font-medium">Easy background pixel</p>
                <p className="text-sm text-muted-foreground mt-1">
                  Model predicts <InlineMath math="p = 0.05" /> (correctly low
                  for background). So <InlineMath math="p_t = 1 - 0.05 = 0.95" />.
                </p>
                <div className="mt-2 text-sm font-mono">
                  <p><InlineMath math="(1 - 0.95)^2 \cdot [-\log(0.95)]" /></p>
                  <p className="mt-1"><InlineMath math="= 0.0025 \times 0.0513 = 0.000128" /></p>
                </div>
                <p className="text-sm text-muted-foreground mt-2">
                  <strong>Nearly zero loss.</strong> This pixel is already
                  correctly classified; focal loss suppresses it.
                </p>
              </div>
              <div className="rounded-lg bg-rose-500/10 border border-rose-500/30 p-4">
                <p className="text-sm font-medium">Hard foreground pixel</p>
                <p className="text-sm text-muted-foreground mt-1">
                  Model predicts <InlineMath math="p = 0.1" /> (incorrectly low
                  for foreground). So <InlineMath math="p_t = 0.1" />.
                </p>
                <div className="mt-2 text-sm font-mono">
                  <p><InlineMath math="(1 - 0.1)^2 \cdot [-\log(0.1)]" /></p>
                  <p className="mt-1"><InlineMath math="= 0.81 \times 2.303 = 1.865" /></p>
                </div>
                <p className="text-sm text-muted-foreground mt-2">
                  <strong>High loss.</strong> This pixel is misclassified; focal
                  loss amplifies the gradient signal.
                </p>
              </div>
            </div>
            <p className="text-muted-foreground">
              The ratio: <InlineMath math="1.865 / 0.000128 \approx 14{,}600\times" />.
              Focal loss makes the hard foreground pixel contribute{' '}
              <strong>~14,600&times; more</strong> to the gradient than the easy
              background pixel. Multiplying both by <InlineMath math="\alpha = 0.25" /> does
              not change this ratio. The model automatically focuses on the
              pixels that matter.
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* Dice loss */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-sm font-medium text-foreground">
              Dice loss: directly optimize mask overlap
            </p>
            <p className="text-muted-foreground">
              While focal loss operates <strong>per-pixel</strong> (each pixel
              contributes independently), dice loss operates on the{' '}
              <strong>mask as a whole</strong>, directly measuring the overlap
              between the predicted mask and the ground truth:
            </p>
            <div className="py-3 px-6 bg-muted/50 rounded-lg">
              <BlockMath math="\mathcal{L}_\text{dice} = 1 - \frac{2 \sum_i p_i \cdot g_i + 1}{\sum_i p_i + \sum_i g_i + 1}" />
            </div>
            <p className="text-muted-foreground">
              where <InlineMath math="p_i" /> is the predicted probability for
              pixel <InlineMath math="i" /> and <InlineMath math="g_i" /> is the
              ground truth (0 or 1). The numerator measures intersection, the
              denominator measures total area. Perfect overlap gives{' '}
              <InlineMath math="\mathcal{L}_\text{dice} = 0" />; no overlap
              gives <InlineMath math="\mathcal{L}_\text{dice} = 1" />. The +1
              smoothing term prevents division by zero when both masks are empty.
            </p>
            <p className="text-muted-foreground">
              <strong>Why two losses?</strong> Focal loss provides local, per-pixel
              gradients (&ldquo;this specific pixel is wrong&rdquo;). Dice loss
              provides a global signal about mask quality (&ldquo;the overall
              overlap is poor&rdquo;). Together they give both fine-grained and
              holistic training signal.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Local + Global">
            Focal loss: &ldquo;fix this pixel.&rdquo; Dice loss: &ldquo;the
            overall shape is wrong.&rdquo; Using both ensures the model gets
            gradients at both the pixel level and the mask level.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Combined loss */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-sm font-medium text-foreground">
              Combined loss with SAM&rsquo;s actual weights
            </p>
            <div className="py-3 px-6 bg-muted/50 rounded-lg">
              <BlockMath math="\mathcal{L}_\text{mask} = 20 \cdot \mathcal{L}_\text{focal} + 1 \cdot \mathcal{L}_\text{dice}" />
            </div>
            <p className="text-muted-foreground">
              SAM weights the focal loss <strong>20&times; higher</strong> than
              the dice loss. This makes sense: focal loss provides the
              per-pixel gradient signal that drives boundary precision, while
              dice loss provides a gentler global correction. The 20&times;
              weight ensures the model prioritizes getting individual pixels
              right.
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* Multi-mask minimum-loss assignment */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-sm font-medium text-foreground">
              Multi-mask training: minimum-loss assignment
            </p>
            <p className="text-muted-foreground">
              SAM outputs 3 masks per prompt. During training, a clever strategy
              prevents all three masks from collapsing to the same prediction:
            </p>
            <ul className="list-disc list-inside text-muted-foreground space-y-1 ml-4">
              <li>Compute the loss for <strong>all 3 masks</strong> against the ground truth</li>
              <li>Backpropagate only through the mask with the <strong>lowest loss</strong></li>
            </ul>
            <div className="py-3 px-6 bg-muted/50 rounded-lg">
              <BlockMath math="\mathcal{L} = \min(\mathcal{L}_1, \mathcal{L}_2, \mathcal{L}_3)" />
              <p className="text-sm text-muted-foreground text-center mt-1">
                where <InlineMath math="\mathcal{L}_i = 20 \cdot \mathcal{L}_\text{focal}(\text{mask}_i, \text{gt}) + 1 \cdot \mathcal{L}_\text{dice}(\text{mask}_i, \text{gt})" />
              </p>
            </div>
            <p className="text-muted-foreground">
              This is <strong>minimum-loss assignment</strong>: each mask
              candidate is free to specialize in a different granularity. Without
              this, averaging the loss across all 3 masks would pressure all
              three to be similar&mdash;why produce 3 identical masks? With
              minimum-loss, one mask learns &ldquo;small region,&rdquo; another
              learns &ldquo;full object,&rdquo; another learns &ldquo;object +
              context.&rdquo; The model is rewarded for having at least{' '}
              <strong>one</strong> good mask, not for having three mediocre ones.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Why Min, Not Mean?">
            Averaging: &ldquo;all 3 masks must be decent.&rdquo; Minimum:
            &ldquo;at least 1 mask must be great.&rdquo; The minimum
            objective lets each mask specialize freely, producing the
            diversity of granularities that makes multi-mask output useful.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* IoU prediction loss */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-sm font-medium text-foreground">
              IoU prediction loss: teaching the model to estimate its own quality
            </p>
            <p className="text-muted-foreground">
              The IoU prediction head is trained with a separate MSE loss:
            </p>
            <div className="py-3 px-6 bg-muted/50 rounded-lg">
              <BlockMath math="\mathcal{L}_\text{IoU} = \text{MSE}(\hat{\text{IoU}}, \text{IoU}_\text{actual})" />
            </div>
            <p className="text-muted-foreground">
              The actual IoU is computed during training by comparing each
              predicted mask (thresholded at 0.5) with the ground truth. This
              teaches the model to <strong>estimate how good its own masks
              are</strong>&mdash;essential for ranking the 3 candidates at
              inference time.
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* Loss function PyTorch pseudocode */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-sm font-medium text-foreground">
              PyTorch pseudocode: SAM loss computation
            </p>
            <CodeBlock
              code={`def compute_loss(pred_masks, pred_ious, gt_mask, alpha=0.25, gamma=2.0):
    # pred_masks: [3, H, W] -- 3 candidate masks (logits)
    # pred_ious: [3] -- predicted IoU for each mask
    # gt_mask: [H, W] -- ground truth binary mask

    losses = []
    actual_ious = []
    for i in range(3):
        # Focal loss (per-pixel, focuses on hard examples)
        p = pred_masks[i].sigmoid()
        ce = F.binary_cross_entropy_with_logits(
            pred_masks[i], gt_mask, reduction='none'
        )
        p_t = p * gt_mask + (1 - p) * (1 - gt_mask)
        focal = alpha * (1 - p_t) ** gamma * ce

        # Dice loss (mask-level overlap)
        intersection = (p * gt_mask).sum()
        dice = 1 - (2 * intersection + 1) / (p.sum() + gt_mask.sum() + 1)

        # Combined: 20x focal + 1x dice
        losses.append(20.0 * focal.mean() + 1.0 * dice)

        # Compute actual IoU for IoU prediction training
        pred_binary = (p > 0.5).float()
        inter = (pred_binary * gt_mask).sum()
        union = pred_binary.sum() + gt_mask.sum() - inter
        actual_ious.append(inter / (union + 1e-6))

    # Minimum loss assignment: backprop through best mask only
    losses = torch.stack(losses)
    best_idx = losses.argmin()
    mask_loss = losses[best_idx]

    # IoU prediction loss
    actual_ious = torch.stack(actual_ious)
    iou_loss = F.mse_loss(pred_ious, actual_ious)

    return mask_loss + iou_loss`}
              language="python"
              filename="sam_loss.py"
            />
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="Loss Summary">
            <ul className="space-y-1 text-sm">
              <li>&bull; <strong>Focal:</strong> per-pixel, suppresses easy pixels</li>
              <li>&bull; <strong>Dice:</strong> mask-level overlap</li>
              <li>&bull; <strong>Combined:</strong> 20&times; focal + 1&times; dice</li>
              <li>&bull; <strong>Multi-mask:</strong> min-loss assignment</li>
              <li>&bull; <strong>IoU:</strong> MSE on predicted quality</li>
            </ul>
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 11: Check 1 -- Predict and Verify (expanded)
          ================================================================ */}
      <Row>
        <Row.Content>
          <GradientCard title="Predict and Verify" color="cyan">
            <div className="space-y-3 text-sm">
              <p className="mt-2">
                <strong>Question 1:</strong> You load a 4000&times;3000
                photograph into SAM. How many times does the ViT image encoder
                run?
              </p>
              <details className="mt-2">
                <summary className="font-medium cursor-pointer text-primary">
                  Reveal
                </summary>
                <div className="mt-2">
                  <p>
                    <strong>Once.</strong> The image embedding is computed once
                    and reused for all subsequent prompts on that image.
                  </p>
                </div>
              </details>

              <p className="mt-4">
                <strong>Question 2:</strong> You click 5 different objects in
                the image. Approximately how long does this take?
              </p>
              <details className="mt-2">
                <summary className="font-medium cursor-pointer text-primary">
                  Reveal
                </summary>
                <div className="mt-2">
                  <p>
                    <strong>~400ms total</strong>&mdash;~150ms for the image
                    encoding + 5 &times; 50ms for the mask decoder = ~400ms.
                    Not 5 &times; 200ms = 1,000ms. The encoder cost is paid
                    once.
                  </p>
                </div>
              </details>

              <p className="mt-4">
                <strong>Question 3:</strong> You click a point on the boundary
                between two touching objects (a cup sitting on a plate). What
                does SAM output?
              </p>
              <details className="mt-2">
                <summary className="font-medium cursor-pointer text-primary">
                  Reveal
                </summary>
                <div className="mt-2">
                  <p>
                    <strong>Three masks at different granularities</strong>&mdash;likely
                    the region near the click, one of the two objects (whichever
                    the model&rsquo;s features more strongly associate with the
                    click point), or both together. A boundary click is
                    inherently one of SAM&rsquo;s hardest cases&mdash;this is
                    where box prompts or multiple point prompts help
                    disambiguate.
                  </p>
                </div>
              </details>

              <p className="mt-4">
                <strong>Question 4:</strong> Why does SAM use minimum-loss
                assignment instead of averaging the loss across all 3 masks?
              </p>
              <details className="mt-2">
                <summary className="font-medium cursor-pointer text-primary">
                  Reveal
                </summary>
                <div className="mt-2">
                  <p>
                    Averaging would pressure all 3 masks to be similar&mdash;they
                    would converge to the same prediction. Minimum-loss lets each
                    mask <strong>specialize</strong> in a different granularity
                    (small region, full object, object + context). The model is
                    rewarded for having at least one good mask, not for having
                    three mediocre ones.
                  </p>
                </div>
              </details>
            </div>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 12: The SA-1B Data Engine
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The SA-1B Data Engine"
            subtitle="1 billion masks through human-AI partnership"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              SAM&rsquo;s training data is unprecedented:{' '}
              <strong>1.1 billion masks on 11 million images</strong>. For
              comparison, COCO (a standard benchmark) has ~860K masks on 160K
              images. SA-1B has <strong>400&times; more masks</strong> than any
              prior segmentation dataset.
            </p>
            <p className="text-muted-foreground">
              But this data did not exist&mdash;Meta had to create it. The
              approach is called the <strong>data engine</strong>: a three-stage
              process where the model and human annotators improve each other.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="A Different Kind of Scaling">
            This is not the same scaling you have seen in language models (just
            add more text from the internet). SAM&rsquo;s data required{' '}
            <strong>active creation</strong> through a human-AI partnership.
            The model helps create the data that trains the next version of
            itself.
          </InsightBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <div className="space-y-3">
            <PhaseCard number={1} title="Assisted-Manual" subtitle="High quality, slow" color="cyan">
              <p>
                Human annotators use an early version of SAM to help draw masks.
                SAM suggests, humans correct. ~120K images, 4.3M masks. Slow but
                high-quality.
              </p>
            </PhaseCard>
            <PhaseCard number={2} title="Semi-Automatic" subtitle="Faster annotation, growing dataset" color="blue">
              <p>
                SAM automatically suggests masks; humans verify and correct. The
                model is better now (trained on Stage 1 data), so less human
                effort per mask.
              </p>
            </PhaseCard>
            <PhaseCard number={3} title="Fully Automatic" subtitle="Scales to billions" color="violet">
              <p>
                SAM processes images with a grid of point prompts, generating
                masks <strong>without human input</strong>. This is what scales
                to 11M images and 1.1B masks.
              </p>
            </PhaseCard>
          </div>
        </Row.Content>
      </Row>

      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The key insight: <strong>the model improves the data, and better
              data improves the model</strong>. This virtuous cycle is how SAM
              scaled to 1 billion masks. The model that creates the annotations
              gets better as it trains on the annotations it helped create.
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 13: SAM 2 -- Video Extension (EXPANDED)
          ================================================================ */}
      <Row>
        <Row.Content>
          <div className="rounded-lg border border-muted bg-muted/20 p-4">
            <p className="text-muted-foreground text-sm">
              <strong>Important framing:</strong> SAM 2 and SAM 3 are not
              separate models built from scratch. Each version{' '}
              <strong>adds</strong> to the previous one. The core
              architecture you just learned&mdash;ViT encoder, prompt
              encoder, mask decoder&mdash;carries through all three versions.
            </p>
          </div>
        </Row.Content>
      </Row>

      <Row>
        <Row.Content>
          <SectionHeader
            title="SAM 2: Video Extension"
            subtitle="From single images to tracking objects across time"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              SAM 1 processes single images. SAM 2 (2024) extends to{' '}
              <strong>video</strong>. The problem: in video, an object appears
              across many frames. Clicking on every frame is impractical. The
              model needs to <strong>track</strong> the object across time.
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* Memory encoder */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-sm font-medium text-foreground">
              The memory mechanism: how SAM 2 remembers across frames
            </p>
            <PhaseCard number={1} title="Memory Encoder" subtitle="Storing what was segmented" color="cyan">
              <div className="space-y-2">
                <p>
                  After segmenting a frame, the predicted mask and image features
                  are combined. The mask is downsampled and passed through
                  lightweight convolutions, then combined with the image encoder
                  output via element-wise addition. Result:{' '}
                  <strong>memory tokens</strong> for this frame (64&times;64&times;64
                  spatial memory features).
                </p>
              </div>
            </PhaseCard>
            <PhaseCard number={2} title="Memory Bank" subtitle="Recent frames + prompted frames" color="blue">
              <div className="space-y-2">
                <p>
                  Stores memory tokens from the <strong>N most recent
                  frames</strong> (default N=6). Also stores memory tokens from
                  any frame where the user provided a prompt&mdash;prompted
                  frame memories are <strong>never evicted</strong>. Structure:
                  a set of [N_frames, 64&times;64, 64] spatial memory features.
                </p>
              </div>
            </PhaseCard>
            <PhaseCard number={3} title="Memory Attention" subtitle="Current frame reads the memory" color="violet">
              <div className="space-y-2">
                <p>
                  Current frame&rsquo;s image features attend to the memory
                  bank via <strong>cross-attention</strong>. Q from the current
                  frame, K/V from the memory bank. Same formula you
                  know&mdash;applied across <strong>time</strong> instead of
                  across spatial positions.
                </p>
              </div>
            </PhaseCard>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="Memory as Cross-Attention">
            SAM 2&rsquo;s memory mechanism is a form of cross-attention&mdash;the
            current frame&rsquo;s features attend to stored features from
            previous frames. The memory bank IS the context window, just for
            video instead of text.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* Memory attention tensor shapes */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-sm font-medium text-foreground">
              Memory attention: tensor shapes
            </p>
            <p className="text-muted-foreground text-sm">
              Memory tokens are 64-dim (from the memory encoder), but the
              current frame&rsquo;s features are 256-dim. A learned linear
              projection maps the memory tokens up to 256-dim before
              cross-attention:
            </p>
            <div className="space-y-2">
              <div className="rounded-lg bg-muted/30 p-3">
                <p className="text-sm font-mono">
                  Memory bank (raw): &nbsp;&nbsp;[N_frames &times; 4096, 64]
                </p>
              </div>
              <div className="rounded-lg bg-muted/30 p-3">
                <p className="text-sm font-mono">
                  Memory bank (projected): &nbsp;&nbsp;[N_frames &times; 4096, 64] &rarr; linear &rarr; [N_frames &times; 4096, 256]
                </p>
              </div>
              <div className="rounded-lg bg-muted/30 p-3">
                <p className="text-sm font-mono">
                  Q = current frame features: &nbsp;&nbsp;[4096, 256]
                </p>
              </div>
              <div className="rounded-lg bg-muted/30 p-3">
                <p className="text-sm font-mono">
                  K, V = projected memory tokens: &nbsp;&nbsp;[N_frames &times; 4096, 256]
                </p>
              </div>
              <div className="rounded-lg bg-primary/10 border border-primary/30 p-3">
                <p className="text-sm font-mono">
                  Scores = Q &middot; K<sup>T</sup> / &radic;256: &nbsp;&nbsp;[4096, 256] &times; [256, N&times;4096] = [4096, N&times;4096]
                </p>
              </div>
              <div className="rounded-lg bg-muted/30 p-3">
                <p className="text-sm font-mono">
                  Output = softmax(Scores) &middot; V: &nbsp;&nbsp;[4096, N&times;4096] &times; [N&times;4096, 256] = <strong>[4096, 256]</strong>
                </p>
              </div>
            </div>
            <p className="text-muted-foreground text-sm">
              The current frame &ldquo;reads&rdquo; the memory of recent
              frames. Each of the 4,096 spatial positions in the current frame
              produces an attention distribution over <strong>all spatial
              positions across all stored frames</strong>. High attention
              weights point to memory positions where the tracked object
              appeared before. This is how temporal consistency works: the
              current frame sees where the object was in previous frames.
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* Occlusion handling */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-sm font-medium text-foreground">
              Occlusion handling: no special mechanism needed
            </p>
            <p className="text-muted-foreground">
              When the object is fully occluded (hidden behind something), the
              mask prediction for that frame will be empty. But the memory bank
              still contains pre-occlusion memories. When the object reappears,
              the current frame&rsquo;s features will have{' '}
              <strong>high cross-attention scores</strong> with the
              pre-occlusion memory tokens&mdash;because the object looks
              similar to how it looked before.
            </p>
            <p className="text-muted-foreground">
              <em>Of course</em> you would store what the object looked like
              before&mdash;that is how you find it again when it reappears.
              This is not a special mechanism. It falls out naturally from
              cross-attention over a memory bank that retains past frames. The
              same elegant pattern: cross-attention handles temporal consistency
              the same way it handles spatial attention.
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* Video processing tensor shape summary */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-sm font-medium text-foreground">
              Full pipeline for one frame of video
            </p>
            <CodeBlock
              code={`Frame t:
  Image encoder: frame [3, 1024, 1024] -> features [256, 64, 64]
  Memory projection: memory_bank [N*4096, 64] -> linear -> [N*4096, 256]
  Memory attention: features [4096, 256] x projected_memory [N*4096, 256]
                    -> updated features [4096, 256]
  Prompt encoder: click/box -> prompt tokens [N_prompt, 256]
  Mask decoder: (updated features, prompt tokens)
                -> masks [3, 256, 256], IoU scores [3]
  Memory encoder: (mask, features) -> memory [64, 64, 64]
                  -> added to memory bank`}
              language="python"
              filename="sam2_video_pipeline.py"
            />
          </div>
        </Row.Content>
      </Row>

      {/* SAM 1 vs SAM 2 comparison */}
      <Row>
        <Row.Content>
          <ComparisonRow
            left={{
              title: 'SAM 1 (Images)',
              color: 'blue',
              items: [
                'Single image input',
                'User must prompt each image separately',
                'No temporal awareness',
                'No object tracking',
              ],
            }}
            right={{
              title: 'SAM 2 (+ Video)',
              color: 'violet',
              items: [
                'Video (sequence of frames)',
                'Prompt once, track across all frames',
                'Memory bank for temporal consistency',
                'Far fewer user interactions per video',
              ],
            }}
          />
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 14: SAM 3 -- Concepts and Language
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="SAM 3: Concepts and Language"
            subtitle='From "click to segment" to "describe to segment"'
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              SAM 1 and SAM 2 have a fundamental limitation: you must{' '}
              <strong>show</strong> the model what to segment&mdash;click a
              point, draw a box. You cannot <strong>say</strong> what to
              segment. If you want to segment all red cars in a street scene,
              you must click each car individually.
            </p>
            <p className="text-muted-foreground">
              SAM 3 (November 2025) adds the final piece:{' '}
              <strong>language understanding</strong>. <em>Of course</em> the
              next step after &ldquo;click to segment&rdquo; is &ldquo;describe
              to segment&rdquo;&mdash;that is the same trajectory language
              models followed.
            </p>
          </div>
        </Row.Content>
      </Row>

      <Row>
        <Row.Content>
          <GradientCard title="SAM 3's Innovation: Concept-Level Segmentation" color="emerald">
            <div className="space-y-2">
              <ul className="space-y-1 ml-4">
                <li>&bull; Accept <strong>text prompts</strong>: &ldquo;red car,&rdquo; &ldquo;shipping container,&rdquo; &ldquo;striped umbrella&rdquo;</li>
                <li>&bull; Accept <strong>exemplar prompts</strong>: provide an image of the target concept</li>
                <li>&bull; Find <strong>all instances</strong> of the concept in the image or video, each with a unique mask and ID</li>
                <li>&bull; <strong>Open-vocabulary</strong>: not limited to a fixed set of categories</li>
              </ul>
              <p className="text-sm opacity-75 mt-2">
                The text prompt is encoded via a text encoder (similar to how
                CLIP/SigLIP encode text) that produces embedding tokens. These
                tokens enter the same prompt-to-image fusion pipeline you
                already know.
              </p>
            </div>
          </GradientCard>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The GPT-3 Parallel">
            SAM 1 &rarr; SAM 3 mirrors the trajectory you have seen in
            language: from manual interaction (click each object) to
            language-based interaction (describe what you want). The same
            &ldquo;promptable foundation model&rdquo; pattern, extended with
            language.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Concrete SAM 1 vs SAM 3 comparison */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              To feel the difference, imagine a street scene with 4 red cars
              parked along the curb:
            </p>
            <ComparisonRow
              left={{
                title: 'SAM 1: Click Each One',
                color: 'blue',
                items: [
                  'Click car #1 \u2192 mask for car #1',
                  'Click car #2 \u2192 mask for car #2',
                  'Click car #3 \u2192 mask for car #3',
                  'Click car #4 \u2192 mask for car #4',
                  '4 clicks, 4 separate interactions',
                ],
              }}
              right={{
                title: 'SAM 3: Describe Once',
                color: 'emerald',
                items: [
                  'Type "red car"',
                  '\u2192 4 masks, one per car',
                  'Each mask has a unique instance ID',
                  'One prompt found them all',
                  'Works for any count \u2014 4 cars or 40',
                ],
              }}
            />
            <p className="text-muted-foreground text-sm">
              SAM 1 requires you to know where each car is and click it. SAM 3
              requires you to know <em>what</em> you are looking for. The model
              handles the <em>where</em> and the <em>how many</em>.
            </p>
          </div>
        </Row.Content>
      </Row>

      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-sm font-medium text-foreground">
              Architecture evolution
            </p>
            <ul className="list-disc list-inside text-muted-foreground space-y-1 ml-4">
              <li>SAM 3 uses a unified <strong>Perception Encoder</strong> that fuses visual features with text/exemplar embeddings</li>
              <li>A transformer-based detector finds all instances of the prompted concept</li>
              <li>The SAM 2 memory/tracking module handles video consistency</li>
              <li>A <strong>global presence head</strong> determines if the concept exists <em>before</em> trying to localize it (&ldquo;recognition before localization&rdquo;)</li>
            </ul>
            <p className="text-muted-foreground">
              <strong>Performance:</strong> 840M parameters (~3.4 GB). 30ms
              inference per image with 100+ detected objects on an H200 GPU.
              Doubles the segmentation quality (cgF1 score) compared to prior
              systems.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Not Two Separate Encoders">
            SAM 3&rsquo;s Perception Encoder is <strong>not</strong> the same
            as CLIP&rsquo;s dual-encoder setup. Instead of two separate
            encoders (one for images, one for text), SAM 3 uses a{' '}
            <strong>unified encoder</strong> that fuses vision and language
            together. Different approach from the &ldquo;two encoders, one
            shared space&rdquo; pattern you know from CLIP.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* SAM Evolution Diagram */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-sm font-medium text-foreground">
              The SAM trajectory: each version is additive
            </p>
            <MermaidDiagram chart={`
              graph LR
                A["SAM 1 (2023): Images"] -->|"+ memory"| B["SAM 2 (2024): + Video"]
                B -->|"+ language"| C["SAM 3 (2025): + Concepts/Text"]
            `} />
            <p className="text-muted-foreground text-sm">
              SAM 3 still supports point/box/mask prompts from SAM 1. Each
              version extends the previous one&mdash;the core architecture
              carries through all three.
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* SA-Co dataset */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-sm font-medium text-foreground">
              The SA-Co dataset
            </p>
            <p className="text-muted-foreground">
              SAM 3 required a new dataset for concept-level understanding:{' '}
              <strong>SA-Co</strong>&mdash;5.2M images, 52.5K videos, 4M+
              unique noun phrases, ~1.4B masks. Created with a data engine
              combining AI annotators, human reviewers, and LLM-generated
              concept ontologies. The same data engine philosophy from SA-1B,
              now applied to concept-level annotations.
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 15: Check 2 -- Transfer Question
          ================================================================ */}
      <Row>
        <Row.Content>
          <GradientCard title="Check Your Understanding" color="cyan">
            <div className="space-y-3 text-sm">
              <p>
                A colleague is building a wildlife monitoring system. Cameras in
                a national park capture thousands of images daily. They want to
                automatically segment and count all elephants in each image.
              </p>

              <p className="mt-4">
                <strong>Question 1:</strong> Which version of SAM would they
                need, and why?
              </p>
              <details className="mt-2">
                <summary className="font-medium cursor-pointer text-primary">
                  Reveal
                </summary>
                <div className="mt-2">
                  <p>
                    <strong>SAM 3.</strong> They need text prompting
                    (&ldquo;elephant&rdquo;) to find all instances
                    automatically. SAM 1/2 would require clicking each elephant
                    manually in every image&mdash;impractical for thousands of
                    daily images.
                  </p>
                </div>
              </details>

              <p className="mt-4">
                <strong>Question 2:</strong> Why is SAM&rsquo;s image encoder
                amortization less important for this use case?
              </p>
              <details className="mt-2">
                <summary className="font-medium cursor-pointer text-primary">
                  Reveal
                </summary>
                <div className="mt-2">
                  <p>
                    It is automated batch processing, not interactive use. Each
                    image is processed once with one prompt
                    (&ldquo;elephant&rdquo;). The per-prompt efficiency matters
                    less than total throughput. The amortization shines during{' '}
                    <em>interactive</em> exploration.
                  </p>
                </div>
              </details>

              <p className="mt-4">
                <strong>Question 3:</strong> If the cameras also capture video,
                what SAM 2 capability becomes useful?
              </p>
              <details className="mt-2">
                <summary className="font-medium cursor-pointer text-primary">
                  Reveal
                </summary>
                <div className="mt-2">
                  <p>
                    The <strong>memory mechanism</strong> for tracking elephants
                    across frames. An elephant partially hidden behind a tree
                    can be tracked because the memory stores what it looked like
                    before occlusion. The cross-attention over the memory bank
                    naturally handles re-identification when the elephant
                    reappears.
                  </p>
                </div>
              </details>
            </div>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 16: SAM in the Vision Ecosystem
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="SAM in the Vision Ecosystem"
            subtitle="Impact beyond segmentation"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              SAM&rsquo;s impact extends beyond segmentation into two major
              areas:
            </p>
            <p className="text-muted-foreground">
              <strong>As a data annotation tool:</strong> SAM dramatically
              accelerates the creation of segmentation datasets for any domain.
              What took hours per image now takes seconds. This changes the
              economics of training specialized models&mdash;you no longer need
              millions of dollars in annotation costs.
            </p>
            <p className="text-muted-foreground">
              <strong>As a foundation model component:</strong> SAM&rsquo;s
              image encoder (trained on 1B masks) produces rich visual features
              that transfer to other vision tasks. Like using CLIP or SigLIP as
              a vision backbone, SAM features encode fine-grained spatial
              understanding that classification-trained models do not have.
            </p>
          </div>
        </Row.Content>
      </Row>

      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-sm font-medium text-foreground">
              The foundation model pattern: three instances you have seen
            </p>
            <div className="grid gap-4 md:grid-cols-3">
              <GradientCard title="GPT" color="blue">
                <p>
                  One model for all language tasks, prompted with text. Trained
                  on massive text corpora.
                </p>
              </GradientCard>
              <GradientCard title="CLIP / SigLIP" color="violet">
                <p>
                  One model for vision-language alignment, prompted with images
                  or text. Trained on image-text pairs.
                </p>
              </GradientCard>
              <GradientCard title="SAM" color="emerald">
                <p>
                  One model for all segmentation tasks, prompted with points,
                  boxes, or text. Trained on 1B+ masks.
                </p>
              </GradientCard>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The Foundation Model Recipe">
            Every foundation model in this course follows the same recipe:
            train on a massive, diverse dataset using a task that forces the
            model to learn general representations. Then use{' '}
            <strong>prompting</strong> to steer the model to specific tasks
            without retraining.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 17: Summary (expanded)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SummaryBlock
            title="Key Takeaways"
            items={[
              {
                headline:
                  'Image segmentation produces per-pixel masks\u2014richer than classification (one label) or detection (bounding box).',
                description:
                  'Segmentation traces exact object boundaries. A mask is a grid of 0s and 1s the same size as the image, capturing every pixel of the object.',
              },
              {
                headline:
                  'SAM is a promptable segmentation foundation model: ViT encoder (heavy, once) + prompt encoder + lightweight mask decoder (fast, per prompt).',
                description:
                  'The asymmetric compute design\u2014heavy encoder once, light decoder per prompt\u2014enables interactive use. Three familiar components, one new composition.',
              },
              {
                headline:
                  'Prompt encoding uses Fourier positional encoding + learned type embeddings.',
                description:
                  'A click at (x, y) is encoded using sin/cos at 128 frequency bands\u2014the same sinusoidal encoding idea from transformers, applied to 2D spatial coordinates. Projected to 256-dim, plus a type embedding.',
              },
              {
                headline:
                  'The mask decoder runs three attention operations per layer: self-attention, token-to-image, and image-to-token cross-attention.',
                description:
                  'Five tokens (1 prompt + 3 mask + 1 IoU) interact with 4,096 image positions across just 2 layers. Masks are generated via dot product between learned mask tokens and upsampled image features.',
              },
              {
                headline:
                  'The loss combines focal loss (hard-pixel focus) + dice loss (global overlap) with minimum-loss assignment.',
                description:
                  'Focal loss auto-suppresses easy pixels (14,600\u00d7 ratio). Dice loss measures mask-level overlap. Minimum-loss across the 3 candidates lets each mask specialize in a different granularity.',
              },
              {
                headline:
                  'SAM 2 adds video via memory: cross-attention over stored frame memories.',
                description:
                  'A memory bank stores features from recent frames. Current-frame features attend to memory\u2014same cross-attention formula, applied across time. Occlusion handling falls out naturally.',
              },
              {
                headline:
                  'SAM evolved additively: SAM 1 (images) \u2192 SAM 2 (+ video memory) \u2192 SAM 3 (+ text/concept understanding).',
                description:
                  'Each version extends the previous. SAM 3 still supports point/box prompts. The core architecture carries through all three.',
              },
              {
                headline:
                  'SAM follows the foundation model pattern: massive pretraining + prompting.',
                description:
                  'The same trajectory as GPT (language) and CLIP/SigLIP (vision-language). Train on broad data, prompt for specific tasks. SAM applies this recipe to segmentation.',
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
              SAM is the universal cookie cutter. Traditional segmentation needs
              one cutter per shape. SAM adjusts its shape based on what you
              point at&mdash;or, with SAM 3, based on what you describe.
            </strong>
          </InsightBlock>
        </Row.Content>
      </Row>

      {/* References */}
      <Row>
        <Row.Content>
          <ReferencesBlock
            references={[
              {
                title: 'Segment Anything',
                authors: 'Kirillov et al., 2023 (Meta AI)',
                url: 'https://arxiv.org/abs/2304.02643',
                note: 'The original SAM paper. Section 3 describes the architecture; Section 4 covers the data engine. Appendix A has prompt encoding details.',
              },
              {
                title: 'SAM 2: Segment Anything in Images and Videos',
                authors: 'Ravi et al., 2024 (Meta FAIR)',
                url: 'https://arxiv.org/abs/2408.00714',
                note: 'SAM 2 paper. Section 3 covers the streaming architecture, memory mechanism, and memory attention.',
              },
              {
                title: 'Focal Loss for Dense Object Detection',
                authors: 'Lin et al., 2017 (Facebook AI Research)',
                url: 'https://arxiv.org/abs/1708.02002',
                note: 'The original focal loss paper (RetinaNet). Section 3 derives the focal loss formula and explains the gamma parameter.',
              },
              {
                title: 'SAM 2.1 and SA-V Dataset Release',
                authors: 'Meta FAIR, 2024',
                url: 'https://github.com/facebookresearch/sam2',
                note: 'Open-source SAM 2 code and model weights. Includes interactive demo.',
              },
              {
                title: 'Segment Anything Model 3 (SA3)',
                authors: 'Meta FAIR, 2025',
                url: 'https://ai.meta.com/blog/segment-anything-model-3/',
                note: 'SAM 3 announcement. Covers concept-level segmentation, text prompting, and the SA-Co dataset.',
              },
            ]}
          />
        </Row.Content>
      </Row>

      {/* Next Step */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              SAM showed us how the &ldquo;foundation model for X&rdquo; pattern
              applies to segmentation. Every domain in vision is following this
              same trajectory: build a foundation model, train on unprecedented
              data, and make it promptable. The Special Topics series is here for
              whatever you want to explore next.
            </p>
          </div>
        </Row.Content>
      </Row>

      <Row>
        <Row.Content>
          <NextStepBlock
            href="/app"
            title="Lesson Complete"
            description="Return to the course home to explore more Special Topics or review your progress."
          />
        </Row.Content>
      </Row>
    </LessonLayout>
  )
}

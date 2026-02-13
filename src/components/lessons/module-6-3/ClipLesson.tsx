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
import { CodeBlock } from '@/components/common/CodeBlock'
import { MermaidDiagram } from '@/components/widgets/MermaidDiagram'
import 'katex/dist/katex.min.css'
import { InlineMath, BlockMath } from 'react-katex'

/**
 * CLIP — Contrastive Learning and Shared Embedding Spaces
 *
 * Lesson 3 in Module 6.3 (Architecture & Conditioning). Lesson 12 overall in Series 6.
 * Cognitive load: STRETCH (genuinely new training paradigm).
 *
 * Teaches how CLIP learns a shared embedding space where text and images
 * can be compared directly:
 * - Contrastive learning as a training paradigm
 * - CLIP's dual-encoder architecture (image + text encoder)
 * - The shared embedding space and cosine similarity
 * - The contrastive loss function (symmetric cross-entropy on similarity matrix)
 * - Zero-shot transfer as an emergent property
 * - What CLIP embeddings represent (and what they do NOT)
 *
 * Core concepts:
 * - Contrastive learning paradigm: DEVELOPED
 * - CLIP dual-encoder architecture: INTRODUCED
 * - Shared embedding space: DEVELOPED
 * - Zero-shot classification: INTRODUCED
 * - Contrastive loss formula: INTRODUCED
 *
 * Previous: Conditioning the U-Net (module 6.3, lesson 2)
 * Next: Text Conditioning & Guidance (module 6.3, lesson 4)
 */

export function ClipLesson() {
  return (
    <LessonLayout>
      {/* Section 1: Header */}
      <Row>
        <Row.Content>
          <LessonHeader
            title="CLIP"
            description="How contrastive learning trains two separate encoders to put matching text and images near each other in a shared embedding space&mdash;giving the U-Net a way to understand what to generate."
            category="Architecture & Conditioning"
          />
        </Row.Content>
      </Row>

      {/* Section 2: Context + Constraints */}
      <Row>
        <Row.Content>
          <ObjectiveBlock>
            Understand how CLIP trains two encoders (one for images, one for text) to
            produce embeddings in a shared space where matching text-image pairs end up
            nearby&mdash;using contrastive learning with cosine similarity and symmetric
            cross-entropy loss. By the end, you will know exactly what a CLIP text
            embedding <em>is</em> and why it is the right signal for controlling image
            generation.
          </ObjectiveBlock>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="What You Already Know">
            In <strong>Embeddings and Position</strong>, you saw tokens cluster by
            meaning in embedding space. In <strong>Exploring Latent Spaces</strong>,
            you saw images organize by similarity in VAE latent space. In{' '}
            <strong>Queries and Keys</strong>, you used dot products to measure
            similarity. All of these building blocks come together in CLIP.
          </TipBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <ConstraintBlock
            title="Scope for This Lesson"
            items={[
              'How CLIP trains two encoders to produce aligned embeddings (the contrastive learning paradigm)',
              'The shared embedding space and why it enables text-image comparison',
              'The contrastive loss function (symmetric cross-entropy on the similarity matrix)',
              'Zero-shot classification as an emergent property of the shared space',
              'NOT: how CLIP connects to the U-Net—that is the next lesson',
              'NOT: cross-attention or classifier-free guidance—next lesson',
              'NOT: implementing CLIP from scratch—it requires enormous compute; understanding is the goal',
              'NOT: Vision Transformer (ViT) architecture details—mentioned, not developed',
            ]}
          />
        </Row.Content>
      </Row>

      {/* Section 3: Recap — Embedding Spaces and Self-Supervision */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Quick Recap: Separate Spaces"
            subtitle="Text and images have lived in different worlds"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Remember how the EmbeddingSpaceExplorer showed &ldquo;dog&rdquo; and
              &ldquo;puppy&rdquo; clustering nearby in embedding space? And how the
              VAE latent space put similar images near each other?{' '}
              <strong>Geometry encodes meaning.</strong> But so far, text embeddings
              and image embeddings have lived in completely separate spaces. A token
              embedding for &ldquo;cat&rdquo; and a VAE encoding of a cat photo share
              no coordinate system&mdash;comparing them by cosine similarity would be
              meaningless. Like comparing GPS coordinates to temperatures.
            </p>
            <p className="text-muted-foreground">
              You have also seen models that generate their own training labels.
              Autoencoders use the input as the target. Language models use shifted
              text as the target. CLIP takes this further&mdash;the internet already
              paired images with text descriptions. Every image on the web has
              alt-text, a title, a caption. No human labeler needed. Four hundred
              million naturally occurring text-image pairs, ready to learn from.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="Three Forms of Free Supervision">
            <ul className="space-y-1 text-sm">
              <li>&bull; <strong>Autoencoder:</strong> input = target</li>
              <li>&bull; <strong>Language model:</strong> shifted text = target</li>
              <li>&bull; <strong>CLIP:</strong> internet text-image pairs = target</li>
            </ul>
            Same principle: find supervision in data that already exists.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* Section 4: Hook — The Control Problem */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Control Problem"
            subtitle="Your diffusion model cannot understand words"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Your diffusion model generates images&mdash;but you cannot tell it{' '}
              <em>what</em> to generate. The U-Net processes tensors. It has no idea
              what the word &ldquo;cat&rdquo; means. To control generation with text,
              you need to turn words into the same kind of mathematical object the
              U-Net already works with: <strong>vectors</strong>. But not just any
              vectors&mdash;vectors that <em>mean the same thing</em> as the words.
            </p>
            <p className="text-muted-foreground">
              This is not a trivial problem. A text encoder (transformer) and an image
              encoder (CNN) produce vectors in completely different spaces. Cosine
              similarity between their outputs would be random noise&mdash;the two
              encoders were never told that &ldquo;a photo of a cat&rdquo; and an
              actual photo of a cat should be related.
            </p>
            <p className="text-muted-foreground">
              The answer: <strong>train them together</strong> so they{' '}
              <em>learn</em> to produce compatible vectors. That is CLIP.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Same Building Blocks, Different Question">
            CLIP uses CNN encoders (Series 3), transformer encoders (Series 4),
            cosine similarity (<strong>Queries and Keys</strong>), and cross-entropy
            loss (<strong>Transfer Learning</strong>). Every piece is familiar.
            The question is new: &ldquo;do this text and this image match?&rdquo;
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Section 5a: Explain — The Dual-Encoder Architecture */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Dual-Encoder Architecture"
            subtitle="Two encoders, one shared space"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              CLIP is <strong>not a single neural network</strong>. It is two separate
              encoders trained together:
            </p>
            <div className="grid gap-4 md:grid-cols-2">
              <GradientCard title="Image Encoder" color="blue">
                <ul className="space-y-1">
                  <li>&bull; A CNN (ResNet) or Vision Transformer (ViT)</li>
                  <li>&bull; Input: an image (224&times;224 pixels)</li>
                  <li>&bull; Output: a single vector (e.g., 512 dimensions)</li>
                  <li>&bull; Like the feature extractor from <strong>Transfer Learning</strong>&mdash;but with no classification head</li>
                </ul>
              </GradientCard>
              <GradientCard title="Text Encoder" color="violet">
                <ul className="space-y-1">
                  <li>&bull; A Transformer (like the ones from Series 4)</li>
                  <li>&bull; Input: a text caption (tokenized)</li>
                  <li>&bull; Output: a single vector (same 512 dimensions)</li>
                  <li>&bull; The final token&rsquo;s embedding, projected to match the image encoder&rsquo;s output size</li>
                </ul>
              </GradientCard>
            </div>
            <p className="text-muted-foreground">
              Both encoders project their output into the <strong>same
              dimensionality</strong>. At inference time, you can use either encoder
              alone&mdash;the image encoder can encode images without any text, and
              vice versa. They share no weights. The only thing connecting them is the
              loss function.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Two Encoders, Not One">
            Most models you have seen are single networks (one CNN, one transformer,
            one U-Net). CLIP has <strong>two</strong> separate encoders that process
            their inputs independently. They are trained simultaneously but share
            no weights&mdash;only the loss function connects them.
          </WarningBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <MermaidDiagram chart={`
graph LR
    IMG["Image\\n(224×224)"] --> IE["Image Encoder\\n(ResNet or ViT)"]
    IE --> IV["Image Vector\\n(512-dim)"]
    TXT["Caption\\n(tokens)"] --> TE["Text Encoder\\n(Transformer)"]
    TE --> TV["Text Vector\\n(512-dim)"]
    IV --> COS["Cosine\\nSimilarity"]
    TV --> COS

    style IE fill:#1e293b,stroke:#3b82f6,color:#e2e8f0
    style TE fill:#1e293b,stroke:#8b5cf6,color:#e2e8f0
    style COS fill:#1e293b,stroke:#f59e0b,color:#e2e8f0
`} />
          <p className="text-sm text-muted-foreground italic mt-2">
            Two independent encoders produce vectors in the same 512-dimensional
            space. Cosine similarity measures how well they match.
          </p>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="Vision Transformer (ViT)">
            ViT processes image patches like tokens&mdash;the same transformer
            architecture from Series 4, applied to image patches instead of words.
            CLIP can use either a CNN or ViT as its image encoder. The key point:
            any encoder that produces a fixed-size vector from its input works.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* Section 5b: The Shared Embedding Space */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Shared Embedding Space"
            subtitle="Where text meaning meets visual meaning"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              After training, both encoders map their inputs into the same geometric
              space. A photo of a cat and the text &ldquo;a photo of a cat&rdquo;
              end up as nearby points. A photo of a cat and the text &ldquo;a red
              sports car&rdquo; end up far apart.
            </p>
            <p className="text-muted-foreground">
              This is the same principle you experienced in the EmbeddingSpaceExplorer
              from <strong>Embeddings and Position</strong>&mdash;but now{' '}
              <strong>both text and images</strong> live in the same space.
              &ldquo;Dog&rdquo; the word and a photo of a dog are neighbors.
            </p>

            {/* Before/after embedding space diagram */}
            <div className="grid gap-6 md:grid-cols-2">
              <div className="rounded-lg bg-muted/30 p-4 space-y-3">
                <p className="text-sm font-medium text-foreground text-center">
                  Before Training
                </p>
                <svg viewBox="0 0 220 180" className="w-full max-w-[220px] mx-auto" aria-label="Before training: image and text embeddings scattered randomly with no alignment">
                  {/* Image embeddings (circles) — clustered left */}
                  <circle cx="35" cy="45" r="6" fill="#3b82f6" opacity="0.8" />
                  <circle cx="55" cy="70" r="6" fill="#3b82f6" opacity="0.8" />
                  <circle cx="30" cy="100" r="6" fill="#3b82f6" opacity="0.8" />
                  <circle cx="60" cy="130" r="6" fill="#3b82f6" opacity="0.8" />
                  {/* Text embeddings (triangles) — clustered right */}
                  <polygon points="175,40 181,52 169,52" fill="#8b5cf6" opacity="0.8" />
                  <polygon points="155,80 161,92 149,92" fill="#8b5cf6" opacity="0.8" />
                  <polygon points="185,110 191,122 179,122" fill="#8b5cf6" opacity="0.8" />
                  <polygon points="160,145 166,157 154,157" fill="#8b5cf6" opacity="0.8" />
                  {/* Labels */}
                  <text x="45" y="165" fontSize="9" fill="#94a3b8" textAnchor="middle">images</text>
                  <text x="170" y="165" fontSize="9" fill="#94a3b8" textAnchor="middle">text</text>
                  {/* Question mark in center */}
                  <text x="110" y="95" fontSize="18" fill="#475569" textAnchor="middle">?</text>
                </svg>
                <p className="text-xs text-muted-foreground text-center">
                  Two separate clusters. No alignment&mdash;cosine similarity between them is meaningless.
                </p>
              </div>
              <div className="rounded-lg bg-muted/30 p-4 space-y-3">
                <p className="text-sm font-medium text-foreground text-center">
                  After Training
                </p>
                <svg viewBox="0 0 220 180" className="w-full max-w-[220px] mx-auto" aria-label="After training: matching image-text pairs nearby, non-matching pairs far apart, with push-together and push-apart arrows">
                  {/* Pair 1: cat — nearby */}
                  <circle cx="45" cy="40" r="6" fill="#3b82f6" opacity="0.8" />
                  <polygon points="60,35 66,47 54,47" fill="#8b5cf6" opacity="0.8" />
                  {/* Push-together arrow */}
                  <line x1="51" y1="40" x2="55" y2="40" stroke="#22c55e" strokeWidth="1.5" markerEnd="url(#arrowGreen)" />
                  <text x="52" y="30" fontSize="7" fill="#94a3b8" textAnchor="middle">cat</text>
                  {/* Pair 2: car — nearby */}
                  <circle cx="160" cy="45" r="6" fill="#3b82f6" opacity="0.8" />
                  <polygon points="175,40 181,52 169,52" fill="#8b5cf6" opacity="0.8" />
                  <line x1="166" y1="45" x2="170" y2="45" stroke="#22c55e" strokeWidth="1.5" markerEnd="url(#arrowGreen)" />
                  <text x="168" y="32" fontSize="7" fill="#94a3b8" textAnchor="middle">car</text>
                  {/* Pair 3: mountain — nearby */}
                  <circle cx="50" cy="135" r="6" fill="#3b82f6" opacity="0.8" />
                  <polygon points="65,130 71,142 59,142" fill="#8b5cf6" opacity="0.8" />
                  <line x1="56" y1="135" x2="60" y2="135" stroke="#22c55e" strokeWidth="1.5" markerEnd="url(#arrowGreen)" />
                  <text x="57" y="122" fontSize="7" fill="#94a3b8" textAnchor="middle">mountain</text>
                  {/* Pair 4: ramen — nearby */}
                  <circle cx="165" cy="140" r="6" fill="#3b82f6" opacity="0.8" />
                  <polygon points="180,135 186,147 174,147" fill="#8b5cf6" opacity="0.8" />
                  <line x1="171" y1="140" x2="175" y2="140" stroke="#22c55e" strokeWidth="1.5" markerEnd="url(#arrowGreen)" />
                  <text x="172" y="127" fontSize="7" fill="#94a3b8" textAnchor="middle">ramen</text>
                  {/* Push-apart arrow between cat and car */}
                  <line x1="70" y1="42" x2="150" y2="44" stroke="#ef4444" strokeWidth="1" strokeDasharray="4,3" />
                  {/* Push-apart arrow between cat and mountain */}
                  <line x1="50" y1="50" x2="50" y2="125" stroke="#ef4444" strokeWidth="1" strokeDasharray="4,3" />
                  {/* Legend */}
                  <circle cx="30" cy="170" r="4" fill="#3b82f6" opacity="0.8" />
                  <text x="38" y="173" fontSize="7" fill="#94a3b8">image</text>
                  <polygon points="70,166 74,174 66,174" fill="#8b5cf6" opacity="0.8" />
                  <text x="80" y="173" fontSize="7" fill="#94a3b8">text</text>
                  <line x1="105" y1="170" x2="115" y2="170" stroke="#22c55e" strokeWidth="1.5" />
                  <text x="120" y="173" fontSize="7" fill="#94a3b8">push together</text>
                  <line x1="165" y1="170" x2="175" y2="170" stroke="#ef4444" strokeWidth="1" strokeDasharray="4,3" />
                  <text x="180" y="173" fontSize="7" fill="#94a3b8">push apart</text>
                  {/* Arrow marker definitions */}
                  <defs>
                    <marker id="arrowGreen" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
                      <path d="M0,0 L6,3 L0,6 Z" fill="#22c55e" />
                    </marker>
                  </defs>
                </svg>
                <p className="text-xs text-muted-foreground text-center">
                  Matching pairs pulled together (green). Non-matching pairs pushed apart (red dashed).
                </p>
              </div>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Geometry Encodes Meaning—Across Modalities">
            In the token embedding space, &ldquo;dog&rdquo; and &ldquo;puppy&rdquo;
            clustered nearby. In the VAE latent space, similar images clustered
            nearby. In CLIP&rsquo;s space, an image of a dog and the text
            &ldquo;a photo of a dog&rdquo; cluster nearby. Same principle, new
            reach.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Address misconception: CLIP vs VAE latent space */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              This looks like the VAE latent space, but there is a{' '}
              <strong>critical difference</strong>. The VAE has a decoder&mdash;you
              can sample a point and generate an image. CLIP has{' '}
              <strong>no decoder</strong>. You cannot generate images from CLIP
              embeddings. CLIP&rsquo;s space is for{' '}
              <strong>comparison</strong>: &ldquo;is this text similar to this
              image?&rdquo; The VAE&rsquo;s space is for{' '}
              <strong>generation</strong>: &ldquo;what image lives at this
              point?&rdquo;
            </p>
            <ComparisonRow
              left={{
                title: 'VAE Latent Space',
                color: 'amber',
                items: [
                  'Has a decoder (encode → decode → image)',
                  'Trained with reconstruction + KL loss',
                  'Can sample and generate new images',
                  'Purpose: generation',
                ],
              }}
              right={{
                title: 'CLIP Embedding Space',
                color: 'blue',
                items: [
                  'No decoder (encode only)',
                  'Trained with contrastive loss',
                  'Can compare, not generate',
                  'Purpose: comparison and matching',
                ],
              }}
            />
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="No Decoder, No Generation">
            You experienced latent space interpolation, arithmetic, and sampling
            with VAEs. None of that transfers to CLIP. There is no decoder to
            map CLIP embeddings back to images. The spaces serve fundamentally
            different purposes.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* Negative example: independently trained encoders */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              <strong>Negative example: why alignment is not free.</strong> What if
              you took a pretrained ResNet (image encoder) and a pretrained
              transformer (text encoder) and compared their embeddings with cosine
              similarity? The result: random correlation. The two encoders were
              trained independently with different objectives. Their embedding
              spaces have no shared structure&mdash;the image encoder organized
              its space for classification, the text encoder organized its space for
              language modeling. Matching them is like trying to use a French-English
              dictionary to translate Mandarin.
            </p>
            <p className="text-muted-foreground">
              The shared space is not a property of the encoders. It is a property
              of the <strong>training</strong>. The contrastive loss function is what
              forces the two spaces into alignment. Same building blocks, different
              question&mdash;and the question is everything.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The Loss Creates the Alignment">
            Any two encoders can produce 512-dim vectors. That does not make them
            compatible. The contrastive loss function <em>teaches</em> them to
            agree on what the dimensions mean. Without it, they speak different
            languages.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Section 5c: Contrastive Learning */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Contrastive Learning"
            subtitle="The training paradigm that aligns two spaces"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              How do you train two encoders to produce aligned embeddings? You need a
              training signal that rewards matching pairs for being nearby and
              penalizes non-matching pairs for being close. This is{' '}
              <strong>contrastive learning</strong>.
            </p>
            <p className="text-muted-foreground">
              <strong>Analogy: name tag matching at a conference.</strong> N people
              enter a room, each wearing a name tag (text) and carrying a photo of
              themselves (image). The training game: given any photo, pick the
              matching name tag from everyone in the room. Given any name tag, pick
              the matching photo. With 2 people, this is trivial. With 32,768
              people, this forces you to learn genuinely discriminative
              representations&mdash;you need to notice the details that distinguish
              each person from 32,767 others.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="The Conference Analogy">
            <ul className="space-y-1 text-sm">
              <li>&bull; <strong>People in the room</strong> = batch of text-image pairs</li>
              <li>&bull; <strong>Name tags</strong> = text embeddings</li>
              <li>&bull; <strong>Photos</strong> = image embeddings</li>
              <li>&bull; <strong>More people</strong> = harder task = better learning</li>
            </ul>
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* First example: 4x4 similarity matrix */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-sm font-medium text-foreground">
              Concrete example: 4 text-image pairs
            </p>
            <p className="text-muted-foreground">
              Four images and four captions. Each caption goes through the text
              encoder to produce a 512-dim vector. Each image goes through the image
              encoder to produce a 512-dim vector. Compute cosine similarity for all
              16 pairs:
            </p>
            {/* Visual similarity matrix heatmap */}
            <div className="overflow-x-auto">
              <table className="mx-auto text-xs border-collapse">
                <thead>
                  <tr>
                    <th className="p-2" />
                    <th className="p-2 text-center font-medium text-muted-foreground">cat image</th>
                    <th className="p-2 text-center font-medium text-muted-foreground">car image</th>
                    <th className="p-2 text-center font-medium text-muted-foreground">mountain image</th>
                    <th className="p-2 text-center font-medium text-muted-foreground">ramen image</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td className="p-2 text-right font-medium text-muted-foreground pr-3">&ldquo;a tabby cat&rdquo;</td>
                    <td className="p-2 text-center rounded font-bold text-white bg-violet-500">0.85</td>
                    <td className="p-2 text-center rounded text-muted-foreground bg-muted/40">0.12</td>
                    <td className="p-2 text-center rounded text-muted-foreground bg-muted/30">0.08</td>
                    <td className="p-2 text-center rounded text-muted-foreground bg-muted/30">0.10</td>
                  </tr>
                  <tr>
                    <td className="p-2 text-right font-medium text-muted-foreground pr-3">&ldquo;a red sports car&rdquo;</td>
                    <td className="p-2 text-center rounded text-muted-foreground bg-muted/30">0.11</td>
                    <td className="p-2 text-center rounded font-bold text-white bg-violet-500">0.82</td>
                    <td className="p-2 text-center rounded text-muted-foreground bg-muted/40">0.15</td>
                    <td className="p-2 text-center rounded text-muted-foreground bg-muted/30">0.09</td>
                  </tr>
                  <tr>
                    <td className="p-2 text-right font-medium text-muted-foreground pr-3">&ldquo;a mountain view&rdquo;</td>
                    <td className="p-2 text-center rounded text-muted-foreground bg-muted/30">0.07</td>
                    <td className="p-2 text-center rounded text-muted-foreground bg-muted/30">0.10</td>
                    <td className="p-2 text-center rounded font-bold text-white bg-violet-500">0.88</td>
                    <td className="p-2 text-center rounded text-muted-foreground bg-muted/40">0.13</td>
                  </tr>
                  <tr>
                    <td className="p-2 text-right font-medium text-muted-foreground pr-3">&ldquo;a bowl of ramen&rdquo;</td>
                    <td className="p-2 text-center rounded text-muted-foreground bg-muted/30">0.09</td>
                    <td className="p-2 text-center rounded text-muted-foreground bg-muted/40">0.14</td>
                    <td className="p-2 text-center rounded text-muted-foreground bg-muted/30">0.11</td>
                    <td className="p-2 text-center rounded font-bold text-white bg-violet-500">0.84</td>
                  </tr>
                </tbody>
              </table>
            </div>
            <p className="text-xs text-muted-foreground text-center mt-1">
              Bright diagonal = matching pairs (high similarity). Dark off-diagonal = non-matching pairs.
              Same shape as the attention weight matrices from <strong>Queries and Keys</strong>.
            </p>
            <p className="text-muted-foreground">
              <strong>The goal of training: make the diagonal bright and everything
              else dark.</strong> Each diagonal entry is a correct match. Each
              off-diagonal entry is a mismatch that the loss pushes apart.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="You Have Seen This Shape Before">
            This similarity matrix looks like the attention weight matrices from{' '}
            <strong>Queries and Keys</strong>. Both are square matrices where
            entry (i, j) measures how much item i relates to item j. Similar
            shape, different use.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Walk through one row: classification framing */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Walk through <strong>one row</strong>. For the text &ldquo;a tabby
              cat,&rdquo; the similarities are [0.85, 0.12, 0.08, 0.10]. This is a{' '}
              <strong>classification problem</strong>: the &ldquo;correct
              answer&rdquo; is column 0 (the cat image). Apply cross-entropy with
              label = 0.
            </p>
            <p className="text-muted-foreground">
              Walk through <strong>one column</strong>. For the cat image, the
              similarities are [0.85, 0.11, 0.07, 0.09]. Same thing: the
              &ldquo;correct answer&rdquo; is row 0 (the cat caption). Apply
              cross-entropy with label = 0.
            </p>
            <p className="text-muted-foreground">
              Every row is a classification problem: &ldquo;which image matches this
              text?&rdquo; Every column is a classification problem: &ldquo;which
              text matches this image?&rdquo; The labels are always the diagonal:
              [0, 1, 2, ..., N&minus;1].
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="It Is Just Cross-Entropy">
            CLIP&rsquo;s loss is cross-entropy&mdash;the same loss you used for
            MNIST classification and language modeling. The only difference: the
            &ldquo;classes&rdquo; are the other items in the batch, and the
            &ldquo;logits&rdquo; are cosine similarities.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Address misconception: where do negatives come from? */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              <strong>Where did the negative examples come from?</strong> In the 4x4
              matrix, each image has 1 matching caption and 3 non-matching ones. But
              nobody labeled any pair as &ldquo;not matching.&rdquo; The other 3
              images in the batch <em>are</em> the negatives. In a batch of N
              text-image pairs, each image has 1 positive match and N&minus;1
              negatives&mdash;<strong>for free</strong>. The batch structure provides
              the negatives automatically.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="No Explicit Negative Labels">
            Every training setup you have seen uses explicit labels or
            self-generated targets. Contrastive learning is different: the
            non-matching pairs arise naturally from the other correct pairs in
            the batch. No human ever labels anything as &ldquo;not
            matching.&rdquo;
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* Section 5d: The Loss Function (symbolic) */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Loss Function"
            subtitle="Symmetric cross-entropy on the similarity matrix"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The CLIP loss is cross-entropy applied twice&mdash;once per modality:
            </p>
            <div className="py-4 px-6 bg-muted/50 rounded-lg space-y-3">
              <p className="text-sm font-medium text-foreground">
                Step 1: Compute the similarity matrix
              </p>
              <BlockMath math="S_{ij} = \cos(\mathbf{I}_i, \mathbf{T}_j) \cdot e^{\tau}" />
              <p className="text-sm font-medium text-foreground">
                Step 2: Labels are the diagonal
              </p>
              <BlockMath math="\text{labels} = [0, 1, 2, \ldots, N{-}1]" />
              <p className="text-sm font-medium text-foreground">
                Step 3: Cross-entropy in both directions
              </p>
              <BlockMath math="L_{\text{image}} = \text{CrossEntropy}(S, \text{labels})" />
              <BlockMath math="L_{\text{text}} = \text{CrossEntropy}(S^\top, \text{labels})" />
              <p className="text-sm font-medium text-foreground">
                Step 4: Average
              </p>
              <BlockMath math="L = \frac{1}{2}(L_{\text{image}} + L_{\text{text}})" />
            </div>
            <p className="text-muted-foreground">
              That is the entire loss. <InlineMath math="L_{\text{image}}" /> treats
              each row as a classification: &ldquo;which text matches this
              image?&rdquo; <InlineMath math="L_{\text{text}}" /> treats each column
              as a classification: &ldquo;which image matches this text?&rdquo; The
              average ensures neither direction dominates.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="The Temperature Parameter">
            In <strong>What Is a Language Model</strong>, you divided logits by
            temperature&mdash;higher T meant a <em>softer</em> distribution.
            Here, <InlineMath math="e^{\tau}" /> <em>multiplies</em> the
            logits, so higher <InlineMath math="\tau" /> means a{' '}
            <em>sharper</em> distribution. Same principle, opposite
            convention&mdash;multiplying by a large number has the same effect
            as dividing by a small one. CLIP learns{' '}
            <InlineMath math="\tau" /> during training.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* Code version of the loss */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              In PyTorch, the entire training step fits in a few lines:
            </p>
            <CodeBlock
              code={`# CLIP training step (pseudocode)
image_features = image_encoder(images)          # [N, 512]
text_features  = text_encoder(captions)         # [N, 512]

# Normalize to unit vectors (so dot product = cosine similarity)
image_features = image_features / image_features.norm(dim=-1, keepdim=True)
text_features  = text_features  / text_features.norm(dim=-1, keepdim=True)

# Similarity matrix, scaled by learned temperature
logits = (image_features @ text_features.T) * exp(temperature)  # [N, N]

# Labels: each image i matches text i
labels = torch.arange(N)                         # [0, 1, 2, ..., N-1]

# Symmetric cross-entropy
loss_i = cross_entropy(logits, labels)           # rows: which text?
loss_t = cross_entropy(logits.T, labels)         # cols: which image?
loss   = (loss_i + loss_t) / 2`}
              language="python"
              filename="clip_training.py"
            />
            <p className="text-muted-foreground">
              No exotic loss function. No adversarial training. No reconstruction
              objective. Just cross-entropy on a similarity matrix&mdash;the same
              cognitive relief as &ldquo;DDPM&rsquo;s loss is just MSE&rdquo; from{' '}
              <strong>Learning to Denoise</strong>.
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* Section 6: Check — Predict and Verify */}
      <Row>
        <Row.Content>
          <GradientCard title="Predict and Verify" color="cyan">
            <div className="space-y-3 text-sm">
              <p>
                <strong>Question 1:</strong> You have a batch of 8 text-image pairs.
                Row 3 (for the text &ldquo;a sunset over the ocean&rdquo;) has
                similarities [0.1, 0.1, 0.2, 0.9, 0.1, 0.1, 0.1, 0.2]. What is
                the cross-entropy label for this row?
              </p>
              <details className="mt-2">
                <summary className="font-medium cursor-pointer text-primary">
                  Reveal
                </summary>
                <div className="mt-2">
                  <p>
                    <strong>3</strong>&mdash;the diagonal entry. Image 3 is the
                    matching image for text 3. The high similarity at position [3,3]
                    = 0.9 is correct; the loss rewards this.
                  </p>
                </div>
              </details>

              <p className="mt-4">
                <strong>Question 2:</strong> What happens if the model accidentally
                makes the similarity between &ldquo;a sunset over the ocean&rdquo;
                and a mountain photo very high (entry [3,2] = 0.8)?
              </p>
              <details className="mt-2">
                <summary className="font-medium cursor-pointer text-primary">
                  Reveal
                </summary>
                <div className="mt-2">
                  <p>
                    <strong>High loss for row 3.</strong> Cross-entropy penalizes the
                    model for putting probability mass on the wrong entry. The
                    gradient pushes the model to increase entry [3,3] (the correct
                    match) and decrease entry [3,2] (the false match).
                  </p>
                </div>
              </details>

              <p className="mt-4">
                <strong>Question 3:</strong> In a batch of 32,768, how many negative
                examples does each image see?
              </p>
              <details className="mt-2">
                <summary className="font-medium cursor-pointer text-primary">
                  Reveal
                </summary>
                <div className="mt-2">
                  <p>
                    <strong>32,767.</strong> One positive match (its own caption) and
                    32,767 negatives (everyone else&rsquo;s captions). The task is
                    far harder than with a batch of 4, which forces the model to
                    learn highly discriminative features.
                  </p>
                </div>
              </details>
            </div>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* Section 7: Explore — Scaling Thought Experiment */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Why Scale Matters"
            subtitle="Batch size changes the difficulty, not just the speed"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              In standard training (classification, language modeling), larger batch
              sizes primarily affect gradient noise and training speed. In contrastive
              learning, batch size fundamentally changes the{' '}
              <strong>difficulty of the task</strong>.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Batch Size Is Different Here">
            In Series 1&ndash;2, larger batches meant faster training and smoother
            gradients. In contrastive learning, larger batches create a{' '}
            <strong>harder task</strong> with more informative gradients. This is
            qualitatively different.
          </WarningBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <div className="space-y-4">
            <div className="grid gap-4 md:grid-cols-3">
              <GradientCard title="Batch = 2" color="emerald">
                <ul className="space-y-1">
                  <li>&bull; Distinguish from 1 wrong caption</li>
                  <li>&bull; 50% random chance</li>
                  <li>&bull; Any rough feature suffices</li>
                  <li>&bull; Task too easy&mdash;model barely learns</li>
                </ul>
              </GradientCard>
              <GradientCard title="Batch = 32" color="blue">
                <ul className="space-y-1">
                  <li>&bull; Distinguish from 31 wrong captions</li>
                  <li>&bull; ~3% random chance</li>
                  <li>&bull; Needs category-level features</li>
                  <li>&bull; Learns broad distinctions</li>
                </ul>
              </GradientCard>
              <GradientCard title="Batch = 32,768" color="violet">
                <ul className="space-y-1">
                  <li>&bull; Distinguish from 32,767 wrong captions</li>
                  <li>&bull; ~0.003% random chance</li>
                  <li>&bull; Needs fine-grained features</li>
                  <li>&bull; Learns highly discriminative representations</li>
                </ul>
              </GradientCard>
            </div>
            <p className="text-muted-foreground">
              CLIP used a batch size of 32,768. With 32,767 negatives per example,
              the model must learn extremely fine-grained representations to tell
              matching pairs apart from near-matches. This is why contrastive
              learning at small scale often fails&mdash;with only 32 negatives, the
              task is too easy and the model does not learn discriminative features.
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* Section 8a: Elaborate — Scale and Training Data */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Training at Internet Scale"
            subtitle="400 million text-image pairs from the web"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              CLIP trained on 400 million text-image pairs scraped from the
              internet&mdash;the WebImageText (WIT) dataset. The captions are not
              carefully curated. They are alt-text, titles, descriptions that humans
              wrote for their own purposes.
            </p>
            <p className="text-muted-foreground">
              <strong>Second example:</strong> a web page has an image of a golden
              retriever and the alt-text &ldquo;photo of a golden retriever playing
              fetch in the park.&rdquo; This is one training pair. Multiply by 400
              million. The captions are noisy, informal, sometimes wrong&mdash;but
              400 million of them provide an enormously rich training signal. No
              human sat down and categorized images into 1,000 classes like
              ImageNet. The supervision is free and abundant.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Natural Language Supervision">
            ImageNet has 1,000 predefined categories. CLIP has no predefined
            categories at all. The &ldquo;categories&rdquo; are whatever text
            humans wrote. This means CLIP&rsquo;s representations are not limited
            to a fixed label set&mdash;they generalize to any text description.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Section 8b: Zero-Shot Classification */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Zero-Shot Classification"
            subtitle="Classification without training on the target classes"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              In <strong>Transfer Learning</strong>, you used a pretrained ResNet
              on a new dataset by replacing the classification head and fine-tuning.
              Feature extraction required retraining the head on the new classes.
              CLIP does not even need that.
            </p>
            <p className="text-muted-foreground">
              To classify an ImageNet image with CLIP&mdash;without any training
              on ImageNet:
            </p>
            <ol className="list-decimal list-inside text-muted-foreground space-y-2 ml-4">
              <li>Encode each of 1,000 class names as text: &ldquo;a photo of a [class name]&rdquo;</li>
              <li>Encode the test image</li>
              <li>Find the text embedding with highest cosine similarity to the image embedding</li>
              <li>That is the predicted class</li>
            </ol>
            <p className="text-muted-foreground">
              This works because the shared embedding space generalizes beyond
              training pairs. If CLIP has seen cats and text about cats during
              training, it can match &ldquo;a photo of a cat&rdquo; to a new cat
              image it has never seen. The text encoder <em>is</em> the
              classification head.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Beyond Transfer Learning">
            In <strong>Transfer Learning</strong>, you replaced the classification
            head and retrained on the new dataset. CLIP skips both steps. The
            shared embedding space already encodes the relationship between
            visual concepts and their text descriptions.
          </InsightBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <CodeBlock
            code={`# Zero-shot classification with CLIP (pseudocode)
# No training on ImageNet — just encode and compare

class_names = ["cat", "dog", "car", "airplane", ...]   # 1,000 classes
text_prompts = [f"a photo of a {name}" for name in class_names]

# Encode all class descriptions (once, then reuse)
text_features = text_encoder(text_prompts)               # [1000, 512]
text_features = text_features / text_features.norm(dim=-1, keepdim=True)

# Encode the test image
image_feature = image_encoder(test_image)                # [1, 512]
image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True)

# Find best match
similarities = image_feature @ text_features.T           # [1, 1000]
predicted_class = similarities.argmax()                   # the winner`}
            language="python"
            filename="zero_shot_classification.py"
          />
        </Row.Content>
      </Row>

      {/* Section 8c: What CLIP Does NOT Understand */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="What CLIP Does NOT Understand"
            subtitle="Useful does not mean perfect"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              CLIP learns statistical co-occurrence, not deep understanding. The
              impressive zero-shot results can make it tempting to anthropomorphize,
              but CLIP has clear, systematic failure modes:
            </p>
            <div className="grid gap-4 md:grid-cols-2">
              <GradientCard title="Typographic Attacks" color="rose">
                <p>
                  Placing the word &ldquo;iPod&rdquo; on a physical apple makes CLIP
                  classify it as an iPod. The model cannot distinguish text printed
                  on an object from the object itself.
                </p>
              </GradientCard>
              <GradientCard title="Spatial Relationships" color="rose">
                <p>
                  CLIP struggles to distinguish &ldquo;a red cube on a blue
                  cube&rdquo; from &ldquo;a blue cube on a red cube.&rdquo; The
                  embedding does not reliably encode spatial structure.
                </p>
              </GradientCard>
              <GradientCard title="Counting" color="rose">
                <p>
                  CLIP cannot reliably distinguish &ldquo;three dogs&rdquo; from
                  &ldquo;five dogs.&rdquo; Quantity is not well represented in the
                  embedding space.
                </p>
              </GradientCard>
              <GradientCard title="Novel Compositions" color="rose">
                <p>
                  CLIP handles common pairings well but struggles with unusual
                  combinations it rarely saw during training, like &ldquo;an
                  astronaut riding a horse.&rdquo;
                </p>
              </GradientCard>
            </div>
            <p className="text-muted-foreground">
              CLIP&rsquo;s embedding space encodes{' '}
              <strong>what things are</strong> and{' '}
              <strong>what words mean</strong>, well enough to match them. It does
              not encode spatial relationships, counts, or causal reasoning.
              Understanding the limitations is as important as understanding the
              capabilities.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="Statistical Co-occurrence">
            CLIP learned that the word &ldquo;iPod&rdquo; co-occurs with images
            containing iPod-like visual features&mdash;but also with images
            containing the text &ldquo;iPod.&rdquo; It has no grounding in
            physical reality. Patterns in data, not understanding of the world.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* Section 9: Check — Transfer Questions */}
      <Row>
        <Row.Content>
          <GradientCard title="Check Your Understanding" color="cyan">
            <div className="space-y-3 text-sm">
              <p>
                <strong>Question 1:</strong> Your colleague says: &ldquo;I&rsquo;ll
                just take a pretrained ResNet image encoder and a pretrained BERT
                text encoder, compute cosine similarity between their outputs, and
                get the same results as CLIP.&rdquo; Why won&rsquo;t this work?
              </p>
              <details className="mt-2">
                <summary className="font-medium cursor-pointer text-primary">
                  Reveal
                </summary>
                <div className="mt-2">
                  <p>
                    The two encoders were trained independently with different
                    objectives. Their embedding spaces are not aligned. Cosine
                    similarity between them is meaningless. CLIP&rsquo;s
                    contrastive training is what <strong>creates</strong> the
                    alignment&mdash;without it, the two encoders speak different
                    languages.
                  </p>
                </div>
              </details>

              <p className="mt-4">
                <strong>Question 2:</strong> Could you use CLIP to generate images
                from text?
              </p>
              <details className="mt-2">
                <summary className="font-medium cursor-pointer text-primary">
                  Reveal
                </summary>
                <div className="mt-2">
                  <p>
                    <strong>No.</strong> CLIP can compare text and images but
                    cannot generate. It has no decoder. To generate, you need a
                    model like the diffusion model from Module 6.2&mdash;but you
                    can use CLIP&rsquo;s text embeddings to{' '}
                    <strong>steer</strong> that generation. That is the next lesson.
                  </p>
                </div>
              </details>
            </div>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* Section 10: Notebook Exercises */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Practice: Hands-On Exercises"
            subtitle="Explore CLIP embeddings in a notebook"
          />
          <div className="space-y-4">
            <div className="rounded-lg border-2 border-primary/50 bg-primary/5 p-6">
              <div className="space-y-4">
                <p className="text-muted-foreground">
                  Compute similarity matrices, explore a pretrained CLIP model, and
                  probe its capabilities and limitations yourself.
                </p>
                <a
                  href="https://colab.research.google.com/github/brennancheung/CourseAI/blob/main/notebooks/6-3-3-clip.ipynb"
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
                  Exercise 1: Compute a Similarity Matrix by Hand (Guided)
                </p>
                <p className="text-sm text-muted-foreground">
                  Given 4 pre-computed embedding vectors (2-dim for simplicity) for
                  images and 4 for text, compute all 16 cosine similarities. Fill in
                  a 4&times;4 matrix. Identify the diagonal. Compute cross-entropy
                  loss for one row. The key insight: each row IS a classification
                  problem.
                </p>
              </div>
              <div className="rounded-lg bg-muted/30 p-4 space-y-2">
                <p className="text-sm font-medium text-foreground">
                  Exercise 2: Explore a Pretrained CLIP Model (Guided)
                </p>
                <p className="text-sm text-muted-foreground">
                  Load OpenAI&rsquo;s CLIP model (ViT-B/32). Encode 5 provided
                  images and 5 text prompts. Compute the 5&times;5 similarity
                  matrix. Visualize as a heatmap and observe the bright diagonal.
                  Find cases where off-diagonal entries are suspiciously high.
                </p>
              </div>
              <div className="rounded-lg bg-muted/30 p-4 space-y-2">
                <p className="text-sm font-medium text-foreground">
                  Exercise 3: Zero-Shot Classification (Supported)
                </p>
                <p className="text-sm text-muted-foreground">
                  Use CLIP to classify 10 images from CIFAR-10 without any training.
                  Create text prompts &ldquo;a photo of a [class]&rdquo; for all 10
                  classes. For each image, find the most similar text prompt. Compare
                  to random chance (10%) and to a trained classifier.
                </p>
              </div>
              <div className="rounded-lg bg-muted/30 p-4 space-y-2">
                <p className="text-sm font-medium text-foreground">
                  Exercise 4: Probing CLIP&rsquo;s Limitations (Independent)
                </p>
                <p className="text-sm text-muted-foreground">
                  Test CLIP with edge cases: spatial relationships (&ldquo;a cat on
                  a table&rdquo; vs &ldquo;a table on a cat&rdquo;), counting
                  (&ldquo;three dogs&rdquo; vs &ldquo;five dogs&rdquo;), and
                  abstract concepts. Document where CLIP succeeds and where it fails.
                </p>
              </div>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="What to Focus On">
            Exercises 1&ndash;3 are the core. Exercise 4 is the most fun&mdash;you
            get to break things. If time is short, prioritize Exercises 2 and 3
            (the ones using real CLIP).
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Section 11: Summary */}
      <Row>
        <Row.Content>
          <SummaryBlock
            title="Key Takeaways"
            items={[
              {
                headline:
                  'Two encoders, one shared space\u2014the loss function creates the alignment.',
                description:
                  'CLIP trains a separate image encoder and text encoder simultaneously. The contrastive loss forces their embedding spaces into alignment. The architecture does not create the shared space\u2014the training does.',
              },
              {
                headline:
                  'Negatives come from the batch\u2014no explicit negative labels needed.',
                description:
                  'In a batch of N text-image pairs, each image has 1 positive match and N\u22121 negatives. The batch structure provides the negatives for free. Larger batches create a harder task with more informative gradients.',
              },
              {
                headline:
                  'The shared space enables zero-shot transfer\u2014match any text to any image.',
                description:
                  'Because text and image embeddings live in the same space, you can classify images by comparing to text descriptions of classes. No task-specific training required.',
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
              CLIP trains two encoders to put matching text and images near each
              other in a shared embedding space.
            </strong>{' '}
            The training signal is contrastive: in a batch of N pairs, maximize
            similarity for the N matches, minimize it for the N&sup2;&minus;N
            non-matches. Same building blocks (CNN, transformer, cross-entropy
            loss), different question (&ldquo;do this text and image
            match?&rdquo;).
          </InsightBlock>
        </Row.Content>
      </Row>

      {/* Connection back to the course */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              That contrastive question gives us text embeddings that encode visual
              meaning&mdash;exactly what the U-Net needs to generate images from text
              descriptions. The next step: how to inject those embeddings so the
              U-Net actually listens.
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* References */}
      <Row>
        <Row.Content>
          <ReferencesBlock
            references={[
              {
                title: 'Learning Transferable Visual Models From Natural Language Supervision',
                authors: 'Radford et al., 2021 (OpenAI)',
                url: 'https://arxiv.org/abs/2103.00020',
                note: 'The CLIP paper. Section 2 describes the contrastive training approach. Section 3 covers zero-shot transfer.',
              },
              {
                title: 'An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale',
                authors: 'Dosovitskiy et al., 2021',
                url: 'https://arxiv.org/abs/2010.11929',
                note: 'The Vision Transformer (ViT) paper. CLIP uses ViT as one of its image encoder options.',
              },
              {
                title: 'High-Resolution Image Synthesis with Latent Diffusion Models',
                authors: 'Rombach et al., 2022',
                url: 'https://arxiv.org/abs/2112.10752',
                note: 'The Stable Diffusion paper. Section 3.3 describes how CLIP text embeddings condition the U-Net via cross-attention.',
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
              You now know what CLIP text embeddings represent: vectors in a shared
              space where text meaning aligns with visual meaning. The next question:
              how do you inject these embeddings into the U-Net so it actually uses
              them during denoising? The answer is cross-attention&mdash;a mechanism
              you already know from Series 4.2, applied in a new context.
            </p>
            <p className="text-muted-foreground">
              The timestep tells the network <em>when</em> it is in the denoising
              process. The text will tell it <em>what</em> to generate.
            </p>
          </div>
        </Row.Content>
      </Row>

      <Row>
        <Row.Content>
          <NextStepBlock
            href="/app"
            title="Up Next: Text Conditioning & Guidance"
            description="How cross-attention injects CLIP text embeddings into the U-Net&mdash;and how classifier-free guidance amplifies their influence."
          />
        </Row.Content>
      </Row>
    </LessonLayout>
  )
}

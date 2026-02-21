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
  ModuleCompleteBlock,
  ReferencesBlock,
  LessonLink,
} from '@/components/lessons'
import { CodeBlock } from '@/components/common/CodeBlock'
import { MermaidDiagram } from '@/components/widgets/MermaidDiagram'
import { ExternalLink } from 'lucide-react'

const NOTEBOOK_URL =
  'https://colab.research.google.com/github/brennancheung/CourseAI/blob/main/notebooks/6-5-3-textual-inversion.ipynb'

/**
 * Textual Inversion
 *
 * Lesson 3 in Module 6.5 (Customization). Final lesson of Series 6.
 * Cognitive load: STRETCH (2 new concepts).
 *
 * Two genuinely new concepts:
 *   1. Pseudo-token creation and embedding optimization (add a new token,
 *      initialize its embedding, optimize it via the DDPM training loop
 *      while freezing all model weights)
 *   2. Textual inversion's expressiveness boundaries compared to LoRA
 *      (one 768-dim vector vs thousands of LoRA parameters)
 *
 * The optimization target (a single embedding vector rather than model
 * weights or inference configuration) is conceptually novel even though
 * every component is familiar. The surprise is not mechanical complexity
 * but conceptual reframing: instead of changing what the model does,
 * you change what the model hears.
 *
 * Previous: Img2Img and Inpainting (Module 6.5, Lesson 2)
 * Next: Series 7 (Post-SD Advances)
 */

export function TextualInversionLesson() {
  return (
    <LessonLayout>
      {/* ================================================================
          Section 1: Header
          ================================================================ */}
      <Row>
        <Row.Content>
          <LessonHeader
            title="Textual Inversion"
            description="Don't change the model. Don't change the inference process. Change what the model hears&mdash;optimize a single embedding vector to teach it a new word."
            category="Customization"
          />
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 2: Objective + Constraints
          ================================================================ */}
      <Row>
        <Row.Content>
          <ObjectiveBlock>
            Understand how textual inversion creates a new pseudo-token
            (e.g., {'<my-cat>'}) and optimizes its 768-dimensional embedding
            vector in CLIP&rsquo;s embedding space to represent a novel
            concept, while keeping the entire model&mdash;U-Net, VAE, CLIP
            text encoder&mdash;completely frozen.
          </ObjectiveBlock>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="What You Already Know">
            You built token embeddings from scratch in{' '}
            <strong>Embeddings &amp; Position</strong>, explored CLIP&rsquo;s
            shared embedding space, and just completed two customization
            lessons (LoRA + img2img/inpainting). Every component of textual
            inversion is familiar&mdash;the surprise is where you point the
            optimizer.
          </TipBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <ConstraintBlock
            title="Scope for This Lesson"
            items={[
              'Creating a pseudo-token and optimizing its embedding in CLIP\'s text embedding space',
              'The two-stage pipeline (embedding lookup → CLIP transformer) and where textual inversion intervenes',
              'The training loop: same DDPM loop, only one embedding vector receives gradients',
              'Initialization strategies (random vs from a related word)',
              'Expressiveness comparison with LoRA (one vector vs thousands of parameters)',
              'When to use textual inversion vs LoRA vs img2img',
              'NOT: DreamBooth (full or partial model fine-tuning)',
              'NOT: multi-token textual inversion (mentioned only)',
              'NOT: hypernetworks or other embedding-space methods',
              'NOT: CLIP internal architecture details or reimplementing CLIP',
            ]}
          />
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 3: Recap
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Three Premises"
            subtitle="Everything you need to derive this technique yourself"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              <strong>Premise 1: The two-stage CLIP text encoding pipeline.</strong>{' '}
              From <LessonLink slug="stable-diffusion-architecture">The Stable Diffusion Pipeline</LessonLink>: the tokenizer
              converts text to integer IDs, <code>nn.Embedding</code> looks up
              initial vectors for each token, then the CLIP transformer applies
              self-attention across all tokens to produce contextual embeddings
              of shape [77, 768]. Those contextual embeddings feed into the
              U-Net&rsquo;s cross-attention as K and V.
            </p>
            <p className="text-muted-foreground">
              <strong>Premise 2: Embeddings are learned parameters.</strong>{' '}
              From <LessonLink slug="embeddings-and-position">Embeddings &amp; Position</LessonLink>: each row of the
              embedding table is a trainable tensor with{' '}
              <code>requires_grad=True</code>. You verified this yourself&mdash;
              <code>embedding.weight[i]</code> is the same tensor you get from{' '}
              <code>embedding(tensor([i]))</code>. Embeddings are the first
              layer of the model, not a preprocessing step.
            </p>
            <p className="text-muted-foreground">
              <strong>Premise 3: CLIP&rsquo;s embedding space is continuous
              and meaningful.</strong> From <LessonLink slug="clip">CLIP</LessonLink>: nearby
              vectors represent similar concepts. You explored this geometry
              interactively with the EmbeddingSpaceExplorer&mdash;semantic
              clusters, meaningful distances, the insight that
              &ldquo;geometry encodes meaning.&rdquo;
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="Why These Three">
            These are not arbitrary facts. They are the three premises from
            which textual inversion follows logically. If you internalize
            them, you can derive the technique before being told.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 4: Hook — "What if you could invent a new word?"
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="What If You Could Invent a New Word?"
            subtitle="A challenge you can solve from what you already know"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              You have five photos of your cat. You want to generate
              &ldquo;my cat wearing a spacesuit on Mars.&rdquo; No existing
              prompt can describe your specific cat precisely enough for
              Stable Diffusion to produce it. LoRA could work, but it modifies
              thousands of parameters across the entire U-Net. You do not
              need to change how the model processes text or generates
              images. You need to change what one word means.
            </p>
            <GradientCard title="Derivation Challenge" color="cyan">
              <p className="text-sm">
                You know three things: (1) Every word maps to a 768-dim
                vector via an embedding lookup. (2) These vectors are
                trainable parameters. (3) CLIP&rsquo;s embedding space is
                continuous and meaningful.
              </p>
              <p className="text-sm mt-2">
                From these three facts alone, can you derive a technique
                for teaching the model your cat?
              </p>
              <details className="mt-3">
                <summary className="font-medium cursor-pointer text-primary text-sm">
                  Reveal the technique
                </summary>
                <div className="mt-2 text-sm text-muted-foreground space-y-2">
                  <p>
                    Create a new token&mdash;call it{' '}
                    <code>&lt;my-cat&gt;</code>&mdash;and give it a trainable
                    embedding vector. Freeze the entire model (U-Net, VAE,
                    CLIP encoder). Optimize only that one embedding vector
                    so that, when the prompt &ldquo;a photo of{' '}
                    <code>&lt;my-cat&gt;</code>&rdquo; is processed, the
                    model generates images that look like your cat.
                  </p>
                  <p>
                    This technique is called <strong>textual inversion</strong>{' '}
                    (Gal et al., 2022). You just reinvented it.
                  </p>
                </div>
              </details>
            </GradientCard>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Not Changing the Machine">
            LoRA changes what the model <em>does</em> with embeddings
            (modified cross-attention projections). Img2img/inpainting
            changes the inference <em>process</em>. Textual inversion
            changes what the model <em>hears</em>&mdash;the input
            representation itself.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 5: Explain — The Textual Inversion Mechanism
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Pseudo-Token"
            subtitle="Adding one row to the embedding table"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              CLIP&rsquo;s vocabulary has 49,408 entries. Each entry maps a
              token ID to a 768-dimensional vector. Textual inversion adds
              one more entry: assign the pseudo-token{' '}
              <code>&lt;my-cat&gt;</code> the next available ID (49409) and
              extend the embedding table by one row. That row&mdash;a single
              768-float vector&mdash;is the{' '}
              <strong>only trainable parameter</strong>.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="One Row">
            The embedding table goes from [49408, 768] to [49409, 768].
            One new row. 768 new trainable floats. That is the entire
            &ldquo;model&rdquo; you are training.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Initialization strategies */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Initialization"
            subtitle="Where does the new embedding start?"
          />
          <div className="mt-2">
            <ComparisonRow
              left={{
                title: 'Random Initialization',
                color: 'amber',
                items: [
                  'Start at a random point in CLIP\'s 768-dim space',
                  'No prior knowledge of the concept',
                  'Works, but slower convergence',
                  'The optimizer must navigate from an arbitrary starting point',
                ],
              }}
              right={{
                title: 'Initialize from Related Word',
                color: 'emerald',
                items: [
                  'Start from the embedding of "cat"',
                  'Begins in a semantically relevant region',
                  'Faster convergence—less distance to travel',
                  'Geometry encodes meaning: nearby = similar concept',
                ],
              }}
            />
          </div>
          <div className="space-y-4 mt-4">
            <p className="text-muted-foreground">
              Initializing from a related word connects directly to the
              &ldquo;geometry encodes meaning&rdquo; mental model from{' '}
              <strong>Embeddings &amp; Position</strong>. If similar tokens
              cluster nearby in embedding space, starting near
              &ldquo;cat&rdquo; puts the optimization in the right
              neighborhood. The optimizer refines from there rather than
              searching the entire 768-dimensional space.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Geometry Guides Optimization">
            You explored semantic clusters in the EmbeddingSpaceExplorer.
            &ldquo;Cat,&rdquo; &ldquo;kitten,&rdquo; and &ldquo;tabby&rdquo;
            were nearby. Starting the optimization in that neighborhood means
            starting where the space already encodes &ldquo;cat-like
            things.&rdquo;
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Training loop comparison */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Training Loop"
            subtitle="Same DDPM loop. Different optimization target."
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              You have seen the DDPM training loop three times: in{' '}
              <LessonLink slug="learning-to-denoise">Learning to Denoise</LessonLink> (the original), in{' '}
              <LessonLink slug="build-a-diffusion-model">Build a Diffusion Model</LessonLink> (your implementation),
              and in <LessonLink slug="lora-finetuning">LoRA Fine-Tuning</LessonLink> (with frozen base and
              LoRA params). Textual inversion uses the same loop a fourth
              time. Compare it to the LoRA version:
            </p>
          </div>
          <div className="mt-4">
            <ComparisonRow
              left={{
                title: 'LoRA Training Step',
                color: 'violet',
                items: [
                  '1. Load (image, caption) pair',
                  '2. VAE encode image (frozen)',
                  '3. Sample t, add noise to latent',
                  '4. U-Net predicts noise (LoRA adapters active)',
                  '5. MSE loss on noise prediction',
                  '6. Backprop into LoRA params only',
                ],
              }}
              right={{
                title: 'Textual Inversion Training Step',
                color: 'cyan',
                items: [
                  '1. Load (image, caption with <my-cat>)',
                  '2. VAE encode image (frozen)',
                  '3. Sample t, add noise to latent',
                  '4. U-Net predicts noise (all weights frozen)',
                  '5. MSE loss on noise prediction',
                  '6. Backprop into one embedding vector only',
                ],
              }}
            />
          </div>
          <div className="space-y-4 mt-4">
            <p className="text-muted-foreground">
              Steps 1&ndash;5 are nearly identical. The only difference is
              step 6: <strong>what receives the gradient update</strong>.
              LoRA updates adapter matrices at the cross-attention projections.
              Textual inversion updates a single row in the embedding table.
              The &ldquo;freeze everything except X&rdquo; pattern from{' '}
              <strong>LoRA Fine-Tuning</strong> transfers directly&mdash;the
              only change is what X is.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="Same Pattern, Different Target">
            <code>for p in model.parameters():</code><br />
            <code>&nbsp;&nbsp;p.requires_grad_(False)</code><br />
            <code>embed[new_id].requires_grad_(True)</code><br />
            <br />
            The &ldquo;freeze everything except X&rdquo; pattern. In LoRA,
            X was the adapter matrices. Here, X is one embedding row.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* Gradient flow diagram */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Gradient Flow"
            subtitle="How gradients reach a single embedding through frozen layers"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              In LoRA training, gradients flowed into the cross-attention
              projections&mdash;relatively close to the loss. In textual
              inversion, gradients must flow all the way back through the
              frozen U-Net <em>and</em> the frozen CLIP text encoder to reach
              the embedding table at the very start of the pipeline. Every
              intermediate layer is frozen (no weight updates), but gradients
              still pass through them to reach the one trainable parameter.
            </p>
          </div>
          <div className="mt-4">
            <MermaidDiagram chart={`
              graph LR
                EMB["Embedding Lookup<br/>(ONE row TRAINABLE)"]:::trainable
                CLIP["CLIP Transformer<br/>(frozen)"]:::frozen
                CROSS["Cross-Attention K/V<br/>(frozen)"]:::frozen
                UNET["U-Net Denoise<br/>(frozen)"]:::frozen
                LOSS["MSE Loss"]:::loss

                EMB -->|"forward"| CLIP
                CLIP -->|"[77, 768]"| CROSS
                CROSS -->|"spatial features"| UNET
                UNET -->|"noise prediction"| LOSS
                LOSS -.->|"gradient ∇"| UNET
                UNET -.->|"gradient ∇"| CROSS
                CROSS -.->|"gradient ∇"| CLIP
                CLIP -.->|"gradient ∇ updates!"| EMB

                classDef trainable fill:#7c3aed,stroke:#a78bfa,color:#f5f3ff
                classDef frozen fill:#374151,stroke:#6b7280,color:#d1d5db
                classDef loss fill:#065f46,stroke:#34d399,color:#d1fae5
            `} />
          </div>
          <div className="space-y-4 mt-4">
            <p className="text-muted-foreground">
              Solid arrows: the forward pass. Dashed arrows: gradient
              backpropagation. Every frozen layer (gray) passes gradients
              through without updating its own weights. Only the embedding
              row (violet) receives the update. Compare this to the LoRA
              diagram from <LessonLink slug="lora-finetuning">LoRA Fine-Tuning</LessonLink>&mdash;there,
              the violet was at the cross-attention projections. Here,
              it is at the embedding table.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Frozen ≠ Invisible to Gradients">
            Frozen layers do not update their weights, but gradients still
            flow <em>through</em> them during backpropagation. The CLIP
            encoder is a frozen pathway&mdash;gradients pass through its
            transformer layers to reach the embedding, but its ~123M
            parameters have zero gradient updates.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* Zero weight change verification */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <GradientCard title="Zero Weight Change Verification" color="orange">
              <div className="space-y-2 text-sm">
                <p>
                  After textual inversion training, run any prompt that does
                  <strong> not</strong> contain{' '}
                  <code>&lt;my-cat&gt;</code> with the same seed. The output
                  is <strong>identical</strong> to before training. No weights
                  changed. The U-Net, VAE, and CLIP encoder behave exactly
                  as before for every prompt that does not route through the
                  new embedding.
                </p>
                <p>
                  Only prompts containing <code>&lt;my-cat&gt;</code>{' '}
                  produce different results, because only they look up the
                  new embedding row. The model is unchanged&mdash;the
                  dictionary grew by one word.
                </p>
              </div>
            </GradientCard>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Inventing a New Word">
            Imagine you speak a language where every word has a precise
            geometric position. Your cat has no word in this language.
            Textual inversion coins a new word&mdash;finding the exact
            position where, if a word existed there, speakers (the U-Net)
            would understand it to mean your specific cat. You are not
            changing the grammar (U-Net weights) or the accent (inference
            process). You are adding one word to the dictionary (embedding
            table).
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Pseudocode */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Setup in Code"
            subtitle="Five lines to configure, then the same training loop"
          />
          <div className="mt-2">
            <CodeBlock
              code={`# 1. Add new token to tokenizer
tokenizer.add_tokens(["<my-cat>"])
new_token_id = tokenizer.convert_tokens_to_ids("<my-cat>")  # 49409

# 2. Resize embedding table (adds one row of random values)
text_encoder.resize_token_embeddings(len(tokenizer))  # [49408, 768] → [49409, 768]

# 3. Optionally initialize from a related word
token_embeds = text_encoder.get_input_embeddings().weight.data
cat_id = tokenizer.convert_tokens_to_ids("cat")
token_embeds[new_token_id] = token_embeds[cat_id].clone()  # start near "cat"

# 4. Freeze EVERYTHING
for p in unet.parameters(): p.requires_grad_(False)
for p in vae.parameters(): p.requires_grad_(False)
for p in text_encoder.parameters(): p.requires_grad_(False)

# 5. Unfreeze ONLY the new embedding row
token_embeds[new_token_id].requires_grad_(True)
optimizer = Adam([token_embeds[new_token_id]], lr=5e-4)

# Training loop — identical to LoRA training from 6.5.1
for image, caption in dataset:          # caption: "a photo of <my-cat>"
    z_0 = vae.encode(image)             # [4, 64, 64]  (frozen)
    t = randint(0, T)
    epsilon = randn_like(z_0)
    z_t = sqrt(alpha_bar_t) * z_0 + sqrt(1 - alpha_bar_t) * epsilon
    text_emb = text_encoder(tokenize(caption))  # [1, 77, 768]  (frozen)
    epsilon_hat = unet(z_t, t, text_emb)        # [4, 64, 64]  (frozen)
    loss = MSE(epsilon, epsilon_hat)
    loss.backward()    # gradients flow through frozen layers to the one embedding
    optimizer.step()   # updates only token_embeds[49409]`}
              language="python"
              filename="textual_inversion_setup.py"
            />
          </div>
          <div className="space-y-4 mt-4">
            <p className="text-muted-foreground">
              Lines 1&ndash;5 are the setup. The training loop body is
              identical to <strong>LoRA Fine-Tuning</strong>&rsquo;s loop&mdash;
              the only difference is that <code>optimizer.step()</code>{' '}
              updates one 768-float vector instead of LoRA adapter matrices.
            </p>
            <GradientCard title="Pseudocode vs Reality" color="amber">
              <div className="space-y-2 text-sm">
                <p>
                  Lines 4&ndash;5 are <strong>conceptually simplified</strong>.
                  In PyTorch, calling{' '}
                  <code>requires_grad_(True)</code> on a single row of an
                  embedding table does not isolate that row for
                  optimization&mdash;the embedding table is one tensor, and
                  gradients flow to all rows that participate in the forward
                  pass via CLIP&rsquo;s self-attention.
                </p>
                <p>
                  The actual pattern: enable <code>requires_grad</code> on
                  the <em>full</em> embedding table, run the optimizer step,
                  then <strong>restore all rows except the
                  pseudo-token&rsquo;s</strong> to their original values.
                  The notebook shows this working implementation.
                </p>
              </div>
            </GradientCard>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Parameter Count">
            <ul className="space-y-1 text-sm">
              <li>&bull; Textual inversion: <strong>768</strong> parameters</li>
              <li>&bull; LoRA (r=4, ~16 blocks): <strong>~200K&ndash;400K</strong></li>
              <li>&bull; Full U-Net fine-tuning: <strong>~860M</strong></li>
            </ul>
            <p className="text-sm mt-2">
              Three orders of magnitude between textual inversion and LoRA.
            </p>
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 6: Check — Predict-and-Verify
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Check: Mechanism Understanding"
            subtitle="Predict, then verify"
          />
          <div className="space-y-4">
            <GradientCard title="Prediction 1: Unchanged Prompts" color="cyan">
              <p className="text-sm">
                After textual inversion training, you generate an image with
                the prompt &ldquo;a beautiful sunset over the ocean&rdquo;
                (no <code>&lt;my-cat&gt;</code> token). Has the output changed
                compared to before training?
              </p>
              <details className="mt-2">
                <summary className="font-medium cursor-pointer text-primary text-sm">
                  Reveal
                </summary>
                <p className="mt-2 text-sm text-muted-foreground">
                  No. The prompt does not contain the pseudo-token, so the
                  new embedding row is never looked up. The tokenizer maps
                  &ldquo;a beautiful sunset over the ocean&rdquo; to the
                  same token IDs as before. The CLIP encoder processes the
                  same input embeddings. The U-Net receives the same K/V
                  tensors. Identical output.
                </p>
              </details>
            </GradientCard>
            <GradientCard title="Prediction 2: Novel Composition" color="cyan">
              <p className="text-sm">
                You trained <code>&lt;my-cat&gt;</code> on cat photos. Then
                you try the prompt &ldquo;a photo of{' '}
                <code>&lt;my-cat&gt;</code> playing piano.&rdquo; Will the
                model know what a piano is, even though no training image
                showed a piano?
              </p>
              <details className="mt-2">
                <summary className="font-medium cursor-pointer text-primary text-sm">
                  Reveal
                </summary>
                <div className="mt-2 text-sm text-muted-foreground space-y-2">
                  <p>
                    Yes. The model&rsquo;s knowledge of pianos, composition,
                    and spatial relationships lives in the U-Net&rsquo;s
                    frozen weights. Textual inversion only taught the model
                    what <code>&lt;my-cat&gt;</code> looks like. The U-Net
                    already knows how to combine a subject with a scene.
                  </p>
                  <p>
                    Cross-attention lets spatial locations attend to both{' '}
                    <code>&lt;my-cat&gt;</code> and &ldquo;piano&rdquo;
                    independently. No special mechanism is needed&mdash;the
                    pseudo-token participates in cross-attention like any
                    other token.
                  </p>
                </div>
              </details>
            </GradientCard>
          </div>
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 7: Explore — Two-Stage Pipeline Distinction
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Two-Stage Pipeline"
            subtitle="Textual inversion optimizes stage 1—CLIP contextualizes in stage 2"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Trace the data flow for the prompt &ldquo;a photo of{' '}
              <code>&lt;my-cat&gt;</code> sitting on grass.&rdquo; At each
              stage, identify what is frozen and what is trained:
            </p>
            <ol className="list-decimal list-inside text-muted-foreground space-y-2 ml-4">
              <li>
                <strong>Tokenizer</strong> (frozen): maps characters to
                integer IDs, including the new{' '}
                <code>&lt;my-cat&gt;</code> &rarr; 49409
              </li>
              <li>
                <strong>Embedding lookup</strong>: row 49409 is the{' '}
                <strong>trained</strong> vector. All other rows are frozen.
              </li>
              <li>
                <strong>CLIP transformer self-attention</strong> (frozen):
                contextualizes the new embedding <em>with surrounding
                tokens</em>&mdash;this is the critical point
              </li>
              <li>
                <strong>Output</strong>: [77, 768] contextual embeddings,
                fed to the U-Net as cross-attention K/V (frozen)
              </li>
            </ol>
          </div>
          <div className="space-y-4 mt-4">
            <p className="text-muted-foreground">
              The key insight: textual inversion optimizes the embedding{' '}
              <strong>before</strong> the CLIP transformer processes it. The
              CLIP transformer then contextualizes it with surrounding words.
              This means the same initial embedding produces different
              contextual embeddings depending on the surrounding prompt.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Context Matters">
            &ldquo;a photo of <code>&lt;my-cat&gt;</code>&rdquo; and
            &ldquo;a sketch of <code>&lt;my-cat&gt;</code>&rdquo; start with
            the same initial embedding for the pseudo-token&mdash;but produce
            different contextual embeddings because the CLIP transformer sees
            different contexts. The pseudo-token adapts to context, just like
            real words do.
          </InsightBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <GradientCard
            title="This Is a Feature, Not a Limitation"
            color="emerald"
          >
            <div className="space-y-2 text-sm">
              <p>
                If the pseudo-token&rsquo;s embedding were processed in
                isolation&mdash;going directly to cross-attention without
                the CLIP transformer&mdash;context would not matter. The
                prompt &ldquo;a photo of <code>&lt;my-cat&gt;</code>&rdquo;
                and &ldquo;a watercolor of{' '}
                <code>&lt;my-cat&gt;</code>&rdquo; would produce identical
                K/V inputs.
              </p>
              <p>
                Instead, the CLIP transformer&rsquo;s self-attention lets
                surrounding words influence the pseudo-token&rsquo;s contextual
                representation. The result: a single learned embedding
                can adapt to diverse prompts, producing style-appropriate
                and composition-appropriate outputs.
              </p>
            </div>
          </GradientCard>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Common Misconception">
            The pseudo-token&rsquo;s embedding is <strong>not</strong>{' '}
            processed in isolation. It passes through CLIP&rsquo;s
            transformer, where self-attention contextualizes it with every
            other token in the prompt. The optimization target is the
            initial embedding (stage 1), but the U-Net sees the
            contextualized version (stage 2).
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 8: Elaborate — Expressiveness & the Complete Spectrum
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Expressiveness and Limitations"
            subtitle="What one vector can and cannot encode"
          />
          <div className="mt-2">
            <ComparisonRow
              left={{
                title: 'Textual Inversion',
                color: 'cyan',
                items: [
                  '768 trainable parameters (one embedding vector)',
                  'Trains in 3,000–5,000 steps',
                  'Output file: ~4 KB',
                  'Captures: object identity, simple visual attributes',
                  'Limited for: complex spatial styles, compositional recipes',
                ],
              }}
              right={{
                title: 'LoRA',
                color: 'violet',
                items: [
                  '200K–400K trainable parameters',
                  'Trains in 500–2,000 steps',
                  'Output file: 2–50 MB',
                  'Captures: style patterns, spatial composition, color palettes',
                  'Works well for: "how should images in this style look?"',
                ],
              }}
            />
          </div>
        </Row.Content>
      </Row>

      {/* When to use which */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="When to Use Which"
            subtitle="Three knobs for three different needs"
          />
          <div className="grid gap-3 md:grid-cols-3 mt-2">
            <GradientCard title="Textual Inversion" color="cyan">
              <ul className="space-y-1 text-sm">
                <li>&bull; &ldquo;I want to add MY specific cat/object/texture to any prompt&rdquo;</li>
                <li>&bull; Small file (~4 KB), easy to share</li>
                <li>&bull; No weight changes at all</li>
                <li>&bull; Best for: specific visual concepts</li>
              </ul>
            </GradientCard>
            <GradientCard title="LoRA" color="violet">
              <ul className="space-y-1 text-sm">
                <li>&bull; &ldquo;I want the model to generate in a specific style&rdquo;</li>
                <li>&bull; More parameters, more expressive</li>
                <li>&bull; Modifies cross-attention weights</li>
                <li>&bull; Best for: complex visual patterns</li>
              </ul>
            </GradientCard>
            <GradientCard title="Img2img / Inpainting" color="blue">
              <ul className="space-y-1 text-sm">
                <li>&bull; &ldquo;I want to edit or transform a specific existing image&rdquo;</li>
                <li>&bull; No training at all</li>
                <li>&bull; Inference-time only</li>
                <li>&bull; Best for: per-image editing</li>
              </ul>
            </GradientCard>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Three Knobs, Three Locations">
            LoRA: changes the <strong>weights</strong> (cross-attention
            projections). Img2img/inpainting: changes the{' '}
            <strong>inference process</strong> (starting point, spatial mask).
            Textual inversion: changes the <strong>embeddings</strong>{' '}
            (one entry in the embedding table).
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Limitations */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The &ldquo;One Word&rdquo; Ceiling"
            subtitle="What a single embedding vector cannot encode"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Textual inversion can encode what a concept <em>looks like</em>{' '}
              but not how to compose or light it. A single token participates
              in cross-attention at the token level&mdash;it advertises its
              content via K/V and spatial locations attend to it via Q. It
              cannot encode compositional instructions like &ldquo;always
              place the subject in the lower-left with dramatic
              backlighting.&rdquo;
            </p>
            <GradientCard
              title="Negative Example: Complex Style"
              color="rose"
            >
              <div className="space-y-2 text-sm">
                <p>
                  Training <code>&lt;my-cat&gt;</code> on photos of your cat
                  works well&mdash;a specific visual object maps naturally to
                  a single token descriptor. But training{' '}
                  <code>&lt;ghibli-style&gt;</code> to capture
                  &ldquo;Studio Ghibli animation style&rdquo; works poorly.
                  That style involves spatial composition, color grading,
                  lighting patterns, and rendering choices that span
                  multiple attention layers and spatial resolutions.
                </p>
                <p>
                  LoRA modifies thousands of projection parameters and can
                  encode spatial style patterns because it operates at the
                  architectural level. Textual inversion encodes a concept
                  descriptor, not a spatial recipe.
                </p>
              </div>
            </GradientCard>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="Multi-Token Variants">
            Using 2&ndash;4 pseudo-tokens for one concept gives more
            capacity (1,536&ndash;3,072 parameters). Each token can
            specialize&mdash;one for texture, one for color, one for shape.
            This extends the expressiveness, but remains less powerful than
            LoRA for complex styles.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* The learned embedding in geometric space */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Where the Learned Embedding Lives"
            subtitle="A new point in the continuous space"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              After training, the embedding sits at a specific point in
              CLIP&rsquo;s 768-dimensional space. It is <strong>not</strong>{' '}
              at any existing vocabulary token&rsquo;s position. If you
              compute the cosine similarity between the learned vector and
              all 49,408 existing entries, the learned vector is not close
              to any single token. It typically sits in a region of
              embedding space &ldquo;between&rdquo; several related
              tokens&mdash;near &ldquo;cat&rdquo; and &ldquo;fur&rdquo; and
              &ldquo;tabby&rdquo; but identical to none of them.
            </p>
            <p className="text-muted-foreground">
              Connect this to the EmbeddingSpaceExplorer from{' '}
              <strong>Embeddings &amp; Position</strong>: the continuous
              space has infinitely more positions than the discrete
              vocabulary has entries. Textual inversion finds a point that
              was always valid in the space but had no word assigned to it.
              The optimization does not search through existing vocabulary
              entries for the best match&mdash;it navigates the continuous
              space to find a genuinely new point.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Continuous vs Discrete">
            CLIP&rsquo;s vocabulary has 49,408 entries. CLIP&rsquo;s
            embedding space has <em>infinite</em> valid positions. Textual
            inversion finds one of the infinitely many unused positions
            that encodes your specific concept.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* CLIP model compatibility */}
      <Row>
        <Row.Content>
          <GradientCard title="Model Compatibility" color="blue">
            <div className="space-y-2 text-sm">
              <p>
                A textual inversion embedding trained on SD v1.4 transfers
                to SD v1.5 because both use the same CLIP text encoder
                (openai/clip-vit-large-patch14). But it does{' '}
                <strong>not</strong> transfer to SD v2.0, which uses a
                different CLIP encoder (OpenCLIP ViT-H). The embedding is
                a vector in a specific embedding space&mdash;it only means
                something in the space where it was trained.
              </p>
              <p>
                Think of it as speaking a word from one language in a
                conversation conducted in another language. The
                &ldquo;word&rdquo; only has meaning in the embedding space
                where it was optimized.
              </p>
            </div>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 9: Check — Transfer Questions
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Transfer Check"
            subtitle="Apply the mechanism to new situations"
          />
          <div className="space-y-4">
            <GradientCard title="Company Logo" color="cyan">
              <p className="text-sm">
                A colleague wants to teach the model what their company logo
                looks like, so they can generate marketing images in various
                contexts (&ldquo;our logo on a coffee mug,&rdquo; &ldquo;our
                logo as a neon sign&rdquo;). Would you recommend textual
                inversion or LoRA? Why?
              </p>
              <details className="mt-2">
                <summary className="font-medium cursor-pointer text-primary text-sm">
                  Reveal
                </summary>
                <div className="mt-2 text-sm text-muted-foreground space-y-2">
                  <p>
                    Textual inversion. A logo is a specific visual
                    object&mdash;exactly the kind of concept that maps to a
                    single token descriptor. The colleague wants to use the
                    logo in novel contexts, which means the model&rsquo;s
                    existing compositional knowledge should be preserved.
                    Textual inversion does not modify the model at all, so
                    all compositional abilities remain intact.
                  </p>
                  <p>
                    LoRA would work too but is overkill: modifying thousands
                    of weights to teach one visual concept. Textual
                    inversion&rsquo;s 768 parameters are sufficient and
                    produce a tiny, easily shareable file.
                  </p>
                </div>
              </details>
            </GradientCard>
            <GradientCard title="Cross-Model Transfer" color="cyan">
              <p className="text-sm">
                You trained a textual inversion embedding for your cat using
                SD v1.5. Now you want to use it with SDXL. Will the
                embedding work?
              </p>
              <details className="mt-2">
                <summary className="font-medium cursor-pointer text-primary text-sm">
                  Reveal
                </summary>
                <p className="mt-2 text-sm text-muted-foreground">
                  No. SD v1.5 uses openai/clip-vit-large-patch14 (768-dim
                  embeddings). SDXL uses a different text encoder
                  architecture (OpenCLIP ViT-bigG + a second encoder). The
                  embedding is a vector in a specific embedding space&mdash;it
                  only means something in the space where it was trained. You
                  would need to retrain the embedding using SDXL&rsquo;s
                  text encoder.
                </p>
              </details>
            </GradientCard>
          </div>
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 10: Practice — Notebook
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Practice: Textual Inversion"
            subtitle="Hands-on notebook exercises"
          />
          <div className="rounded-lg border-2 border-primary/50 bg-primary/5 p-6">
            <div className="space-y-4">
              <p className="text-muted-foreground">
                The notebook grounds the embedding mechanism in real models.
                You will inspect the CLIP embedding table, verify gradient
                flow through frozen layers, train a real textual inversion
                embedding, and compare it to LoRA.
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
              <ul className="text-sm text-muted-foreground space-y-2">
                <li>
                  <strong>Exercise 1 (Guided):</strong> Explore the CLIP
                  embedding table. Load a pretrained CLIP text encoder,
                  inspect the embedding table shape (49408 &times; 768), look
                  up embeddings for &ldquo;cat,&rdquo; &ldquo;dog,&rdquo;
                  &ldquo;kitten,&rdquo; compute cosine similarities. Add a
                  new token and verify the table grows to 49409 &times; 768.
                </li>
                <li>
                  <strong>Exercise 2 (Guided):</strong> One textual inversion
                  training step by hand. Load a training image, VAE encode
                  it, create the prompt &ldquo;a photo of{' '}
                  <code>&lt;my-concept&gt;</code>,&rdquo; run through the
                  full DDPM training step. Backprop and verify that frozen
                  model weights have zero gradients, then see the
                  restore-original-rows pattern that ensures only the
                  pseudo-token&rsquo;s embedding is actually updated.
                </li>
                <li>
                  <strong>Exercise 3 (Supported):</strong> Train a full
                  textual inversion embedding on 5&ndash;8 concept images for
                  3,000 steps. Generate images with and without the
                  pseudo-token. Compare results at checkpoints (500, 1500,
                  3000 steps).
                </li>
                <li>
                  <strong>Exercise 4 (Independent):</strong> Compare textual
                  inversion vs LoRA. Train both on the same concept images.
                  Generate from the same prompts using each. Compare file
                  size, training time, and output quality for simple vs
                  complex prompts.
                </li>
              </ul>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Exercise Progression">
            <ol className="space-y-1 text-sm list-decimal list-inside">
              <li>Inspect the embedding table (ground the mechanism)</li>
              <li>Verify gradient flow (one training step by hand)</li>
              <li>Train a real embedding (full workflow)</li>
              <li>Compare to LoRA (experience the tradeoff)</li>
            </ol>
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 11: Summary
          ================================================================ */}
      <Row>
        <Row.Content>
          <SummaryBlock
            title="Key Takeaways"
            items={[
              {
                headline:
                  'Textual inversion optimizes a single 768-dimensional embedding vector to represent a novel concept.',
                description:
                  'Create a pseudo-token, add one row to the embedding table, freeze the entire model (U-Net, VAE, CLIP encoder), and optimize only that one row using the standard DDPM training loop. 768 trainable parameters, ~4 KB output file.',
              },
              {
                headline:
                  'The U-Net cannot distinguish "real" word embeddings from the optimized one.',
                description:
                  'The optimized embedding enters the U-Net through the normal cross-attention pathway. To the U-Net, the pseudo-token is just another column in the K/V matrices. No special mechanism is needed—the existing pipeline handles it naturally.',
              },
              {
                headline:
                  'Textual inversion is the most lightweight customization but least expressive for complex styles.',
                description:
                  'One vector captures object identity and simple visual attributes. For complex spatial styles, color palettes, and rendering techniques, LoRA\u2019s thousands of cross-attention parameters are more appropriate. Choose the tool that matches the scope of the customization.',
              },
            ]}
          />
        </Row.Content>
      </Row>

      {/* Mental model echo */}
      <Row>
        <Row.Content>
          <InsightBlock title="Mental Model">
            <strong>Inventing a new word in CLIP&rsquo;s language</strong>{' '}
            &mdash;finding the right point in embedding space where, if a word
            existed there, the model would understand it as your concept.
            You are not changing the grammar. You are not changing the
            accent. You are adding one word to the dictionary.
          </InsightBlock>
        </Row.Content>
      </Row>

      {/* Module and Series completion */}
      <Row>
        <Row.Content>
          <InsightBlock title="The Complete Customization Spectrum">
            Three knobs on three different parts of the pipeline: LoRA
            changes the <strong>weights</strong>, img2img/inpainting changes
            the <strong>inference process</strong>, textual inversion changes
            the <strong>embeddings</strong>. Each is appropriate for different
            customization needs. You now understand all three.
          </InsightBlock>
        </Row.Content>
      </Row>

      {/* Module completion */}
      <Row>
        <Row.Content>
          <ModuleCompleteBlock
            module="6.5"
            title="Customization & Fine-Tuning"
            achievements={[
              'LoRA fine-tuning: modifying cross-attention projections for style and subject',
              'Img2img and inpainting: reconfiguring inference without training',
              'Textual inversion: optimizing a single embedding to teach the model a new word',
              'The complete customization spectrum: weights, inference, embeddings',
            ]}
            nextModule="7.1"
            nextTitle="Post-SD Advances"
          />
        </Row.Content>
      </Row>

      {/* Series completion message */}
      <Row>
        <Row.Content>
          <GradientCard title="Series 6 Complete" color="emerald">
            <div className="space-y-2 text-sm">
              <p>
                You have traveled from &ldquo;what is a generative
                model?&rdquo; through autoencoders, VAEs, diffusion theory,
                U-Net architecture, CLIP, cross-attention, classifier-free
                guidance, latent diffusion, samplers, and three customization
                techniques. You did not just learn to use Stable
                Diffusion&mdash;you built the intuition for every component.
              </p>
              <p>
                When you adjust a parameter, you know what changes inside the
                pipeline. When a new technique is announced, you have the
                foundation to understand why it works.
              </p>
            </div>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* References */}
      <Row>
        <Row.Content>
          <ReferencesBlock
            references={[
              {
                title:
                  'An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion',
                authors: 'Gal et al., 2022',
                url: 'https://arxiv.org/abs/2208.01618',
                note: 'The original textual inversion paper. Section 3 describes the optimization of the embedding vector. Section 4 covers the experimental results on subject and style personalization.',
              },
              {
                title:
                  'DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation',
                authors: 'Ruiz et al., 2022',
                url: 'https://arxiv.org/abs/2208.12242',
                note: 'An alternative personalization approach that fine-tunes the full model (or a subset). Useful for comparison with textual inversion\'s embedding-only approach.',
              },
              {
                title:
                  'LoRA: Low-Rank Adaptation of Large Language Models',
                authors: 'Hu et al., 2021',
                url: 'https://arxiv.org/abs/2106.09685',
                note: 'The LoRA paper. Compare LoRA\'s weight modification approach with textual inversion\'s embedding optimization for different personalization needs.',
              },
            ]}
          />
        </Row.Content>
      </Row>

      {/* Next step — Series 7 */}
      <Row>
        <Row.Content>
          <GradientCard title="What Comes Next" color="violet">
            <p className="text-sm">
              Series 7 explores what came after Stable Diffusion: ControlNet
              for structural control, SDXL for higher resolution, consistency
              models for faster generation, and flow matching as a new
              framework. Every one of these builds on what you now understand.
            </p>
          </GradientCard>
        </Row.Content>
      </Row>
    </LessonLayout>
  )
}

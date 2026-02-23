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
  TryThisBlock,
} from '@/components/lessons'
import { CodeBlock } from '@/components/common/CodeBlock'
import { MermaidDiagram } from '@/components/widgets/MermaidDiagram'
import { ExercisePanel } from '@/components/widgets/ExercisePanel'
import { SafetyStackSimulator } from '@/components/widgets/SafetyStackSimulator'
import 'katex/dist/katex.min.css'
import { InlineMath } from 'react-katex'

/**
 * Image Generation Safety -- How Production Systems Prevent Harmful Content
 *
 * Lesson 1 in Module 8.2 (Safety & Content Moderation). Series 8 Special Topics.
 * Cognitive load: BUILD (familiar tools applied to a new problem domain).
 *
 * Teaches the multi-layered safety stack for image generation:
 * - Layer 1: Prompt-level filtering (keyword blocklists, text classifiers, LLM rewriting)
 * - Layer 2: Inference-time guidance (negative prompts, Safe Latent Diffusion)
 * - Layer 3: Post-generation image classification (SD safety checker, NSFW classifiers)
 * - Layer 4: Model-level concept erasure (ESD, UCE)
 * - How real production systems compose these layers
 *
 * Core concepts:
 * - Safety stack as a design pattern: DEVELOPED
 * - CLIP-based safety classification: INTRODUCED (nearly DEVELOPED)
 * - Safe Latent Diffusion: INTRODUCED
 * - Concept erasure (ESD): INTRODUCED
 *
 * Previous: SAM 3 (module 8.1, lesson 2)
 * Next: TBD
 */

export function ImageGenerationSafetyLesson() {
  return (
    <LessonLayout>
      {/* ================================================================
          Section 1: Header
          ================================================================ */}
      <Row>
        <Row.Content>
          <LessonHeader
            title="Image Generation Safety"
            description="How production image generation systems prevent harmful content through a multi-layered defense stack&mdash;from keyword blocklists through inference-time guidance to model-level concept erasure."
            category="Safety & Content Moderation"
          />
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 2: Context + Constraints (Outline 1)
          ================================================================ */}
      <Row>
        <Row.Content>
          <ObjectiveBlock>
            Understand the multi-layered safety stack that production image generation
            systems use to prevent harmful content. Trace how each layer works using tools
            you already know&mdash;CLIP embeddings, classifier-free guidance, fine-tuning,
            and cross-attention&mdash;and explain why no single layer is sufficient.
          </ObjectiveBlock>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="What You Already Know">
            From <LessonLink slug="clip">CLIP</LessonLink>, you know shared embedding spaces
            and cosine similarity. From{' '}
            <LessonLink slug="text-conditioning-and-guidance">Text Conditioning and Guidance</LessonLink>,
            you know classifier-free guidance and negative prompts. From{' '}
            <LessonLink slug="unet-architecture">U-Net Architecture</LessonLink>, you know
            cross-attention as the gateway for text conditioning. Every technique in this
            lesson is built on those foundations.
          </TipBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <ConstraintBlock
            title="Scope for This Lesson"
            items={[
              'How each layer of the safety stack works technically (prompt filtering, inference guidance, image classification, model-level erasure)',
              'Why layered defense is necessary (each layer\'s failure modes)',
              'How real production systems compose these layers (DALL-E 3, Midjourney, Stability AI)',
              'NOT: building a complete safety system from scratch',
              'NOT: the ethics/politics of what should be filtered (this is an engineering lesson)',
              'NOT: adversarial attack methods in detail (mentioned as motivation, not taught)',
              'NOT: training data filtering, watermarking, or provenance tracking',
            ]}
          />
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 3: Hook (Outline 2)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Problem: Power Without Control"
            subtitle="You built a powerful generative model. Now what?"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              You have spent the last several series learning to build incredibly powerful
              generative models&mdash;models that can produce photorealistic images from
              any text description. But imagine you have just deployed one as an API.
              Within hours, users discover it can generate: photorealistic violence, NSFW
              content, images of real public figures in compromising scenarios, and
              copyrighted characters. Your model can do <strong>all of this</strong> because
              it learned from internet-scale data. You need to stop it. What do you build?
            </p>
            <p className="text-muted-foreground">
              The obvious answer is to put a classifier at the end: generate the image,
              check if it is unsafe, block it if so. But this &ldquo;one classifier at the
              end&rdquo; approach has critical flaws:
            </p>
            <ul className="list-disc list-inside text-muted-foreground space-y-1 ml-4">
              <li>The classifier can be bypassed (adversarial perturbations to the latents)</li>
              <li>It has false positives (blocking medical imagery, classical art)</li>
              <li>It wastes all the compute generating images that will just be thrown away</li>
              <li>It only catches what it was trained on (NSFW but not violence, or vice versa)</li>
            </ul>
            <p className="text-muted-foreground">
              The real answer is not one technique&mdash;it is a <strong>layered
              defense stack</strong> where each layer catches what the others miss.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Defense in Depth">
            Think of a castle: walls, a moat, guards, and a locked keep. Each layer can be
            breached, but an attacker must breach <strong>all</strong> of them. No single
            layer is trusted alone.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Defense stack architecture diagram */}
      <Row>
        <Row.Content>
          <MermaidDiagram chart={`
            graph LR
              P["User Prompt"] --> L1["Layer 1: Prompt Filtering"]
              L1 -->|pass| L2["Layer 2: Inference Guidance (SLD)"]
              L2 --> L3["Layer 3: Output Classifier"]
              L3 -->|pass| O["Delivered"]
              L1 -->|blocked| R1["Rejected"]
              L3 -->|blocked| R2["Rejected"]
              ME["Layer 4: Model Erasure"] -.->|modifies weights| L2

              style L1 fill:#78350f,stroke:#f59e0b,color:#fbbf24
              style L2 fill:#312e81,stroke:#8b5cf6,color:#c4b5fd
              style L3 fill:#064e3b,stroke:#10b981,color:#6ee7b7
              style ME fill:#4c1d95,stroke:#a78bfa,color:#ddd6fe
              style R1 fill:#7f1d1d,stroke:#ef4444,color:#fca5a5
              style R2 fill:#7f1d1d,stroke:#ef4444,color:#fca5a5
              style O fill:#064e3b,stroke:#10b981,color:#6ee7b7
              style P fill:#1e293b,stroke:#64748b,color:#e2e8f0
          `} />
          <p className="text-xs text-muted-foreground text-center mt-2">
            The safety stack: each subsequent section fills in one layer of this diagram.
          </p>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="Four Lines of Defense">
            <ul className="space-y-1 text-sm">
              <li>&bull; <strong>Prompt filtering</strong> catches intent before compute</li>
              <li>&bull; <strong>Inference guidance</strong> steers the generation process</li>
              <li>&bull; <strong>Output classification</strong> catches the result</li>
              <li>&bull; <strong>Model erasure</strong> removes the capability entirely</li>
            </ul>
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 4: Layer 1 -- Prompt-Level Filtering (Outline 3)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Layer 1: Prompt-Level Filtering"
            subtitle="Catch the intent before spending any compute"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The cheapest and fastest line of defense: analyze the text prompt
              before the diffusion model ever sees it. Three sub-techniques, increasing
              in sophistication:
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* 3a: Keyword Blocklists */}
      <Row>
        <Row.Content>
          <PhaseCard number={1} title="Keyword Blocklists" subtitle="The wall: obvious, easy to see over" color="amber">
            <div className="space-y-3">
              <p className="text-sm text-muted-foreground">
                A list of banned terms. If the prompt contains one, reject immediately.
                Near-zero latency, 100% recall for exact matches, trivially auditable&mdash;you
                can read the list.
              </p>
              <p className="text-sm text-muted-foreground">
                <strong>Limitations:</strong> Easily bypassed with misspellings (<code>nud3</code>),
                Unicode substitutions (Armenian characters that look like Latin), and
                circumlocutions. But they stop casual attempts with zero GPU cost.
              </p>
            </div>
          </PhaseCard>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Don&rsquo;t Dismiss Blocklists">
            Blocklists are not naive. They are <strong>fast, cheap, and predictable</strong>.
            DALL-E 3 uses blocklists alongside GPT-4 analysis. Even the smartest AI
            classifier is overkill for catching &ldquo;generate a nude woman.&rdquo;
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* 3b: Text Embedding Classifiers */}
      <Row>
        <Row.Content>
          <PhaseCard number={2} title="Text Embedding Classifiers" subtitle="The moat: harder to cross" color="blue">
            <div className="space-y-3">
              <p className="text-sm text-muted-foreground">
                Encode the prompt with a text encoder (DistilBERT, RoBERTa, or even CLIP&rsquo;s
                text encoder). Compare the embedding against known unsafe concept embeddings,
                or pass through a classification head trained on safe/unsafe prompts.
              </p>
              <p className="text-sm text-muted-foreground">
                <strong>Example: DiffGuard</strong>&mdash;a DistilBERT/RoBERTa-based classifier
                (67&ndash;125M params) fine-tuned on 250K prompts, achieving 94% F1.
                It catches semantic meaning that blocklists miss: &ldquo;nud3&rdquo; embeds
                near &ldquo;nude&rdquo; in the learned space.
              </p>
              <p className="text-sm text-muted-foreground">
                <strong>Limitations:</strong> Still text-only. Cannot catch adversarial token
                sequences that look safe as text but produce unsafe images through
                interactions in the model&rsquo;s embedding space.
              </p>
            </div>
          </PhaseCard>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Of Course">
            A prompt text classifier is doing what CLIP does for images&mdash;encoding text
            and comparing against known categories. The pattern is identical to zero-shot
            classification. You have already seen this.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* 3c: LLM-Based Prompt Analysis */}
      <Row>
        <Row.Content>
          <PhaseCard number={3} title="LLM-Based Prompt Analysis" subtitle="The smartest guard: slow but understands context" color="purple">
            <div className="space-y-3">
              <p className="text-sm text-muted-foreground">
                DALL-E 3&rsquo;s approach: GPT-4 rewrites the user&rsquo;s prompt before
                DALL-E 3 ever sees it. The LLM understands intent, context, and subtlety
                that keyword and embedding classifiers miss. It can refuse, rewrite, or
                sanitize.
              </p>
              <p className="text-sm text-muted-foreground">
                <strong>Trade-off:</strong> Highest latency, highest accuracy, most
                expensive. A user asking for &ldquo;a photo of [celebrity] committing
                a crime&rdquo; is refused by GPT-4 before DALL-E 3 even starts generating.
              </p>
            </div>
          </PhaseCard>
        </Row.Content>
      </Row>

      {/* Pseudocode for prompt classifier pipeline */}
      <Row>
        <Row.Content>
          <CodeBlock
            code={`def check_prompt(prompt: str, blocklist: set[str], classifier: TextClassifier) -> str:
    # Layer 1a: Keyword blocklist (near-zero latency)
    for word in prompt.lower().split():
        if word in blocklist:
            return "BLOCKED"

    # Layer 1b: Embedding classifier (semantic check)
    embedding = classifier.encode(prompt)       # e.g., DistilBERT
    score = classifier.predict(embedding)       # P(unsafe | prompt)
    if score > 0.85:
        return "BLOCKED"

    # Layer 1c: LLM analysis (expensive, context-aware)
    # Only used in production systems like DALL-E 3
    # llm_decision = gpt4_analyze(prompt)

    return "PASS"`}
            language="python"
            filename="prompt_classifier.py"
          />
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Defense Economics">
            Notice the ordering: cheapest first, most expensive last. Each layer only
            runs if the previous one passed. This minimizes GPU cost for obviously
            unsafe prompts.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 5: Check 1 -- Predict-and-Verify (Outline 4)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Check: Which Layer Catches Which?"
            subtitle="Predict before reading the answer"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Three prompts enter the prompt-level filtering stack. For each,
              predict which sub-layer catches it (or if it passes through):
            </p>
            <div className="space-y-3">
              <GradientCard title="Prompt 1: &ldquo;Generate a nude woman&rdquo;" color="amber">
                <details className="group">
                  <summary className="cursor-pointer text-sm font-medium text-primary hover:text-primary/80 transition-colors">
                    Show answer
                  </summary>
                  <div className="mt-3 text-sm border-t border-primary/20 pt-3">
                    <p className="text-muted-foreground">
                      Keyword blocklist catches this immediately.
                      &ldquo;Nude&rdquo; is a banned keyword. No embedding classification needed.
                    </p>
                  </div>
                </details>
              </GradientCard>
              <GradientCard title="Prompt 2: &ldquo;Generate a n.u.d.e woman&rdquo;" color="blue">
                <details className="group">
                  <summary className="cursor-pointer text-sm font-medium text-primary hover:text-primary/80 transition-colors">
                    Show answer
                  </summary>
                  <div className="mt-3 text-sm border-t border-primary/20 pt-3">
                    <p className="text-muted-foreground">
                      Keyword blocklist misses it (&ldquo;n.u.d.e&rdquo;
                      is not &ldquo;nude&rdquo;). But the embedding classifier catches it because
                      &ldquo;n.u.d.e&rdquo; embeds close to &ldquo;nude&rdquo; in the learned
                      semantic space (0.87 cosine similarity).
                    </p>
                  </div>
                </details>
              </GradientCard>
              <GradientCard title="Prompt 3: &ldquo;A renaissance painting featuring classical figure studies&rdquo;" color="purple">
                <details className="group">
                  <summary className="cursor-pointer text-sm font-medium text-primary hover:text-primary/80 transition-colors">
                    Show answer
                  </summary>
                  <div className="mt-3 text-sm border-t border-primary/20 pt-3">
                    <p className="text-muted-foreground">
                      Ambiguous. No banned keywords. The embedding
                      classifier sees artistic context and may pass it. An LLM analyzer (DALL-E 3&rsquo;s
                      GPT-4 layer) would assess whether the artistic framing is genuine or a
                      circumlocution. This is where nuanced understanding matters.
                    </p>
                  </div>
                </details>
              </GradientCard>
            </div>
          </div>
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 6: Layer 2 -- Inference-Level Safety (Outline 5)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Layer 2: Inference-Time Safety"
            subtitle="Steer the generation process itself"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Even if a prompt passes all text-level filters, the generation process
              itself can be steered toward safety. You already know the tool for
              this&mdash;classifier-free guidance.
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* Negative prompts as soft safety */}
      <Row>
        <Row.Content>
          <GradientCard title="Negative Prompts as Soft Safety" color="violet">
            <div className="space-y-3">
              <p className="text-sm text-muted-foreground">
                You already use negative prompts to steer away from unwanted content.
                Using &ldquo;nudity, violence, gore, nsfw&rdquo; as a negative prompt
                modifies the unconditional prediction in CFG, steering generation away
                from unsafe content. Simple, no extra models needed.
              </p>
              <p className="text-sm text-muted-foreground">
                <strong>But probabilistic:</strong> it turns down the volume, it does not
                mute the channel. Negative prompts reduce probability but do not guarantee
                safety. An adversarial prompt can still produce harmful content despite
                negative prompts.
              </p>
            </div>
          </GradientCard>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="Connection">
            From <LessonLink slug="text-conditioning-and-guidance">Text Conditioning and Guidance</LessonLink>:
            negative prompts replace the unconditional prediction with a &ldquo;negative&rdquo;
            prediction, steering away from unwanted content. Safety-guided diffusion is
            the same idea, formalized and strengthened.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* Safe Latent Diffusion */}
      <Row>
        <Row.Content>
          <GradientCard title="Safe Latent Diffusion (SLD)" color="blue">
            <div className="space-y-3">
              <p className="text-sm text-muted-foreground">
                SLD formalizes and strengthens negative-prompt safety. It adds a dedicated
                safety guidance term to the denoising step. Remember the CFG formula:
              </p>
              <div className="py-3 px-4 bg-muted/50 rounded-lg">
                <InlineMath math="\epsilon_\theta = \epsilon_{\text{uncond}} + s \cdot (\epsilon_{\text{cond}} - \epsilon_{\text{uncond}})" />
              </div>
              <p className="text-sm text-muted-foreground">
                SLD adds one more term pushing in the opposite direction of a safety concept:
              </p>
              <div className="py-3 px-4 bg-muted/50 rounded-lg">
                <InlineMath math="\epsilon_\theta = \epsilon_{\text{uncond}} + s \cdot (\epsilon_{\text{cond}} - \epsilon_{\text{uncond}}) - s_{\text{sld}} \cdot (\epsilon_{\text{safety}} - \epsilon_{\text{uncond}})" />
              </div>
              <p className="text-sm text-muted-foreground">
                Where <InlineMath math="\epsilon_{\text{safety}}" /> is the noise prediction
                conditioned on a safety concept text (e.g., &ldquo;nudity, violence,
                harm&rdquo;). The third term actively pushes away from unsafe content at
                each denoising step.
              </p>
              <p className="text-sm text-muted-foreground">
                SLD also has <strong>warmup steps</strong> (kicks in after step ~10 to avoid
                disrupting early structure), <strong>momentum</strong> (accumulated safety
                signal), and configurable strength (WEAK/MEDIUM/STRONG/MAX).
              </p>
            </div>
          </GradientCard>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Of Course">
            Same geometric idea as CFG&mdash;vector arithmetic in noise-prediction space.
            CFG extrapolates toward the prompt. SLD simultaneously extrapolates <strong>away</strong> from
            a safety concept. One more term, same framework.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* SLD pseudocode */}
      <Row>
        <Row.Content>
          <CodeBlock
            code={`def denoise_step_with_sld(
    model, x_t, t, prompt_embed, safety_embed,
    guidance_scale=7.5, sld_scale=2000.0, warmup_steps=10
):
    # Standard CFG predictions
    eps_uncond = model(x_t, t, null_embed)
    eps_cond   = model(x_t, t, prompt_embed)
    eps_safety = model(x_t, t, safety_embed)

    # Standard CFG direction
    noise_pred = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

    # SLD safety term (active after warmup)
    # t counts DOWN from T (pure noise) to 0 (clean image).
    # steps_completed counts UP from 0 (start) to T (done).
    steps_completed = T - t
    if steps_completed > warmup_steps:
        safety_direction = eps_safety - eps_uncond
        noise_pred = noise_pred - sld_scale * safety_direction

    return noise_pred`}
            language="python"
            filename="safe_latent_diffusion.py"
          />
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Why Warmup?">
            The first ~10 denoising steps establish global structure (composition, layout).
            Applying the safety term too early disrupts image coherence. SLD waits until
            details emerge before steering.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 7: Interactive Widget (Outline 6)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Explore: The Safety Stack in Action"
            subtitle="Toggle layers on and off to see what leaks through"
          />
          <p className="text-muted-foreground mb-4">
            This simulator shows example prompts flowing through the safety stack. Toggle
            individual layers on and off to see which prompts get caught, which slip through,
            and why no single layer is enough. Pay attention to false positives too&mdash;the
            &ldquo;museum statue&rdquo; prompt is legitimate artistic content that the keyword
            blocklist blocks. Every safety system faces this precision-recall tradeoff: aggressive
            filtering catches more unsafe content but also blocks more legitimate content. A
            medical imaging platform needs high recall for anatomy; a children&rsquo;s app
            can tolerate more false positives. The right threshold depends on the application.
          </p>
          <ExercisePanel title="Safety Stack Simulator" subtitle="Toggle layers and watch what gets through">
            <SafetyStackSimulator />
          </ExercisePanel>
        </Row.Content>
        <Row.Aside>
          <TryThisBlock title="Experiments to Try">
            <ul className="space-y-2 text-sm">
              <li>&bull; Turn off the blocklist. Which prompts still get caught?</li>
              <li>&bull; Leave only the output classifier on. Does anything leak?</li>
              <li>&bull; Turn off everything except the text classifier. What slips through?</li>
              <li>&bull; Notice: the &ldquo;violence&rdquo; prompt leaks when only the output classifier is active. Why?</li>
              <li>&bull; Notice: the &ldquo;museum statue&rdquo; gets <strong>blocked</strong> by the blocklist&mdash;a <strong>false positive</strong>. What is the tradeoff?</li>
            </ul>
          </TryThisBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 8: Layer 3 -- Post-Generation Image Classification (Outline 7)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Layer 3: Post-Generation Image Classification"
            subtitle="The last line of defense before delivery"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Regardless of what the prompt said or how inference was guided, the generated
              image itself can be classified. This is the most important concrete
              example because the Stable Diffusion safety checker is fully open-source.
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* SD Safety Checker architecture */}
      <Row>
        <Row.Content>
          <GradientCard title="The Stable Diffusion Safety Checker" color="emerald">
            <div className="space-y-3">
              <p className="text-sm text-muted-foreground">
                <strong>Architecture:</strong> Uses CLIP ViT-L/14 as the image encoder.
                The generated image is encoded into CLIP embedding space (768-dim vector).
                This embedding is compared via cosine similarity against <strong>17 fixed
                concept embeddings</strong>&mdash;the concepts are obfuscated (only their
                embedding vectors are public, not the text descriptions). Each concept
                has a threshold. If any cosine similarity exceeds its threshold, the image
                is flagged and replaced with a black rectangle.
              </p>
              <p className="text-sm text-muted-foreground">
                Two tiers: <code>special_care_embeds</code> (3 embeddings, lower
                thresholds&mdash;more sensitive) and regular <code>concept_embeds</code> (14
                embeddings). An <code>adjustment</code> parameter globally shifts all
                thresholds.
              </p>
            </div>
          </GradientCard>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Of Course">
            This is literally zero-shot CLIP classification. You have already done this&mdash;comparing
            an embedding against reference embeddings and thresholding on cosine similarity.
            The safety checker is the simplest possible application of CLIP.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Traced computation */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground font-medium">Traced computation:</p>
            <ComparisonRow
              left={{
                title: 'Unsafe Image',
                color: 'rose',
                items: [
                  'Image [3, 512, 512]',
                  '→ CLIP ViT-L/14 encoder',
                  '→ embedding [768]',
                  '→ cosine_similarity(emb, concept₇) = 0.83',
                  '→ threshold₇ = 0.78',
                  '→ 0.83 > 0.78 → FLAGGED',
                ],
              }}
              right={{
                title: 'Safe Image',
                color: 'emerald',
                items: [
                  'Image [3, 512, 512]',
                  '→ CLIP ViT-L/14 encoder',
                  '→ embedding [768]',
                  '→ cosine_similarity(emb, concept₇) = 0.31',
                  '→ threshold₇ = 0.78',
                  '→ 0.31 < 0.78 → PASS',
                ],
              }}
            />
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="Same Math, New Use">
            From <LessonLink slug="clip">CLIP</LessonLink>: cosine similarity measures
            semantic proximity in the shared embedding space. The safety checker just asks:
            &ldquo;Is this image embedding close to any of our 17 unsafe concept embeddings?&rdquo;
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* Safety checker pseudocode */}
      <Row>
        <Row.Content>
          <CodeBlock
            code={`def safety_check(image: Tensor, clip_model, concept_embeds, thresholds) -> bool:
    """Returns True if the image is FLAGGED (unsafe)."""

    # Step 1: Encode image into CLIP space
    image_embed = clip_model.encode_image(image)         # [768]
    image_embed = image_embed / image_embed.norm()       # L2 normalize

    # Step 2: Cosine similarity against all 17 concepts
    similarities = image_embed @ concept_embeds.T        # [17]

    # Step 3: Threshold check
    for i in range(len(similarities)):
        if similarities[i] > thresholds[i]:
            return True    # FLAGGED

    return False           # PASS`}
            language="python"
            filename="sd_safety_checker.py"
          />
        </Row.Content>
      </Row>

      {/* Dedicated classifiers + limitations */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Beyond CLIP-based checking, some systems use <strong>purpose-trained
              image classifiers</strong> (ResNet/InceptionV3-based, like NudeNet). These
              are standard image classification models trained on labeled safe/unsafe
              datasets. Higher accuracy for specific categories (nudity detection) but
              require labeled training data and do not generalize to new harm categories
              without retraining.
            </p>
          </div>
        </Row.Content>
      </Row>

      <Row>
        <Row.Content>
          <WarningBlock title="Known Limitations of Output Classifiers">
            <ul className="space-y-1 text-sm text-muted-foreground">
              <li>&bull; <strong>Narrow scope:</strong> SD safety checker only checks sexual content, not violence, hate symbols, or deepfakes</li>
              <li>&bull; <strong>Adversarial vulnerability:</strong> Small perturbations to latents can fool the classifier</li>
              <li>&bull; <strong>Latency cost:</strong> CLIP encoding is not free (adds inference time)</li>
              <li>&bull; <strong>False positives:</strong> Medical imagery, classical art, swimwear can trigger flags</li>
            </ul>
          </WarningBlock>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Not Just NSFW">
            Production systems filter for violence, gore, self-harm, hate symbols, real
            public figures, copyrighted characters, drug manufacturing, weapons, and child
            safety. OpenAI&rsquo;s categories include at least 10 distinct harm types. The
            SD safety checker&rsquo;s sexual-only scope is a known limitation, not the
            intended design.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 9: Check 2 -- Explain-It-Back (Outline 8)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Check: Why Obfuscate the Concepts?"
            subtitle="Think about it before reading the answer"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The SD safety checker uses 17 concept embeddings but their text descriptions
              are hidden. Only the embedding vectors are public. Why would you obfuscate
              the concepts? What would happen if the concept texts were public?
            </p>
            <GradientCard title="The Reasoning" color="violet">
              <details className="group">
                <summary className="cursor-pointer text-sm font-medium text-primary hover:text-primary/80 transition-colors">
                  Show answer
                </summary>
                <div className="mt-3 text-sm border-t border-primary/20 pt-3">
                  <p className="text-muted-foreground">
                    Publishing the exact concepts would make it trivial to craft adversarial
                    prompts that land just below the threshold for each concept, or to
                    specifically target the gaps between concepts. Obfuscation is a form of
                    security through obscurity&mdash;imperfect, but it raises the cost of
                    targeted attacks. You cannot optimize against a threshold you do not
                    know precisely.
                  </p>
                </div>
              </details>
            </GradientCard>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Arms Race Thinking">
            Every published detail of a safety system becomes a target for adversarial
            optimization. This is why safety is an ongoing engineering effort, not a
            one-time deployment.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 10: Layer 4 -- Model-Level Concept Erasure (Outline 9)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Layer 4: Model-Level Concept Erasure"
            subtitle="Remove the capability from the weights entirely"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The previous layers all operate around the model&mdash;filtering inputs or
              classifying outputs. Concept erasure takes a different approach: modify the
              model&rsquo;s weights so that it literally <strong>cannot generate</strong> certain
              content. Remove the concept from inside the castle rather than guarding
              the gate.
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* ESD */}
      <Row>
        <Row.Content>
          <GradientCard title="Erased Stable Diffusion (ESD)" color="purple">
            <div className="space-y-3">
              <p className="text-sm text-muted-foreground">
                <strong>Core mechanism:</strong> Use classifier-free guidance logic at
                TRAINING time. The frozen pretrained model provides both conditional
                (concept-present) and unconditional noise predictions. The edited model
                is trained to produce, when given the concept prompt, a prediction that
                would guide AWAY from the concept. Essentially: &ldquo;when asked to
                generate [concept], generate the opposite of what you would have
                generated.&rdquo;
              </p>
              <p className="text-sm text-muted-foreground">
                <strong>Two variants:</strong>
              </p>
              <ul className="list-disc list-inside text-sm text-muted-foreground space-y-1 ml-2">
                <li>
                  <strong>ESD-x</strong> (cross-attention only): Erases text-to-concept
                  associations. Good for style removal (&ldquo;Van Gogh&rdquo;) because the
                  style IS a text-conditioned concept. Targeted, minimal collateral damage.
                </li>
                <li>
                  <strong>ESD-u</strong> (all non-cross-attention layers): Erases concepts
                  globally, even without explicit text triggers. Better for NSFW removal
                  because unsafe content can emerge from certain style descriptions without
                  explicit prompting. Broader erasure, more collateral damage.
                </li>
              </ul>
            </div>
          </GradientCard>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Of Course">
            ESD-x targets cross-attention because cross-attention is where text conditioning
            enters the U-Net. The model learns &ldquo;Van Gogh = swirly brushstrokes&rdquo;
            through cross-attention. Erase those weights, erase that association.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ESD example: style erasure */}
      <Row>
        <Row.Content>
          <ComparisonRow
            left={{
              title: 'Standard SD: "A painting in the style of Van Gogh"',
              color: 'amber',
              items: [
                'Cross-attention maps text "Van Gogh" to learned style features',
                'Output: swirling brushstrokes, vibrant yellows and blues',
                'The model has a strong text → style association',
              ],
            }}
            right={{
              title: 'ESD-x Model (Van Gogh erased)',
              color: 'blue',
              items: [
                'Cross-attention weights for "Van Gogh" have been modified',
                'Output: generic painting with no Van Gogh characteristics',
                'Same prompt, different model weights → different result',
              ],
            }}
          />
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="Fine-Tuning in Reverse">
            ESD is fine-tuning with a reversed objective. Instead of teaching the model
            a new concept, you teach it to un-learn one. Same optimizer, same weight
            updates, opposite direction.
            <br /><br />
            <strong>Important:</strong> This is not just negating the loss. The frozen model
            provides reference predictions; the edited model trains to produce the
            unconditional prediction when given the concept-conditioned prompt.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* ESD pseudocode */}
      <Row>
        <Row.Content>
          <CodeBlock
            code={`def esd_training_step(
    frozen_model, edited_model, concept_text: str, optimizer
):
    """One ESD training step: teach the model to 'forget' a concept."""
    # Sample random noise and timestep
    x_t = sample_noisy_latent()
    t   = sample_timestep()

    # Frozen model provides reference predictions
    with torch.no_grad():
        eps_cond   = frozen_model(x_t, t, encode(concept_text))  # "what Van Gogh looks like"
        eps_uncond = frozen_model(x_t, t, null_embed)             # "what anything looks like"

    # Target: guide AWAY from the concept
    # (same formula as negative guidance in CFG)
    eps_target = eps_uncond - guidance_scale * (eps_cond - eps_uncond)

    # Edited model learns to produce this anti-concept prediction
    eps_edited = edited_model(x_t, t, encode(concept_text))
    loss = mse_loss(eps_edited, eps_target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()`}
            language="python"
            filename="esd_training.py"
          />
        </Row.Content>
      </Row>

      {/* UCE */}
      <Row>
        <Row.Content>
          <GradientCard title="Unified Concept Editing (UCE)" color="cyan">
            <div className="space-y-3">
              <p className="text-sm text-muted-foreground">
                A closed-form solution (no gradient-based training) that edits
                cross-attention weights directly. Can debias, erase, and moderate multiple
                concepts simultaneously. Faster than ESD (no fine-tuning loop), but limited
                to cross-attention modifications.
              </p>
              <p className="text-sm text-muted-foreground">
                UCE computes the exact weight update needed to project out a concept from
                the key/value matrices of cross-attention layers. Think of it as surgically
                removing a direction from the weight space rather than iteratively training
                it out.
              </p>
            </div>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* Tradeoffs of model-level erasure */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Why Not Just Erase Everything?"
            subtitle="The limits of model-level erasure"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              If ESD can erase concepts from model weights, why not just erase all
              harmful concepts and skip the other layers entirely? Four critical problems:
            </p>
          </div>
          <div className="grid gap-3 md:grid-cols-2 mt-4">
            <GradientCard title="Collateral Damage" color="rose">
              <p className="text-sm text-muted-foreground">
                Erasing &ldquo;nudity&rdquo; also degrades images of people in swimwear,
                medical imagery, and classical art. The concept boundaries in embedding
                space are not surgically precise.
              </p>
            </GradientCard>
            <GradientCard title="Reversibility" color="rose">
              <p className="text-sm text-muted-foreground">
                Adversarial fine-tuning can recover erased concepts in ~1000 steps. If
                the model weights are public (open-source), anyone can undo the erasure.
              </p>
            </GradientCard>
            <GradientCard title="Coverage Gaps" color="amber">
              <p className="text-sm text-muted-foreground">
                You must anticipate every harmful concept in advance. New concepts
                (new slang, new cultural context) require re-erasure. You cannot erase
                what you have not foreseen.
              </p>
            </GradientCard>
            <GradientCard title="Irreversibility of Deployment" color="amber">
              <p className="text-sm text-muted-foreground">
                Once you ship an erased model, you cannot add the concept back for
                legitimate uses (medical imagery, art restoration, research).
              </p>
            </GradientCard>
          </div>
          <div className="mt-4">
            <p className="text-muted-foreground mb-3">
              <strong>Collateral damage in practice:</strong> here is what happens when
              ESD-u erases &ldquo;nudity&rdquo; from the model&mdash;legitimate use
              cases break:
            </p>
            <ComparisonRow
              left={{
                title: 'Before ESD-u: "A medical textbook diagram of human anatomy"',
                color: 'emerald',
                items: [
                  'Clear, detailed anatomical illustration',
                  'Appropriate for educational context',
                  'The model treats this as medical content',
                ],
              }}
              right={{
                title: 'After ESD-u (nudity erased): same prompt',
                color: 'rose',
                items: [
                  'Distorted, unusable output',
                  'Anatomical features degraded or missing',
                  'The erasure cannot distinguish medical from exploitative',
                ],
              }}
            />
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Not a Silver Bullet">
            Model-level erasure is the most technically impressive technique in the safety
            stack, but it is also the most fragile. It complements the other layers; it
            does not replace them.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 11: How Real Systems Compose the Stack (Outline 10)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="How Real Systems Compose the Stack"
            subtitle="DALL-E 3, Midjourney, and Stability AI"
          />
        </Row.Content>
      </Row>

      <Row>
        <Row.Content>
          <div className="space-y-3">
            <GradientCard title="DALL-E 3 (OpenAI)" color="blue">
              <ul className="space-y-1 text-sm text-muted-foreground">
                <li>&bull; Keyword blocklists for obvious terms</li>
                <li>&bull; GPT-4 prompt rewriting/refusal (the most sophisticated layer)</li>
                <li>&bull; Bespoke CLIP-based output classifier</li>
                <li>&bull; Training data filtering (removed problematic content before training)</li>
                <li>&bull; The GPT-4 layer understands context, intent, and subtlety that no classifier can match</li>
              </ul>
            </GradientCard>
            <GradientCard title="Midjourney" color="purple">
              <ul className="space-y-1 text-sm text-muted-foreground">
                <li>&bull; AI-driven content moderation evaluating prompts holistically (not just keywords)</li>
                <li>&bull; Post-generation image analysis</li>
                <li>&bull; Community reporting system</li>
                <li>&bull; Dynamic, regularly updated filter</li>
                <li>&bull; No public architecture details (closed-source advantage)</li>
              </ul>
            </GradientCard>
            <GradientCard title="Stability AI (Stable Diffusion)" color="emerald">
              <ul className="space-y-1 text-sm text-muted-foreground">
                <li>&bull; Open-source safety checker (CLIP-based, 17 concepts)</li>
                <li>&bull; SD 2.0 added training data filtering (removed NSFW from training set)</li>
                <li>&bull; Community-built tools (NudeNet, custom classifiers)</li>
                <li>&bull; Open-source means users CAN disable safety&mdash;the safety stack is advisory, not enforced</li>
              </ul>
            </GradientCard>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Closed vs Open">
            Closed-source systems (DALL-E, Midjourney) can enforce all layers because they
            control the full pipeline. Open-source systems (SD) can only include layers as
            defaults that users can disable. This is a fundamental architectural constraint,
            not just a policy choice.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 11b: The Arms Race (Misconception #5)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Arms Race: Every Layer Has Been Bypassed"
            subtitle="Safety is an ongoing challenge, not a solved problem"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The safety stack sounds robust: four layers, each catching different failure
              modes. But every single layer has been bypassed in practice. If you walk
              away from this lesson thinking a well-designed stack is bulletproof,
              you are holding a dangerous misconception.
            </p>
          </div>
          <ul className="list-disc list-inside text-muted-foreground space-y-2 ml-4 mt-4">
            <li><strong>Blocklists</strong> &mdash; bypassed with misspellings and Unicode substitutions (as we saw in Layer 1), requiring zero technical skill</li>
            <li><strong>Text classifiers</strong> &mdash; SurrogatePrompt (Tsai et al., 2024) achieves <strong>70%+ bypass rates</strong> by optimizing prompts that embed far from unsafe clusters but still produce unsafe images through token interactions in the diffusion model</li>
            <li><strong>Output classifiers</strong> &mdash; adding carefully crafted noise to latents before decoding shifts the CLIP embedding below the safety threshold while the image remains visually unsafe</li>
            <li><strong>Concept erasure</strong> &mdash; erased concepts can be recovered via fine-tuning (as we saw in Layer 4), and open-source weights make this accessible to anyone</li>
          </ul>
          <div className="mt-4">
            <p className="text-muted-foreground">
              The lesson is not that safety is futile&mdash;it is that safety is an
              <strong>ongoing engineering effort</strong>. Every published detail of a
              safety system becomes a target for adversarial optimization. Defense in depth
              raises the cost of attacks, but it does not eliminate them. The most realistic
              mental model is an arms race: attackers find bypasses, defenders patch them,
              attackers find new bypasses, and the cycle continues.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Not Bulletproof">
            A well-designed safety stack raises the cost of generating harmful content
            from &ldquo;trivial&rdquo; to &ldquo;requires specialized knowledge.&rdquo; It
            does not make it impossible. Every production system operates under this
            assumption.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 12: Check 3 -- Transfer Question (Outline 11)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Check: Design a Safety Stack"
            subtitle="Apply what you have learned to a new scenario"
          />
          <div className="space-y-4">
            <GradientCard title="Scenario" color="cyan">
              <p className="text-sm text-muted-foreground">
                A startup is building a <strong>children&rsquo;s illustration
                generator</strong>. They want to use an open-source diffusion model.
                Design their safety stack: which layers would you include and why? What
                is the biggest risk they should worry about?
              </p>
            </GradientCard>
            <GradientCard title="Key Reasoning" color="violet">
              <details className="group">
                <summary className="cursor-pointer text-sm font-medium text-primary hover:text-primary/80 transition-colors">
                  Show answer
                </summary>
                <div className="mt-3 border-t border-primary/20 pt-3">
                  <div className="space-y-2 text-sm text-muted-foreground">
                    <p>
                      <strong>All four layers are needed</strong>, plus extra-conservative
                      thresholds on the output classifier. Consider ESD-u for the broadest concept
                      erasure despite collateral damage&mdash;children&rsquo;s illustrations do not
                      need anatomical accuracy.
                    </p>
                    <p>
                      <strong>Biggest risk:</strong> Subtly inappropriate content that passes all
                      automated layers but is harmful in context (e.g., inappropriately sexualized
                      poses that no single classifier catches, or cultural stereotypes). A human
                      review layer may be necessary for a children&rsquo;s product.
                    </p>
                    <p>
                      <strong>Open-source consideration:</strong> Since users interact through
                      their API (not the raw model), they CAN enforce all layers server-side&mdash;the
                      open-source model is inside their controlled pipeline.
                    </p>
                  </div>
                </div>
              </details>
            </GradientCard>
          </div>
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 13: Summary (Outline 12)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SummaryBlock
            items={[
              {
                headline: 'Image generation safety requires a multi-layered defense stack.',
                description: 'No single technique is sufficient. Each layer catches what the others miss, like a castle with walls, a moat, guards, and a locked keep.',
              },
              {
                headline: 'Prompt filtering catches intent before spending compute.',
                description: 'Keyword blocklists (fast, cheap), text embedding classifiers (semantic), and LLM-based analysis (contextual) form the first line of defense.',
              },
              {
                headline: 'Inference-time guidance steers the generation process itself.',
                description: 'Safe Latent Diffusion adds a safety term to the CFG formula you already know, pushing away from unsafe content at each denoising step.',
              },
              {
                headline: 'Output classification catches results regardless of prompt intent.',
                description: 'The SD safety checker is literally zero-shot CLIP classification: encode the image, compute cosine similarity against 17 concept embeddings, threshold.',
              },
              {
                headline: 'Model-level concept erasure removes capabilities from the weights.',
                description: 'ESD uses CFG-style guidance at training time to teach the model to "forget" concepts. Powerful but has collateral damage and can be reversed.',
              },
              {
                headline: 'Every technique is built on tools you already understand.',
                description: 'CLIP embeddings power the safety checker, CFG powers SLD, fine-tuning powers ESD, cross-attention explains why ESD-x targets specific layers.',
              },
              {
                headline: 'Safety is an ongoing engineering challenge, not a one-time deployment.',
                description: 'Every layer has been bypassed. Every published detail becomes a target for adversarial optimization. Defense in depth is the only viable strategy.',
              },
            ]}
          />
        </Row.Content>
      </Row>

      {/* ================================================================
          References
          ================================================================ */}
      <Row>
        <Row.Content>
          <ReferencesBlock
            references={[
              {
                title: 'Safe Latent Diffusion: Mitigating Inappropriate Degeneration in Diffusion Models',
                authors: 'Schramowski et al., 2023',
                url: 'https://arxiv.org/abs/2211.05105',
                note: 'Introduces the SLD safety guidance term. Section 3 covers the modified denoising formula.',
              },
              {
                title: 'Erased Stable Diffusion: Forget What You Have Learned',
                authors: 'Gandikota et al., 2023',
                url: 'https://arxiv.org/abs/2303.07345',
                note: 'The ESD paper. Section 3 explains ESD-x vs ESD-u. Figure 3 shows style erasure results.',
              },
              {
                title: 'Unified Concept Editing in Diffusion Models',
                authors: 'Gandikota et al., 2024',
                url: 'https://arxiv.org/abs/2308.14761',
                note: 'UCE closed-form concept editing. Section 3 derives the projection-based weight update.',
              },
              {
                title: 'DALL-E 3 System Card',
                authors: 'OpenAI, 2023',
                url: 'https://cdn.openai.com/papers/DALL_E_3_System_Card.pdf',
                note: 'Describes the multi-layered safety approach including GPT-4 prompt analysis.',
              },
            ]}
          />
        </Row.Content>
      </Row>

      {/* ================================================================
          Next Step (Outline 13)
          ================================================================ */}
      <Row>
        <Row.Content>
          <NextStepBlock
            href="/app"
            title="Lesson Complete"
            description="This lesson focused on the engineering of safety systems. Adjacent topics worth exploring: training data curation and filtering, watermarking and provenance tracking, adversarial robustness, and constitutional approaches to generative model alignment."
          />
        </Row.Content>
      </Row>
    </LessonLayout>
  )
}

# Lesson: From Classification to Generation

**Module:** 6.1 — Generative Foundations
**Position:** Lesson 1 of 4
**Slug:** `from-classification-to-generation`
**Type:** Conceptual / interactive (no notebook)
**Cognitive load:** BUILD

---

## Phase 1: Orient — Student State

The student has completed Series 1-3 and Modules 4.1-4.2. Every model they have built or studied has been discriminative. They have strong foundations in neural network mechanics and a deep understanding of the transformer architecture, but have never encountered the generative framing.

### Relevant Concepts with Depths

| Concept | Depth | Source | Relevance |
|---------|-------|--------|-----------|
| ML as function approximation | INTRODUCED | 1.1 (what-is-ml) | Core frame being extended. Student thinks of ML as "learn f: input -> label." This lesson reframes to "learn the distribution that input comes from." |
| MSE loss function | DEVELOPED | 1.1 (loss-landscape) | Familiar loss that will reappear as reconstruction loss in Lesson 2. Establishes that loss = measure of wrongness. |
| Training loop (forward -> loss -> backward -> update) | DEVELOPED | 1.1, 2.1 | Universal pattern. Generation still uses this loop — the loss function changes, not the training procedure. |
| Softmax function | INTRODUCED | 2.2 (real-data) | Converts logits to probabilities. Used in classification output. Student has seen it produce a probability distribution over classes. |
| Cross-entropy loss | INTRODUCED | 2.2 (real-data) | Classification loss. The discriminative objective. Contrasts with reconstruction loss and likelihood-based objectives. |
| CNN architecture (conv-pool-fc) | APPLIED | 3.1 (mnist-cnn-project) | Student has built CNNs for classification. The same conv layers will appear in autoencoders (Lesson 2). Important for "same building blocks, different objective." |
| Transfer learning | DEVELOPED | 3.2 (transfer-learning) | Demonstrates that learned features are reusable. Latent representations in autoencoders extend this idea. |
| Feature hierarchy in CNNs | DEVELOPED | 3.1-3.3 | "Edges -> textures -> parts -> objects." The encoder in an autoencoder learns a similar hierarchy. |
| Language modeling as next-token prediction | DEVELOPED | 4.1 (what-is-a-language-model) | The student's ONE encounter with something that looks generative: predicting (and sampling) next tokens. Critical bridge concept. |
| Temperature as distribution reshaping | INTRODUCED | 4.1 (what-is-a-language-model) | Student has interacted with a temperature slider and seen how it changes the shape of a probability distribution. Directly relevant to understanding sampling. |
| Autoregressive generation (sample, append, repeat) | DEVELOPED | 4.1 (what-is-a-language-model) | The feedback loop for text generation. Student has seen generation as repeated sampling. This lesson generalizes the idea. |

### Mental Models Already Established

- **"ML is function approximation"** — The defining frame. This lesson extends it to "ML can also approximate the data distribution itself."
- **"Training loop = forward -> loss -> backward -> update"** — The student knows this is universal. Generation doesn't change the loop.
- **"Architecture encodes assumptions about data"** — From CNNs. Different architectures for different problems. The encoder-decoder is another architectural choice.
- **"Temperature changes sampling, not knowledge"** — From language models. The student has manipulated a distribution and seen how it changes what gets sampled.
- **"A language model approximates P(next token | context)"** — The closest thing to a generative model the student has seen.

### What Was Explicitly NOT Covered

- Generative models of any kind (GANs, VAEs, diffusion, autoregressive image models)
- Probability density functions over continuous data
- Reconstruction loss
- Encoder-decoder architecture (in the generative sense; the transformer has an encoder-decoder variant but student only studied decoder-only)
- Latent representations (beyond what feature hierarchies implicitly provide)
- Any image generation

### Readiness Assessment

The student is well-prepared. They have strong discriminative foundations and, crucially, have already seen sampling from a learned distribution (language model temperature slider, autoregressive generation). The conceptual leap is not from zero — it's from "I've sampled tokens from a distribution" to "what if we could sample images from a distribution?" The language model experience is the bridge.

---

## Phase 2: Analyze

### Target Concept

This lesson teaches the student to articulate the difference between discriminative models (learning decision boundaries between categories) and generative models (learning the distribution of the data itself), and to understand that generation means sampling from a learned probability distribution.

### Prerequisites Table

| Concept | Required Depth | Actual Depth | Source Lesson | Status | Reasoning |
|---------|---------------|-------------|---------------|--------|-----------|
| ML as function approximation | INTRODUCED | INTRODUCED | 1.1 | OK | Needed as the frame being extended, not built upon at higher depth. Student recognizes ML as "approximate a function from examples." |
| Softmax as probability distribution | INTRODUCED | INTRODUCED | 2.2 | OK | Student has seen softmax produce probabilities that sum to 1. Need this intuition for "distribution over data." |
| Classification (input -> label) | DEVELOPED | APPLIED | 3.1 (CNN project) | OK | The discriminative paradigm. Student has practiced classification end-to-end. |
| Next-token prediction as learned distribution | INTRODUCED | DEVELOPED | 4.1 | OK | Exceeds requirement. Student understands P(next token | context) and has seen how sampling from it produces text. |
| Temperature / distribution reshaping | INTRODUCED | INTRODUCED | 4.1 | OK | Student has interacted with temperature slider. Understands that the same distribution can be sampled differently. |
| Probability distribution (basic concept) | INTRODUCED | INTRODUCED | 2.2 (softmax), 4.1 (temperature) | OK | Student has seen discrete distributions over classes and tokens. This lesson extends to distributions over data. |

All prerequisites are OK. No gaps to resolve.

### Misconceptions Table

| Misconception | Why They'd Think This | Negative Example That Disproves It | Where to Address |
|---------------|----------------------|-----------------------------------|-----------------|
| "Generative models memorize training images and spit them back" | The outputs look like training data, and the student has no framework for how novel images could emerge from a model. Nearest mental model: lookup table. | Show two images side by side: a training image and a generated image that is clearly similar in style but depicts something never in the dataset (e.g., a face with a specific combination of features that doesn't match any training example). If the model memorized, it could only reproduce exact training images. Generation of novel combinations proves it learned structure, not instances. | Section 5 (Elaborate) — after core concept is established. Address head-on with the "memorization vs structure" framing. |
| "You need completely different neural network operations for generation (not the same layers)" | Every model they've built uses specific layers (linear, conv, relu) for classification. Generation seems like a fundamentally different task. | The decoder in an autoencoder uses the same conv layers, just transposed. The language model they studied in 4.1 uses the same transformer blocks for generation as BERT uses for classification. Same weighted sums, same activations, same backprop. The difference is the objective (what the loss function measures), not the building blocks. | Section 4 (Explain) — as part of the "same tools, different question" framing. |
| "Generation is just classification in reverse (undo the classification to get back an image)" | Natural inversion intuition. If classification maps image -> label, generation must map label -> image. This misses the many-to-one nature of classification. | "Cat" doesn't map to ONE image — there are millions of possible cat images. Classification is many-to-one (all cats -> "cat"), so inverting it is one-to-many (which cat?). You'd need to somehow specify WHICH cat image. This is exactly the role of sampling from a distribution. | Section 4 (Explain) — immediately after the discriminative recap. The many-to-one observation motivates the distributional framing. |
| "A probability distribution over images means the model stores a probability for every possible image" | Student's experience with distributions is discrete (10 classes, 50K tokens). Images are continuous and high-dimensional. Extrapolating "store a probability per item" to images is intractable. | A 28x28 grayscale image has 256^784 possible pixel configurations — more than atoms in the universe. No model could store a probability for each. Instead, the model learns a compact representation of which images are likely (the structure of the distribution), not an explicit probability for every possibility. | Section 4 (Explain) — when introducing "distribution over data." The impossibility of explicit enumeration motivates learning structure. |

### Examples Planned

| Example | Type | Purpose | Why This Example |
|---------|------|---------|-----------------|
| Language model as a generative model the student already knows | Positive | Bridge from familiar to new. The student has already seen a generative model (LM predicts next tokens, samples to generate text) without calling it that. Reveals that "generative" isn't alien. | The student studied language models in 4.1 and interacted with the temperature slider. Reframing what they already know as "generative" removes the intimidation factor and provides a concrete anchor. |
| MNIST classifier vs MNIST generator side-by-side | Positive | The core comparison. Same dataset, two different questions: "what digit is this?" vs "what digits could exist?" Shows the discriminative/generative split on familiar ground. | MNIST is the student's most familiar dataset (used in Series 2 and 3). Using it strips away novelty so the conceptual difference is isolated. The classifier is something they literally built. |
| Inverting a classifier to generate ("just run it backward") | Negative | Exposes why generation is not the reverse of classification. Classification is many-to-one (all 7s -> "7"), so inversion is one-to-many. Which 7? This impossibility motivates the distributional approach. | Directly addresses the "generation = classification in reverse" misconception. The student needs to feel the problem before seeing the solution (probability distributions). |
| Histogram of handwritten digit strokes as a simple distribution | Positive (stretch) | Makes "distribution over data" concrete with a low-dimensional example. Before jumping to "distribution over 784-pixel images," show a distribution over a single feature (stroke width, slant angle). Sampling from this 1D distribution is just picking a value from the histogram. | Reduces the abstraction. Going directly from "distribution over 10 classes" to "distribution over 784 pixels" is too large a jump. A 1D feature distribution is an intermediate step. |

---

## Phase 3: Design

### Narrative Arc

You have spent 40+ lessons learning to answer questions about data: Is this a 7 or a 9? Is this a cat or a dog? What word comes next? Every model you have built takes an input and produces a judgment. But there is a fundamentally different question a neural network can learn to answer: not "what is this?" but "what could exist?" When you trained a classifier on MNIST, the model learned where the boundaries between digits lie. But imagine a model that instead learned what handwritten digits actually look like — the range of stroke widths, the ways a loop can be drawn, the typical slant of a 4. Such a model would not just recognize digits; it could create new ones that look like they belong to the same distribution of handwriting. This is the generative turn: from modeling decision boundaries to modeling the data itself. And you have already seen it once without knowing it — when you watched a language model sample the next token, it was generating from a learned distribution. This lesson makes that implicit knowledge explicit, and opens the door to generating images rather than just text.

### Modalities Planned

| Modality | What Specifically | Why This Modality for This Concept |
|----------|------------------|------------------------------------|
| **Visual** | Side-by-side diagram: discriminative model (input -> boundary -> label) vs generative model (distribution -> sample -> new instance). Two-panel layout showing the same MNIST data viewed through two lenses. | The core distinction is spatial/geometric: discriminative models draw boundaries IN the data space, generative models model the DENSITY of the data space. A visual makes this tangible. |
| **Verbal/Analogy** | Art critic vs artist analogy. A critic (discriminative) looks at a painting and judges: "This is impressionist, not cubist." An artist (generative) has internalized what impressionist paintings look like and can paint a new one. Both understand impressionism, but the critic draws boundaries between styles while the artist models the space of possible paintings within a style. | The analogy maps precisely: same domain knowledge, different capability. It also naturally addresses the "different kind of network" misconception — the critic and artist both studied the same paintings. |
| **Concrete example** | Interactive widget: a 2D scatter plot showing two classes of points (e.g., two clusters). A discriminative toggle shows the decision boundary. A generative toggle shows the probability density (heatmap or contour) for each class. Sampling mode lets the student click "generate" and see new points drawn from the learned density. | Seeing samples appear from the density — rather than being classified — is the "aha" moment. The student manipulates the same data under two paradigms. 2D keeps it tractable while preserving the essential distinction. |
| **Intuitive** | The language model callback: "You already know a generative model. When the language model assigned probabilities to next tokens and you sampled one, that was generation. The temperature slider changed the distribution. The model never memorized specific sentences — it learned the structure of language well enough to produce new ones." | Connects to existing knowledge the student has at DEVELOPED depth. The insight that they've already done generation — just with tokens, not pixels — removes the conceptual barrier. |
| **Symbolic** | P(y|x) for discriminative (probability of label given input) vs P(x) for generative (probability of the data itself). Brief, clearly defined, no derivation. Just naming the two objectives so the student has precise language. | Provides the formal anchor. The student has seen P(y|x) implicitly (softmax output of classifier) and P(next_token | context) explicitly. Naming P(x) completes the picture. Not heavy math — just notation for a concept already understood intuitively. |

### Cognitive Load Assessment

- **New concepts in this lesson:** 2-3 (generative vs discriminative framing, probability distributions over data, sampling as generation)
- **Previous lesson load:** N/A (first lesson in a new series, but the most recent lesson the student completed was in Module 4.2 which ended with a CONSOLIDATE lesson)
- **Assessment:** BUILD is appropriate. The concepts are genuinely new in framing but connect directly to things the student already knows (classification, softmax distributions, language model sampling). No new math, no new code, no new architecture. The lesson is a perspective shift, not a difficulty spike.

### Connections to Prior Concepts

- **"ML is function approximation"** -> extended to "ML can also approximate the data distribution." The approximation target changes from a function f(x)->y to a distribution P(x).
- **Softmax output as a distribution** -> The student has seen a model produce a distribution over 10 classes (MNIST) and over 50K tokens (language model). This lesson generalizes: what if the "distribution" is over images themselves?
- **Temperature slider** -> Direct callback. The student adjusted temperature and saw how it changed which tokens get sampled. Same idea applies to image generation: the model learns a distribution, and generation = sampling from it.
- **Autoregressive generation loop** -> The student has seen "sample, append, repeat" for text. Image generation follows a similar principle (sample from a distribution to produce the output), though the mechanism differs.
- **"Architecture encodes assumptions about data"** -> Extended. Discriminative architectures encode assumptions about what distinguishes classes. Generative architectures encode assumptions about what the data looks like.

**Potentially misleading prior analogies:**
- **"ML is function approximation"** could be misleading if taken too literally. A generative model doesn't approximate a function in the same way a classifier does — it approximates a distribution. The lesson needs to explicitly extend the mental model rather than break it.
- **Next-token prediction** is autoregressive generation, which is one form of generative modeling. The student might overgeneralize and think all generation is autoregressive. Scope boundary: this lesson mentions autoregressive as ONE approach and notes that image generation uses different strategies (preview for the rest of the module).

### Scope Boundaries

**This lesson IS about:**
- The conceptual distinction between discriminative and generative models
- What it means to "learn a distribution" at an intuitive level
- Why sampling from a distribution produces novel instances
- Connecting language model generation (familiar) to image generation (new)

**This lesson is NOT about:**
- Any specific generative architecture (autoencoders, VAEs, GANs, diffusion) — those come in Lessons 2-4 and beyond
- Probability density functions, likelihood, or any formal probability theory
- How to train a generative model (loss functions, objectives)
- Image generation in practice
- The quality of different generative approaches
- Latent spaces or representations (Lesson 2)

**Target depth for core concepts:**
- Generative vs discriminative framing: INTRODUCED (can explain the distinction, hasn't built one)
- Probability distributions over data: INTRODUCED (intuitive understanding, not formal)
- Sampling as generation: INTRODUCED (understands the concept via language model callback, hasn't done it for images)

### Lesson Outline

1. **Context + Constraints** (~2 paragraphs)
   - What this lesson is about: a change in perspective, not a new technique
   - What we are NOT doing: building a generative model (that starts next lesson), formal probability, specific architectures
   - Series 6 roadmap: where we are headed (Stable Diffusion) and what this module covers (generative foundations)

2. **Hook: "You've already seen a generative model"** (type: misconception reveal)
   - Open with: "Every model you've built answers 'what is this?'. But you've already seen a model that answers 'what could come next?'"
   - Callback to language model temperature slider from 4.1. The student dragged the slider and watched the distribution reshape. When they sampled, the model generated text it had never seen in training.
   - Reframe: "That was generation. The language model learned the distribution of text well enough to produce new text. This lesson asks: can we do the same thing with images?"
   - Why this hook: the student already has the concept but doesn't know they have it. Revealing this removes the intimidation of "generation" and provides an anchor for everything that follows.

3. **Explain: The Discriminative Paradigm (recap + reframe)** (~3 paragraphs)
   - Brief recap of what the student has built: classifiers that draw decision boundaries. MNIST: 784 pixels -> 10 probabilities. CNN: spatial features -> class. LLM: context -> next token distribution.
   - The defining characteristic: all of these learn P(y|x) — the probability of a label given an input. The model doesn't need to understand what the input IS, only which category it belongs to.
   - Visual: a 2D scatter plot with two classes and a decision boundary. The boundary is the model's entire output. Everything on one side is "class A," everything on the other is "class B." The model says nothing about what the data looks like — only where the boundary is.

4. **Explain: The Generative Question** (~4-5 paragraphs)
   - "What if, instead of learning where the boundary is, we learned what the data looks like?"
   - The negative example: "Can you invert the classifier?" All 7s map to the label "7." Which 7 do you get back? There are millions of possible 7s. Classification is many-to-one; inversion is one-to-many. You need to know WHICH 7.
   - The key shift: instead of P(y|x), learn P(x) — the distribution of the data itself. "What does a 7 look like?" has an answer, but it's not a single image — it's a distribution over possible 7s.
   - Art critic vs artist analogy. Both study impressionism. The critic judges; the artist creates. Both have internalized the same knowledge, but the artist can sample from it.
   - Notation: P(y|x) = discriminative, P(x) = generative. Brief, no derivation. Just naming the objectives.

5. **Check: Predict-and-verify**
   - Question: "A model has learned the distribution of handwritten 5s. You sample from it twice. Do you get the same image both times? Why or why not?"
   - Expected answer: No — sampling from a distribution is stochastic. Each sample is a different plausible 5, just as each sample from the language model is a different plausible next token.
   - This checks whether the student connects "distribution" to "stochastic sampling" — the essential link.

6. **Explore: Interactive widget — Discriminative vs Generative on 2D data**
   - Widget: GenerativeVsDiscriminativeWidget
   - A 2D scatter plot with two classes (e.g., two Gaussian clusters, visually resembling the data the student has seen in Series 1).
   - Toggle between two modes:
     - **Discriminative mode:** Shows the decision boundary. Points near the boundary get uncertain classifications. The model says nothing about the density of the data.
     - **Generative mode:** Shows the learned density as a heatmap/contour. No decision boundary — instead, darker regions = more likely data. A "Sample" button draws new points from the density.
   - The student sees: (a) the discriminative view only cares about the boundary, (b) the generative view captures the shape/density of each cluster, (c) sampling produces novel points that fall within the high-density regions.
   - TryThisBlock prompts: "Click Sample 5 times. Are the new points identical? Do they all fall within the training data exactly? Notice how they cluster in high-density regions but are never exact copies of training points."

7. **Elaborate: Why this is hard (and why it's not just memorization)** (~3-4 paragraphs)
   - The scale problem: a 28x28 grayscale image has 256^784 possible pixel configurations. You cannot enumerate them. A model that "memorized" every training image would have only 60,000 points in a space of 10^1888 possibilities. Memorization cannot explain generation of novel images.
   - What the model actually learns: structure. Digits have strokes. Strokes have thickness and curvature. 7s have a horizontal bar and a diagonal. The model learns these structural regularities — the "rules" of what makes a plausible digit — not a catalog of specific digits.
   - Connection to feature hierarchy: "Remember how CNN layers learned edges -> textures -> parts -> objects? A generative model learns similar structure, but uses it to CREATE rather than CLASSIFY."
   - Scope limit: we are not yet talking about HOW a model learns this distribution. That starts next lesson with autoencoders. For now: the concept is clear, the "why" is motivated, and the "how" is the open question.

8. **Check: Transfer question**
   - "Your colleague says: 'Generative AI just memorizes images from the internet and remixes them.' Using what you learned about the dimensionality of image space and the size of training sets, explain why this can't be the full story."
   - Expected reasoning: The space of possible images is astronomically larger than any training set. If the model only memorized, it could only reproduce exact training images. But generative models produce novel images — combinations of features never seen together in training. The model must have learned STRUCTURE (stroke patterns, spatial relationships, textures), not a lookup table.

9. **Summarize** (~2 paragraphs)
   - Two questions, two paradigms: "What is this?" (discriminative, P(y|x)) vs "What could exist?" (generative, P(x))
   - Generation = sampling from a learned distribution. The model doesn't memorize — it learns the structure of the data well enough to produce plausible new instances.
   - Echo the language model callback: "You've been generating since Module 4.1. This module teaches you how to do it with images."
   - Key mental models: "discriminative draws boundaries, generative models density" and "generation is sampling from a learned distribution"

10. **Next step** (~1 paragraph)
    - "We know WHAT generation means — sample from a learned distribution. But HOW does a neural network learn a distribution over images? Next lesson: autoencoders. The idea is simple and powerful: force a network to compress an image into a tiny representation, then reconstruct it. If the reconstruction is good, the tiny representation captured the essential structure. And that tiny representation is our first step toward a space we can sample from."

---

## Checklists

### Prerequisite Audit
- [x] Every assumed concept listed with required depth
- [x] Each traced via the records (not the curriculum plan)
- [x] Depth match verified for each
- [x] No untaught concepts remain
- [x] No multi-concept jumps in widgets/exercises
- [x] All gaps have explicit resolution plans (no gaps found)

### Pedagogical Design
- [x] Narrative motivation stated as a coherent paragraph (problem before solution)
- [x] At least 3 modalities planned for the core concept, each with rationale (5 planned: visual, verbal/analogy, concrete/interactive, intuitive, symbolic)
- [x] At least 2 positive examples + 1 negative example, each with stated purpose (3 positive + 1 negative)
- [x] At least 3 misconceptions identified with negative examples (4 identified)
- [x] Cognitive load = 2-3 new concepts (generative vs discriminative, distribution over data, sampling as generation)
- [x] Every new concept connected to at least one existing concept
- [x] Scope boundaries explicitly stated

---

## Review — 2026-02-09 (Iteration 1/3)

### Summary
- Critical: 1
- Improvement: 3
- Polish: 2

### Verdict: NEEDS REVISION

No issues that would leave the student completely lost, but one critical finding (language model classification contradiction) creates real confusion at a pivotal moment in the lesson. The improvement findings address a missing planned example, an unaddressed misconception, and a missed opportunity in the widget. Polish items are minor.

### Findings

#### [CRITICAL] — Language model listed as both generative and discriminative without resolution

**Location:** Section 2 ("You've Already Seen a Generative Model") and Section 3 ("The Discriminative Paradigm"), specifically the bullet list at lines 186-197.

**Issue:** Section 2 establishes clearly that the language model IS a generative model: "When the model sampled 'mat' after 'The cat sat on the,' it was generating." Two scrolls later, Section 3 lists the language model as discriminative: "Language model: context tokens -> distribution over next token (which word comes next?)" alongside MNIST classifiers and CNNs.

The lesson never resolves this contradiction. The student reads "language model = generative" then "language model = discriminative" and has no framework for understanding how both can be true. The reality is nuanced: a language model learns P(next token | context), which is technically a conditional distribution (discriminative in form), but the autoregressive feedback loop makes it generative in capability. This subtlety needs to be surfaced, not left as an unresolved contradiction.

**Student impact:** The student would feel genuinely confused. They just had the "aha" moment ("I already know a generative model!") and then it appears in a list of discriminative models. This undermines the hook and creates distrust in the lesson's clarity. The student might conclude either: (a) the lesson contradicts itself, or (b) they don't understand the distinction as well as they thought.

**Suggested fix:** In the Section 3 bullet list, either (a) remove the language model from the discriminative list entirely, or (b) include it with a clear qualifier like: "Language model: context tokens -> distribution over next token. Technically discriminative in form (it learns P(next_token | context)), but the autoregressive loop turns it into a generator. We'll return to this nuance." Option (a) is cleaner for the lesson flow. If the language model is the hook for generation, don't also use it as an example of discrimination. Replace it with another discriminative example the student knows (e.g., transfer learning on flowers from Series 3, or sentiment analysis as a concept they'd recognize).

---

#### [IMPROVEMENT] — Missing planned example: histogram of digit strokes as a simple distribution

**Location:** Section 4 ("The Generative Question"), between the introduction of P(x) and the art critic analogy (around lines 270-280).

**Issue:** The planning document included a fourth example: "Histogram of handwritten digit strokes as a simple distribution." The purpose was to bridge from discrete distributions the student knows (softmax over 10 classes, vocabulary of 50K tokens) to the idea of a distribution over images. The plan noted: "Going directly from 'distribution over 10 classes' to 'distribution over 784 pixels' is too large a jump. A 1D feature distribution is an intermediate step."

This example is absent from the built lesson. Instead, the lesson jumps from "learn P(x) -- the distribution of the data itself" (abstract) directly to the art critic analogy (metaphorical). The student has no concrete, low-dimensional example of what "a distribution over data" looks like before being asked to imagine one over 784-dimensional images.

**Student impact:** The student understands distributions over discrete categories (10 digits, 50K tokens). They are now asked to imagine a distribution over continuous, high-dimensional data (images). Without an intermediate example, "distribution over data" remains abstract. A 1D histogram (e.g., "here's the distribution of stroke widths in the digit 7 -- most are medium thickness, some are thin, few are very thick") would make the concept tangible before scaling it up.

**Suggested fix:** Add a brief paragraph or small visual between the P(x) introduction and the art critic analogy. Something like: "What does a distribution over data look like? Start simple. If you measured the stroke width of every 7 in MNIST, you'd get a histogram -- most 7s have medium stroke width, some are thin, some are thick. That histogram IS a distribution over one feature of 7s. Sampling from it gives you a stroke width for a new 7. Now imagine doing this for every feature simultaneously -- stroke angle, crossbar presence, overall slant. A distribution over all these features together is a distribution over 7s. That's what P(x) captures."

---

#### [IMPROVEMENT] — Misconception 4 (distribution stores probability for every image) not explicitly surfaced

**Location:** Section 4, around the introduction of P(x) (lines 270-280), and Section 7 (lines 430-450).

**Issue:** The planning document identified four misconceptions. Misconception 4 -- "A probability distribution over images means the model stores a probability for every possible image" -- was planned to be addressed "when introducing 'distribution over data'" in Section 4. The scale argument in Section 7 (256^784 possibilities) implicitly shows that explicit enumeration is impossible, but the misconception is never surfaced as a misconception. The student might read Section 4 and form exactly this wrong mental model ("P(x) means there's a probability stored for every possible image"), then only encounter the scale argument three sections later, by which point the wrong model is entrenched.

**Student impact:** The student's experience with distributions is discrete: softmax over 10 classes (store 10 probabilities), vocabulary distribution over 50K tokens (store 50K probabilities). Extrapolating to "store a probability for each of 10^1888 possible images" is the natural (wrong) conclusion. Without explicit correction at the point where P(x) is introduced, the student builds a wrong mental model that persists until Section 7.

**Suggested fix:** When P(x) is first introduced (Section 4, around line 275), add 1-2 sentences that explicitly address this: "P(x) does NOT mean the model stores a separate probability for every possible image -- that would require more storage than all the atoms in the universe. Instead, the model learns a compact representation of which regions of image space are likely. We'll see the staggering scale of image space in the next section, which makes it clear why the model must learn structure rather than enumerate possibilities."

---

#### [IMPROVEMENT] — Widget does not show annotations or labels explaining what is pedagogically significant

**Location:** Section 6, the GenerativeVsDiscriminativeWidget component.

**Issue:** The widget shows the visual contrast well (boundary vs heatmap, sampling), but it relies entirely on the TryThisBlock in the aside to guide interpretation. The widget itself has no annotations, callouts, or guided observations. For example, in discriminative mode, the boundary region could be labeled "The model only cares about this line." In generative mode, high-density regions could have annotations like "More likely here" or the heatmap legend could show a density scale.

The widget's description panel below the toggle ("Discriminative: The model learns a decision boundary...") helps, but it's text below the toggle, not integrated into the visualization. The student is doing visual processing (looking at the plot) and reading processing (reading the aside) simultaneously, which splits attention.

**Student impact:** Some students will look at the widget, toggle back and forth, and get the "aha" immediately. Others (especially with ADHD, as noted in the design principles) may look at the pretty heatmap and not extract the pedagogical point without reading the aside carefully. Annotations on the visualization itself would make the key insight available in the visual channel directly.

**Suggested fix:** Add minimal text annotations to the SVG itself. In discriminative mode, a label near the boundary line: "Decision boundary -- the only thing this model learned." In generative mode, a small label in the densest region: "High density -- data is likely here." These would reinforce the key contrast without requiring the student to cross-reference with the aside.

---

#### [POLISH] — Section 3 paragraph about 2D scatter plot is verbal-only; no actual visual provided there

**Location:** Section 3, lines 205-210: "Think of a 2D scatter plot with two classes of points and a line separating them."

**Issue:** The lesson asks the student to "think of" a 2D scatter plot, but doesn't show one until Section 6 (the widget). The verbal description works, but the interactive widget that actually shows this is three sections away. A small inline diagram or a forward reference ("which you'll see in the interactive widget below") would help.

**Student impact:** Minor. The student can imagine a 2D scatter plot. But the lesson is asking the student to visualize something that it will show them later. A forward reference would set expectations.

**Suggested fix:** Add a brief forward reference: "Think of a 2D scatter plot with two classes of points and a line separating them (you'll interact with exactly this in a moment). The discriminative model's entire job is drawing that line."

---

#### [POLISH] — "models density" ambiguity in mental models echo

**Location:** Mental Models section, line 566: "Discriminative draws boundaries, generative models density."

**Issue:** "models density" is grammatically ambiguous. "Models" could be read as a noun (generative models) or a verb (the generative approach models density). On first read, the sentence parses as "discriminative draws boundaries, generative models [are about] density" which is awkward. The intended reading is "discriminative [models] draw boundaries, generative [models] model density" where the second "models" is a verb.

**Student impact:** Negligible. The student will likely parse it correctly from context. But mental model summaries should be instantly clear since they're meant to be memorable anchors.

**Suggested fix:** Rephrase to remove ambiguity: "Discriminative models draw boundaries; generative models learn density." Or: "A discriminative model asks 'which side of the line?' A generative model asks 'how dense is this region?'" (The second version is already present in the next sentence, so the first option is cleaner.)

### Review Notes

**What works well:**
- The language model callback as the hook is excellent. It leverages the student's strongest experiential knowledge (temperature slider from 4.1) and reframes it. This is exactly the kind of bridge that makes a conceptual lesson feel like a natural extension rather than a cold start.
- The negative example (inverting the classifier) is one of the best in the curriculum so far. The many-to-one / one-to-many reasoning is simple, concrete, and genuinely illuminating.
- The scope boundaries are crisp and well-communicated. The student knows exactly what this lesson is and isn't about.
- The ComparisonRow (art critic vs artist) is well-mapped and memorable.
- The widget is well-designed: clean, focused, and the sampling mechanism makes the abstract concept tangible.
- The pacing is appropriate for a BUILD lesson. No section is too dense.

**Systemic observation:**
The one critical finding (language model contradiction) is a byproduct of the language model being genuinely hard to categorize. It learns a conditional distribution P(next_token | context) (discriminative form) but generates via sampling (generative capability). The lesson needs to either own this complexity with a brief aside, or cleanly separate the language model into the "generative" camp and not use it as a discriminative example. The current approach of using it in both roles without acknowledgment is the worst option.

**Overall assessment:**
This is a strong conceptual lesson. The narrative arc, modality coverage, and connection to prior knowledge are well-executed. The critical finding is localized (a few lines in Section 3) and straightforward to fix. The improvement findings are genuine strengthening opportunities, not nitpicks. After revision, this should pass on the next review iteration.

---

## Review — 2026-02-09 (Iteration 2/3)

### Summary
- Critical: 0
- Improvement: 0
- Polish: 1

### Verdict: PASS

All six findings from iteration 1 have been resolved effectively. The critical finding (language model contradiction) was fixed with the best possible approach: removing the LM from the discriminative list and adding an italic paragraph that explicitly addresses the nuance ("discriminative in form, but the autoregressive loop turns that prediction into generation"). The three improvement findings (missing histogram example, unsurfaced misconception 4, widget annotations) are all present in the revised lesson and widget. Both polish findings (forward reference for scatter plot, "models density" ambiguity) are fixed. One new minor issue was introduced by the fixes, documented below.

### Findings

#### [POLISH] — Scale numbers (10^1888, atoms in universe) appear twice in nearly identical phrasing

**Location:** Section 4 (lines 295-302, the misconception 4 address) and Section 7 (lines 460-473, the memorization argument).

**Issue:** The iteration 1 fix for misconception 4 correctly added a clarification when P(x) is first introduced: "With 784 pixels each taking 256 values, there are roughly 10^1888 possible images -- more than the atoms in the universe." Three sections later, the memorization argument repeats nearly the same calculation: "A 28x28 grayscale image has 784 pixels, each taking a value from 0 to 255. The number of possible pixel configurations is 256^784 -- approximately 10^1888. That is incomprehensibly larger than the number of atoms in the observable universe (roughly 10^80)."

The two uses serve different pedagogical purposes: Section 4 explains why P(x) can't be an explicit lookup table; Section 7 uses the ratio of training data to possibility space to argue against memorization. The memorization argument IS additive (it introduces the 60,000 training images vs 10^1888 possibilities ratio). But the student will experience deja vu because the impressive numbers are restated rather than referenced.

**Student impact:** Minor. The student might think "didn't we already establish this?" when reaching Section 7. The memorization argument's unique contribution (the training set size ratio) risks being lost in the feeling of repetition.

**Suggested fix:** In Section 7, reference back rather than re-derive: "Remember the scale we saw earlier: roughly 10^1888 possible pixel configurations for a 28x28 image. The MNIST training set has 60,000 images. A model that memorized every single training image would have 60,000 points in a space of 10^1888 possibilities." This preserves the memorization argument while acknowledging the student already knows the scale number.

### Iteration 1 Fix Verification

| Iteration 1 Finding | Severity | Fix Status | Notes |
|---------------------|----------|------------|-------|
| Language model listed as both generative and discriminative | CRITICAL | RESOLVED | LM removed from discriminative list. Italic paragraph (lines 199-204) explicitly addresses the nuance: "discriminative in form" but "autoregressive loop turns that prediction into generation." Clean, accurate, and doesn't undermine the hook. Best possible resolution. |
| Missing histogram of digit strokes example | IMPROVEMENT | RESOLVED | Added in Section 4 (lines 286-293). Builds incrementally from 1D (stroke width) to multi-feature to full distribution. Well-placed between P(x) introduction and art critic analogy. Exactly as suggested. |
| Misconception 4 not explicitly surfaced | IMPROVEMENT | RESOLVED | Added in Section 4 (lines 295-302). Explicit "P(x) does NOT mean the model stores a separate probability for every possible image" with the impossibility argument. Placed at the point of introduction, before the wrong model can form. Introduced a minor redundancy with Section 7 (see Polish finding above). |
| Widget lacks inline annotations | IMPROVEMENT | RESOLVED | Discriminative mode: "Decision boundary -- the only thing this model learned" annotated near the boundary line (widget line 303). Generative mode: "High density -- sample new points from here" annotated above Cluster A (widget lines 508-518). Both use white text at 0.85 opacity, readable against the dark background. |
| 2D scatter plot verbal-only, no forward reference | POLISH | RESOLVED | Forward reference added: "(you'll interact with exactly this in a moment)" at line 216. |
| "models density" ambiguity | POLISH | RESOLVED | Rephrased to "Discriminative models draw boundaries; generative models learn density" (line 591). Unambiguous -- "models" is clearly a noun, "learn" is the verb. |

### Review Notes

**What works well (reinforced from iteration 1, with additions):**
- The language model nuance paragraph is the standout fix. Rather than simply removing the LM from the discriminative list (which would dodge the nuance), the lesson owns the complexity in a way that actually enriches the student's understanding. The phrase "discriminative in form" vs "generative in capability" is a genuinely useful distinction that will serve the student well.
- The histogram example is exactly the right intermediate step. The progression from "distribution over 10 classes" (student knows this) to "distribution over stroke widths" (1D, concrete) to "distribution over all features simultaneously" (multi-dimensional, still grounded) to P(x) (abstract notation for what they now understand) is a model scaffold.
- The misconception 4 address prevents the wrong mental model from forming at all. The student reads P(x), immediately sees "this does NOT mean a probability for every image," and arrives at the scale argument in Section 7 already understanding structure-vs-enumeration. Section 7 then extends the argument to memorization, which is a natural next step.
- Widget annotations make the key insight available in the visual channel without requiring the student to read the aside. The annotations are minimal and well-placed.
- The sentiment classifier example in the discriminative list (which replaced the language model) is a transparent example that doesn't require prior teaching. The student can understand "text -> positive or negative" from general knowledge.

**Overall assessment:**
The lesson is now clean, complete, and effective. All planned examples are present. All four misconceptions are addressed at appropriate locations. All five modalities are working. The narrative flows smoothly from hook through concept development to application. The single remaining finding is genuinely Polish-level -- a minor redundancy that doesn't impair learning. This lesson is ready to ship.

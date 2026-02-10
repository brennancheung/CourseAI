# Lesson: Pretraining — The Training Loop

**Module:** 4.3 (Building & Training GPT)
**Position:** Lesson 2 of 4
**Slug:** `pretraining`
**Notebook:** `4-3-2-pretraining.ipynb`
**Cognitive load:** STRETCH
**Previous lesson:** building-nanogpt (BUILD)
**Next lesson:** scaling-and-efficiency (BUILD)

---

## Phase 1: Orient — Student State

The student arrives with a working GPT model in PyTorch that generates gibberish. They built every class (Head, CausalSelfAttention, FeedForward, Block, GPT), verified parameter counts (~124.4M), and ran the generate() method to produce random text. The architecture is complete; it just needs to learn.

### Relevant Concepts (with depths and sources)

| Concept | Depth | Source | Notes |
|---------|-------|--------|-------|
| Complete PyTorch training loop (forward -> loss -> backward -> update) | DEVELOPED | Series 1 (1.1.6), DEVELOPED again in Series 2 (2.1.3, 2.1.4) | The universal heartbeat. Student has written this loop many times for MNIST, Fashion-MNIST. "Same heartbeat, new instruments." |
| Cross-entropy loss | INTRODUCED | Series 2 (2.2.1) | nn.CrossEntropyLoss takes raw logits + integer labels. "Confidence penalty." Student has used it for classification (10 classes). Has NOT used it for next-token prediction over a 50K vocabulary. |
| Dataset/DataLoader abstraction | DEVELOPED | Series 2 (2.2.1) | Custom Dataset subclass + DataLoader for batching and shuffling. Student knows the pattern well. |
| GPT architecture in PyTorch (all 5 classes) | APPLIED | building-nanogpt (4.3.1) | Student wrote every class, verified shapes, ran the model. This is what they are training. |
| Autoregressive generation (generate method) | DEVELOPED | building-nanogpt (4.3.1) | torch.no_grad(), crop to block_size, forward pass, last position logits, temperature, sample, append. |
| Language modeling as next-token prediction | DEVELOPED | what-is-a-language-model (4.1.1) | P(x_t | x_1,...,x_{t-1}). Self-supervised labels: text provides its own training data. |
| BPE tokenization | APPLIED | tokenization (4.1.2) | Built BPE from scratch. Merge table IS the tokenizer. Student understands how text becomes token IDs. |
| Token embeddings as learned lookup | DEVELOPED | embeddings-and-position (4.1.3) | nn.Embedding. 50K x 768 matrix. One-hot times matrix = row selection. |
| Weight initialization for transformers | DEVELOPED | building-nanogpt (4.3.1) | Normal with sigma=0.02, residual projection scaling by 1/sqrt(2N). Concrete activation statistics. |
| Gradient clipping | MENTIONED | Series 1 (1.3.3) | Named as "safety net for exploding gradients." Not practiced, not explained in depth. |
| Learning rate as hyperparameter | DEVELOPED | Series 1 (1.1.5) | Goldilocks zone, oscillation/divergence failure modes. Constant LR only — no scheduling. |
| Adam optimizer | DEVELOPED | Series 2 (2.1.4) | Momentum + RMSProp + bias correction. defaults lr=0.001. One-line swap from SGD. |
| AdamW | MENTIONED | Series 1 (1.3.7) | Named as "practical default optimizer." Student has not used it directly. |
| Training curves as diagnostic | DEVELOPED | Series 1 (1.3.6) | Plot train + val loss, three phases, "scissors" pattern for overfitting. |
| Checkpoint pattern | DEVELOPED | Series 2 (2.3.1) | model + optimizer + epoch + loss dict, periodic saves, resume training. |
| Mixed precision training (autocast + GradScaler) | INTRODUCED | Series 2 (2.3.2) | Float16 forward, float32 gradients. 4-line addition. Conceptual but has written the code. |
| Causal masking | DEVELOPED | decoder-only-transformers (4.2.6), building-nanogpt (4.3.1) | Lower-triangular mask. Training is parallel with masking; inference is sequential. |

### Mental models already established

- "Training loop = forward -> loss -> backward -> update" (universal heartbeat)
- "Same heartbeat, new instruments" (PyTorch automates the familiar loop)
- "Clear, compute, use" (gradient lifecycle: zero_grad, backward, step)
- "A language model approximates P(next token | context)" (the objective)
- "Untrained gibberish is a success" (correct architecture, needs training)
- "Autoregressive generation is a feedback loop" (outputs become inputs)
- "Tokenization defines what the model can see"
- "Baseline-then-improve" (experimental methodology)

### What was explicitly NOT covered

- Learning rate scheduling of any kind (warmup, cosine decay, step decay)
- Gradient clipping in practice (only MENTIONED as a concept)
- Training a language model (next-token prediction loss over a full vocabulary)
- Text dataset preparation (chunking text into training examples)
- AdamW in practice (only named, never used)
- Training dynamics at transformer scale (loss magnitude, convergence behavior)
- Watching generated text quality improve during training

### Readiness Assessment

The student is well-prepared. The training loop pattern is deeply familiar from Series 1-2 (DEVELOPED/APPLIED across multiple lessons). The GPT model is built and verified. The conceptual framework for next-token prediction is solid from Module 4.1. The genuinely new content is: (1) how to prepare a text dataset for language modeling (context windows, input/target offset), (2) learning rate scheduling (warmup + cosine decay), and (3) gradient clipping in practice. These are all extensions of familiar concepts, not entirely new paradigms. The emotional payoff (watching text improve from gibberish to recognizable patterns) will sustain motivation through the STRETCH load.

---

## Phase 2: Analyze

### Target Concept

This lesson teaches the student to train a GPT model from scratch — preparing a text dataset, building the training loop with LR scheduling and gradient clipping, and interpreting loss curves and generated text quality as training progresses.

### Prerequisites Table

| Concept | Required Depth | Actual Depth | Source Lesson | Status | Reasoning |
|---------|---------------|-------------|---------------|--------|-----------|
| PyTorch training loop | DEVELOPED | DEVELOPED | 2.1.3, 2.1.4 | OK | Student needs to adapt the familiar loop, not learn it. The pattern (forward, loss, backward, step) is second nature. |
| Cross-entropy loss (nn.CrossEntropyLoss) | DEVELOPED | INTRODUCED | 2.2.1 | GAP | Student has used cross-entropy for 10-class classification. Needs to understand it for next-token prediction over 50K tokens and how the loss is computed at every position simultaneously. |
| Dataset/DataLoader | DEVELOPED | DEVELOPED | 2.2.1 | OK | Student knows the abstraction well. Text dataset is a new domain but same pattern. |
| GPT model (all 5 classes) | APPLIED | APPLIED | building-nanogpt | OK | The model to be trained. Student built every line. |
| Language modeling as next-token prediction | DEVELOPED | DEVELOPED | 4.1.1 | OK | Conceptual understanding of the objective is solid. Now they implement the loss for it. |
| BPE tokenization | DEVELOPED | APPLIED | 4.1.2 | OK | Student needs to tokenize a dataset. They built BPE from scratch — using a pretrained tokenizer is simpler. |
| Adam / AdamW optimizer | DEVELOPED | DEVELOPED (Adam) / MENTIONED (AdamW) | 2.1.4, 1.3.7 | GAP | Student has used Adam extensively but AdamW only named. Need brief recap of why weight decay is decoupled in AdamW. |
| Learning rate scheduling | INTRODUCED | MISSING | — | MISSING | Student has only used constant LR. Warmup + cosine decay is entirely new. |
| Gradient clipping | INTRODUCED | MENTIONED | 1.3.3 | GAP | Named as "safety net for exploding gradients" but never practiced. Need to explain the mechanic (clip norm) and why transformers need it. |
| Training curves | DEVELOPED | DEVELOPED | 1.3.6 | OK | Student can read and interpret loss curves. Language model loss scale (nats/bits per token) is new but the skill transfers. |
| Checkpointing | DEVELOPED | DEVELOPED | 2.3.1 | OK | Save/resume pattern is known. Apply it here. |

### Gap Resolution

| Gap | Size | Resolution |
|-----|------|------------|
| Cross-entropy for next-token prediction (50K vocab, all positions) | Small | Brief recap section (2-3 paragraphs). Student knows CE for 10-class classification. Extension: same formula, but now vocab_size classes and T positions per sequence. Reshape logits from (B, T, vocab_size) to (B*T, vocab_size), targets from (B, T) to (B*T). "Cross-entropy doesn't care whether it's classifying a shirt vs sneaker or predicting the next word — it's the same formula." |
| AdamW vs Adam | Small | One paragraph aside. Student knows Adam thoroughly. AdamW = same thing, but weight decay is applied directly to parameters instead of added to loss. In practice: swap `optim.Adam` for `optim.AdamW`. torch.optim.AdamW is the transformer default. |
| Learning rate scheduling (warmup + cosine decay) | Medium | Dedicated section within this lesson. This is genuinely new. Motivate: constant LR is fragile at this scale. Too high at start = instability with random weights. Too low throughout = slow training. Solution: warmup (ramp up from near-zero) + cosine decay (gradually reduce). Visual: LR schedule plot. Code: torch.optim.lr_scheduler or manual lambda. |
| Gradient clipping in practice | Small | Brief section (2-3 paragraphs + code). Student knows the concept name. Explain: compute the global norm of all gradients, if it exceeds a threshold, scale all gradients down proportionally. One line: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`. Why transformers need it: occasional large gradients from long sequences/attention instability. |

### Misconceptions Table

| Misconception | Why They'd Think This | Negative Example That Disproves It | Where to Address |
|---------------|----------------------|-----------------------------------|-----------------|
| "Training a language model requires a fundamentally different loop from MNIST training." | The model is much bigger, the task sounds different (generate text vs classify images), and the scale feels qualitatively different. | Side-by-side code comparison: MNIST loop vs GPT loop. The structure is identical: forward pass, compute loss, backward, step. The differences are the data format and a couple of extra lines (LR scheduling, gradient clipping). Same heartbeat. | Hook section — address immediately by showing the structural similarity |
| "The model sees the entire sequence and predicts just the last token." | Autoregressive generation (the generate() method from Lesson 1) processes one token at a time and only uses the last position. Natural to assume training works the same way. | During training with causal masking, EVERY position predicts its next token simultaneously. A sequence of length T produces T training examples in one forward pass. Show: for "The cat sat", position 0 predicts "cat", position 1 predicts "sat". This is the entire reason training is efficient — not T separate forward passes. | Core explanation section — before the training loop code |
| "Learning rate scheduling is an optimization trick — the model would still converge with a constant LR." | In Series 2, constant LR worked fine for MNIST. Seems like scheduling is unnecessary complexity. | Run training with constant LR vs scheduled LR on the same data. Constant LR at 3e-4: trains OK but suboptimal. Constant LR at 6e-4: training becomes unstable early (random weights + high LR = gradient explosion). Warmup avoids the instability; cosine decay finds lower loss. The schedule is not a trick — it navigates different training regimes (early instability, mid-training exploration, late convergence). | LR scheduling section — show the failure case first |
| "Loss should decrease smoothly, like in MNIST." | MNIST training curves are relatively smooth with small models on curated data. | Language model loss is noisy, especially early in training. Some batches have easy sequences, some have hard ones. Show a real loss curve: jagged with clear downward trend but lots of local variation. Log-scale smoothing helps see the trend. The noise is normal, not a sign of something wrong. | Loss curves section — normalize expectations |
| "Lower loss = better text." | In classification, lower loss correlates directly with accuracy. Seems like the same should hold for text quality. | A model with loss 4.0 produces gibberish. Loss 3.0 produces word-like fragments. Loss 2.5 produces grammatical fragments. Loss 2.0 produces mostly coherent sentences. BUT the relationship is logarithmic, not linear. Going from 4.0 to 3.0 is a dramatic qualitative leap; going from 2.0 to 1.8 is subtle. Also, a model can have low loss but still fail on specific tasks. The mapping from loss to text quality is nonlinear and eventually diminishes. | Generated text samples section — show text at different loss values |

### Examples Planned

| Example | Type | Purpose | Why This Example |
|---------|------|---------|-----------------|
| Side-by-side: MNIST training loop vs GPT training loop | Positive | Show the structural identity of all training loops. "Same heartbeat, new instruments" made concrete. | The student's emotional anchor is the MNIST loop. Showing the GPT loop as the same pattern with minor additions (LR scheduling, gradient clipping, text generation) reduces intimidation and reinforces the universal training loop mental model. |
| Training run with periodic text generation (loss 6.0 -> 4.0 -> 3.0 -> 2.0) | Positive | The emotional core of the lesson. Watch gibberish become recognizable language. Show loss values alongside generated text at each milestone. | This is the payoff the student has been building toward. The progression from random characters to English words to grammatical phrases makes the abstract loss number tangible. Every learner in Karpathy's nanoGPT walkthrough remembers this moment. |
| Constant LR failure at transformer scale | Negative | Demonstrate that constant LR, which worked for MNIST, fails here. Too high causes instability; too low is wasteful. | Motivates LR scheduling by showing the problem it solves. The student needs to feel the limitation of their current tools before accepting the new one. Concrete: same model, same data, only LR schedule differs. |
| Text dataset preparation: input-target offset | Positive | Show how a text sequence becomes training examples. "The cat sat on" -> input "The cat sat", target "cat sat on". Every position is a training example. | This is the key new concept for language model training. The student has prepared image datasets (MNIST) but never text datasets. The input/target offset pattern is simple but non-obvious. Using a short sentence makes it immediately visible. |
| Gradient clipping saving a training run | Positive (illustrative) | Show gradient norm spiking during training (a specific batch causes a large gradient), and how clipping prevents the spike from corrupting learned weights. | Connects MENTIONED concept from 1.3.3 to practice. Without clipping, occasional spikes can undo many steps of training. With clipping, the spike is bounded. This makes the one-line addition feel justified rather than cargo-culted. |

---

## Phase 3: Design

### Narrative Arc

The student has a complete GPT model that generates random gibberish. The architecture is verified — 124 million parameters, shapes match, generation runs. But the model has learned nothing. Every weight is random noise. This lesson takes the model from gibberish to recognizable English by doing the one thing the student has done many times before: training. The core insight is that training a 124M-parameter transformer is structurally identical to training the 3-parameter linear model from Lesson 1.1.6. Forward pass, compute loss, backward pass, update weights. The loop hasn't changed. What's new is the scale — and at this scale, two practical techniques become essential that weren't needed for MNIST: learning rate scheduling (because random transformer weights are fragile — too much learning early means instability) and gradient clipping (because occasional large gradients from long sequences can corrupt an entire training run). The emotional journey: "Wait, this is the same loop? ... It really is the same loop. ... But I need these two extras to make it work at this scale. ... The text is getting better. ... It's writing sentences now." The student emerges understanding that training a language model is not fundamentally different from training any neural network — it's the same algorithm, with two practical additions that become necessary at scale.

### Modalities Planned

| Modality | What Specifically | Why This Modality for This Concept |
|----------|------------------|------------------------------------|
| Verbal/Analogy | "Same heartbeat, new instruments" — the training loop from Series 2 is the same heartbeat; LR scheduling and gradient clipping are two new instruments added to the band. LR scheduling is like warming up before a workout (start gentle, ramp up, then gradually cool down). | The student needs to feel that this is familiar territory with incremental additions, not a new paradigm. The workout warmup analogy is physically intuitive for LR scheduling. |
| Visual | (1) LR schedule plot: warmup ramp + cosine decay curve with annotations showing why each phase matters. (2) Loss curve with periodic text samples annotated at different loss values. (3) Side-by-side code comparison: MNIST loop vs GPT loop with highlights on the 3-4 added lines. | The LR schedule is best understood as a shape (ramp up, curve down). The loss-to-text-quality mapping is best understood by seeing both simultaneously. The code comparison makes the structural identity visually undeniable. |
| Symbolic/Code | Complete training loop code with shape comments. Dataset preparation code with the input/target offset pattern. LR scheduling implementation. Gradient clipping one-liner. All in the notebook. | This is a hands-on lesson. The code IS the primary deliverable. Every concept must be implemented, not just described. |
| Concrete example | (1) "The cat sat on the mat" tokenized and split into input/target pairs at every position. (2) Actual generated text at loss 6.0, 4.0, 3.0, 2.5, 2.0 from a real training run. (3) Gradient norm values before/after clipping. | Abstract concepts like "next-token prediction loss at every position" only click when the student traces through a specific sentence. The text quality progression is the most memorable concrete example in the entire course. |
| Intuitive | "Cross-entropy doesn't care whether it's classifying shirts or predicting words — same formula, different vocabulary size." "Warmup exists because random weights are fragile — big updates early could send you somewhere unrecoverable." "The loop is the loop is the loop." | These one-sentence intuitions compress the key insights into memorable frames that connect to existing knowledge. |

### Cognitive Load Assessment

- **New concepts in this lesson:** 3
  1. Text dataset preparation for language modeling (context windows, input/target offset, all-positions training)
  2. Learning rate scheduling (warmup + cosine decay)
  3. Gradient clipping in practice
- **Previous lesson load:** BUILD (building-nanogpt)
- **This lesson's load:** STRETCH — appropriate. The student just had a BUILD lesson. LR scheduling is entirely new territory. Dataset preparation for language modeling requires a conceptual shift (every position is a training example). Gradient clipping is a smaller addition but still new. Three new concepts is at the limit. The emotional payoff (watching text improve) sustains motivation through the stretch.
- **Next lesson load:** BUILD (scaling-and-efficiency) — good trajectory: BUILD -> STRETCH -> BUILD

### Connections to Prior Concepts

| New Concept | Connects To | How |
|-------------|-------------|-----|
| Text dataset preparation | Dataset/DataLoader from 2.2.1 | Same abstraction (custom Dataset, DataLoader wraps it). The domain changes (text instead of images) but the pattern is identical. |
| Input/target offset | Language modeling as next-token prediction from 4.1.1 | The student learned that the objective is P(next token | context). Now they implement it: input = tokens[:-1], target = tokens[1:]. The conceptual understanding becomes concrete code. |
| Cross-entropy over vocabulary | Cross-entropy for classification from 2.2.1 | Exact same formula, just 50K classes instead of 10. Reshape and it's the same nn.CrossEntropyLoss call. "Wrongness score v2" extends to "wrongness score v3." |
| Training loop structure | Universal training loop from 1.1.6, 2.1.3, 2.1.4 | "Same heartbeat, new instruments." The student's most practiced pattern. |
| LR scheduling warmup | Learning rate as hyperparameter from 1.1.5 | Student knows constant LR and its failure modes (too high = oscillation, too low = slow). Scheduling navigates between these dynamically. "What if the best LR changes as training progresses?" |
| Gradient clipping | Exploding gradients from 1.3.3, gradient clipping MENTIONED in 1.3.3 | Student knows exploding gradients cause NaN. Gradient clipping is the "safety net" they heard about — now they use it. One line of code. |
| Loss curves interpretation | Training curves from 1.3.6 | Same skill (read the curve), new domain. Language model curves are noisier but the diagnostic principles transfer. |
| Checkpointing during training | Checkpoint pattern from 2.3.1 | Save model + optimizer periodically. Already practiced. |
| AdamW optimizer | Adam from 2.1.4, weight decay from 1.3.7 | AdamW = Adam + decoupled weight decay. One-line change. |

**Potentially misleading prior analogies:** The "shuffle=True is how you get random sampling" mental model from 2.2.1 needs nuance for text data. We shuffle chunks/documents, not individual tokens. The data is sequential — you can't randomly sample positions without preserving local context.

### Scope Boundaries

**This lesson IS about:**
- Preparing a text dataset for language model training (TinyShakespeare or similar small corpus)
- Building the complete training loop with all necessary components
- Learning rate scheduling (warmup + cosine decay) — DEVELOPED depth
- Gradient clipping — INTRODUCED depth (use it, understand why, but not deep theory)
- Interpreting loss curves for language model training
- Generating text periodically to observe quality improvement
- Cross-entropy loss applied to next-token prediction across all positions
- AdamW as the transformer training optimizer

**This lesson is NOT about:**
- GPU optimization, mixed precision, flash attention (Lesson 3: scaling-and-efficiency)
- Multi-GPU or distributed training (out of scope for series)
- Different tokenization strategies or training your own tokenizer (Module 4.1 covered this)
- Different model architectures or sizes (one model, one dataset, one training run)
- Hyperparameter search (use known-good hyperparameters from Karpathy's nanoGPT)
- Evaluation beyond loss and qualitative text inspection (perplexity, downstream benchmarks)
- Fine-tuning or transfer learning (Module 4.4)
- Data quality, filtering, deduplication (scaling concerns for Lesson 3 or Module 4.4)

**Target depth:** The training loop at APPLIED (student runs it and can modify it). LR scheduling at DEVELOPED (understands why each phase exists, can adjust parameters). Gradient clipping at INTRODUCED (uses it correctly, understands the motivation, but doesn't need to implement custom clipping strategies). Dataset preparation at DEVELOPED (understands the input/target offset, context windows, and can prepare new datasets).

### Lesson Outline

1. **Context + Constraints**
   - "We have a 124M-parameter GPT that generates gibberish. Today we make it learn."
   - Scope: one model, one dataset, one training run. No GPU optimization, no hyperparameter search, no evaluation benchmarks. Just the training loop and watching text get better.
   - What's new vs familiar: "The training loop is the same one from Series 2. Three things are new: how we prepare text data, learning rate scheduling, and gradient clipping."

2. **Hook: Side-by-side code comparison** (Type: before/after + misconception reveal)
   - Show the MNIST training loop from Series 2 next to the GPT training loop. Highlight that they are structurally identical — same forward/loss/backward/step pattern. The GPT version has 3-4 extra lines. "Wait, that's it?" This immediately addresses the misconception that training a language model requires a fundamentally different approach.

3. **Explain: Dataset Preparation** (New concept 1)
   - **Problem first:** "We have raw text. The model expects tensors of token IDs. How do we turn Shakespeare into training data?"
   - Tokenize the entire corpus into a long sequence of token IDs.
   - Context windows: slice the sequence into chunks of block_size (e.g., 256 tokens).
   - The input/target offset: input = tokens[i:i+T], target = tokens[i+1:i+T+1]. Every position predicts the next token.
   - **Concrete example:** "The cat sat on the mat" -> trace through all 6 prediction tasks from a single sequence.
   - **Misconception addressed:** "The model doesn't predict just the last token — every position predicts its next token simultaneously. One sequence of length T = T training examples."
   - Modalities: Verbal explanation, concrete trace-through with a short sentence, code implementation.

4. **Explain: Cross-Entropy for Next-Token Prediction** (Gap resolution)
   - Brief recap: "You know cross-entropy from MNIST — same formula, 50K classes instead of 10."
   - The reshape trick: logits (B, T, vocab_size) -> (B*T, vocab_size), targets (B, T) -> (B*T). Then nn.CrossEntropyLoss works identically.
   - Expected initial loss: random predictions over 50K tokens -> -ln(1/50257) ~= 10.8. If initial loss is near this, the model is correctly initialized (uniform predictions).

5. **Check: Predict the initial loss**
   - Before running the first forward pass, ask: "If the untrained model assigns equal probability to all 50,257 tokens, what should the cross-entropy loss be?"
   - Student computes: -ln(1/50257) ~= 10.8
   - Run it and verify. If it's close, initialization is correct. If it's much higher, something is wrong.
   - This is the "parameter count as architecture verification" pattern from Lesson 1, applied to the training setup.

6. **Explain: The Training Loop** (Familiar territory + new additions)
   - Start with the bare loop: forward, loss, backward, step. "This is the same loop."
   - Add gradient clipping: one line — `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`. Explain: if the global gradient norm exceeds 1.0, scale all gradients down proportionally. Why transformers need it: attention can produce occasional large gradients, especially early in training. The "safety net" from 1.3.3, now deployed.
   - Add periodic text generation: every N steps, run model.generate() and print the output. This is how we monitor quality beyond the loss number.
   - Add periodic loss logging for plotting.

7. **Explain: Learning Rate Scheduling** (New concept 2)
   - **Problem first:** "Constant LR worked for MNIST. Why wouldn't it work here?"
   - **Negative example:** Show training with constant LR at different values. High constant LR (6e-4): training destabilizes early because random weights + large updates = chaos. Low constant LR (1e-5): trains but painfully slowly.
   - **The solution:** LR scheduling with two phases:
     - **Warmup** (first ~5-10% of training): Start LR near zero, linearly ramp up to peak LR. "Random weights are fragile. Start gentle."
     - **Cosine decay** (remaining 90-95%): Gradually decrease LR following a cosine curve from peak down to ~10% of peak. "As you get closer to a good solution, take smaller steps."
   - **Visual:** LR schedule plot with warmup ramp + cosine curve, annotated with the rationale for each phase.
   - **Implementation:** Manual lambda scheduler or `torch.optim.lr_scheduler.CosineAnnealingLR` with warmup. Show both approaches.
   - **Connection:** "Remember the Goldilocks zone from 1.1.5? LR scheduling is dynamic Goldilocks — the best LR changes as training progresses."

8. **Explore: Run the Training** (Notebook exercise)
   - Student runs the full training loop on TinyShakespeare (or similar small corpus).
   - Observe: loss curve descending (noisy but downward), gradient norms staying bounded, LR schedule following the planned curve.
   - Periodically generated text improving:
     - Step 0 (loss ~10.8): Random characters/tokens
     - Step 500 (loss ~4.0): Word-like fragments, some common words
     - Step 2000 (loss ~3.0): Recognizable English phrases, crude grammar
     - Step 5000 (loss ~2.5): Grammatical sentences, some Shakespeare-like patterns
     - Step 10000 (loss ~2.0): Coherent Shakespeare-like passages
   - This is the emotional climax of the lesson — and arguably the module.

9. **Check: Interpret the loss curve**
   - Show the full training loss curve. Ask:
     - "Why is the curve noisy?" (Different batches have different difficulty levels)
     - "What would it mean if loss suddenly spiked and didn't recover?" (Gradient explosion — clipping should prevent this, but extreme cases exist)
     - "The loss went from 10.8 to 2.0. Is that good? How do you know?" (Initial loss = random guessing; final loss should be compared to what's achievable for this dataset/model size)

10. **Elaborate: Training Dynamics Nuances**
    - **Loss noise is normal:** Language model loss is inherently noisier than MNIST because text difficulty varies wildly batch to batch. Use smoothed loss (running average) for trend, raw loss for health monitoring.
    - **The loss-to-quality mapping is nonlinear:** Address the misconception. Going from loss 6.0 to 4.0 is a dramatic qualitative leap (gibberish to word fragments). Going from 2.0 to 1.8 is subtle. Show the text samples again with this framing.
    - **When to stop:** For this lesson, we train for a fixed number of steps. In practice, you'd monitor validation loss (the student knows this from 1.3.6). But with a small dataset and a large model, overfitting is inevitable — the model will eventually memorize Shakespeare. That's OK for learning; we'll address data quality and scale in Lesson 3.

11. **Summarize: Key Takeaways**
    - "The training loop is the same loop. Always." (forward -> loss -> backward -> update)
    - Three new tools for transformer-scale training: text dataset preparation (input/target offset), LR scheduling (warmup + cosine decay), gradient clipping (one line, essential safety net).
    - Cross-entropy doesn't change with vocabulary size — same formula, more classes.
    - Loss-to-text-quality is nonlinear — the biggest qualitative leaps happen early.
    - The mental model: "Training a language model is training. The same training. At scale."

12. **Next step: Scaling & Efficiency**
    - "Your training loop works. But it's slow — training 124M parameters on a single GPU with float32 precision. What changes when you need to go faster? What changes when the model is 100x bigger? That's Lesson 3."
    - Forward reference to: GPU utilization, mixed precision, KV caching, flash attention, scaling laws.

### Widget Needed

**No dedicated interactive widget.** The notebook IS the interactive component. The student runs the training loop, watches loss decrease, reads generated text, and plots loss curves. The experiential learning happens in the notebook cells. A training-visualization widget would duplicate what the notebook already provides and add implementation overhead without pedagogical benefit.

However, the lesson page should include:
- A static or animated LR schedule diagram (can be a simple SVG or Recharts plot)
- A static loss curve with annotated text samples at different loss values
- The side-by-side MNIST vs GPT training loop code comparison

These are informational visuals, not interactive widgets.

### Exercise Design

**Notebook exercise (guided -> supported):**

The notebook is scaffolded in phases:
1. **Guided:** Load and tokenize the dataset (code provided, student runs it)
2. **Guided:** Create the Dataset class and DataLoader (code provided with explanations)
3. **Supported:** Build the training loop (skeleton provided, student fills in the loss computation and gradient clipping lines)
4. **Supported:** Implement LR scheduling (function signature and warmup phase provided, student implements cosine decay)
5. **Independent:** Run training, plot loss curves, generate text at checkpoints, interpret results
6. **Independent:** Experiment: try different peak LR values, try removing gradient clipping, observe what happens

---

## Checklists

### Prerequisite Audit
- [x] Every assumed concept listed with required depth
- [x] Each traced via the records (not the curriculum plan)
- [x] Depth match verified for each
- [x] No untaught concepts remain (all gaps have resolution plans)
- [x] No multi-concept jumps in widgets/exercises
- [x] All gaps have explicit resolution plans (cross-entropy vocab extension, AdamW aside, LR scheduling dedicated section, gradient clipping brief section)

### Pedagogical Design
- [x] Narrative motivation stated as a coherent paragraph (problem before solution)
- [x] At least 3 modalities planned for the core concept, each with rationale (verbal/analogy, visual, symbolic/code, concrete example, intuitive — 5 modalities)
- [x] At least 2 positive examples + 1 negative example, each with stated purpose (4 positive + 1 negative)
- [x] At least 3 misconceptions identified with negative examples (5 misconceptions)
- [x] Cognitive load <= 3 new concepts (exactly 3: dataset prep, LR scheduling, gradient clipping)
- [x] Every new concept connected to at least one existing concept
- [x] Scope boundaries explicitly stated

---

## Review — 2026-02-09 (Iteration 1/3)

### Summary
- Critical: 2
- Improvement: 5
- Polish: 3

### Verdict: MAJOR REVISION

Two critical findings must be resolved before this lesson is usable. The hook contains a code inconsistency that would undermine the core "same loop" mental model, and a planned misconception with its negative example was dropped entirely. The improvement findings address a missing validation loss evaluation, a weak gradient clipping section that lacks the motivating example the plan called for, and a few structural issues.

### Findings

#### [CRITICAL] — Hook training loop structure contradicts the real training loop

**Location:** Section 3 (The Same Heartbeat), GPT training loop code block
**Issue:** The hook's GPT training loop shows a nested structure: `for step in range(max_steps): for x, y in train_loader:`, implying every step iterates over the entire DataLoader. But the real training loop in Section 10 uses `x, y = next(iter(train_loader))` (sample a single batch per step). These two code snippets represent fundamentally different training strategies. The student's first exposure to the GPT loop is the hook, and they will carry that mental model forward. When they reach Section 10 and see a different pattern, they will be confused about which is correct, or worse, they will not notice the difference and form the wrong mental model.
**Student impact:** The core message of the hook is "same loop, two new lines." If the loop structure itself is inconsistent between the hook and the implementation, the student cannot trust either version. The "aha" moment is undermined.
**Suggested fix:** Make the hook's GPT loop match the actual implementation. Use the step-based iteration pattern (no nested `for x, y in train_loader`). Show `x, y = next(train_iter)` or simplify to pseudocode that abstracts the data loading. The MNIST loop can keep its epoch-based structure since it accurately represents what the student wrote before. The key comparison should highlight that both have the same forward/loss/backward/step core, even if the outer iteration differs.

#### [CRITICAL] — Misconception #3 (constant LR failure) lacks its planned negative example

**Location:** Section 8 (Learning Rate Scheduling)
**Issue:** The planning document specifies a negative example: "Run training with constant LR vs scheduled LR on the same data. Constant LR at 3e-4: trains OK but suboptimal. Constant LR at 6e-4: training becomes unstable early." The built lesson describes the problem in prose and uses a ComparisonRow (bullet points), but never shows the actual failure. There is no loss curve, no concrete numbers, no before/after comparison. The plan called this out as a key negative example and explicitly stated: "show the failure case first." The lesson tells the student constant LR fails but does not show it.
**Student impact:** Without seeing the failure, the student has no reason to believe that constant LR actually breaks. Their experience from Series 2 says constant LR works fine. A prose claim is not sufficient to override lived experience. The motivation for LR scheduling rests on this negative example.
**Suggested fix:** Add a concrete negative example. Options: (1) A static image or SVG showing two loss curves (constant LR high = diverges, constant LR low = slow, scheduled = good). (2) A callout box with actual loss values at step 100/500/1000 for each case. (3) In the notebook, have the student run a short experiment with constant LR before introducing the schedule. At minimum, the lesson page needs a visual comparison, not just bullet points.

#### [IMPROVEMENT] — No validation loss evaluation anywhere in the lesson

**Location:** Entire lesson (missing)
**Issue:** The planning document's outline (Section 9, "Check: Interpret the loss curve") asks "The loss went from 10.8 to 2.0. Is that good? How do you know?" and the scope includes "Interpret loss curves." The student knows about validation loss from Series 1 (Training Curves, 1.3.6, DEVELOPED). However, the lesson only logs training loss. The complete training loop (Section 10) has no `val_loader` evaluation. The loss curve section (Section 12) discusses curve shapes but never mentions validation loss or how to detect overfitting. The student has the "scissors pattern" mental model from 1.3.6 but is never prompted to apply it here. The plan's Elaborate section (point 3, "When to stop") explicitly mentions monitoring validation loss.
**Student impact:** The student will wonder whether they should be checking validation loss (they were taught to in Series 1) and may feel something is missing but not know what. This is a lost opportunity to reinforce a DEVELOPED concept.
**Suggested fix:** Add a brief validation evaluation step to the training loop code (every N steps, compute val loss) and add 1-2 sentences to Section 12 about monitoring train vs val loss. The plan acknowledges overfitting is inevitable with TinyShakespeare and says "that's OK for learning" — include that framing but still show the student how to check.

#### [IMPROVEMENT] — Gradient clipping section lacks the illustrative example from the plan

**Location:** Section 9 (Gradient Clipping)
**Issue:** The planning document includes a positive example: "Show gradient norm spiking during training (a specific batch causes a large gradient), and how clipping prevents the spike from corrupting learned weights." The built lesson explains clipping mechanically (how the math works, the one line of code, why transformers need it) but never shows a concrete case of gradient norms spiking and being clipped. The three-step formula box is symbolic, not experiential.
**Student impact:** The student understands the mechanism but has no visceral sense of when and how often clipping actually fires. It feels like a theoretical safeguard rather than a practical tool. The connection from the MENTIONED concept in 1.3.3 to practice is weaker than it should be.
**Suggested fix:** Add a small concrete example: "During training, you might see gradient norms like [0.8, 0.6, 0.9, 12.3, 0.7]. That 12.3 is an outlier that would cause a massive parameter update. Clipping scales it down to 1.0, preserving the direction but bounding the damage." This can be a simple callout box, not a full visualization.

#### [IMPROVEMENT] — Training loop uses `next(iter(train_loader))` anti-pattern

**Location:** Section 10 (The Complete Training Loop), line 752
**Issue:** The code uses `x, y = next(iter(train_loader))` inside the training loop. This creates a new iterator every step and always returns the first batch from the DataLoader (after shuffling). It does not iterate through the full dataset. The correct pattern for step-based training is to create the iterator once outside the loop and handle `StopIteration` to reset, or to use an infinite data iterator. As written, the student may not see all their training data.
**Student impact:** The student has used DataLoader with epoch-based loops before. This step-based pattern is new but the code as written is subtly wrong. If the student runs this code, they will repeatedly sample from the beginning of the shuffled dataset. They might not notice (training will still work because shuffle=True re-randomizes each time `iter()` is called), but it teaches a bad habit.
**Suggested fix:** Either (1) create the iterator outside the loop and reset it properly, or (2) use an infinite iterator pattern like `itertools.cycle`, or (3) acknowledge the simplification with a comment like `# For simplicity: each step creates a fresh random batch`. Option 3 is the easiest and honest about the pedagogical choice.

#### [IMPROVEMENT] — Misconception #5 (lower loss = better text) is addressed only in an aside

**Location:** Section 11, Row.Aside (InsightBlock "The Biggest Leaps Are Early")
**Issue:** The planning document identifies "Lower loss = better text" as a misconception and plans for it to be addressed in the generated text samples section with concrete text at different loss values. The lesson does show the text progression (excellent), but the explicit reframing of the nonlinear relationship is relegated to a sidebar InsightBlock. The main content just says "This is the payoff" without addressing the misconception directly. The aside is easily skippable.
**Student impact:** The student sees the text progression and naturally forms the conclusion "lower loss = proportionally better text." The nonlinear mapping is the misconception to preempt, but it is in a sidebar rather than the main flow. The student may never read it.
**Suggested fix:** Move the nonlinear loss-to-quality insight into the main content area, directly after the text progression boxes. A sentence like "Notice the pattern: the leap from 10.8 to 4.0 (random to words) is enormous. The leap from 2.5 to 2.0 is subtle. The relationship between loss and text quality is logarithmic, not linear." Keep the aside for additional color, but the core insight belongs in the main flow.

#### [IMPROVEMENT] — No checkpointing mentioned despite student having this at DEVELOPED

**Location:** Entire lesson (missing)
**Issue:** The planning document's connections table lists "Checkpointing during training" connecting to the checkpoint pattern from 2.3.1 (DEVELOPED). The plan also lists checkpointing in the scope ("save/resume pattern is known. Apply it here."). The built lesson has no mention of saving checkpoints. The student is training a model for 10,000 steps and has no way to save progress.
**Student impact:** The student knows how to checkpoint from Series 2 and would naturally expect to see it here. Its absence is a missed reinforcement opportunity and a practical gap (if training crashes at step 8000, they lose everything).
**Suggested fix:** Add 2-3 lines to the training loop (save checkpoint every N steps) and a brief aside noting this is the same pattern from Series 2. This also reinforces a DEVELOPED concept that has not been used in several lessons, fulfilling the Reinforcement Rule.

#### [POLISH] — `&apos;` entity in aside text reads awkwardly in source

**Location:** Section 10 aside (WarningBlock "Manual LR Update"), line 791
**Issue:** The code uses `param_group[&apos;lr&apos;]` which is technically correct HTML escaping for single quotes inside a JSX attribute, but it reads poorly in source and could produce rendering issues in some edge cases. Using backtick template strings or different quoting would be cleaner.
**Student impact:** None if rendering is correct. Minor source maintainability issue.
**Suggested fix:** Use `param_group['lr']` inside the code element or wrap in curly braces with a JS string.

#### [POLISH] — SVG LR schedule not responsive

**Location:** Section 8, the SVG visualization (lines 544-606)
**Issue:** The SVG has a fixed width of 500px and height of 200px. On narrow mobile viewports, this will overflow or be cut off. Other visuals in the course may use responsive patterns.
**Student impact:** On mobile, the student may not see the full LR schedule diagram, which is one of the key visuals for this lesson.
**Suggested fix:** Add `className="w-full max-w-[500px]"` and use a responsive viewBox pattern, or wrap in a div with `overflow-x-auto`.

#### [POLISH] — Concrete example box title says "The cat sat on" but shows 6 tokens including "the" and "mat"

**Location:** Section 4 (Dataset Preparation), the concrete example box (lines 244-263)
**Issue:** The section header references `"The cat sat on"` as the example sentence in the introduction paragraph but the actual trace-through uses 6 tokens: "The", "cat", "sat", "on", "the", "mat". The full sentence is "The cat sat on the mat" which produces 5 training examples from 6 tokens. This is fine content-wise, but the narrative says `"The cat sat on"` and then the box shows more tokens than that phrase contains. Minor mismatch.
**Student impact:** Negligible. The student would likely just read through it. But precision in examples matters.
**Suggested fix:** Change the introductory text from referencing `"The cat sat on"` to `"The cat sat on the mat"` to match the actual trace-through, or reduce the example to only use the 4 tokens mentioned.

### Review Notes

**What works well:**
- The hook (side-by-side MNIST vs GPT loop) is excellent in concept and delivers on the "same heartbeat" mental model. The structural comparison is immediately compelling.
- The text progression from gibberish to coherent Shakespeare (Section 11) is the emotional core and it lands well. The color-coded boxes with real-ish generated text are effective.
- The dataset preparation section (Section 4) with the concrete trace-through of input/target offsets is clear and well-structured.
- The LR schedule SVG visualization is a good addition and communicates the warmup + decay shape effectively.
- Scope boundaries are clearly stated and consistently enforced. The lesson does not drift.
- The cross-entropy gap resolution (Section 5) is clean and correctly minimal. "Same formula, more classes" is the right framing.
- The "Predict the Initial Loss" checkpoint (Section 6) is a strong pedagogical move that reinforces the sanity-check pattern from Lesson 1.

**Pattern concerns:**
- The lesson is prose-heavy for a STRETCH lesson. Most concepts are explained with words + code, which is fine, but the two key negative examples (constant LR failure, gradient norm spike) were planned as visuals/concrete data and were replaced with prose descriptions. For a student who needs to feel the problem before accepting the solution, showing is more persuasive than telling.
- The plan's "Check: Interpret the loss curve" section (outline item 9) was partially absorbed into the "Reading the Loss Curve" section but lost its interactive/questioning character. The plan framed it as asking the student specific diagnostic questions. The built version presents the information declaratively. Converting at least one GradientCard to a question format would preserve the active learning intent.

---

## Review — 2026-02-09 (Iteration 2/3)

### Summary
- Critical: 0
- Improvement: 2
- Polish: 2

### Verdict: NEEDS REVISION

All 10 findings from Iteration 1 were addressed. The two critical issues (hook loop inconsistency and missing constant LR failure example) are fully resolved. The five improvement issues (validation loss, gradient clipping example, iterator anti-pattern, misconception #5 placement, checkpointing) are all fixed. The three polish items are also fixed. Two new issues surfaced during this review.

### Findings

#### [IMPROVEMENT] — Validation evaluation loop uses `next(iter(val_loader))` anti-pattern

**Location:** Section 10 (The Complete Training Loop), line 836
**Issue:** The validation evaluation loop runs `for _ in range(val_steps): xv, yv = next(iter(val_loader))`. This creates a new iterator on every iteration of the inner loop, which means it always returns the first batch from `val_loader`. Since `val_loader` has `shuffle=False`, the validation loss is computed on the same 20 identical batches (actually the same first batch, 20 times). The train loader was correctly fixed to use `iter(cycle(train_loader))` but the val loader was not given the same treatment.
**Student impact:** The student will compute validation loss on a single repeated batch rather than sampling across the validation set. The reported val loss will be unrepresentative. If the student notices the pattern and applies their DataLoader knowledge, they will be confused about why the code does this. If they do not notice, they learn a subtly incorrect evaluation pattern.
**Suggested fix:** Create a `val_iter` outside the validation block or iterate directly: `for xv, yv in itertools.islice(val_loader, val_steps):`. Alternatively, use a simple `for xv, yv in val_loader:` loop with a `break` after `val_steps` batches. The key is to not recreate `iter()` on every step.

#### [IMPROVEMENT] — Loss curve section is entirely declarative, missing the planned diagnostic questions

**Location:** Section 12 (Reading the Loss Curve)
**Issue:** The planning document's outline item 9 ("Check: Interpret the loss curve") frames this section as active learning with specific questions for the student: "Why is the curve noisy?", "What would it mean if loss suddenly spiked and didn't recover?", "The loss went from 10.8 to 2.0. Is that good? How do you know?" The iteration 1 review noted this as a pattern concern. The built lesson presents all four loss curve concepts as GradientCards with declarative explanations. The student reads the answers without engaging with the questions. This was noted in the iteration 1 review notes but not formally classified as a finding. It should be, because it undermines the active learning intent of the plan.
**Student impact:** The student passively absorbs "noisy curves are normal" and "plateaus mean low LR" without being prompted to reason about these diagnostics themselves. When they encounter an unusual loss curve in the notebook, they will have read about what to look for but not practiced the diagnostic thinking. Converting even one or two of the GradientCards into a question-first format (ask, then reveal) would engage the student's reasoning.
**Suggested fix:** Convert the first GradientCard ("Normal: The curve is noisy") and the third ("Warning: Loss spikes and does not recover") into question format. For example: "Your loss curve looks jagged with lots of up-and-down variation. Is something wrong?" followed by the explanation. This preserves the diagnostic exercise intent from the plan. The other two cards can remain declarative.

#### [POLISH] — Em-dashes with spaces in text milestone labels

**Location:** Section 11 (Watching the Model Learn), lines 920, 928, 936, 944, 952
**Issue:** The milestone labels use `Step 0 &mdash; Loss ~10.8` with spaces on both sides of the em-dash. The writing style rule requires no spaces around em-dashes: `word—word` not `word — word`. While these are label-like text rather than flowing prose, they are rendered content that the student reads. Consistency with the style rule throughout the lesson is preferred.
**Student impact:** None functionally. Minor inconsistency with the rest of the lesson's em-dash usage.
**Suggested fix:** Change to `Step 0—Loss ~10.8` or, if the spacing looks too tight for label readability, use an en-dash with spaces (`Step 0 &ndash; Loss ~10.8`) which is the conventional punctuation for ranges/separators. Alternatively, use a pipe or colon separator: `Step 0 | Loss ~10.8`.

#### [POLISH] — Python code comments use spaced em-dashes inconsistently

**Location:** Section 10 (The Complete Training Loop), lines 808, 812, 815, 822
**Issue:** The Python code comments use spaced em-dashes: `# Get batch — cycle restarts...`, `# Forward — same as always`, etc. These are inside code strings rendered by CodeBlock, not lesson prose, so the writing style rule arguably does not apply. However, lines 443 (`# The transformer default — same API`) in Section 7 and the Python comments in Section 10 are the only places in the lesson using this pattern. If the student copies this code into their notebook, the dashes will render correctly in Python. This is a stylistic preference issue, not a correctness issue.
**Student impact:** None. The comments are clear and useful regardless of dash style.
**Suggested fix:** No action needed. Noting for completeness. Python comments commonly use spaced em-dashes for readability.

### Review Notes

**What was fixed well:**
- The hook loop inconsistency fix is clean. Both loops now show their actual iteration pattern: MNIST uses nested `for images, labels in train_loader`, GPT uses `next(train_iter)`. The comparison is honest and the "same heartbeat" message still lands.
- The constant LR failure example is now concrete and compelling. The three-column comparison with actual loss values at specific steps makes the failure visceral. The student can see that 6e-4 explodes at step 300, 1e-5 crawls to loss 4.3 at step 10000, and the schedule reaches loss 2.0. This is exactly what the plan called for.
- The `itertools.cycle` fix for the train iterator is the right pattern. The comment "cycle restarts when the dataset is exhausted" explains the behavior clearly.
- The gradient clipping concrete example (norm values 0.83, 0.61, 0.92, 12.3, 0.74) is effective and placed well. The highlighted spike line draws the eye.
- The validation loss integration is well-placed in the training loop and the scissors pattern reference in Section 12 ties back to Series 1 knowledge correctly.
- Checkpointing is cleanly added with a brief comment connecting to Series 2.
- The nonlinear loss-to-quality insight is now in the main content flow (lines 967-975) with the aside providing additional color. The core insight is no longer skippable.

**Remaining pattern concern:**
- The val_loader `next(iter(...))` issue is the same anti-pattern that was fixed for the train_loader but was introduced in the new validation code. This suggests the fix was applied narrowly rather than as a general principle. The lesson should be consistent about iterator usage across both train and val loaders.

**Overall assessment:**
The lesson is in strong shape. The iteration 1 fixes addressed every finding correctly. The two remaining improvement findings are both fixable with small, targeted changes. The lesson's core pedagogical arc (same heartbeat, three new instruments, emotional payoff from text improvement) is effective and well-executed. After addressing the val_loader iterator and adding at least one diagnostic question to the loss curve section, this lesson should pass.

---

## Review — 2026-02-09 (Iteration 3/3)

### Summary
- Critical: 0
- Improvement: 0
- Polish: 1

### Verdict: PASS

All findings from Iterations 1 and 2 have been resolved. The two Iteration 2 improvement findings (val_loader iterator and diagnostic questions) are both correctly fixed. No new critical or improvement issues were found.

### Iteration 2 Fix Verification

**Val_loader iterator (was IMPROVEMENT):** FIXED. Line 834 creates `val_iter = iter(val_loader)` once before the inner loop, and `next(val_iter)` at line 837 correctly iterates through successive batches rather than recreating the iterator on each step. The pattern is consistent with the `iter(cycle(train_loader))` pattern used for the training iterator.

**Loss curve diagnostic questions (was IMPROVEMENT):** FIXED. GradientCards 1 and 3 in Section 12 now use question-first format: "Your loss curve looks jagged with lots of up-and-down variation. Is something wrong?" and "Your loss suddenly spiked at step 3000 and hasn't come back down. What happened?" This engages the student's diagnostic reasoning before presenting the answer, preserving the active learning intent from the plan.

**Em-dash spacing in milestone labels (was POLISH):** FIXED. Labels now use `Step 0&mdash;Loss ~10.8` with no spaces around the em-dash.

**Python code comment dashes (was POLISH):** Correctly assessed as acceptable in Iteration 2. No change needed.

### Findings

#### [POLISH] — Skull emoji in constant LR failure example

**Location:** Section 8 (Learning Rate Scheduling), line 530
**Issue:** The constant LR = 6e-4 failure case uses a skull-and-crossbones emoji (`\u2620\uFE0F`) after "Step 300: loss NaN." While it communicates the failure vividly, the project conventions discourage emoji usage unless explicitly requested. The rest of the lesson uses no emojis. This is the only instance.
**Student impact:** Negligible. The emoji adds visual emphasis that communicates "this is catastrophically bad" effectively. The student understands the point with or without it.
**Suggested fix:** Replace with a text indicator like "(exploded)" or "(diverged)", or leave it as a deliberate pedagogical choice. This is a style preference, not a pedagogical issue.

### Review Notes

**What works well (final assessment):**

- **The hook delivers.** The side-by-side MNIST vs GPT comparison is the lesson's most effective pedagogical move. Both loops are shown honestly (epoch-based vs step-based iteration) while the structural identity (forward, loss, backward, step) is undeniable. The "same heartbeat, two new instruments" frame is memorable.

- **The constant LR failure example is now compelling.** The three-column comparison with actual loss values at specific steps (6e-4 explodes at step 300, 1e-5 crawls to 4.3, schedule reaches 2.0) makes the motivation for LR scheduling visceral rather than theoretical. This was the plan's key negative example and it now delivers as intended.

- **The text progression is the emotional core of the module.** Five color-coded boxes showing generated text from gibberish to coherent Shakespeare, with the nonlinear loss-to-quality insight stated in the main flow. This is the moment the student has been building toward across the entire module.

- **The diagnostic question format in Section 12 works.** Converting two GradientCards to question-first format ("Is something wrong?" / "What happened?") engages the student's reasoning before revealing the answer. This preserves the plan's active learning intent.

- **All five misconceptions are addressed.** Each one is placed correctly in the lesson flow and has a concrete counter-example or reframing. The timing is right -- misconceptions are preempted before the student would form them.

- **The validation loop fix is clean.** Creating `val_iter` once before the inner loop is the correct pattern, consistent with the training iterator approach.

- **Scope boundaries hold.** The lesson does not drift into GPU optimization, hyperparameter search, or fine-tuning. The forward reference to Lesson 3 is clear and well-placed.

**Overall:** This lesson is ready to ship. The three-iteration review process caught and resolved 2 critical issues, 7 improvement issues, and 5 polish issues across the three passes. The final lesson is pedagogically sound, structurally honest, and emotionally effective. The student will leave understanding that training a language model is the same training they have done since Series 1, with three practical additions for transformer scale.

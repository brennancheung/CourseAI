# Module 4.4: Beyond Pretraining -- Record

**Goal:** The student can explain and apply the full post-pretraining pipeline -- adapting a pretrained language model for classification tasks, understanding how instruction tuning and alignment transform base models into assistants, and using parameter-efficient methods to make this practical on real hardware.
**Status:** Complete (5 of 5 lessons built)

## Concept Index

| Concept | Depth | Lesson | Notes |
|---------|-------|--------|-------|
| Adding a classification head to a pretrained language model (replace lm_head with nn.Linear(d_model, num_classes)) | DEVELOPED | finetuning-for-classification | Core pattern. Explicitly bridges CNN transfer learning (replace model.fc) to transformer transfer learning (replace lm_head). Includes breaking weight tying with embedding layer. |
| Extracting a sequence representation from a causal transformer via last-token hidden state (hidden_states[:, -1, :]) | DEVELOPED | finetuning-for-classification | The one genuinely new concept. Motivated by causal masking: last token has attended to all previous tokens, so it has full sequence context. Negative example: first token sees only itself = worst choice. BERT contrast: bidirectional model uses first token ([CLS]) because every position sees everything. |
| Frozen backbone finetuning for transformers (requires_grad=False on transformer parameters) | DEVELOPED | finetuning-for-classification | Same pattern as CNN feature extraction. Train only classification head (1,536 params for binary) while 124M backbone params frozen. Connected to requires_grad from Series 3. |
| Full finetuning vs frozen backbone tradeoffs for LLMs | INTRODUCED | finetuning-for-classification | ComparisonRow: frozen = fast/safe/low memory vs full = slow/risky/higher potential accuracy. Overfitting argument: 124M params on small dataset = capacity mismatch. Partial unfreezing with differential LR as middle ground. |
| Catastrophic forgetting in finetuned language models | INTRODUCED | finetuning-for-classification | Frozen backbone: model generates identical text before/after finetuning. Aggressive full finetuning on narrow task: model can lose general language abilities. Observable in notebook. |
| Transformer as general text feature extractor (same backbone supports multiple classification tasks by swapping the head) | INTRODUCED | finetuning-for-classification | Second positive example: swap nn.Linear(768, 2) to nn.Linear(768, 4) for topic classification. Only head dimensions and dataset change. |
| Instruction dataset format (instruction/response pairs as SFT training data) | DEVELOPED | instruction-tuning | Multiple concrete examples shown: factual question, creative task, coding task, summarization. Diversity of task types teaches the meta-pattern: instruction in, response out. Alpaca 52K and LIMA 1K as dataset size references. |
| Chat templates and special tokens as structural delimiters (<\|im_start\|>, <\|im_end\|>, role markers) | DEVELOPED | instruction-tuning | ChatML format shown in detail. Special tokens are real vocabulary entries with no pretrained meaning -- they acquire meaning entirely from SFT. Multiple template formats compared (ChatML, Llama, Alpaca). Wrong-template negative example shows degraded output. |
| Loss masking / prompt masking (label = -100 for instruction tokens, compute loss only on response tokens) | DEVELOPED | instruction-tuning | Inline SVG LossMaskingDiagram color-codes masked vs loss-contributing tokens. PyTorch's CrossEntropyLoss ignore_index=-100 convention. Motivated by efficiency: training signal should focus on response generation, not prompt prediction. |
| SFT teaches format, not knowledge (central insight) | DEVELOPED | instruction-tuning | The core misconception ("instruction-tuned models know more") defeated through "capital of France" dual-prompt evidence. Expert-in-monologue analogy. Reinforced by data efficiency argument (format is simpler than knowledge). |
| SFT training mechanics (same cross-entropy loss, same training loop, different data) | APPLIED | instruction-tuning | Side-by-side ComparisonRow: pretraining loop vs SFT loop -- 5 steps, 4 identical, step 1 differs (data source). No new head, no new loss function. Student runs SFT in notebook. |
| Classification finetuning vs SFT structural distinction (new head vs same lm_head, narrow task vs broad task) | DEVELOPED | instruction-tuning | ComparisonRow contrasting the two approaches. Classification changes WHAT the model outputs (class labels). SFT changes HOW the model uses its existing output mechanism (instruction-appropriate tokens). Explicitly notes "feature extractor" mental model from Lesson 1 does NOT apply to SFT. |
| SFT data efficiency (small datasets produce large behavioral changes because format is simpler than knowledge) | DEVELOPED | instruction-tuning | LIMA 1,000 examples result. Connected to classification finetuning data efficiency (1,536-param head learns from small dataset because backbone already extracts features). Same principle at behavior level. |
| The alignment problem: SFT-only models can be harmful, sycophantic, or confidently wrong despite following instructions | DEVELOPED | rlhf-and-alignment | Three concrete failure mode examples (lock-picking, flat Earth agreement, confident misrepresentation). Motivated by "SFT teaches format but has no signal for quality." |
| Human preference data format (comparison pairs: prompt + two responses + which is better) | DEVELOPED | rlhf-and-alignment | Quantum computing explanation preference pair. Relative signal (A > B) more reliable than absolute scoring. InstructGPT ~33K comparisons. Core insight: "I cannot define a perfect response, but I can tell you which of these two is better." |
| Reward model architecture (pretrained LM backbone + scalar output head trained on preference pairs) | INTRODUCED | rlhf-and-alignment | Same pattern as classification finetuning: backbone + head. Instead of class probabilities, outputs a single scalar quality score. RewardModelDiagram SVG. Explicit callback to finetuning-for-classification architecture. Concrete training trace: reward(preferred) - reward(dispreferred) pushes positive. |
| PPO for language models (generate-score-update loop with KL penalty) | INTRODUCED | rlhf-and-alignment | Three-step loop: generate response, score with reward model, update policy. RlhfLoopDiagram SVG. First time the training loop changes shape (breaks "same heartbeat" pattern). Two models involved (policy + reward model). |
| KL penalty as soft constraint preventing reward hacking (continuous version of "freeze the backbone") | INTRODUCED | rlhf-and-alignment | Objective: maximize reward minus KL divergence from SFT model. Connected to catastrophic forgetting from finetuning-for-classification. Editor-with-blind-spots analogy for reward hacking. ComparisonRow: reward hacking examples vs KL-constrained behavior. |
| Reward hacking (model exploits imperfections in learned reward model to achieve high scores without genuine quality) | INTRODUCED | rlhf-and-alignment | Negative example: excessive verbosity, confident-sounding filler, formatting tricks. Editor analogy extended: "An editor who only read articles from one genre might overly reward that genre's conventions." Motivates the KL penalty as essential, not optional. |
| DPO as preference optimization without a separate reward model | INTRODUCED | rlhf-and-alignment | Directly increases probability of preferred responses, decreases dispreferred. ComparisonRow: PPO pipeline (4 steps, two models) vs DPO pipeline (4 steps, one model). Partially restores familiar training loop shape. Comparable results on benchmarks (Llama 2 used PPO; Zephyr, Mistral used DPO). Implicit KL penalty built into formulation. |
| RLHF teaches what humans prefer, not what is objectively true (preference correlates with truth but is not identical) | INTRODUCED | rlhf-and-alignment | Seeded early in reward model section, fully elaborated in "Why Alignment Matters Beyond Safety." Sycophancy example: "You're right!" often preferred over correction. Alignment improves but does not perfect truthfulness. |
| Training memory breakdown (weights + gradients + optimizer states = ~12 bytes/param for mixed-precision Adam) | DEVELOPED | lora-and-quantization | Four PhaseCards with concrete arithmetic: bf16 weights (14 GB) + bf16 gradients (14 GB) + fp32 Adam momentum (28 GB) + fp32 Adam variance (28 GB) = ~84 GB for 7B model. Optimizer states dominate at two-thirds of total. Memory bar chart SVG. ComparisonRow: GPT-2 (~1.5 GB) vs Llama 2 7B (~84 GB). |
| The memory wall problem (full finetuning does not scale to real model sizes) | DEVELOPED | lora-and-quantization | Motivational hook: full finetuning 7B model requires ~84 GB minimum, exceeds A100 80 GB with activations, impossible on consumer RTX 4090 (24 GB). "Two problems, two solutions" framing: LoRA for training, quantization for inference. |
| Low-rank decomposition / matrix factorization (large matrix W approximated as B times A where B is m x r, A is r x n) | INTRODUCED | lora-and-quantization | Built from matrix multiplication the student already knows. Concrete 4x4 rank-1 example (16 entries captured by 8 numbers). GPT-2 scale: 768x768 = 589,824 params vs rank-8 decomposition = 12,288 params (48x reduction). No SVD or eigenvalues. "Low-rank means the rows are not all independent -- they lie in a small subspace." |
| LoRA: Low-Rank Adaptation (freeze base weights, add trainable low-rank bypass matrices B and A) | DEVELOPED | lora-and-quantization | Architecture: h = Wx + BAx * (alpha/r). B initialized to zeros (starts identical to pretrained model). A from random normal. LoRA bypass SVG diagram (highway + detour). Minimal LoRALinear PyTorch class. Applied to W_Q and W_V attention projections. Rank r=4,8,16 as hyperparameter. Alpha scaling factor. Merge at inference: W_merged = W + BA*(alpha/r), zero additional inference cost. |
| Why LoRA works: finetuning weight changes are low-rank because finetuning is a refinement, not a revolution | DEVELOPED | lora-and-quantization | Hu et al. (2021) results: LoRA matches or exceeds full finetuning (GPT-3 175B SST-2: 95.1% vs 95.2%). Low-rank constraint as implicit regularization. ComparisonRow: when LoRA excels (classification, SFT, domain adaptation) vs when it may underperform (new language, radical domain shift). Connected to overfitting argument from finetuning-for-classification. |
| LoRA hyperparameters (rank r, alpha scaling, target modules W_Q and W_V) | DEVELOPED | lora-and-quantization | r=4,8,16 common values. Alpha often 2r or fixed at 16. Standard practice: apply to W_Q and W_V, not every layer. Misconception addressed: LoRA is NOT applied to every layer. Checkpoint question: rank-4 vs rank-64 overfitting on simple classification task. |
| Quantization (mapping floating-point weights to fixed-point integers for reduced memory) | DEVELOPED | lora-and-quantization | Bridge from bfloat16 (precision spectrum). Continuous float values mapped to discrete int8/int4 grid. Neural networks tolerate precision loss because weight distributions are approximately Gaussian. Quantization number line SVG diagram. |
| Absmax quantization (scale = max abs value / 127, q = round(w / scale)) | DEVELOPED | lora-and-quantization | Full worked example: w = [-0.8, 0.3, 1.2, -0.5] -> scale = 1.2/127 -> q = [-85, 32, 127, -53]. Four PhaseCards: find scale, quantize, store, dequantize. Reconstruction error shown to be tiny. |
| Absmax outlier problem (extreme values waste int8 range) | DEVELOPED | lora-and-quantization | Negative example: w = [-0.1, 0.05, 0.02, -0.03, 8.5]. Small values near zero all map to q near 0, lose information. Most int8 range wasted. Motivates zero-point quantization and GPTQ/AWQ. |
| Zero-point quantization (shifted range for asymmetric distributions) | INTRODUCED | lora-and-quantization | q = round(w/scale) + zero_point. Maps min to -128, max to 127. Better range utilization. One extra integer per group. Formula shown but not worked through in same detail as absmax. |
| QLoRA (quantized 4-bit base model + bfloat16 LoRA adapters) | INTRODUCED | lora-and-quantization | Combination of both techniques for different problems. Memory breakdown: 4-bit base (~3.5 GB) + LoRA adapters (~10-50 MB) + gradients (~10-50 MB) + optimizer states (~20-100 MB) = ~4 GB total. ~21x more memory-efficient than full finetuning. Memory comparison bar chart SVG. |
| GPTQ (post-training quantization with calibration dataset, layer-by-layer error compensation) | MENTIONED | lora-and-quantization | Name-drop level. Post-training quantization. Uses calibration dataset. Achieves INT4 with less than 1% perplexity degradation. |
| AWQ (Activation-Aware Weight Quantization, protects salient weights) | MENTIONED | lora-and-quantization | Name-drop level. Identifies important weights by activation magnitudes, keeps those at higher precision. |
| NF4 / NormalFloat4 (4-bit data type designed for normally-distributed weights) | MENTIONED | lora-and-quantization | Non-uniform quantization levels with more precision near zero. Used in QLoRA. Brief mention only. |
| KV cache quantization (quantize cached K,V tensors for long-sequence memory savings) | INTRODUCED | lora-and-quantization | Brief callback to KV caching from scaling-and-efficiency. KV cache can be quantized from float16 to int8, halving memory with minimal quality impact. Relevant for long sequences where KV cache dominates memory. |
| PEFT library (HuggingFace parameter-efficient finetuning) | INTRODUCED | lora-and-quantization | Used in notebook Exercise 4. LoRA finetuning as a few lines of code. Conceptual understanding from Exercise 3 makes the library transparent. |
| bitsandbytes library (quantized model loading for inference) | INTRODUCED | lora-and-quantization | Used in notebook Exercise 5. Load 4-bit or 8-bit quantized models. Compare memory, speed, and quality vs full precision. |
| The complete LLM pipeline as a sequential dependency chain (pretraining -> SFT -> alignment -> LoRA -> quantization) | DEVELOPED | putting-it-all-together | CONSOLIDATE synthesis. No new concept -- connects all prior stages into a single coherent pipeline. Each stage depends on the previous. FullPipelineDiagram SVG as capstone visual. |
| What each pipeline stage adds and cannot provide (knowledge, format, judgment, specialization, accessibility) | DEVELOPED | putting-it-all-together | CONSOLIDATE synthesis. One-word "adds" label per stage, unpacked into full explanation drawing on prior lessons. Dependency framing: each stage depends on the output of the previous stage. |
| Mapping pipeline stages to open-source model artifacts (base models, instruct models, LoRA adapters, quantized variants) | DEVELOPED | putting-it-all-together | CONSOLIDATE synthesis. Llama 3 family traced through pipeline. "Llama 3 70B Instruct, 4-bit GPTQ" decoded word by word. Addresses "base models are useless" misconception. |
| The adaptation spectrum: every method answers "how much should I change?" (no adaptation -> classification head -> LoRA/QLoRA -> SFT -> RLHF -> full finetuning) | DEVELOPED | putting-it-all-together | CONSOLIDATE synthesis. Extends the "frozen backbone -> KL penalty -> LoRA" spectrum from Lesson 4 to encompass all adaptation methods from the module. |

## Per-Lesson Summaries

### Lesson 1: Finetuning for Classification (BUILD)

**Concepts taught:**
- Adding a classification head to pretrained GPT-2 (DEVELOPED)
- Last-token hidden state as sequence representation for causal models (DEVELOPED)
- Frozen backbone finetuning for transformers (DEVELOPED)
- Frozen vs unfrozen comparison and tradeoffs (INTRODUCED)
- Catastrophic forgetting (INTRODUCED)
- Transformer as general text feature extractor (INTRODUCED)

**Mental models established:**
- "A pretrained transformer is a text feature extractor" -- extension of CNN transfer learning mental model. Add a head, freeze the backbone, train the head.
- "Of course you use the last token" -- causal masking means the last position has full sequence context. Architecture dictates the choice, not convention.

**Key analogies used:**
- CNN transfer learning side-by-side comparison (ResNet freeze+replace fc vs GPT-2 freeze+replace lm_head). ComparisonRow shows the steps are nearly word-for-word identical.
- "Hire experienced, train specific" callback from Series 3 transfer learning.
- "Same heartbeat, new instruments" callback for the training loop structure.

**How concepts were taught:**
- **Classification head:** Code-first with GPT2ForClassification class. Worked example tracing "This movie was terrible" through tokenization -> forward pass -> last token selection -> linear layer -> logits with concrete shapes at each step.
- **Last-token selection:** Inline SVG causal attention diagram showing per-position context count. Green highlighting on last row (5 tokens of context). Negative example: first token has 1 token of context. BERT bidirectional contrast.
- **Frozen backbone:** Code showing requires_grad=False loop + optimizer only on classifier params. Direct callback to CNN freezing pattern.
- **Frozen vs unfrozen:** ComparisonRow with concrete tradeoffs. Overfitting argument (124M params on thousands of examples = capacity mismatch). Partial unfreezing with differential LR mentioned.
- **Architecture diagram:** Inline SVG showing full pipeline: input tokens -> embeddings -> transformer blocks (labeled FROZEN) -> layer norm -> last token selection (labeled LAST TOKEN) -> nn.Linear (labeled NEW HEAD) -> class logits. Grayed-out dashed "lm_head (removed)" annotation.

**What is NOT covered:**
- Instruction tuning or SFT (Lesson 2)
- RLHF or alignment (Lesson 3)
- LoRA or parameter-efficient finetuning (Lesson 4)
- Token classification (NER, POS tagging)
- Prompt-based classification (zero-shot, few-shot)
- HuggingFace Trainer API
- Padding/batching variable-length sequences in depth

**Notebook:** `4-4-1-finetuning-for-classification.ipynb` -- Load pretrained GPT-2, tokenize SST-2, implement classification head, train with frozen backbone, evaluate accuracy, unfreeze last N blocks with differential LR, generate text before/after finetuning to observe catastrophic forgetting.

### Lesson 2: Instruction Tuning / SFT (STRETCH)

**Concepts taught:**
- Instruction dataset format (instruction/response pairs) (DEVELOPED)
- Chat templates and special tokens as structural delimiters (DEVELOPED)
- Loss masking / prompt masking (label = -100 for instruction tokens) (DEVELOPED)
- "SFT teaches format, not knowledge" -- the central insight (DEVELOPED)
- SFT training mechanics (same loop, same loss, different data) (APPLIED)
- Classification finetuning vs SFT structural distinction (DEVELOPED)
- SFT data efficiency (DEVELOPED)

**Mental models established:**
- "SFT teaches format, not knowledge" -- The base model is a brilliant expert who only speaks in monologue. SFT does not give the expert new knowledge; it teaches conversational form.
- "Same heartbeat" extended a third time -- pretraining, classification finetuning, and SFT all share the identical forward-loss-backward-step training loop. Only the data changes.

**Key analogies used:**
- Expert-in-monologue: brilliant expert who knows everything but cannot hold a conversation. SFT teaches them to answer questions instead of giving lectures.
- "Same heartbeat" callback from pretraining lesson -- the training loop is identical for the third time.
- "Prediction Checkpoint" (guess a new head, get told that guess is wrong) -- leverages and then corrects the classification finetuning mental model.

**How concepts were taught:**
- **Instruction datasets:** Code block showing 4 diverse JSONL examples (factual, creative, coding, summarization). Diversity emphasized as teaching the meta-pattern. Alpaca (52K) and LIMA (1K) as size references.
- **Chat templates:** ChatML format shown in detail with special tokens. Multiple formats compared (ChatML, Llama, Alpaca). Wrong-template negative example with concrete garbled output in Check 2 proves templates are functional, not cosmetic.
- **Loss masking:** Inline SVG LossMaskingDiagram with color-coded tokens (purple = masked/instruction, green = loss-contributing/response). Labels showing -100 vs target. Code block with simplified implementation using ignore_index=-100.
- **Central insight ("format not knowledge"):** "Capital of France" dual-prompt example. Base model completes "The capital of France is" correctly (has the knowledge) but treats "What is the capital of France?" as document continuation (lacks the behavior). GradientCard with expert analogy.
- **SFT mechanics:** ComparisonRow showing pretraining loop vs SFT loop side-by-side (5 steps, 4 identical). SftPipelineDiagram inline SVG showing pretraining -> SFT -> RLHF pipeline with "THIS LESSON" and "NEXT LESSON" labels.
- **Classification vs SFT:** ComparisonRow with 6 structural contrasts. Explicitly addresses that "feature extractor" mental model from Lesson 1 does not apply.
- **"Why so little data?":** Reframed using student's own three-training trajectory (pretraining -> classification -> instruction). "Of course the model learns to produce responses to instructions. Same mechanism every time."

**What is NOT covered:**
- RLHF, DPO, or preference-based alignment (Lesson 3)
- LoRA or parameter-efficient finetuning (Lesson 4)
- Building production-quality instruction datasets
- Multi-turn conversation handling in depth
- Constitutional AI or RLAIF (Series 5)
- Evaluation of instruction-tuned models (beyond qualitative observation)
- Prompt engineering or in-context learning

**Notebook:** `4-4-2-instruction-tuning.ipynb` -- Explore an instruction dataset (Alpaca format), implement chat template formatting with special tokens, tokenize formatted examples, implement loss masking (labels = -100 for prompt tokens), run SFT training for a small number of steps, compare base vs instruction-tuned model on several prompts to observe the behavioral shift.

### Lesson 3: RLHF & Alignment (BUILD)

**Concepts taught:**
- The alignment problem: why SFT alone is insufficient (DEVELOPED)
- Human preference data format: comparison pairs, not absolute scores (DEVELOPED)
- Reward models: pretrained LM + scalar head, trained on preferences (INTRODUCED)
- PPO for language models: generate-score-update loop (INTRODUCED)
- KL penalty as soft constraint preventing reward hacking (INTRODUCED)
- Reward hacking as failure mode of unconstrained optimization (INTRODUCED)
- DPO: same goal, no separate reward model (INTRODUCED)
- RLHF teaches preference, not truth (INTRODUCED)

**Mental models established:**
- "SFT gives the model a voice; alignment gives it judgment" -- Mute (base) to speaking (SFT) to speaking wisely (aligned). Each stage adds something essential the previous stage could not provide.
- "For the first time, the training loop changes shape" -- PPO breaks the "same heartbeat" pattern established across pretraining, classification finetuning, and SFT. Generate-score-update with two models, operating at response level instead of token level.
- "The reward model is an experienced editor" -- Does not write the article but can tell you which draft is better. Learned judgment from thousands of comparisons, not from rules. Has blind spots that can be exploited.
- "KL penalty is the continuous version of 'freeze the backbone'" -- Same purpose as frozen backbone (prevent catastrophic forgetting of previous stage's learning), but soft rather than binary.

**Key analogies used:**
- Voice vs judgment (pretraining = knowledge, SFT = voice, alignment = judgment).
- Experienced editor (reward model as learned judge, not rule-based scorer).
- Editor with blind spots (reward hacking: the model finds patterns the reward model over-rewards, just as a clever writer could game an editor's stylistic preferences).
- Continuous version of freeze-the-backbone (KL penalty as soft constraint, connected to catastrophic forgetting from Lesson 1).

**How concepts were taught:**
- **Alignment problem:** Three PhaseCards with concrete SFT failure modes: harmful helpfulness (lock-picking instructions), sycophancy (flat Earth agreement), confident incorrectness (misrepresented paper summary). The common thread: format without quality signal.
- **Human preference data:** Side-by-side preference pair: quantum computing prompt, jargon-heavy Response A (dispreferred) vs age-appropriate Response B (preferred). Human label B > A. Relative signal easier than absolute scoring.
- **Reward model:** RewardModelDiagram inline SVG showing pretrained LM backbone + nn.Linear scalar head. Explicit callback to classification finetuning architecture from Lesson 1. Concrete training trace with scores 0.3 vs 0.7.
- **PPO loop:** RlhfLoopDiagram inline SVG showing three-step cycle (generate, score, update) with KL penalty callout and frozen SFT reference model. Three numbered PhaseCards explaining each step.
- **KL penalty / reward hacking:** ComparisonRow: reward hacking examples (verbosity, confident filler, formatting tricks) vs KL-constrained behavior. Editor-with-blind-spots analogy. Connected to catastrophic forgetting.
- **DPO:** ComparisonRow contrasting PPO pipeline (4 steps, two models, complex) vs DPO pipeline (4 steps, one model, closer to supervised learning). Concrete trace reusing the quantum computing preference pair. Results: comparable quality, widely adopted.
- **Preference vs truth:** Seeded in reward model section ("what humans prefer, not what is objectively true"), fully elaborated in "Why Alignment Matters Beyond Safety" section with sycophancy example.

**What is NOT covered:**
- Implementing RLHF or DPO in code (no notebook -- conceptual lesson)
- PPO algorithm details (clipping, value function, advantage estimation)
- RL formalism beyond minimum needed (policy = model behavior, reward = score)
- Constitutional AI or RLAIF (Series 5)
- Red teaming, adversarial evaluation, safety benchmarks
- Political/philosophical aspects of alignment
- Multi-objective alignment in depth

**Notebook:** None. Conceptual lesson. The implementation complexity of RLHF exceeds what can be done with GPT-2 in a Colab notebook. Focus on why and what, not how.

### Lesson 4: LoRA, Quantization & Inference (STRETCH)

**Concepts taught:**
- Training memory breakdown: weights + gradients + optimizer states (DEVELOPED)
- The memory wall problem: full finetuning does not scale (DEVELOPED)
- Low-rank decomposition / matrix factorization (INTRODUCED)
- LoRA: Low-Rank Adaptation architecture, forward pass, where to apply, parameter savings, merge at inference (DEVELOPED)
- Why LoRA works: low-rank constraint as implicit regularization (DEVELOPED)
- LoRA hyperparameters: rank r, alpha scaling, target modules (DEVELOPED)
- Quantization: mapping floats to integers (DEVELOPED)
- Absmax quantization with worked example (DEVELOPED)
- Absmax outlier problem -- negative example (DEVELOPED)
- Zero-point quantization (INTRODUCED)
- QLoRA: quantized base + LoRA adapters (INTRODUCED)
- GPTQ, AWQ (MENTIONED)
- NF4 / NormalFloat4 (MENTIONED)
- KV cache quantization (INTRODUCED)
- PEFT library, bitsandbytes library (INTRODUCED)

**Mental models established:**
- "LoRA is the surgical version of 'freeze the backbone'" -- Same philosophy as frozen backbone finetuning (preserve pretrained knowledge, adapt minimally), but applied inside the model with tiny detours alongside frozen weights, rather than only at the output. The "highway and detour" metaphor.
- "Finetuning is a refinement, not a revolution" -- Weight changes during finetuning are low-rank because you are adjusting, not rewriting. The adaptation lives in a small subspace. Of course the update is low-rank.
- "Frozen backbone -> KL penalty -> LoRA" spectrum -- Three approaches to the same challenge (adapt without forgetting), at increasing levels of surgical precision. Binary constraint, soft constraint, targeted bypass.
- "The precision spectrum continues" -- float32 -> bfloat16 -> int8 -> int4. Each step trades precision for memory, and neural networks tolerate it because weight distributions are compressible.

**Key analogies used:**
- Highway and detour (LoRA bypass): frozen W is the highway, B*A is a small trainable detour. Outputs are summed. The detour starts at zero.
- English and business emails: "You learned English over 20 years. Learning formal business emails does not rewrite your knowledge of English -- it adds a small adjustment." Low-rank finetuning intuition.
- "The classification head is tiny" callback: LoRA adapters are also tiny (~2% of full matrix), but distributed inside the model rather than added at the end.
- Precision spectrum callback: bfloat16 trade from scaling-and-efficiency extended to int8 and int4. "You already traded precision once and nothing broke."

**How concepts were taught:**
- **Memory wall:** Four PhaseCards with concrete per-component arithmetic (bf16 weights 14 GB + bf16 gradients 14 GB + fp32 Adam states 56 GB = ~84 GB). ComparisonRow: GPT-2 vs Llama 2 7B. InsightBlock: optimizer states dominate at two-thirds of total.
- **Low-rank decomposition:** Concrete 4x4 rank-1 matrix example where every row is a scaled version of [1,2,3,4]. Decomposed as b(4x1) * a(1x4). Then scaled to GPT-2 dimensions: 768x768 = 589,824 vs rank-8 = 12,288 (48x reduction). No SVD. "Of course" framing.
- **LoRA architecture:** Forward pass formula h = Wx + BAx*(alpha/r). LoRA bypass inline SVG diagram (highway + detour, sum at output). Minimal LoRALinear PyTorch class (~10 lines). B initialized to zeros so model starts identical to pretrained. Where to apply: W_Q and W_V. Merge at inference.
- **Why LoRA works:** Hu et al. GPT-3 175B SST-2 benchmark (95.1% vs 95.2%). Low-rank constraint as regularization. ComparisonRow: when LoRA excels vs when it may underperform. Checkpoint question: rank-4 outperforms rank-64 on simple task (overfitting).
- **Absmax quantization:** Full 4-step worked example with concrete numbers [-0.8, 0.3, 1.2, -0.5]. Four PhaseCards tracing scale, quantize, store, dequantize. Quantization number line SVG (float32 continuous -> int8 discrete grid with arrows).
- **Outlier problem:** Negative example with extreme outlier [values near 0, then 8.5]. Shows how absmax wastes range. Motivates zero-point and GPTQ/AWQ.
- **QLoRA:** PhaseCards showing 4-bit base (~3.5 GB) + LoRA adapters + gradients + optimizer states = ~4 GB. Memory comparison bar chart SVG (84 GB full finetune vs 16 GB LoRA vs 4 GB QLoRA vs 3.5 GB quantized inference). RTX 4090 24 GB reference line.

**What is NOT covered:**
- SVD, eigenvalues, or formal linear algebra beyond rank/decomposition intuition
- Implementing LoRA from scratch in the lesson (notebook uses PEFT library)
- Quantization-aware training (QAT) -- post-training quantization only
- Mixture of experts, pruning, distillation, or other efficiency techniques
- Detailed GPTQ/AWQ algorithms (name-drop level only)
- Production deployment or serving infrastructure
- Flash attention (covered in scaling-and-efficiency, Module 4.3)

**Notebook:** `4-4-4-lora-and-quantization.ipynb` -- Five exercises: (1) Memory calculator: compute memory for inference and training at different precisions (Guided). (2) Quantization by hand: apply absmax quantization step by step, compute reconstruction error, try with outliers (Guided). (3) LoRA from scratch: implement LoRALinear layer, verify frozen base, count trainable vs total params (Supported). (4) LoRA finetuning with PEFT: use HuggingFace PEFT library for practical LoRA finetuning (Supported). (5) Quantized inference: load quantized model with bitsandbytes, compare memory/speed/quality (Minimal Scaffolding).

### Lesson 5: Putting It All Together (CONSOLIDATE)

**Concepts taught:**
- No new concepts. This is a CONSOLIDATE lesson that synthesizes the entire 18-lesson series.
- The complete LLM pipeline as a sequential dependency chain (DEVELOPED -- synthesis)
- What each pipeline stage adds and cannot provide (DEVELOPED -- synthesis)
- Mapping pipeline stages to open-source model artifacts (DEVELOPED -- synthesis)
- The adaptation spectrum expanded to all module methods (DEVELOPED -- synthesis)

**Mental models established:**
- "Assembly, not invention" extended from the GPT architecture (Module 4.2) to the entire LLM pipeline -- each stage is a known technique, and the pipeline is their careful composition.
- "Every adaptation method answers the same question: how much should I change?" -- extends the "frozen backbone -> KL penalty -> LoRA" spectrum from Lesson 4 to encompass all adaptation methods: no adaptation, classification head, LoRA/QLoRA, SFT, RLHF/DPO, full finetuning.

**Key analogies used:**
- "Assembly, not invention" callback from Module 4.2 (causal-masking-and-gpt), extended from architecture to pipeline.
- One-word "adds" labels per pipeline stage: knowledge (pretraining), format (SFT), judgment (alignment), specialization (LoRA), accessibility (quantization).
- Callbacks to every established mental model: "format not knowledge," "voice then judgment," "highway and detour," "the precision spectrum continues."

**How concepts were taught:**
- **Complete pipeline:** FullPipelineDiagram inline SVG showing 8 stages from tokenization through quantization, color-coded by module, with "adds:" labels and module references. "Raw Text Corpus" visually differentiated as starting point (dashed border, italic), not a pipeline stage.
- **What each stage adds:** Five GradientCards (pretraining, SFT, alignment, LoRA, quantization), each with "Adds," "Cannot provide," and "Depends on" sections. WarningBlock on tradeoffs (heavy RLHF = over-cautious, aggressive quantization = quality loss).
- **Missing-stage check:** Three reveal-answer scenarios: missing SFT (text completion instead of answering), missing alignment (confident incorrectness + harmful helpfulness), missing quantization (80 GB memory requirement). Straightforward comprehension check.
- **Open-source ecosystem:** 2x2 grid of GradientCards mapping pipeline to real artifacts (base models, instruct models, LoRA adapters, quantized models). Llama 3 concrete example traced through pipeline. ComparisonRow: base models are not useless vs you do not start from scratch. InsightBlock: reading a HuggingFace model card -- every word maps to a pipeline stage.
- **Adaptation spectrum:** Six GradientCards ordered by amount of change (no adaptation, classification head, LoRA/QLoRA, SFT, RLHF/DPO, full finetuning). QLoRA explicitly included with concrete memory numbers (~4 GB for 7B).
- **Match the method check:** Three reveal-answer scenarios matching practical needs to adaptation methods (classification -> frozen backbone + head, chatbot -> SFT + RLHF, style adaptation -> LoRA/QLoRA).
- **What You Have Built:** Four module-level GradientCards summarizing the 18-lesson journey. Five key mental models echoed as final reinforcement.
- **Series 5 preview:** Three GradientCards (Constitutional AI, Reasoning Models, Multimodal) as previews, not lessons. Each connected to a prior concept the student already knows.

**What is NOT covered:**
- Any new concept, technique, or algorithm
- Implementation details (no code, no notebook)
- Comparing specific models (no Llama vs Mistral benchmarks)
- Production deployment, MLOps, or serving infrastructure
- Constitutional AI, reasoning models, multimodal (Series 5 previews only, one sentence each)

**Notebook:** None. CONSOLIDATE lesson -- synthesis only.

## Key Mental Models and Analogies

1. **"A pretrained transformer is a text feature extractor"** -- Same pattern as CNN transfer learning. The backbone extracts features; the head maps them to task outputs. Only the feature extractor changed (CNN -> transformer).
2. **"Of course you use the last token"** -- In a causal model, the last token is the only position with full sequence context. The architecture dictates this choice. (Contrasts with BERT's [CLS] at position 0, which works because BERT is bidirectional.)
3. **"The classification head is tiny"** -- 768 x 2 = 1,536 parameters for binary classification vs 124M total. Training 0.001% of the model. Overfitting the backbone on a small dataset is the real risk.
4. **"SFT teaches format, not knowledge"** -- The base model already has vast knowledge from pretraining. SFT on instruction-response pairs teaches it to express that knowledge in a conversational, instruction-following format. The expert-in-monologue analogy.
5. **"Same heartbeat, third time"** -- The training loop (forward, loss, backward, step) is identical across pretraining, classification finetuning, and SFT. Only the data changes. Extended from "same heartbeat, new instruments."
6. **"SFT gives the model a voice; alignment gives it judgment"** -- Mute (base) to speaking (SFT) to speaking wisely (aligned). Each stage adds something essential the previous stage could not provide.
7. **"For the first time, the training loop changes shape"** -- PPO breaks the "same heartbeat" pattern. Generate-score-update with two models at response level, not token level. DPO partially restores familiar loop shape.
8. **"The reward model is an experienced editor"** -- Learned judgment from human comparisons, not rules. Has blind spots that can be exploited (reward hacking).
9. **"KL penalty is the continuous version of 'freeze the backbone'"** -- Soft constraint preventing drift from SFT model. Same purpose as frozen backbone (catastrophic forgetting prevention), but gradient rather than binary.
10. **"LoRA is the surgical version of 'freeze the backbone'"** -- Same philosophy (preserve pretrained knowledge, adapt minimally) but applied inside the model with tiny trainable detours alongside frozen weights, rather than only adding a head at the output. Highway and detour metaphor.
11. **"Finetuning is a refinement, not a revolution"** -- Weight changes during finetuning are low-rank because you are adjusting, not rewriting. The adaptation lives in a small subspace. This is why LoRA works.
12. **"Frozen backbone -> KL penalty -> LoRA" spectrum** -- Three approaches to adapting without forgetting, at increasing surgical precision. Binary constraint (frozen backbone), soft constraint (KL penalty), targeted bypass (LoRA).
13. **"The precision spectrum continues"** -- float32 -> bfloat16 -> int8 -> int4. Each step trades precision for memory, and neural networks tolerate it because weight distributions are compressible.
14. **"The pipeline is assembly, not invention"** -- Extension of the GPT architecture principle to the entire LLM pipeline. Each stage (pretraining, SFT, alignment, LoRA, quantization) is a known technique; the pipeline is their careful composition. No stage can be skipped or reordered.
15. **"Every adaptation method answers the same question: how much should I change?"** -- Extension of the "frozen backbone -> KL penalty -> LoRA" spectrum to encompass all adaptation methods in the module: no adaptation, classification head, LoRA/QLoRA, SFT, RLHF/DPO, full finetuning. A spectrum from "change nothing" to "change everything."

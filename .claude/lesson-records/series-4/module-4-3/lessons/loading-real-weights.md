# Lesson: Loading Real Weights

**Module:** 4.3 (Building & Training GPT)
**Position:** Lesson 4 of 4 (Module Capstone)
**Slug:** `loading-real-weights`
**Type:** Hands-on (notebook: `4-3-3-loading-real-weights.ipynb`)
**Cognitive load:** CONSOLIDATE

---

## Phase 1: Orient -- Student State

The student has built a complete GPT model from scratch (building-nanogpt), trained it on TinyShakespeare (pretraining), and studied the engineering techniques that make transformers practical at scale (scaling-and-efficiency). They have generated text from their own trained model and watched quality improve from gibberish to recognizable Shakespeare. They have never loaded externally-trained weights into their own architecture. Their model uses the exact same class structure as GPT-2 (GPTConfig, Head/CausalSelfAttention, FeedForward, Block, GPT), but the internal naming conventions (e.g., `self.transformer.wte`, `self.transformer.h[i].attn.c_attn`) may or may not match OpenAI's published weight names. The student has done checkpointing (save/load state_dict) in Series 2 and pretraining, but always with models they trained themselves -- same code, same names, no mapping needed.

### Relevant Concepts with Depths

| Concept | Depth | Source | Notes |
|---------|-------|--------|-------|
| Complete GPT architecture in PyTorch (Head, CausalSelfAttention, FeedForward, Block, GPT) | APPLIED | building-nanogpt | Student wrote every class. Knows every layer name, every shape, every forward pass detail. This is the architecture they will load weights into. |
| GPTConfig dataclass for hyperparameter management | APPLIED | building-nanogpt | Debug config and GPT-2 config. Student knows how config maps to architecture dimensions. |
| Weight tying (embedding and output projection share weights) | DEVELOPED | building-nanogpt | `self.transformer.wte.weight = self.lm_head.weight`. Saves ~38M parameters. Student understands the reverse mapping. |
| GPT-2 parameter counting (~124.4M verified) | DEVELOPED | decoder-only-transformers, building-nanogpt | Per-component breakdown verified programmatically. Distribution: embeddings ~31%, attention ~23%, FFN ~46%. |
| Autoregressive generation (generate method) | DEVELOPED | building-nanogpt | Sample-append-repeat loop. torch.no_grad(), crop to block_size, temperature, sampling. Student has seen quality improve during training. |
| Checkpointing (torch.save / torch.load with state_dict) | APPLIED | pretraining, gpu-training (2.3.2) | Student has saved and loaded their own model checkpoints. Same architecture, same code, no name mapping required. |
| state_dict as an ordered dictionary of parameter name -> tensor | DEVELOPED | gpu-training (2.3.2) | Student knows model.state_dict() returns a dict with keys like "transformer.h.0.attn.c_attn.weight" and values are tensors. |
| nn.Module parameter naming hierarchy (nested modules produce dotted key paths) | DEVELOPED | building-nanogpt | Student's GPT uses nn.ModuleDict and nn.ModuleList, producing keys like "transformer.h.0.attn.c_attn.weight". They have printed parameter names during counting. |
| Cross-entropy for next-token prediction over 50K vocabulary | DEVELOPED | pretraining | Same formula, reshape (B, T, V) to (B*T, V). Initial loss sanity check: -ln(1/50257) ~= 10.82. |
| Temperature and sampling for generation | DEVELOPED | what-is-a-language-model, building-nanogpt | Student has used temperature parameter in generate(). Low temp = confident, high temp = creative. |
| Mixed precision / bfloat16 | DEVELOPED | scaling-and-efficiency | Student knows about precision formats, master weights pattern. Relevant when loading weights that may be in different dtypes. |
| Tiktoken tokenizer (GPT-2 BPE, 50257 vocab) | DEVELOPED | tokenization, pretraining | Student used tiktoken for TinyShakespeare training. Knows enc.encode() and enc.decode(). |
| Hugging Face transformers library | MENTIONED | (not directly taught) | Student may have encountered it but has not used it in this course. Will need a brief introduction for loading reference weights. |

### Established Mental Models

- "The formula IS the code" -- transformer block forward() directly implements the math
- "Five PyTorch operations build the entire GPT" -- nn.Linear, nn.Embedding, nn.LayerNorm, nn.GELU, nn.Dropout
- "Parameter count = architecture verification" -- one number confirms the entire structure
- "Same heartbeat, new instruments" -- training loop is structurally identical across all models
- "Untrained gibberish is a success" -- correct architecture producing random text means everything works
- "The math is elegant; the engineering makes it work" -- from scaling-and-efficiency

### What Was Explicitly NOT Covered That Is Relevant Here

- Loading weights from a different codebase into your own model (weight name mapping)
- Handling shape transpositions between different implementations (e.g., HuggingFace Conv1D vs nn.Linear)
- Using the HuggingFace transformers library to download pretrained models
- Verifying your implementation by comparing outputs against a reference model
- Generating coherent, high-quality text from a real pretrained model

### Readiness Assessment

The student is fully prepared. They have APPLIED depth on the GPT architecture (they wrote every class), DEVELOPED depth on weight tying and parameter counting (they understand where every parameter lives), and APPLIED depth on checkpointing (they have saved and loaded their own models). The only genuinely new skill is weight name mapping between their code and OpenAI's naming convention. Everything else -- loading state dicts, verifying shapes, generating text -- builds directly on existing skills. This is CONSOLIDATE: the novelty is low and the satisfaction is high.

---

## Phase 2: Analyze

### Target Concept

**This lesson teaches the student to load pretrained GPT-2 weights from OpenAI's published model into their own GPT architecture, verify that the loaded model produces correct outputs, and generate coherent text.**

### Prerequisites Table

| Concept | Required Depth | Actual Depth | Source Lesson | Status | Reasoning |
|---------|---------------|-------------|---------------|--------|-----------|
| Complete GPT architecture (all nn.Module classes) | APPLIED | APPLIED | building-nanogpt | OK | Student must know every layer name and shape in their own model to build the mapping. They wrote every class. |
| Weight tying (embedding shares with lm_head) | DEVELOPED | DEVELOPED | building-nanogpt | OK | Weight tying creates a special case during loading: two parameter names pointing to the same tensor. Student must understand this to handle it correctly. |
| state_dict (ordered dict of name -> tensor) | DEVELOPED | DEVELOPED | gpu-training (2.3.2), pretraining | OK | Student must be able to inspect, compare, and manipulate state dicts. They have used state_dict for checkpointing. |
| nn.Module parameter naming (dotted paths from nested modules) | DEVELOPED | DEVELOPED | building-nanogpt | OK | Student must understand how their class hierarchy produces parameter names (e.g., "transformer.h.0.attn.c_attn.weight"). They printed these during parameter counting. |
| GPT-2 parameter counting (124.4M) | DEVELOPED | DEVELOPED | decoder-only-transformers, building-nanogpt | OK | After loading, the student can verify the parameter count still matches. Serves as a sanity check. |
| Autoregressive generation (generate method) | DEVELOPED | DEVELOPED | building-nanogpt | OK | The payoff of the lesson: generate coherent text from the loaded model. Already implemented. |
| Tiktoken tokenizer | DEVELOPED | DEVELOPED | tokenization, pretraining | OK | Needed to encode prompts and decode generated tokens. Already used extensively. |
| Checkpointing (torch.save/load) | APPLIED | APPLIED | pretraining, gpu-training | OK | Provides the foundation: student knows how to save/load model state. This lesson extends to loading state from a different source. |
| Hugging Face transformers library | INTRODUCED | MENTIONED | (not taught) | GAP (small) | Student has not used HuggingFace in this course. Need a brief 2-3 paragraph introduction to downloading a pretrained model. No deep understanding needed -- just "here is how to get the reference weights." |

### Gap Resolution

| Gap | Size | Action |
|-----|------|--------|
| HuggingFace transformers library (MENTIONED -> INTRODUCED) | Small | Brief introduction (2-3 paragraphs) explaining: HuggingFace is the standard way to download pretrained models, `from transformers import GPT2LMHeadModel; model = GPT2LMHeadModel.from_pretrained("gpt2")` gives you the reference model. No deep dive into the library -- we use it only to obtain the weights and reference outputs. The student's own architecture is what matters. |

### Misconceptions Table

| Misconception | Why They'd Think This | Negative Example That Disproves It | Where to Address |
|---------------|----------------------|-----------------------------------|-----------------|
| "If my architecture is correct, loading pretrained weights should 'just work' with model.load_state_dict()" | In their own checkpointing experience, saving and loading always worked seamlessly because the same code produced both the model and the checkpoint. They have never encountered a name mismatch. | Print the keys from the HuggingFace GPT-2 state_dict alongside the keys from your own model. They do not match. HuggingFace uses names like "transformer.h.0.attn.c_attn.weight" while your model might use "transformer.h.0.attn.c_attn.weight" (sometimes matching) or different names entirely. Even if names happen to match, shapes may differ: HuggingFace's GPT-2 uses Conv1D (weight shape transposed relative to nn.Linear). load_state_dict() with strict=True will throw an error listing every mismatch. | Opening exploration -- print and compare the two state dicts side by side. |
| "Weight mapping is a tedious one-off task with no deeper lesson" | The mapping looks like bookkeeping -- match names from column A to column B. Why spend a lesson on it? | The mapping IS the verification. If your architecture has a bug -- wrong number of layers, wrong dimension, missing projection -- the weight mapping will fail because shapes do not match. A successful load is proof that every component in your architecture matches the original. Every shape mismatch is a bug report. | Frame in the narrative arc: the mapping is not bookkeeping, it is the ultimate architecture test. |
| "HuggingFace's GPT-2 uses the same nn.Linear convention as my model (weight shape is [out_features, in_features])" | nn.Linear stores weight as (out_features, in_features) -- this is the PyTorch convention the student has used throughout. Why would HuggingFace be different? | HuggingFace's GPT-2 implementation uses a custom Conv1D class that stores weight as (in_features, out_features) -- transposed relative to nn.Linear. Loading without transposing produces a model that runs without errors (the shapes broadcast or happen to be square for some layers) but generates garbage. The weight values are correct; the orientation is wrong. This is the most insidious type of bug: silent wrongness. | Weight mapping section, when handling linear layers. Show what happens without transposing: the model generates incoherent text despite having "loaded" the weights. |
| "If the model generates text after loading weights, the loading must be correct" | Text generation is a black-box test -- if output looks reasonable, everything is fine. The student's prior experience: untrained model produces gibberish, trained model produces coherent text, so coherent text = correct model. | Generate text from the model with one weight tensor transposed incorrectly. The text may look superficially plausible (real words, some grammar) but will be qualitatively worse than the correctly loaded model. Compare both outputs on the same prompt. Better verification: compare logits on a known input against the reference model. If logits match (within floating-point tolerance), the loading is correct. If they differ, something is wrong, even if the generated text looks "okay." | Verification section -- logit comparison before text generation. |
| "The weight-tied parameters (embedding and lm_head) need to be loaded twice into both locations" | Weight tying means two parameters share the same tensor. When loading from a checkpoint that stored them separately, the student might think they need to copy the same tensor into both places. | With weight tying, `self.transformer.wte.weight` and `self.lm_head.weight` are literally the same tensor object (same memory). Setting one sets the other. If the source checkpoint stores the embedding weight under one name, loading it into `transformer.wte.weight` automatically updates `lm_head.weight`. Loading it into both would be redundant (harmless but indicates a misunderstanding of what "tying" means). | Weight tying handling section. Show `self.transformer.wte.weight is self.lm_head.weight` returns True. |

### Examples Planned

| Example | Type | Purpose | Why This Example |
|---------|------|---------|-----------------|
| Side-by-side print of HuggingFace state_dict keys vs student model state_dict keys, with matching and non-matching keys highlighted | Positive | Makes the weight mapping problem concrete and immediate. Student sees the exact names they need to map. | This is "problem before solution" -- before writing any mapping code, the student sees the raw data. The names are strings they can read and compare. Some may match; some will not. The mismatch is the problem statement. |
| Loading weights without transposing Conv1D layers, then generating text -- output is incoherent despite no error | Negative | Demonstrates that silent shape transposition errors produce a model that runs but generates garbage. Disproves "if it runs, it's correct." | This is the most dangerous type of bug: no error, no crash, just wrong output. The student must feel this to understand why careful mapping matters. It also teaches a debugging skill: when output is wrong but no error occurs, check weight orientations. |
| Logit comparison: same input to student model (with loaded weights) vs HuggingFace reference model, verified with torch.allclose | Positive | Establishes the gold standard for verification: exact numerical match on logits, not just "looks reasonable" text comparison. | Logit comparison is definitive. If logits match within floating-point tolerance, the model is correct. If they differ, there is a bug. This is a more rigorous test than text comparison, which is subjective and stochastic (sampling adds randomness). |
| Generating coherent, multi-paragraph text from the correctly loaded model on several prompts | Positive (payoff) | The "I built GPT" moment. The student's own code, with real weights, generating genuinely coherent English text. The emotional closure for the entire module. | This is not primarily pedagogical -- it is motivational. The student has earned this moment across four lessons. The contrast with the gibberish from building-nanogpt's untrained model is the full arc of the module. |

---

## Phase 3: Design

### Narrative Arc

You built a GPT from scratch. You trained it on Shakespeare and watched gibberish become recognizable English. You learned what makes that process efficient at real scale. One question remains: is your implementation actually correct? Your TinyShakespeare model generates plausible-looking text, but "plausible-looking" is not the same as "correct." The real test is not training on a toy dataset -- it is loading the actual weights that OpenAI trained on billions of tokens and checking whether your model produces the same outputs as theirs. If your architecture has any bug -- a wrong dimension, a missing layer, a transposed weight -- the loading will fail or the outputs will diverge. This is the most satisfying kind of verification: not a unit test, not a shape assertion, but a living proof that your code implements the same model that powers ChatGPT's ancestor. And when it works, you get the payoff: your code, your classes, your forward pass, generating genuinely coherent English text. Not Shakespeare trained on a laptop -- real GPT-2, running on code you wrote from scratch.

### Modalities Planned

| Modality | What Specifically | Why This Modality for This Concept |
|----------|------------------|------------------------------------|
| **Symbolic (code)** | The weight mapping function translating HuggingFace parameter names to student model parameter names; the loading loop; the logit comparison; the generation code | This is a hands-on lesson. The core skill IS writing the mapping code. The student traces the correspondence between two sets of names and writes the translation. |
| **Concrete example** | Side-by-side state_dict key comparison (exact strings printed); logit comparison with torch.allclose (exact numbers); generated text before and after correct loading; generated text with one weight transposed incorrectly | Every claim is verified with concrete, printable evidence. The student can see the names, see the numbers, read the text. No abstract arguments about correctness -- concrete verification at every step. |
| **Visual** | Table showing the name mapping between HuggingFace keys and student model keys, organized by component (embeddings, attention, FFN, layer norm). Color-coded: green for direct matches, amber for name changes, red for shape transpositions. | The mapping is fundamentally a correspondence -- a visual table makes the pattern visible. The student can scan for patterns (e.g., all attention weights need transposing, all layer norms match directly) rather than handling each parameter individually. |
| **Intuitive** | "The mapping IS the test" -- the callback to parameter counting as architecture verification, extended to weight loading as architecture verification. If the shapes match and the outputs are correct, your implementation is right. | Connects to the existing mental model "parameter count = architecture verification." Loading real weights is a more powerful version of the same idea: instead of verifying one aggregate number, you verify every individual tensor shape. |

### Cognitive Load Assessment

- **New concepts in this lesson:** 1 genuinely new concept
  1. Weight name mapping between different codebases (translating parameter names, handling shape transpositions, dealing with weight tying during loading)
- **Reinforced/extended concepts:** 3
  1. state_dict inspection and manipulation (DEVELOPED -> APPLIED in a new context)
  2. Verification through numerical comparison (extends parameter counting verification)
  3. Autoregressive generation (DEVELOPED, used here as the payoff rather than taught)
- **Previous lesson load:** BUILD (scaling-and-efficiency)
- **This lesson load:** CONSOLIDATE -- appropriate. Only one new skill (weight mapping). Everything else is applying existing skills in a deeply satisfying context. After a BUILD lesson, CONSOLIDATE is the right closing move. The module ends on a high note rather than introducing more complexity.

### Connections to Prior Concepts

| New Concept | Prior Connection | How |
|-------------|-----------------|-----|
| Weight name mapping | nn.Module parameter naming (building-nanogpt) | The student already printed parameter names during parameter counting. They know the naming hierarchy. The new skill is comparing their names against a different codebase's names and building the translation. |
| Shape transposition handling | nn.Linear weight shape (Series 2, building-nanogpt) | The student knows nn.Linear stores (out_features, in_features). The new insight: not every library follows this convention. HuggingFace Conv1D stores (in_features, out_features). The fix is `.t()` -- a single transpose. |
| Logit comparison as verification | Parameter counting as architecture verification (building-nanogpt) | "Parameter count verifies the structure. Logit comparison verifies the computation. If both match, your implementation is correct." An escalation of the same verification strategy. |
| Loading external weights | Checkpointing (pretraining, gpu-training) | Checkpointing loads your own weights back into your own model. This lesson loads someone else's weights into your model. The mechanism (state_dict) is the same; the challenge is the name mapping. |

### Analogies from Prior Lessons That Might Be Misleading

- **"Parameter count = architecture verification"** -- This is mostly true but incomplete. A correct parameter count means the shapes are right in aggregate, but it does not catch weight transposition errors (a transposed weight has the same number of parameters). The lesson extends this: "parameter count verifies the shapes; logit comparison verifies the computation." The prior analogy is not wrong, just insufficient.

### Scope Boundaries

**This lesson IS about:**
- Loading GPT-2 (124M, "gpt2") pretrained weights into the student's own GPT class
- Building a weight name mapping between HuggingFace's naming convention and the student's naming convention
- Handling Conv1D vs nn.Linear weight transposition
- Handling weight tying during loading (embedding shared with lm_head)
- Verifying correctness by comparing logits against the HuggingFace reference model
- Generating coherent text from the correctly loaded model
- The emotional closure of the "I built GPT" arc

**This lesson is NOT about:**
- Training or fine-tuning the loaded model (Module 4.4)
- Loading different GPT-2 sizes (medium, large, XL) -- mentioned as a stretch exercise
- Understanding the HuggingFace transformers library in depth (used only as a weight source)
- Quantization or model compression (Module 4.4)
- Serving or deploying the model
- KV caching implementation during generation (covered conceptually in scaling-and-efficiency)
- Comparing performance against OpenAI's API
- Understanding what GPT-2 was trained on or how it was trained (the student already understands pretraining)

**Target depths:**
- Weight name mapping between codebases: APPLIED (student writes the mapping and uses it)
- Conv1D vs nn.Linear transposition: DEVELOPED (student understands why and handles it, but this is a one-time skill, not a deep concept)
- Logit comparison as verification: DEVELOPED (student performs it and understands why it is more rigorous than text comparison)
- HuggingFace transformers library: INTRODUCED (just enough to download and inspect the model)

### Lesson Outline

**1. Context + Constraints**
What this lesson covers (loading real GPT-2 weights into your architecture, verifying correctness, generating text) and what it does not (no training, no fine-tuning, no deployment, no other model sizes beyond a stretch exercise). Frame: "This is the final test. If real weights work in your model, every component you built is correct."

**2. Hook: "The Real Test"**
Type: Challenge preview + callback.
Callback to building-nanogpt: "Remember generating text from the untrained model? It was gibberish -- but it was YOUR gibberish, proof that the architecture runs." Callback to pretraining: "You trained on TinyShakespeare and the gibberish became recognizable English." Now: "But was your implementation truly correct, or did it just happen to produce something that looks okay on a small dataset?" The real test: load the weights that OpenAI trained on WebText (40GB of internet text, billions of tokens) and check whether your model produces the same outputs as their official implementation. If it does, your code is right -- not just "works on a toy dataset" right, but "implements the exact same model" right.

**3. Brief Introduction: Getting Reference Weights**
HuggingFace transformers library: `pip install transformers`, `from transformers import GPT2LMHeadModel`, `model_hf = GPT2LMHeadModel.from_pretrained("gpt2")`. This downloads the official GPT-2 weights. We are using HuggingFace only as a source of weights and a reference implementation -- the student's own model is the one we care about. Brief: 2-3 paragraphs + code, not a deep dive.

**4. Explore: Comparing State Dicts**
The "feel the problem" moment. Print both state dicts side by side:
- `model_hf.state_dict().keys()` -- HuggingFace's parameter names
- `model_ours.state_dict().keys()` -- Our parameter names

Observation 1: Some names match (or nearly match). Observation 2: Some names are different. Observation 3: Some shapes are different even when names seem to correspond. Observation 4: The number of keys is different (weight tying affects this).

The student sees the raw problem: these are not the same state dict. `model_ours.load_state_dict(model_hf.state_dict())` will fail. Let them try it and read the error message.

**5. Explain: Why the Names Don't Match**
Different codebases make different naming choices. HuggingFace's GPT-2 was implemented by different people than the student. The architecture is the same (same components, same shapes, same forward pass) but the code is organized differently. Parameter names come from the nn.Module hierarchy -- different class names and nesting produce different dotted paths.

The transposition issue: HuggingFace uses a custom Conv1D class (a historical artifact from the original OpenAI GPT-2 release) that stores weight tensors as (in_features, out_features) instead of nn.Linear's convention of (out_features, in_features). Same parameters, different storage layout. The fix: transpose (.t()) when copying from Conv1D to nn.Linear.

**6. Explain: Building the Weight Map**
Walk through the mapping systematically, component by component:
- **Embeddings:** Token embedding (`wte`) and position embedding (`wpe`) -- names and shapes typically match directly.
- **Attention:** Q, K, V projections and output projection. HuggingFace stores Q/K/V as a single combined weight (`c_attn.weight` of shape `[d_model, 3*d_model]` in Conv1D convention); the student's model may store them the same way (if using CausalSelfAttention) or separately. Shape transposition needed (Conv1D -> nn.Linear).
- **FFN:** Two linear layers (`c_fc` and `c_proj`). Shape transposition needed.
- **Layer Norm:** `ln_1`, `ln_2` per block, plus final `ln_f`. Weights and biases -- no transposition (LayerNorm parameters are 1D vectors, not matrices).
- **Weight tying:** HuggingFace stores the embedding weight and may or may not store a separate `lm_head.weight`. In our model, `lm_head.weight` IS `transformer.wte.weight` (same tensor). During loading, we load the embedding weight once; weight tying handles the rest.

Show the mapping as a table organized by component, with color coding: green (direct match), amber (name change only), red (name change + transpose).

**7. Check: Predict the Shape**
Before running the mapping code, prediction exercise: "HuggingFace's `transformer.h.0.attn.c_attn.weight` has shape `[768, 2304]` (Conv1D convention: in_features, out_features). What shape should the corresponding tensor have in your model, using nn.Linear convention?" Answer: `[2304, 768]` -- transposed. "What is 2304?" Answer: 3 * 768 = 3 * d_model, because Q, K, V are concatenated.

**8. Explore: Running the Load**
The student runs the mapping function in the notebook. For each parameter:
1. Look up the corresponding name in the HuggingFace state dict
2. If it is a Conv1D weight (2D, needs transposing), transpose it
3. Copy the tensor into the student's model

After loading, verify:
- Parameter count matches (same ~124.4M)
- No missing or unexpected keys
- All shapes match

**9. Negative Example: What Happens Without Transposing**
Load the weights but skip the transposition for one layer (e.g., the first attention block's c_attn weight). Generate text. The output is incoherent -- real words perhaps, but no meaning, no grammar, nothing like GPT-2. Compare with the correctly loaded model on the same prompt. The difference is stark.

The lesson: the model runs without errors. The shapes happen to work (768x2304 can multiply against either orientation of the input for some operations, or the combined QKV weight happens to be usable in either orientation because matrix multiplication is valid both ways with different semantics). But the computation is wrong. Silent failure. This is why logit comparison matters -- it catches errors that text comparison might miss.

**10. Explain: Logit Verification**
The gold standard: feed the same input tokens to both models (your loaded model and the HuggingFace reference) and compare the output logits.

```python
input_ids = torch.tensor([[...]])  # same tokens
with torch.no_grad():
    logits_ours = model_ours(input_ids)
    logits_hf = model_hf(input_ids).logits
print(torch.allclose(logits_ours, logits_hf, atol=1e-5))  # True
```

If True: your implementation is correct. Every component -- every projection, every layer norm, every residual connection -- produces the same output as the reference. This is stronger than parameter counting (which verifies shapes) and stronger than text comparison (which is stochastic and subjective).

Connection to prior verification: "In building-nanogpt, parameter counting verified your architecture had the right structure. Now, logit comparison verifies your architecture computes the right function. Together: right structure AND right computation."

**11. Check: What Could Cause Logit Mismatch?**
Transfer question: "Your logits are close but not exactly matching (allclose with atol=1e-5 returns False, but atol=1e-3 returns True). What could cause small numerical differences?" Possible answers: floating-point operation ordering (associativity), different implementations of the same function (e.g., fused vs unfused layer norm), different dtypes (float32 vs bfloat16). "Would these small differences affect text generation quality?" No -- the differences are at the level of floating-point noise. The top-k tokens and their relative probabilities would be essentially identical.

**12. Payoff: Generate Real Text**
The moment the entire module has been building toward. The student's own GPT class, with OpenAI's pretrained weights, generating coherent English text.

Multiple prompts to try:
- A straightforward prompt ("The meaning of life is")
- A factual prompt ("The capital of France is")
- A creative prompt ("Once upon a time in a land far away")
- A technical prompt ("The transformer architecture consists of")

The contrast with building-nanogpt's gibberish is the full arc: `random weights -> gibberish -> trained on Shakespeare -> recognizable English -> real GPT-2 weights -> coherent, knowledgeable text`. The student's code did not change. The weights changed. That is what pretraining buys.

**13. Elaborate: What the Weights Contain**
Brief reflection (not a new concept -- a connecting observation). The weights encode everything GPT-2 learned from WebText: grammar, facts, reasoning patterns, writing styles. When the student loaded those weights into their architecture, they transferred all of that knowledge into their code. The architecture is the vessel; the weights are the knowledge. This connects to the Module 4.4 theme: fine-tuning starts with these pretrained weights and adapts them to specific tasks.

**14. Summarize: What You Built**
Echo the full module arc:
- Lesson 1: Assembled the architecture. Generated gibberish. "The architecture works."
- Lesson 2: Trained on Shakespeare. Watched quality improve. "Training works."
- Lesson 3: Understood the engineering that makes it practical at scale.
- Lesson 4: Loaded real weights. Verified against OpenAI's implementation. Generated coherent text. "Your implementation is correct."

The verification chain: parameter count (right shapes) + logit comparison (right computation) + coherent generation (right behavior). Three levels of evidence, all confirmed.

Mental model echo: "The mapping IS the verification. Every shape match is a component verified. Every matching logit is a computation confirmed."

**15. Next Step**
"You have a working, verified GPT-2 implementation. In the next module, you will learn what comes after pretraining: taking these pretrained weights and adapting them for specific tasks through fine-tuning, instruction tuning, and alignment."

### Widget Decision

**No custom interactive widget needed.** This is a hands-on notebook lesson where the interactivity comes from the notebook itself -- the student runs the mapping, inspects state dicts, compares logits, and generates text. The lesson's visualizations are tables (weight name mapping, organized by component) and code outputs (printed state dict keys, logit comparison results, generated text). These are best rendered as static tables and code blocks in the lesson page, with the real exploration happening in the notebook.

Rationale: The core activity is mapping between two sets of string keys and handling tensor transpositions. This is fundamentally a code task, not a visualization task. An interactive widget would add complexity without pedagogical value. The notebook is the interactive element.

---

## Checklists

### Prerequisite Audit
- [x] Every assumed concept listed with required depth
- [x] Each traced via the records (not the curriculum plan)
- [x] Depth match verified for each
- [x] No untaught concepts remain (HuggingFace gap is small and has resolution plan)
- [x] No multi-concept jumps in widgets/exercises
- [x] All gaps have explicit resolution plans

### Pedagogical Design
- [x] Narrative motivation stated as a coherent paragraph (problem before solution)
- [x] At least 3 modalities planned for the core concept, each with rationale (4 modalities)
- [x] At least 2 positive examples + 1 negative example, each with stated purpose (3 positive + 1 negative)
- [x] At least 3 misconceptions identified with negative examples (5 misconceptions)
- [x] Cognitive load: 1 new concept (well within limit for CONSOLIDATE)
- [x] Every new concept connected to at least one existing concept
- [x] Scope boundaries explicitly stated

---

## Review — 2026-02-09 (Iteration 1/3)

### Summary
- Critical: 0
- Improvement: 4
- Polish: 3

### Verdict: NEEDS REVISION

No critical issues -- the lesson is structurally sound, the narrative arc is compelling, the ordering rules are respected, and the student would not get lost or form wrong mental models. However, four improvement-level findings would make the lesson significantly stronger with relatively modest changes.

### Findings

#### [IMPROVEMENT] — Misconception 5 ("loading tied weights twice") addressed but not with a concrete negative example

**Location:** Section 8 (Weight Tying During Loading)
**Issue:** The planning document identifies 5 misconceptions, each requiring a concrete negative example. Misconception 5 -- "The weight-tied parameters need to be loaded twice into both locations" -- is addressed in the prose and code comment ("Don't load it into both locations -- that's redundant (harmless but shows a misunderstanding)"). However, there is no concrete demonstration of *what happens* if you do load both. The plan called for showing `self.transformer.wte.weight is self.lm_head.weight` returning True. The code block does show this `is` check, which is good, but the misconception treatment is weaker than the other four. The student might still think "well, loading both is fine, just redundant" without understanding *why* it proves misunderstanding.
**Student impact:** The student understands weight tying conceptually but may not viscerally feel the "one tensor, two names" insight. This is minor because the concept was already DEVELOPED in building-nanogpt, but the lesson could reinforce it more concretely.
**Suggested fix:** Add a brief aside or inline note showing the identity check in a more prominent position -- e.g., a small "try this" prompt: "After loading, check: `model_ours.transformer.wte.weight.data_ptr() == model_ours.lm_head.weight.data_ptr()` -- same memory address." This makes the single-tensor reality concrete and memorable. Alternatively, show what `model_ours.state_dict()` contains (one entry or two?) after weight tying, making the state_dict key count discrepancy a concrete puzzle.

#### [IMPROVEMENT] — Modality count for the core concept is light on the visual/geometric side

**Location:** Entire lesson
**Issue:** The plan lists 4 modalities: symbolic/code, concrete example, visual (table), and intuitive. The built lesson delivers on symbolic/code (extensively), concrete examples (strong -- the transposition negative example and logit comparison are excellent), and intuitive ("the mapping IS the test"). The visual modality is present as the weight mapping table (section 7), the ComparisonRow (section 6), and the color-coded generated text comparisons (section 11). However, none of these are true *diagrams* -- they are text-formatted tables and styled boxes. For a lesson about mapping between two parallel structures (HuggingFace's module hierarchy vs the student's), a visual diagram showing the two architectures side by side with connecting lines (even a simple text diagram or inline SVG) would make the correspondence far more intuitive than a table alone. The table tells you *what* maps to *what*; a diagram would show *why* (the structural parallel).
**Student impact:** The student can follow the table, but it feels like a lookup reference rather than an insight. A visual of two parallel module trees with connecting lines would make the "same architecture, different names" insight feel obvious rather than requiring careful table reading.
**Suggested fix:** Add a simple inline SVG or a visual representation (even ASCII-art style) showing both module hierarchies side-by-side with arrows connecting corresponding parameters. Highlight which connections require transposition (red arrows) vs direct copy (green arrows). This would complement the existing table and add a genuinely distinct visual modality.

#### [IMPROVEMENT] — The "What Happens Without Transposing" section shows the result but does not walk through *why*

**Location:** Section 11 (What Happens Without Transposing)
**Issue:** The negative example is present and effective -- the student sees coherent vs incoherent text side by side. The prose explains that "the shapes happen to work" and "the attention scores are meaningless." But it does not walk through *why* the model still runs without error at a concrete level. The student is told "the combined QKV weight is used in a matrix multiplication that is valid in either orientation" but is not shown the actual shapes that make this work. Since the student has APPLIED depth on the GPT architecture and knows the shapes, a brief concrete trace ("c_attn.weight is [2304, 768] in your model. If you load the HuggingFace weight without transposing, you get [768, 2304]. The forward pass does `x @ weight.T`, which... still produces a tensor of the right output shape, just with scrambled values") would close the loop.
**Student impact:** The student accepts on faith that "it still runs." A 2-3 sentence shape trace would make this feel inevitable rather than surprising, reinforcing their shape-tracking skills from building-nanogpt.
**Suggested fix:** Add a brief shape trace after "The shapes happen to work" showing the specific matrix multiplication dimensions that produce valid-but-wrong outputs. This connects to the shape verification discipline from building-nanogpt and makes the silent failure *predictable* from the math rather than just *reported* as a fact.

#### [IMPROVEMENT] — Missing explicit treatment of Misconception 2 ("weight mapping is tedious bookkeeping")

**Location:** Throughout, but specifically sections 6-7
**Issue:** The planning document identifies Misconception 2: "Weight mapping is a tedious one-off task with no deeper lesson." The planned treatment was to frame the mapping as the ultimate architecture test. The lesson does this well in the asides ("The Mapping IS the Test" insight block in section 6, "Every Shape Match = Verification" in section 10) and in the summary. However, the misconception is addressed only through *positive framing* (telling the student it IS important) rather than through the planned *negative example* approach. A concrete negative example would be more powerful: show (or describe) what happens when an architecture has a bug -- e.g., "if your c_attn had `3 * d_model + 1` instead of `3 * d_model`, loading would fail at exactly that parameter with a shape mismatch of [768, 2304] vs [768, 2305]. The mapping catches what parameter counting might miss." This would disprove the misconception concretely rather than arguing against it rhetorically.
**Student impact:** The student hears "the mapping is important" but doesn't experience it catching a bug. The lesson shows what happens when the mapping is wrong (transposition), but not what happens when the *architecture* is wrong and the mapping catches it. These are different lessons: one about the mapping code, one about the mapping as a diagnostic tool.
**Suggested fix:** Add a brief paragraph (2-3 sentences) with a concrete hypothetical: "Imagine your FeedForward used `4 * d_model + 1` instead of `4 * d_model`. Parameter counting would still produce approximately 124.4M (off by a few thousand among millions). But the weight mapping would fail instantly: `AssertionError: shape [768, 3073] != shape [768, 3072]`. One number off, caught immediately. The mapping is a per-component X-ray, not an aggregate checksum."

#### [POLISH] — Section 8 (Weight Tying) has no SectionHeader

**Location:** Section 8 (Weight Tying During Loading)
**Issue:** Every other major content section has a `<SectionHeader>` component with a title and subtitle. Section 8 (weight tying) jumps directly into prose starting with a bold "Weight tying:" label. This creates a visual inconsistency -- the student experiences a jarring transition where the layout pattern breaks.
**Student impact:** Minor visual disruption. The student might not consciously notice, but the section feels less structured than the others.
**Suggested fix:** Add `<SectionHeader title="Weight Tying During Loading" subtitle="One tensor, two names" />` before the prose content.

#### [POLISH] — The `load_hf_weights` function iterates over `sd_ours` keys but copies into `sd_ours` dict, then calls `load_state_dict` -- redundant final step

**Location:** Section 10 (Running the Load), the `load_hf_weights` code block
**Issue:** The function copies data into `sd_ours` (which is a fresh dict from `model_ours.state_dict()`) using `sd_ours[key].copy_(sd_hf[key].t())`, which writes directly to the model's parameter tensors (since `state_dict()` returns references to the actual parameter data). Then it calls `model_ours.load_state_dict(sd_ours)` at the end, which is redundant -- the data is already in place. This is not *wrong* (it is a harmless redundancy), but a student who understands `state_dict()` deeply might be confused about why `load_state_dict()` is needed after in-place copies. Since this lesson is at CONSOLIDATE cognitive load, a brief comment clarifying this would help: either remove the `load_state_dict()` call (since `.copy_()` already modified the tensors in place) or add a comment explaining why it is included (e.g., "redundant but explicit -- makes the intent clear").
**Student impact:** A careful student might pause and wonder "wait, didn't `copy_()` already update the model?" This is a teaching opportunity about `state_dict()` returning references vs copies, but it is not addressed.
**Suggested fix:** Either (a) remove the `model_ours.load_state_dict(sd_ours)` line and the `return` line (the in-place copies are sufficient), or (b) add a comment explaining the choice. Option (a) is cleaner and avoids the confusion.

#### [POLISH] — Notebook filename inconsistency with lesson position

**Location:** Section 16 (notebook link) and doc header
**Issue:** The notebook is named `4-3-3-loading-real-weights.ipynb` but this is lesson 4 of 4 in Module 4.3. The numbering convention would suggest `4-3-4`. This appears to be because the scaling-and-efficiency lesson (lesson 3) had no notebook, so the notebook numbering follows the notebook count rather than the lesson count. This is not wrong -- it is a deliberate choice -- but it could confuse the student who looks at the lesson sequence (1, 2, 3, 4) and sees notebook numbers (1, 2, _, 3).
**Student impact:** Minimal. Most students would not notice or care. But if they look at the notebooks directory, the numbering gap might cause a moment of "did I miss one?"
**Suggested fix:** Either rename to `4-3-4-loading-real-weights.ipynb` for consistency with lesson position, or add a note in the notebook link section explaining the numbering. This is low priority.

### Review Notes

**What works well:**
- The narrative arc is excellent. The hook connects to building-nanogpt and pretraining with specific callbacks ("Remember generating text from the untrained model?"), creating a genuine emotional thread across the module.
- The problem-before-solution ordering is consistently maintained. The student tries `load_state_dict()` and sees it fail before learning about weight mapping. They see incoherent text before learning about logit verification. Every solution is preceded by a felt need.
- The negative example (section 11 -- incoherent text from skipped transposition) is one of the strongest in the curriculum. The side-by-side comparison with styled boxes makes the silent failure visceral and memorable.
- The checkpoint exercises are well-designed. "Predict the Shape" tests understanding of both the transposition convention and the combined QKV projection. "What Could Cause Logit Mismatch?" extends thinking to floating-point semantics.
- The summary and module completion blocks provide genuine closure. The three-level verification chain (parameter count, logit comparison, coherent generation) is a satisfying synthesis.
- The scope boundaries are crisp and consistently enforced -- the lesson never strays into fine-tuning, deployment, or HuggingFace depth.
- The cognitive load is appropriate for CONSOLIDATE: one genuinely new concept (weight mapping), everything else is application and verification of existing skills.

**Systemic observation:**
The lesson relies heavily on code blocks and prose, with tables as the primary "visual" element. While this is appropriate for a hands-on notebook lesson, adding even one small diagram would diversify the modalities meaningfully. The weight mapping table is effective but could be complemented by a structural diagram. This is the main area for improvement.

---

## Review — 2026-02-09 (Iteration 2/3)

### Summary
- Critical: 0
- Improvement: 0
- Polish: 1

### Verdict: PASS

All four iteration 1 improvement findings have been addressed effectively. No new critical or improvement issues emerged. The lesson is ready to ship.

### Iteration 1 Fix Verification

**[IMPROVEMENT] Misconception 5 (weight tying) -- RESOLVED.** Section 8 now has a second code block showing `data_ptr()` comparison and the 148 vs 147 key count discrepancy. The prose ("The key count difference is not a bug -- it is weight tying in action. One tensor, two names in the module hierarchy, but only one entry in the state dict because they share the same memory.") makes the single-tensor reality concrete. The misconception treatment is now on par with the other four.

**[IMPROVEMENT] Visual modality gap -- RESOLVED.** A full `WeightMappingDiagram` inline SVG component (lines 63-264) has been added, showing HuggingFace and student module hierarchies side-by-side with color-coded connecting arrows (green for direct copy, red for transpose required), `.t()` labels on red arrows, and a legend. This is a genuinely distinct visual modality that complements the weight mapping table. The structural parallel between the two codebases is now visible at a glance rather than requiring careful table reading.

**[IMPROVEMENT] "What Happens Without Transposing" lacks *why* -- RESOLVED.** Section 11 now includes a detailed shape trace (lines 1063-1079): "Your `c_attn` expects a weight of shape `[2304, 768]` (nn.Linear: out, in). The HuggingFace weight without transposing is `[768, 2304]`. But `copy_()` does not check semantics -- it copies raw values. Once the wrong-orientation values are in your `[2304, 768]` tensor, the forward pass computes `x @ W.T + b` as usual -- dimensions align, matmul succeeds, and a tensor of the correct output shape emerges. The values are simply meaningless because the rows and columns of the weight matrix have been swapped: features that should map to Q are mapping to K, and vice versa." This makes the silent failure predictable from the math rather than reported as a fact.

**[IMPROVEMENT] Misconception 2 (bookkeeping) -- RESOLVED.** Section 10 now includes the concrete hypothetical (lines 984-996): "Imagine your FeedForward used `4 * d_model + 1` instead of `4 * d_model`. Parameter counting would still produce approximately 124.4M (off by a few thousand among millions). But the weight mapping would fail instantly: `AssertionError: shape [768, 3073] != [768, 3072]`. One number off, caught immediately." This concretely disproves the misconception rather than arguing against it rhetorically.

**[POLISH] Section 8 SectionHeader -- RESOLVED.** Section 8 now has `<SectionHeader title="Weight Tying During Loading" subtitle="One tensor, two names" />`.

**[POLISH] Redundant `load_state_dict()` -- RESOLVED.** The `load_state_dict()` call has been removed and replaced with a clear comment: "No need for model_ours.load_state_dict(sd_ours) here: sd_ours holds references to the model's actual parameter tensors, so copy_() already modified the model's weights in place."

**[POLISH] Notebook filename (4-3-3 vs 4-3-4) -- DEFERRED (intentional).** This remains as-is. The numbering follows notebook count rather than lesson count, which is a deliberate convention since scaling-and-efficiency has no notebook.

### Findings

#### [POLISH] — Section 15 ("What the Weights Contain") has no SectionHeader

**Location:** Lines 1309-1329 (the "What the Weights Contain" section)
**Issue:** This section is a brief connecting observation between the text generation payoff and the summary. Unlike other content sections, it has no `<SectionHeader>`. While this can be read as an intentional continuation of the "Generate Real Text" section, it introduces a new conceptual point (the weights encode knowledge, connecting to fine-tuning in Module 4.4) that arguably warrants its own section header for consistency with the rest of the lesson.
**Student impact:** Minimal. The section reads naturally as a continuation. Some students might feel a slight transition gap, but the content flows logically from what precedes it.
**Suggested fix:** Optionally add a lightweight SectionHeader or leave as-is. This is a genuinely borderline call -- the section is short (2 paragraphs) and transitional, and adding a header could make it feel over-formalized. No fix required.

### Review Notes

**What works well (iteration 2 additions):**
- The `WeightMappingDiagram` SVG is well-executed. The column headers ("HuggingFace GPT-2 (Conv1D)" and "Your GPT (nn.Linear)"), indentation reflecting module hierarchy, color-coded arrows, `.t()` labels on transpose connections, and the three-item legend (direct copy, transpose required, skipped/weight tying) combine to create a visualization that communicates the full mapping at a glance. This was the biggest gap from iteration 1 and the fix is strong.
- The shape trace in section 11 (explaining *why* the model runs without error when transposition is skipped) is excellent. It walks through the exact matrix multiplication dimensions (`copy_()` ignoring semantics, `x @ W.T + b` producing valid-shape but meaningless output, Q features mapping to K positions) and transforms the negative example from "trust me, it breaks silently" to "of course it breaks silently -- trace the shapes and you can predict it."
- The `data_ptr()` check and key count discrepancy (148 vs 147) in the weight tying section make an abstract concept ("same tensor, two names") into something the student can verify in one line of code. This is the kind of concrete grounding that makes concepts stick.
- The hypothetical architecture bug in section 10 (`3073 != 3072`) effectively reframes the weight mapping from bookkeeping to diagnostic tool. It extends the "parameter count = verification" mental model from building-nanogpt by showing that the weight mapping is *finer-grained* verification (per-component vs aggregate).

**Overall assessment:**
The lesson is pedagogically sound across all seven review dimensions. The narrative arc is compelling (hook connects to three prior lessons, builds through problem -> mapping -> verification -> payoff). The ordering rules are consistently respected (problem before solution at every turn). The modality count is now genuinely strong (code, concrete examples, visual diagram, table, intuitive insight). All five planned misconceptions are addressed with concrete demonstrations. The cognitive load is appropriate for CONSOLIDATE. The module capstone function is fulfilled -- the student ends with verified, working GPT-2 running on code they wrote from scratch.

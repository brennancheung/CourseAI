# Series 5: Recent LLM Advances -- Plan

**Status:** Complete (Modules 5.1, 5.2, 5.3)
**Prerequisites:** Series 4 (LLMs & Transformers), Series 1-3 (foundations, PyTorch, CNNs)

## Series Goal

Understand the key innovations that transformed base LLMs into the capable, reasoning, multimodal systems deployed today -- and the research frontiers that may reshape them next. From constitutional AI through multimodal architectures to diffusion language models and subquadratic sequence processing, the student finishes able to read any current ML paper and understand the techniques being discussed.

## Pedagogical Approach

Series 4 built the complete transformer pipeline from scratch: pretrain, SFT, align. The student can implement GPT, understands attention deeply, and has intuitive mental models for RLHF, DPO, and reward hacking. Series 5 builds on top of that foundation in two ways:

1. **Deepening alignment:** Series 4 INTRODUCED RLHF/DPO at an intuitive level (no notebook, no implementation). Series 5 picks up where that left off -- constitutional AI replaces human annotators with AI, RLAIF scales preference data, and the student sees how alignment evolved from "ask thousands of humans" to "use AI to supervise AI." The mental model "the reward model is an experienced editor" extends to "what if the editor is also an AI?"

2. **Expanding capability frontiers:** The student understands the basic transformer. Now: how do you make it reason step-by-step? How do you feed it images alongside text? How do you make it 10x larger without 10x compute? Each module introduces a new capability frontier, grounded in the architecture the student already knows. The recurring pattern: identify a limitation of the current approach, feel why it matters, then see the solution.

**Conceptual, not implementation-heavy:** Unlike Series 4 (which built GPT from scratch), Series 5 is primarily conceptual. The innovations here involve training at scales the student cannot reproduce in a notebook. Where possible, lessons include notebooks that demonstrate concepts on smaller-scale proxies (chain-of-thought prompting, multimodal feature inspection, MoE routing visualization), but the emphasis is on understanding mechanisms and building mental models for reading papers, not on reimplementation.

**Durability over novelty:** The field moves fast. This series focuses on concepts that will remain relevant regardless of which specific model is state-of-the-art: the principle behind constitutional AI (AI-supervised alignment), the mechanism of chain-of-thought (decomposition as computation), the architecture of multimodal models (projection into shared embedding space), the economics of mixture of experts (conditional computation). Specific model names are examples, not the point.

## Modules and Lessons

### Module 5.1: Advanced Alignment (4 lessons)

Picks up directly from Series 4's RLHF lesson. The student has RLHF and DPO at INTRODUCED depth. This module develops alignment further: constitutional AI removes the human bottleneck, RLAIF scales it, red teaming tests it, and evaluation measures it.

| # | Slug | Title | Description |
|---|------|-------|-------------|
| 1 | constitutional-ai | Constitutional AI | What if the reward signal comes from AI instead of humans? Principles as prompts, critique-and-revision, RLAIF -- the "editor" from Series 4 becomes an AI editor |
| 2 | alignment-techniques-landscape | The Alignment Techniques Landscape | DPO variations (IPO, KTO, ORPO), online vs offline preference optimization, iterative RLHF -- mapping the design space beyond the PPO/DPO binary from Series 4 |
| 3 | red-teaming-and-adversarial-evaluation | Red Teaming & Adversarial Evaluation | How do you find what alignment missed? Automated red teaming, jailbreaks, the cat-and-mouse dynamic, why "safe" is never a finished state |
| 4 | evaluating-llms | Evaluating LLMs | Benchmarks, contamination, Goodhart's law for evaluation, human eval vs automated eval, why evaluation is harder than training |

### Module 5.2: Reasoning & In-Context Learning (4 lessons)

The student knows language models predict next tokens. This module explores the surprising capabilities that emerge from that simple objective: in-context learning, chain-of-thought reasoning, and the training techniques that amplify them.

| # | Slug | Title | Description |
|---|------|-------|-------------|
| 5 | in-context-learning | In-Context Learning | Zero-shot, few-shot, why examples in the prompt work without updating weights -- the transformer as a learning algorithm at inference time |
| 6 | prompt-engineering | Prompt Engineering | Systematic prompting techniques -- structured output, role prompting, retrieval-augmented generation (RAG) overview -- moving from "talk to the model" to "program the model" |
| 7 | chain-of-thought | Chain-of-Thought Reasoning | Why "let's think step by step" works: intermediate tokens as computation, the limits of single-forward-pass reasoning, process vs outcome supervision |
| 8 | reasoning-models | Reasoning Models | How models like o1 are trained to reason: reinforcement learning for reasoning, test-time compute scaling, search during inference, the shift from "bigger model" to "more thinking time" |

### Module 5.3: Scaling Architecture (3 lessons)

The student understands scaling laws from Series 4 (Chinchilla). This module covers architectural innovations that change the scaling equation: mixture of experts for conditional computation, long-context techniques, and the infrastructure that makes it all work.

| # | Slug | Title | Description |
|---|------|-------|-------------|
| 9 | mixture-of-experts | Mixture of Experts | Conditional computation -- not every parameter activates for every token. Router networks, expert specialization, why MoE models are "bigger but not slower" |
| 10 | long-context-and-efficient-attention | Long Context & Efficient Attention | RoPE in depth, context window extension techniques, sparse/linear attention variants, the memory-compute tradeoff at long sequence lengths |
| 11 | training-and-serving-at-scale | Training & Serving at Scale | Data parallelism, tensor parallelism, pipeline parallelism, speculative decoding, continuous batching -- the engineering that turns a research model into a product |

### Module 5.4: Multimodal Models (4 lessons)

The student knows transformers process token sequences and has seen CLIP in Series 6 (contrastive text-image embeddings). This module covers how language models gain the ability to see, and the architectural patterns that make it work.

| # | Slug | Title | Description |
|---|------|-------|-------------|
| 12 | vision-encoders-for-llms | Vision Encoders for LLMs | Vision Transformers (ViT), patch embeddings, how images become token sequences -- the same "everything is tokens" paradigm the student knows from text |
| 13 | vision-language-models | Vision-Language Models | Projection layers, visual tokens in the LLM context, architectures like LLaVA -- how a language model learns to "see" without changing the transformer |
| 14 | multimodal-training | Multimodal Training | Pretraining with interleaved text-image data, instruction tuning for visual QA, the data challenge of multimodal alignment |
| 15 | beyond-text-and-images | Beyond Text and Images | Audio (Whisper architecture), video, tool use and agents -- the pattern: encode any modality as tokens, project into the transformer's embedding space |

### Module 5.5: Research Frontiers (6 lessons)

The student now understands the production LLM stack. This module looks at what might replace or reshape it. Each lesson challenges an assumption the student has internalized: that generation must be autoregressive, that attention must be quadratic, that models are black boxes, that training is a one-shot process. The module doubles as a course capstone — after this, the student can read any current ML paper and orient themselves.

| # | Slug | Title | Description |
|---|------|-------|-------------|
| 16 | diffusion-language-models | Diffusion Language Models | The same diffusion process from Series 6, applied to text. Parallel generation, discrete vs continuous diffusion, MDLM/LLaDA, Gemini Diffusion -- what if autoregressive is not the only way? |
| 17 | subquadratic-architectures | Subquadratic Architectures | State space models from first principles, Mamba's selective mechanism, the attention-SSM duality (Mamba-2), and the theoretical impossibility result that proves purely subquadratic models have fundamental limits |
| 18 | hybrid-architectures-and-efficient-attention | Hybrid Architectures & Efficient Attention | Why hybrids win: ~90% SSM + ~10% attention (DeepSeek V3.2, Qwen3-Next). Multi-Head Latent Attention (MLA), Grouped Query Attention (GQA), KV cache compression -- what production frontier models actually use today |
| 19 | mechanistic-interpretability | Mechanistic Interpretability | Sparse autoencoders decompose superimposed representations into interpretable features. Superposition, feature decomposition, hallucination detection, model steering -- "opening the black box" |
| 20 | continual-learning | Continual Learning | Catastrophic forgetting revisited (callback to Series 4 finetuning). Parameter isolation, regularization approaches, Google's Nested Learning. Why "forgetting" is partly an illusion (spurious forgetting). The key barrier to models that learn from the world |
| 21 | open-frontiers | Open Frontiers | Model merging and weight surgery (why does averaging weights work?), Kolmogorov-Arnold Networks (learnable activation functions), world models (V-JEPA 2), and the unsolved problems. Where to keep learning after this course |

## Scope Boundaries

### In scope

- Constitutional AI and RLAIF (extending RLHF from Series 4)
- Post-DPO alignment techniques (IPO, KTO, ORPO -- landscape, not implementation)
- Red teaming and adversarial evaluation
- LLM evaluation and benchmarking
- In-context learning and prompt engineering
- Chain-of-thought reasoning and process supervision
- Reasoning models (o1-style test-time compute)
- Mixture of experts architecture
- Long-context techniques (RoPE, context extension)
- Training and serving infrastructure (parallelism, speculative decoding)
- Vision Transformers and multimodal LLMs
- Audio models (Whisper, at INTRODUCED depth)
- Tool use and agents (at INTRODUCED depth, as the "everything is tokens" pattern)
- Diffusion language models (MDLM, LLaDA, discrete diffusion for text)
- Subquadratic architectures (state space models, Mamba, SSM-attention duality)
- Hybrid architectures and efficient attention engineering (MLA, GQA, KV cache compression)
- Mechanistic interpretability and sparse autoencoders
- Continual learning (catastrophic forgetting solutions, nested learning)
- Model merging and weight surgery (at INTRODUCED depth)
- Kolmogorov-Arnold Networks (at INTRODUCED depth, as alternative architecture)
- World models (at MENTIONED depth, as future direction)

### Out of scope

- Implementing RLHF or DPO from scratch (Series 4 covered the concepts; this series extends them conceptually)
- Full RL formalism (kept intuitive as in Series 4)
- Specific model benchmarks or leaderboard positions (outdated before the lesson is finished)
- Political/philosophical alignment debates (focus on mechanisms, not policy)
- Diffusion model advances for images (Series 7)
- Code generation models in depth (covered as examples of SFT/RLHF application, not as their own topic)
- Retrieval-augmented generation in depth (INTRODUCED as a prompting technique, not a full module)
- Production MLOps, deployment pipelines, model serving frameworks
- Autonomous agents in depth (MENTIONED as frontier, not developed)
- Full state space theory derivation (intuition and duality, not mathematical proofs)
- Implementing SSMs or Mamba from scratch (conceptual understanding, not reimplementation)

## Connections to Prior Series

| Series 5 Concept | Earlier Concept | Source |
|-------------------|----------------|--------|
| Constitutional AI (AI as preference annotator) | Reward models, human preference data | Series 4.4 (rlhf-and-alignment) |
| RLAIF scaling | "The reward model is an experienced editor" analogy | Series 4.4 (rlhf-and-alignment) |
| DPO variations (IPO, KTO) | DPO as simpler alternative to PPO | Series 4.4 (rlhf-and-alignment) |
| KL penalty in alignment variations | "KL penalty is the continuous version of freeze the backbone" | Series 4.4 (rlhf-and-alignment) |
| In-context learning | Attention as data-dependent computation | Series 4.2 (attention mechanism) |
| Chain-of-thought as intermediate computation | Autoregressive generation feedback loop | Series 4.1 (what-is-a-language-model) |
| Process supervision for reasoning | Loss masking / training signal design | Series 4.4 (instruction-tuning) |
| Mixture of experts routing | Softmax for classification / gating | Series 1, 4.2 |
| MoE conditional computation | FFN as key-value memories, "attention reads, FFN writes" | Series 4.2 (transformer-block) |
| RoPE (positional encoding in depth) | Sinusoidal positional encoding, learned PE | Series 4.1 (embeddings-and-position) |
| Vision Transformer (ViT) patch embeddings | CNN feature extraction, token embeddings | Series 3, 4.1 |
| Multimodal projection layers | Embedding spaces, learned projections (Q/K/V) | Series 4.2 |
| CLIP as bridge (text-image embeddings) | CLIP lesson in Series 6 | Series 6.3 (clip) |
| Parallelism strategies | Compute-bound vs memory-bound, arithmetic intensity | Series 4.3 (scaling-and-efficiency) |
| Speculative decoding | Autoregressive generation bottleneck, KV caching | Series 4.3 |
| Diffusion LLMs (discrete diffusion for text) | Diffusion process (forward/reverse, noise schedules, DDPM) | Series 6.2 |
| Diffusion LLMs (parallel generation) | Autoregressive generation as feedback loop | Series 4.1 (what-is-a-language-model) |
| State space models (recurrence) | RNN concepts, sequential processing | Series 4.1 (implicit) |
| SSM-attention duality (Mamba-2) | Attention as data-dependent weights | Series 4.2 |
| Hybrid architectures (SSM + attention layers) | Transformer block as repeating unit | Series 4.2 (transformer-block) |
| Multi-Head Latent Attention (low-rank KV compression) | KV caching, attention head structure | Series 4.2, 4.3 |
| Sparse autoencoders | Autoencoders (encoder-decoder, bottleneck, reconstruction) | Series 6.1 (autoencoders) |
| Superposition in neural networks | Embedding spaces, learned representations | Series 4.1 (embeddings) |
| Continual learning / catastrophic forgetting | Catastrophic forgetting in finetuning | Series 4.4 (finetuning-for-classification) |
| Model merging (weight averaging) | Loss landscapes, optimization | Series 1.3 |
| KANs (learnable activation functions) | MLPs, activation functions (ReLU, GELU) | Series 1.2, 4.2 |

## Key Pedagogical Notes

- **RLHF depth upgrade:** Series 4 INTRODUCED RLHF, DPO, reward models, and reward hacking. Series 5 Module 5.1 builds on this foundation but does NOT re-teach RLHF from scratch. The Reinforcement Rule applies: if Series 4 Lessons 16-18 are more than 3 lessons back, Module 5.1 Lesson 1 should open with a brief recap connecting to established mental models ("the reward model is an experienced editor," "KL penalty is the continuous version of freeze the backbone") before extending to constitutional AI.

- **CLIP as cross-series bridge:** The student encounters CLIP in Series 6 Module 6.3 (contrastive text-image learning). Module 5.4 (multimodal) builds on the idea that text and images can share an embedding space. If Series 6 was completed before Series 5, this is REVIEW. If not, Module 5.4 needs to INTRODUCE CLIP. The module plan should check the student's actual state when the time comes.

- **Conceptual emphasis:** Most lessons in this series are conceptual (no notebook or lightweight notebook). The innovations covered operate at scales that cannot be reproduced on a laptop. Where notebooks exist, they demonstrate principles on smaller proxies: chain-of-thought prompting on GPT-2, MoE routing visualization, ViT patch embedding visualization. The student should not expect to train a reasoning model or a multimodal LLM -- they should expect to understand how they work.

- **Reasoning models are the fastest-moving topic.** The o1-style reasoning model lesson should focus on the durable insight (test-time compute scaling, RL for reasoning, search during inference) rather than specific model architectures that may be outdated. Frame as: "the paradigm shift from 'bigger model' to 'more thinking time'" rather than "how o1 works."

- **Module 5.3 (Scaling Architecture) builds on Series 4's scaling lesson.** Series 4 INTRODUCED scaling laws, mixed precision, KV caching, and flash attention. Module 5.3 extends this: MoE changes the parameter-compute relationship, long-context changes the memory equation, parallelism strategies change how you distribute training. The Chinchilla mental model ("scale both, not just one") should be explicitly revisited and extended.

- **Module ordering flexibility:** Modules 5.1-5.4 are relatively independent. A student could potentially do Module 5.2 (reasoning) before Module 5.1 (alignment) without significant gaps. However, the planned order reflects logical progression: alignment techniques first (directly extending Series 4), then reasoning capabilities, then architectural scaling, then multimodal expansion. Each module moves further from the Series 4 foundation into new territory.

- **Module 5.5 is the course capstone.** The final lesson ("Open Frontiers") serves as both a module capstone and a course capstone. It should leave the student with a sense of what's unsolved, what's exciting, and where to keep learning. The tone shifts from "here's how this works" to "here's what nobody has figured out yet."

- **Diffusion LLMs connect to Series 6.** The student built a pixel-space diffusion model in Series 6. Lesson 16 should explicitly activate that knowledge: "You know how diffusion works for images — forward process adds noise, reverse process removes it, the model predicts the noise. Now: what if the 'image' is a sequence of tokens?" The key new concept is discrete diffusion (masking tokens vs adding Gaussian noise). The connection to Series 6 should feel like a payoff, not a coincidence.

- **Subquadratic needs two lessons because the conceptual depth is real.** Lesson 17 covers the theory: what state space models are, why they're O(n), Mamba's selective mechanism, and the duality result showing SSMs and attention are two views of the same operation. Lesson 18 covers the practical reality: why pure subquadratic fails (theoretical impossibility for retrieval), why hybrids win, and the specific engineering innovations (MLA, GQA) that production models use. This split prevents cognitive overload — the student digests the paradigm shift before seeing the messy engineering reality.

- **Mechanistic interpretability is the most hands-on lesson in this module.** Unlike most Series 5 lessons (conceptual, no notebook), SAEs can actually be run on small models. The notebook should let the student decompose GPT-2 representations and find interpretable features. This is the "I can see inside the model" moment — potentially the most memorable lesson in the module.

- **Continual learning closes a loop from Series 4.** The student felt catastrophic forgetting when finetuning GPT-2 in Module 4.4 Lesson 1. This lesson should explicitly callback to that experience: "Remember when finetuning destroyed your model's ability to generate text? That's catastrophic forgetting. Now: what if we could update models without that happening?" The key insight from recent research (spurious forgetting) that much "forgetting" is actually task-alignment loss, not knowledge loss, is a satisfying twist.

- **Open Frontiers should inspire, not overwhelm.** Cover model merging, KANs, and world models at INTRODUCED/MENTIONED depth. The goal is to show the student that the field is wide open and point them toward rabbit holes they can explore independently. End with "you now have the mental models to read any of these papers yourself."

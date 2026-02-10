# Module 4.4: Beyond Pretraining -- Plan

**Module goal:** The student can explain and apply the full post-pretraining pipeline -- adapting a pretrained language model for classification tasks, understanding how instruction tuning and alignment transform base models into assistants, and using parameter-efficient methods to make this practical on real hardware.

## Narrative Arc

The student arrives having built GPT from scratch, trained it on TinyShakespeare, loaded real GPT-2 weights, and generated coherent text. They have a deep understanding of what pretraining produces: a next-token predictor. But a next-token predictor is not a useful tool. You cannot ask it questions. You cannot give it instructions. It just continues text. This module is about the transformation from "predicts the next word" to "follows your instructions."

The arc follows the historical and practical progression:

1. **Finetuning for classification** (Lesson 1) -- The most direct form of adaptation. Take the pretrained model, add a task head, train on labeled data. This is the LLM version of transfer learning from CNNs -- the student already knows this pattern. The callback is explicit: "You froze a ResNet backbone and added a classification head. Now you are doing the same thing with GPT-2." This grounds the module in familiar territory before venturing into new concepts.

2. **Instruction tuning (SFT)** (Lesson 2) -- Classification was about narrow tasks (sentiment, NER). But what if the task is "follow instructions"? Supervised finetuning on instruction-response pairs transforms a base model into something that looks like ChatGPT. The dataset format changes everything: instead of (text, label), it is (instruction, response). Chat templates, special tokens, and the mechanics of SFT.

3. **RLHF & alignment** (Lesson 3) -- SFT gets you a model that follows instructions, but not one that follows them well. It can be helpful but also harmful, verbose, or wrong in confident ways. RLHF introduces a second training signal: human preferences. Reward models, PPO basics, DPO as the simpler alternative. This is the most conceptual lesson -- no notebook, but essential for understanding why aligned models behave differently from SFT models.

4. **LoRA & quantization** (Lesson 4) -- The practical reality: full finetuning requires the same GPU memory as pretraining. LoRA makes finetuning accessible by training tiny rank-decomposition matrices. Quantization makes inference accessible by reducing precision. KV caching makes generation fast. These are the tools that let you run and adapt models on real hardware.

5. **Putting it all together** (Lesson 5) -- The synthesis. No new concepts. Walk the full pipeline: pretrain -> SFT -> alignment -> serve. Connect every module in Series 4 into a coherent mental model. What Series 5 builds on.

The key pedagogical move: the module starts from the most familiar transformation (classification finetuning = CNN transfer learning) and progressively introduces less familiar concepts (instruction tuning, alignment, efficient methods). Each lesson answers "why isn't the previous step enough?"

## Lesson Sequence

| Lesson | Core Concept | Type | Rationale for Position |
|--------|-------------|------|----------------------|
| finetuning-for-classification | Adapting a pretrained LLM for classification via task heads | BUILD | Must come first: most familiar adaptation pattern (direct callback to CNN transfer learning). Low novelty (student knows the pattern), new context (applying it to transformers). Establishes "pretrained model + task head" as the foundation for all subsequent adaptation. |
| instruction-tuning | SFT on instruction-response pairs transforms base model into assistant | STRETCH | Builds on Lesson 1's "adapt pretrained model" but introduces genuinely new concepts: instruction datasets, chat templates, the conceptual shift from narrow tasks to general instruction following. Requires understanding finetuning mechanics from Lesson 1. |
| rlhf-and-alignment | Human preference training (reward models, PPO, DPO) aligns model behavior | BUILD | Conceptual lesson after the STRETCH of instruction tuning. No notebook. Requires understanding SFT as the baseline that alignment improves. Answers "SFT follows instructions, but not always helpfully -- how do we fix that?" |
| lora-and-quantization | Parameter-efficient finetuning and inference optimization | STRETCH | Requires understanding full finetuning (Lessons 1-2) to appreciate why parameter efficiency matters. Introduces genuinely new math (low-rank decomposition, quantization). Practical payoff: student can actually run these techniques. |
| putting-it-all-together | Full pipeline synthesis: pretrain -> SFT -> align -> serve | CONSOLIDATE | Must be last: synthesizes everything. No new concepts. The mental model capstone for the entire series. |

## Rough Topic Allocation

- **Lesson 1 (finetuning-for-classification):** Why a pretrained LM needs adaptation for specific tasks (it predicts next tokens, not sentiment labels). The classification head pattern: take the last token's hidden state, project to num_classes. Freezing vs unfreezing layers (callback to CNN transfer learning). Training on a classification dataset (e.g., SST-2 sentiment). Comparing frozen backbone + head vs full finetuning. The "feature extractor" framing: pretrained transformer as a learned text encoder. Notebook: finetune GPT-2 for sentiment classification.

- **Lesson 2 (instruction-tuning):** What a base model actually does vs what a chat model does (complete the text vs follow instructions). Instruction dataset format: (instruction, input, output) triples. Chat templates and special tokens (<|im_start|>, <|im_end|>, role markers). Supervised finetuning mechanics: same training loop, different data format. Why SFT works: the model already "knows" things from pretraining, SFT teaches it the FORMAT of being helpful. Catastrophic forgetting risk. Notebook: SFT a small model on an instruction dataset.

- **Lesson 3 (rlhf-and-alignment):** Why SFT isn't enough (helpful but also harmful, sycophantic, confidently wrong). The alignment problem framed concretely. Reward models: train a model to predict human preferences between response pairs. PPO at a high level (policy, reward, KL penalty -- the intuition, not the full algorithm). DPO as the simpler alternative (no reward model needed, preference pairs directly optimize the policy). Why alignment matters beyond safety: it makes models genuinely more useful. No notebook -- conceptual understanding.

- **Lesson 4 (lora-and-quantization):** The memory problem: full finetuning GPT-2 requires storing all 124M gradients + optimizer states. LoRA: freeze base weights, add small trainable rank-decomposition matrices (A and B, rank << hidden dim). Why low rank works (weight changes during finetuning are low-rank). Quantization: reducing float32/float16 to int8/int4 for inference. Absmax and zero-point quantization. GPTQ/AWQ as post-training quantization. KV caching revisited (now with quantized models). Notebook: LoRA finetuning + quantized inference.

- **Lesson 5 (putting-it-all-together):** The complete pipeline: pretraining (Module 4.3) -> SFT (Lesson 2) -> alignment (Lesson 3) -> efficient serving (Lesson 4). What each stage adds. Where open-source models fit (base models, instruct models, quantized models). What Series 5 explores (constitutional AI, reasoning, multimodal). No new concepts -- synthesis and mental model consolidation.

## Cognitive Load Trajectory

| Lesson | Load | Rationale |
|--------|------|-----------|
| finetuning-for-classification | BUILD | Direct callback to CNN transfer learning. Same pattern (freeze backbone, add head, train head), new domain (text instead of images, transformer instead of CNN). Low novelty, high connection to prior knowledge. |
| instruction-tuning | STRETCH | Genuinely new concepts: instruction datasets, chat templates, SFT mechanics. The shift from "narrow task" to "general instruction following" is conceptually demanding. |
| rlhf-and-alignment | BUILD | Conceptual lesson with no implementation. The ideas (reward models, preference learning) are new but explained at an intuitive level. Follows a STRETCH. |
| lora-and-quantization | STRETCH | New math (rank decomposition, quantization). But practical payoff is high and the student is ready for it after the conceptual foundation of Lessons 1-3. |
| putting-it-all-together | CONSOLIDATE | No new concepts. Synthesis and connection. Capstone feeling. |

## Module-Level Misconceptions

- **"Finetuning a language model is fundamentally different from transfer learning on CNNs."** It is the same pattern: pretrained feature extractor + task-specific head. The features are different (text representations vs image features) but the strategy is identical.

- **"Instruction-tuned models know more than base models."** SFT does not add knowledge -- the base model already has all the information from pretraining. SFT teaches the model a new FORMAT for expressing what it knows (instruction-following rather than text completion).

- **"RLHF teaches the model what is true."** RLHF teaches the model what humans PREFER, which includes being helpful, harmless, and honest -- but preferences are not the same as truth. A model can learn to be confidently wrong in a way humans prefer.

- **"LoRA is an approximation that sacrifices quality."** For many tasks, LoRA achieves quality comparable to full finetuning. The key insight is that weight updates during finetuning are typically low-rank, so LoRA is not losing important information.

- **"You need a massive GPU to do anything useful with LLMs."** Quantization + LoRA + KV caching make it possible to finetune and run models on consumer hardware. The module should leave the student feeling empowered, not excluded.

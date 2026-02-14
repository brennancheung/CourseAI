# Module 5.1: Advanced Alignment -- Plan

**Status:** In progress (1 of 4 lessons built)
**Prerequisites:** Module 4.4 (Beyond Pretraining) -- specifically Lesson 3 (RLHF & Alignment) and the full finetuning/SFT pipeline

## Module Goal

The student can explain how alignment evolved beyond human-annotated RLHF -- from AI-supervised alignment (constitutional AI) through the expanding landscape of preference optimization techniques, to how alignment is stress-tested (red teaming) and measured (evaluation) -- and can critically assess the strengths and limitations of each approach.

## Narrative Arc

Module 4.4 ended with the student understanding the RLHF pipeline: humans label preferences, a reward model learns from those labels, and PPO optimizes against that reward model (with DPO as a simpler alternative). But the student also saw the limitations -- reward hacking, the "editor with blind spots" problem, and the fundamental constraint that human annotation is expensive and doesn't scale.

Module 5.1 picks up that thread and follows it forward. **The opening question: what if we could remove the human bottleneck?** Constitutional AI answers this by replacing human annotators with AI-generated feedback guided by principles. This is the conceptual leap -- "the editor becomes an AI editor" -- and it sets up the rest of the module.

From there, the landscape widens. DPO was introduced as "PPO but simpler." Now: what if even DPO isn't the right formulation? Lesson 2 maps the design space of preference optimization (IPO, KTO, ORPO), showing that the PPO/DPO binary from Series 4 was a starting point, not the final answer.

Then comes the adversarial turn. Lessons 1-2 built alignment techniques; Lesson 3 asks how you break them. Red teaming and adversarial evaluation reveal that alignment is never "done" -- it's a dynamic between attack and defense. This gives the student a realistic mental model: alignment is an ongoing process, not a solved problem.

The module closes with evaluation. How do you measure whether alignment worked? Benchmarks, contamination, Goodhart's law applied to metrics, and why evaluation may be harder than training itself. This is the capstone: the student can now think critically about alignment claims rather than taking "aligned" at face value.

**The progression:** Build it (Lessons 1-2) -> Break it (Lesson 3) -> Measure it (Lesson 4).

## Lesson Sequence with Rationale

| Lesson | Core Concept | Type | Rationale for Position |
|--------|-------------|------|----------------------|
| constitutional-ai | AI-supervised alignment via principles, critique-and-revision, RLAIF | STRETCH | First lesson extends RLHF directly -- the "editor" analogy from Series 4 becomes "an AI editor." Conceptual leap from human-annotated to AI-annotated preferences. Opens with recap of RLHF/DPO (Reinforcement Rule: >3 lessons back). |
| alignment-techniques-landscape | DPO variations (IPO, KTO, ORPO), online vs offline, design space mapping | BUILD | Builds on constitutional-ai's RLAIF foundation and Series 4's DPO. Maps the full design space rather than introducing another single technique. Lower cognitive load -- organizing known concepts rather than introducing a radically new one. |
| red-teaming-and-adversarial-evaluation | Automated red teaming, jailbreaks, the attack-defense dynamic | BUILD | Requires understanding of what alignment techniques DO (Lessons 1-2) before exploring how they FAIL. Introduces adversarial thinking as complement to alignment. |
| evaluating-llms | Benchmarks, contamination, Goodhart's law for evaluation, human vs automated eval | CONSOLIDATE | Capstone. Requires understanding of alignment techniques AND their failure modes. Integrates everything: you need to know what alignment does, how it can fail, and now how to measure whether it worked. Lower new concept load -- applies critical thinking to evaluation rather than introducing new mechanisms. |

## Rough Topic Allocation

- **Lesson 1 (constitutional-ai):** Recap of RLHF bottleneck (human annotation cost/scale), constitutional AI principles as prompts, critique-and-revision loop, RLAIF (AI-generated preference labels replacing human labels), the "AI supervising AI" paradigm, comparison with human RLHF
- **Lesson 2 (alignment-techniques-landscape):** DPO recap as anchor point, IPO (bounded preferences), KTO (single-response signal, no pairs needed), ORPO (odds ratio, no reference model), online vs offline preference optimization, iterative RLHF, mapping the design space along key axes (reference model needed? paired data needed? reward model needed?)
- **Lesson 3 (red-teaming-and-adversarial-evaluation):** What red teaming is and why it matters, jailbreak categories (prompt injection, role-play exploits, encoding tricks), automated red teaming (using LLMs to find LLM weaknesses), the cat-and-mouse dynamic, why "safe" is never a finished state, defense-in-depth
- **Lesson 4 (evaluating-llms):** Benchmark zoo overview, contamination problem, Goodhart's law applied to LLM metrics, human evaluation challenges (inter-annotator agreement, cost, bias), automated evaluation (LLM-as-judge, limitations), evaluation as harder than training, what benchmarks actually measure vs what they claim to measure

## Cognitive Load Trajectory

| Lesson | Load | Rationale |
|--------|------|-----------|
| constitutional-ai | STRETCH | New paradigm: AI-supervised alignment. Extends RLHF mental model in a non-obvious direction. Critique-and-revision is a novel mechanism. |
| alignment-techniques-landscape | BUILD | Organizing variations of known concepts (DPO from Series 4) rather than introducing a fundamentally new idea. The challenge is breadth, not depth. |
| red-teaming-and-adversarial-evaluation | BUILD | Adversarial thinking is new but accessible -- "how would you break this?" is intuitive. Builds on concrete understanding of what alignment does. |
| evaluating-llms | CONSOLIDATE | Integrates the full module. Few genuinely new mechanisms; mostly applying critical thinking and connecting prior concepts. |

No two STRETCH lessons are adjacent.

## Module-Level Misconceptions

- **"Constitutional AI replaces RLHF"** -- Students may think CAI is a wholesale replacement. In practice, it's a variant of RLHF where the preference annotation source changes (AI instead of humans). The RL training loop (or DPO-style optimization) still happens. CAI changes where the signal comes from, not the optimization mechanism.

- **"More alignment techniques = more aligned models"** -- The landscape lesson risks giving the impression that alignment is solved by picking the right technique. Students may think IPO/KTO/ORPO are strictly "better" than DPO. In reality, they make different tradeoffs, and the choice depends on data availability, compute budget, and what failure modes you care about.

- **"Alignment is a one-time process"** -- Students may think you align a model once and it's done. Red teaming reveals that alignment is dynamic: new attacks surface new failures, which require new defenses, which create new attack surfaces.

- **"Benchmarks measure what they claim to measure"** -- Students may take benchmark scores at face value. Goodhart's law, contamination, and the gap between benchmark performance and real-world behavior challenge this assumption.

- **"AI feedback is less reliable than human feedback"** -- Students may assume human feedback is the gold standard and AI feedback is a compromise. Constitutional AI shows that AI feedback can be more consistent and scalable, though it inherits the biases of the supervising model.

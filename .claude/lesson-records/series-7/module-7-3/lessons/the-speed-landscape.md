# Lesson: The Speed Landscape (7.3.3)

**Module:** 7.3 -- Fast Generation
**Position:** Lesson 3 of 3 (final lesson in Module 7.3 and Series 7's speed narrative)
**Type:** CONSOLIDATE (0 new concepts)
**Slug:** `the-speed-landscape`

---

## Phase 1: Orient -- Student State

The student arrives at this lesson with an unusually rich and well-connected set of acceleration concepts spanning two series. This is a synthesis lesson: the student knows every piece and now needs the map.

### Relevant Concepts with Depths and Sources

| Concept | Depth | Source | Notes |
|---------|-------|--------|-------|
| DDPM reverse sampling (many small steps, Markov chain constraint) | DEVELOPED | 6.2.4 (sampling-and-generation) | Student understands WHY DDPM needs ~1000 steps: coefficients calibrated for adjacent timesteps only |
| DDIM predict-and-leap mechanism | DEVELOPED | 6.4.2 (samplers-and-efficiency) | Predict x_0, leap to target timestep via alpha-bar. Deterministic with sigma=0. ~50 steps. |
| ODE perspective on diffusion (noise predictions define a vector field; samplers are ODE solvers) | INTRODUCED | 6.4.2 (samplers-and-efficiency) | Connected to Euler's method and gradient descent. Foundation for everything in 7.2 and 7.3. |
| DPM-Solver / higher-order ODE solvers | INTRODUCED | 6.4.2 (samplers-and-efficiency) | 15-20 steps by reading trajectory curvature. Three tiers (first/second/third order). |
| Sampler as inference-time choice (zero retraining) | DEVELOPED | 6.4.2 (samplers-and-efficiency) | Key insight: swap schedulers without retraining the model. |
| Score function and score-noise equivalence | DEVELOPED | 7.2.1 (score-functions-and-sdes) | Noise prediction IS a scaled score function. Unified framework. |
| Probability flow ODE (deterministic reverse process) | INTRODUCED | 7.2.1, deepened from MENTIONED in 6.4.2 | Named, formalized, connected to DDIM. |
| Flow matching (straight-line interpolation, velocity prediction) | DEVELOPED | 7.2.2 (flow-matching) | Straight paths by construction. 20-30 steps. "Design the trajectory, then derive the training objective." |
| Straight vs curved trajectories (core geometric insight) | DEVELOPED | 7.2.2 (flow-matching) | Euler's method on straight paths is exact in one step. Curvature sets a floor on step count. |
| Rectified flow (iterative trajectory straightening) | INTRODUCED | 7.2.2 (flow-matching) | 1-2 rounds of straightening significantly reduces steps. |
| Self-consistency property of ODE trajectories | DEVELOPED | 7.3.1 (consistency-models) | Any point on the same deterministic ODE trajectory maps to the same endpoint. |
| Consistency function f(x_t, t) = x_0 (direct mapping, no solver) | DEVELOPED | 7.3.1 (consistency-models) | One function evaluation replaces multi-step trajectory following. "Teleport to the destination." |
| Consistency distillation (teacher-student with pretrained diffusion model) | DEVELOPED | 7.3.1 (consistency-models) | Four-step training procedure. Teacher provides trajectory estimates. FID ~3.5 on ImageNet 64x64. |
| Multi-step consistency generation (2-4 refinement steps) | INTRODUCED | 7.3.1 (consistency-models) | Restart pattern (teleport to x_0, re-noise, teleport again). Recovers quality gap. |
| "Three levels of speed" framework | DEVELOPED | 7.3.1, extended in 7.3.2 | Level 1: better solvers, Level 2: straighter paths, Level 3a: consistency teleport, Level 3b: adversarial teleport. |
| Latent Consistency Models (LCM) | DEVELOPED | 7.3.2 (latent-consistency-and-turbo) | Consistency distillation on latent diffusion. "Same recipe, bigger kitchen." |
| LCM-LoRA ("speed as a skill") | DEVELOPED | 7.3.2 (latent-consistency-and-turbo) | ~4 MB adapter, universal across compatible fine-tunes. Reframes LoRA from style to behavioral skill. |
| Adversarial diffusion distillation (ADD / SDXL Turbo) | DEVELOPED | 7.3.2 (latent-consistency-and-turbo) | Hybrid loss (diffusion + adversarial). Sharp 1-step outputs. Specific model, not adapter. |
| Level 3a/3b distinction | DEVELOPED | 7.3.2 (latent-consistency-and-turbo) | 3a consistency-based (LCM/LCM-LoRA), 3b adversarial (ADD/SDXL Turbo). Different teacher signals. |

### Mental Models Already Established

- **"Three levels of speed"** -- better solvers / straighter paths / teleport. The organizing framework since 7.3.1.
- **"Teleport to the destination"** -- consistency models bypass the trajectory. One evaluation, no solver.
- **"GPS recalculating vs straight highway"** -- curved paths need many corrections, straight paths do not.
- **"Symptom vs cause"** -- better solvers treat the symptom (curvature), flow matching treats the cause (the trajectory itself).
- **"Same landscape, four lenses"** -- diffusion SDE, probability flow ODE, flow matching, consistency models. Same start (noise), same end (data), different routes.
- **"Same recipe, bigger kitchen"** -- LCM is consistency distillation at latent diffusion scale.
- **"Speed as a skill"** -- LCM-LoRA captures "how to generate fast" as an adapter.
- **"Two teachers, two lessons"** -- ADD's dual training signals (diffusion + adversarial).
- **"Discovery vs learning"** -- consistency training discovers; distillation learns from established knowledge.

### What Was NOT Covered in Prior Lessons That Is Relevant

- No comprehensive side-by-side comparison of ALL approaches with the same evaluation criteria
- No practical decision framework ("given these requirements, use this approach")
- No explicit discussion of composability across levels (e.g., flow matching teacher + consistency distillation)
- No treatment of how these approaches interact with model customization (LoRA, ControlNet)

### Readiness Assessment

The student is fully prepared. Every concept needed for this synthesis lesson has been taught at INTRODUCED or DEVELOPED depth across six prior lessons. The "three levels of speed" framework was specifically designed to scaffold this consolidation -- the student already has the organizing structure and now needs it filled in with decision-relevant detail. No gaps to fill.

---

## Phase 2: Analyze

### Target Concept

This lesson teaches the student to **choose the right acceleration approach for a given generation scenario by comparing all approaches along quality, speed, flexibility, and composability dimensions**.

### Prerequisites Table

| Concept | Required Depth | Actual Depth | Source Lesson | Status | Reasoning |
|---------|---------------|-------------|---------------|--------|-----------|
| DDIM predict-and-leap | INTRODUCED (recognize as baseline) | DEVELOPED | 6.4.2 | OK | Overqualified. Student deeply understands the mechanism. |
| DPM-Solver / higher-order solvers | INTRODUCED (recognize as Level 1) | INTRODUCED | 6.4.2 | OK | Exact match. Student knows the concept and step counts. |
| Flow matching (straight-line trajectories) | INTRODUCED (recognize as Level 2) | DEVELOPED | 7.2.2 | OK | Overqualified. Student can explain velocity prediction and training. |
| Consistency distillation | INTRODUCED (recognize as Level 3a mechanism) | DEVELOPED | 7.3.1 | OK | Overqualified. Student knows the training procedure. |
| LCM / LCM-LoRA | INTRODUCED (recognize as practical Level 3a) | DEVELOPED | 7.3.2 | OK | Overqualified. Student has run these in notebooks. |
| Adversarial diffusion distillation (ADD) | INTRODUCED (recognize as Level 3b mechanism) | DEVELOPED | 7.3.2 | OK | Overqualified. Student understands the hybrid loss. |
| "Three levels of speed" framework | DEVELOPED (use as organizing structure) | DEVELOPED | 7.3.1, extended 7.3.2 | OK | Exact match. Framework was designed for this lesson. |
| Sampler as inference-time choice | INTRODUCED (contrast with training-time changes) | DEVELOPED | 6.4.2 | OK | Overqualified. Key for distinguishing inference-only vs retraining. |
| LoRA composability | INTRODUCED (evaluate flexibility) | INTRODUCED | 7.3.2 (LCM-LoRA + style LoRA) | OK | Exact match. Student has composed LoRA adapters. |
| ControlNet conditioning | INTRODUCED (evaluate compatibility) | DEVELOPED | 7.1.1-7.1.2 | OK | Overqualified. Student knows the architecture. |

**Gap resolution:** No gaps. All prerequisites met or exceeded.

### Misconceptions Table

Since this is a CONSOLIDATE lesson with 0 new concepts, the misconceptions are about the RELATIONSHIPS between known concepts, not about the concepts themselves.

| Misconception | Why They'd Think This | Negative Example That Disproves It | Where to Address |
|---------------|----------------------|-----------------------------------|-----------------|
| "There is one best approach -- just use the fastest one (SDXL Turbo) for everything" | Natural optimization instinct: if a method is faster, it dominates. Reinforced by marketing emphasis on step counts. | SDXL Turbo is a fixed model -- cannot use your custom LoRA, cannot add ControlNet, cannot use a different base checkpoint. A photographer who wants 4-step generation from their custom portrait model MUST use LCM-LoRA, not SDXL Turbo. Speed alone is not the decision criterion. | Decision framework section -- the "fastest wins" assumption is the first thing to dismantle |
| "These approaches are mutually exclusive -- pick one" | Each was taught in a separate lesson, creating an implicit sense of competition. | Flow matching + consistency distillation is strictly better than either alone. A flow matching teacher produces straighter trajectories, giving the consistency model a better training signal. LCM-LoRA + style LoRA is another example: speed and style composed simultaneously. Approaches stack across levels. | "Composability" section -- show that levels combine |
| "Fewer steps always means worse quality -- there must be a linear tradeoff" | Intuitive: more computation = better result. DDPM's 1000 steps produce the highest quality, so fewer steps must always mean lower quality. | DPM-Solver++ at 20 steps produces quality comparable to DDPM at 1000 steps. The "linear tradeoff" assumption confuses necessary computation with wasted computation. DDPM's 1000 steps include enormous redundancy that better methods eliminate without quality loss. The real tradeoff appears below ~10 steps, where compression becomes lossy. | Quality-speed mapping -- show the curve is NOT linear, it has a plateau |
| "Level 1, 2, 3 are a progression where each replaces the previous" | The numbering suggests iteration: version 1, then 2, then 3. Each seems to improve on the last. | Level 1 (DPM-Solver++) runs on ANY existing diffusion model with zero retraining. Level 3b (SDXL Turbo) requires a specific distilled model. The levels have different requirements, different costs, and different flexibility. A researcher training a new architecture would use Level 1 immediately and maybe never need Level 3. They are options, not upgrades. | Framework section -- reframe from progression to menu |
| "The decision is purely about speed vs quality" | These are the two most salient dimensions, and prior lessons emphasized the speed-quality tradeoff. | A user with a custom LoRA fine-tune needs LCM-LoRA (composable) rather than SDXL Turbo (faster but locked model). A user who needs ControlNet conditioning needs to verify compatibility. The decision has at least four dimensions: speed, quality, flexibility (model compatibility), and composability (works with adapters/ControlNet). | Decision framework -- explicitly add flexibility and composability as decision dimensions |

### Examples Planned

| Example | Type | Purpose | Why This Example |
|---------|------|---------|-----------------|
| **The photographer scenario** -- custom portrait model, wants fast generation, needs ControlNet for pose, uses style LoRA | Positive (scenario) | Demonstrates that flexibility and composability drive the decision as much as speed. Answer: LCM-LoRA (composable with style LoRA and ControlNet, works with custom base model). SDXL Turbo fails here (locked model, no LoRA composability). | Forces evaluation across all four dimensions, not just speed. The "obvious" fastest answer is wrong. |
| **The prototype scenario** -- just installed diffusers, has a vanilla SD 1.5 model, wants to see faster results immediately | Positive (scenario) | Demonstrates that Level 1 has zero barrier to entry. Answer: swap scheduler to DPM-Solver++, set steps=20. Done. No downloading, no LoRA loading, no retraining. | Shows that the simplest solution is often correct. Counterbalances the excitement about Level 3 approaches. |
| **The maximum quality 1-step scenario** -- building a real-time application, latency is king, quality must be competitive | Positive (scenario) | Demonstrates Level 3b's unique strength. Answer: SDXL Turbo (ADD gives sharper 1-step output than consistency distillation). Trade flexibility for 1-step quality. | Shows a genuine case where the "locked model" tradeoff is worth it. Balances the photographer example. |
| **The "just stack everything" scenario** -- uses flow matching model + LCM-LoRA + high-order solver + SDXL Turbo | Negative | Demonstrates that approaches do NOT all compose. LCM-LoRA and SDXL Turbo address the same level (3) with incompatible mechanisms. A high-order solver is irrelevant when the consistency model already bypasses the trajectory. You cannot have a DPM-Solver step schedule on a consistency model -- there is no trajectory to step along. | Exposes the boundaries of composability. Prevents the misconception that more acceleration = faster. |
| **The quality curve** -- DDPM 1000 steps, DPM-Solver 20 steps, LCM 4 steps, SDXL Turbo 1 step: where does quality actually drop? | Positive (analytical) | Demonstrates the nonlinear quality-speed relationship. The drop from 1000 to 20 steps is nearly invisible (Level 1 eliminates redundancy). The drop from 20 to 4 is small but measurable. The drop from 4 to 1 is noticeable (softness or artifacts). | Dismantles the "linear tradeoff" misconception with specific quality regions. |

---

## Phase 3: Design

### Narrative Arc

The student has spent three modules (6.4, 7.2, 7.3) learning acceleration approaches one at a time, each presented as a response to a limitation of the previous approach. Better solvers reduced DDPM's 1000 steps to 20, but curved trajectories set a floor. Flow matching straightened the trajectories, but you still need ODE solver steps. Consistency models bypassed the trajectory entirely, but 1-step output is softer than 50-step diffusion. Adversarial distillation sharpened the 1-step output, but locked you into a specific model. Each lesson answered one question and raised another. The student now has six tools and no workshop manual. This lesson IS the workshop manual: a decision framework that organizes what they know by what matters -- not by chronological order of invention, but by what you need for your specific generation scenario. The feeling should be: "I can look at any diffusion pipeline and immediately identify which acceleration strategy it uses and whether it is the right choice for the task."

### Modalities Planned

| Modality | What Specifically | Why This Modality for This Concept |
|----------|------------------|------------------------------------|
| **Visual (table/grid)** | Comprehensive comparison grid: all approaches on one axis, evaluation dimensions (steps, quality, flexibility, composability, requirements) on the other. A single artifact the student can screenshot and reference. | A synthesis lesson needs a synthesis artifact. The student has seen each approach described in text across six lessons. A structured grid is the only way to make relationships visible at a glance. This is the equivalent of the periodic table for acceleration approaches. |
| **Verbal/Analogy** | "Menu, not upgrade path" reframing of the three levels. The levels are not version numbers (each replacing the last) but menu items (pick what fits your constraints). | The numbering of levels strongly implies progression. The analogy explicitly breaks this implication and reframes the relationship. |
| **Concrete scenario** | Three decision scenarios (photographer, prototype, real-time app) worked through the framework step by step. Each scenario applies the same evaluation dimensions but reaches a different answer. | Scenarios are the natural modality for a decision framework. Abstract comparison tables are inert until applied to specific situations. The student needs to practice the decision process, not memorize the table. |
| **Visual (quality curve)** | Nonlinear quality-speed curve showing three regions: "free" speedup (1000->20 steps, no quality loss), "cheap" speedup (20->4 steps, small quality cost), "expensive" speedup (4->1 step, noticeable quality cost). | Dismantles the linear tradeoff misconception visually. The curve shape is the key insight: most of the speedup is free. |
| **Intuitive** | "Composability map" showing which approaches combine and which conflict. Arrows for "composes with," X marks for "conflicts with." | The student needs to see composability as a dimension, not an afterthought. A visual map makes the compatibility relationships scannable. |

### Cognitive Load Assessment

- **New concepts in this lesson:** 0. This is pure synthesis.
- **Previous lesson load:** BUILD (latent-consistency-and-turbo: LCM, LCM-LoRA, ADD applied known patterns at scale).
- **Load trajectory:** STRETCH -> BUILD -> CONSOLIDATE. This follows the ideal pattern: stretch introduces the hard concept, build applies it at scale, consolidate organizes and synthesizes.
- **Assessment:** Appropriate. A CONSOLIDATE lesson after BUILD gives the student breathing room to integrate. The cognitive work here is organizational (seeing relationships between known concepts) rather than acquisitional (learning new concepts). This is less taxing but deeply satisfying.

### Connections to Prior Concepts

| Existing Concept | Connection in This Lesson |
|-----------------|--------------------------|
| "Three levels of speed" (7.3.1, 7.3.2) | Reframed from a progression to a menu. All three levels (plus 3a/3b) organized in a single comparison grid. The student already knows this framework -- this lesson fills it with decision-relevant detail. |
| "Symptom vs cause" (7.2.2) | Extended to the full taxonomy: Level 1 treats the symptom (curvature), Level 2 treats the cause (the trajectory shape), Level 3 bypasses both (direct mapping). |
| "Sampler as inference-time choice" (6.4.2) | Becomes the key distinguishing dimension: Level 1 is inference-only (swap scheduler). Level 2 requires retraining. Level 3 requires distillation from an existing model. The "what must change" dimension. |
| LoRA composability (6.5.1, 7.3.2) | LCM-LoRA's composability with style LoRA is a decisive advantage in the decision framework. Extends the "speed as a skill" mental model. |
| ControlNet conditioning (7.1.1-7.1.2) | Compatibility with ControlNet becomes an evaluation dimension. LCM-LoRA preserves ControlNet compatibility; SDXL Turbo may not (different base architecture). |
| "GPS recalculating vs straight highway" (7.2.2) | Brief callback when placing flow matching in the taxonomy. Not re-taught. |
| Sampler comparison (6.4.2) | This lesson is the spiritual successor to the sampler comparison from 6.4.2, elevated from "which sampler" to "which acceleration paradigm." The student did this once at a smaller scale and now does it at the full scale. |

**Potentially misleading prior analogies:**
- The level numbering (1, 2, 3) could imply that 3 is always better than 2, which is better than 1. The "menu, not upgrade path" reframing directly addresses this.

### Scope Boundaries

**What this lesson IS:**
- A decision framework for choosing acceleration approaches
- A synthesis of all acceleration concepts from Series 6 and Module 7.2-7.3
- A composability map showing what combines and what conflicts
- A satisfying conclusion to Module 7.3 and the speed narrative of Series 7

**What this lesson is NOT:**
- New technical content (no new concepts, formulas, or mechanisms)
- A review lesson (not re-teaching anything -- assumes solid knowledge)
- An exhaustive survey of every possible acceleration method (no progressive distillation, no DDIM-specific inversion tricks, no quantization or hardware acceleration)
- A production deployment guide (no inference server optimization, no batching strategies)

**Target depth:** CONSOLIDATE. The student organizes existing INTRODUCED/DEVELOPED concepts into a decision framework. No concept changes depth -- the framework itself is the deliverable.

### Lesson Outline

#### 1. Context + Constraints
What this lesson is about: organizing every acceleration approach the student has learned into a decision framework. What we are NOT doing: learning new techniques, going deeper on any single approach, or covering deployment optimization. The student already knows every piece -- this lesson assembles the map.

#### 2. Recap (brief)
No gap-fill needed. Brief activation of the "three levels of speed" framework from 7.3.1 as the organizing skeleton. One ConceptBlock bridge: "You have learned six ways to generate images faster. Each was a response to a limitation of the previous approach. But which one should you actually use?" Transition from chronological learning order to decision-relevant organization.

#### 3. Hook: "The Wrong Question"
**Type:** Misconception reveal.
**Content:** Open with the question the student is probably asking: "Which approach is fastest?" Then reveal this is the wrong question. Show two scenarios where the fastest approach (SDXL Turbo at 1 step) is the wrong answer: (1) custom portrait model with style LoRA -- SDXL Turbo cannot compose, (2) research prototype where you just want to iterate faster -- swapping the scheduler takes 1 line. The right question is not "which is fastest" but "which is right for what I need." Set up the four-dimensional evaluation framework: speed, quality, flexibility, composability.

#### 4. Explain: The Complete Taxonomy
Organize all approaches into the comparison grid. NOT a re-teaching -- each row references the lesson where the concept was taught and activates the mental model with one sentence.

**The grid (structured as a comprehensive comparison):**

| Approach | Level | Steps | What Changes | Quality Trade | Flexibility | Composability |
|----------|-------|-------|-------------|---------------|-------------|---------------|
| DPM-Solver++ | 1 | 15-20 | Scheduler only (inference) | None at 20 steps | Any model, any checkpoint | Full (nothing changes about the model) |
| Flow matching (SD3/Flux) | 2 | 20-30 | Training objective + model weights | None at 25 steps | Requires retraining or flow-matching-native model | Full (standard diffusion pipeline) |
| LCM | 3a | 2-4 | Student model distilled from teacher | Small at 4 steps, noticeable at 1 | Requires distillation per base model | Moderate (standard pipeline, but fixed to teacher's capabilities) |
| LCM-LoRA | 3a | 4-8 | LoRA adapter loaded at inference | Moderate (LoRA approximation) | Any compatible base model | High (composable with style LoRA, ControlNet) |
| SDXL Turbo (ADD) | 3b | 1-4 | Dedicated distilled model | Sharpest at 1 step (adversarial) | Locked to SDXL Turbo checkpoint | Low (cannot swap base model, limited adapter support) |
| Consistency training (no teacher) | 3a | 1-4 | Trained from scratch | Lower than distillation (FID ~6.2 vs ~3.5) | Requires full training | Same as the trained model |

For each row: one-sentence mechanism summary (callback to source lesson), step count, what the student needs to change to use it.

**The quality-speed curve:** Visual description of the nonlinear relationship. Three regions described as GradientCards:
- Emerald (free speedup): 1000 -> 20 steps. DPM-Solver++ eliminates DDPM redundancy with no quality loss. This is wasted computation removed.
- Amber (cheap speedup): 20 -> 4 steps. Consistency distillation or flow matching reduces further with small, measurable quality cost. Detail softening at edges, some high-frequency texture loss.
- Rose (expensive speedup): 4 -> 1 step. Single-step generation shows noticeable softness (consistency) or occasional texture artifacts (adversarial). Quality-speed tradeoff becomes real here.

InsightBlock: "Most of the speedup is free." The first 50x speedup (1000->20) costs nothing. The next 5x (20->4) costs a little. The last 4x (4->1) costs the most per step gained.

#### 5. Check #1 (predict-and-verify)
Three questions:
1. A researcher trains a brand-new diffusion model on a novel dataset. Which acceleration level can they use immediately without any additional training? (Level 1: swap scheduler to DPM-Solver++)
2. A user wants 4-step generation from their custom SD 1.5 fine-tune. Which approach should they consider? (LCM-LoRA: composable with their custom checkpoint, no retraining needed)
3. True or false: "Flow matching (Level 2) makes Level 1 approaches obsolete." (False. Even flow matching models benefit from DPM-Solver++ over naive Euler. The levels compose.)

#### 6. Explore: The Decision Framework (interactive scenarios)
Three worked scenarios using the comparison grid:

**Scenario 1: The Photographer** (detailed walkthrough)
- Needs: custom portrait model (SD 1.5 fine-tune), style LoRA for "film grain" look, ControlNet for pose, wants fast generation for client previews
- Evaluate: SDXL Turbo? No -- locked model, no LoRA, different architecture from SD 1.5. LCM? Requires distilling from their specific model. LCM-LoRA? Yes -- plug into their model, compose with style LoRA and ControlNet, 4-step generation.
- Answer: LCM-LoRA + style LoRA + ControlNet at 4-8 steps.

**Scenario 2: The Prototype Builder** (brief)
- Needs: just installed diffusers, vanilla SD 1.5, wants faster iteration during prompt exploration
- Evaluate: any Level 3? Overkill -- requires downloading adapters or dedicated models. Level 2? Requires a flow matching model. Level 1? One line change: `pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)`, set steps=20.
- Answer: DPM-Solver++ at 20 steps. Zero friction.

**Scenario 3: The Real-Time App** (brief)
- Needs: latency under 500ms, competitive quality, does not need custom styles or adapters
- Evaluate: Level 1 at 20 steps? Too slow for real-time. LCM-LoRA at 4 steps? Possible but not the sharpest 1-step option. SDXL Turbo at 1 step? Yes -- sharpest 1-step quality, acceptable flexibility tradeoff for a fixed deployment.
- Answer: SDXL Turbo (ADD) at 1-4 steps. Trade flexibility for latency.

#### 7. Elaborate: Composability and the Negative Example

**Composability map:** Show which approaches combine across levels and which conflict within levels.

Composes well:
- Level 1 + Level 2: DPM-Solver++ on a flow matching model. The straight trajectories make the solver even more effective. Different levels, complementary.
- Level 2 + Level 3a: flow matching teacher + consistency distillation (this IS what LCM does). Straighter trajectories give the consistency model better training signal.
- LCM-LoRA + style LoRA: speed + style as additive LoRA bypasses. Same model, different skills.
- LCM-LoRA + ControlNet: speed adapter + spatial control. Target different parts of the pipeline.

Does NOT compose:
- LCM-LoRA + SDXL Turbo: both address Level 3 with incompatible mechanisms. SDXL Turbo is a complete model, not an adapter.
- DPM-Solver++ + consistency model: the consistency model does not follow the trajectory. There is no ODE to solve with a higher-order method. The solver has nothing to solve.
- Two different Level 3 approaches on the same model: pick one bypass mechanism.

**Negative example (worked through):** "The Kitchen Sink" -- someone loads a flow matching model, adds LCM-LoRA, sets DPM-Solver++ as scheduler, and sets steps=20. What happens? The LCM-LoRA has been trained to produce quality output in 4 steps via the consistency objective. Running 20 steps with DPM-Solver++ is solving an ODE that the consistency model was trained to bypass. The result is unpredictable and likely worse than either approach used correctly. The approaches do not just fail to help -- they actively conflict.

#### 8. Check #2 (transfer questions)
Two questions:
1. Can you use a higher-order ODE solver (DPM-Solver++) to improve the output quality of an LCM model at 4 steps? Why or why not? (No. LCM produces output via the consistency function, not by ODE solving. DPM-Solver++ assumes a trajectory to step along -- LCM has no trajectory. Use multi-step consistency refinement instead.)
2. A team trained a diffusion model using flow matching (Level 2). They now want to push to 4-step generation. What should they consider? (Consistency distillation from their flow matching model -- Level 2 + Level 3a compose. The straight trajectories from flow matching give the consistency model better training signal. Or: distill an LCM-LoRA from the model for universal plug-and-play.)

#### 9. Practice: Notebook Exercises (Colab)
**Notebook:** `notebooks/7-3-3-the-speed-landscape.ipynb`

Four exercises, all analysis/comparison rather than training:

**Exercise 1 (Guided): The Sampler Swap**
- Load a vanilla SD 1.5 model. Generate the same image with DDPM (50 steps), DPM-Solver++ (20 steps), and DPM-Solver++ (10 steps). Compare quality and time.
- Purpose: Level 1 in action. The student has done sampler swaps before (6.4.2) but now frames it as the first and most accessible acceleration approach.
- Key insight: the 50->20 step reduction is nearly free. The 20->10 reduction starts showing quality loss.

**Exercise 2 (Guided): LCM-LoRA Speed Comparison**
- Load the same SD 1.5 model + LCM-LoRA. Generate at 4 steps and 8 steps. Compare with DPM-Solver++ at 20 steps (Exercise 1).
- Purpose: Level 3a vs Level 1 head-to-head. Same model, different acceleration strategy.
- Key insight: LCM-LoRA at 4 steps is faster than DPM-Solver++ at 20 steps, with some quality tradeoff.

**Exercise 3 (Supported): The Composability Test**
- Combine LCM-LoRA + a style LoRA. Generate at 4 steps. Compare with: (a) LCM-LoRA alone, (b) style LoRA alone at 20 steps with DPM-Solver++.
- Purpose: verify that speed + style compose as predicted by the framework.
- Key insight: the composition works because LoRA bypasses are additive and target different behaviors.

**Exercise 4 (Independent): Build Your Decision Cheat Sheet**
- Given three generation scenarios (described in markdown cells), the student writes which approach they would choose and why, using the four-dimensional framework (speed, quality, flexibility, composability). Then they verify one of their choices by implementing it and comparing with an alternative.
- Purpose: the student applies the decision framework independently. The reasoning matters more than the code.
- Key insight: there is no universal best -- the answer depends on the constraints.

**Exercise design notes:**
- Exercises are independent (can be done in any order) except Exercise 2 builds on the model loaded in Exercise 1.
- Progression: Guided -> Guided -> Supported -> Independent mirrors the lesson arc from concrete comparison to decision-making.
- No SDXL Turbo exercises (requires separate model download and significant VRAM). Its position in the framework is established conceptually.

#### 10. Summarize: The Map

Five key takeaways, echoing the mental models and reframing:

1. **Speed is a menu, not an upgrade path.** The three levels (better solvers, straighter trajectories, trajectory bypass) are options with different requirements and tradeoffs, not versions where each replaces the last.
2. **Most speedup is free.** Going from 1000 to 20 steps (Level 1) costs nothing. The quality-speed tradeoff only becomes real below ~10 steps.
3. **Four dimensions, not two.** Speed and quality are obvious. Flexibility (works with your model?) and composability (works with your adapters?) often determine the practical choice.
4. **Levels compose across, not within.** Flow matching (Level 2) + consistency distillation (Level 3a) is strictly better than either alone. But two Level 3 approaches on the same model conflict.
5. **The question is not "which is fastest" but "which is right for what I need."** A photographer with a custom model needs LCM-LoRA. A researcher with a new architecture needs DPM-Solver++. A real-time app needs SDXL Turbo. The framework tells you which.

#### 11. Next Step: Module and Series Completion

ModuleCompleteBlock: Module 7.3 (Fast Generation) complete. The student can now:
- Explain the self-consistency property and how it enables 1-step generation (7.3.1)
- Use LCM-LoRA for practical fast generation and understand ADD/SDXL Turbo as an alternative (7.3.2)
- Choose the right acceleration approach for any given scenario using the four-dimensional decision framework (7.3.3)

Bridge to Module 7.4 (Next-Generation Architectures): "You have seen how to make diffusion FASTER. The next module asks: what if we also change the architecture? SDXL stretches the U-Net to its limits. Then the Diffusion Transformer (DiT) replaces it entirely -- bringing the transformer architecture from Series 4 into the diffusion world. SD3 and Flux combine DiT + flow matching + better text encoding. Everything you have learned converges."

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
- [x] At least 3 modalities planned for the core concept, each with rationale (5 modalities: visual grid, verbal/analogy, concrete scenario, visual curve, intuitive composability map)
- [x] At least 2 positive examples + 1 negative example, each with stated purpose (3 positive scenarios + 1 quality curve + 1 negative "kitchen sink")
- [x] At least 3 misconceptions identified with negative examples (5 misconceptions)
- [x] Cognitive load: 0 new concepts (CONSOLIDATE)
- [x] Every concept connected to existing concepts (all synthesis of prior work)
- [x] Scope boundaries explicitly stated

---

## Review -- 2026-02-20 (Iteration 1/3)

### Summary
- Critical: 0
- Improvement: 2
- Polish: 1

### Verdict: NEEDS REVISION

No critical issues -- the student will not be lost or form wrong mental models. However, two planned visual modalities were downgraded to text during implementation, which weakens the lesson's modality coverage and misses the plan's explicit design intent. One polish item in the notebook.

### Findings

#### [IMPROVEMENT] -- Quality-speed curve is textual, not visual

**Location:** "The Quality-Speed Curve" section (three GradientCards: Free Speedup, Cheap Speedup, Expensive Speedup)
**Issue:** The planning document explicitly called for a "nonlinear quality-speed curve showing three regions" as a **visual modality** -- one of five planned modalities. The built lesson describes the three regions in GradientCards with text, but does not include an actual chart or diagram showing the curve shape. The plan's rationale stated: "Dismantles the linear tradeoff misconception visually. The curve shape is the key insight: most of the speedup is free." A text description of a curve is verbal, not visual -- the shape itself is the insight.
**Student impact:** The student reads about three regions but does not see the nonlinear curve shape. The most powerful rebuttal of the "linear tradeoff" misconception is seeing that the curve is flat from 1000-to-20 steps and then drops steeply below 4 steps. Without the visual, the student must construct this shape mentally from text descriptions, which is harder and less memorable. The "most speedup is free" insight lands weaker without seeing the shape.
**Suggested fix:** Add a simple stepped diagram or use Recharts to show a schematic quality-vs-steps curve. It does not need real FID data -- a schematic curve with annotated regions (flat region labeled "Free," gentle slope labeled "Cheap," steep drop labeled "Expensive") would accomplish the goal. Even a simple ASCII-style representation using styled divs would be better than pure text. This is the one modality gap that matters most for this lesson because it directly serves the "linear tradeoff" misconception.

#### [IMPROVEMENT] -- Composability map is textual, not visual

**Location:** "Composability" section (two GradientCards: "Composes Well" and "Does NOT Compose")
**Issue:** The planning document described this modality as a "Composability map showing which approaches combine across levels and which conflict within them. Arrows for 'composes with,' X marks for 'conflicts with.'" The plan's rationale: "The student needs to see composability as a dimension, not an afterthought. A visual map makes the compatibility relationships scannable." The built lesson uses two text-based lists in GradientCards. This is verbal, not visual/spatial.
**Student impact:** The student reads which pairs compose and which conflict, but does not get a scannable map that shows the pattern at a glance. The key insight -- "across levels compose, within levels conflict" -- is stated in the aside InsightBlock but is not visually reinforced by the composability data itself. A list of pairs requires the student to mentally sort which are cross-level and which are same-level; a visual map would make this pattern immediately obvious.
**Suggested fix:** Create a simple grid or matrix visualization. Rows and columns are the approaches (DPM-Solver++, Flow Matching, LCM, LCM-LoRA, SDXL Turbo). Cells show checkmarks or X marks. The cross-level vs within-level pattern becomes visually apparent because compatible pairs fall on off-diagonal positions (different levels) and incompatible pairs fall along the diagonal (same level). This could be a simple HTML table with green/red cells, or styled divs. Does not need to be interactive.

#### [POLISH] -- Notebook uses spaced em dashes throughout

**Location:** Multiple markdown cells and f-string labels across all exercises in `notebooks/7-3-3-the-speed-landscape.ipynb`
**Issue:** The writing style rule requires em dashes with no spaces (`word--word` or `word—word`), but the notebook uses spaced em dashes (` — `) extensively. Examples: "time each one — see where," "No new theory — just hands-on," "DDPM — 50 steps" (in plot labels), "The simplest and most accessible speedup — the," etc. This appears in approximately 40+ locations across the notebook's markdown cells and code string literals.
**Student impact:** Minimal -- this is a style consistency issue, not a comprehension issue. The student reads the notebook without confusion.
**Suggested fix:** Find-and-replace ` — ` with `—` in all notebook markdown cells. For f-string plot labels (e.g., `"DDPM — 50 steps"`), the spaced form may actually be more readable in plot titles where the em dash serves as a visual separator -- consider keeping those or using ` | ` instead. The markdown prose should use the no-space form to match the lesson component's style.

### Review Notes

**What works well:**
- The hook ("The Wrong Question") is one of the strongest hooks in the course so far. It immediately engages the student by subverting their assumption and then showing two concrete scenarios where it fails. This is textbook problem-before-solution.
- The three decision scenarios are perfectly differentiated -- each reaches a genuinely different conclusion using the same framework, which proves the framework's value rather than just asserting it.
- The Kitchen Sink negative example is excellent. It concretely shows what happens when you violate the composability rules, and the ComparisonRow providing two correct alternatives is pedagogically strong.
- The lesson's scope discipline is impressive. It resists the temptation to re-teach any concept and trusts the student's preparation. Every reference to a prior concept is a brief activation, not a review. This is exactly right for a CONSOLIDATE lesson.
- The narrative arc from "six tools and no workshop manual" to "this lesson IS the workshop manual" is satisfying and well-executed.
- The notebook is well-structured with clear progression (Guided -> Guided -> Supported -> Independent), good predict-before-run prompts, helpful "What Just Happened" reflections, and complete solutions with reasoning.

**Pattern observations:**
- Both improvement findings share the same root cause: visual modalities described in the plan were implemented as text during building. This is a common pattern when building lessons -- text is easier to write than visualizations, so planned visuals get downgraded. The plan specifically identified these as important modalities with clear rationales. For the quality-speed curve in particular, the visual IS the insight -- the nonlinear shape is what dismantles the linear tradeoff misconception.
- The lesson is very close to passing. The improvement findings target planned modalities that would strengthen the lesson but whose absence does not prevent learning. If these two visuals are added, this lesson should pass on the next iteration.

---

## Review -- 2026-02-20 (Iteration 2/3)

### Summary
- Critical: 0
- Improvement: 1
- Polish: 1

### Verdict: NEEDS REVISION

All three iteration 1 findings were addressed. The QualitySpeedChart and ComposabilityGrid are both solid additions that fulfill the plan's visual modality goals. The notebook em dashes are fixed. However, the ComposabilityGrid's legend text introduces a new inaccuracy that could confuse the student about the composability rule. One polish item in the new chart titles.

### Iteration 1 Resolution Check

**[IMPROVEMENT] Quality-speed curve is textual, not visual -- RESOLVED.**
The QualitySpeedChart component (Recharts LineChart with log-scale x-axis) shows the nonlinear curve with three colored ReferenceArea regions (Free/Cheap/Expensive), labeled data points (DDPM, DPM-Solver++, LCM/Flow, SDXL Turbo), and reference lines at region boundaries. The schematic data is clearly labeled as illustrative. The reversed log-scale axis correctly puts 1000 steps on the left and 1 step on the right, making the flat-then-steep shape immediately visible. The caption reinforces the key insight. This is a strong implementation that fulfills the plan's rationale: "Dismantles the linear tradeoff misconception visually."

**[IMPROVEMENT] Composability map is textual, not visual -- RESOLVED (with new issue).**
The ComposabilityGrid component implements the exact suggestion from iteration 1: a 5x5 matrix with rows/columns for each approach, checkmarks (green) and X marks (red) in each cell, level labels on each approach, and a legend. The data logic in `getComposability()` correctly encodes which pairs compose and which conflict. The visual is scannable and immediately shows the pattern. However, the legend text introduces a new issue (see finding below).

**[POLISH] Notebook uses spaced em dashes throughout -- RESOLVED.**
All spaced em dashes (`word — word`) in the notebook's markdown cells have been replaced with unspaced em dashes (`word—word`). Verified by grep: zero instances of ` — ` remain in the notebook.

### Findings

#### [IMPROVEMENT] -- ComposabilityGrid legend oversimplifies and contradicts matrix data

**Location:** ComposabilityGrid component, legend section (lines 353-365 of the lesson)
**Issue:** The legend labels checkmarks as "Composes (different levels)" and X marks as "Conflicts (same level)." The parenthetical annotations claim that composition happens between different levels and conflicts happen within the same level. But the matrix shows four cross-level conflicts: DPM-Solver++ (L1) vs LCM (L3a), DPM-Solver++ (L1) vs LCM-LoRA (L3a), DPM-Solver++ (L1) vs SDXL Turbo (L3b), and Flow Matching (L2) vs SDXL Turbo (L3b). These are all different levels, yet they conflict. The legend's parenthetical claim is directly contradicted by the data the student is looking at.
**Student impact:** The student reads the legend, forms the rule "different levels compose, same levels conflict," then looks at the matrix and sees four red X marks between different levels. This creates a contradiction: either the legend is wrong, or the student is misreading the matrix. The lesson's prose already explains these specific cross-level conflicts thoroughly (DPM-Solver++ has no trajectory to solve when combined with a consistency model; SDXL Turbo is a locked model that cannot swap training objectives). But the legend's oversimplification could undermine the student's confidence in the framework. The InsightBlock aside ("Across, Not Within") has the same oversimplification but is less prominent.
**Suggested fix:** Remove the parenthetical annotations from the legend labels. Change from "Composes (different levels)" to just "Composes" and from "Conflicts (same level)" to just "Conflicts." The legend should describe what the symbols mean, not assert a rule. The actual composability rules are explained in the surrounding prose and GradientCards, which correctly handle the nuances. Alternatively, refine the parentheticals to something accurate like "Composes (complementary mechanisms)" and "Conflicts (incompatible mechanisms)."

#### [POLISH] -- Chart titles use spaced em dashes

**Location:** QualitySpeedChart title (line 85) and ComposabilityGrid title (line 282)
**Issue:** Both new chart components use spaced em dashes in their titles: "Quality vs Steps — The Nonlinear Tradeoff" and "Composability Matrix — Which Approaches Combine?" The writing style rule requires no spaces around em dashes.
**Student impact:** Minimal. These are chart labels in monospace font, where the spaced form acts as a visual separator. The readability impact is negligible.
**Suggested fix:** Change to unspaced form: "Quality vs Steps—The Nonlinear Tradeoff" and "Composability Matrix—Which Approaches Combine?" Or, since these are chart labels rather than prose, consider using a different separator like a colon or pipe character if the unspaced em dash looks cramped in monospace.

### Review Notes

**What works well:**
- The QualitySpeedChart is well-executed. The log-scale axis, colored reference areas, labeled data points, and caption combine to make the nonlinear curve shape immediately visible. The student can see at a glance that quality is flat from 1000 to 20 steps and drops steeply below 4 steps. This is a significant pedagogical improvement over the text-only version.
- The ComposabilityGrid data is accurate. The `getComposability()` function correctly encodes all 10 unique pairs with appropriate comments explaining why each composes or conflicts. The visual format (colored cells with level labels) makes the compatibility pattern scannable.
- The notebook em dash fix is clean and consistent. All 40+ instances were corrected.

**What remains:**
- The single improvement finding is a legend text issue in the new ComposabilityGrid component. The matrix data itself is correct; only the legend's parenthetical annotations oversimplify the composability rules. This is a quick text change (remove or refine two parenthetical annotations). The InsightBlock aside has the same issue but is less prominent.
- After this fix, the lesson should pass on iteration 3. No structural changes needed.

# Module 8.5: Preference Optimization Deep Dives

## Module Goal

The student can explain how specific preference optimization techniques work at the mathematical level--deriving the loss functions, understanding what each term does, and implementing them in code--building on the conceptual understanding from Series 4-5.

## Narrative Arc

Series 4 and 5 gave the student the "what" and "why" of preference optimization: what DPO is, why it exists, where it sits in the design space of alignment techniques. But the mathematical "how" was explicitly deferred. This module fills that gap. Each lesson takes a technique the student already recognizes at INTRODUCED depth and develops it to DEVELOPED or APPLIED depth--deriving the math, building intuition for what the formulas mean, and implementing the technique in code. The progression is from conceptual recognition to mathematical fluency: "I know DPO skips the reward model" becomes "I can derive DPO's loss from the RLHF objective and explain why every term is there."

## Lesson Sequence

| Lesson | Core Concept | Type | Rationale for Position |
|--------|-------------|------|----------------------|
| direct-preference-optimization | DPO loss derivation, the Bradley-Terry preference model, and the implicit reward model insight | STRETCH | First and currently only lesson. STRETCH because the student needs to follow a multi-step mathematical derivation (Bradley-Terry -> RLHF objective -> closed-form solution -> DPO loss), which is genuinely new territory--the student has never derived a training objective from first principles in this course. Mathematical reasoning at this level is a new skill, not just a new topic. |

## Topic Allocation

- **Lesson 1 (direct-preference-optimization):** The Bradley-Terry model for pairwise preferences. How the RLHF objective (maximize reward minus KL divergence) has a closed-form optimal solution. Substituting that solution back to eliminate the reward model, yielding the DPO loss. What each term in the DPO loss does (log-ratio of policy to reference for preferred vs dispreferred, implicit KL penalty via the reference model). The "implicit reward model" insight--DPO defines a reward function in terms of the policy itself. Implementation of DPO training on a small model with preference data.

## Cognitive Load Trajectory

| Lesson | Load | Rationale |
|--------|------|-----------|
| direct-preference-optimization | STRETCH | Multi-step mathematical derivation is a new skill. The student has strong conceptual foundations (DPO at INTRODUCED, reward models at INTRODUCED, KL penalty at INTRODUCED) but has never followed a derivation of this length. Three genuinely new concepts: Bradley-Terry model, the closed-form RLHF solution, and the DPO loss function itself. |

## Module-Level Misconceptions

- Students may think DPO is an approximation of RLHF that sacrifices quality for simplicity (it is mathematically equivalent under certain assumptions, not an approximation)
- Students may think the reference model in DPO is optional or just a regularizer (it is structurally essential--without it, the loss is undefined and the connection to RLHF breaks)
- Students may think DPO "just increases probability of good responses" without understanding that the log-ratio structure simultaneously pushes down dispreferred responses relative to the reference
- Students may think the Bradley-Terry model is specific to DPO (it is a general preference model used across many fields--psychometrics, chess ratings, recommendation systems)

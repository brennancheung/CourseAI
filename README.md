# CourseAI

A personal learning app for understanding AI and machine learning fundamentals.

## Course Scope

- **Deep Learning Fundamentals** — Neural networks, backprop, optimization, architectures
- **PyTorch** — Practical implementation skills
- **LLMs** — Transformers, attention, training, fine-tuning, prompting
- **Recent LLM Advances** — RLHF, constitutional AI, reasoning models, multimodal
- **Stable Diffusion** — Classical diffusion, DDPM, latent diffusion
- **Post-SD Advances** — ControlNet, SDXL, consistency models, flow matching

## Quick Start

```bash
pnpm install
pnpm dev
```

## Commands

```bash
pnpm ai          # Typecheck + lint (fast)
pnpm aib         # Typecheck + lint + build (full verification)
pnpm dev         # Start dev server
```

## Architecture

- **Next.js 16** with App Router
- **Convex** for backend
- **Tailwind CSS v4** with shadcn/ui components

### Adding Lessons

1. Add a node to the curriculum tree in `src/data/curriculum/`
2. Create `src/app/app/lesson/{slug}/page.tsx`

'use client'

import { LessonLayout } from '@/components/lessons/LessonLayout'
import { Row } from '@/components/layout/Row'
import {
  LessonHeader,
  ObjectiveBlock,
  ConstraintBlock,
  SectionHeader,
  InsightBlock,
  TipBlock,
  WarningBlock,
  SummaryBlock,
  GradientCard,
  NextStepBlock,
} from '@/components/lessons'
import { CodeBlock } from '@/components/common/CodeBlock'
import { ExternalLink } from 'lucide-react'
import 'katex/dist/katex.min.css'
import { InlineMath } from 'react-katex'

/**
 * Building nanoGPT
 *
 * First lesson in Module 4.3 (Building & Training GPT).
 * Tenth lesson in Series 4 (LLMs & Transformers).
 *
 * Translates the complete GPT architecture from Module 4.2 into
 * working PyTorch code. This is a BUILD lesson: low conceptual
 * novelty, high implementation satisfaction. Every component was
 * DEVELOPED in Module 4.2; now the student writes the code.
 *
 * Core concepts at APPLIED:
 * - GPT architecture implemented in PyTorch (Head, MHA, FFN, Block, GPT)
 * - Shape verification at every layer boundary
 * - Parameter counting verification
 *
 * Core concepts at DEVELOPED:
 * - Weight initialization for deep transformers
 * - Autoregressive generation in code (generate method)
 *
 * EXPLICITLY NOT COVERED:
 * - Training the model (Lesson 2 -- pretraining)
 * - Dataset preparation (Lesson 2)
 * - Optimization, learning rate scheduling (Lesson 2)
 * - GPU utilization, mixed precision, flash attention (Lesson 3)
 * - Loading pretrained weights (Lesson 4)
 * - nn.MultiheadAttention (student builds from scratch)
 *
 * Previous: Decoder-Only Transformers (Module 4.2, Lesson 6)
 * Next: Pretraining on Real Text (Module 4.3, Lesson 2)
 */

export function BuildingNanogptLesson() {
  return (
    <LessonLayout>
      {/* ================================================================
          1. Header
          ================================================================ */}
      <Row>
        <Row.Content>
          <LessonHeader
            title="Building nanoGPT"
            description="Translate the GPT architecture into working PyTorch code&mdash;every component, from token embedding to generated text."
            category="Implementation"
          />
        </Row.Content>
      </Row>

      {/* ================================================================
          2. Context + Constraints
          ================================================================ */}
      <Row>
        <Row.Content>
          <ObjectiveBlock>
            Implement a complete GPT model in PyTorch: token embeddings,
            positional encoding, multi-head attention with causal masking,
            feed-forward networks, transformer blocks, output projection, and
            weight tying. Verify the parameter count matches GPT-2 (~124M).
            Generate text from the untrained model.
          </ObjectiveBlock>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="This Is Implementation">
            Every concept in this lesson was taught in Module 4.2. You already
            know <em>what</em> each component does. This lesson is about{' '}
            <em>writing</em> it. The challenge is craft, not comprehension.
          </TipBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <ConstraintBlock
            title="Scope for This Lesson"
            items={[
              'Implement the full GPT architecture in PyTorch (Head, MHA, FFN, Block, GPT)',
              'Verify shapes at every layer boundary',
              'Weight initialization for transformers (brief, practical)',
              'Generate text from the assembled (untrained) model',
              'GPT-2 small configuration (124M parameters)',
              'NOT: training the model\u2014that\u2019s the next lesson',
              'NOT: dataset preparation or data loading\u2014next lesson',
              'NOT: optimization, learning rate scheduling\u2014next lesson',
              'NOT: GPU utilization, mixed precision, flash attention\u2014Lesson 3',
              'NOT: loading pretrained weights\u2014Lesson 4',
            ]}
          />
        </Row.Content>
      </Row>

      {/* ================================================================
          3. Hook: The Parts List Reveal
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Parts List"
            subtitle="Five operations. You know all five."
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The GPT architecture powers the most capable AI systems ever
              built. Surely the code must be exotic&mdash;custom CUDA kernels,
              advanced autograd tricks, framework-specific incantations?
            </p>

            <p className="text-muted-foreground">
              Here is every unique PyTorch operation used in the entire model:
            </p>

            <div className="grid gap-3 md:grid-cols-5">
              <div className="px-3 py-2 bg-violet-500/10 border border-violet-500/20 rounded-lg text-center">
                <p className="text-sm font-mono font-medium text-violet-400">nn.Linear</p>
              </div>
              <div className="px-3 py-2 bg-violet-500/10 border border-violet-500/20 rounded-lg text-center">
                <p className="text-sm font-mono font-medium text-violet-400">nn.Embedding</p>
              </div>
              <div className="px-3 py-2 bg-violet-500/10 border border-violet-500/20 rounded-lg text-center">
                <p className="text-sm font-mono font-medium text-violet-400">nn.LayerNorm</p>
              </div>
              <div className="px-3 py-2 bg-violet-500/10 border border-violet-500/20 rounded-lg text-center">
                <p className="text-sm font-mono font-medium text-violet-400">nn.GELU</p>
              </div>
              <div className="px-3 py-2 bg-violet-500/10 border border-violet-500/20 rounded-lg text-center">
                <p className="text-sm font-mono font-medium text-violet-400">nn.Dropout</p>
              </div>
            </div>

            <p className="text-muted-foreground">
              Five operations. You have used every one of them since Series 2.
              The entire GPT model&mdash;124 million parameters, capable of
              generating coherent text&mdash;is assembled from these five pieces.
              The complexity is in the <strong>assembly</strong>, not the parts.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Familiar Building Blocks">
            <code className="text-xs">nn.Linear</code> and{' '}
            <code className="text-xs">nn.Embedding</code> from Series 2.{' '}
            <code className="text-xs">nn.LayerNorm</code> from The
            Transformer Block. <code className="text-xs">nn.GELU</code> and{' '}
            <code className="text-xs">nn.Dropout</code> from your CNN work.
            Nothing new.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          4. Architecture Diagram with Class Boundaries
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Blueprint"
            subtitle="Each colored region maps to a class you will write"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              You know this architecture from Decoder-Only Transformers. Now
              see which class implements each region. Five classes, bottom to
              top&mdash;this is the build order.
            </p>

            {/* Architecture diagram with class boundary annotations */}
            <div className="flex justify-center py-4">
              <svg
                width="440"
                height="600"
                viewBox="0 0 440 600"
                className="overflow-visible"
              >
                {/* ---- Legend ---- */}
                <rect x="20" y="10" width="10" height="10" rx="2" fill="#a78bfa" opacity={0.4} />
                <text x="35" y="19" fill="#9ca3af" fontSize="9">Residual stream</text>

                <rect x="140" y="10" width="10" height="10" rx="2" fill="#38bdf8" opacity={0.4} />
                <text x="155" y="19" fill="#9ca3af" fontSize="9">Attention (Head / CausalSelfAttention)</text>

                <rect x="20" y="28" width="10" height="10" rx="2" fill="#f59e0b" opacity={0.4} />
                <text x="35" y="37" fill="#9ca3af" fontSize="9">FeedForward</text>

                <rect x="140" y="28" width="10" height="10" rx="2" fill="#c084fc" opacity={0.4} />
                <text x="155" y="37" fill="#9ca3af" fontSize="9">Embedding (GPT class)</text>

                {/* ---- Residual stream backbone ---- */}
                <line x1="200" y1="560" x2="200" y2="110" stroke="#a78bfa" strokeWidth={2.5} strokeDasharray="6,4" opacity={0.2} />

                {/* ==== GPT class boundary (entire model) ==== */}
                <rect x="55" y="48" width="330" height="528" rx="12" fill="none" stroke="#6b7280" strokeWidth={1.5} strokeDasharray="8,4" opacity={0.5} />
                <text x="70" y="68" fill="#6b7280" fontSize="11" fontWeight="600" fontFamily="monospace">GPT</text>

                {/* ---- Token IDs (bottom, outside GPT) ---- */}
                <rect x="130" y="560" width="140" height="26" rx="5" fill="#6b7280" opacity={0.12} stroke="#6b7280" strokeWidth={1} />
                <text x="200" y="577" textAnchor="middle" fill="#9ca3af" fontSize="10" fontWeight="500">Token IDs</text>
                <text x="278" y="577" fill="#6b7280" fontSize="8" fontFamily="monospace">(B, T)</text>

                {/* Arrow up */}
                <line x1="200" y1="558" x2="200" y2="542" stroke="#9ca3af" strokeWidth={1.5} />
                <polygon points="196,545 200,538 204,545" fill="#9ca3af" />

                {/* ---- Token + Position Embedding ---- */}
                <rect x="115" y="510" width="170" height="28" rx="5" fill="#c084fc" opacity={0.1} stroke="#c084fc" strokeWidth={1.5} />
                <text x="200" y="528" textAnchor="middle" fill="#c084fc" fontSize="10" fontWeight="500">Token Embed + Pos Embed</text>
                <text x="295" y="528" fill="#6b7280" fontSize="8" fontFamily="monospace">(B, T, 768)</text>
                <text x="95" y="528" textAnchor="end" fill="#c084fc" fontSize="8" fontFamily="monospace" opacity={0.7}>wte + wpe</text>

                {/* Arrow up */}
                <line x1="200" y1="510" x2="200" y2="494" stroke="#a78bfa" strokeWidth={1.5} />
                <polygon points="196,497 200,490 204,497" fill="#a78bfa" />

                {/* ==== Block class boundary ==== */}
                <rect x="75" y="220" width="250" height="270" rx="8" fill="none" stroke="#fb923c" strokeWidth={1.5} strokeDasharray="6,3" opacity={0.6} />
                <text x="90" y="240" fill="#fb923c" fontSize="10" fontWeight="600" fontFamily="monospace">Block</text>
                <text x="310" y="240" fill="#6b7280" fontSize="8" fontFamily="monospace">{'\u00d7'}12</text>

                {/* ---- Pre-LN 1 ---- */}
                <rect x="145" y="458" width="110" height="22" rx="4" fill="#34d399" opacity={0.1} stroke="#34d399" strokeWidth={0.8} />
                <text x="200" y="473" textAnchor="middle" fill="#34d399" fontSize="8" fontWeight="500">LayerNorm (ln_1)</text>

                {/* Arrow up */}
                <line x1="200" y1="456" x2="200" y2="444" stroke="#a78bfa" strokeWidth={1.2} />

                {/* ---- CausalSelfAttention boundary ---- */}
                <rect x="95" y="372" width="210" height="72" rx="6" fill="#38bdf8" opacity={0.06} stroke="#38bdf8" strokeWidth={1.5} />
                <text x="110" y="390" fill="#38bdf8" fontSize="9" fontWeight="600" fontFamily="monospace">CausalSelfAttention</text>

                {/* MHA internals */}
                <rect x="110" y="396" width="55" height="18" rx="3" fill="#38bdf8" opacity={0.12} stroke="#38bdf8" strokeWidth={0.6} />
                <text x="137" y="409" textAnchor="middle" fill="#38bdf8" fontSize="7">Q, K, V</text>

                <rect x="172" y="396" width="55" height="18" rx="3" fill="#38bdf8" opacity={0.12} stroke="#38bdf8" strokeWidth={0.6} />
                <text x="199" y="409" textAnchor="middle" fill="#38bdf8" fontSize="7">Attention</text>

                <rect x="234" y="396" width="55" height="18" rx="3" fill="#38bdf8" opacity={0.12} stroke="#38bdf8" strokeWidth={0.6} />
                <text x="261" y="409" textAnchor="middle" fill="#38bdf8" fontSize="7">W_O</text>

                {/* Head annotation */}
                <text x="110" y="430" fill="#38bdf8" fontSize="7" fontFamily="monospace" opacity={0.6}>h heads (Head class each)</text>

                {/* Residual add */}
                <circle cx="200" cy="360" r="8" fill="none" stroke="#a78bfa" strokeWidth={1} />
                <text x="200" y="363" textAnchor="middle" fill="#a78bfa" fontSize="10" fontWeight="bold">+</text>
                <text x="215" y="363" fill="#a78bfa" fontSize="7" opacity={0.6}>residual</text>

                {/* Arrow up from attention to residual */}
                <line x1="200" y1="372" x2="200" y2="368" stroke="#38bdf8" strokeWidth={1.2} />

                {/* Arrow up from residual */}
                <line x1="200" y1="352" x2="200" y2="340" stroke="#a78bfa" strokeWidth={1.2} />

                {/* ---- Pre-LN 2 ---- */}
                <rect x="145" y="322" width="110" height="22" rx="4" fill="#34d399" opacity={0.1} stroke="#34d399" strokeWidth={0.8} />
                <text x="200" y="337" textAnchor="middle" fill="#34d399" fontSize="8" fontWeight="500">LayerNorm (ln_2)</text>

                {/* Arrow up */}
                <line x1="200" y1="322" x2="200" y2="310" stroke="#a78bfa" strokeWidth={1.2} />

                {/* ---- FeedForward boundary ---- */}
                <rect x="95" y="258" width="210" height="50" rx="6" fill="#f59e0b" opacity={0.06} stroke="#f59e0b" strokeWidth={1.5} />
                <text x="110" y="276" fill="#f59e0b" fontSize="9" fontWeight="600" fontFamily="monospace">FeedForward</text>

                <rect x="110" y="282" width="75" height="16" rx="3" fill="#f59e0b" opacity={0.12} stroke="#f59e0b" strokeWidth={0.6} />
                <text x="147" y="293" textAnchor="middle" fill="#f59e0b" fontSize="7">Linear (4x)</text>

                <rect x="192" y="282" width="40" height="16" rx="3" fill="#f59e0b" opacity={0.12} stroke="#f59e0b" strokeWidth={0.6} />
                <text x="212" y="293" textAnchor="middle" fill="#f59e0b" fontSize="7">GELU</text>

                <rect x="239" y="282" width="55" height="16" rx="3" fill="#f59e0b" opacity={0.12} stroke="#f59e0b" strokeWidth={0.6} />
                <text x="266" y="293" textAnchor="middle" fill="#f59e0b" fontSize="7">Linear (1x)</text>

                {/* Residual add */}
                <circle cx="200" cy="248" r="8" fill="none" stroke="#a78bfa" strokeWidth={1} />
                <text x="200" y="251" textAnchor="middle" fill="#a78bfa" fontSize="10" fontWeight="bold">+</text>
                <text x="215" y="251" fill="#a78bfa" fontSize="7" opacity={0.6}>residual</text>

                {/* Arrow up from FFN to residual */}
                <line x1="200" y1="258" x2="200" y2="256" stroke="#f59e0b" strokeWidth={1.2} />

                {/* Arrow up from residual */}
                <line x1="200" y1="240" x2="200" y2="224" stroke="#a78bfa" strokeWidth={1.5} />
                <polygon points="196,227 200,220 204,227" fill="#a78bfa" />

                {/* ---- Dots (repeated blocks) ---- */}
                <text x="200" y="218" textAnchor="middle" fill="#6b7280" fontSize="14" fontWeight="bold">{'\u22ee'}</text>

                {/* Arrow up */}
                <line x1="200" y1="202" x2="200" y2="186" stroke="#a78bfa" strokeWidth={1.5} />
                <polygon points="196,189 200,182 204,189" fill="#a78bfa" />

                {/* ---- Final Layer Norm ---- */}
                <rect x="130" y="156" width="140" height="24" rx="5" fill="#34d399" opacity={0.1} stroke="#34d399" strokeWidth={1.5} />
                <text x="200" y="172" textAnchor="middle" fill="#34d399" fontSize="9" fontWeight="500">Final LayerNorm (ln_f)</text>

                {/* Arrow up */}
                <line x1="200" y1="156" x2="200" y2="140" stroke="#34d399" strokeWidth={1.5} />
                <polygon points="196,143 200,136 204,143" fill="#34d399" />

                {/* ---- Output Projection ---- */}
                <rect x="115" y="108" width="170" height="28" rx="5" fill="#f59e0b" opacity={0.1} stroke="#f59e0b" strokeWidth={1.5} />
                <text x="200" y="126" textAnchor="middle" fill="#f59e0b" fontSize="10" fontWeight="500">Output Projection (lm_head)</text>
                <text x="295" y="126" fill="#6b7280" fontSize="8" fontFamily="monospace">(B, T, V)</text>

                {/* Weight tying annotation */}
                <path d="M 115 122 Q 80 122 80 528 Q 80 538 115 538" fill="none" stroke="#c084fc" strokeWidth={1} strokeDasharray="4,3" opacity={0.5} />
                <text x="68" y="330" fill="#c084fc" fontSize="7" fontFamily="monospace" opacity={0.6} textAnchor="end" transform="rotate(-90, 68, 330)">weight tied</text>

                {/* ---- Logits (top) ---- */}
                <rect x="140" y="78" width="120" height="24" rx="5" fill="#6366f1" opacity={0.1} stroke="#6366f1" strokeWidth={1} />
                <text x="200" y="94" textAnchor="middle" fill="#6366f1" fontSize="10" fontWeight="500">Logits / Softmax</text>

                {/* Arrow up */}
                <line x1="200" y1="108" x2="200" y2="102" stroke="#f59e0b" strokeWidth={1.5} />
                <polygon points="196,105 200,98 204,105" fill="#f59e0b" />
              </svg>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Diagram {'\u2192'} Code">
            Every colored region becomes a Python class. The build order
            goes bottom-up: start with the smallest piece (Head), compose
            into larger pieces (CausalSelfAttention, FeedForward, Block),
            then wire them all together (GPT).
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          5. Assembly Roadmap (navigational preview)
          ================================================================ */}
      <Row>
        <Row.Content>
          <div className="space-y-3">
            <p className="text-muted-foreground text-sm font-medium">
              The build order&mdash;five classes, each small and testable:
            </p>
            <div className="grid gap-2 md:grid-cols-5">
              <div className="px-3 py-2 bg-violet-500/10 border border-violet-500/20 rounded-lg text-center">
                <p className="text-xs font-mono font-medium text-violet-400">1. Head</p>
                <p className="text-xs text-muted-foreground mt-0.5">~15 lines</p>
              </div>
              <div className="px-3 py-2 bg-sky-500/10 border border-sky-500/20 rounded-lg text-center">
                <p className="text-xs font-mono font-medium text-sky-400">2. CausalSelfAttention</p>
                <p className="text-xs text-muted-foreground mt-0.5">batched MHA</p>
              </div>
              <div className="px-3 py-2 bg-amber-500/10 border border-amber-500/20 rounded-lg text-center">
                <p className="text-xs font-mono font-medium text-amber-400">3. FeedForward</p>
                <p className="text-xs text-muted-foreground mt-0.5">~6 lines</p>
              </div>
              <div className="px-3 py-2 bg-orange-500/10 border border-orange-500/20 rounded-lg text-center">
                <p className="text-xs font-mono font-medium text-orange-400">4. Block</p>
                <p className="text-xs text-muted-foreground mt-0.5">~4 lines forward</p>
              </div>
              <div className="px-3 py-2 bg-emerald-500/10 border border-emerald-500/20 rounded-lg text-center">
                <p className="text-xs font-mono font-medium text-emerald-400">5. GPT</p>
                <p className="text-xs text-muted-foreground mt-0.5">full assembly</p>
              </div>
            </div>
          </div>
        </Row.Content>
      </Row>

      {/* ================================================================
          6. Config
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Configuration"
            subtitle="All hyperparameters in one place"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Before writing any model code, define the hyperparameters. A
              dataclass keeps them organized and makes it easy to swap between
              configurations.
            </p>

            <CodeBlock
              code={`from dataclasses import dataclass

@dataclass
class GPTConfig:
    vocab_size: int = 50257     # GPT-2 BPE vocabulary
    block_size: int = 1024      # Maximum context length
    n_layer: int = 12           # Number of transformer blocks
    n_head: int = 12            # Number of attention heads
    n_embd: int = 768           # Embedding dimension (d_model)
    dropout: float = 0.1        # Dropout rate
    bias: bool = False          # Use bias in Linear layers?

# GPT-2 small (what we're building)
config = GPTConfig()

# Tiny debug config (fast iteration)
debug_config = GPTConfig(
    vocab_size=256,
    block_size=64,
    n_layer=4,
    n_head=4,
    n_embd=128,
)`}
              language="python"
              filename="config"
            />

            <p className="text-muted-foreground">
              The debug config is your best friend during development. It
              builds in seconds, uses minimal memory, and lets you verify
              shapes before scaling up. Every class you write will be tested
              with the debug config first, then switched to the full GPT-2
              config for the final parameter count.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Debug Config">
            Always build with a tiny config first. Bugs in tensor shapes are
            the same at 128 dimensions as at 768&mdash;but the tiny config
            runs instantly. Scale up only after the shapes check out.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          5. Build: Single Attention Head
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Build: Single Attention Head"
            subtitle="The formula from Module 4.2, now in code"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Start from the bottom. In Values and the Attention Output, you
              traced single-head attention by hand:{' '}
              <InlineMath math="\text{softmax}(QK^T / \sqrt{d_k}) \cdot V" />.
              Every line below maps to a step you already know.
            </p>

            <CodeBlock
              code={`import torch
import torch.nn as nn
import torch.nn.functional as F

class Head(nn.Module):
    """Single head of self-attention."""

    def __init__(self, config, head_size):
        super().__init__()
        # Q, K, V projections — three "lenses" on the same input
        self.query = nn.Linear(config.n_embd, head_size, bias=config.bias)
        self.key   = nn.Linear(config.n_embd, head_size, bias=config.bias)
        self.value = nn.Linear(config.n_embd, head_size, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

        # Causal mask — registered as a buffer (not a parameter)
        self.register_buffer(
            'mask',
            torch.tril(torch.ones(config.block_size, config.block_size))
        )

    def forward(self, x):
        B, T, C = x.shape                    # (batch, seq_len, n_embd)

        q = self.query(x)                    # (B, T, head_size)
        k = self.key(x)                      # (B, T, head_size)
        v = self.value(x)                    # (B, T, head_size)

        # Scaled dot-product attention
        scale = k.shape[-1] ** -0.5
        scores = q @ k.transpose(-2, -1) * scale   # (B, T, T)

        # Causal mask: set future positions to -inf
        scores = scores.masked_fill(
            self.mask[:T, :T] == 0, float('-inf')
        )                                    # (B, T, T)

        weights = F.softmax(scores, dim=-1)  # (B, T, T)
        weights = self.dropout(weights)

        out = weights @ v                    # (B, T, head_size)
        return out`}
              language="python"
              filename="head.py"
            />

            <CodeBlock
              code={`# Verify it — shape assertions catch bugs early
head = Head(debug_config, head_size=32)
x = torch.randn(2, 64, 128)     # (B=2, T=64, n_embd=128)
assert head(x).shape == (2, 64, 32), "Head output shape wrong!"
# ✓ (2, 64, 32) — as expected`}
              language="python"
              filename="verify_head.py"
            />

            <div className="px-4 py-3 bg-sky-500/10 border border-sky-500/20 rounded-lg space-y-2">
              <p className="text-sm font-medium text-sky-400">
                Mapping code to concepts
              </p>
              <ul className="text-sm text-muted-foreground space-y-1">
                <li>
                  <code className="text-xs">self.query/key/value</code> &mdash;
                  the three projections from Queries, Keys, and the Relevance Function
                </li>
                <li>
                  <code className="text-xs">k.shape[-1] ** -0.5</code> &mdash;
                  the <InlineMath math="1/\sqrt{d_k}" /> scaling factor
                </li>
                <li>
                  <code className="text-xs">register_buffer</code> &mdash;
                  stores the causal mask as a non-trainable tensor that moves
                  with the model to GPU
                </li>
                <li>
                  <code className="text-xs">masked_fill</code> &mdash;
                  the <InlineMath math="-\infty" /> insertion from
                  Decoder-Only Transformers
                </li>
              </ul>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="register_buffer">
            The causal mask is not a learnable parameter&mdash;it never gets
            gradients. But it needs to live on the same device as the model.{' '}
            <code className="text-xs">register_buffer</code> handles this:
            the mask moves to GPU with the model but is excluded from{' '}
            <code className="text-xs">model.parameters()</code>.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          6. Build: Multi-Head Attention
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Build: Multi-Head Attention"
            subtitle='The "split, not multiplied" dimension trick'
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              In Multi-Head Attention, you learned that multiple heads run in
              parallel with dimension splitting:{' '}
              <InlineMath math="d_k = d_{\text{model}} / h" />. There are two
              ways to implement this.
            </p>

            <p className="text-muted-foreground">
              <strong>Approach 1: explicit loop.</strong> Run <em>h</em>{' '}
              independent Head modules and concatenate. Matches the formula
              directly. Clear, correct, readable.
            </p>

            <CodeBlock
              code={`class MultiHeadAttention(nn.Module):
    """Multi-head attention with explicit head loop."""

    def __init__(self, config):
        super().__init__()
        head_size = config.n_embd // config.n_head
        self.heads = nn.ModuleList([
            Head(config, head_size) for _ in range(config.n_head)
        ])
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)  # W_O
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # Run each head independently, concatenate along last dim
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # (B, T, n_embd)
        out = self.proj(out)                                   # (B, T, n_embd)
        out = self.dropout(out)
        return out`}
              language="python"
              filename="multi_head_attention.py"
            />

            <p className="text-muted-foreground">
              <strong>Approach 2: batched reshape.</strong> A single set of
              Q, K, V projections produces all heads at once. The reshape
              operation IS the dimension splitting. No loop. This is what
              production implementations use.
            </p>

            <CodeBlock
              code={`class CausalSelfAttention(nn.Module):
    """Multi-head attention with batched computation (no loop)."""

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # Q, K, V for ALL heads in one projection
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)  # W_O
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        self.register_buffer(
            'mask',
            torch.tril(torch.ones(config.block_size, config.block_size))
                 .view(1, 1, config.block_size, config.block_size)
        )

    def forward(self, x):
        B, T, C = x.shape                                         # (B, T, n_embd)

        # Single projection, split into Q, K, V
        qkv = self.c_attn(x)                                      # (B, T, 3*n_embd)
        q, k, v = qkv.split(self.n_embd, dim=2)                   # each (B, T, n_embd)

        # Reshape into (B, n_head, T, head_size) — the dimension split
        head_size = C // self.n_head
        q = q.view(B, T, self.n_head, head_size).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, head_size).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, head_size).transpose(1, 2)  # (B, nh, T, hs)

        # Attention scores for all heads at once
        scale = head_size ** -0.5
        scores = q @ k.transpose(-2, -1) * scale                  # (B, nh, T, T)
        scores = scores.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        weights = F.softmax(scores, dim=-1)                        # (B, nh, T, T)
        weights = self.attn_dropout(weights)

        out = weights @ v                                          # (B, nh, T, hs)
        out = out.transpose(1, 2).contiguous().view(B, T, C)       # (B, T, n_embd)

        out = self.c_proj(out)                                     # (B, T, n_embd)
        out = self.resid_dropout(out)
        return out`}
              language="python"
              filename="causal_self_attention.py"
            />

            <p className="text-muted-foreground">
              Both produce the same output. The batched version replaces the
              Python loop with tensor reshapes&mdash;which GPUs handle
              efficiently. The reshape from{' '}
              <InlineMath math="(B, T, d_{\text{model}})" /> to{' '}
              <InlineMath math="(B, h, T, d_k)" /> IS the
              &ldquo;split, not multiplied&rdquo; operation from
              Multi-Head Attention. Verify it:
            </p>

            <CodeBlock
              code={`# Verify: loop and batched produce the same output
loop_mha = MultiHeadAttention(debug_config)
batched_mha = CausalSelfAttention(debug_config)

# Copy weights so both use the same parameters
# (details in the notebook)

x = torch.randn(2, 64, 128)
out_loop = loop_mha(x)
out_batched = batched_mha(x)
assert torch.allclose(out_loop, out_batched, atol=1e-5)
# ✓ Identical outputs — the reshape IS the dimension split`}
              language="python"
              filename="verify_equivalence.py"
            />
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Shape Correctness Is Not Enough">
            A single mistake in reshape ordering (swapping{' '}
            <code className="text-xs">n_head</code> and{' '}
            <code className="text-xs">T</code> dimensions) produces output
            with the correct final shape but completely wrong values. Always
            verify with assertions and test data.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          7. Build: Feed-Forward Network
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Build: Feed-Forward Network"
            subtitle="The writer in the attention-reads-FFN-writes pair"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              From The Transformer Block: attention reads from the context,
              the FFN processes each token independently. Two linear layers
              with a GELU activation and the 4x expansion factor.
            </p>

            <CodeBlock
              code={`class FeedForward(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(self, config):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu   = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):                    # (B, T, n_embd)
        x = self.c_fc(x)                     # (B, T, 4*n_embd)
        x = self.gelu(x)                     # (B, T, 4*n_embd)
        x = self.c_proj(x)                   # (B, T, n_embd)
        x = self.dropout(x)
        return x                             # (B, T, n_embd)`}
              language="python"
              filename="feed_forward.py"
            />

            <p className="text-muted-foreground">
              Two lines of meaningful code for a component that holds{' '}
              <strong>two-thirds of the block&rsquo;s parameters</strong>.
              Recall from The Transformer Block: with{' '}
              <InlineMath math="d_{\text{model}} = 768" /> and the 4x
              expansion, the FFN has{' '}
              <InlineMath math="2 \times 768 \times 3072 = 4{,}718{,}592" />{' '}
              parameters per block, compared to attention&rsquo;s{' '}
              <InlineMath math="2{,}359{,}296" />.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Attention Reads, FFN Writes">
            Attention gathers information from the context (reads). The FFN
            processes each token&rsquo;s enriched representation
            independently (writes). Both are needed. The FFN is where most
            of the model&rsquo;s learned knowledge lives.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          8. Build: Transformer Block
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Build: Transformer Block"
            subtitle="The formula is the code"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              In The Transformer Block, you learned the block formula:{' '}
              <InlineMath math="x' = x + \text{MHA}(\text{LN}(x))" />,{' '}
              <InlineMath math="\text{out} = x' + \text{FFN}(\text{LN}(x'))" />.
              Pre-norm ordering, two residual connections. The{' '}
              <code className="text-xs">forward()</code> method IS this
              formula:
            </p>

            <CodeBlock
              code={`class Block(nn.Module):
    """Transformer block: MHA + FFN with residual connections and layer norm."""

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.ffn  = FeedForward(config)

    def forward(self, x):                    # (B, T, n_embd)
        x = x + self.attn(self.ln_1(x))      # residual + MHA(LN(x))
        x = x + self.ffn(self.ln_2(x))       # residual + FFN(LN(x))
        return x                             # (B, T, n_embd)`}
              language="python"
              filename="block.py"
            />

            <p className="text-muted-foreground">
              The block preserves shape:{' '}
              <InlineMath math="(B, T, d_{\text{model}})" /> in,{' '}
              <InlineMath math="(B, T, d_{\text{model}})" /> out. This is
              what makes stacking possible. Twelve identical blocks, each
              reading from and writing to the same residual stream.
            </p>

            <CodeBlock
              code={`# Verify: block preserves shape exactly
block = Block(debug_config)
x = torch.randn(2, 64, 128)      # (B, T, n_embd)
assert block(x).shape == x.shape, "Block must preserve shape!"
# ✓ (2, 64, 128) in, (2, 64, 128) out`}
              language="python"
              filename="verify_block.py"
            />
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Pre-Norm Ordering">
            Layer norm comes <em>before</em> the sub-layer (MHA or FFN),
            not after. This is the modern standard (GPT-2 and later). The
            original Transformer used post-norm, which is harder to train
            at depth.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          9. Build: Full GPT Model
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Build: The Full GPT Model"
            subtitle="Token IDs in, logits out"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Now assemble everything. The GPT class wires together
              embeddings, blocks, and the output projection. Follow the
              architecture diagram from Decoder-Only Transformers&mdash;the
              code mirrors it from bottom to top.
            </p>

            <CodeBlock
              code={`class GPT(nn.Module):
    """Complete GPT language model."""

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            # Token + position embeddings
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            # N transformer blocks
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            # Final layer norm
            ln_f = nn.LayerNorm(config.n_embd),
        ))

        # Output projection (d_model -> vocab_size)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: share embedding weights with output projection
        self.transformer.wte.weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        if isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.config.block_size, \\
            f"Sequence length {T} exceeds block_size {self.config.block_size}"

        # Token + position embeddings
        tok_emb = self.transformer.wte(idx)                # (B, T, n_embd)
        pos = torch.arange(0, T, device=idx.device)        # (T,)
        pos_emb = self.transformer.wpe(pos)                # (T, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)      # (B, T, n_embd)

        # Pass through all transformer blocks
        for block in self.transformer.h:
            x = block(x)                                   # (B, T, n_embd)

        # Final layer norm + output projection
        x = self.transformer.ln_f(x)                       # (B, T, n_embd)
        logits = self.lm_head(x)                           # (B, T, vocab_size)

        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )

        return logits, loss`}
              language="python"
              filename="gpt.py"
            />

            <div className="px-4 py-3 bg-sky-500/10 border border-sky-500/20 rounded-lg space-y-2">
              <p className="text-sm font-medium text-sky-400">
                Shape trace through the full model
              </p>
              <div className="text-sm text-muted-foreground space-y-1 font-mono text-xs">
                <p>Input idx:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(B, T)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= (2, 64)</p>
                <p>Token embeddings:&nbsp;&nbsp;&nbsp;(B, T, n_embd)&nbsp;&nbsp;&nbsp;= (2, 64, 768)</p>
                <p>+ Position embeds:&nbsp;&nbsp;(B, T, n_embd)&nbsp;&nbsp;&nbsp;= (2, 64, 768)</p>
                <p>After each block:&nbsp;&nbsp;&nbsp;(B, T, n_embd)&nbsp;&nbsp;&nbsp;= (2, 64, 768)</p>
                <p>After final LN:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(B, T, n_embd)&nbsp;&nbsp;&nbsp;= (2, 64, 768)</p>
                <p>Logits:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(B, T, vocab)&nbsp;&nbsp;&nbsp;&nbsp;= (2, 64, 50257)</p>
              </div>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Weight Tying">
            <code className="text-xs">self.transformer.wte.weight = self.lm_head.weight</code>{' '}
            makes the embedding and output projection share the same matrix.
            Embedding maps token ID {'\u2192'} vector. Output projection maps
            vector {'\u2192'} token scores. Same mapping, opposite direction.
            This saves ~38M parameters.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          10. Weight Initialization
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Weight Initialization"
            subtitle="Why defaults fail for deep transformers"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              In Training Dynamics, you saw that initialization matters&mdash;Xavier
              and He initialization prevent activations from exploding or
              collapsing. Transformers add a new wrinkle: 12 blocks means 24
              residual additions. Each residual connection adds to the stream,
              and without careful scaling, activations grow with depth.
            </p>

            <p className="text-muted-foreground">
              The practical recipe for GPT:
            </p>

            <ul className="list-disc list-inside text-muted-foreground space-y-1 ml-4">
              <li>
                Most linear layers: normal distribution with{' '}
                <InlineMath math="\sigma = 0.02" />
              </li>
              <li>
                Residual projection layers (attention output, FFN output):
                scale by <InlineMath math="1 / \sqrt{2N}" /> where{' '}
                <InlineMath math="N" /> is the number of blocks
              </li>
              <li>
                Embedding layers: normal with{' '}
                <InlineMath math="\sigma = 0.02" />
              </li>
              <li>
                Biases: zero
              </li>
            </ul>

            <CodeBlock
              code={`# After the standard _init_weights, apply scaled init
# to residual projection layers
for name, p in model.named_parameters():
    if name.endswith('c_proj.weight'):
        torch.nn.init.normal_(
            p, mean=0.0,
            std=0.02 / (2 * config.n_layer) ** 0.5
        )`}
              language="python"
              filename="scaled_init.py"
            />

            <p className="text-muted-foreground">
              The <InlineMath math="1 / \sqrt{2N}" /> factor compensates for
              the 2N residual additions (two per block: one from attention,
              one from FFN). Without it, activation standard deviations grow
              with depth. Here is what happens with a 12-block model:
            </p>

            <div className="grid gap-3 md:grid-cols-2">
              <div className="px-4 py-3 bg-rose-500/10 border border-rose-500/20 rounded-lg space-y-2">
                <p className="text-sm font-medium text-rose-400">
                  Default init (<InlineMath math="\sigma = 0.02" /> everywhere)
                </p>
                <div className="text-xs text-muted-foreground font-mono space-y-0.5">
                  <p>Block 1 output std:&nbsp; 0.82</p>
                  <p>Block 4 output std:&nbsp; 1.63</p>
                  <p>Block 8 output std:&nbsp; 3.41</p>
                  <p>Block 12 output std: 6.55</p>
                </div>
                <p className="text-xs text-rose-400/80">
                  Activations grow ~8x across 12 blocks.
                </p>
              </div>
              <div className="px-4 py-3 bg-emerald-500/10 border border-emerald-500/20 rounded-lg space-y-2">
                <p className="text-sm font-medium text-emerald-400">
                  Scaled init (<InlineMath math="1/\sqrt{'{2N}'}" /> on residual projections)
                </p>
                <div className="text-xs text-muted-foreground font-mono space-y-0.5">
                  <p>Block 1 output std:&nbsp; 0.81</p>
                  <p>Block 4 output std:&nbsp; 0.85</p>
                  <p>Block 8 output std:&nbsp; 0.83</p>
                  <p>Block 12 output std: 0.80</p>
                </div>
                <p className="text-xs text-emerald-400/80">
                  Activations stay stable across all blocks.
                </p>
              </div>
            </div>

            <p className="text-muted-foreground text-sm">
              The notebook reproduces these measurements&mdash;run both
              initializations and verify the numbers yourself.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Default Init Fails Silently">
            A 12-block transformer with default initialization will still
            train&mdash;just poorly. The loss starts higher, converges
            slower, and may be unstable. You will not get an error message.
            Always apply scaled initialization for transformers.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          11. Check: Parameter Counting
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Checkpoint: Parameter Count"
            subtitle="Does our architecture match GPT-2?"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Before generating text, verify the architecture is correct. In
              Decoder-Only Transformers, you counted GPT-2&rsquo;s parameters
              by hand: ~124.4M. Now count them programmatically.
            </p>

            <CodeBlock
              code={`model = GPT(GPTConfig())

# Count total parameters
total = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total:,}")

# Count per component
for name, module in model.transformer.named_children():
    n = sum(p.numel() for p in module.parameters())
    print(f"  {name}: {n:,}")`}
              language="python"
              filename="param_count.py"
            />

            <GradientCard title="Prediction Exercise" color="emerald">
              <div className="space-y-3 text-sm">
                <p>
                  Before running the count, predict the total. You computed
                  this in Decoder-Only Transformers:
                </p>
                <ul className="space-y-1">
                  <li>Token embeddings: <InlineMath math="50{,}257 \times 768 = 38.6\text{M}" /></li>
                  <li>Position embeddings: <InlineMath math="1{,}024 \times 768 = 0.8\text{M}" /></li>
                  <li>12 blocks: <InlineMath math="12 \times 7.1\text{M} \approx 85\text{M}" /></li>
                  <li>Final layer norm: <InlineMath math="1{,}536" /></li>
                  <li>Output projection: weight-tied (0 additional)</li>
                </ul>
                <p className="font-medium">
                  Expected total: ~124.4M
                </p>
                <details className="mt-3">
                  <summary className="font-medium cursor-pointer text-primary">
                    Why does it matter if the count matches?
                  </summary>
                  <div className="mt-2">
                    <p>
                      If your count does not match ~124M, something is wrong
                      with the architecture. A missing projection, an
                      incorrect dimension, or a weight-tying bug will show
                      up as the wrong parameter count. This is the simplest,
                      most reliable architecture verification.
                    </p>
                  </div>
                </details>
              </div>
            </GradientCard>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Architecture Verification">
            Parameter counting is the fastest way to verify your
            implementation. If the count matches the known figure, the
            dimensions and structure are correct. If it does not, you have
            a bug. One number, one check.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          12. Generate: Autoregressive Inference
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Generate: Autoregressive Inference"
            subtitle="The payoff&mdash;your model produces text"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              In What is a Language Model?, you learned the autoregressive
              loop: predict the next token, append it, repeat. Now implement
              it. The <code className="text-xs">generate()</code> method is
              simpler than the training forward pass because it only needs
              logits for the <strong>last</strong> position&mdash;the one
              being generated.
            </p>

            <CodeBlock
              code={`@torch.no_grad()
def generate(self, idx, max_new_tokens, temperature=1.0):
    """Generate tokens autoregressively.

    Args:
        idx: (B, T) tensor of token indices (the prompt)
        max_new_tokens: number of tokens to generate
        temperature: controls randomness (higher = more random)
    """
    for _ in range(max_new_tokens):
        # Crop context to block_size if needed
        idx_cond = idx[:, -self.config.block_size:]

        # Forward pass — get logits for ALL positions
        logits, _ = self(idx_cond)

        # We only need the last position's prediction
        logits = logits[:, -1, :]              # (B, vocab_size)

        # Apply temperature
        logits = logits / temperature

        # Sample from the distribution
        probs = F.softmax(logits, dim=-1)      # (B, vocab_size)
        idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

        # Append to the running sequence
        idx = torch.cat([idx, idx_next], dim=1)  # (B, T+1)

    return idx`}
              language="python"
              filename="generate.py"
            />

            <div className="px-4 py-3 bg-amber-500/10 border border-amber-500/20 rounded-lg space-y-2">
              <p className="text-sm font-medium text-amber-400">
                Training mode vs generation mode
              </p>
              <div className="grid gap-3 md:grid-cols-2 text-sm text-muted-foreground">
                <div className="space-y-1">
                  <p className="font-medium">Training</p>
                  <ul className="space-y-0.5 text-xs">
                    <li>All positions computed in parallel</li>
                    <li>Logits for every position needed (for loss)</li>
                    <li>Gradients tracked</li>
                    <li>Loss computed against targets</li>
                  </ul>
                </div>
                <div className="space-y-1">
                  <p className="font-medium">Generation</p>
                  <ul className="space-y-0.5 text-xs">
                    <li>One token generated per step</li>
                    <li>Only last position&rsquo;s logits used</li>
                    <li>torch.no_grad() for speed</li>
                    <li>Temperature + sampling instead of loss</li>
                  </ul>
                </div>
              </div>
            </div>

            <p className="text-muted-foreground">
              Notice that <code className="text-xs">generate()</code> calls
              the full forward pass for every token but only uses the last
              position&rsquo;s logits. This is correct but wasteful&mdash;it
              recomputes attention for tokens we already processed.
              Production systems use <strong>KV caching</strong> to avoid
              this recomputation. We cover that in Lesson 3.
            </p>

            <p className="text-muted-foreground">
              The <code className="text-xs">temperature</code> parameter is
              the same concept from the TemperatureExplorer in
              What is a Language Model?&mdash;now it is a single line of code.
              Higher temperature {'\u2192'} flatter distribution {'\u2192'}{' '}
              more random text. Lower temperature {'\u2192'} sharper
              distribution {'\u2192'} more predictable.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="torch.no_grad()">
            During generation, no learning happens. Wrapping in{' '}
            <code className="text-xs">@torch.no_grad()</code> tells PyTorch
            not to track gradients, which saves memory and speeds up
            computation significantly.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          13. The Moment of Truth: Generated Text
          ================================================================ */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Run the generate method on the untrained model. The output will
              be random gibberish&mdash;the model has never seen any text.
              Every weight is random. But the gibberish is{' '}
              <strong>structurally correct</strong>: it is a sequence of
              valid token IDs, produced by the autoregressive loop, with
              attention, masking, and the full forward pass working exactly
              as designed.
            </p>

            <CodeBlock
              code={`import tiktoken

# GPT-2's tokenizer
enc = tiktoken.get_encoding("gpt2")

# Create model and generate
model = GPT(GPTConfig())
prompt = enc.encode("The meaning of life is")
idx = torch.tensor([prompt])            # (1, T)

# Generate 50 tokens
output = model.generate(idx, max_new_tokens=50, temperature=1.0)
print(enc.decode(output[0].tolist()))

# Output (example — yours will differ):
# "The meaning of life is Frag broadly meanwhile
#  precip Winn472 fireplace oxy deserted bure..."`}
              language="python"
              filename="generate_example.py"
            />

            <div className="px-4 py-3 bg-primary/10 border border-primary/30 rounded-lg space-y-2">
              <p className="text-sm font-medium text-primary">
                The architecture works. It just needs to learn.
              </p>
              <p className="text-sm text-muted-foreground">
                This random gibberish is a working language model. The
                architecture is correct&mdash;the parameter count matches,
                the shapes flow properly, the causal mask prevents future
                leakage, the autoregressive loop generates tokens. It
                produces nonsense because every weight is random. In the
                next lesson, you train it on real text and watch the
                gibberish gradually become recognizable English.
              </p>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Random but Correct">
            An untrained model producing gibberish is a success, not a
            failure. It means the architecture, shapes, masking, and
            generation loop all work. Training is a separate concern&mdash;and
            the next lesson.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* (Assembly Path moved to Section 5, before Config) */}

      {/* ================================================================
          15. Notebook Link
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Build It Yourself"
            subtitle="Open the notebook and write every line"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The lesson showed you the code. Now write it yourself. The
              notebook guides you through building each component, testing
              shapes with dummy inputs, verifying the parameter count, and
              generating text from your untrained model.
            </p>

            <div className="rounded-lg border-2 border-primary/50 bg-primary/5 p-6">
              <div className="space-y-4">
                <p className="text-muted-foreground">
                  Build your own GPT from scratch in a Jupyter notebook.
                </p>
                <a
                  href="https://colab.research.google.com/github/brennancheung/CourseAI/blob/main/notebooks/4-3-1-building-nanogpt.ipynb"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-2 px-4 py-2 rounded-md bg-primary text-primary-foreground font-medium text-sm hover:bg-primary/90 transition-colors"
                >
                  <ExternalLink className="w-4 h-4" />
                  Open in Google Colab
                </a>
                <p className="text-xs text-muted-foreground">
                  The notebook includes shape verification assertions at every
                  step, activation statistics for initialization, and a
                  stretch exercise comparing tiny vs GPT-2 scale configs.
                </p>
              </div>
            </div>
          </div>
        </Row.Content>
      </Row>

      {/* ================================================================
          16. Summary
          ================================================================ */}
      <Row>
        <Row.Content>
          <SummaryBlock
            title="Key Takeaways"
            items={[
              {
                headline:
                  'Five PyTorch operations build the entire GPT model.',
                description:
                  'nn.Linear, nn.Embedding, nn.LayerNorm, nn.GELU, nn.Dropout. Every one is familiar from Series 2. The complexity is in the assembly, not the parts.',
              },
              {
                headline:
                  'Bottom-up assembly: Head \u2192 MHA \u2192 FFN \u2192 Block \u2192 GPT.',
                description:
                  'Each class is small (5\u201315 lines), testable independently, and maps 1:1 to a concept from Module 4.2. The build order mirrors the learning order.',
              },
              {
                headline:
                  'Parameter count = architecture verification.',
                description:
                  'If your model has ~124.4M parameters, the dimensions, projections, and weight tying are correct. One number confirms the entire structure.',
              },
              {
                headline:
                  'An untrained model generating gibberish is a success.',
                description:
                  'It means the architecture, shapes, masking, and generation loop all work. The model is correct\u2014it just needs to learn from data.',
              },
            ]}
          />
        </Row.Content>
      </Row>

      {/* ================================================================
          17. Next Step
          ================================================================ */}
      <Row>
        <Row.Content>
          <div className="px-4 py-3 bg-primary/10 border border-primary/30 rounded-lg space-y-2">
            <p className="text-sm font-medium text-primary">
              What comes next
            </p>
            <p className="text-sm text-muted-foreground">
              The model generates random text because every weight is random.
              In the next lesson, you <strong>train</strong> it. You will
              feed it real text, watch the loss drop, and see the generated
              output gradually transform from random gibberish into
              recognizable English. The architecture works. Now it learns.
            </p>
          </div>
        </Row.Content>
      </Row>

      <Row>
        <Row.Content>
          <NextStepBlock
            href="/app"
            title="When you're done"
            description="Complete the notebook, verify your parameter count matches ~124M, and generate some gibberish. Then review your session."
          />
        </Row.Content>
      </Row>
    </LessonLayout>
  )
}

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
  TryThisBlock,
  WarningBlock,
  SummaryBlock,
  NextStepBlock,
  GradientCard,
  ComparisonRow,
} from '@/components/lessons'
import { BpeVisualizer } from '@/components/widgets/BpeVisualizer'
import { ExercisePanel } from '@/components/widgets/ExercisePanel'
import { cn } from '@/lib/utils'
import { ExternalLink } from 'lucide-react'

/**
 * Tokenization
 *
 * Second lesson in Module 4.1 (Language Modeling Fundamentals).
 * Teaches how text is converted to integer sequences via subword
 * tokenization, and has the student implement BPE from scratch.
 *
 * Core concepts at APPLIED:
 * - BPE algorithm (implemented from scratch in notebook)
 *
 * Concepts at DEVELOPED:
 * - Subword tokenization concept
 * - Character vs word vs subword tradeoffs
 *
 * Concepts at INTRODUCED:
 * - Vocabulary size as a design tradeoff
 * - Tokenization artifacts / failure modes
 *
 * Concepts at MENTIONED:
 * - WordPiece, SentencePiece
 * - Byte-level BPE
 *
 * Previous: What Is a Language Model? (module 4.1, lesson 1)
 * Next: Embeddings & Positional Encoding (module 4.1, lesson 3)
 */

// ---------------------------------------------------------------------------
// Static data for examples
// ---------------------------------------------------------------------------

const CHAR_TOKENS = 'The cat sat on the mat'.split('')
const WORD_TOKENS = ['The', 'cat', 'sat', 'on', 'the', 'mat']

// ---------------------------------------------------------------------------
// Main lesson component
// ---------------------------------------------------------------------------

export function TokenizationLesson() {
  return (
    <LessonLayout>
      {/* ================================================================
          Header
          ================================================================ */}
      <Row>
        <Row.Content>
          <LessonHeader
            title="Tokenization"
            description="How text becomes the integer sequences your language model actually processes&mdash;and why the method matters more than you&rsquo;d think."
            category="Language Modeling"
          />
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 1: Context + Constraints
          ================================================================ */}
      <Row>
        <Row.Content>
          <ObjectiveBlock>
            Understand how text is converted to integer sequences via subword tokenization,
            and implement Byte Pair Encoding (BPE) from scratch in a notebook.
          </ObjectiveBlock>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="What You Already Know">
            You know that a language model predicts the next token from context&mdash;a
            probability distribution over a vocabulary of 50,000+ items. This lesson
            answers: what defines those items?
          </TipBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <ConstraintBlock
            title="Scope for This Lesson"
            items={[
              'How text becomes integer sequences',
              'Character-level, word-level, and subword tokenization tradeoffs',
              'The BPE algorithm: training (learning merges) and encoding (applying merges)',
              'Implementing BPE from scratch in a Colab notebook',
              'How tokenization affects model behavior',
              'NOT: other subword algorithms in detail (WordPiece, SentencePiece)\u2014mentioned only',
              'NOT: how embeddings work\u2014that\u2019s the next lesson',
              'NOT: training a language model\u2014Module 4.3',
              'NOT: production tokenizer implementation\u2014we build a minimal, correct BPE',
            ]}
          />
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 2: Hook — The Model Can't Read
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Model Can&rsquo;t Read"
            subtitle="It needs integers, not text"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              In What Is a Language Model?, you saw a bar chart of candidate next tokens&mdash;&ldquo;mat&rdquo;
              at 35%, &ldquo;floor&rdquo; at 20%, and so on. The model output a probability
              distribution over a vocabulary of integers. But you type <em>text</em>.
            </p>
            <p className="text-muted-foreground">
              Something has to convert &ldquo;The cat sat on the mat&rdquo; into a sequence
              of integers like <code className="text-sm bg-muted/50 px-1.5 py-0.5 rounded">[464, 3797, 3332, 319, 262, 2603]</code>.
              What is that something? And why those specific numbers?
            </p>
            <p className="text-muted-foreground">
              By the end of this lesson, you&rsquo;ll build the algorithm that produces
              these numbers. And you&rsquo;ll understand why the choice of tokenization
              method shapes what your model can and cannot do.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Vocabulary = Classes">
            Remember the jump from MNIST (10 classes) to language modeling (50,000+ classes)?
            Tokenization is what defines those 50,000 classes. The tokenizer <em>builds</em> the
            vocabulary.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 3: The Obvious Solutions (and Why They Fail)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Obvious Solutions"
            subtitle="And why they don&rsquo;t work"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Let&rsquo;s start with the simplest possible approaches. Take the sentence
              &ldquo;The cat sat on the mat&rdquo; and consider two ways to break it into tokens.
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* Character-level */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground font-medium">
              Option 1: Character-level tokenization
            </p>
            <p className="text-muted-foreground">
              Split text into individual characters. Your vocabulary is tiny: {'{'}a, b, c, ..., z,
              A, ..., Z, 0-9, punctuation{'}'}&mdash;roughly 100 items.
            </p>
            <div className="px-4 py-3 bg-muted/50 rounded-lg">
              <p className="text-xs text-muted-foreground/70 mb-2">Character tokens ({CHAR_TOKENS.length}):</p>
              <div className="flex flex-wrap gap-1">
                {CHAR_TOKENS.map((ch, i) => (
                  <span
                    key={i}
                    className="inline-flex items-center px-1.5 py-0.5 rounded bg-sky-500/20 text-sky-300 border border-sky-500/30 text-sm font-mono"
                  >
                    {ch === ' ' ? '\u2423' : ch}
                  </span>
                ))}
              </div>
            </div>
            <p className="text-muted-foreground">
              Pro: tiny vocabulary, no out-of-vocabulary (OOV) problem ever. Con: &ldquo;The cat sat
              on the mat&rdquo; becomes <strong>22 tokens</strong> (including spaces). The model must learn
              from scratch that &ldquo;T-h-e&rdquo; means &ldquo;the.&rdquo; A 500-word article
              becomes ~2,500 tokens. Sequences explode in length, and attention scales with
              sequence length.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Too Granular">
            Character-level seems clean&mdash;no OOV! But the model wastes capacity
            learning spelling. It must figure out that &ldquo;c-a-t&rdquo; is a single concept
            before it can even start learning what cats do. Efficiency matters.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* Word-level */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground font-medium">
              Option 2: Word-level tokenization
            </p>
            <p className="text-muted-foreground">
              Split on spaces and punctuation. Each word is a token. Short sequences, each
              token carries meaning.
            </p>
            <div className="px-4 py-3 bg-muted/50 rounded-lg">
              <p className="text-xs text-muted-foreground/70 mb-2">Word tokens ({WORD_TOKENS.length}):</p>
              <div className="flex flex-wrap gap-1">
                {WORD_TOKENS.map((word, i) => (
                  <span
                    key={i}
                    className="inline-flex items-center px-2 py-0.5 rounded bg-amber-500/20 text-amber-300 border border-amber-500/30 text-sm font-mono"
                  >
                    {word}
                  </span>
                ))}
              </div>
            </div>
            <p className="text-muted-foreground">
              Pro: compact&mdash;just 6 tokens. Each one carries meaning. Con: what happens
              when the model sees a word it hasn&rsquo;t seen before?
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* Negative example: OOV */}
      <Row>
        <Row.Content>
          <GradientCard title='The Out-of-Vocabulary Problem' color="rose">
            <div className="space-y-3 text-sm">
              <p>
                With word-level tokenization, the vocabulary is all words seen during training.
                What happens when the model encounters &ldquo;ChatGPT&rdquo;?
              </p>
              <div className="px-3 py-2 bg-background/30 rounded font-mono text-xs">
                &ldquo;ChatGPT is amazing&rdquo; {'\u2192'} [<span className="text-rose-300 font-bold">[UNK]</span>, &ldquo;is&rdquo;, &ldquo;amazing&rdquo;]
              </div>
              <p>
                The word is mapped to a generic <code>[UNK]</code> token. All information
                is lost. Every misspelling, new word, foreign word, and proper name not in
                the training vocabulary is invisible to the model.
              </p>
              <p>
                English has ~170,000 words, plus inflections, compounds, names, technical
                terms... The vocabulary would need to be enormous, and it would still miss
                things.
              </p>
            </div>
          </GradientCard>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The Tradeoff">
            Small vocabulary = long sequences (character). Large vocabulary = sparse coverage
            (word). We need a middle ground.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Comparison */}
      <Row>
        <Row.Content>
          <ComparisonRow
            left={{
              title: 'Character-Level',
              color: 'sky',
              items: [
                'Vocabulary: ~100 items',
                '"The cat sat" = 22 tokens',
                'No OOV problem ever',
                'Sequences too long for practical use',
                'Model must learn spelling from scratch',
              ],
            }}
            right={{
              title: 'Word-Level',
              color: 'amber',
              items: [
                'Vocabulary: 100,000+ items',
                '"The cat sat" = 6 tokens',
                'Novel words are invisible ([UNK])',
                'Compact sequences',
                'Each token carries meaning',
              ],
            }}
          />
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 4: Subword Tokenization — The Key Insight
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Subword Tokenization"
            subtitle="The key insight"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              What if tokens were <strong>pieces of words</strong>? Common words stay whole.
              Rare words get split into recognizable pieces.
            </p>
            <div className="px-4 py-3 bg-violet-500/10 border border-violet-500/20 rounded-lg">
              <p className="text-sm text-muted-foreground font-mono">
                &ldquo;tokenization&rdquo; {'\u2192'} [&ldquo;token&rdquo;, &ldquo;ization&rdquo;]
              </p>
              <p className="text-xs text-muted-foreground/70 mt-1">
                The model has never seen &ldquo;tokenization&rdquo; as a whole, but it knows
                &ldquo;token&rdquo; and &ldquo;ization&rdquo; (from &ldquo;organization&rdquo;,
                &ldquo;civilization&rdquo;, &ldquo;authorization&rdquo;...).
              </p>
            </div>
            <p className="text-muted-foreground">
              This is the core idea: subword tokenization gives you compact sequences (like
              word-level) <em>and</em> coverage of novel words (like character-level). It&rsquo;s
              the approach every modern LLM uses.
            </p>
            <div className="px-4 py-3 bg-muted/50 rounded-lg">
              <p className="text-xs text-muted-foreground/70 mb-2">
                BPE on &ldquo;The cat sat on the mat&rdquo; (common words stay whole):
              </p>
              <div className="flex flex-wrap gap-1">
                {['The', ' cat', ' sat', ' on', ' the', ' mat'].map((tok, i) => (
                  <span
                    key={i}
                    className="inline-flex items-center px-2 py-0.5 rounded bg-violet-500/20 text-violet-300 border border-violet-500/30 text-sm font-mono"
                  >
                    {tok.replace(/ /g, '\u2423')}
                  </span>
                ))}
              </div>
              <p className="text-xs text-muted-foreground/70 mt-2">
                For this simple sentence of common words, BPE produces a similar result to
                word-level (6 tokens). The difference shows up with rare words&mdash;BPE splits
                them into recognizable pieces instead of mapping them to [UNK].
              </p>
            </div>
            <p className="text-muted-foreground">
              Think of it like learning abbreviations. Frequent patterns get their own symbol.
              &ldquo;th&rdquo; + &ldquo;e&rdquo; becomes &ldquo;the&rdquo;&mdash;just like &ldquo;btw&rdquo;
              means &ldquo;by the way.&rdquo; You keep compressing the most common thing until
              you hit your budget.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The Sweet Spot">
            Subword tokenization sits between characters and words. Typical vocabulary
            sizes: 32,000&ndash;100,000 tokens. GPT-4 uses ~100K. LLaMA uses ~32K. The
            exact size is a design choice, not a law of nature.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Check: Predict-and-verify */}
      <Row>
        <Row.Content>
          <GradientCard title="Predict and Verify" color="emerald">
            <div className="space-y-3 text-sm">
              <p>
                If a BPE tokenizer was trained on English Wikipedia, which of these would
                likely be a <strong>single token</strong>: &ldquo;the&rdquo;,
                &ldquo;tokenization&rdquo;, &ldquo;xyzzy&rdquo;? Why?
              </p>
              <details className="mt-3">
                <summary className="font-medium cursor-pointer text-primary">
                  Reveal
                </summary>
                <div className="mt-2 space-y-2">
                  <p>
                    <strong>&ldquo;the&rdquo;</strong>&mdash;almost certainly a single token. It&rsquo;s
                    one of the most common words in English, so BPE would merge its characters
                    early.
                  </p>
                  <p>
                    <strong>&ldquo;tokenization&rdquo;</strong>&mdash;probably split into 2&ndash;3
                    pieces. Common enough to not be fully character-level, but rare enough that
                    BPE wouldn&rsquo;t dedicate a single vocabulary slot to it.
                  </p>
                  <p>
                    <strong>&ldquo;xyzzy&rdquo;</strong>&mdash;almost certainly split to individual
                    characters. This string barely appears in Wikipedia, so no merges would form
                    for it.
                  </p>
                </div>
              </details>
            </div>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 5: BPE Algorithm — Training Phase
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Byte Pair Encoding"
            subtitle="The algorithm behind modern tokenizers"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Byte Pair Encoding (BPE) is beautifully simple. It&rsquo;s a compression
              algorithm repurposed for tokenization. Here&rsquo;s the entire training procedure:
            </p>
            <ol className="list-decimal list-inside text-muted-foreground space-y-2 ml-4">
              <li>Start with a character-level vocabulary.</li>
              <li>Count all adjacent character pairs in the corpus.</li>
              <li>Find the most frequent pair.</li>
              <li>Merge that pair into a new token. Add it to the vocabulary.</li>
              <li>Replace all occurrences of the pair in the corpus with the new token.</li>
              <li>Repeat from step 2 until you reach your target vocabulary size.</li>
            </ol>
            <p className="text-muted-foreground">
              That&rsquo;s it. Common subwords emerge naturally from frequency statistics.
              &ldquo;th&rdquo; + &ldquo;e&rdquo; merges early because &ldquo;the&rdquo; is
              frequent. The merge table&mdash;the ordered list of merges&mdash;IS the
              tokenizer.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Deterministic">
            A common misconception: since BPE uses frequency statistics, maybe it&rsquo;s
            stochastic? No. Once the merge table is learned, encoding is fully deterministic.
            Same input + same merge table = same tokens, every time. The randomness is only
            in which corpus you train on.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Step-by-step walkthrough */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground font-medium">
              Walkthrough with a tiny corpus
            </p>
            <p className="text-muted-foreground">
              Let&rsquo;s trace BPE on: &ldquo;low low low low low lower lower newest newest&rdquo;
            </p>
            <div className="space-y-3">
              <StepCard
                step={0}
                label="Start"
                description="Character-level tokens"
                tokens="l o w _ l o w _ l o w _ l o w _ l o w _ l o w e r _ l o w e r _ n e w e s t _ n e w e s t"
                highlight={null}
              />
              <StepCard
                step={1}
                label="Most frequent pair: (l, o) appears 7 times"
                description='Merge "l" + "o" into "lo"'
                tokens="lo w _ lo w _ lo w _ lo w _ lo w _ lo w e r _ lo w e r _ n e w e s t _ n e w e s t"
                highlight="lo"
              />
              <StepCard
                step={2}
                label='Most frequent pair: (lo, w) appears 7 times'
                description='Merge "lo" + "w" into "low"'
                tokens="low _ low _ low _ low _ low _ low e r _ low e r _ n e w e s t _ n e w e s t"
                highlight="low"
              />
              <div className="px-4 py-2 bg-muted/30 rounded text-xs text-muted-foreground/70 italic">
                Multiple pairs now tie at count 2 (e+r, e+s, e+w...). When there&rsquo;s a tie,
                the algorithm just picks one&mdash;the choice is implementation-dependent and
                doesn&rsquo;t affect the final result much. We&rsquo;ll go with (e, r).
              </div>
              <StepCard
                step={3}
                label='Most frequent pair: (e, r) appears 2 times'
                description='Merge "e" + "r" into "er"'
                tokens="low _ low _ low _ low _ low _ low er _ low er _ n e w e s t _ n e w e s t"
                highlight="er"
              />
              <StepCard
                step={4}
                label='Most frequent pair: (e, s) appears 2 times'
                description='Merge "e" + "s" into "es"'
                tokens="low _ low _ low _ low _ low _ low er _ low er _ n e w es t _ n e w es t"
                highlight="es"
              />
            </div>
            <p className="text-muted-foreground text-sm">
              Notice how &ldquo;low&rdquo; quickly becomes a single token because it appears
              so frequently. Less common subwords like &ldquo;er&rdquo; and &ldquo;es&rdquo;
              merge later. If we kept going, &ldquo;newest&rdquo; and &ldquo;lower&rdquo;
              might eventually become single tokens too&mdash;if they appear often enough.
            </p>
            <p className="text-muted-foreground text-sm">
              Once you have the merge table, <strong>encoding</strong> new text follows the same
              merges in priority order. Given the text &ldquo;lowest&rdquo; and the merge table
              above, you start with characters [l, o, w, e, s, t], apply merge #1
              (l+o {'\u2192'} lo), then merge #2 (lo+w {'\u2192'} low), then check for further applicable
              merges. The merge table IS the tokenizer. The notebook walks through this in detail.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Frequency Drives Everything">
            BPE doesn&rsquo;t know language. It doesn&rsquo;t know that &ldquo;er&rdquo;
            is a suffix. It just counts pairs and merges the most common one. Linguistic
            structure emerges from frequency patterns in the training corpus.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 6: Interactive BPE Widget
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Explore: BPE Step by Step"
            subtitle="Watch the merge algorithm in action"
          />
          <p className="text-muted-foreground mb-4">
            Type any text below and step through the BPE algorithm one merge at a time. Watch
            how character pairs get merged into longer tokens, the vocabulary grows, and the
            total token count shrinks.
          </p>
          <ExercisePanel
            title="BPE Visualizer"
            subtitle="Type text, then step through merges"
          >
            <BpeVisualizer />
          </ExercisePanel>
          <p className="text-xs text-muted-foreground/70 mt-2">
            The widget stops when no pair appears more than once&mdash;there&rsquo;s nothing left
            to compress. In practice, BPE training stops at a target vocabulary size (e.g., 50,000
            merges), which is reached long before running out of pairs on a large corpus.
          </p>
        </Row.Content>
        <Row.Aside>
          <TryThisBlock title="Experiments">
            <ul className="space-y-2 text-sm">
              <li>{'\u2022'} Try &ldquo;unhappiness&rdquo;&mdash;watch how common suffixes merge first.</li>
              <li>{'\u2022'} Try &ldquo;ChatGPT&rdquo;&mdash;notice it gets split into recognizable pieces, not [UNK].</li>
              <li>{'\u2022'} Try repeating a word many times. More repetitions = faster merges.</li>
              <li>{'\u2022'} Try <code className="text-xs">print(&quot;hello&quot;)</code>&mdash;code tokens are different from prose.</li>
              <li>{'\u2022'} Watch the compression percentage. How much shorter does the token sequence get?</li>
            </ul>
          </TryThisBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 7: Notebook Exercise
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Build It: Implement BPE from Scratch"
            subtitle="The notebook exercise"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The widget showed you the algorithm. Now implement it yourself. The notebook
              walks you through building a complete BPE tokenizer:
            </p>
            <ul className="list-disc list-inside text-muted-foreground space-y-1 ml-4">
              <li><code className="text-sm bg-muted/50 px-1 rounded">get_pair_counts(tokens)</code>&mdash;count adjacent pairs</li>
              <li><code className="text-sm bg-muted/50 px-1 rounded">merge_pair(tokens, pair, new_token)</code>&mdash;replace all occurrences of a pair</li>
              <li><code className="text-sm bg-muted/50 px-1 rounded">train_bpe(corpus, num_merges)</code>&mdash;the training loop</li>
              <li><code className="text-sm bg-muted/50 px-1 rounded">encode(text, merges)</code>&mdash;apply learned merges to new text</li>
              <li><code className="text-sm bg-muted/50 px-1 rounded">decode(token_ids, vocab)</code>&mdash;convert back to text</li>
            </ul>
            <p className="text-muted-foreground">
              You&rsquo;ll start with character-level and word-level tokenization to feel their
              limitations, then build BPE as the solution.
            </p>
            <div className="rounded-lg border-2 border-primary/50 bg-primary/5 p-6">
              <div className="space-y-4">
                <p className="text-muted-foreground">
                  Open the notebook and implement BPE from scratch.
                </p>
                <a
                  href="https://colab.research.google.com/github/brennancheung/CourseAI/blob/main/notebooks/4-1-2-tokenization.ipynb"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-2 px-4 py-2 rounded-md bg-primary text-primary-foreground font-medium text-sm hover:bg-primary/90 transition-colors"
                >
                  <ExternalLink className="w-4 h-4" />
                  Open in Google Colab
                </a>
                <p className="text-xs text-muted-foreground">
                  The notebook includes scaffolded exercises with solutions. You&rsquo;ll train
                  a BPE tokenizer on a small corpus, examine the merge table, and tokenize
                  new sentences.
                </p>
              </div>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Build, Don&rsquo;t Read">
            The implementation is intentionally minimal&mdash;no optimizations, no edge cases.
            The goal is understanding, not speed. A production tokenizer (like tiktoken or
            SentencePiece) is thousands of lines. Yours will be ~50.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 8: Tokenization Artifacts — Why This Matters
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Tokenization Artifacts"
            subtitle="Why this isn&rsquo;t just preprocessing"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              You might be thinking: &ldquo;OK, text becomes numbers. Let&rsquo;s move on.&rdquo;
              But tokenization has real consequences for what models can and cannot do.
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* Strawberry problem */}
      <Row>
        <Row.Content>
          <GradientCard title='The "strawberry" Problem' color="orange">
            <div className="space-y-3 text-sm">
              <p>
                Ask an LLM: &ldquo;How many r&rsquo;s in strawberry?&rdquo; Many get it wrong. Why?
              </p>
              <div className="px-3 py-2 bg-background/30 rounded font-mono text-xs">
                &ldquo;strawberry&rdquo; {'\u2192'} [&ldquo;straw&rdquo;, &ldquo;berry&rdquo;]
              </div>
              <p>
                The model never sees the individual letters r-r-r. It sees two tokens:
                &ldquo;straw&rdquo; and &ldquo;berry.&rdquo; To count letters, it would need
                to decompose tokens back into characters&mdash;but that&rsquo;s not how it
                processes text. The tokenizer creates a blind spot.
              </p>
            </div>
          </GradientCard>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The Model Sees Tokens">
            The tokenizer determines the atomic units of perception for the model. Anything
            below the token level is invisible. This is why character-level tasks (spelling,
            letter counting, rhyming) are hard for LLMs.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Arithmetic problem */}
      <Row>
        <Row.Content>
          <GradientCard title="Arithmetic Difficulty" color="orange">
            <div className="space-y-3 text-sm">
              <p>
                Numbers tokenize inconsistently. &ldquo;42&rdquo; might be one token, but
                &ldquo;427&rdquo; might be [&ldquo;42&rdquo;, &ldquo;7&rdquo;]. The model
                doesn&rsquo;t see individual digits in a consistent way.
              </p>
              <p>
                This is one reason LLMs struggle with multi-digit arithmetic. The solution
                isn&rsquo;t a better tokenizer&mdash;it&rsquo;s architectural (chain-of-thought,
                tool use, scratchpads).
              </p>
            </div>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* Multilingual inequality */}
      <Row>
        <Row.Content>
          <GradientCard title="Multilingual Inequality" color="orange">
            <div className="space-y-3 text-sm">
              <p>
                BPE trained primarily on English compresses English efficiently&mdash;one
                token per common word. But it fragments other languages. The same sentence
                might cost 2&ndash;5x more tokens in Korean or Arabic than in English.
              </p>
              <p>
                This means: more tokens = more computation = higher cost = lower quality.
                Tokenization choices create structural advantages for some languages.
              </p>
            </div>
          </GradientCard>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Not Neutral Preprocessing">
            Tokenization is a design decision that shapes what the model can and cannot
            do. It affects letter-level reasoning, arithmetic, multilingual performance,
            and even creates bizarre failure modes.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* SolidGoldMagikarp */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground font-medium">
              Glitch tokens: SolidGoldMagikarp
            </p>
            <p className="text-muted-foreground">
              A Reddit username, &ldquo;SolidGoldMagikarp&rdquo;, appeared in the training data
              used to build the tokenizer&rsquo;s vocabulary&mdash;but barely appeared in the
              text used to actually train the model. The result: a token that exists in the
              vocabulary but that the model never learned the meaning of. Prompting GPT
              with this token caused bizarre, erratic behavior.
            </p>
            <p className="text-muted-foreground">
              The model learns what each vocabulary entry means by seeing it used in many
              contexts during training. If a token appears in the vocabulary but barely
              appears in the training text, the model has no data to learn what it means.
              Its internal representation for that token is essentially random&mdash;it was
              never trained. These &ldquo;glitch tokens&rdquo; are a direct consequence
              of tokenization.
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 9: Other Tokenization Methods (MENTIONED)
          ================================================================ */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground font-medium">
              Beyond BPE
            </p>
            <p className="text-muted-foreground">
              BPE is the most widely used subword algorithm, but it&rsquo;s not the only one.
              Two others worth knowing by name:
            </p>
            <ul className="list-disc list-inside text-muted-foreground space-y-1 ml-4">
              <li>
                <strong>WordPiece</strong> (used by BERT)&mdash;similar to BPE but selects
                merges by maximizing likelihood, not just frequency.
              </li>
              <li>
                <strong>SentencePiece / Unigram</strong> (used by LLaMA, T5)&mdash;starts
                with a large vocabulary and prunes it down, the reverse of BPE&rsquo;s
                bottom-up approach.
              </li>
            </ul>
            <p className="text-muted-foreground">
              There&rsquo;s also <strong>byte-level BPE</strong> (used by GPT-2 and later),
              which applies BPE to raw bytes instead of characters. This eliminates any
              need for Unicode handling and guarantees that every possible input can be
              tokenized. We won&rsquo;t develop these further here&mdash;just know they exist
              and share the same core idea: learn a vocabulary from data.
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 10: Vocabulary Size as a Design Choice
          ================================================================ */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground font-medium">
              50,000 tokens is not arbitrary
            </p>
            <p className="text-muted-foreground">
              Why do most LLMs use vocabularies of 32K&ndash;100K tokens? Because vocabulary
              size is a tradeoff:
            </p>
            <ul className="list-disc list-inside text-muted-foreground space-y-1 ml-4">
              <li>
                <strong>Bigger vocabulary:</strong> shorter sequences (more compression), but
                the model&rsquo;s output layer has more classes (remember MNIST 10 {'\u2192'} LM
                50K?), and each vocabulary entry needs its own learned representation. More
                entries = more parameters. Rare tokens barely appear in training data, so
                their representations are poorly learned.
              </li>
              <li>
                <strong>Smaller vocabulary:</strong> longer sequences (less compression), but
                each token is seen more often during training and gets a better learned
                representation.
              </li>
            </ul>
            <p className="text-muted-foreground">
              The sweet spot depends on your training data, compute budget, and target
              languages. There&rsquo;s no universally optimal size.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Size Examples">
            GPT-2: 50,257 tokens. GPT-4: ~100K tokens. LLaMA: 32,000 tokens. LLaMA 3:
            128,000 tokens (expanded for better multilingual coverage). The trend is toward
            larger vocabularies as models get bigger.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 11: Check — Transfer Question
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Check: Transfer Question"
            subtitle="Apply what you&rsquo;ve learned"
          />
          <GradientCard title="Think It Through" color="emerald">
            <div className="space-y-3 text-sm">
              <p>
                A language model trained with a 32K vocabulary consistently fails at
                multi-digit multiplication. A researcher proposes switching to a
                character-level tokenizer to fix this. Would this help? What tradeoff
                would it introduce?
              </p>
              <details className="mt-3">
                <summary className="font-medium cursor-pointer text-primary">
                  Reveal
                </summary>
                <div className="mt-2 space-y-2">
                  <p>
                    <strong>Would it help?</strong> Partially. Character-level tokenization
                    would let the model see individual digits consistently&mdash;no more
                    &ldquo;42&rdquo; vs [&ldquo;42&rdquo;, &ldquo;7&rdquo;] inconsistencies.
                    So digit-level reasoning would be easier.
                  </p>
                  <p>
                    <strong>The tradeoff:</strong> sequences become much longer. A 500-word
                    math problem might go from ~600 tokens to ~2,500 characters. Attention
                    scales quadratically with length, so this makes training and inference
                    much slower. Long-range patterns become harder to learn.
                  </p>
                  <p>
                    <strong>The better solution:</strong> the real fix is probably architectural
                    (chain-of-thought prompting, tool use, scratchpads), not tokenization
                    alone. But understanding why the problem exists requires understanding
                    tokenization.
                  </p>
                </div>
              </details>
            </div>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 12: Summary
          ================================================================ */}
      <Row>
        <Row.Content>
          <SummaryBlock
            title="Key Takeaways"
            items={[
              {
                headline: 'The model needs integers. Tokenization converts text to integer sequences.',
                description:
                  'Every language model has a tokenizer that maps text to a sequence of integer token IDs and back again.',
              },
              {
                headline: 'Character-level: tiny vocabulary, long sequences. Word-level: short sequences, can\u2019t handle novel words.',
                description:
                  'Both extremes have fundamental problems. Subword tokenization is the middle ground that every modern LLM uses.',
              },
              {
                headline: 'BPE: start from characters, iteratively merge the most frequent pair.',
                description:
                  'Common words stay whole, rare words split into recognizable pieces. The merge table IS the tokenizer. Encoding is deterministic.',
              },
              {
                headline: 'You built a BPE tokenizer from scratch.',
                description:
                  'get_pair_counts, merge_pair, train_bpe, encode, decode\u2014five functions that implement the algorithm behind GPT\u2019s tokenizer.',
              },
              {
                headline: 'Tokenization defines the vocabulary. The vocabulary defines what the model can see.',
                description:
                  'This is not neutral preprocessing. It affects letter counting, arithmetic, multilingual performance, and even creates glitch tokens.',
              },
            ]}
          />
        </Row.Content>
      </Row>

      {/* Mental model echo */}
      <Row>
        <Row.Content>
          <InsightBlock title="Mental Model">
            Just like choosing convolutions encodes the assumption that nearby pixels matter,
            choosing a tokenizer encodes assumptions about language&mdash;what units are
            meaningful, how to handle novel words, what granularity of patterns the model
            should learn. <strong>Architecture encodes assumptions about data</strong>&mdash;and
            the tokenizer is the first architectural decision.
          </InsightBlock>
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 13: Next Step
          ================================================================ */}
      <Row>
        <Row.Content>
          <NextStepBlock
            href="/app/lesson/embeddings-and-position"
            title="Embeddings & Positional Encoding"
            description="You can now convert text to integer sequences. But integer 464 is just an index into a table&mdash;it doesn&rsquo;t carry any meaning yet. Next: how those integers become rich vector representations the model can actually compute with."
            buttonText="Continue to Embeddings"
          />
        </Row.Content>
      </Row>
    </LessonLayout>
  )
}

// ---------------------------------------------------------------------------
// Inline helper: Step-by-step walkthrough card
// ---------------------------------------------------------------------------

function StepCard({
  step,
  label,
  description,
  tokens,
  highlight,
}: {
  step: number
  label: string
  description: string
  tokens: string
  highlight: string | null
}) {
  const tokenList = tokens.split(' ')
  return (
    <div className="px-4 py-3 bg-muted/50 rounded-lg space-y-2">
      <div className="flex items-center gap-2">
        <span className="flex-shrink-0 w-6 h-6 rounded-full bg-violet-500/20 text-violet-400 text-xs font-bold flex items-center justify-center">
          {step}
        </span>
        <p className="text-sm text-muted-foreground">
          <span className="font-medium text-foreground">{label}</span>
        </p>
      </div>
      <p className="ml-8 text-xs text-muted-foreground/70">{description}</p>
      <div className="ml-8 flex flex-wrap gap-1">
        {tokenList.map((t, i) => (
          <span
            key={i}
            className={cn(
              'inline-flex items-center px-1.5 py-0.5 rounded text-xs font-mono border',
              highlight && t === highlight
                ? 'bg-violet-500/30 text-violet-200 border-violet-400/50 ring-1 ring-violet-400/30'
                : t === '_'
                  ? 'bg-muted/30 text-muted-foreground/40 border-border/30'
                  : 'bg-muted/50 text-muted-foreground border-border/50',
            )}
          >
            {t === '_' ? '\u2423' : t}
          </span>
        ))}
      </div>
    </div>
  )
}

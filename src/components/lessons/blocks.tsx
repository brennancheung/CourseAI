import { ReactNode } from 'react'
import {
  Target,
  Fence,
  Lightbulb,
  AlertTriangle,
  Sparkles,
  ArrowRight,
  Play,
  BookOpen,
  Check,
  Clock,
} from 'lucide-react'

/**
 * Lesson Block Components
 *
 * Visual language for different content types in lessons.
 * Each block has distinct styling to aid quick recognition.
 */

// =============================================================================
// OBJECTIVE BLOCK
// What you'll learn - appears at the top of lessons
// =============================================================================

interface ObjectiveBlockProps {
  children: ReactNode
}

export function ObjectiveBlock({ children }: ObjectiveBlockProps) {
  return (
    <div className="flex gap-3 items-start p-4 rounded-lg bg-primary/5 border border-primary/20">
      <Target className="w-5 h-5 text-primary flex-shrink-0 mt-0.5" />
      <div className="text-sm">
        <span className="font-medium text-primary">Objective: </span>
        <span className="text-foreground">{children}</span>
      </div>
    </div>
  )
}

// =============================================================================
// CONSTRAINT BLOCK
// Scope boundaries - what you're NOT doing
// =============================================================================

interface ConstraintBlockProps {
  items: string[]
  title?: string
}

export function ConstraintBlock({
  items,
  title = 'Constraints',
}: ConstraintBlockProps) {
  return (
    <div className="relative pl-4 border-l-4 border-amber-500/70 bg-amber-500/5 rounded-r-lg py-4 pr-4">
      <div className="flex items-center gap-2 mb-2">
        <Fence className="w-4 h-4 text-amber-600 dark:text-amber-400" />
        <h3 className="font-semibold text-sm text-amber-700 dark:text-amber-300">
          {title}
        </h3>
      </div>
      <p className="text-xs text-muted-foreground mb-3">
        These limits help you focus. Don&apos;t try to do more than this.
      </p>
      <ul className="space-y-1.5">
        {items.map((item, i) => (
          <li key={i} className="text-sm flex items-start gap-2">
            <span className="text-amber-500 mt-1 text-xs">&#x25A0;</span>
            <span>{item}</span>
          </li>
        ))}
      </ul>
    </div>
  )
}

// =============================================================================
// STEP LIST
// Numbered instructions - the core "what to do"
// =============================================================================

interface StepListProps {
  steps: string[]
  title?: string
}

export function StepList({ steps, title = 'Instructions' }: StepListProps) {
  return (
    <div className="space-y-4">
      <h2 className="font-semibold text-lg">{title}</h2>
      <ol className="space-y-4">
        {steps.map((step, i) => (
          <li key={i} className="flex gap-4">
            <span className="flex-shrink-0 w-7 h-7 rounded-full bg-primary/10 text-primary flex items-center justify-center text-sm font-semibold">
              {i + 1}
            </span>
            <p className="text-sm pt-1 leading-relaxed">{step}</p>
          </li>
        ))}
      </ol>
    </div>
  )
}

// =============================================================================
// TIP BLOCK
// Helpful hints and advice
// =============================================================================

interface TipBlockProps {
  children: ReactNode
  title?: string
}

export function TipBlock({ children, title = 'Tip' }: TipBlockProps) {
  return (
    <div className="relative pl-4 border-l-4 border-sky-500/70 bg-sky-500/5 rounded-r-lg py-3 pr-4">
      <div className="flex items-center gap-2 mb-1">
        <Lightbulb className="w-4 h-4 text-sky-600 dark:text-sky-400" />
        <h3 className="font-medium text-sm text-sky-700 dark:text-sky-300">
          {title}
        </h3>
      </div>
      <div className="text-sm text-muted-foreground">{children}</div>
    </div>
  )
}

// =============================================================================
// WARNING BLOCK
// Common mistakes and cautions
// =============================================================================

interface WarningBlockProps {
  children: ReactNode
  title?: string
}

export function WarningBlock({
  children,
  title = 'Watch out',
}: WarningBlockProps) {
  return (
    <div className="relative pl-4 border-l-4 border-rose-500/70 bg-rose-500/5 rounded-r-lg py-3 pr-4">
      <div className="flex items-center gap-2 mb-1">
        <AlertTriangle className="w-4 h-4 text-rose-600 dark:text-rose-400" />
        <h3 className="font-medium text-sm text-rose-700 dark:text-rose-300">
          {title}
        </h3>
      </div>
      <div className="text-sm text-muted-foreground">{children}</div>
    </div>
  )
}

// =============================================================================
// INSIGHT BLOCK
// Key takeaways and important concepts
// =============================================================================

interface InsightBlockProps {
  children: ReactNode
  title?: string
}

export function InsightBlock({
  children,
  title = 'Key Insight',
}: InsightBlockProps) {
  return (
    <div className="relative pl-4 border-l-4 border-violet-500/70 bg-violet-500/5 rounded-r-lg py-3 pr-4">
      <div className="flex items-center gap-2 mb-1">
        <Sparkles className="w-4 h-4 text-violet-600 dark:text-violet-400" />
        <h3 className="font-medium text-sm text-violet-700 dark:text-violet-300">
          {title}
        </h3>
      </div>
      <div className="text-sm text-muted-foreground">{children}</div>
    </div>
  )
}

// =============================================================================
// TRY THIS BLOCK
// Interactive prompts and exercises
// =============================================================================

interface TryThisBlockProps {
  children: ReactNode
  title?: string
}

export function TryThisBlock({
  children,
  title = 'Try This',
}: TryThisBlockProps) {
  return (
    <div className="rounded-lg border-2 border-dashed border-emerald-500/50 bg-emerald-500/5 p-4">
      <div className="flex items-center gap-2 mb-2">
        <Play className="w-4 h-4 text-emerald-600 dark:text-emerald-400" />
        <h3 className="font-medium text-sm text-emerald-700 dark:text-emerald-300">
          {title}
        </h3>
      </div>
      <div className="text-sm">{children}</div>
    </div>
  )
}

// =============================================================================
// CONCEPT BLOCK
// Detailed theory and explanations - works in sidebar or main content
// =============================================================================

interface ConceptBlockProps {
  title: string
  children: ReactNode
}

export function ConceptBlock({ title, children }: ConceptBlockProps) {
  return (
    <div className="rounded-lg border bg-card p-4">
      <div className="flex items-center gap-2 mb-2">
        <BookOpen className="w-4 h-4 text-muted-foreground" />
        <h3 className="font-medium text-sm">{title}</h3>
      </div>
      <div className="text-sm text-muted-foreground">{children}</div>
    </div>
  )
}

// =============================================================================
// CONCEPT CARD
// Brief concept overview - for grids showing multiple related concepts
// =============================================================================

interface ConceptCardProps {
  title: string
  description: string
  icon?: React.ReactNode
}

export function ConceptCard({ title, description, icon }: ConceptCardProps) {
  return (
    <div className="rounded-lg border bg-card/50 p-4 flex gap-3">
      {icon && (
        <div className="w-8 h-8 rounded-md bg-primary/10 flex items-center justify-center flex-shrink-0">
          <span className="text-primary">{icon}</span>
        </div>
      )}
      <div>
        <p className="font-medium text-sm">{title}</p>
        <p className="text-xs text-muted-foreground mt-0.5">{description}</p>
      </div>
    </div>
  )
}

// =============================================================================
// NEXT STEP BLOCK
// Session completion and what to do after
// =============================================================================

interface NextStepBlockProps {
  href: string
  title?: string
  description?: string
  buttonText?: string
}

export function NextStepBlock({
  href,
  title = "When you're done",
  description = 'Review your session with Claude Code to capture what you learned.',
  buttonText = 'Complete Session',
}: NextStepBlockProps) {
  return (
    <div className="rounded-lg bg-gradient-to-br from-primary/10 via-primary/5 to-transparent border border-primary/20 p-5">
      <div className="flex items-start gap-3">
        <div className="w-8 h-8 rounded-full bg-primary/20 flex items-center justify-center flex-shrink-0">
          <ArrowRight className="w-4 h-4 text-primary" />
        </div>
        <div className="space-y-3">
          <div>
            <h3 className="font-semibold">{title}</h3>
            <p className="text-sm text-muted-foreground mt-1">{description}</p>
          </div>
          <a
            href={href}
            className="inline-flex items-center justify-center rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90 transition-colors"
          >
            {buttonText}
          </a>
        </div>
      </div>
    </div>
  )
}

// =============================================================================
// SIDEBAR SECTION
// Wrapper for sidebar content groups
// =============================================================================

interface SidebarSectionProps {
  title?: string
  children: ReactNode
}

export function SidebarSection({ title, children }: SidebarSectionProps) {
  return (
    <div className="space-y-3">
      {title && (
        <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">
          {title}
        </h3>
      )}
      <div className="space-y-3">{children}</div>
    </div>
  )
}

// =============================================================================
// SUMMARY BLOCK
// Key takeaways with gradient background
// =============================================================================

interface SummaryItem {
  headline: string
  description: string
}

interface SummaryBlockProps {
  title?: string
  items: SummaryItem[]
}

export function SummaryBlock({
  title = 'Key Takeaways',
  items,
}: SummaryBlockProps) {
  return (
    <div className="bg-gradient-to-r from-primary/15 to-violet-500/10 rounded-lg p-6 border border-primary/30">
      <h3 className="font-semibold text-sm text-primary mb-4">{title}</h3>
      <ul className="space-y-3 text-muted-foreground">
        {items.map((item, i) => (
          <li key={i} className="flex gap-2">
            <Check className="w-4 h-4 text-emerald-400 flex-shrink-0 mt-0.5" />
            <span>
              <strong className="text-foreground">{item.headline}</strong>{' '}
              {item.description}
            </span>
          </li>
        ))}
      </ul>
    </div>
  )
}

// =============================================================================
// SECTION HEADER
// Section titles within lessons
// =============================================================================

interface SectionHeaderProps {
  title: string
  subtitle?: string
}

export function SectionHeader({ title, subtitle }: SectionHeaderProps) {
  return (
    <div className="space-y-1">
      <h2 className="text-xl font-semibold text-foreground">{title}</h2>
      {subtitle && (
        <p className="text-sm text-muted-foreground">{subtitle}</p>
      )}
    </div>
  )
}

// =============================================================================
// LESSON HEADER
// Top of lessons with title, description, and optional badges
// =============================================================================

interface LessonHeaderProps {
  title: string
  description: string
  duration?: string
  category?: string
}

export function LessonHeader({
  title,
  description,
  duration,
  category,
}: LessonHeaderProps) {
  return (
    <div className="space-y-4">
      <div className="space-y-2">
        <h1 className="text-2xl font-bold text-foreground">{title}</h1>
        <p className="text-muted-foreground">{description}</p>
      </div>
      {(duration || category) && (
        <div className="flex gap-2">
          {category && (
            <span className="inline-flex items-center gap-1 rounded-md bg-primary/10 px-2 py-1 text-xs font-medium text-primary">
              {category}
            </span>
          )}
          {duration && (
            <span className="inline-flex items-center gap-1 rounded-md bg-muted px-2 py-1 text-xs font-medium text-muted-foreground">
              <Clock className="w-3 h-3" />
              {duration}
            </span>
          )}
        </div>
      )}
    </div>
  )
}

// =============================================================================
// GRADIENT CARDS
// Visually prominent cards with colored gradients for categorization
//
// WHEN TO USE WHICH:
//
// GradientCard - Single standalone card with gradient background
//   Use for: Feature highlights, categorized info, any content needing visual prominence
//   Colors convey meaning: amber=warning, blue=primary, cyan=info, orange=highlight, purple=insight
//
// ComparisonRow - Two side-by-side gradient cards
//   Use for: A vs B comparisons, contrasting approaches, before/after
//   Always use with two complementary colors (e.g., amber+blue, cyan+orange)
//
// PhaseCard - Numbered sequential card with icon
//   Use for: Multi-step processes, timeline phases, song structure sections
//   Colors should progress logically (e.g., cyan→orange→purple for build→hit→tail)
//
// =============================================================================

type GradientColor = 'amber' | 'blue' | 'cyan' | 'orange' | 'purple' | 'emerald' | 'rose' | 'violet' | 'sky'

const gradientColorMap: Record<GradientColor, { bg: string; title: string }> = {
  amber: { bg: 'from-amber-500/10', title: 'text-amber-400' },
  blue: { bg: 'from-blue-500/10', title: 'text-blue-400' },
  cyan: { bg: 'from-cyan-500/10', title: 'text-cyan-400' },
  orange: { bg: 'from-orange-500/10', title: 'text-orange-400' },
  purple: { bg: 'from-purple-500/10', title: 'text-purple-400' },
  emerald: { bg: 'from-emerald-500/10', title: 'text-emerald-400' },
  rose: { bg: 'from-rose-500/10', title: 'text-rose-400' },
  violet: { bg: 'from-violet-500/10', title: 'text-violet-400' },
  sky: { bg: 'from-sky-500/10', title: 'text-sky-400' },
}

// =============================================================================
// GRADIENT CARD
// Single card with gradient background - for categorized content
// =============================================================================

interface GradientCardProps {
  title: string
  color: GradientColor
  children: ReactNode
  direction?: 'vertical' | 'horizontal'
}

export function GradientCard({
  title,
  color,
  children,
  direction = 'vertical',
}: GradientCardProps) {
  const colors = gradientColorMap[color]
  const gradientDir = direction === 'vertical' ? 'bg-gradient-to-b' : 'bg-gradient-to-r'

  return (
    <div className={`p-4 rounded-lg border ${gradientDir} ${colors.bg} to-transparent`}>
      <h4 className={`font-semibold text-sm mb-2 ${colors.title}`}>{title}</h4>
      <div className="text-xs text-muted-foreground">{children}</div>
    </div>
  )
}

// =============================================================================
// COMPARISON ROW
// Two side-by-side gradient cards for A vs B comparisons
// =============================================================================

interface ComparisonItem {
  title: string
  color: GradientColor
  items: string[]
}

interface ComparisonRowProps {
  left: ComparisonItem
  right: ComparisonItem
}

export function ComparisonRow({ left, right }: ComparisonRowProps) {
  const leftColors = gradientColorMap[left.color]
  const rightColors = gradientColorMap[right.color]

  return (
    <div className="grid gap-4 sm:grid-cols-2">
      <div className={`p-4 rounded-lg border bg-gradient-to-b ${leftColors.bg} to-transparent`}>
        <h4 className={`font-semibold text-sm mb-2 ${leftColors.title}`}>
          {left.title}
        </h4>
        <ul className="text-xs text-muted-foreground space-y-1">
          {left.items.map((item, i) => (
            <li key={i}>• {item}</li>
          ))}
        </ul>
      </div>
      <div className={`p-4 rounded-lg border bg-gradient-to-b ${rightColors.bg} to-transparent`}>
        <h4 className={`font-semibold text-sm mb-2 ${rightColors.title}`}>
          {right.title}
        </h4>
        <ul className="text-xs text-muted-foreground space-y-1">
          {right.items.map((item, i) => (
            <li key={i}>• {item}</li>
          ))}
        </ul>
      </div>
    </div>
  )
}

// =============================================================================
// PHASE CARD
// Numbered sequential card with icon - for multi-step processes
// =============================================================================

interface PhaseCardProps {
  number: number
  title: string
  subtitle?: string
  color: GradientColor
  children: ReactNode
}

export function PhaseCard({
  number,
  title,
  subtitle,
  color,
  children,
}: PhaseCardProps) {
  const colors = gradientColorMap[color]

  return (
    <div className={`p-4 rounded-lg border bg-gradient-to-r ${colors.bg} to-transparent`}>
      <div className="flex items-center gap-3 mb-3">
        <div
          className={`w-10 h-10 rounded-full bg-${color}-500/20 flex items-center justify-center ${colors.title} font-bold`}
        >
          {number}
        </div>
        <div>
          <h4 className="font-semibold">{title}</h4>
          {subtitle && (
            <p className="text-xs text-muted-foreground">{subtitle}</p>
          )}
        </div>
      </div>
      <div className="text-sm text-muted-foreground">{children}</div>
    </div>
  )
}

// =============================================================================
// MODULE COMPLETE BLOCK
// Celebration block for end of module
// =============================================================================

interface ModuleCompleteBlockProps {
  /** Module number (e.g., "1.1") */
  module: string
  /** Module title (e.g., "The Learning Problem") */
  title: string
  /** List of achievements/concepts learned */
  achievements: string[]
  /** Next module number (e.g., "1.2") */
  nextModule: string
  /** Next module title */
  nextTitle: string
}

export function ModuleCompleteBlock({
  module,
  title,
  achievements,
  nextModule,
  nextTitle,
}: ModuleCompleteBlockProps) {
  return (
    <div className="rounded-lg bg-gradient-to-br from-emerald-500/10 via-emerald-500/5 to-transparent border border-emerald-500/20 p-6">
      <h3 className="font-semibold text-lg text-emerald-400 mb-2">
        🎉 Module {module} Complete!
      </h3>
      <p className="text-muted-foreground mb-4">
        You&apos;ve learned the core concepts of {title.toLowerCase()}:
      </p>
      <ul className="list-disc list-inside text-muted-foreground space-y-1 ml-4 mb-4">
        {achievements.map((achievement, i) => (
          <li key={i}>{achievement}</li>
        ))}
      </ul>
      <p className="text-muted-foreground">
        Next up: <strong>Module {nextModule} — {nextTitle}</strong>
      </p>
    </div>
  )
}


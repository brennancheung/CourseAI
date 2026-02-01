import { ReactNode } from 'react'
import { Clock } from 'lucide-react'
import { ASIDE_WIDTH, GAP } from '@/components/layout/Row'

interface LessonLayoutProps {
  children: ReactNode
}

/**
 * Container for lesson content.
 * Provides consistent max-width and spacing.
 * Use Row from @/components/layout/Row for content + aside pairs.
 */
export function LessonLayout({ children }: LessonLayoutProps) {
  return <div className="space-y-8">{children}</div>
}

interface LessonRowProps {
  children: ReactNode
  aside?: ReactNode
}

/**
 * @deprecated Use `Row` from `@/components/layout/Row` instead.
 * LessonRow doesn't auto-inject empty asides, causing layout breaks.
 *
 * Migration:
 * ```tsx
 * // Old
 * <LessonRow aside={<Tip />}>content</LessonRow>
 *
 * // New
 * <Row>
 *   <Row.Content>content</Row.Content>
 *   <Row.Aside><Tip /></Row.Aside>
 * </Row>
 * ```
 */
export function LessonRow({ children, aside }: LessonRowProps) {
  if (process.env.NODE_ENV === 'development') {
    console.warn(
      'LessonRow is deprecated. Use Row from @/components/layout/Row instead.'
    )
  }
  return (
    <div className={`flex flex-col lg:flex-row ${GAP} items-start`}>
      <div className="flex-1 min-w-0 space-y-6">{children}</div>
      <aside className={`hidden lg:block ${ASIDE_WIDTH} flex-shrink-0 space-y-4`}>
        {aside}
      </aside>
    </div>
  )
}

interface ExpandableRowProps {
  children: ReactNode
  aside?: ReactNode
  isExpanded: boolean
}

/**
 * @deprecated Use `Row` from `@/components/layout/Row` instead.
 *
 * Migration:
 * ```tsx
 * <Row>
 *   <Row.Content>content</Row.Content>
 *   {isExpanded && <Row.Aside>tips</Row.Aside>}
 * </Row>
 * ```
 */
export function ExpandableRow({ children, aside, isExpanded }: ExpandableRowProps) {
  if (process.env.NODE_ENV === 'development') {
    console.warn(
      'ExpandableRow is deprecated. Use Row from @/components/layout/Row instead.'
    )
  }
  return (
    <div className={`flex flex-col lg:flex-row ${GAP} items-start`}>
      <div className="flex-1 min-w-0 space-y-4">{children}</div>
      <aside className={`hidden lg:block ${ASIDE_WIDTH} flex-shrink-0 space-y-4`}>
        {isExpanded && aside}
      </aside>
    </div>
  )
}

interface LessonHeaderProps {
  title: string
  description: string
  duration?: string
  category?: string
}

/**
 * Header for a lesson page with title, description, and metadata.
 */
export function LessonHeader({ title, description, duration, category }: LessonHeaderProps) {
  return (
    <div className="space-y-2">
      <div className="flex items-center gap-3">
        {category && (
          <span className="text-xs font-medium px-2 py-0.5 rounded bg-primary/10 text-primary">
            {category}
          </span>
        )}
        {duration && (
          <span className="text-xs text-muted-foreground flex items-center gap-1">
            <Clock className="w-3 h-3" />
            {duration}
          </span>
        )}
      </div>
      <h1 className="text-2xl font-bold">{title}</h1>
      <p className="text-muted-foreground">{description}</p>
    </div>
  )
}

interface SectionHeaderProps {
  title: string
  subtitle?: string
}

/**
 * Section header within a lesson.
 */
export function SectionHeader({ title, subtitle }: SectionHeaderProps) {
  return (
    <div className="space-y-1">
      <h2 className="text-xl font-semibold">{title}</h2>
      {subtitle && <p className="text-sm text-muted-foreground">{subtitle}</p>}
    </div>
  )
}

'use client'

import { ReactNode, Children, isValidElement } from 'react'

/**
 * Row - Compound Component for consistent 2-column layout
 *
 * Usage:
 * <Row>
 *   <Row.Content>Main content here</Row.Content>
 *   <Row.Aside>Sidebar content (optional)</Row.Aside>
 * </Row>
 *
 * The aside column is ALWAYS rendered to maintain consistent layout.
 * Pass empty <Row.Aside /> if you don't need sidebar content.
 */

// Layout constants - change these in ONE place
export const ASIDE_WIDTH = 'lg:w-64' // 256px
export const GAP = 'gap-8' // 32px

interface RowProps {
  children: ReactNode
}

interface ContentProps {
  children: ReactNode
}

interface AsideProps {
  children?: ReactNode
}

/**
 * Content area - takes remaining space
 */
const Content = ({ children }: ContentProps) => {
  return <div className="flex-1 min-w-0 space-y-6">{children}</div>
}

/**
 * Aside renders a fixed-width column that's always present on large screens.
 * This maintains consistent layout even when empty.
 */
const Aside = ({ children }: AsideProps) => {
  return (
    <aside className={`hidden lg:block ${ASIDE_WIDTH} flex-shrink-0 space-y-4`}>
      {children}
    </aside>
  )
}

/**
 * Default aside that Row renders when no Row.Aside is provided.
 * Ensures the 2-column layout is always maintained.
 */
const DefaultAside = () => {
  return <aside className={`hidden lg:block ${ASIDE_WIDTH} flex-shrink-0`} />
}

/**
 * Row container with flex layout
 */
const RowBase = ({ children }: RowProps) => {
  // Check if Row.Aside is among children
  const hasAside = Children.toArray(children).some(
    (child) => isValidElement(child) && child.type === Aside
  )

  return (
    <div className={`flex flex-col lg:flex-row ${GAP} items-start`}>
      {children}
      {!hasAside && <DefaultAside />}
    </div>
  )
}

// Export sub-components for direct import (RSC compatible)
export const RowContent = Content
export const RowAside = Aside

// Create compound component with proper typing
// Note: Row.Content and Row.Aside work in client components but may not serialize
// properly across RSC boundaries. Use RowContent/RowAside as alternatives.
export const Row = Object.assign(RowBase, {
  Content,
  Aside,
})

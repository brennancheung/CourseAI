'use client'

import {
  useState,
  useCallback,
  useEffect,
  useContext,
  createContext,
  cloneElement,
  isValidElement,
  ReactNode,
  ReactElement,
} from 'react'
import { createPortal } from 'react-dom'
import { Maximize2, Minimize2, X } from 'lucide-react'
import { Button } from '@/components/ui/button'

/**
 * ExercisePanel - Compound component for interactive exercises
 *
 * Provides a bordered panel with header (title + expand button) and content area.
 * Supports fullscreen modal expansion.
 *
 * Usage:
 * <ExercisePanel>
 *   <ExercisePanel.Header title="Find the Sweet Spot" />
 *   <ExercisePanel.Content>
 *     <LearningRateExplorer mode="interactive" />
 *   </ExercisePanel.Content>
 * </ExercisePanel>
 *
 * Or with shorthand:
 * <ExercisePanel title="Find the Sweet Spot">
 *   <LearningRateExplorer mode="interactive" />
 * </ExercisePanel>
 */

// Context for sharing state between compound components
type ExercisePanelContextType = {
  isExpanded: boolean
  setIsExpanded: (expanded: boolean) => void
  expandedWidth: number
  title?: string
}

const ExercisePanelContext = createContext<ExercisePanelContextType | null>(null)

function useExercisePanelContext() {
  const context = useContext(ExercisePanelContext)
  if (!context) {
    throw new Error('ExercisePanel compound components must be used within ExercisePanel')
  }
  return context
}

// --- Sub-components ---

type HeaderProps = {
  title: string
  subtitle?: string
}

function Header({ title, subtitle }: HeaderProps) {
  const { isExpanded, setIsExpanded } = useExercisePanelContext()

  return (
    <div className="flex items-center justify-between px-4 py-3 border-b border-border bg-muted/30">
      <div>
        <h3 className="font-semibold text-sm">{title}</h3>
        {subtitle && (
          <p className="text-xs text-muted-foreground">{subtitle}</p>
        )}
      </div>
      <Button
        variant="ghost"
        size="icon"
        className="h-8 w-8"
        onClick={() => setIsExpanded(!isExpanded)}
        title={isExpanded ? 'Exit fullscreen' : 'Expand to fullscreen'}
      >
        {isExpanded ? (
          <Minimize2 className="w-4 h-4" />
        ) : (
          <Maximize2 className="w-4 h-4" />
        )}
      </Button>
    </div>
  )
}

type ContentProps = {
  children: ReactElement<{ width?: number }>
}

function Content({ children }: ContentProps) {
  const { isExpanded, expandedWidth } = useExercisePanelContext()

  // Clone child with expanded width when in fullscreen
  const renderChild = () => {
    if (!isValidElement(children)) return children

    if (isExpanded) {
      return cloneElement(children, {
        width: expandedWidth,
      })
    }

    return children
  }

  return (
    <div className="p-4">
      {renderChild()}
    </div>
  )
}

// --- Main component ---

type ExercisePanelProps = {
  children: ReactNode
  /** Shorthand: if provided, auto-generates Header */
  title?: string
  /** Shorthand: subtitle for auto-generated Header */
  subtitle?: string
}

function ExercisePanelBase({ children, title, subtitle }: ExercisePanelProps) {
  const [isExpanded, setIsExpanded] = useState(false)
  const [portalContainer, setPortalContainer] = useState<HTMLElement | null>(null)
  const [expandedWidth, setExpandedWidth] = useState(900)

  // Set up portal container
  useEffect(() => {
    setPortalContainer(document.body)
  }, [])

  // Calculate expanded width
  useEffect(() => {
    if (!isExpanded) return

    const updateWidth = () => {
      const width = Math.min(window.innerWidth - 96, 1000)
      setExpandedWidth(width)
    }

    updateWidth()
    window.addEventListener('resize', updateWidth)
    return () => window.removeEventListener('resize', updateWidth)
  }, [isExpanded])

  // Handle ESC key
  useEffect(() => {
    if (!isExpanded) return

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setIsExpanded(false)
      }
    }

    document.addEventListener('keydown', handleKeyDown)
    return () => document.removeEventListener('keydown', handleKeyDown)
  }, [isExpanded])

  // Prevent body scroll when expanded
  useEffect(() => {
    if (isExpanded) {
      document.body.style.overflow = 'hidden'
    } else {
      document.body.style.overflow = ''
    }
    return () => {
      document.body.style.overflow = ''
    }
  }, [isExpanded])

  const handleBackdropClick = useCallback((e: React.MouseEvent) => {
    if (e.target === e.currentTarget) {
      setIsExpanded(false)
    }
  }, [])

  const contextValue: ExercisePanelContextType = {
    isExpanded,
    setIsExpanded,
    expandedWidth,
    title,
  }

  // Check if children include Header, or if we should auto-generate one
  const hasHeader = Array.isArray(children)
    ? children.some((child) => isValidElement(child) && child.type === Header)
    : isValidElement(children) && children.type === Header

  // Extract Content child for rendering
  const getContent = () => {
    if (Array.isArray(children)) {
      return children.find((child) => isValidElement(child) && child.type === Content)
    }
    if (isValidElement(children) && children.type === Content) {
      return children
    }
    // Shorthand: wrap bare children in Content
    if (!hasHeader && title) {
      return <Content>{children as ReactElement<{ width?: number }>}</Content>
    }
    return children
  }

  const panelContent = (
    <>
      {!hasHeader && title && <Header title={title} subtitle={subtitle} />}
      {hasHeader ? children : getContent()}
    </>
  )

  // Inline panel
  const inlinePanel = (
    <div className="border border-border rounded-lg overflow-hidden bg-card">
      {panelContent}
    </div>
  )

  // Fullscreen modal
  const fullscreenModal = isExpanded && portalContainer && createPortal(
    <div
      className="fixed inset-0 z-50 bg-black/90 flex items-start justify-center overflow-y-auto py-8"
      onClick={handleBackdropClick}
    >
      <div className="bg-card rounded-lg overflow-hidden max-w-[1100px] w-full mx-6 relative">
        {/* Modal close button */}
        <Button
          variant="ghost"
          size="icon"
          className="absolute top-2 right-2 z-10 hover:bg-muted"
          onClick={() => setIsExpanded(false)}
        >
          <X className="w-5 h-5" />
        </Button>

        {panelContent}

        {/* Help text */}
        <p className="text-muted-foreground text-sm py-3 text-center border-t border-border">
          Press ESC or click outside to close
        </p>
      </div>
    </div>,
    portalContainer
  )

  return (
    <ExercisePanelContext.Provider value={contextValue}>
      {inlinePanel}
      {fullscreenModal}
    </ExercisePanelContext.Provider>
  )
}

// Export compound component
export const ExercisePanel = Object.assign(ExercisePanelBase, {
  Header,
  Content,
})

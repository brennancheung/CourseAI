'use client'

import { useState, useCallback, useEffect, cloneElement, isValidElement, ReactElement } from 'react'
import { createPortal } from 'react-dom'
import { Maximize2, X } from 'lucide-react'
import { Button } from '@/components/ui/button'

/**
 * ExpandableWidget - Wrapper that allows any widget to be expanded to fullscreen
 *
 * Wraps the ENTIRE widget (canvas + controls + stats) and shows it all in a modal.
 *
 * Features:
 * - Shows widget inline at normal size
 * - Expand button opens fullscreen modal with entire widget
 * - ESC key closes modal
 * - Click backdrop to close
 * - Passes expanded width to child for canvas sizing
 */

type ExpandableWidgetProps = {
  children: ReactElement<{ width?: number; height?: number }>
  /** Title shown in expanded view */
  title?: string
}

export function ExpandableWidget({ children, title }: ExpandableWidgetProps) {
  const [isExpanded, setIsExpanded] = useState(false)
  const [portalContainer, setPortalContainer] = useState<HTMLElement | null>(null)
  const [expandedWidth, setExpandedWidth] = useState(900)

  // Set up portal container
  useEffect(() => {
    setPortalContainer(document.body)
  }, [])

  // Calculate expanded canvas width (leave room for padding)
  useEffect(() => {
    if (!isExpanded) return

    const updateWidth = () => {
      // Max canvas width, leaving room for modal padding
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

  // Clone child with expanded width when in modal
  const renderExpandedChild = () => {
    if (!isValidElement(children)) return children

    // Pass larger width for canvas, let height be natural
    return cloneElement(children, {
      width: expandedWidth,
      // Don't pass height - let widget determine its own height
    })
  }

  return (
    <>
      {/* Inline view with expand button */}
      <div className="relative group">
        {children}
        <Button
          variant="ghost"
          size="icon"
          className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity bg-background/80 hover:bg-background z-10"
          onClick={() => setIsExpanded(true)}
          title="Expand to fullscreen"
        >
          <Maximize2 className="w-4 h-4" />
        </Button>
      </div>

      {/* Fullscreen modal */}
      {isExpanded && portalContainer && createPortal(
        <div
          className="fixed inset-0 z-50 bg-black/90 flex items-start justify-center overflow-y-auto py-8"
          onClick={handleBackdropClick}
        >
          <div className="bg-card rounded-lg p-6 max-w-[1100px] w-full mx-6 relative">
            {/* Header */}
            <div className="flex items-center justify-between mb-4">
              {title && (
                <h2 className="text-xl font-semibold">{title}</h2>
              )}
              <div className="flex-1" />
              <Button
                variant="ghost"
                size="icon"
                className="hover:bg-muted"
                onClick={() => setIsExpanded(false)}
              >
                <X className="w-5 h-5" />
              </Button>
            </div>

            {/* Widget content - full widget with controls */}
            <div>
              {renderExpandedChild()}
            </div>

            {/* Help text */}
            <p className="text-muted-foreground text-sm mt-4 text-center">
              Press ESC or click outside to close
            </p>
          </div>
        </div>,
        portalContainer
      )}
    </>
  )
}

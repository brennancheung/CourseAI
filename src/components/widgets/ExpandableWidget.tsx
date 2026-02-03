'use client'

import { useState, useCallback, useEffect, cloneElement, isValidElement, ReactElement } from 'react'
import { createPortal } from 'react-dom'
import { Maximize2, X } from 'lucide-react'
import { Button } from '@/components/ui/button'

/**
 * ExpandableWidget - Wrapper that allows any widget to be expanded to fullscreen
 *
 * Features:
 * - Shows widget inline at normal size
 * - Expand button opens fullscreen modal
 * - ESC key closes modal
 * - Click backdrop to close
 * - Passes expanded dimensions to child widget
 */

type ExpandableWidgetProps = {
  children: ReactElement<{ width?: number; height?: number }>
  /** Title shown in expanded view */
  title?: string
}

export function ExpandableWidget({ children, title }: ExpandableWidgetProps) {
  const [isExpanded, setIsExpanded] = useState(false)
  const [portalContainer, setPortalContainer] = useState<HTMLElement | null>(null)
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 })

  // Set up portal container
  useEffect(() => {
    setPortalContainer(document.body)
  }, [])

  // Calculate expanded dimensions
  useEffect(() => {
    if (!isExpanded) return

    const updateDimensions = () => {
      // Leave padding for header and margins
      const width = Math.min(window.innerWidth - 48, 1200)
      const height = window.innerHeight - 120
      setDimensions({ width, height })
    }

    updateDimensions()
    window.addEventListener('resize', updateDimensions)
    return () => window.removeEventListener('resize', updateDimensions)
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

  // Clone child with expanded dimensions when in modal
  const renderChild = (expanded: boolean) => {
    if (!isValidElement(children)) return children

    if (expanded) {
      return cloneElement(children, {
        width: dimensions.width,
        height: dimensions.height,
      })
    }

    return children
  }

  return (
    <>
      {/* Inline view with expand button */}
      <div className="relative group">
        {renderChild(false)}
        <Button
          variant="ghost"
          size="icon"
          className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity bg-background/80 hover:bg-background"
          onClick={() => setIsExpanded(true)}
          title="Expand to fullscreen"
        >
          <Maximize2 className="w-4 h-4" />
        </Button>
      </div>

      {/* Fullscreen modal */}
      {isExpanded && portalContainer && createPortal(
        <div
          className="fixed inset-0 z-50 bg-black/90 flex flex-col items-center justify-center p-6"
          onClick={handleBackdropClick}
        >
          {/* Header */}
          <div className="w-full max-w-[1200px] flex items-center justify-between mb-4">
            {title && (
              <h2 className="text-lg font-semibold text-white">{title}</h2>
            )}
            <div className="flex-1" />
            <Button
              variant="ghost"
              size="icon"
              className="text-white hover:bg-white/10"
              onClick={() => setIsExpanded(false)}
            >
              <X className="w-5 h-5" />
            </Button>
          </div>

          {/* Widget container */}
          <div
            className="bg-card rounded-lg overflow-hidden"
            style={{ width: dimensions.width, height: dimensions.height }}
          >
            {renderChild(true)}
          </div>

          {/* Help text */}
          <p className="text-white/50 text-sm mt-4">
            Press ESC or click outside to close • Scroll to zoom • Drag to pan • Double-click to reset
          </p>
        </div>,
        portalContainer
      )}
    </>
  )
}

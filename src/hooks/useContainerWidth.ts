'use client'

import { useRef, useState, useEffect, RefObject } from 'react'

/**
 * useContainerWidth - Measures the width of a container element using ResizeObserver
 *
 * Attach `containerRef` to the outer div of your widget. The hook returns the
 * measured width (or `fallback` before the first measurement).
 *
 * Widgets that accept a `width` prop from ExercisePanel fullscreen should use:
 *   const width = widthOverride ?? measuredWidth
 */
export function useContainerWidth(fallback = 500): {
  containerRef: RefObject<HTMLDivElement | null>
  width: number
} {
  const containerRef = useRef<HTMLDivElement>(null)
  const [width, setWidth] = useState(fallback)

  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    const updateWidth = () => {
      setWidth(container.getBoundingClientRect().width)
    }

    updateWidth()
    const observer = new ResizeObserver(updateWidth)
    observer.observe(container)
    return () => observer.disconnect()
  }, [])

  return { containerRef, width }
}

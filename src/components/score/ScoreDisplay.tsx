'use client'

import { useRef, useEffect } from 'react'
import abcjs from 'abcjs'

type ScoreDisplayProps = {
  abc: string
  className?: string
  onRender?: (tuneObject: abcjs.TuneObject) => void
}

/**
 * Renders ABC notation as an SVG score.
 * This is a simple display-only component without playback.
 */
export function ScoreDisplay({ abc, className = '', onRender }: ScoreDisplayProps) {
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!containerRef.current) return

    const tuneObjects = abcjs.renderAbc(containerRef.current, abc, {
      responsive: 'resize',
      add_classes: true,
      paddingtop: 10,
      paddingbottom: 10,
      paddingleft: 10,
      paddingright: 10,
    })

    if (onRender && tuneObjects.length > 0) {
      onRender(tuneObjects[0])
    }
  }, [abc, onRender])

  return (
    <div
      ref={containerRef}
      className={`score-display ${className}`}
      style={{ colorScheme: 'light', color: 'black' }}
    />
  )
}

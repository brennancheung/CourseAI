'use client'

import { ReactNode, useMemo, createContext, useContext } from 'react'
import { Stage, Layer } from 'react-konva'

/**
 * MathCanvas - Foundation for mathematical visualizations
 *
 * Provides coordinate transformation from math space to pixel space.
 * All child components can use the context to convert coordinates.
 */

export type ViewBox = {
  xMin: number
  xMax: number
  yMin: number
  yMax: number
}

export type CoordinateSystem = {
  /** Convert math x to pixel x */
  toPixelX: (x: number) => number
  /** Convert math y to pixel y */
  toPixelY: (y: number) => number
  /** Convert pixel x to math x */
  toMathX: (px: number) => number
  /** Convert pixel y to math y */
  toMathY: (py: number) => number
  /** Scale a math distance to pixels (for sizes) */
  scaleX: (dx: number) => number
  scaleY: (dy: number) => number
  /** The viewbox in math coordinates */
  viewBox: ViewBox
  /** Canvas dimensions in pixels */
  width: number
  height: number
}

const CoordinateContext = createContext<CoordinateSystem | null>(null)

export function useCoordinates(): CoordinateSystem {
  const ctx = useContext(CoordinateContext)
  if (!ctx) {
    throw new Error('useCoordinates must be used within a MathCanvas')
  }
  return ctx
}

type MathCanvasProps = {
  /** Width in pixels */
  width: number
  /** Height in pixels */
  height: number
  /** View box in math coordinates */
  viewBox: ViewBox
  /** Background color */
  backgroundColor?: string
  /** Children to render */
  children: ReactNode
}

export function MathCanvas({
  width,
  height,
  viewBox,
  backgroundColor = '#1e1e2e',
  children,
}: MathCanvasProps) {
  const coords = useMemo((): CoordinateSystem => {
    const { xMin, xMax, yMin, yMax } = viewBox
    const mathWidth = xMax - xMin
    const mathHeight = yMax - yMin

    return {
      toPixelX: (x: number) => ((x - xMin) / mathWidth) * width,
      toPixelY: (y: number) => height - ((y - yMin) / mathHeight) * height, // Flip Y
      toMathX: (px: number) => xMin + (px / width) * mathWidth,
      toMathY: (py: number) => yMax - (py / height) * mathHeight, // Flip Y
      scaleX: (dx: number) => (dx / mathWidth) * width,
      scaleY: (dy: number) => (dy / mathHeight) * height,
      viewBox,
      width,
      height,
    }
  }, [width, height, viewBox])

  return (
    <CoordinateContext.Provider value={coords}>
      <Stage width={width} height={height} style={{ backgroundColor }}>
        <Layer>{children}</Layer>
      </Stage>
    </CoordinateContext.Provider>
  )
}

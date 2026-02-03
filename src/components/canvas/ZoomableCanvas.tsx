'use client'

import { useRef, useEffect, ReactNode } from 'react'
import { Stage, Layer } from 'react-konva'
import Konva from 'konva'

/**
 * ZoomableCanvas - A react-konva Stage with pan and zoom support
 *
 * Features:
 * - Mouse wheel zoom (centered on cursor)
 * - Trackpad pinch zoom
 * - Touch pinch zoom
 * - Drag to pan
 * - Double-click to reset view
 */

type ZoomableCanvasProps = {
  width: number
  height: number
  children: ReactNode
  /** Background color */
  backgroundColor?: string
  /** Minimum zoom level */
  minScale?: number
  /** Maximum zoom level */
  maxScale?: number
  /** Initial scale */
  initialScale?: number
  /** Initial X offset */
  initialX?: number
  /** Initial Y offset */
  initialY?: number
}

// Helper to get distance between two touch points
const getDistance = (p1: Touch, p2: Touch): number =>
  Math.sqrt(Math.pow(p2.clientX - p1.clientX, 2) + Math.pow(p2.clientY - p1.clientY, 2))

// Helper to get center point between two touches
const getCenter = (p1: Touch, p2: Touch): { x: number; y: number } => ({
  x: (p1.clientX + p2.clientX) / 2,
  y: (p1.clientY + p2.clientY) / 2,
})

// Zoom sensitivity for trackpad gestures
const ZOOM_SENSITIVITY = 0.01

export function ZoomableCanvas({
  width,
  height,
  children,
  backgroundColor = '#1a1a2e',
  minScale = 0.25,
  maxScale = 4,
  initialScale = 1,
  initialX = 0,
  initialY = 0,
}: ZoomableCanvasProps) {
  const stageRef = useRef<Konva.Stage>(null)

  // Track pinch gesture state (for touchscreen devices)
  const lastDistanceRef = useRef<number | null>(null)
  const lastCenterRef = useRef<{ x: number; y: number } | null>(null)

  // Store initial values for reset
  const initialState = useRef({ scale: initialScale, x: initialX, y: initialY })

  // Disable browser's native touch handling on the canvas
  useEffect(() => {
    const timer = setTimeout(() => {
      const stage = stageRef.current
      if (!stage) return
      const container = stage.container()
      container.style.touchAction = 'none'
    }, 0)
    return () => clearTimeout(timer)
  }, [])

  // Set initial scale/position
  useEffect(() => {
    const stage = stageRef.current
    if (!stage) return
    stage.scale({ x: initialScale, y: initialScale })
    stage.position({ x: initialX, y: initialY })
  }, [initialScale, initialX, initialY])

  const handleWheel = (e: Konva.KonvaEventObject<WheelEvent>) => {
    e.evt.preventDefault()

    const stage = stageRef.current
    if (!stage) return

    const pointer = stage.getPointerPosition()
    if (!pointer) return

    const oldScale = stage.scaleX()
    const mousePointTo = {
      x: (pointer.x - stage.x()) / oldScale,
      y: (pointer.y - stage.y()) / oldScale,
    }

    // Apply sensitivity dampening
    const delta = -e.evt.deltaY * ZOOM_SENSITIVITY
    let newScale = oldScale * (1 + delta)

    // Clamp scale
    newScale = Math.max(minScale, Math.min(maxScale, newScale))

    // Update stage directly (no React state = no re-render lag)
    stage.scale({ x: newScale, y: newScale })
    stage.position({
      x: pointer.x - mousePointTo.x * newScale,
      y: pointer.y - mousePointTo.y * newScale,
    })
  }

  const handleTouchMove = (e: Konva.KonvaEventObject<TouchEvent>) => {
    const touches = e.evt.touches
    if (touches.length !== 2) return

    e.evt.preventDefault()

    const stage = stageRef.current
    if (!stage) return

    const touch1 = touches[0]
    const touch2 = touches[1]
    const newDistance = getDistance(touch1, touch2)
    const newCenter = getCenter(touch1, touch2)

    // Get stage position relative to the viewport
    const stageContainer = stage.container()
    const stageRect = stageContainer.getBoundingClientRect()

    // Convert center to stage coordinates
    const centerOnStage = {
      x: newCenter.x - stageRect.left,
      y: newCenter.y - stageRect.top,
    }

    // If this is the first touch event with 2 fingers, just store the initial values
    if (lastDistanceRef.current === null || lastCenterRef.current === null) {
      lastDistanceRef.current = newDistance
      lastCenterRef.current = centerOnStage
      return
    }

    // Calculate scale change with sensitivity dampening
    const rawScaleChange = newDistance / lastDistanceRef.current
    const touchPinchSensitivity = 0.5
    const scaleChange = 1 + (rawScaleChange - 1) * touchPinchSensitivity
    const oldScale = stage.scaleX()
    const newScale = Math.max(minScale, Math.min(maxScale, oldScale * scaleChange))

    // Calculate the point to zoom around
    const pointTo = {
      x: (centerOnStage.x - stage.x()) / oldScale,
      y: (centerOnStage.y - stage.y()) / oldScale,
    }

    // Calculate pan delta based on center movement
    const lastCenterOnStage = lastCenterRef.current
    const dx = centerOnStage.x - lastCenterOnStage.x
    const dy = centerOnStage.y - lastCenterOnStage.y

    // Update stage directly
    stage.scale({ x: newScale, y: newScale })
    stage.position({
      x: centerOnStage.x - pointTo.x * newScale + dx,
      y: centerOnStage.y - pointTo.y * newScale + dy,
    })

    lastDistanceRef.current = newDistance
    lastCenterRef.current = centerOnStage
  }

  const handleTouchEnd = () => {
    lastDistanceRef.current = null
    lastCenterRef.current = null
  }

  const handleDoubleClick = () => {
    const stage = stageRef.current
    if (!stage) return

    // Reset to initial view
    stage.scale({ x: initialState.current.scale, y: initialState.current.scale })
    stage.position({ x: initialState.current.x, y: initialState.current.y })
  }

  return (
    <Stage
      ref={stageRef}
      width={width}
      height={height}
      draggable
      onWheel={handleWheel}
      onTouchMove={handleTouchMove}
      onTouchEnd={handleTouchEnd}
      onDblClick={handleDoubleClick}
      onDblTap={handleDoubleClick}
      style={{ backgroundColor }}
    >
      <Layer>{children}</Layer>
    </Stage>
  )
}

'use client'

import { useState, useMemo, useCallback } from 'react'
import { useContainerWidth } from '@/hooks/useContainerWidth'
import { ZoomableCanvas } from '@/components/canvas/ZoomableCanvas'
import { DiagramNode } from './DiagramNode'
import { DiagramEdge } from './DiagramEdge'
import { DiagramGroup } from './DiagramGroup'
import { InfoPanel } from './InfoPanel'
import type { ArchitectureDiagramData, DiagramNodeDef } from './types'

type ArchitectureDiagramProps = {
  data: ArchitectureDiagramData
  width?: number
  height?: number
}

const DEFAULT_NODE_WIDTH = 140
const DEFAULT_NODE_HEIGHT = 48

function getNodeBounds(node: DiagramNodeDef) {
  return {
    x: node.x,
    y: node.y,
    width: node.width ?? DEFAULT_NODE_WIDTH,
    height: node.height ?? DEFAULT_NODE_HEIGHT,
  }
}

export function ArchitectureDiagram({ data, width: widthOverride, height: heightOverride }: ArchitectureDiagramProps) {
  const { containerRef, width: measuredWidth } = useContainerWidth(700)
  const containerWidth = widthOverride ?? measuredWidth

  const [hoveredNodeId, setHoveredNodeId] = useState<string | null>(null)
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null)

  const activeNodeId = selectedNodeId ?? hoveredNodeId

  // Build a lookup map for nodes
  const nodeMap = useMemo(() => {
    const map = new Map<string, DiagramNodeDef>()
    for (const node of data.nodes) {
      map.set(node.id, node)
    }
    return map
  }, [data.nodes])

  // Build bounds map for edge computation
  const boundsMap = useMemo(() => {
    const map = new Map<string, { x: number; y: number; width: number; height: number }>()
    for (const node of data.nodes) {
      map.set(node.id, getNodeBounds(node))
    }
    return map
  }, [data.nodes])

  // Find highlighted edges and nodes when active node is set
  const { highlightedEdgeIds, highlightedNodeIds } = useMemo(() => {
    if (!activeNodeId) {
      return { highlightedEdgeIds: new Set<string>(), highlightedNodeIds: new Set<string>() }
    }

    const edgeIds = new Set<string>()
    const nodeIds = new Set<string>([activeNodeId])

    for (const edge of data.edges) {
      if (edge.from === activeNodeId || edge.to === activeNodeId) {
        edgeIds.add(edge.id)
        nodeIds.add(edge.from)
        nodeIds.add(edge.to)
      }
    }

    return { highlightedEdgeIds: edgeIds, highlightedNodeIds: nodeIds }
  }, [activeNodeId, data.edges])

  // Compute scale to fit canvas content into available space
  const { scale, canvasHeight, offsetX, offsetY } = useMemo(() => {
    const scaleX = containerWidth / data.canvasWidth
    // Inline: use aspect ratio to fill width, no arbitrary cap
    const aspectHeight = data.canvasHeight * scaleX
    const baseHeight = heightOverride ?? aspectHeight
    const scaleY = baseHeight / data.canvasHeight
    const s = Math.min(scaleX, scaleY, 1.5)
    const ox = (containerWidth - data.canvasWidth * s) / 2
    const oy = Math.max((baseHeight - data.canvasHeight * s) / 2, 0)
    return { scale: s, canvasHeight: baseHeight, offsetX: ox, offsetY: oy }
  }, [containerWidth, heightOverride, data.canvasWidth, data.canvasHeight])

  const handleNodeEnter = useCallback((id: string) => setHoveredNodeId(id), [])
  const handleNodeLeave = useCallback(() => setHoveredNodeId(null), [])
  const handleNodeClick = useCallback((id: string) => {
    setSelectedNodeId(prev => (prev === id ? null : id))
  }, [])
  const handleCloseInfo = useCallback(() => setSelectedNodeId(null), [])

  const selectedNode = selectedNodeId ? nodeMap.get(selectedNodeId) : undefined

  return (
    <div ref={containerRef} className="space-y-0">
      <ZoomableCanvas
        width={containerWidth}
        height={canvasHeight}
        backgroundColor="#0f1219"
        initialScale={scale}
        initialX={offsetX}
        initialY={offsetY}
        minScale={0.2}
        maxScale={3}
      >
        {/* Groups (background) */}
        {data.groups.map(group => (
          <DiagramGroup key={group.id} group={group} />
        ))}

        {/* Edges */}
        {data.edges.map(edge => {
          const fromNode = nodeMap.get(edge.from)
          const toNode = nodeMap.get(edge.to)
          const fromBounds = boundsMap.get(edge.from)
          const toBounds = boundsMap.get(edge.to)
          if (!fromNode || !toNode || !fromBounds || !toBounds) return null

          return (
            <DiagramEdge
              key={edge.id}
              edge={edge}
              fromNode={fromNode}
              toNode={toNode}
              fromBounds={fromBounds}
              toBounds={toBounds}
              isHighlighted={highlightedEdgeIds.has(edge.id)}
              isDimmed={!!activeNodeId && !highlightedEdgeIds.has(edge.id)}
            />
          )
        })}

        {/* Nodes (on top, receive events) */}
        {data.nodes.map(node => (
          <DiagramNode
            key={node.id}
            node={node}
            isHighlighted={highlightedNodeIds.has(node.id)}
            isDimmed={!!activeNodeId && !highlightedNodeIds.has(node.id)}
            isSelected={selectedNodeId === node.id}
            onMouseEnter={() => handleNodeEnter(node.id)}
            onMouseLeave={handleNodeLeave}
            onClick={() => handleNodeClick(node.id)}
          />
        ))}
      </ZoomableCanvas>

      {selectedNode && (
        <InfoPanel node={selectedNode} onClose={handleCloseInfo} />
      )}

      <p className="text-xs text-muted-foreground mt-2">
        Hover to highlight connections. Click a node for details. Scroll to zoom, drag to pan, double-click to reset.
      </p>
    </div>
  )
}

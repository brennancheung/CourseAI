export type DiagramNodeDef = {
  id: string
  label: string
  description?: string
  groupId?: string
  x: number
  y: number
  width?: number
  height?: number
  fillColor?: string
  strokeColor?: string
}

export type DiagramEdgeDef = {
  id: string
  from: string
  to: string
  label?: string
  style: 'solid' | 'dashed'
  color?: string
  waypoints?: Array<{ x: number; y: number }>
  tension?: number
}

export type DiagramGroupDef = {
  id: string
  title: string
  x: number
  y: number
  width: number
  height: number
  fillColor?: string
  strokeColor?: string
  strokeStyle?: 'solid' | 'dashed'
}

export type ArchitectureDiagramData = {
  nodes: DiagramNodeDef[]
  edges: DiagramEdgeDef[]
  groups: DiagramGroupDef[]
  canvasWidth: number
  canvasHeight: number
}

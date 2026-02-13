import type { DiagramNodeDef } from './types'

type InfoPanelProps = {
  node: DiagramNodeDef
  onClose: () => void
}

export function InfoPanel({ node, onClose }: InfoPanelProps) {
  if (!node.description) return null

  return (
    <div className="mt-3 rounded-lg border border-amber-500/30 bg-amber-500/5 p-3 text-sm animate-in fade-in slide-in-from-bottom-1 duration-200">
      <div className="flex items-start justify-between gap-2">
        <div className="space-y-1">
          <div className="font-semibold text-amber-400">{node.label}</div>
          <p className="text-muted-foreground leading-relaxed">{node.description}</p>
        </div>
        <button
          onClick={onClose}
          className="shrink-0 text-muted-foreground hover:text-foreground transition-colors cursor-pointer text-xs px-1.5 py-0.5 rounded hover:bg-muted"
        >
          &times;
        </button>
      </div>
    </div>
  )
}

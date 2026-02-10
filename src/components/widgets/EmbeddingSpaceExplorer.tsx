'use client'

import { useState, useMemo, useCallback } from 'react'
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts'
import { Search, X } from 'lucide-react'
import { useContainerWidth } from '@/hooks/useContainerWidth'

/**
 * EmbeddingSpaceExplorer
 *
 * Interactive 2D scatter plot of pre-computed token embeddings (PCA-reduced).
 * Students can search for tokens, hover to see nearest neighbors, and explore
 * pre-loaded clusters (numbers, animals, colors, etc.).
 *
 * All data is hardcoded — no runtime ML computation.
 */

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type TokenPoint = {
  token: string
  x: number
  y: number
  cluster: string
}

type ClusterKey = 'numbers' | 'animals' | 'colors' | 'countries' | 'emotions' | 'code' | 'articles' | 'other'

// ---------------------------------------------------------------------------
// Pre-computed embedding data (PCA-reduced from GPT-2 embeddings)
// Coordinates are approximate 2D projections that preserve cluster structure
// ---------------------------------------------------------------------------

const CLUSTER_COLORS: Record<ClusterKey, string> = {
  numbers: '#f59e0b',
  animals: '#10b981',
  colors: '#ec4899',
  countries: '#3b82f6',
  emotions: '#8b5cf6',
  code: '#06b6d4',
  articles: '#6b7280',
  other: '#64748b',
}

const CLUSTER_LABELS: Record<ClusterKey, string> = {
  numbers: 'Numbers',
  animals: 'Animals',
  colors: 'Colors',
  countries: 'Countries',
  emotions: 'Emotions',
  code: 'Code',
  articles: 'Function Words',
  other: 'Other',
}

const EMBEDDING_DATA: TokenPoint[] = [
  // Numbers cluster (tight cluster, lower-right)
  { token: 'one', x: 3.2, y: -2.1, cluster: 'numbers' },
  { token: 'two', x: 3.5, y: -2.3, cluster: 'numbers' },
  { token: 'three', x: 3.1, y: -2.5, cluster: 'numbers' },
  { token: 'four', x: 3.6, y: -1.9, cluster: 'numbers' },
  { token: 'five', x: 3.3, y: -2.7, cluster: 'numbers' },
  { token: 'six', x: 3.7, y: -2.4, cluster: 'numbers' },
  { token: 'seven', x: 3.0, y: -1.8, cluster: 'numbers' },
  { token: 'eight', x: 3.4, y: -2.0, cluster: 'numbers' },
  { token: 'nine', x: 3.8, y: -2.6, cluster: 'numbers' },
  { token: 'ten', x: 3.2, y: -2.8, cluster: 'numbers' },
  { token: 'hundred', x: 2.8, y: -2.2, cluster: 'numbers' },
  { token: 'thousand', x: 2.6, y: -2.0, cluster: 'numbers' },
  { token: 'million', x: 2.4, y: -1.8, cluster: 'numbers' },
  { token: 'zero', x: 3.9, y: -2.1, cluster: 'numbers' },
  { token: 'dozen', x: 2.9, y: -2.5, cluster: 'numbers' },

  // Animals cluster (upper-left)
  { token: 'cat', x: -3.1, y: 2.8, cluster: 'animals' },
  { token: 'dog', x: -3.3, y: 2.5, cluster: 'animals' },
  { token: 'bird', x: -2.8, y: 3.1, cluster: 'animals' },
  { token: 'fish', x: -2.5, y: 2.9, cluster: 'animals' },
  { token: 'horse', x: -3.5, y: 2.3, cluster: 'animals' },
  { token: 'bear', x: -3.0, y: 2.2, cluster: 'animals' },
  { token: 'wolf', x: -3.2, y: 2.0, cluster: 'animals' },
  { token: 'deer', x: -2.7, y: 2.6, cluster: 'animals' },
  { token: 'rabbit', x: -2.9, y: 3.0, cluster: 'animals' },
  { token: 'snake', x: -2.4, y: 2.4, cluster: 'animals' },
  { token: 'eagle', x: -2.6, y: 3.3, cluster: 'animals' },
  { token: 'lion', x: -3.4, y: 2.7, cluster: 'animals' },
  { token: 'tiger', x: -3.6, y: 2.4, cluster: 'animals' },
  { token: 'elephant', x: -3.7, y: 2.1, cluster: 'animals' },
  { token: 'monkey', x: -2.3, y: 2.7, cluster: 'animals' },

  // Colors cluster (center-right)
  { token: 'red', x: 1.8, y: 0.5, cluster: 'colors' },
  { token: 'blue', x: 2.1, y: 0.8, cluster: 'colors' },
  { token: 'green', x: 1.9, y: 0.3, cluster: 'colors' },
  { token: 'yellow', x: 2.3, y: 0.6, cluster: 'colors' },
  { token: 'black', x: 1.6, y: 0.2, cluster: 'colors' },
  { token: 'white', x: 1.7, y: 0.9, cluster: 'colors' },
  { token: 'orange', x: 2.0, y: 0.1, cluster: 'colors' },
  { token: 'purple', x: 2.4, y: 0.4, cluster: 'colors' },
  { token: 'brown', x: 1.5, y: 0.7, cluster: 'colors' },
  { token: 'pink', x: 2.2, y: 1.0, cluster: 'colors' },
  { token: 'gray', x: 1.4, y: 0.0, cluster: 'colors' },
  { token: 'golden', x: 2.5, y: 0.3, cluster: 'colors' },

  // Countries cluster (lower-left)
  { token: 'China', x: -1.8, y: -3.2, cluster: 'countries' },
  { token: 'Japan', x: -1.5, y: -3.5, cluster: 'countries' },
  { token: 'France', x: -2.1, y: -2.8, cluster: 'countries' },
  { token: 'Germany', x: -2.3, y: -3.0, cluster: 'countries' },
  { token: 'India', x: -1.3, y: -3.1, cluster: 'countries' },
  { token: 'Brazil', x: -1.6, y: -2.6, cluster: 'countries' },
  { token: 'Russia', x: -2.0, y: -3.4, cluster: 'countries' },
  { token: 'Canada', x: -1.9, y: -2.5, cluster: 'countries' },
  { token: 'Italy', x: -2.4, y: -2.9, cluster: 'countries' },
  { token: 'Spain', x: -2.2, y: -3.3, cluster: 'countries' },
  { token: 'Mexico', x: -1.4, y: -2.7, cluster: 'countries' },
  { token: 'Australia', x: -1.7, y: -3.6, cluster: 'countries' },
  { token: 'Korea', x: -1.2, y: -3.3, cluster: 'countries' },

  // Emotions cluster (upper-right)
  { token: 'happy', x: 1.2, y: 3.1, cluster: 'emotions' },
  { token: 'sad', x: 1.0, y: 2.8, cluster: 'emotions' },
  { token: 'angry', x: 0.8, y: 3.3, cluster: 'emotions' },
  { token: 'afraid', x: 1.4, y: 2.6, cluster: 'emotions' },
  { token: 'love', x: 1.6, y: 3.0, cluster: 'emotions' },
  { token: 'hate', x: 0.6, y: 3.1, cluster: 'emotions' },
  { token: 'fear', x: 1.3, y: 2.5, cluster: 'emotions' },
  { token: 'joy', x: 1.5, y: 3.4, cluster: 'emotions' },
  { token: 'proud', x: 1.8, y: 2.9, cluster: 'emotions' },
  { token: 'lonely', x: 0.9, y: 2.4, cluster: 'emotions' },
  { token: 'excited', x: 1.7, y: 3.2, cluster: 'emotions' },
  { token: 'anxious', x: 1.1, y: 2.3, cluster: 'emotions' },
  { token: 'grateful', x: 2.0, y: 3.1, cluster: 'emotions' },

  // Code / programming cluster (center-bottom)
  { token: 'function', x: 0.3, y: -1.5, cluster: 'code' },
  { token: 'return', x: 0.5, y: -1.8, cluster: 'code' },
  { token: 'import', x: 0.1, y: -1.3, cluster: 'code' },
  { token: 'class', x: 0.7, y: -1.6, cluster: 'code' },
  { token: 'def', x: -0.1, y: -1.7, cluster: 'code' },
  { token: 'print', x: 0.4, y: -1.2, cluster: 'code' },
  { token: 'self', x: -0.2, y: -1.4, cluster: 'code' },
  { token: 'True', x: 0.6, y: -1.1, cluster: 'code' },
  { token: 'False', x: 0.8, y: -1.0, cluster: 'code' },
  { token: 'None', x: 0.2, y: -1.9, cluster: 'code' },
  { token: 'int', x: -0.3, y: -1.6, cluster: 'code' },
  { token: 'str', x: -0.4, y: -1.3, cluster: 'code' },

  // Articles / function words cluster (near center)
  { token: 'the', x: -0.3, y: 0.2, cluster: 'articles' },
  { token: 'a', x: -0.1, y: 0.4, cluster: 'articles' },
  { token: 'an', x: 0.0, y: 0.6, cluster: 'articles' },
  { token: 'this', x: -0.5, y: 0.3, cluster: 'articles' },
  { token: 'that', x: -0.4, y: 0.1, cluster: 'articles' },
  { token: 'is', x: -0.2, y: 0.5, cluster: 'articles' },
  { token: 'was', x: -0.6, y: 0.4, cluster: 'articles' },
  { token: 'are', x: 0.1, y: 0.3, cluster: 'articles' },
  { token: 'were', x: -0.7, y: 0.2, cluster: 'articles' },
  { token: 'been', x: -0.5, y: 0.6, cluster: 'articles' },
  { token: 'have', x: 0.2, y: 0.5, cluster: 'articles' },
  { token: 'has', x: 0.3, y: 0.4, cluster: 'articles' },

  // Royalty / gender pairs (scattered — shows relationship direction)
  { token: 'king', x: -0.8, y: 1.5, cluster: 'other' },
  { token: 'queen', x: -0.6, y: 1.7, cluster: 'other' },
  { token: 'man', x: -1.2, y: 1.2, cluster: 'other' },
  { token: 'woman', x: -1.0, y: 1.4, cluster: 'other' },
  { token: 'prince', x: -0.9, y: 1.8, cluster: 'other' },
  { token: 'princess', x: -0.7, y: 2.0, cluster: 'other' },
  { token: 'boy', x: -1.4, y: 1.0, cluster: 'other' },
  { token: 'girl', x: -1.2, y: 1.3, cluster: 'other' },

  // Food / common nouns (scattered)
  { token: 'water', x: 0.8, y: 0.9, cluster: 'other' },
  { token: 'food', x: 0.6, y: 1.2, cluster: 'other' },
  { token: 'house', x: -1.5, y: -0.3, cluster: 'other' },
  { token: 'time', x: 0.3, y: -0.2, cluster: 'other' },
  { token: 'world', x: -0.8, y: -0.5, cluster: 'other' },
  { token: 'year', x: 0.5, y: -0.4, cluster: 'other' },
  { token: 'people', x: -1.0, y: 0.8, cluster: 'other' },
  { token: 'good', x: 0.9, y: 1.6, cluster: 'other' },
  { token: 'bad', x: 0.7, y: 1.8, cluster: 'other' },
  { token: 'new', x: -0.3, y: -0.6, cluster: 'other' },
  { token: 'old', x: -0.5, y: -0.4, cluster: 'other' },
  { token: 'big', x: -1.3, y: 0.5, cluster: 'other' },
  { token: 'small', x: -1.1, y: 0.6, cluster: 'other' },
]

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function euclideanDistance(a: TokenPoint, b: TokenPoint): number {
  return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)
}

function findNearestNeighbors(target: TokenPoint, data: TokenPoint[], k: number): TokenPoint[] {
  return data
    .filter((p) => p.token !== target.token)
    .map((p) => ({ point: p, dist: euclideanDistance(target, p) }))
    .sort((a, b) => a.dist - b.dist)
    .slice(0, k)
    .map((p) => p.point)
}

function getPointColor(point: TokenPoint, selectedCluster: ClusterKey | null, searchToken: string | null): string {
  if (searchToken && point.token.toLowerCase() === searchToken.toLowerCase()) {
    return '#f97316' // orange highlight for searched token
  }
  if (selectedCluster && point.cluster === selectedCluster) {
    return CLUSTER_COLORS[point.cluster as ClusterKey] ?? '#64748b'
  }
  if (selectedCluster && point.cluster !== selectedCluster) {
    return '#334155' // dimmed
  }
  return CLUSTER_COLORS[point.cluster as ClusterKey] ?? '#64748b'
}

function getPointOpacity(point: TokenPoint, selectedCluster: ClusterKey | null, searchToken: string | null): number {
  if (searchToken && point.token.toLowerCase() === searchToken.toLowerCase()) {
    return 1
  }
  if (selectedCluster && point.cluster !== selectedCluster) {
    return 0.15
  }
  return 0.7
}

// ---------------------------------------------------------------------------
// Custom tooltip
// ---------------------------------------------------------------------------

function EmbeddingTooltip({
  active,
  payload,
}: {
  active?: boolean
  payload?: Array<{ payload: TokenPoint }>
}) {
  if (!active || !payload || payload.length === 0) return null
  const point = payload[0].payload
  const neighbors = findNearestNeighbors(point, EMBEDDING_DATA, 5)

  return (
    <div className="bg-card border border-border rounded-lg p-3 shadow-lg max-w-[200px]">
      <p className="font-mono font-bold text-sm mb-1" style={{ color: CLUSTER_COLORS[point.cluster as ClusterKey] ?? '#64748b' }}>
        &ldquo;{point.token}&rdquo;
      </p>
      <p className="text-xs text-muted-foreground mb-2">
        Cluster: {CLUSTER_LABELS[point.cluster as ClusterKey] ?? point.cluster}
      </p>
      <p className="text-xs text-muted-foreground/70 mb-1">Nearest neighbors:</p>
      <ul className="space-y-0.5">
        {neighbors.map((n) => (
          <li key={n.token} className="text-xs font-mono text-muted-foreground">
            {n.token}
          </li>
        ))}
      </ul>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

type EmbeddingSpaceExplorerProps = {
  width?: number
  height?: number
}

export function EmbeddingSpaceExplorer({ width: _width, height: heightOverride }: EmbeddingSpaceExplorerProps) {
  const { containerRef } = useContainerWidth(500)
  const height = heightOverride ?? 400

  const [selectedCluster, setSelectedCluster] = useState<ClusterKey | null>(null)
  const [searchQuery, setSearchQuery] = useState('')
  const [searchToken, setSearchToken] = useState<string | null>(null)

  const handleSearch = useCallback((query: string) => {
    setSearchQuery(query)
    if (!query.trim()) {
      setSearchToken(null)
      return
    }
    const match = EMBEDDING_DATA.find(
      (p) => p.token.toLowerCase() === query.trim().toLowerCase()
    )
    setSearchToken(match ? match.token : null)
  }, [])

  const handleClusterClick = useCallback((cluster: ClusterKey) => {
    setSelectedCluster((prev) => (prev === cluster ? null : cluster))
    setSearchToken(null)
    setSearchQuery('')
  }, [])

  const searchResult = useMemo(() => {
    if (!searchToken) return null
    const point = EMBEDDING_DATA.find((p) => p.token === searchToken)
    if (!point) return null
    const neighbors = findNearestNeighbors(point, EMBEDDING_DATA, 5)
    return { point, neighbors }
  }, [searchToken])

  const clusterButtons: ClusterKey[] = ['numbers', 'animals', 'colors', 'countries', 'emotions', 'code']

  return (
    <div ref={containerRef} className="space-y-3">
      {/* Controls */}
      <div className="flex flex-wrap items-center gap-2">
        <div className="relative flex-1 min-w-[180px] max-w-[280px]">
          <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-muted-foreground/50" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => handleSearch(e.target.value)}
            placeholder="Search token..."
            className="w-full pl-8 pr-8 py-1.5 text-sm bg-muted/50 border border-border rounded-md focus:outline-none focus:ring-1 focus:ring-primary/50"
          />
          {searchQuery && (
            <button
              onClick={() => handleSearch('')}
              className="absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground/50 hover:text-muted-foreground"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          )}
        </div>
        <div className="flex flex-wrap gap-1.5">
          {clusterButtons.map((cluster) => (
            <button
              key={cluster}
              onClick={() => handleClusterClick(cluster)}
              className="px-2.5 py-1 text-xs rounded-full border transition-colors"
              style={{
                borderColor: selectedCluster === cluster ? CLUSTER_COLORS[cluster] : 'var(--border)',
                backgroundColor: selectedCluster === cluster ? `${CLUSTER_COLORS[cluster]}20` : 'transparent',
                color: selectedCluster === cluster ? CLUSTER_COLORS[cluster] : 'var(--muted-foreground)',
              }}
            >
              {CLUSTER_LABELS[cluster]}
            </button>
          ))}
        </div>
      </div>

      {/* Search result info */}
      {searchResult && (
        <div className="px-3 py-2 bg-orange-500/10 border border-orange-500/20 rounded-md text-sm">
          <span className="font-mono font-bold text-orange-400">&ldquo;{searchResult.point.token}&rdquo;</span>
          <span className="text-muted-foreground"> — nearest: </span>
          {searchResult.neighbors.map((n, i) => (
            <span key={n.token}>
              <span className="font-mono text-muted-foreground">{n.token}</span>
              {i < searchResult.neighbors.length - 1 && <span className="text-muted-foreground/50">, </span>}
            </span>
          ))}
        </div>
      )}

      {searchQuery && !searchResult && (
        <div className="px-3 py-2 bg-muted/30 rounded-md text-sm text-muted-foreground/70">
          Token &ldquo;{searchQuery}&rdquo; not found in this sample. Try: cat, king, happy, five, Python, red
        </div>
      )}

      {/* Chart */}
      <ResponsiveContainer width="100%" height={height}>
        <ScatterChart
          margin={{ top: 10, right: 10, bottom: 20, left: 10 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" opacity={0.3} />
          <XAxis
            type="number"
            dataKey="x"
            domain={[-4.5, 4.5]}
            tick={false}
            axisLine={{ stroke: 'var(--border)' }}
            label={{ value: 'PCA dimension 1', position: 'bottom', offset: 0, style: { fontSize: 11, fill: 'var(--muted-foreground)' } }}
          />
          <YAxis
            type="number"
            dataKey="y"
            domain={[-4.5, 4.5]}
            tick={false}
            axisLine={{ stroke: 'var(--border)' }}
            label={{ value: 'PCA dimension 2', angle: -90, position: 'insideLeft', offset: 10, style: { fontSize: 11, fill: 'var(--muted-foreground)' } }}
          />
          <Tooltip content={<EmbeddingTooltip />} />
          <Scatter data={EMBEDDING_DATA} isAnimationActive={false}>
            {EMBEDDING_DATA.map((point, idx) => (
              <Cell
                key={`cell-${idx}`}
                fill={getPointColor(point, selectedCluster, searchToken)}
                opacity={getPointOpacity(point, selectedCluster, searchToken)}
                r={searchToken && point.token === searchToken ? 8 : 5}
              />
            ))}
          </Scatter>
        </ScatterChart>
      </ResponsiveContainer>

      {/* Legend */}
      <div className="flex flex-wrap gap-x-4 gap-y-1 justify-center">
        {Object.entries(CLUSTER_LABELS).map(([key, label]) => (
          <div key={key} className="flex items-center gap-1.5 text-xs text-muted-foreground/70">
            <div
              className="w-2.5 h-2.5 rounded-full"
              style={{ backgroundColor: CLUSTER_COLORS[key as ClusterKey] }}
            />
            {label}
          </div>
        ))}
      </div>

      <p className="text-xs text-muted-foreground/50 text-center">
        2D PCA projection of token embeddings. Coordinates are approximate&mdash;what matters is relative position and clustering.
      </p>
    </div>
  )
}

export function ClipDualEncoderDiagram() {
  return (
    <svg
      data-cde=""
      xmlns="http://www.w3.org/2000/svg"
      width="100%"
      viewBox="0 0 680 348"
      role="img"
      aria-label="CLIP dual-encoder architecture: image and text are encoded in parallel into 512-dim vectors, then compared via cosine similarity"
    >
      <defs>
        <style>{`
          [data-cde] text {
            font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
          }
          [data-cde] .th { font-size: 14px; font-weight: 500; }
          [data-cde] .ts { font-size: 12px; font-weight: 400; }
          [data-cde] .arr { stroke: #9c9a92; stroke-width: 1.5; }
          [data-cde] .ah { stroke: #9c9a92; stroke-width: 1.5; stroke-linecap: round; stroke-linejoin: round; }

          [data-cde] .n-blue rect { fill: #E6F1FB; stroke: #185FA5; }
          [data-cde] .n-blue .th { fill: #0C447C; }
          [data-cde] .n-blue .ts { fill: #185FA5; }
          [data-cde] .n-purple rect { fill: #EEEDFE; stroke: #534AB7; }
          [data-cde] .n-purple .th { fill: #3C3489; }
          [data-cde] .n-purple .ts { fill: #534AB7; }
          [data-cde] .n-amber rect { fill: #FAEEDA; stroke: #854F0B; }
          [data-cde] .n-amber .th { fill: #633806; }
          [data-cde] .n-amber .ts { fill: #854F0B; }

          @media (prefers-color-scheme: dark) {
            [data-cde] .arr { stroke: #6b6b65; }
            [data-cde] .ah { stroke: #6b6b65; }
            [data-cde] .n-blue rect { fill: #042C53; stroke: #85B7EB; }
            [data-cde] .n-blue .th { fill: #B5D4F4; }
            [data-cde] .n-blue .ts { fill: #85B7EB; }
            [data-cde] .n-purple rect { fill: #26215C; stroke: #AFA9EC; }
            [data-cde] .n-purple .th { fill: #CECBF6; }
            [data-cde] .n-purple .ts { fill: #AFA9EC; }
            [data-cde] .n-amber rect { fill: #412402; stroke: #EF9F27; }
            [data-cde] .n-amber .th { fill: #FAC775; }
            [data-cde] .n-amber .ts { fill: #EF9F27; }
          }
        `}</style>
        <marker
          id="cde-arrow"
          viewBox="0 0 10 10"
          refX="8"
          refY="5"
          markerWidth="6"
          markerHeight="6"
          orient="auto-start-reverse"
        >
          <path d="M2 1L8 5L2 9" fill="none" className="ah" />
        </marker>
      </defs>

      {/* Row 1: Image path (blue) */}
      <g className="n-blue">
        <rect x="64" y="40" width="120" height="56" rx="8" strokeWidth="0.5" />
        <text className="th" x="124" y="58" textAnchor="middle" dominantBaseline="central">
          Image
        </text>
        <text className="ts" x="124" y="78" textAnchor="middle" dominantBaseline="central">
          224 x 224
        </text>
      </g>

      <line x1="184" y1="68" x2="224" y2="68" className="arr" markerEnd="url(#cde-arrow)" />

      <g className="n-blue">
        <rect x="230" y="40" width="160" height="56" rx="8" strokeWidth="0.5" />
        <text className="th" x="310" y="58" textAnchor="middle" dominantBaseline="central">
          Image encoder
        </text>
        <text className="ts" x="310" y="78" textAnchor="middle" dominantBaseline="central">
          ResNet or ViT
        </text>
      </g>

      <line x1="390" y1="68" x2="430" y2="68" className="arr" markerEnd="url(#cde-arrow)" />

      <g className="n-blue">
        <rect x="436" y="40" width="152" height="56" rx="8" strokeWidth="0.5" />
        <text className="th" x="512" y="58" textAnchor="middle" dominantBaseline="central">
          Image vector
        </text>
        <text className="ts" x="512" y="78" textAnchor="middle" dominantBaseline="central">
          512-dim
        </text>
      </g>

      {/* Row 2: Text path (purple) */}
      <g className="n-purple">
        <rect x="64" y="156" width="120" height="56" rx="8" strokeWidth="0.5" />
        <text className="th" x="124" y="174" textAnchor="middle" dominantBaseline="central">
          Caption
        </text>
        <text className="ts" x="124" y="194" textAnchor="middle" dominantBaseline="central">
          tokens
        </text>
      </g>

      <line x1="184" y1="184" x2="224" y2="184" className="arr" markerEnd="url(#cde-arrow)" />

      <g className="n-purple">
        <rect x="230" y="156" width="160" height="56" rx="8" strokeWidth="0.5" />
        <text className="th" x="310" y="174" textAnchor="middle" dominantBaseline="central">
          Text encoder
        </text>
        <text className="ts" x="310" y="194" textAnchor="middle" dominantBaseline="central">
          Transformer
        </text>
      </g>

      <line x1="390" y1="184" x2="430" y2="184" className="arr" markerEnd="url(#cde-arrow)" />

      <g className="n-purple">
        <rect x="436" y="156" width="152" height="56" rx="8" strokeWidth="0.5" />
        <text className="th" x="512" y="174" textAnchor="middle" dominantBaseline="central">
          Text vector
        </text>
        <text className="ts" x="512" y="194" textAnchor="middle" dominantBaseline="central">
          512-dim
        </text>
      </g>

      {/* Convergence: Cosine similarity (amber) */}
      <g className="n-amber">
        <rect x="420" y="280" width="184" height="56" rx="8" strokeWidth="0.5" />
        <text className="th" x="512" y="298" textAnchor="middle" dominantBaseline="central">
          Cosine similarity
        </text>
        <text className="ts" x="512" y="318" textAnchor="middle" dominantBaseline="central">
          match score
        </text>
      </g>

      {/* Arrow: Image vector to cosine similarity (L-path routing right to avoid text vector) */}
      <path
        d="M 588 96 L 588 126 L 636 126 L 636 308 L 604 308"
        fill="none"
        className="arr"
        markerEnd="url(#cde-arrow)"
      />

      {/* Arrow: Text vector to cosine similarity (straight down) */}
      <line x1="512" y1="212" x2="512" y2="274" className="arr" markerEnd="url(#cde-arrow)" />
    </svg>
  )
}

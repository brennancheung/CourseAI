export function UnetAttentionBlocksDiagram() {
  return (
    <svg
      data-uab=""
      xmlns="http://www.w3.org/2000/svg"
      width="100%"
      viewBox="0 0 680 262"
      role="img"
      aria-label="U-Net attention block sequence: residual blocks, self-attention, and cross-attention repeating in a chain"
    >
      <defs>
        <style>{`
          [data-uab] text {
            font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
          }
          [data-uab] .th { font-size: 14px; font-weight: 500; }
          [data-uab] .ts { font-size: 12px; font-weight: 400; }
          [data-uab] .arr { stroke: #9c9a92; stroke-width: 1.5; }
          [data-uab] .ah { stroke: #9c9a92; stroke-width: 1.5; stroke-linecap: round; stroke-linejoin: round; }

          [data-uab] .n-amber rect { fill: #FAEEDA; stroke: #854F0B; }
          [data-uab] .n-amber .th { fill: #633806; }
          [data-uab] .n-amber .ts { fill: #854F0B; }

          [data-uab] .n-blue rect { fill: #E6F1FB; stroke: #185FA5; }
          [data-uab] .n-blue .th { fill: #0C447C; }
          [data-uab] .n-blue .ts { fill: #185FA5; }

          [data-uab] .n-purple rect { fill: #EEEDFE; stroke: #534AB7; }
          [data-uab] .n-purple .th { fill: #3C3489; }
          [data-uab] .n-purple .ts { fill: #534AB7; }

          @media (prefers-color-scheme: dark) {
            [data-uab] .arr { stroke: #6b6b65; }
            [data-uab] .ah { stroke: #6b6b65; }

            [data-uab] .n-amber rect { fill: #412402; stroke: #EF9F27; }
            [data-uab] .n-amber .th { fill: #FAC775; }
            [data-uab] .n-amber .ts { fill: #EF9F27; }

            [data-uab] .n-blue rect { fill: #042C53; stroke: #85B7EB; }
            [data-uab] .n-blue .th { fill: #B5D4F4; }
            [data-uab] .n-blue .ts { fill: #85B7EB; }

            [data-uab] .n-purple rect { fill: #26215C; stroke: #AFA9EC; }
            [data-uab] .n-purple .th { fill: #CECBF6; }
            [data-uab] .n-purple .ts { fill: #AFA9EC; }
          }
        `}</style>
        <marker
          id="uab-arrow"
          viewBox="0 0 10 10"
          refX="8"
          refY="5"
          markerWidth="6"
          markerHeight="6"
          orient="auto-start-reverse"
        >
          <path
            d="M2 1L8 5L2 9"
            fill="none"
            className="ah"
          />
        </marker>
      </defs>

      {/* Row 1: RB1 -> SA1 -> CA1 */}

      <g className="n-amber">
        <rect x="40" y="40" width="170" height="56" rx="8" strokeWidth="0.5" />
        <text className="th" x="125" y="58" textAnchor="middle" dominantBaseline="central">
          Residual block
        </text>
        <text className="ts" x="125" y="78" textAnchor="middle" dominantBaseline="central">
          AdaGN + t_emb
        </text>
      </g>

      <line
        x1="210" y1="68" x2="230" y2="68"
        className="arr"
        markerEnd="url(#uab-arrow)"
      />

      <g className="n-blue">
        <rect x="236" y="40" width="170" height="56" rx="8" strokeWidth="0.5" />
        <text className="th" x="321" y="58" textAnchor="middle" dominantBaseline="central">
          Self-attention
        </text>
        <text className="ts" x="321" y="78" textAnchor="middle" dominantBaseline="central">
          Spatial context
        </text>
      </g>

      <line
        x1="406" y1="68" x2="426" y2="68"
        className="arr"
        markerEnd="url(#uab-arrow)"
      />

      <g className="n-purple">
        <rect x="432" y="40" width="170" height="56" rx="8" strokeWidth="0.5" />
        <text className="th" x="517" y="58" textAnchor="middle" dominantBaseline="central">
          Cross-attention
        </text>
        <text className="ts" x="517" y="78" textAnchor="middle" dominantBaseline="central">
          text_emb
        </text>
      </g>

      {/* Wrap arrow: CA1 bottom -> RB2 top */}
      <path
        d="M 517 96 L 517 131 L 125 131 L 125 160"
        fill="none"
        className="arr"
        markerEnd="url(#uab-arrow)"
      />

      {/* Row 2: RB2 -> SA2 -> CA2 */}

      <g className="n-amber">
        <rect x="40" y="166" width="170" height="56" rx="8" strokeWidth="0.5" />
        <text className="th" x="125" y="184" textAnchor="middle" dominantBaseline="central">
          Residual block
        </text>
        <text className="ts" x="125" y="204" textAnchor="middle" dominantBaseline="central">
          AdaGN + t_emb
        </text>
      </g>

      <line
        x1="210" y1="194" x2="230" y2="194"
        className="arr"
        markerEnd="url(#uab-arrow)"
      />

      <g className="n-blue">
        <rect x="236" y="166" width="170" height="56" rx="8" strokeWidth="0.5" />
        <text className="th" x="321" y="184" textAnchor="middle" dominantBaseline="central">
          Self-attention
        </text>
        <text className="ts" x="321" y="204" textAnchor="middle" dominantBaseline="central">
          Spatial context
        </text>
      </g>

      <line
        x1="406" y1="194" x2="426" y2="194"
        className="arr"
        markerEnd="url(#uab-arrow)"
      />

      <g className="n-purple">
        <rect x="432" y="166" width="170" height="56" rx="8" strokeWidth="0.5" />
        <text className="th" x="517" y="184" textAnchor="middle" dominantBaseline="central">
          Cross-attention
        </text>
        <text className="ts" x="517" y="204" textAnchor="middle" dominantBaseline="central">
          text_emb
        </text>
      </g>
    </svg>
  )
}

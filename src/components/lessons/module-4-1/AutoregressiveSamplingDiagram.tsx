export function AutoregressiveSamplingDiagram() {
  return (
    <svg
      data-asd=""
      xmlns="http://www.w3.org/2000/svg"
      width="100%"
      viewBox="0 0 680 184"
      role="img"
      aria-label="Autoregressive sampling loop: context flows through language model, probability distribution, sample token, append to context, then loops back"
    >
      <defs>
        <style>{`
          [data-asd] text {
            font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
          }
          [data-asd] .th { font-size: 14px; font-weight: 500; }
          [data-asd] .ts { font-size: 12px; font-weight: 400; }
          [data-asd] .arr { stroke: #9c9a92; stroke-width: 1.5; }
          [data-asd] .ah { stroke: #9c9a92; stroke-width: 1.5; stroke-linecap: round; stroke-linejoin: round; }

          [data-asd] .n-blue rect { fill: #E6F1FB; stroke: #185FA5; }
          [data-asd] .n-blue .th { fill: #0C447C; }
          [data-asd] .n-blue .ts { fill: #185FA5; }
          [data-asd] .n-purple rect { fill: #EEEDFE; stroke: #534AB7; }
          [data-asd] .n-purple .th { fill: #3C3489; }
          [data-asd] .n-purple .ts { fill: #534AB7; }
          [data-asd] .n-teal rect { fill: #E1F5EE; stroke: #0F6E56; }
          [data-asd] .n-teal .th { fill: #085041; }
          [data-asd] .n-teal .ts { fill: #0F6E56; }
          [data-asd] .n-amber rect { fill: #FAEEDA; stroke: #854F0B; }
          [data-asd] .n-amber .th { fill: #633806; }
          [data-asd] .n-amber .ts { fill: #854F0B; }

          @media (prefers-color-scheme: dark) {
            [data-asd] .arr { stroke: #6b6b65; }
            [data-asd] .ah { stroke: #6b6b65; }
            [data-asd] .n-blue rect { fill: #042C53; stroke: #85B7EB; }
            [data-asd] .n-blue .th { fill: #B5D4F4; }
            [data-asd] .n-blue .ts { fill: #85B7EB; }
            [data-asd] .n-purple rect { fill: #26215C; stroke: #AFA9EC; }
            [data-asd] .n-purple .th { fill: #CECBF6; }
            [data-asd] .n-purple .ts { fill: #AFA9EC; }
            [data-asd] .n-teal rect { fill: #04342C; stroke: #5DCAA5; }
            [data-asd] .n-teal .th { fill: #9FE1CB; }
            [data-asd] .n-teal .ts { fill: #5DCAA5; }
            [data-asd] .n-amber rect { fill: #412402; stroke: #EF9F27; }
            [data-asd] .n-amber .th { fill: #FAC775; }
            [data-asd] .n-amber .ts { fill: #EF9F27; }
          }
        `}</style>
        <marker
          id="asd-arrow"
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

      {/* Node 1: Context (blue) — x=57, w=92 */}
      <g className="n-blue">
        <rect x="57" y="40" width="92" height="56" rx="8" strokeWidth="0.5" />
        <text className="th" x="103" y="58" textAnchor="middle" dominantBaseline="central">
          Context
        </text>
        <text className="ts" x="103" y="78" textAnchor="middle" dominantBaseline="central">
          tokens
        </text>
      </g>

      {/* Arrow 1 → 2 */}
      <line x1="149" y1="68" x2="163" y2="68" className="arr" markerEnd="url(#asd-arrow)" />

      {/* Node 2: Language model (purple) — x=163, w=100 */}
      <g className="n-purple">
        <rect x="163" y="40" width="100" height="56" rx="8" strokeWidth="0.5" />
        <text className="th" x="213" y="58" textAnchor="middle" dominantBaseline="central">
          Language
        </text>
        <text className="ts" x="213" y="78" textAnchor="middle" dominantBaseline="central">
          model
        </text>
      </g>

      {/* Arrow 2 → 3 */}
      <line x1="263" y1="68" x2="277" y2="68" className="arr" markerEnd="url(#asd-arrow)" />

      {/* Node 3: Probability distribution (teal) — x=277, w=128 */}
      <g className="n-teal">
        <rect x="277" y="40" width="128" height="56" rx="8" strokeWidth="0.5" />
        <text className="th" x="341" y="58" textAnchor="middle" dominantBaseline="central">
          Probability
        </text>
        <text className="ts" x="341" y="78" textAnchor="middle" dominantBaseline="central">
          distribution
        </text>
      </g>

      {/* Arrow 3 → 4 */}
      <line x1="405" y1="68" x2="419" y2="68" className="arr" markerEnd="url(#asd-arrow)" />

      {/* Node 4: Sample token (amber) — x=419, w=84 */}
      <g className="n-amber">
        <rect x="419" y="40" width="84" height="56" rx="8" strokeWidth="0.5" />
        <text className="th" x="461" y="58" textAnchor="middle" dominantBaseline="central">
          Sample
        </text>
        <text className="ts" x="461" y="78" textAnchor="middle" dominantBaseline="central">
          token
        </text>
      </g>

      {/* Arrow 4 → 5 */}
      <line x1="503" y1="68" x2="517" y2="68" className="arr" markerEnd="url(#asd-arrow)" />

      {/* Node 5: Append to context (amber) — x=517, w=106 */}
      <g className="n-amber">
        <rect x="517" y="40" width="106" height="56" rx="8" strokeWidth="0.5" />
        <text className="th" x="570" y="58" textAnchor="middle" dominantBaseline="central">
          Append
        </text>
        <text className="ts" x="570" y="78" textAnchor="middle" dominantBaseline="central">
          to context
        </text>
      </g>

      {/* Return arrow: Append → Context (path below the row) */}
      <path
        d="M 570 96 L 570 140 L 103 140 L 103 106"
        fill="none"
        className="arr"
        markerEnd="url(#asd-arrow)"
      />

      {/* Loop label on the return path */}
      <text
        className="ts"
        x="336"
        y="156"
        textAnchor="middle"
        dominantBaseline="central"
        fill="#9c9a92"
      >
        repeat until done
      </text>
    </svg>
  )
}

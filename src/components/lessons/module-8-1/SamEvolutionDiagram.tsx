export function SamEvolutionDiagram() {
  return (
    <svg
      data-sed=""
      xmlns="http://www.w3.org/2000/svg"
      width="100%"
      viewBox="0 0 680 136"
      role="img"
      aria-label="SAM evolution: SAM 1 images, SAM 2 adds video, SAM 3 adds concepts and text"
    >
      <defs>
        <style>{`
          [data-sed] text {
            font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
          }
          [data-sed] .th { font-size: 14px; font-weight: 500; }
          [data-sed] .ts { font-size: 12px; font-weight: 400; }
          [data-sed] .arr { stroke: #9c9a92; stroke-width: 1.5; }
          [data-sed] .lbl { fill: #9c9a92; font-size: 12px; font-weight: 400; }
          [data-sed] .ah { stroke: #9c9a92; stroke-width: 1.5; stroke-linecap: round; stroke-linejoin: round; }

          [data-sed] .n-blue rect { fill: #E6F1FB; stroke: #185FA5; }
          [data-sed] .n-blue .th { fill: #0C447C; }
          [data-sed] .n-blue .ts { fill: #185FA5; }
          [data-sed] .n-purple rect { fill: #EEEDFE; stroke: #534AB7; }
          [data-sed] .n-purple .th { fill: #3C3489; }
          [data-sed] .n-purple .ts { fill: #534AB7; }
          [data-sed] .n-teal rect { fill: #E1F5EE; stroke: #0F6E56; }
          [data-sed] .n-teal .th { fill: #085041; }
          [data-sed] .n-teal .ts { fill: #0F6E56; }

          @media (prefers-color-scheme: dark) {
            [data-sed] .arr { stroke: #6b6b65; }
            [data-sed] .lbl { fill: #6b6b65; }
            [data-sed] .ah { stroke: #6b6b65; }
            [data-sed] .n-blue rect { fill: #042C53; stroke: #85B7EB; }
            [data-sed] .n-blue .th { fill: #B5D4F4; }
            [data-sed] .n-blue .ts { fill: #85B7EB; }
            [data-sed] .n-purple rect { fill: #26215C; stroke: #AFA9EC; }
            [data-sed] .n-purple .th { fill: #CECBF6; }
            [data-sed] .n-purple .ts { fill: #AFA9EC; }
            [data-sed] .n-teal rect { fill: #04342C; stroke: #5DCAA5; }
            [data-sed] .n-teal .th { fill: #9FE1CB; }
            [data-sed] .n-teal .ts { fill: #5DCAA5; }
          }
        `}</style>
        <marker
          id="sed-arrow"
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

      {/* Node 1: SAM 1 (blue) */}
      <g className="n-blue">
        <rect x="40" y="40" width="140" height="56" rx="8" strokeWidth="0.5" />
        <text className="th" x="110" y="58" textAnchor="middle" dominantBaseline="central">
          SAM 1 (2023)
        </text>
        <text className="ts" x="110" y="76" textAnchor="middle" dominantBaseline="central">
          Images
        </text>
      </g>

      {/* Arrow 1 -> 2 */}
      <line x1="180" y1="68" x2="246" y2="68" className="arr" markerEnd="url(#sed-arrow)" />
      <text className="lbl" x="213" y="56" textAnchor="middle" dominantBaseline="central">
        + memory
      </text>

      {/* Node 2: SAM 2 (purple) */}
      <g className="n-purple">
        <rect x="252" y="40" width="152" height="56" rx="8" strokeWidth="0.5" />
        <text className="th" x="328" y="58" textAnchor="middle" dominantBaseline="central">
          SAM 2 (2024)
        </text>
        <text className="ts" x="328" y="76" textAnchor="middle" dominantBaseline="central">
          + Video
        </text>
      </g>

      {/* Arrow 2 -> 3 */}
      <line x1="404" y1="68" x2="470" y2="68" className="arr" markerEnd="url(#sed-arrow)" />
      <text className="lbl" x="437" y="56" textAnchor="middle" dominantBaseline="central">
        + language
      </text>

      {/* Node 3: SAM 3 (teal) */}
      <g className="n-teal">
        <rect x="476" y="40" width="164" height="56" rx="8" strokeWidth="0.5" />
        <text className="th" x="558" y="58" textAnchor="middle" dominantBaseline="central">
          SAM 3 (2025)
        </text>
        <text className="ts" x="558" y="76" textAnchor="middle" dominantBaseline="central">
          + Concepts/Text
        </text>
      </g>
    </svg>
  )
}

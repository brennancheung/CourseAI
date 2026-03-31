export function DataPipelineDiagram() {
  return (
    <svg
      data-dpd=""
      xmlns="http://www.w3.org/2000/svg"
      width="100%"
      viewBox="0 0 680 136"
      role="img"
      aria-label="Data pipeline: raw data flows through Dataset, DataLoader, then training loop"
    >
      <defs>
        <style>{`
          [data-dpd] text {
            font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
          }
          [data-dpd] .th { font-size: 14px; font-weight: 500; }
          [data-dpd] .ts { font-size: 12px; font-weight: 400; }
          [data-dpd] .arr { stroke: #9c9a92; stroke-width: 1.5; }
          [data-dpd] .ah { stroke: #9c9a92; stroke-width: 1.5; stroke-linecap: round; stroke-linejoin: round; }

          [data-dpd] .n-blue rect { fill: #E6F1FB; stroke: #185FA5; }
          [data-dpd] .n-blue .th { fill: #0C447C; }
          [data-dpd] .n-blue .ts { fill: #185FA5; }
          [data-dpd] .n-purple rect { fill: #EEEDFE; stroke: #534AB7; }
          [data-dpd] .n-purple .th { fill: #3C3489; }
          [data-dpd] .n-purple .ts { fill: #534AB7; }
          [data-dpd] .n-teal rect { fill: #E1F5EE; stroke: #0F6E56; }
          [data-dpd] .n-teal .th { fill: #085041; }
          [data-dpd] .n-teal .ts { fill: #0F6E56; }
          [data-dpd] .n-coral rect { fill: #FAECE7; stroke: #993C1D; }
          [data-dpd] .n-coral .th { fill: #712B13; }
          [data-dpd] .n-coral .ts { fill: #993C1D; }

          @media (prefers-color-scheme: dark) {
            [data-dpd] .arr { stroke: #6b6b65; }
            [data-dpd] .ah { stroke: #6b6b65; }
            [data-dpd] .n-blue rect { fill: #042C53; stroke: #85B7EB; }
            [data-dpd] .n-blue .th { fill: #B5D4F4; }
            [data-dpd] .n-blue .ts { fill: #85B7EB; }
            [data-dpd] .n-purple rect { fill: #26215C; stroke: #AFA9EC; }
            [data-dpd] .n-purple .th { fill: #CECBF6; }
            [data-dpd] .n-purple .ts { fill: #AFA9EC; }
            [data-dpd] .n-teal rect { fill: #04342C; stroke: #5DCAA5; }
            [data-dpd] .n-teal .th { fill: #9FE1CB; }
            [data-dpd] .n-teal .ts { fill: #5DCAA5; }
            [data-dpd] .n-coral rect { fill: #4A1B0C; stroke: #F0997B; }
            [data-dpd] .n-coral .th { fill: #F5C4B3; }
            [data-dpd] .n-coral .ts { fill: #F0997B; }
          }
        `}</style>
        <marker
          id="dpd-arrow"
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

      {/* Raw data */}
      <g className="n-blue">
        <rect x="40" y="40" width="132" height="56" rx="8" strokeWidth="0.5" />
        <text className="th" x="106" y="58" textAnchor="middle" dominantBaseline="central">
          Raw data
        </text>
        <text className="ts" x="106" y="78" textAnchor="middle" dominantBaseline="central">
          files, arrays, etc.
        </text>
      </g>

      {/* Arrow: Raw data -> Dataset */}
      <line x1="172" y1="68" x2="192" y2="68" className="arr" markerEnd="url(#dpd-arrow)" />

      {/* Dataset */}
      <g className="n-purple">
        <rect x="196" y="40" width="132" height="56" rx="8" strokeWidth="0.5" />
        <text className="th" x="262" y="58" textAnchor="middle" dominantBaseline="central">
          Dataset
        </text>
        <text className="ts" x="262" y="78" textAnchor="middle" dominantBaseline="central">
          getitem(i)
        </text>
      </g>

      {/* Arrow: Dataset -> DataLoader */}
      <line x1="328" y1="68" x2="348" y2="68" className="arr" markerEnd="url(#dpd-arrow)" />

      {/* DataLoader */}
      <g className="n-teal">
        <rect x="352" y="40" width="132" height="56" rx="8" strokeWidth="0.5" />
        <text className="th" x="418" y="58" textAnchor="middle" dominantBaseline="central">
          DataLoader
        </text>
        <text className="ts" x="418" y="78" textAnchor="middle" dominantBaseline="central">
          batch + shuffle
        </text>
      </g>

      {/* Arrow: DataLoader -> Training loop */}
      <line x1="484" y1="68" x2="504" y2="68" className="arr" markerEnd="url(#dpd-arrow)" />

      {/* Training loop */}
      <g className="n-coral">
        <rect x="508" y="40" width="132" height="56" rx="8" strokeWidth="0.5" />
        <text className="th" x="574" y="58" textAnchor="middle" dominantBaseline="central">
          Training loop
        </text>
        <text className="ts" x="574" y="78" textAnchor="middle" dominantBaseline="central">
          fwd / bwd / step
        </text>
      </g>
    </svg>
  )
}

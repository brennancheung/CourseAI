export function VqvaeTokenizationDiagram() {
  return (
    <svg
      data-vtd=""
      xmlns="http://www.w3.org/2000/svg"
      width="100%"
      viewBox="0 0 680 252"
      role="img"
      aria-label="VQ-VAE tokenization pipeline: image to encoder to continuous vectors to codebook lookup to integer tokens to text-compatible format"
    >
      <defs>
        <style>{`
          [data-vtd] text {
            font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
          }
          [data-vtd] .th { font-size: 14px; font-weight: 500; }
          [data-vtd] .ts { font-size: 12px; font-weight: 400; }
          [data-vtd] .arr { stroke: #9c9a92; stroke-width: 1.5; }
          [data-vtd] .ah { stroke: #9c9a92; stroke-width: 1.5; stroke-linecap: round; stroke-linejoin: round; }

          [data-vtd] .n-blue rect { fill: #E6F1FB; stroke: #185FA5; }
          [data-vtd] .n-blue .th { fill: #0C447C; }
          [data-vtd] .n-blue .ts { fill: #185FA5; }
          [data-vtd] .n-purple rect { fill: #EEEDFE; stroke: #534AB7; }
          [data-vtd] .n-purple .th { fill: #3C3489; }
          [data-vtd] .n-purple .ts { fill: #534AB7; }
          [data-vtd] .n-amber rect { fill: #FAEEDA; stroke: #854F0B; }
          [data-vtd] .n-amber .th { fill: #633806; }
          [data-vtd] .n-amber .ts { fill: #854F0B; }
          [data-vtd] .n-teal rect { fill: #E1F5EE; stroke: #0F6E56; }
          [data-vtd] .n-teal .th { fill: #085041; }
          [data-vtd] .n-teal .ts { fill: #0F6E56; }

          @media (prefers-color-scheme: dark) {
            [data-vtd] .arr { stroke: #6b6b65; }
            [data-vtd] .ah { stroke: #6b6b65; }
            [data-vtd] .n-blue rect { fill: #042C53; stroke: #85B7EB; }
            [data-vtd] .n-blue .th { fill: #B5D4F4; }
            [data-vtd] .n-blue .ts { fill: #85B7EB; }
            [data-vtd] .n-purple rect { fill: #26215C; stroke: #AFA9EC; }
            [data-vtd] .n-purple .th { fill: #CECBF6; }
            [data-vtd] .n-purple .ts { fill: #AFA9EC; }
            [data-vtd] .n-amber rect { fill: #412402; stroke: #EF9F27; }
            [data-vtd] .n-amber .th { fill: #FAC775; }
            [data-vtd] .n-amber .ts { fill: #EF9F27; }
            [data-vtd] .n-teal rect { fill: #04342C; stroke: #5DCAA5; }
            [data-vtd] .n-teal .th { fill: #9FE1CB; }
            [data-vtd] .n-teal .ts { fill: #5DCAA5; }
          }
        `}</style>
        <marker
          id="vtd-arrow"
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

      {/* Row 1: Image → VQ-VAE Encoder → Continuous Vectors */}
      <g className="n-blue">
        <rect x="58" y="40" width="152" height="56" rx="8" strokeWidth="0.5" />
        <text className="th" x="134" y="58" textAnchor="middle" dominantBaseline="central">
          256x256 Image
        </text>
        <text className="ts" x="134" y="76" textAnchor="middle" dominantBaseline="central">
          Input photograph
        </text>
      </g>

      <line x1="210" y1="68" x2="240" y2="68" className="arr" markerEnd="url(#vtd-arrow)" />

      <g className="n-purple">
        <rect x="240" y="40" width="160" height="56" rx="8" strokeWidth="0.5" />
        <text className="th" x="320" y="58" textAnchor="middle" dominantBaseline="central">
          VQ-VAE Encoder
        </text>
        <text className="ts" x="320" y="76" textAnchor="middle" dominantBaseline="central">
          Neural network
        </text>
      </g>

      <line x1="400" y1="68" x2="430" y2="68" className="arr" markerEnd="url(#vtd-arrow)" />

      <g className="n-purple">
        <rect x="430" y="40" width="192" height="56" rx="8" strokeWidth="0.5" />
        <text className="th" x="526" y="58" textAnchor="middle" dominantBaseline="central">
          Continuous Vectors
        </text>
        <text className="ts" x="526" y="76" textAnchor="middle" dominantBaseline="central">
          32x32 grid
        </text>
      </g>

      {/* Connecting arrow: Row 1 → Row 2 (L-shaped path) */}
      <path
        d="M526 96 L526 126 L142 126 L142 156"
        fill="none"
        className="arr"
        markerEnd="url(#vtd-arrow)"
      />

      {/* Row 2: Codebook Lookup → Integer Tokens → Text-compatible */}
      <g className="n-amber">
        <rect x="58" y="156" width="168" height="56" rx="8" strokeWidth="0.5" />
        <text className="th" x="142" y="174" textAnchor="middle" dominantBaseline="central">
          Codebook Lookup
        </text>
        <text className="ts" x="142" y="192" textAnchor="middle" dominantBaseline="central">
          Nearest neighbor
        </text>
      </g>

      <line x1="226" y1="184" x2="256" y2="184" className="arr" markerEnd="url(#vtd-arrow)" />

      <g className="n-amber">
        <rect x="256" y="156" width="160" height="56" rx="8" strokeWidth="0.5" />
        <text className="th" x="336" y="174" textAnchor="middle" dominantBaseline="central">
          Integer Tokens
        </text>
        <text className="ts" x="336" y="192" textAnchor="middle" dominantBaseline="central">
          1,024 per image
        </text>
      </g>

      <line x1="416" y1="184" x2="446" y2="184" className="arr" markerEnd="url(#vtd-arrow)" />

      <g className="n-teal">
        <rect x="446" y="156" width="176" height="56" rx="8" strokeWidth="0.5" />
        <text className="th" x="534" y="174" textAnchor="middle" dominantBaseline="central">
          Text-compatible
        </text>
        <text className="ts" x="534" y="192" textAnchor="middle" dominantBaseline="central">
          Same token format
        </text>
      </g>
    </svg>
  )
}

export function NnModuleResidualDiagram() {
  return (
    <svg
      data-nrd=""
      xmlns="http://www.w3.org/2000/svg"
      width="100%"
      viewBox="0 0 680 168"
      role="img"
      aria-label="Residual block: x flows through ReLU and nn.Linear to add node, with a skip connection from x directly to add"
    >
      <defs>
        <style>{`
          [data-nrd] text {
            font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
          }
          [data-nrd] .th { font-size: 14px; font-weight: 500; }
          [data-nrd] .ts { font-size: 12px; font-weight: 400; }
          [data-nrd] .arr { stroke: #9c9a92; stroke-width: 1.5; }
          [data-nrd] .ah { stroke: #9c9a92; stroke-width: 1.5; stroke-linecap: round; stroke-linejoin: round; }

          [data-nrd] .n-blue rect { fill: #E6F1FB; stroke: #185FA5; }
          [data-nrd] .n-blue .th { fill: #0C447C; }
          [data-nrd] .n-purple rect { fill: #EEEDFE; stroke: #534AB7; }
          [data-nrd] .n-purple .th { fill: #3C3489; }
          [data-nrd] .n-gray rect { fill: #F1EFE8; stroke: #5F5E5A; }
          [data-nrd] .n-gray .th { fill: #444441; }
          [data-nrd] .n-gray .ts { fill: #5F5E5A; }

          @media (prefers-color-scheme: dark) {
            [data-nrd] .arr { stroke: #6b6b65; }
            [data-nrd] .ah { stroke: #6b6b65; }
            [data-nrd] .n-blue rect { fill: #042C53; stroke: #85B7EB; }
            [data-nrd] .n-blue .th { fill: #B5D4F4; }
            [data-nrd] .n-purple rect { fill: #26215C; stroke: #AFA9EC; }
            [data-nrd] .n-purple .th { fill: #CECBF6; }
            [data-nrd] .n-gray rect { fill: #2C2C2A; stroke: #B4B2A9; }
            [data-nrd] .n-gray .th { fill: #D3D1C7; }
            [data-nrd] .n-gray .ts { fill: #B4B2A9; }
          }
        `}</style>
        <marker
          id="nrd-arrow"
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

      {/* x (input) */}
      <g className="n-blue">
        <rect x="95" y="40" width="70" height="44" rx="8" strokeWidth="0.5" />
        <text className="th" x="130" y="62" textAnchor="middle" dominantBaseline="central">
          x
        </text>
      </g>

      <line x1="165" y1="62" x2="179" y2="62" className="arr" markerEnd="url(#nrd-arrow)" />

      {/* ReLU */}
      <g className="n-gray">
        <rect x="185" y="40" width="80" height="44" rx="8" strokeWidth="0.5" />
        <text className="th" x="225" y="62" textAnchor="middle" dominantBaseline="central">
          ReLU
        </text>
      </g>

      <line x1="265" y1="62" x2="279" y2="62" className="arr" markerEnd="url(#nrd-arrow)" />

      {/* nn.Linear */}
      <g className="n-purple">
        <rect x="285" y="40" width="120" height="44" rx="8" strokeWidth="0.5" />
        <text className="th" x="345" y="62" textAnchor="middle" dominantBaseline="central">
          nn.Linear
        </text>
      </g>

      <line x1="405" y1="62" x2="419" y2="62" className="arr" markerEnd="url(#nrd-arrow)" />

      {/* + (add) */}
      <g className="n-gray">
        <rect x="425" y="40" width="70" height="44" rx="8" strokeWidth="0.5" />
        <text className="th" x="460" y="62" textAnchor="middle" dominantBaseline="central">
          +
        </text>
      </g>

      <line x1="495" y1="62" x2="509" y2="62" className="arr" markerEnd="url(#nrd-arrow)" />

      {/* output */}
      <g className="n-blue">
        <rect x="515" y="40" width="70" height="44" rx="8" strokeWidth="0.5" />
        <text className="th" x="550" y="62" textAnchor="middle" dominantBaseline="central">
          output
        </text>
      </g>

      {/* Skip connection: x bottom -> + bottom, routed below all boxes */}
      <path
        d="M130 84 L130 118 L460 118 L460 84"
        fill="none"
        className="arr"
        markerEnd="url(#nrd-arrow)"
      />
      <text className="ts" x="295" y="134" textAnchor="middle" dominantBaseline="central">
        skip connection
      </text>
    </svg>
  )
}

export function AutogradComputationalGraphDiagram() {
  return (
    <svg
      data-acg=""
      xmlns="http://www.w3.org/2000/svg"
      width="100%"
      viewBox="0 0 680 300"
      role="img"
      aria-label="Computational graph: x flows through w1*x, +b1, ReLU, w2*a1, +b2 to MSE loss, with gradient values on parameter nodes"
    >
      <defs>
        <style>{`
          [data-acg] text {
            font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
          }
          [data-acg] .th { font-size: 14px; font-weight: 500; }
          [data-acg] .ts { font-size: 12px; font-weight: 400; }
          [data-acg] .arr { stroke: #9c9a92; stroke-width: 1.5; }
          [data-acg] .ah { stroke: #9c9a92; stroke-width: 1.5; stroke-linecap: round; stroke-linejoin: round; }
          [data-acg] .flow-label { font-size: 12px; font-weight: 400; fill: #9c9a92; opacity: 0.45; }

          [data-acg] .n-purple rect { fill: #EEEDFE; stroke: #534AB7; }
          [data-acg] .n-purple .th { fill: #3C3489; }
          [data-acg] .n-purple .ts { fill: #534AB7; }
          [data-acg] .n-coral rect { fill: #FAECE7; stroke: #993C1D; }
          [data-acg] .n-coral .th { fill: #712B13; }
          [data-acg] .n-coral .ts { fill: #993C1D; }
          [data-acg] .n-gray rect { fill: #F1EFE8; stroke: #5F5E5A; }
          [data-acg] .n-gray .th { fill: #444441; }
          [data-acg] .n-gray .ts { fill: #5F5E5A; }
          [data-acg] .n-blue rect { fill: #E6F1FB; stroke: #185FA5; }
          [data-acg] .n-blue .th { fill: #0C447C; }
          [data-acg] .n-blue .ts { fill: #185FA5; }

          @media (prefers-color-scheme: dark) {
            [data-acg] .arr { stroke: #6b6b65; }
            [data-acg] .ah { stroke: #6b6b65; }
            [data-acg] .flow-label { fill: #6b6b65; }
            [data-acg] .n-purple rect { fill: #26215C; stroke: #AFA9EC; }
            [data-acg] .n-purple .th { fill: #CECBF6; }
            [data-acg] .n-purple .ts { fill: #AFA9EC; }
            [data-acg] .n-coral rect { fill: #4A1B0C; stroke: #F0997B; }
            [data-acg] .n-coral .th { fill: #F5C4B3; }
            [data-acg] .n-coral .ts { fill: #F0997B; }
            [data-acg] .n-gray rect { fill: #2C2C2A; stroke: #B4B2A9; }
            [data-acg] .n-gray .th { fill: #D3D1C7; }
            [data-acg] .n-gray .ts { fill: #B4B2A9; }
            [data-acg] .n-blue rect { fill: #042C53; stroke: #85B7EB; }
            [data-acg] .n-blue .th { fill: #B5D4F4; }
            [data-acg] .n-blue .ts { fill: #85B7EB; }
          }
        `}</style>
        <marker
          id="acg-arrow"
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

      {/* ===== Parameter nodes (top row, y=40, height=56) ===== */}

      {/* w1 */}
      <g className="n-purple">
        <rect x="78" y="40" width="120" height="56" rx="8" strokeWidth="0.5" />
        <text className="th" x="138" y="58" textAnchor="middle" dominantBaseline="central">
          w1 = 0.5
        </text>
        <text className="ts" x="138" y="78" textAnchor="middle" dominantBaseline="central">
          grad: 1.356
        </text>
      </g>

      {/* b1 */}
      <g className="n-purple">
        <rect x="206" y="40" width="120" height="56" rx="8" strokeWidth="0.5" />
        <text className="th" x="266" y="58" textAnchor="middle" dominantBaseline="central">
          b1 = 0.1
        </text>
        <text className="ts" x="266" y="78" textAnchor="middle" dominantBaseline="central">
          grad: 0.678
        </text>
      </g>

      {/* w2 */}
      <g className="n-purple">
        <rect x="365" y="40" width="130" height="56" rx="8" strokeWidth="0.5" />
        <text className="th" x="430" y="58" textAnchor="middle" dominantBaseline="central">
          w2 = -0.3
        </text>
        <text className="ts" x="430" y="78" textAnchor="middle" dominantBaseline="central">
          grad: -2.486
        </text>
      </g>

      {/* b2 */}
      <g className="n-purple">
        <rect x="503" y="40" width="130" height="56" rx="8" strokeWidth="0.5" />
        <text className="th" x="568" y="58" textAnchor="middle" dominantBaseline="central">
          b2 = 0.2
        </text>
        <text className="ts" x="568" y="78" textAnchor="middle" dominantBaseline="central">
          grad: -2.260
        </text>
      </g>

      {/* ===== Arrows: params down to operations ===== */}
      <line x1="138" y1="96" x2="138" y2="162" className="arr" markerEnd="url(#acg-arrow)" />
      <line x1="266" y1="96" x2="236" y2="162" className="arr" markerEnd="url(#acg-arrow)" />
      <line x1="430" y1="96" x2="430" y2="162" className="arr" markerEnd="url(#acg-arrow)" />
      <line x1="568" y1="96" x2="528" y2="162" className="arr" markerEnd="url(#acg-arrow)" />

      {/* ===== Operation nodes (bottom row, y=168, height=56) ===== */}

      {/* x = 2.0 (input) */}
      <g className="n-blue">
        <rect x="40" y="168" width="52" height="56" rx="8" strokeWidth="0.5" />
        <text className="th" x="66" y="186" textAnchor="middle" dominantBaseline="central">
          x
        </text>
        <text className="ts" x="66" y="206" textAnchor="middle" dominantBaseline="central">
          2.0
        </text>
      </g>

      <line x1="92" y1="196" x2="106" y2="196" className="arr" markerEnd="url(#acg-arrow)" />

      {/* w1*x */}
      <g className="n-gray">
        <rect x="108" y="168" width="78" height="56" rx="8" strokeWidth="0.5" />
        <text className="th" x="147" y="186" textAnchor="middle" dominantBaseline="central">
          w1*x
        </text>
        <text className="ts" x="147" y="206" textAnchor="middle" dominantBaseline="central">
          = 1.0
        </text>
      </g>

      <line x1="186" y1="196" x2="200" y2="196" className="arr" markerEnd="url(#acg-arrow)" />

      {/* +b1 */}
      <g className="n-gray">
        <rect x="202" y="168" width="78" height="56" rx="8" strokeWidth="0.5" />
        <text className="th" x="241" y="186" textAnchor="middle" dominantBaseline="central">
          + b1
        </text>
        <text className="ts" x="241" y="206" textAnchor="middle" dominantBaseline="central">
          z1 = 1.1
        </text>
      </g>

      <line x1="280" y1="196" x2="294" y2="196" className="arr" markerEnd="url(#acg-arrow)" />

      {/* ReLU */}
      <g className="n-gray">
        <rect x="296" y="168" width="78" height="56" rx="8" strokeWidth="0.5" />
        <text className="th" x="335" y="186" textAnchor="middle" dominantBaseline="central">
          ReLU
        </text>
        <text className="ts" x="335" y="206" textAnchor="middle" dominantBaseline="central">
          a1 = 1.1
        </text>
      </g>

      <line x1="374" y1="196" x2="388" y2="196" className="arr" markerEnd="url(#acg-arrow)" />

      {/* w2*a1 */}
      <g className="n-gray">
        <rect x="390" y="168" width="82" height="56" rx="8" strokeWidth="0.5" />
        <text className="th" x="431" y="186" textAnchor="middle" dominantBaseline="central">
          w2*a1
        </text>
        <text className="ts" x="431" y="206" textAnchor="middle" dominantBaseline="central">
          = -0.33
        </text>
      </g>

      <line x1="472" y1="196" x2="486" y2="196" className="arr" markerEnd="url(#acg-arrow)" />

      {/* +b2 */}
      <g className="n-gray">
        <rect x="488" y="168" width="82" height="56" rx="8" strokeWidth="0.5" />
        <text className="th" x="529" y="186" textAnchor="middle" dominantBaseline="central">
          + b2
        </text>
        <text className="ts" x="529" y="206" textAnchor="middle" dominantBaseline="central">
          {'\u0177'} = -0.13
        </text>
      </g>

      <line x1="570" y1="196" x2="584" y2="196" className="arr" markerEnd="url(#acg-arrow)" />

      {/* MSE loss */}
      <g className="n-coral">
        <rect x="586" y="168" width="54" height="56" rx="8" strokeWidth="0.5" />
        <text className="th" x="613" y="186" textAnchor="middle" dominantBaseline="central">
          MSE
        </text>
        <text className="ts" x="613" y="206" textAnchor="middle" dominantBaseline="central">
          1.2769
        </text>
      </g>

      {/* ===== Flow direction label ===== */}
      <text
        className="flow-label"
        x="340"
        y="260"
        textAnchor="middle"
        dominantBaseline="central"
      >
        {'forward pass \u2192'}
      </text>
    </svg>
  )
}

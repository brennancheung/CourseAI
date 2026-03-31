export function NnModuleNetworkDiagram() {
  return (
    <svg
      data-nnd=""
      xmlns="http://www.w3.org/2000/svg"
      width="100%"
      viewBox="0 0 680 136"
      role="img"
      aria-label="Simple network: x flows through self.layer1, torch.clamp ReLU, self.layer2, to y_hat"
    >
      <defs>
        <style>{`
          [data-nnd] text {
            font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
          }
          [data-nnd] .th { font-size: 14px; font-weight: 500; }
          [data-nnd] .ts { font-size: 12px; font-weight: 400; }
          [data-nnd] .arr { stroke: #9c9a92; stroke-width: 1.5; }
          [data-nnd] .ah { stroke: #9c9a92; stroke-width: 1.5; stroke-linecap: round; stroke-linejoin: round; }

          [data-nnd] .n-blue rect { fill: #E6F1FB; stroke: #185FA5; }
          [data-nnd] .n-blue .th { fill: #0C447C; }
          [data-nnd] .n-blue .ts { fill: #185FA5; }
          [data-nnd] .n-purple rect { fill: #EEEDFE; stroke: #534AB7; }
          [data-nnd] .n-purple .th { fill: #3C3489; }
          [data-nnd] .n-purple .ts { fill: #534AB7; }
          [data-nnd] .n-gray rect { fill: #F1EFE8; stroke: #5F5E5A; }
          [data-nnd] .n-gray .th { fill: #444441; }
          [data-nnd] .n-gray .ts { fill: #5F5E5A; }

          @media (prefers-color-scheme: dark) {
            [data-nnd] .arr { stroke: #6b6b65; }
            [data-nnd] .ah { stroke: #6b6b65; }
            [data-nnd] .n-blue rect { fill: #042C53; stroke: #85B7EB; }
            [data-nnd] .n-blue .th { fill: #B5D4F4; }
            [data-nnd] .n-blue .ts { fill: #85B7EB; }
            [data-nnd] .n-purple rect { fill: #26215C; stroke: #AFA9EC; }
            [data-nnd] .n-purple .th { fill: #CECBF6; }
            [data-nnd] .n-purple .ts { fill: #AFA9EC; }
            [data-nnd] .n-gray rect { fill: #2C2C2A; stroke: #B4B2A9; }
            [data-nnd] .n-gray .th { fill: #D3D1C7; }
            [data-nnd] .n-gray .ts { fill: #B4B2A9; }
          }
        `}</style>
        <marker
          id="nnd-arrow"
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
        <rect x="62" y="40" width="70" height="56" rx="8" strokeWidth="0.5" />
        <text className="th" x="97" y="58" textAnchor="middle" dominantBaseline="central">
          x
        </text>
        <text className="ts" x="97" y="78" textAnchor="middle" dominantBaseline="central">
          input
        </text>
      </g>

      <line x1="132" y1="68" x2="144" y2="68" className="arr" markerEnd="url(#nnd-arrow)" />

      {/* self.layer1 */}
      <g className="n-purple">
        <rect x="150" y="40" width="124" height="56" rx="8" strokeWidth="0.5" />
        <text className="th" x="212" y="58" textAnchor="middle" dominantBaseline="central">
          self.layer1
        </text>
        <text className="ts" x="212" y="78" textAnchor="middle" dominantBaseline="central">
          nn.Linear(1,1)
        </text>
      </g>

      <line x1="274" y1="68" x2="286" y2="68" className="arr" markerEnd="url(#nnd-arrow)" />

      {/* torch.clamp (ReLU) */}
      <g className="n-gray">
        <rect x="292" y="40" width="112" height="56" rx="8" strokeWidth="0.5" />
        <text className="th" x="348" y="58" textAnchor="middle" dominantBaseline="central">
          torch.clamp
        </text>
        <text className="ts" x="348" y="78" textAnchor="middle" dominantBaseline="central">
          ReLU
        </text>
      </g>

      <line x1="404" y1="68" x2="416" y2="68" className="arr" markerEnd="url(#nnd-arrow)" />

      {/* self.layer2 */}
      <g className="n-purple">
        <rect x="422" y="40" width="124" height="56" rx="8" strokeWidth="0.5" />
        <text className="th" x="484" y="58" textAnchor="middle" dominantBaseline="central">
          self.layer2
        </text>
        <text className="ts" x="484" y="78" textAnchor="middle" dominantBaseline="central">
          nn.Linear(1,1)
        </text>
      </g>

      <line x1="546" y1="68" x2="558" y2="68" className="arr" markerEnd="url(#nnd-arrow)" />

      {/* y_hat (output) */}
      <g className="n-blue">
        <rect x="564" y="40" width="70" height="56" rx="8" strokeWidth="0.5" />
        <text className="th" x="599" y="58" textAnchor="middle" dominantBaseline="central">
          y_hat
        </text>
        <text className="ts" x="599" y="78" textAnchor="middle" dominantBaseline="central">
          output
        </text>
      </g>
    </svg>
  )
}

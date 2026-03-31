export function AutoencoderArchitectureDiagram() {
  return (
    <svg
      data-aad=""
      xmlns="http://www.w3.org/2000/svg"
      width="100%"
      viewBox="0 0 680 176"
      role="img"
      aria-label="Autoencoder hourglass architecture: input compresses through encoder to bottleneck, then reconstructs through decoder to output"
    >
      <defs>
        <style>{`
          [data-aad] text {
            font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
          }
          [data-aad] .th { font-size: 14px; font-weight: 500; }
          [data-aad] .ts { font-size: 12px; font-weight: 400; }
          [data-aad] .arr { stroke: #9c9a92; stroke-width: 1.5; }
          [data-aad] .ah { stroke: #9c9a92; stroke-width: 1.5; stroke-linecap: round; stroke-linejoin: round; }

          [data-aad] .n-blue rect { fill: #E6F1FB; stroke: #185FA5; }
          [data-aad] .n-blue .th { fill: #0C447C; }
          [data-aad] .n-blue .ts { fill: #185FA5; }
          [data-aad] .n-purple rect { fill: #EEEDFE; stroke: #534AB7; }
          [data-aad] .n-purple .th { fill: #3C3489; }
          [data-aad] .n-purple .ts { fill: #534AB7; }
          [data-aad] .n-amber rect { fill: #FAEEDA; stroke: #854F0B; }
          [data-aad] .n-amber .th { fill: #633806; }
          [data-aad] .n-amber .ts { fill: #854F0B; }

          @media (prefers-color-scheme: dark) {
            [data-aad] .arr { stroke: #6b6b65; }
            [data-aad] .ah { stroke: #6b6b65; }
            [data-aad] .n-blue rect { fill: #042C53; stroke: #85B7EB; }
            [data-aad] .n-blue .th { fill: #B5D4F4; }
            [data-aad] .n-blue .ts { fill: #85B7EB; }
            [data-aad] .n-purple rect { fill: #26215C; stroke: #AFA9EC; }
            [data-aad] .n-purple .th { fill: #CECBF6; }
            [data-aad] .n-purple .ts { fill: #AFA9EC; }
            [data-aad] .n-amber rect { fill: #412402; stroke: #EF9F27; }
            [data-aad] .n-amber .th { fill: #FAC775; }
            [data-aad] .n-amber .ts { fill: #EF9F27; }
          }
        `}</style>
        <marker
          id="aad-arrow"
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

      {/* Input */}
      <g className="n-blue">
        <rect x="40" y="53" width="100" height="70" rx="8" strokeWidth="0.5" />
        <text className="th" x="90" y="80" textAnchor="middle" dominantBaseline="central">
          Input
        </text>
        <text className="ts" x="90" y="100" textAnchor="middle" dominantBaseline="central">
          1 x 28 x 28
        </text>
      </g>

      {/* Arrow: Input → Encoder */}
      <line x1="140" y1="88" x2="151" y2="88" className="arr" markerEnd="url(#aad-arrow)" />

      {/* Encoder */}
      <g className="n-purple">
        <rect x="157" y="60" width="108" height="56" rx="8" strokeWidth="0.5" />
        <text className="th" x="211" y="80" textAnchor="middle" dominantBaseline="central">
          Encoder
        </text>
        <text className="ts" x="211" y="98" textAnchor="middle" dominantBaseline="central">
          Conv + ReLU
        </text>
      </g>

      {/* Arrow: Encoder → Bottleneck */}
      <line x1="265" y1="88" x2="276" y2="88" className="arr" markerEnd="url(#aad-arrow)" />

      {/* Bottleneck */}
      <g className="n-amber">
        <rect x="282" y="66" width="114" height="44" rx="8" strokeWidth="0.5" />
        <text className="th" x="339" y="82" textAnchor="middle" dominantBaseline="central">
          Bottleneck
        </text>
        <text className="ts" x="339" y="98" textAnchor="middle" dominantBaseline="central">
          32-dim
        </text>
      </g>

      {/* Arrow: Bottleneck → Decoder */}
      <line x1="396" y1="88" x2="407" y2="88" className="arr" markerEnd="url(#aad-arrow)" />

      {/* Decoder */}
      <g className="n-purple">
        <rect x="413" y="60" width="108" height="56" rx="8" strokeWidth="0.5" />
        <text className="th" x="467" y="80" textAnchor="middle" dominantBaseline="central">
          Decoder
        </text>
        <text className="ts" x="467" y="98" textAnchor="middle" dominantBaseline="central">
          ConvT + ReLU
        </text>
      </g>

      {/* Arrow: Decoder → Output */}
      <line x1="521" y1="88" x2="532" y2="88" className="arr" markerEnd="url(#aad-arrow)" />

      {/* Output */}
      <g className="n-blue">
        <rect x="538" y="53" width="100" height="70" rx="8" strokeWidth="0.5" />
        <text className="th" x="588" y="80" textAnchor="middle" dominantBaseline="central">
          Output
        </text>
        <text className="ts" x="588" y="100" textAnchor="middle" dominantBaseline="central">
          1 x 28 x 28
        </text>
      </g>
    </svg>
  )
}

export function AdagnResidualBlockDiagram() {
  return (
    <svg
      data-arbd=""
      xmlns="http://www.w3.org/2000/svg"
      width="100%"
      viewBox="0 0 680 230"
      role="img"
      aria-label="Residual block with AdaGN timestep conditioning: input flows through Conv, AdaGN, and activation layers with a skip connection and timestep embedding injection"
    >
      <defs>
        <style>{`
          [data-arbd] text {
            font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
          }
          [data-arbd] .th { font-size: 14px; font-weight: 500; }
          [data-arbd] .ts { font-size: 12px; font-weight: 400; }
          [data-arbd] .arr { stroke: #9c9a92; stroke-width: 1.5; }
          [data-arbd] .ah { stroke: #9c9a92; stroke-width: 1.5; stroke-linecap: round; stroke-linejoin: round; }

          [data-arbd] .n-blue rect { fill: #E6F1FB; stroke: #185FA5; }
          [data-arbd] .n-blue .th { fill: #0C447C; }

          [data-arbd] .n-gray rect { fill: #F1EFE8; stroke: #5F5E5A; }
          [data-arbd] .n-gray .th { fill: #444441; }

          [data-arbd] .n-amber rect { fill: #FAEEDA; stroke: #854F0B; }
          [data-arbd] .n-amber .th { fill: #633806; }

          [data-arbd] .n-purple circle { fill: #EEEDFE; stroke: #534AB7; }
          [data-arbd] .n-purple .th { fill: #3C3489; }

          [data-arbd] .skip { stroke: #378ADD; stroke-width: 1.5; stroke-dasharray: 6 3; }
          [data-arbd] .skip-label { fill: #185FA5; }
          [data-arbd] .cond-line { stroke: #BA7517; stroke-width: 1.2; stroke-dasharray: 4 3; }
          [data-arbd] .cond-label { fill: #854F0B; }

          @media (prefers-color-scheme: dark) {
            [data-arbd] .arr { stroke: #6b6b65; }
            [data-arbd] .ah { stroke: #6b6b65; }

            [data-arbd] .n-blue rect { fill: #042C53; stroke: #85B7EB; }
            [data-arbd] .n-blue .th { fill: #B5D4F4; }

            [data-arbd] .n-gray rect { fill: #2C2C2A; stroke: #B4B2A9; }
            [data-arbd] .n-gray .th { fill: #D3D1C7; }

            [data-arbd] .n-amber rect { fill: #412402; stroke: #EF9F27; }
            [data-arbd] .n-amber .th { fill: #FAC775; }

            [data-arbd] .n-purple circle { fill: #26215C; stroke: #AFA9EC; }
            [data-arbd] .n-purple .th { fill: #CECBF6; }

            [data-arbd] .skip { stroke: #85B7EB; }
            [data-arbd] .skip-label { fill: #85B7EB; }
            [data-arbd] .cond-line { stroke: #EF9F27; }
            [data-arbd] .cond-label { fill: #EF9F27; }
          }
        `}</style>
        <marker
          id="arbd-arrow"
          viewBox="0 0 10 10"
          refX="8"
          refY="5"
          markerWidth="6"
          markerHeight="6"
          orient="auto-start-reverse"
        >
          <path d="M2 1L8 5L2 9" fill="none" className="ah" />
        </marker>
        <marker
          id="arbd-arrow-amber"
          viewBox="0 0 10 10"
          refX="8"
          refY="5"
          markerWidth="6"
          markerHeight="6"
          orient="auto-start-reverse"
        >
          <path d="M2 1L8 5L2 9" fill="none" className="cond-line" strokeDasharray="0" />
        </marker>
      </defs>

      {/* Node 1: Input x (blue) */}
      <g className="n-blue">
        <rect x="20" y="60" width="64" height="40" rx="8" strokeWidth="0.5" />
        <text className="th" x="52" y="80" textAnchor="middle" dominantBaseline="central">
          Input x
        </text>
      </g>

      {/* Arrow: Input x -> Conv1 */}
      <line x1="84" y1="80" x2="94" y2="80" className="arr" markerEnd="url(#arbd-arrow)" />

      {/* Node 2: Conv1 (gray) */}
      <g className="n-gray">
        <rect x="96" y="60" width="56" height="40" rx="8" strokeWidth="0.5" />
        <text className="th" x="124" y="80" textAnchor="middle" dominantBaseline="central">
          Conv1
        </text>
      </g>

      {/* Arrow: Conv1 -> AdaGN 1 */}
      <line x1="152" y1="80" x2="162" y2="80" className="arr" markerEnd="url(#arbd-arrow)" />

      {/* Node 3: AdaGN 1 (amber) */}
      <g className="n-amber">
        <rect x="164" y="60" width="60" height="40" rx="8" strokeWidth="0.5" />
        <text className="th" x="194" y="80" textAnchor="middle" dominantBaseline="central">
          AdaGN
        </text>
      </g>

      {/* Arrow: AdaGN 1 -> Activation 1 */}
      <line x1="224" y1="80" x2="234" y2="80" className="arr" markerEnd="url(#arbd-arrow)" />

      {/* Node 4: Activation 1 (gray) */}
      <g className="n-gray">
        <rect x="236" y="60" width="74" height="40" rx="8" strokeWidth="0.5" />
        <text className="th" x="273" y="80" textAnchor="middle" dominantBaseline="central">
          Activation
        </text>
      </g>

      {/* Arrow: Activation 1 -> Conv2 */}
      <line x1="310" y1="80" x2="320" y2="80" className="arr" markerEnd="url(#arbd-arrow)" />

      {/* Node 5: Conv2 (gray) */}
      <g className="n-gray">
        <rect x="322" y="60" width="56" height="40" rx="8" strokeWidth="0.5" />
        <text className="th" x="350" y="80" textAnchor="middle" dominantBaseline="central">
          Conv2
        </text>
      </g>

      {/* Arrow: Conv2 -> AdaGN 2 */}
      <line x1="378" y1="80" x2="388" y2="80" className="arr" markerEnd="url(#arbd-arrow)" />

      {/* Node 6: AdaGN 2 (amber) */}
      <g className="n-amber">
        <rect x="390" y="60" width="60" height="40" rx="8" strokeWidth="0.5" />
        <text className="th" x="420" y="80" textAnchor="middle" dominantBaseline="central">
          AdaGN
        </text>
      </g>

      {/* Arrow: AdaGN 2 -> Activation 2 */}
      <line x1="450" y1="80" x2="460" y2="80" className="arr" markerEnd="url(#arbd-arrow)" />

      {/* Node 7: Activation 2 (gray) */}
      <g className="n-gray">
        <rect x="462" y="60" width="74" height="40" rx="8" strokeWidth="0.5" />
        <text className="th" x="499" y="80" textAnchor="middle" dominantBaseline="central">
          Activation
        </text>
      </g>

      {/* Arrow: Activation 2 -> + add */}
      <line x1="536" y1="80" x2="544" y2="80" className="arr" markerEnd="url(#arbd-arrow)" />

      {/* Node 8: + add (purple circle) */}
      <g className="n-purple">
        <circle cx="564" cy="80" r="18" strokeWidth="0.5" />
        <text className="th" x="564" y="80" textAnchor="middle" dominantBaseline="central">
          +
        </text>
      </g>

      {/* Arrow: + add -> Output */}
      <line x1="582" y1="80" x2="596" y2="80" className="arr" markerEnd="url(#arbd-arrow)" />

      {/* Node 9: Output (blue) */}
      <g className="n-blue">
        <rect x="598" y="60" width="62" height="40" rx="8" strokeWidth="0.5" />
        <text className="th" x="629" y="80" textAnchor="middle" dominantBaseline="central">
          Output
        </text>
      </g>

      {/* Skip connection: Input x -> + add */}
      <path
        d="M 52 100 L 52 120 L 564 120 L 564 98"
        fill="none"
        className="skip"
        markerEnd="url(#arbd-arrow)"
      />
      <text
        className="ts skip-label"
        x="308"
        y="136"
        textAnchor="middle"
        dominantBaseline="central"
      >
        skip connection
      </text>

      {/* Node 10: Timestep embedding t_emb (amber) */}
      <g className="n-amber">
        <rect x="258" y="172" width="164" height="40" rx="8" strokeWidth="0.5" />
        <text className="th" x="340" y="192" textAnchor="middle" dominantBaseline="central">
          Timestep emb. t_emb
        </text>
      </g>

      {/* Conditioning arrow: t_emb -> AdaGN 1 */}
      <path
        d="M 290 172 L 290 152 L 194 152 L 194 100"
        fill="none"
        className="cond-line"
        markerEnd="url(#arbd-arrow-amber)"
      />
      <text
        className="ts cond-label"
        x="234"
        y="148"
        textAnchor="middle"
        dominantBaseline="central"
      >
        {'\u03B3\u2081(t), \u03B2\u2081(t)'}
      </text>

      {/* Conditioning arrow: t_emb -> AdaGN 2 */}
      <path
        d="M 390 172 L 390 152 L 420 152 L 420 100"
        fill="none"
        className="cond-line"
        markerEnd="url(#arbd-arrow-amber)"
      />
      <text
        className="ts cond-label"
        x="454"
        y="148"
        textAnchor="middle"
        dominantBaseline="central"
      >
        {'\u03B3\u2082(t), \u03B2\u2082(t)'}
      </text>
    </svg>
  )
}

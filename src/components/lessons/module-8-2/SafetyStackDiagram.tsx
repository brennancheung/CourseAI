export function SafetyStackDiagram() {
  return (
    <svg
      data-ssd=""
      xmlns="http://www.w3.org/2000/svg"
      width="100%"
      viewBox="0 0 680 576"
      role="img"
      aria-label="Safety stack: four defense layers from prompt filtering to model erasure"
    >
      <defs>
        <style>{`
          [data-ssd] text {
            font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
          }
          [data-ssd] .th { font-size: 14px; font-weight: 500; }
          [data-ssd] .ts { font-size: 12px; font-weight: 400; }
          [data-ssd] .arr { stroke: #9c9a92; stroke-width: 1.5; }
          [data-ssd] .lbl { fill: #9c9a92; font-size: 12px; font-weight: 400; }
          [data-ssd] .ah { stroke: #9c9a92; stroke-width: 1.5; stroke-linecap: round; stroke-linejoin: round; }

          [data-ssd] .n-blue rect { fill: #E6F1FB; stroke: #185FA5; }
          [data-ssd] .n-blue .th { fill: #0C447C; }
          [data-ssd] .n-blue .ts { fill: #185FA5; }
          [data-ssd] .n-amber rect { fill: #FAEEDA; stroke: #854F0B; }
          [data-ssd] .n-amber .th { fill: #633806; }
          [data-ssd] .n-amber .ts { fill: #854F0B; }
          [data-ssd] .n-purple rect { fill: #EEEDFE; stroke: #534AB7; }
          [data-ssd] .n-purple .th { fill: #3C3489; }
          [data-ssd] .n-purple .ts { fill: #534AB7; }
          [data-ssd] .n-teal rect { fill: #E1F5EE; stroke: #0F6E56; }
          [data-ssd] .n-teal .th { fill: #085041; }
          [data-ssd] .n-teal .ts { fill: #0F6E56; }
          [data-ssd] .n-coral rect { fill: #FAECE7; stroke: #993C1D; }
          [data-ssd] .n-coral .th { fill: #712B13; }
          [data-ssd] .n-coral .ts { fill: #993C1D; }
          [data-ssd] .n-green rect { fill: #EAF3DE; stroke: #3B6D11; }
          [data-ssd] .n-green .th { fill: #27500A; }

          @media (prefers-color-scheme: dark) {
            [data-ssd] .arr { stroke: #6b6b65; }
            [data-ssd] .lbl { fill: #6b6b65; }
            [data-ssd] .ah { stroke: #6b6b65; }
            [data-ssd] .n-blue rect { fill: #042C53; stroke: #85B7EB; }
            [data-ssd] .n-blue .th { fill: #B5D4F4; }
            [data-ssd] .n-blue .ts { fill: #85B7EB; }
            [data-ssd] .n-amber rect { fill: #412402; stroke: #EF9F27; }
            [data-ssd] .n-amber .th { fill: #FAC775; }
            [data-ssd] .n-amber .ts { fill: #EF9F27; }
            [data-ssd] .n-purple rect { fill: #26215C; stroke: #AFA9EC; }
            [data-ssd] .n-purple .th { fill: #CECBF6; }
            [data-ssd] .n-purple .ts { fill: #AFA9EC; }
            [data-ssd] .n-teal rect { fill: #04342C; stroke: #5DCAA5; }
            [data-ssd] .n-teal .th { fill: #9FE1CB; }
            [data-ssd] .n-teal .ts { fill: #5DCAA5; }
            [data-ssd] .n-coral rect { fill: #4A1B0C; stroke: #F0997B; }
            [data-ssd] .n-coral .th { fill: #F5C4B3; }
            [data-ssd] .n-coral .ts { fill: #F0997B; }
            [data-ssd] .n-green rect { fill: #173404; stroke: #97C459; }
            [data-ssd] .n-green .th { fill: #C0DD97; }
          }
        `}</style>
        <marker
          id="ssd-arrow"
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

      {/* User prompt */}
      <g className="n-blue">
        <rect x="270" y="40" width="140" height="44" rx="8" strokeWidth="0.5" />
        <text className="th" x="340" y="62" textAnchor="middle" dominantBaseline="central">
          User prompt
        </text>
      </g>

      {/* Arrow: User prompt -> L1 */}
      <line x1="340" y1="84" x2="340" y2="142" className="arr" markerEnd="url(#ssd-arrow)" />

      {/* Prompt filtering (Layer 1) */}
      <g className="n-amber">
        <rect x="252" y="144" width="176" height="56" rx="8" strokeWidth="0.5" />
        <text className="th" x="340" y="162" textAnchor="middle" dominantBaseline="central">
          Prompt filtering
        </text>
        <text className="ts" x="340" y="180" textAnchor="middle" dominantBaseline="central">
          Layer 1
        </text>
      </g>

      {/* Arrow: L1 -> Blocked */}
      <line x1="428" y1="172" x2="486" y2="172" className="arr" markerEnd="url(#ssd-arrow)" />
      <text className="lbl" x="457" y="158" textAnchor="middle" dominantBaseline="central">
        blocked
      </text>

      {/* Blocked 1 */}
      <g className="n-coral">
        <rect x="488" y="150" width="110" height="44" rx="8" strokeWidth="0.5" />
        <text className="th" x="543" y="172" textAnchor="middle" dominantBaseline="central">
          Blocked
        </text>
      </g>

      {/* Arrow: L1 -> L2 (pass) */}
      <line x1="340" y1="200" x2="340" y2="258" className="arr" markerEnd="url(#ssd-arrow)" />
      <text className="lbl" x="356" y="230" textAnchor="start" dominantBaseline="central">
        pass
      </text>

      {/* Model erasure (Layer 4) */}
      <g className="n-coral">
        <rect x="42" y="260" width="152" height="56" rx="8" strokeWidth="0.5" />
        <text className="th" x="118" y="278" textAnchor="middle" dominantBaseline="central">
          Model erasure
        </text>
        <text className="ts" x="118" y="296" textAnchor="middle" dominantBaseline="central">
          Layer 4
        </text>
      </g>

      {/* Dashed arrow: L4 -> L2 (modifies weights) */}
      <line
        x1="194"
        y1="288"
        x2="242"
        y2="288"
        className="arr"
        strokeDasharray="4 3"
        markerEnd="url(#ssd-arrow)"
      />
      <text className="lbl" x="218" y="250" textAnchor="middle" dominantBaseline="central">
        modifies weights
      </text>

      {/* Inference guidance (Layer 2) */}
      <g className="n-purple">
        <rect x="244" y="260" width="192" height="56" rx="8" strokeWidth="0.5" />
        <text className="th" x="340" y="278" textAnchor="middle" dominantBaseline="central">
          Inference guidance
        </text>
        <text className="ts" x="340" y="296" textAnchor="middle" dominantBaseline="central">
          Layer 2 · SLD
        </text>
      </g>

      {/* Arrow: L2 -> L3 */}
      <line x1="340" y1="316" x2="340" y2="374" className="arr" markerEnd="url(#ssd-arrow)" />

      {/* Output classifier (Layer 3) */}
      <g className="n-teal">
        <rect x="246" y="376" width="188" height="56" rx="8" strokeWidth="0.5" />
        <text className="th" x="340" y="394" textAnchor="middle" dominantBaseline="central">
          Output classifier
        </text>
        <text className="ts" x="340" y="412" textAnchor="middle" dominantBaseline="central">
          Layer 3
        </text>
      </g>

      {/* Arrow: L3 -> Blocked */}
      <line x1="434" y1="404" x2="486" y2="404" className="arr" markerEnd="url(#ssd-arrow)" />
      <text className="lbl" x="460" y="390" textAnchor="middle" dominantBaseline="central">
        blocked
      </text>

      {/* Blocked 2 */}
      <g className="n-coral">
        <rect x="488" y="382" width="110" height="44" rx="8" strokeWidth="0.5" />
        <text className="th" x="543" y="404" textAnchor="middle" dominantBaseline="central">
          Blocked
        </text>
      </g>

      {/* Arrow: L3 -> Delivered (pass) */}
      <line x1="340" y1="432" x2="340" y2="490" className="arr" markerEnd="url(#ssd-arrow)" />
      <text className="lbl" x="356" y="462" textAnchor="start" dominantBaseline="central">
        pass
      </text>

      {/* Delivered */}
      <g className="n-green">
        <rect x="278" y="492" width="124" height="44" rx="8" strokeWidth="0.5" />
        <text className="th" x="340" y="514" textAnchor="middle" dominantBaseline="central">
          Delivered
        </text>
      </g>
    </svg>
  )
}

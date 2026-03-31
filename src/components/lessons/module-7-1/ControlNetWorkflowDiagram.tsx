export function ControlNetWorkflowDiagram() {
  return (
    <svg
      data-cnw=""
      xmlns="http://www.w3.org/2000/svg"
      width="100%"
      viewBox="0 0 680 244"
      role="img"
      aria-label="ControlNet workflow: photograph flows through preprocessor, spatial map, ControlNet, and Stable Diffusion pipeline to generated image. Text prompt also feeds into the pipeline."
    >
      <defs>
        <style>{`
          [data-cnw] text {
            font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
          }
          [data-cnw] .th { font-size: 14px; font-weight: 500; }
          [data-cnw] .ts { font-size: 12px; font-weight: 400; }
          [data-cnw] .arr { stroke: #9c9a92; stroke-width: 1.5; }
          [data-cnw] .ah { stroke: #9c9a92; stroke-width: 1.5; stroke-linecap: round; stroke-linejoin: round; }

          [data-cnw] .n-blue rect { fill: #E6F1FB; stroke: #185FA5; }
          [data-cnw] .n-blue .th { fill: #0C447C; }
          [data-cnw] .n-purple rect { fill: #EEEDFE; stroke: #534AB7; }
          [data-cnw] .n-purple .th { fill: #3C3489; }
          [data-cnw] .n-amber rect { fill: #FAEEDA; stroke: #854F0B; }
          [data-cnw] .n-amber .th { fill: #633806; }
          [data-cnw] .n-green rect { fill: #EAF3DE; stroke: #3B6D11; }
          [data-cnw] .n-green .th { fill: #27500A; }
          [data-cnw] .n-gray rect { fill: #F1EFE8; stroke: #5F5E5A; }
          [data-cnw] .n-gray .th { fill: #444441; }
          [data-cnw] .n-coral rect { fill: #FAECE7; stroke: #993C1D; }
          [data-cnw] .n-coral .th { fill: #712B13; }

          @media (prefers-color-scheme: dark) {
            [data-cnw] .arr { stroke: #6b6b65; }
            [data-cnw] .ah { stroke: #6b6b65; }
            [data-cnw] .n-blue rect { fill: #042C53; stroke: #85B7EB; }
            [data-cnw] .n-blue .th { fill: #B5D4F4; }
            [data-cnw] .n-purple rect { fill: #26215C; stroke: #AFA9EC; }
            [data-cnw] .n-purple .th { fill: #CECBF6; }
            [data-cnw] .n-amber rect { fill: #412402; stroke: #EF9F27; }
            [data-cnw] .n-amber .th { fill: #FAC775; }
            [data-cnw] .n-green rect { fill: #173404; stroke: #97C459; }
            [data-cnw] .n-green .th { fill: #C0DD97; }
            [data-cnw] .n-gray rect { fill: #2C2C2A; stroke: #B4B2A9; }
            [data-cnw] .n-gray .th { fill: #D3D1C7; }
            [data-cnw] .n-coral rect { fill: #4A1B0C; stroke: #F0997B; }
            [data-cnw] .n-coral .th { fill: #F5C4B3; }
          }
        `}</style>
        <marker
          id="cnw-arrow"
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

      {/* Row 1: Photograph -> Preprocessor -> Spatial map -> ControlNet */}
      <g className="n-blue">
        <rect x="40" y="40" width="110" height="44" rx="8" strokeWidth="0.5" />
        <text className="th" x="95" y="62" textAnchor="middle" dominantBaseline="central">
          Photograph
        </text>
      </g>

      <g className="n-purple">
        <rect x="172" y="40" width="118" height="44" rx="8" strokeWidth="0.5" />
        <text className="th" x="231" y="62" textAnchor="middle" dominantBaseline="central">
          Preprocessor
        </text>
      </g>

      <g className="n-amber">
        <rect x="312" y="40" width="110" height="44" rx="8" strokeWidth="0.5" />
        <text className="th" x="367" y="62" textAnchor="middle" dominantBaseline="central">
          Spatial map
        </text>
      </g>

      <g className="n-green">
        <rect x="444" y="40" width="110" height="44" rx="8" strokeWidth="0.5" />
        <text className="th" x="499" y="62" textAnchor="middle" dominantBaseline="central">
          ControlNet
        </text>
      </g>

      {/* Row 1 arrows */}
      <line x1="150" y1="62" x2="162" y2="62" className="arr" markerEnd="url(#cnw-arrow)" />
      <line x1="290" y1="62" x2="302" y2="62" className="arr" markerEnd="url(#cnw-arrow)" />
      <line x1="422" y1="62" x2="434" y2="62" className="arr" markerEnd="url(#cnw-arrow)" />

      {/* Row 2: Text prompt -> SD pipeline -> Generated image */}
      <g className="n-blue">
        <rect x="40" y="160" width="120" height="44" rx="8" strokeWidth="0.5" />
        <text className="th" x="100" y="182" textAnchor="middle" dominantBaseline="central">
          Text prompt
        </text>
      </g>

      <g className="n-gray">
        <rect x="220" y="160" width="220" height="44" rx="8" strokeWidth="0.5" />
        <text className="th" x="330" y="182" textAnchor="middle" dominantBaseline="central">
          Stable Diffusion pipeline
        </text>
      </g>

      <g className="n-coral">
        <rect x="480" y="160" width="160" height="44" rx="8" strokeWidth="0.5" />
        <text className="th" x="560" y="182" textAnchor="middle" dominantBaseline="central">
          Generated image
        </text>
      </g>

      {/* ControlNet -> SD pipeline (L-bend: down then left then down) */}
      <path
        d="M499 84 L499 122 L330 122 L330 150"
        fill="none"
        className="arr"
        markerEnd="url(#cnw-arrow)"
      />

      {/* Text prompt -> SD pipeline */}
      <line x1="160" y1="182" x2="210" y2="182" className="arr" markerEnd="url(#cnw-arrow)" />

      {/* SD pipeline -> Generated image */}
      <line x1="440" y1="182" x2="470" y2="182" className="arr" markerEnd="url(#cnw-arrow)" />
    </svg>
  )
}

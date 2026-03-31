export function ImageGenerationPipelineDiagram() {
  return (
    <svg
      data-igpd=""
      xmlns="http://www.w3.org/2000/svg"
      width="100%"
      viewBox="0 0 680 716"
      role="img"
      aria-label="Image generation pipeline: user prompt through Gemini backbone, thinking tokens, autoregressive generation, GemPix decoder, to high-resolution image"
    >
      <defs>
        <style>{`
          [data-igpd] text {
            font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
          }
          [data-igpd] .th { font-size: 14px; font-weight: 500; }
          [data-igpd] .ts { font-size: 12px; font-weight: 400; }
          [data-igpd] .arr { stroke: #9c9a92; stroke-width: 1.5; }
          [data-igpd] .ah { stroke: #9c9a92; stroke-width: 1.5; stroke-linecap: round; stroke-linejoin: round; }

          [data-igpd] .n-blue rect { fill: #E6F1FB; stroke: #185FA5; }
          [data-igpd] .n-blue .th { fill: #0C447C; }
          [data-igpd] .n-blue .ts { fill: #185FA5; }
          [data-igpd] .n-amber rect { fill: #FAEEDA; stroke: #854F0B; }
          [data-igpd] .n-amber .th { fill: #633806; }
          [data-igpd] .n-amber .ts { fill: #854F0B; }
          [data-igpd] .n-purple rect { fill: #EEEDFE; stroke: #534AB7; }
          [data-igpd] .n-purple .th { fill: #3C3489; }
          [data-igpd] .n-purple .ts { fill: #534AB7; }
          [data-igpd] .n-green rect { fill: #EAF3DE; stroke: #3B6D11; }
          [data-igpd] .n-green .th { fill: #27500A; }
          [data-igpd] .n-green .ts { fill: #3B6D11; }

          @media (prefers-color-scheme: dark) {
            [data-igpd] .arr { stroke: #6b6b65; }
            [data-igpd] .ah { stroke: #6b6b65; }
            [data-igpd] .n-blue rect { fill: #042C53; stroke: #85B7EB; }
            [data-igpd] .n-blue .th { fill: #B5D4F4; }
            [data-igpd] .n-blue .ts { fill: #85B7EB; }
            [data-igpd] .n-amber rect { fill: #412402; stroke: #EF9F27; }
            [data-igpd] .n-amber .th { fill: #FAC775; }
            [data-igpd] .n-amber .ts { fill: #EF9F27; }
            [data-igpd] .n-purple rect { fill: #26215C; stroke: #AFA9EC; }
            [data-igpd] .n-purple .th { fill: #CECBF6; }
            [data-igpd] .n-purple .ts { fill: #AFA9EC; }
            [data-igpd] .n-green rect { fill: #173404; stroke: #97C459; }
            [data-igpd] .n-green .th { fill: #C0DD97; }
            [data-igpd] .n-green .ts { fill: #97C459; }
          }
        `}</style>
        <marker
          id="igpd-arrow"
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
        <rect x="220" y="40" width="240" height="56" rx="8" strokeWidth="0.5" />
        <text className="th" x="340" y="58" textAnchor="middle" dominantBaseline="central">
          User prompt
        </text>
        <text className="ts" x="340" y="76" textAnchor="middle" dominantBaseline="central">
          Text input
        </text>
      </g>

      <line x1="340" y1="96" x2="340" y2="156" className="arr" markerEnd="url(#igpd-arrow)" />

      {/* Gemini 3 Pro backbone */}
      <g className="n-blue">
        <rect x="220" y="156" width="240" height="56" rx="8" strokeWidth="0.5" />
        <text className="th" x="340" y="174" textAnchor="middle" dominantBaseline="central">
          Gemini 3 Pro backbone
        </text>
        <text className="ts" x="340" y="192" textAnchor="middle" dominantBaseline="central">
          Language model
        </text>
      </g>

      <line x1="340" y1="212" x2="340" y2="272" className="arr" markerEnd="url(#igpd-arrow)" />

      {/* Thinking tokens */}
      <g className="n-amber">
        <rect x="220" y="272" width="240" height="56" rx="8" strokeWidth="0.5" />
        <text className="th" x="340" y="290" textAnchor="middle" dominantBaseline="central">
          Thinking tokens
        </text>
        <text className="ts" x="340" y="308" textAnchor="middle" dominantBaseline="central">
          Composition planning
        </text>
      </g>

      <line x1="340" y1="328" x2="340" y2="388" className="arr" markerEnd="url(#igpd-arrow)" />

      {/* Image token generation */}
      <g className="n-purple">
        <rect x="220" y="388" width="240" height="56" rx="8" strokeWidth="0.5" />
        <text className="th" x="340" y="406" textAnchor="middle" dominantBaseline="central">
          Image token generation
        </text>
        <text className="ts" x="340" y="424" textAnchor="middle" dominantBaseline="central">
          ~1,120 tokens sequentially
        </text>
      </g>

      <line x1="340" y1="444" x2="340" y2="504" className="arr" markerEnd="url(#igpd-arrow)" />

      {/* GemPix 2 decoder */}
      <g className="n-green">
        <rect x="220" y="504" width="240" height="56" rx="8" strokeWidth="0.5" />
        <text className="th" x="340" y="522" textAnchor="middle" dominantBaseline="central">
          GemPix 2 decoder
        </text>
        <text className="ts" x="340" y="540" textAnchor="middle" dominantBaseline="central">
          Tokens to pixels
        </text>
      </g>

      <line x1="340" y1="560" x2="340" y2="620" className="arr" markerEnd="url(#igpd-arrow)" />

      {/* High-resolution image */}
      <g className="n-green">
        <rect x="220" y="620" width="240" height="56" rx="8" strokeWidth="0.5" />
        <text className="th" x="340" y="638" textAnchor="middle" dominantBaseline="central">
          High-resolution image
        </text>
        <text className="ts" x="340" y="656" textAnchor="middle" dominantBaseline="central">
          Final output
        </text>
      </g>
    </svg>
  )
}

export function SamArchitectureDiagram() {
  return (
    <svg
      data-sad=""
      xmlns="http://www.w3.org/2000/svg"
      width="100%"
      viewBox="0 0 680 360"
      role="img"
      aria-label="SAM architecture: image encoder and prompt encoder converge at mask decoder"
    >
      <defs>
        <style>{`
          [data-sad] text {
            font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
          }
          [data-sad] .th { font-size: 14px; font-weight: 500; }
          [data-sad] .ts { font-size: 12px; font-weight: 400; }
          [data-sad] .arr { stroke: #9c9a92; stroke-width: 1.5; }
          [data-sad] .ah { stroke: #9c9a92; stroke-width: 1.5; stroke-linecap: round; stroke-linejoin: round; }

          [data-sad] .n-blue rect { fill: #E6F1FB; stroke: #185FA5; }
          [data-sad] .n-blue .th { fill: #0C447C; }
          [data-sad] .n-blue .ts { fill: #185FA5; }
          [data-sad] .n-amber rect { fill: #FAEEDA; stroke: #854F0B; }
          [data-sad] .n-amber .th { fill: #633806; }
          [data-sad] .n-amber .ts { fill: #854F0B; }
          [data-sad] .n-teal rect { fill: #E1F5EE; stroke: #0F6E56; }
          [data-sad] .n-teal .th { fill: #085041; }
          [data-sad] .n-teal .ts { fill: #0F6E56; }

          @media (prefers-color-scheme: dark) {
            [data-sad] .arr { stroke: #6b6b65; }
            [data-sad] .ah { stroke: #6b6b65; }
            [data-sad] .n-blue rect { fill: #042C53; stroke: #85B7EB; }
            [data-sad] .n-blue .th { fill: #B5D4F4; }
            [data-sad] .n-blue .ts { fill: #85B7EB; }
            [data-sad] .n-amber rect { fill: #412402; stroke: #EF9F27; }
            [data-sad] .n-amber .th { fill: #FAC775; }
            [data-sad] .n-amber .ts { fill: #EF9F27; }
            [data-sad] .n-teal rect { fill: #04342C; stroke: #5DCAA5; }
            [data-sad] .n-teal .th { fill: #9FE1CB; }
            [data-sad] .n-teal .ts { fill: #5DCAA5; }
          }
        `}</style>
        <marker
          id="sad-arrow"
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

      {/* Row 1 (blue): Image -> ViT-H Image Encoder -> Image Embedding */}

      <g className="n-blue">
        <rect x="40" y="40" width="112" height="56" rx="8" strokeWidth="0.5" />
        <text className="th" x="96" y="58" textAnchor="middle" dominantBaseline="central">
          Image
        </text>
        <text className="ts" x="96" y="76" textAnchor="middle" dominantBaseline="central">
          1024x1024x3
        </text>
      </g>

      <line x1="152" y1="68" x2="162" y2="68" className="arr" markerEnd="url(#sad-arrow)" />

      <g className="n-blue">
        <rect x="168" y="40" width="208" height="56" rx="8" strokeWidth="0.5" />
        <text className="th" x="272" y="58" textAnchor="middle" dominantBaseline="central">
          ViT-H Image Encoder
        </text>
        <text className="ts" x="272" y="76" textAnchor="middle" dominantBaseline="central">
          ~150ms, ONCE
        </text>
      </g>

      <line x1="376" y1="68" x2="386" y2="68" className="arr" markerEnd="url(#sad-arrow)" />

      <g className="n-blue">
        <rect x="392" y="40" width="164" height="56" rx="8" strokeWidth="0.5" />
        <text className="th" x="474" y="58" textAnchor="middle" dominantBaseline="central">
          Image Embedding
        </text>
        <text className="ts" x="474" y="76" textAnchor="middle" dominantBaseline="central">
          64x64x256
        </text>
      </g>

      {/* Row 2 (amber): User Prompt -> Prompt Encoder -> Prompt Tokens */}

      <g className="n-amber">
        <rect x="40" y="156" width="148" height="56" rx="8" strokeWidth="0.5" />
        <text className="th" x="114" y="174" textAnchor="middle" dominantBaseline="central">
          User Prompt
        </text>
        <text className="ts" x="114" y="192" textAnchor="middle" dominantBaseline="central">
          point/box/text
        </text>
      </g>

      <line x1="188" y1="184" x2="198" y2="184" className="arr" markerEnd="url(#sad-arrow)" />

      <g className="n-amber">
        <rect x="204" y="156" width="156" height="56" rx="8" strokeWidth="0.5" />
        <text className="th" x="282" y="174" textAnchor="middle" dominantBaseline="central">
          Prompt Encoder
        </text>
        <text className="ts" x="282" y="192" textAnchor="middle" dominantBaseline="central">
          {'<1ms'}
        </text>
      </g>

      <line x1="360" y1="184" x2="386" y2="184" className="arr" markerEnd="url(#sad-arrow)" />

      <g className="n-amber">
        <rect x="392" y="156" width="164" height="56" rx="8" strokeWidth="0.5" />
        <text className="th" x="474" y="174" textAnchor="middle" dominantBaseline="central">
          Prompt Tokens
        </text>
        <text className="ts" x="474" y="192" textAnchor="middle" dominantBaseline="central">
          1-4 tokens
        </text>
      </g>

      {/* Convergence: route right of column-3 boxes, down, left to Mask Decoder */}

      <path d="M 556 68 L 580 68 L 580 218" fill="none" className="arr" />
      <path d="M 556 184 L 580 184 L 580 218" fill="none" className="arr" />
      <path
        d="M 580 218 L 340 218 L 340 252"
        fill="none"
        className="arr"
        markerEnd="url(#sad-arrow)"
      />

      {/* Row 3 (teal): Mask Decoder -> 3 Candidate Masks */}

      <g className="n-teal">
        <rect x="243" y="258" width="194" height="56" rx="8" strokeWidth="0.5" />
        <text className="th" x="340" y="276" textAnchor="middle" dominantBaseline="central">
          Mask Decoder
        </text>
        <text className="ts" x="340" y="294" textAnchor="middle" dominantBaseline="central">
          ~50ms, PER PROMPT
        </text>
      </g>

      <line x1="437" y1="286" x2="453" y2="286" className="arr" markerEnd="url(#sad-arrow)" />

      <g className="n-teal">
        <rect x="459" y="258" width="180" height="56" rx="8" strokeWidth="0.5" />
        <text className="th" x="549" y="276" textAnchor="middle" dominantBaseline="central">
          3 Candidate Masks
        </text>
        <text className="ts" x="549" y="294" textAnchor="middle" dominantBaseline="central">
          + confidence scores
        </text>
      </g>
    </svg>
  )
}

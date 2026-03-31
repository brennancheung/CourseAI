export function UnetArchitectureDiagram() {
  return (
    <svg
      data-unet=""
      xmlns="http://www.w3.org/2000/svg"
      width="100%"
      viewBox="0 0 680 500"
      role="img"
      aria-label="U-Net encoder-decoder architecture with skip connections showing the U-shaped data flow"
    >
      <defs>
        <style>{`
          [data-unet] text {
            font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
          }
          [data-unet] .th { font-size: 14px; font-weight: 500; }
          [data-unet] .ts { font-size: 12px; font-weight: 400; }
          [data-unet] .arr { stroke: #9c9a92; stroke-width: 1.5; }
          [data-unet] .ah { stroke: #9c9a92; stroke-width: 1.5; stroke-linecap: round; stroke-linejoin: round; }
          [data-unet] .skip-line { stroke: #9c9a92; stroke-width: 1.2; stroke-dasharray: 6 4; }
          [data-unet] .label { font-size: 12px; font-weight: 400; fill: #9c9a92; }
          [data-unet] .group-label { font-size: 12px; font-weight: 400; fill: #9c9a92; opacity: 0.6; }

          [data-unet] .n-teal rect { fill: #E1F5EE; stroke: #0F6E56; }
          [data-unet] .n-teal .th { fill: #085041; }
          [data-unet] .n-teal .ts { fill: #0F6E56; }

          [data-unet] .n-amber rect { fill: #FAEEDA; stroke: #854F0B; }
          [data-unet] .n-amber .th { fill: #633806; }
          [data-unet] .n-amber .ts { fill: #854F0B; }

          [data-unet] .n-purple rect { fill: #EEEDFE; stroke: #534AB7; }
          [data-unet] .n-purple .th { fill: #3C3489; }
          [data-unet] .n-purple .ts { fill: #534AB7; }

          [data-unet] .n-blue rect { fill: #E6F1FB; stroke: #185FA5; }
          [data-unet] .n-blue .th { fill: #0C447C; }
          [data-unet] .n-blue .ts { fill: #185FA5; }

          @media (prefers-color-scheme: dark) {
            [data-unet] .arr { stroke: #6b6b65; }
            [data-unet] .ah { stroke: #6b6b65; }
            [data-unet] .skip-line { stroke: #6b6b65; }
            [data-unet] .label { fill: #6b6b65; }
            [data-unet] .group-label { fill: #6b6b65; }

            [data-unet] .n-teal rect { fill: #04342C; stroke: #5DCAA5; }
            [data-unet] .n-teal .th { fill: #9FE1CB; }
            [data-unet] .n-teal .ts { fill: #5DCAA5; }

            [data-unet] .n-amber rect { fill: #412402; stroke: #EF9F27; }
            [data-unet] .n-amber .th { fill: #FAC775; }
            [data-unet] .n-amber .ts { fill: #EF9F27; }

            [data-unet] .n-purple rect { fill: #26215C; stroke: #AFA9EC; }
            [data-unet] .n-purple .th { fill: #CECBF6; }
            [data-unet] .n-purple .ts { fill: #AFA9EC; }

            [data-unet] .n-blue rect { fill: #042C53; stroke: #85B7EB; }
            [data-unet] .n-blue .th { fill: #B5D4F4; }
            [data-unet] .n-blue .ts { fill: #85B7EB; }
          }
        `}</style>
        <marker
          id="unet-arrow"
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
          id="unet-arrow-skip"
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

      {/* ===== Input (blue, top-left) ===== */}
      <g className="n-blue">
        <rect x="70" y="40" width="200" height="56" rx="8" strokeWidth="0.5" />
        <text className="th" x="170" y="60" textAnchor="middle" dominantBaseline="central">
          64&times;64&times;3
        </text>
        <text className="ts" x="170" y="80" textAnchor="middle" dominantBaseline="central">
          Noisy image
        </text>
      </g>

      {/* ===== Encoder column (left, teal) ===== */}
      <text className="group-label" x="170" y="114" textAnchor="middle" dominantBaseline="auto">
        Encoder
      </text>

      <g className="n-teal">
        <rect x="70" y="120" width="200" height="56" rx="8" strokeWidth="0.5" />
        <text className="th" x="170" y="140" textAnchor="middle" dominantBaseline="central">
          64&times;64&times;64
        </text>
        <text className="ts" x="170" y="160" textAnchor="middle" dominantBaseline="central">
          Edges, textures
        </text>
      </g>

      <g className="n-teal">
        <rect x="70" y="216" width="200" height="56" rx="8" strokeWidth="0.5" />
        <text className="th" x="170" y="236" textAnchor="middle" dominantBaseline="central">
          32&times;32&times;128
        </text>
        <text className="ts" x="170" y="256" textAnchor="middle" dominantBaseline="central">
          Shapes, parts
        </text>
      </g>

      <g className="n-teal">
        <rect x="70" y="312" width="200" height="56" rx="8" strokeWidth="0.5" />
        <text className="th" x="170" y="332" textAnchor="middle" dominantBaseline="central">
          16&times;16&times;256
        </text>
        <text className="ts" x="170" y="352" textAnchor="middle" dominantBaseline="central">
          Object structure
        </text>
      </g>

      {/* ===== Bottleneck (center, amber) ===== */}
      <text className="group-label" x="340" y="398" textAnchor="middle" dominantBaseline="auto">
        Bottleneck
      </text>

      <g className="n-amber">
        <rect x="240" y="404" width="200" height="56" rx="8" strokeWidth="0.5" />
        <text className="th" x="340" y="424" textAnchor="middle" dominantBaseline="central">
          8&times;8&times;512
        </text>
        <text className="ts" x="340" y="444" textAnchor="middle" dominantBaseline="central">
          Global context
        </text>
      </g>

      {/* ===== Decoder column (right, purple) ===== */}
      <text className="group-label" x="510" y="114" textAnchor="middle" dominantBaseline="auto">
        Decoder
      </text>

      <g className="n-purple">
        <rect x="410" y="312" width="200" height="56" rx="8" strokeWidth="0.5" />
        <text className="th" x="510" y="332" textAnchor="middle" dominantBaseline="central">
          16&times;16&times;256
        </text>
        <text className="ts" x="510" y="352" textAnchor="middle" dominantBaseline="central">
          + encoder features
        </text>
      </g>

      <g className="n-purple">
        <rect x="410" y="216" width="200" height="56" rx="8" strokeWidth="0.5" />
        <text className="th" x="510" y="236" textAnchor="middle" dominantBaseline="central">
          32&times;32&times;128
        </text>
        <text className="ts" x="510" y="256" textAnchor="middle" dominantBaseline="central">
          + encoder features
        </text>
      </g>

      <g className="n-purple">
        <rect x="410" y="120" width="200" height="56" rx="8" strokeWidth="0.5" />
        <text className="th" x="510" y="140" textAnchor="middle" dominantBaseline="central">
          64&times;64&times;64
        </text>
        <text className="ts" x="510" y="160" textAnchor="middle" dominantBaseline="central">
          + encoder features
        </text>
      </g>

      {/* ===== Output (blue, top-right) ===== */}
      <g className="n-blue">
        <rect x="410" y="40" width="200" height="56" rx="8" strokeWidth="0.5" />
        <text className="th" x="510" y="60" textAnchor="middle" dominantBaseline="central">
          64&times;64&times;3
        </text>
        <text className="ts" x="510" y="80" textAnchor="middle" dominantBaseline="central">
          Predicted noise
        </text>
      </g>

      {/* ===== Encoder arrows (downward) ===== */}
      <line x1="170" y1="96" x2="170" y2="120" className="arr" markerEnd="url(#unet-arrow)" />

      <line x1="170" y1="176" x2="170" y2="216" className="arr" markerEnd="url(#unet-arrow)" />
      <text className="label" x="158" y="198" textAnchor="end" dominantBaseline="central">
        downsample
      </text>

      <line x1="170" y1="272" x2="170" y2="312" className="arr" markerEnd="url(#unet-arrow)" />
      <text className="label" x="158" y="294" textAnchor="end" dominantBaseline="central">
        downsample
      </text>

      {/* Enc2 to bottleneck (L-bend) */}
      <path
        d="M170,368 L170,386 L340,386 L340,404"
        fill="none"
        className="arr"
        markerEnd="url(#unet-arrow)"
      />
      <text className="label" x="248" y="381" textAnchor="middle" dominantBaseline="auto">
        downsample
      </text>

      {/* ===== Decoder arrows (upward) ===== */}
      {/* Bottleneck to Dec2 (L-bend) */}
      <path
        d="M340,404 L340,386 L510,386 L510,368"
        fill="none"
        className="arr"
        markerEnd="url(#unet-arrow)"
      />
      <text className="label" x="432" y="381" textAnchor="middle" dominantBaseline="auto">
        upsample
      </text>

      <line x1="510" y1="312" x2="510" y2="272" className="arr" markerEnd="url(#unet-arrow)" />
      <text className="label" x="522" y="294" dominantBaseline="central">
        upsample
      </text>

      <line x1="510" y1="216" x2="510" y2="176" className="arr" markerEnd="url(#unet-arrow)" />
      <text className="label" x="522" y="198" dominantBaseline="central">
        upsample
      </text>

      <line x1="510" y1="120" x2="510" y2="96" className="arr" markerEnd="url(#unet-arrow)" />

      {/* ===== Skip connections (horizontal dashed) ===== */}
      <line
        x1="270" y1="148" x2="410" y2="148"
        className="skip-line"
        markerEnd="url(#unet-arrow-skip)"
      />
      <text className="label" x="340" y="138" textAnchor="middle" dominantBaseline="auto">
        skip: concat
      </text>

      <line
        x1="270" y1="244" x2="410" y2="244"
        className="skip-line"
        markerEnd="url(#unet-arrow-skip)"
      />
      <text className="label" x="340" y="234" textAnchor="middle" dominantBaseline="auto">
        skip: concat
      </text>

      <line
        x1="270" y1="340" x2="410" y2="340"
        className="skip-line"
        markerEnd="url(#unet-arrow-skip)"
      />
      <text className="label" x="340" y="330" textAnchor="middle" dominantBaseline="auto">
        skip: concat
      </text>
    </svg>
  )
}

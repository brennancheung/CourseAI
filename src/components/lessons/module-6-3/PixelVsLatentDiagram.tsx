export function PixelVsLatentDiagram() {
  return (
    <svg
      data-pvld=""
      xmlns="http://www.w3.org/2000/svg"
      width="100%"
      viewBox="0 0 680 396"
      role="img"
      aria-label="Comparison of pixel-space diffusion (3 steps) versus latent-space diffusion (7 steps with VAE encode and decode bookending the same U-Net denoise loop)"
    >
      <defs>
        <style>{`
          [data-pvld] text {
            font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
          }
          [data-pvld] .th { font-size: 14px; font-weight: 500; }
          [data-pvld] .ts { font-size: 12px; font-weight: 400; }
          [data-pvld] .arr { stroke: #9c9a92; stroke-width: 1.5; }
          [data-pvld] .ah { stroke: #9c9a92; stroke-width: 1.5; stroke-linecap: round; stroke-linejoin: round; }

          [data-pvld] .n-amber rect { fill: #FAEEDA; stroke: #854F0B; }
          [data-pvld] .n-amber .th { fill: #633806; }
          [data-pvld] .n-amber .ts { fill: #854F0B; }

          [data-pvld] .n-purple rect { fill: #EEEDFE; stroke: #534AB7; }
          [data-pvld] .n-purple .th { fill: #3C3489; }
          [data-pvld] .n-purple .ts { fill: #534AB7; }

          [data-pvld] .n-green rect { fill: #EAF3DE; stroke: #3B6D11; }
          [data-pvld] .n-green .th { fill: #27500A; }
          [data-pvld] .n-green .ts { fill: #3B6D11; }

          [data-pvld] .label-amber { fill: #854F0B; }
          [data-pvld] .label-purple { fill: #534AB7; }

          @media (prefers-color-scheme: dark) {
            [data-pvld] .arr { stroke: #6b6b65; }
            [data-pvld] .ah { stroke: #6b6b65; }

            [data-pvld] .n-amber rect { fill: #412402; stroke: #EF9F27; }
            [data-pvld] .n-amber .th { fill: #FAC775; }
            [data-pvld] .n-amber .ts { fill: #EF9F27; }

            [data-pvld] .n-purple rect { fill: #26215C; stroke: #AFA9EC; }
            [data-pvld] .n-purple .th { fill: #CECBF6; }
            [data-pvld] .n-purple .ts { fill: #AFA9EC; }

            [data-pvld] .n-green rect { fill: #173404; stroke: #97C459; }
            [data-pvld] .n-green .th { fill: #C0DD97; }
            [data-pvld] .n-green .ts { fill: #97C459; }

            [data-pvld] .label-amber { fill: #EF9F27; }
            [data-pvld] .label-purple { fill: #AFA9EC; }
          }
        `}</style>
        <marker
          id="pvld-arrow"
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

      {/* Row 1 label */}
      <text
        className="th label-amber"
        x="340"
        y="30"
        textAnchor="middle"
        dominantBaseline="central"
      >
        Pixel-space diffusion
      </text>

      {/* B1: Image 512x512x3 */}
      <g className="n-amber">
        <rect x="82" y="48" width="112" height="56" rx="8" strokeWidth="0.5" />
        <text className="th" x="138" y="68" textAnchor="middle" dominantBaseline="central">
          Image
        </text>
        <text className="ts" x="138" y="88" textAnchor="middle" dominantBaseline="central">
          512×512×3
        </text>
      </g>

      {/* Arrow B1 → B2 */}
      <line
        x1="194"
        y1="76"
        x2="224"
        y2="76"
        className="arr"
        markerEnd="url(#pvld-arrow)"
      />

      {/* B2: U-Net denoise ×50 steps */}
      <g className="n-amber">
        <rect x="234" y="48" width="156" height="56" rx="8" strokeWidth="0.5" />
        <text className="th" x="312" y="68" textAnchor="middle" dominantBaseline="central">
          U-Net denoise
        </text>
        <text className="ts" x="312" y="88" textAnchor="middle" dominantBaseline="central">
          ×50 steps
        </text>
      </g>

      {/* Arrow B2 → B3 */}
      <line
        x1="390"
        y1="76"
        x2="420"
        y2="76"
        className="arr"
        markerEnd="url(#pvld-arrow)"
      />

      {/* B3: Generated image 512×512×3 */}
      <g className="n-amber">
        <rect x="430" y="48" width="168" height="56" rx="8" strokeWidth="0.5" />
        <text className="th" x="514" y="68" textAnchor="middle" dominantBaseline="central">
          Generated image
        </text>
        <text className="ts" x="514" y="88" textAnchor="middle" dominantBaseline="central">
          512×512×3
        </text>
      </g>

      {/* Row 2 label */}
      <text
        className="th label-purple"
        x="340"
        y="144"
        textAnchor="middle"
        dominantBaseline="central"
      >
        Latent-space diffusion
      </text>

      {/* B4: Image 512×512×3 */}
      <g className="n-purple">
        <rect x="40" y="162" width="104" height="56" rx="8" strokeWidth="0.5" />
        <text className="th" x="92" y="182" textAnchor="middle" dominantBaseline="central">
          Image
        </text>
        <text className="ts" x="92" y="202" textAnchor="middle" dominantBaseline="central">
          512×512×3
        </text>
      </g>

      {/* Arrow B4 → B5 */}
      <line
        x1="144"
        y1="190"
        x2="170"
        y2="190"
        className="arr"
        markerEnd="url(#pvld-arrow)"
      />

      {/* B5: VAE encode */}
      <g className="n-green">
        <rect x="180" y="162" width="112" height="56" rx="8" strokeWidth="0.5" />
        <text className="th" x="236" y="182" textAnchor="middle" dominantBaseline="central">
          VAE encode
        </text>
        <text className="ts" x="236" y="202" textAnchor="middle" dominantBaseline="central">
          compress
        </text>
      </g>

      {/* Arrow B5 → B6 */}
      <line
        x1="292"
        y1="190"
        x2="318"
        y2="190"
        className="arr"
        markerEnd="url(#pvld-arrow)"
      />

      {/* B6: Latent 64×64×4 */}
      <g className="n-purple">
        <rect x="328" y="162" width="104" height="56" rx="8" strokeWidth="0.5" />
        <text className="th" x="380" y="182" textAnchor="middle" dominantBaseline="central">
          Latent
        </text>
        <text className="ts" x="380" y="202" textAnchor="middle" dominantBaseline="central">
          64×64×4
        </text>
      </g>

      {/* Arrow B6 → B7 */}
      <line
        x1="432"
        y1="190"
        x2="462"
        y2="190"
        className="arr"
        markerEnd="url(#pvld-arrow)"
      />

      {/* B7: U-Net denoise ×50 steps */}
      <g className="n-purple">
        <rect x="472" y="162" width="168" height="56" rx="8" strokeWidth="0.5" />
        <text className="th" x="556" y="182" textAnchor="middle" dominantBaseline="central">
          U-Net denoise
        </text>
        <text className="ts" x="556" y="202" textAnchor="middle" dominantBaseline="central">
          ×50 steps
        </text>
      </g>

      {/* L-shaped arrow B7 → B8 */}
      <path
        d="M 556 218 L 556 248 L 252 248 L 252 278"
        fill="none"
        className="arr"
        markerEnd="url(#pvld-arrow)"
      />

      {/* B8: Denoised latent 64×64×4 */}
      <g className="n-purple">
        <rect x="176" y="288" width="152" height="56" rx="8" strokeWidth="0.5" />
        <text className="th" x="252" y="308" textAnchor="middle" dominantBaseline="central">
          Denoised latent
        </text>
        <text className="ts" x="252" y="328" textAnchor="middle" dominantBaseline="central">
          64×64×4
        </text>
      </g>

      {/* Arrow B8 → B9 */}
      <line
        x1="328"
        y1="316"
        x2="358"
        y2="316"
        className="arr"
        markerEnd="url(#pvld-arrow)"
      />

      {/* B9: VAE decode */}
      <g className="n-green">
        <rect x="368" y="288" width="112" height="56" rx="8" strokeWidth="0.5" />
        <text className="th" x="424" y="308" textAnchor="middle" dominantBaseline="central">
          VAE decode
        </text>
        <text className="ts" x="424" y="328" textAnchor="middle" dominantBaseline="central">
          decompress
        </text>
      </g>

      {/* Arrow B9 → B10 */}
      <line
        x1="480"
        y1="316"
        x2="510"
        y2="316"
        className="arr"
        markerEnd="url(#pvld-arrow)"
      />

      {/* B10: Generated 512×512×3 */}
      <g className="n-purple">
        <rect x="520" y="288" width="120" height="56" rx="8" strokeWidth="0.5" />
        <text className="th" x="580" y="308" textAnchor="middle" dominantBaseline="central">
          Generated
        </text>
        <text className="ts" x="580" y="328" textAnchor="middle" dominantBaseline="central">
          512×512×3
        </text>
      </g>
    </svg>
  )
}

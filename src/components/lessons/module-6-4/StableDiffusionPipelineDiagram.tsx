export function StableDiffusionPipelineDiagram() {
  return (
    <svg
      data-sdp=""
      xmlns="http://www.w3.org/2000/svg"
      width="100%"
      viewBox="0 0 680 524"
      role="img"
      aria-label="Stable Diffusion pipeline: text prompt flows through CLIP encoder to produce text embeddings, which guide a 50-step U-Net denoising loop from noise to clean latent, then VAE decoder produces the final image"
    >
      <defs>
        <style>{`
          [data-sdp] text {
            font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
          }
          [data-sdp] .th { font-size: 14px; font-weight: 500; }
          [data-sdp] .ts { font-size: 12px; font-weight: 400; }
          [data-sdp] .arr { stroke: #9c9a92; stroke-width: 1.5; }
          [data-sdp] .lbl { fill: #9c9a92; font-size: 12px; font-weight: 400; }

          [data-sdp] .grp-purple { fill: #EEEDFE; stroke: #534AB7; }
          [data-sdp] .grp-purple-label { fill: #534AB7; font-size: 14px; font-weight: 500; }
          [data-sdp] .grp-blue { fill: #E6F1FB; stroke: #185FA5; }
          [data-sdp] .grp-blue-label { fill: #185FA5; font-size: 14px; font-weight: 500; }
          [data-sdp] .grp-green { fill: #EAF3DE; stroke: #3B6D11; }
          [data-sdp] .grp-green-label { fill: #3B6D11; font-size: 14px; font-weight: 500; }

          [data-sdp] .n-purple rect { fill: #EEEDFE; stroke: #534AB7; }
          [data-sdp] .n-purple .th { fill: #3C3489; }
          [data-sdp] .n-purple .ts { fill: #534AB7; }

          [data-sdp] .n-blue rect { fill: #E6F1FB; stroke: #185FA5; }
          [data-sdp] .n-blue .th { fill: #0C447C; }
          [data-sdp] .n-blue .ts { fill: #185FA5; }

          [data-sdp] .n-green rect { fill: #EAF3DE; stroke: #3B6D11; }
          [data-sdp] .n-green .th { fill: #27500A; }
          [data-sdp] .n-green .ts { fill: #3B6D11; }

          [data-sdp] .conn-purple { stroke: #7F77DD; stroke-width: 1.5; }
          [data-sdp] .conn-purple-label { fill: #7F77DD; font-size: 12px; font-weight: 400; }
          [data-sdp] .conn-blue { stroke: #378ADD; stroke-width: 1.5; }
          [data-sdp] .conn-blue-label { fill: #378ADD; font-size: 12px; font-weight: 400; }

          @media (prefers-color-scheme: dark) {
            [data-sdp] .arr { stroke: #6b6b65; }
            [data-sdp] .lbl { fill: #6b6b65; }

            [data-sdp] .grp-purple { fill: #26215C; stroke: #AFA9EC; }
            [data-sdp] .grp-purple-label { fill: #AFA9EC; }
            [data-sdp] .grp-blue { fill: #042C53; stroke: #85B7EB; }
            [data-sdp] .grp-blue-label { fill: #85B7EB; }
            [data-sdp] .grp-green { fill: #173404; stroke: #97C459; }
            [data-sdp] .grp-green-label { fill: #97C459; }

            [data-sdp] .n-purple rect { fill: #26215C; stroke: #AFA9EC; }
            [data-sdp] .n-purple .th { fill: #CECBF6; }
            [data-sdp] .n-purple .ts { fill: #AFA9EC; }

            [data-sdp] .n-blue rect { fill: #042C53; stroke: #85B7EB; }
            [data-sdp] .n-blue .th { fill: #B5D4F4; }
            [data-sdp] .n-blue .ts { fill: #85B7EB; }

            [data-sdp] .n-green rect { fill: #173404; stroke: #97C459; }
            [data-sdp] .n-green .th { fill: #C0DD97; }
            [data-sdp] .n-green .ts { fill: #97C459; }

            [data-sdp] .conn-purple { stroke: #AFA9EC; }
            [data-sdp] .conn-purple-label { fill: #AFA9EC; }
            [data-sdp] .conn-blue { stroke: #85B7EB; }
            [data-sdp] .conn-blue-label { fill: #85B7EB; }
          }
        `}</style>
        <marker
          id="sdp-arrow"
          viewBox="0 0 10 10"
          refX="8"
          refY="5"
          markerWidth="6"
          markerHeight="6"
          orient="auto-start-reverse"
        >
          <path
            d="M2 1L8 5L2 9"
            fill="none"
            stroke="context-stroke"
            strokeWidth="1.5"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </marker>
      </defs>

      {/* ========== GROUP 1: CLIP (purple) ========== */}
      <rect
        className="grp-purple"
        x="40" y="30" width="600" height="120" rx="16"
        strokeWidth="0.5" opacity="0.35"
      />
      <text
        className="grp-purple-label"
        x="340" y="52"
        textAnchor="middle" dominantBaseline="central"
      >
        CLIP text encoder (frozen, ~123M params)
      </text>

      {/* Text prompt */}
      <g className="n-purple">
        <rect x="60" y="72" width="130" height="56" rx="8" strokeWidth="0.5" />
        <text className="th" x="125" y="92" textAnchor="middle" dominantBaseline="central">
          Text prompt
        </text>
        <text className="ts" x="125" y="112" textAnchor="middle" dominantBaseline="central">
          {'"a cat on mars"'}
        </text>
      </g>

      {/* Arrow: Text prompt -> Tokenizer */}
      <line
        x1="190" y1="100" x2="228" y2="100"
        className="arr" markerEnd="url(#sdp-arrow)"
      />

      {/* Tokenizer */}
      <g className="n-purple">
        <rect x="234" y="78" width="120" height="44" rx="8" strokeWidth="0.5" />
        <text className="th" x="294" y="100" textAnchor="middle" dominantBaseline="central">
          Tokenizer
        </text>
      </g>

      {/* Arrow: Tokenizer -> Text encoder, label above */}
      <line
        x1="354" y1="100" x2="402" y2="100"
        className="arr" markerEnd="url(#sdp-arrow)"
      />
      <text className="lbl" x="378" y="88" textAnchor="middle" dominantBaseline="central">
        [77] int IDs
      </text>

      {/* Text encoder */}
      <g className="n-purple">
        <rect x="408" y="72" width="148" height="56" rx="8" strokeWidth="0.5" />
        <text className="th" x="482" y="92" textAnchor="middle" dominantBaseline="central">
          Text encoder
        </text>
        <text className="ts" x="482" y="112" textAnchor="middle" dominantBaseline="central">
          12 transformer layers
        </text>
      </g>

      {/* ========== Connector: CLIP -> Denoising loop ========== */}
      {/* L-shaped path: text encoder bottom (482,128) -> (482,168) -> (344,168) -> denoising step top (344,210) */}
      <path
        d="M 482 128 L 482 168 L 344 168 L 344 210"
        fill="none" className="conn-purple" markerEnd="url(#sdp-arrow)"
      />
      <text
        className="conn-purple-label"
        x="420" y="162"
        textAnchor="middle" dominantBaseline="central"
      >
        [77, 768] text embeddings
      </text>

      {/* ========== GROUP 2: Denoising loop (blue) ========== */}
      <rect
        className="grp-blue"
        x="40" y="174" width="600" height="170" rx="16"
        strokeWidth="0.5" opacity="0.35"
      />
      <text
        className="grp-blue-label"
        x="340" y="200"
        textAnchor="middle" dominantBaseline="central"
      >
        {'Denoising loop \u2014 U-Net (~860M params) \u00d7 50 steps'}
      </text>

      {/* z_T noise sample */}
      <g className="n-blue">
        <rect x="60" y="218" width="130" height="56" rx="8" strokeWidth="0.5" />
        <text className="th" x="125" y="238" textAnchor="middle" dominantBaseline="central">
          {'z\u1d1b ~ N(0,I)'}
        </text>
        <text className="ts" x="125" y="258" textAnchor="middle" dominantBaseline="central">
          [4, 64, 64]
        </text>
      </g>

      {/* Arrow: z_T -> Per step */}
      <line
        x1="190" y1="246" x2="218" y2="246"
        className="arr" markerEnd="url(#sdp-arrow)"
      />

      {/* Per denoising step */}
      <g className="n-blue">
        <rect x="224" y="216" width="240" height="108" rx="8" strokeWidth="0.5" />
        <text className="th" x="344" y="236" textAnchor="middle" dominantBaseline="central">
          Per denoising step
        </text>
        <text className="ts" x="344" y="258" textAnchor="middle" dominantBaseline="central">
          1. Embed timestep t
        </text>
        <text className="ts" x="344" y="276" textAnchor="middle" dominantBaseline="central">
          2. U-Net pass (uncond)
        </text>
        <text className="ts" x="344" y="294" textAnchor="middle" dominantBaseline="central">
          3. U-Net pass (cond)
        </text>
        <text className="ts" x="344" y="312" textAnchor="middle" dominantBaseline="central">
          4. CFG combine + sched step
        </text>
      </g>

      {/* Arrow: Per step -> z_0 */}
      <line
        x1="464" y1="246" x2="498" y2="246"
        className="arr" markerEnd="url(#sdp-arrow)"
      />

      {/* z_0 clean latent */}
      <g className="n-blue">
        <rect x="504" y="218" width="120" height="56" rx="8" strokeWidth="0.5" />
        <text className="th" x="564" y="238" textAnchor="middle" dominantBaseline="central">
          {'z\u2080'}
        </text>
        <text className="ts" x="564" y="258" textAnchor="middle" dominantBaseline="central">
          [4, 64, 64]
        </text>
      </g>

      {/* ========== Connector: Denoising loop -> VAE ========== */}
      {/* L-shaped path: z_0 bottom (564,274) -> (564,384) -> (340,384) -> decoder top (340,434) */}
      <path
        d="M 564 274 L 564 384 L 340 384 L 340 434"
        fill="none" className="conn-blue" markerEnd="url(#sdp-arrow)"
      />
      <text
        className="conn-blue-label"
        x="578" y="340"
        dominantBaseline="central"
      >
        [4, 64, 64]
      </text>

      {/* ========== GROUP 3: VAE Decoder (green) ========== */}
      <rect
        className="grp-green"
        x="40" y="394" width="600" height="90" rx="16"
        strokeWidth="0.5" opacity="0.35"
      />
      <text
        className="grp-green-label"
        x="340" y="416"
        textAnchor="middle" dominantBaseline="central"
      >
        VAE decoder (frozen, ~84M params)
      </text>

      {/* Decoder */}
      <g className="n-green">
        <rect x="280" y="434" width="120" height="44" rx="8" strokeWidth="0.5" />
        <text className="th" x="340" y="456" textAnchor="middle" dominantBaseline="central">
          Decoder
        </text>
      </g>

      {/* Arrow: Decoder -> Image */}
      <line
        x1="400" y1="456" x2="438" y2="456"
        className="arr" markerEnd="url(#sdp-arrow)"
      />

      {/* Image output */}
      <g className="n-green">
        <rect x="444" y="428" width="128" height="56" rx="8" strokeWidth="0.5" />
        <text className="th" x="508" y="448" textAnchor="middle" dominantBaseline="central">
          Image
        </text>
        <text className="ts" x="508" y="468" textAnchor="middle" dominantBaseline="central">
          [3, 512, 512]
        </text>
      </g>
    </svg>
  )
}

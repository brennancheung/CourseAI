export function IPAdapterAttentionDiagram() {
  return (
    <svg
      data-ipa=""
      xmlns="http://www.w3.org/2000/svg"
      width="100%"
      viewBox="0 0 680 530"
      role="img"
      aria-label="IP-Adapter decoupled cross-attention: spatial features produce shared Q, text embeddings produce frozen K/V for text attention, image embeddings produce trainable K/V for image attention, outputs are summed."
    >
      <defs>
        <style>{`
          [data-ipa] text {
            font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
          }
          [data-ipa] .th { font-size: 14px; font-weight: 500; }
          [data-ipa] .ts { font-size: 12px; font-weight: 400; }
          [data-ipa] .arr { stroke: #9c9a92; stroke-width: 1.5; }
          [data-ipa] .ah { stroke: #9c9a92; stroke-width: 1.5; stroke-linecap: round; stroke-linejoin: round; }

          [data-ipa] .n-gray rect { fill: #F1EFE8; stroke: #5F5E5A; }
          [data-ipa] .n-gray .th { fill: #444441; }
          [data-ipa] .n-gray .ts { fill: #5F5E5A; }

          [data-ipa] .n-purple rect { fill: #EEEDFE; stroke: #534AB7; }
          [data-ipa] .n-purple .th { fill: #3C3489; }
          [data-ipa] .n-purple .ts { fill: #534AB7; }

          [data-ipa] .n-green rect { fill: #EAF3DE; stroke: #3B6D11; }
          [data-ipa] .n-green .th { fill: #27500A; }
          [data-ipa] .n-green .ts { fill: #3B6D11; }

          [data-ipa] .sum-circle { fill: #EAF3DE; stroke: #3B6D11; }
          [data-ipa] .sum-text { font-size: 16px; font-weight: 500; fill: #3B6D11; }
          [data-ipa] .lbl { font-size: 12px; font-weight: 400; fill: #9c9a92; }

          @media (prefers-color-scheme: dark) {
            [data-ipa] .arr { stroke: #6b6b65; }
            [data-ipa] .ah { stroke: #6b6b65; }

            [data-ipa] .n-gray rect { fill: #2C2C2A; stroke: #B4B2A9; }
            [data-ipa] .n-gray .th { fill: #D3D1C7; }
            [data-ipa] .n-gray .ts { fill: #B4B2A9; }

            [data-ipa] .n-purple rect { fill: #26215C; stroke: #AFA9EC; }
            [data-ipa] .n-purple .th { fill: #CECBF6; }
            [data-ipa] .n-purple .ts { fill: #AFA9EC; }

            [data-ipa] .n-green rect { fill: #173404; stroke: #97C459; }
            [data-ipa] .n-green .th { fill: #C0DD97; }
            [data-ipa] .n-green .ts { fill: #97C459; }

            [data-ipa] .sum-circle { fill: #173404; stroke: #97C459; }
            [data-ipa] .sum-text { fill: #97C459; }
            [data-ipa] .lbl { fill: #6b6b65; }
          }
        `}</style>
        <marker
          id="ipa-arrow"
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

      {/* === TIER 0 (y=40): Input embeddings === */}

      <g className="n-gray">
        <rect x="250" y="40" width="180" height="56" rx="8" strokeWidth="0.5" />
        <text className="th" x="340" y="58" textAnchor="middle" dominantBaseline="central">
          Spatial features
        </text>
        <text className="ts" x="340" y="78" textAnchor="middle" dominantBaseline="central">
          From U-Net
        </text>
      </g>

      <g className="n-gray">
        <rect x="45" y="40" width="170" height="56" rx="8" strokeWidth="0.5" />
        <text className="th" x="130" y="58" textAnchor="middle" dominantBaseline="central">
          Text embeddings
        </text>
        <text className="ts" x="130" y="78" textAnchor="middle" dominantBaseline="central">
          77 tokens
        </text>
      </g>

      <g className="n-purple">
        <rect x="465" y="40" width="170" height="56" rx="8" strokeWidth="0.5" />
        <text className="th" x="550" y="58" textAnchor="middle" dominantBaseline="central">
          Image embeddings
        </text>
        <text className="ts" x="550" y="78" textAnchor="middle" dominantBaseline="central">
          257 tokens
        </text>
      </g>

      {/* === Arrows: Tier 0 -> Tier 1 === */}
      <line x1="340" y1="96" x2="340" y2="130" className="arr" markerEnd="url(#ipa-arrow)" />
      <line x1="100" y1="96" x2="90" y2="130" className="arr" markerEnd="url(#ipa-arrow)" />
      <line x1="160" y1="96" x2="205" y2="130" className="arr" markerEnd="url(#ipa-arrow)" />
      <line x1="520" y1="96" x2="500" y2="130" className="arr" markerEnd="url(#ipa-arrow)" />
      <line x1="580" y1="96" x2="620" y2="130" className="arr" markerEnd="url(#ipa-arrow)" />

      {/* === TIER 1 (y=130): Projection weights === */}

      <g className="n-gray">
        <rect x="275" y="130" width="130" height="44" rx="8" strokeWidth="0.5" />
        <text className="th" x="340" y="152" textAnchor="middle" dominantBaseline="central">
          W_Q (shared)
        </text>
      </g>

      <g className="n-gray">
        <rect x="40" y="130" width="100" height="44" rx="8" strokeWidth="0.5" />
        <text className="th" x="90" y="152" textAnchor="middle" dominantBaseline="central">
          W_K_text
        </text>
      </g>

      <g className="n-gray">
        <rect x="155" y="130" width="100" height="44" rx="8" strokeWidth="0.5" />
        <text className="th" x="205" y="152" textAnchor="middle" dominantBaseline="central">
          W_V_text
        </text>
      </g>

      <g className="n-purple">
        <rect x="445" y="130" width="110" height="44" rx="8" strokeWidth="0.5" />
        <text className="th" x="500" y="152" textAnchor="middle" dominantBaseline="central">
          W_K_image
        </text>
      </g>

      <g className="n-purple">
        <rect x="570" y="130" width="100" height="44" rx="8" strokeWidth="0.5" />
        <text className="th" x="620" y="152" textAnchor="middle" dominantBaseline="central">
          W_V_image
        </text>
      </g>

      {/* === Arrows: Tier 1 -> Tier 2 === */}
      <line x1="340" y1="174" x2="340" y2="208" className="arr" markerEnd="url(#ipa-arrow)" />
      <line x1="90" y1="174" x2="90" y2="208" className="arr" markerEnd="url(#ipa-arrow)" />
      <line x1="205" y1="174" x2="205" y2="208" className="arr" markerEnd="url(#ipa-arrow)" />
      <line x1="500" y1="174" x2="500" y2="208" className="arr" markerEnd="url(#ipa-arrow)" />
      <line x1="620" y1="174" x2="620" y2="208" className="arr" markerEnd="url(#ipa-arrow)" />

      {/* === TIER 2 (y=208): Q, K, V intermediate results === */}

      <g className="n-gray">
        <rect x="310" y="208" width="60" height="40" rx="8" strokeWidth="0.5" />
        <text className="th" x="340" y="228" textAnchor="middle" dominantBaseline="central">
          Q
        </text>
      </g>

      <g className="n-gray">
        <rect x="50" y="208" width="80" height="40" rx="8" strokeWidth="0.5" />
        <text className="th" x="90" y="228" textAnchor="middle" dominantBaseline="central">
          K_text
        </text>
      </g>

      <g className="n-gray">
        <rect x="165" y="208" width="80" height="40" rx="8" strokeWidth="0.5" />
        <text className="th" x="205" y="228" textAnchor="middle" dominantBaseline="central">
          V_text
        </text>
      </g>

      <g className="n-purple">
        <rect x="460" y="208" width="80" height="40" rx="8" strokeWidth="0.5" />
        <text className="th" x="500" y="228" textAnchor="middle" dominantBaseline="central">
          K_image
        </text>
      </g>

      <g className="n-purple">
        <rect x="580" y="208" width="80" height="40" rx="8" strokeWidth="0.5" />
        <text className="th" x="620" y="228" textAnchor="middle" dominantBaseline="central">
          V_image
        </text>
      </g>

      {/* === Arrows: Q fans out to both attention boxes (L-shaped) === */}
      <path
        d="M 310 238 L 170 238 L 170 290"
        fill="none"
        className="arr"
        markerEnd="url(#ipa-arrow)"
      />
      <path
        d="M 370 238 L 510 238 L 510 290"
        fill="none"
        className="arr"
        markerEnd="url(#ipa-arrow)"
      />

      {/* K_text, V_text -> Text attention */}
      <line x1="90" y1="248" x2="120" y2="290" className="arr" markerEnd="url(#ipa-arrow)" />
      <line x1="205" y1="248" x2="195" y2="290" className="arr" markerEnd="url(#ipa-arrow)" />

      {/* K_image, V_image -> Image attention */}
      <line x1="500" y1="248" x2="480" y2="290" className="arr" markerEnd="url(#ipa-arrow)" />
      <line x1="620" y1="248" x2="555" y2="290" className="arr" markerEnd="url(#ipa-arrow)" />

      {/* === TIER 3 (y=290): Attention operations === */}

      <g className="n-gray">
        <rect x="85" y="290" width="170" height="44" rx="8" strokeWidth="0.5" />
        <text className="th" x="170" y="312" textAnchor="middle" dominantBaseline="central">
          Text attention
        </text>
      </g>

      <g className="n-purple">
        <rect x="425" y="290" width="170" height="44" rx="8" strokeWidth="0.5" />
        <text className="th" x="510" y="312" textAnchor="middle" dominantBaseline="central">
          Image attention
        </text>
      </g>

      {/* === TIER 4: Weighted sum labels + arrows === */}
      <text className="lbl" x="235" y="362" textAnchor="middle" dominantBaseline="central">
        text_out
      </text>
      <text className="lbl" x="445" y="362" textAnchor="middle" dominantBaseline="central">
        scale * image_out
      </text>

      {/* Text attention -> plus node (L-shaped) */}
      <path
        d="M 170 334 L 170 398 L 310 398"
        fill="none"
        className="arr"
        markerEnd="url(#ipa-arrow)"
      />
      {/* Image attention -> plus node (L-shaped) */}
      <path
        d="M 510 334 L 510 398 L 370 398"
        fill="none"
        className="arr"
        markerEnd="url(#ipa-arrow)"
      />

      {/* Plus / sum circle */}
      <circle cx="340" cy="398" r="16" className="sum-circle" strokeWidth="1" />
      <text className="sum-text" x="340" y="398" textAnchor="middle" dominantBaseline="central">
        +
      </text>

      {/* === TIER 5 (y=444): Combined output === */}
      <line x1="340" y1="414" x2="340" y2="444" className="arr" markerEnd="url(#ipa-arrow)" />

      <g className="n-green">
        <rect x="255" y="444" width="170" height="50" rx="8" strokeWidth="0.5" />
        <text className="th" x="340" y="462" textAnchor="middle" dominantBaseline="central">
          Combined output
        </text>
        <text className="ts" x="340" y="480" textAnchor="middle" dominantBaseline="central">
          To next layer
        </text>
      </g>
    </svg>
  )
}

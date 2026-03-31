export function LoraInjectionDiagram() {
  return (
    <svg
      data-lid=""
      xmlns="http://www.w3.org/2000/svg"
      width="100%"
      viewBox="0 0 680 590"
      role="img"
      aria-label="LoRA injection in U-Net residual block: conv blocks, self-attention, and adaptive GroupNorm are frozen, while cross-attention projections W_Q, W_K, W_V, and W_out have trainable LoRA adapters"
    >
      <defs>
        <style>{`
          [data-lid] text {
            font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
          }
          [data-lid] .th { font-size: 14px; font-weight: 500; }
          [data-lid] .ts { font-size: 12px; font-weight: 400; }
          [data-lid] .arr { stroke: #9c9a92; stroke-width: 1.5; }
          [data-lid] .ah { stroke: #9c9a92; stroke-width: 1.5; stroke-linecap: round; stroke-linejoin: round; }

          [data-lid] .container-fill { fill: #f5f5f0; stroke: rgba(0,0,0,0.15); }
          [data-lid] .container-label { fill: #1a1a1a; }
          [data-lid] .sub-border { stroke: #534AB7; fill: none; }
          [data-lid] .sub-label { fill: #534AB7; }

          [data-lid] .n-gray rect { fill: #F1EFE8; stroke: #5F5E5A; }
          [data-lid] .n-gray .th { fill: #444441; }
          [data-lid] .n-purple rect { fill: #EEEDFE; stroke: #534AB7; }
          [data-lid] .n-purple .th { fill: #3C3489; }

          [data-lid] .legend-frozen { fill: #F1EFE8; stroke: #5F5E5A; }
          [data-lid] .legend-lora { fill: #EEEDFE; stroke: #534AB7; }
          [data-lid] .legend-text { fill: #6b6b65; font-size: 12px; }

          @media (prefers-color-scheme: dark) {
            [data-lid] .arr { stroke: #6b6b65; }
            [data-lid] .ah { stroke: #6b6b65; }
            [data-lid] .container-fill { fill: #2c2c2a; stroke: rgba(255,255,255,0.15); }
            [data-lid] .container-label { fill: #e8e6df; }
            [data-lid] .sub-border { stroke: #AFA9EC; }
            [data-lid] .sub-label { fill: #AFA9EC; }

            [data-lid] .n-gray rect { fill: #2C2C2A; stroke: #B4B2A9; }
            [data-lid] .n-gray .th { fill: #D3D1C7; }
            [data-lid] .n-purple rect { fill: #26215C; stroke: #AFA9EC; }
            [data-lid] .n-purple .th { fill: #CECBF6; }

            [data-lid] .legend-frozen { fill: #2C2C2A; stroke: #B4B2A9; }
            [data-lid] .legend-lora { fill: #26215C; stroke: #AFA9EC; }
            [data-lid] .legend-text { fill: #9c9a92; }
          }
        `}</style>
        <marker
          id="lid-arrow"
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

      {/* Outer container: U-Net Residual Block */}
      <rect
        x="140" y="40" width="400" height="470" rx="20"
        className="container-fill" strokeWidth="0.5"
      />
      <text
        className="th container-label" x="340" y="66"
        textAnchor="middle" dominantBaseline="central"
      >
        U-Net residual block
      </text>

      {/* Conv block 1 (frozen) */}
      <g className="n-gray">
        <rect x="265" y="86" width="150" height="44" rx="8" strokeWidth="0.5" />
        <text className="th" x="340" y="108" textAnchor="middle" dominantBaseline="central">
          Conv block
        </text>
      </g>

      {/* Arrow: Conv 1 → Conv 2 */}
      <line x1="340" y1="130" x2="340" y2="150" className="arr" markerEnd="url(#lid-arrow)" />

      {/* Conv block 2 (frozen) */}
      <g className="n-gray">
        <rect x="265" y="150" width="150" height="44" rx="8" strokeWidth="0.5" />
        <text className="th" x="340" y="172" textAnchor="middle" dominantBaseline="central">
          Conv block
        </text>
      </g>

      {/* Arrow: Conv 2 → Self-attention */}
      <line x1="340" y1="194" x2="340" y2="214" className="arr" markerEnd="url(#lid-arrow)" />

      {/* Self-attention (frozen) */}
      <g className="n-gray">
        <rect x="253" y="214" width="174" height="44" rx="8" strokeWidth="0.5" />
        <text className="th" x="340" y="236" textAnchor="middle" dominantBaseline="central">
          Self-attention
        </text>
      </g>

      {/* Arrow: Self-attention → Cross-attention subgroup */}
      <line x1="340" y1="258" x2="340" y2="282" className="arr" markerEnd="url(#lid-arrow)" />

      {/* Cross-attention subgroup border */}
      <rect
        x="168" y="282" width="344" height="152" rx="12"
        className="sub-border" strokeWidth="0.5" strokeDasharray="4 4" opacity="0.6"
      />
      <text
        className="ts sub-label" x="340" y="300"
        textAnchor="middle" dominantBaseline="central"
      >
        Cross-attention
      </text>

      {/* W_Q + LoRA (trainable) */}
      <g className="n-purple">
        <rect x="185" y="314" width="150" height="44" rx="8" strokeWidth="0.5" />
        <text className="th" x="260" y="336" textAnchor="middle" dominantBaseline="central">
          W_Q + LoRA
        </text>
      </g>

      {/* W_K + LoRA (trainable) */}
      <g className="n-purple">
        <rect x="345" y="314" width="150" height="44" rx="8" strokeWidth="0.5" />
        <text className="th" x="420" y="336" textAnchor="middle" dominantBaseline="central">
          W_K + LoRA
        </text>
      </g>

      {/* W_V + LoRA (trainable) */}
      <g className="n-purple">
        <rect x="185" y="374" width="150" height="44" rx="8" strokeWidth="0.5" />
        <text className="th" x="260" y="396" textAnchor="middle" dominantBaseline="central">
          W_V + LoRA
        </text>
      </g>

      {/* W_out + LoRA (trainable) */}
      <g className="n-purple">
        <rect x="345" y="374" width="150" height="44" rx="8" strokeWidth="0.5" />
        <text className="th" x="420" y="396" textAnchor="middle" dominantBaseline="central">
          W_out + LoRA
        </text>
      </g>

      {/* Arrow: Cross-attention → Adaptive GroupNorm */}
      <line x1="340" y1="434" x2="340" y2="456" className="arr" markerEnd="url(#lid-arrow)" />

      {/* Adaptive GroupNorm (frozen) */}
      <g className="n-gray">
        <rect x="228" y="456" width="224" height="44" rx="8" strokeWidth="0.5" />
        <text className="th" x="340" y="478" textAnchor="middle" dominantBaseline="central">
          Adaptive GroupNorm
        </text>
      </g>

      {/* Legend */}
      <rect x="56" y="530" width="14" height="14" rx="3" className="legend-frozen" strokeWidth="0.5" />
      <text className="legend-text" x="78" y="537" dominantBaseline="central">
        Frozen
      </text>

      <rect x="130" y="530" width="14" height="14" rx="3" className="legend-lora" strokeWidth="0.5" />
      <text className="legend-text" x="152" y="537" dominantBaseline="central">
        Trainable (LoRA)
      </text>
    </svg>
  )
}

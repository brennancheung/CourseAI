export function TextualInversionFlowDiagram() {
  return (
    <svg
      data-tif=""
      xmlns="http://www.w3.org/2000/svg"
      width="100%"
      viewBox="0 0 680 280"
      role="img"
      aria-label="Textual inversion training flow: forward pass through embedding, CLIP, cross-attention, U-Net to loss, with gradient backpropagation returning to the single trainable embedding row"
    >
      <defs>
        <style>{`
          [data-tif] text {
            font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
          }
          [data-tif] .th { font-size: 14px; font-weight: 500; }
          [data-tif] .ts { font-size: 12px; font-weight: 400; }
          [data-tif] .arr { stroke: #9c9a92; stroke-width: 1.5; }
          [data-tif] .ah { stroke: #9c9a92; stroke-width: 1.5; stroke-linecap: round; stroke-linejoin: round; }

          [data-tif] .n-purple rect { fill: #EEEDFE; stroke: #534AB7; }
          [data-tif] .n-purple .th { fill: #3C3489; }
          [data-tif] .n-purple .ts { fill: #534AB7; }
          [data-tif] .n-coral rect { fill: #FAECE7; stroke: #993C1D; }
          [data-tif] .n-coral .th { fill: #712B13; }
          [data-tif] .n-coral .ts { fill: #993C1D; }
          [data-tif] .n-gray rect { fill: #F1EFE8; stroke: #5F5E5A; }
          [data-tif] .n-gray .th { fill: #444441; }
          [data-tif] .n-gray .ts { fill: #5F5E5A; }

          [data-tif] .label-dim { fill: #9c9a92; font-family: system-ui, -apple-system, "Segoe UI", sans-serif; font-size: 12px; font-weight: 400; }
          [data-tif] .grad-line { stroke: #7F77DD; stroke-width: 1.5; }
          [data-tif] .grad-tick { stroke: #7F77DD; stroke-width: 0.8; stroke-dasharray: 3 3; }

          @media (prefers-color-scheme: dark) {
            [data-tif] .arr { stroke: #6b6b65; }
            [data-tif] .ah { stroke: #6b6b65; }
            [data-tif] .n-purple rect { fill: #26215C; stroke: #AFA9EC; }
            [data-tif] .n-purple .th { fill: #CECBF6; }
            [data-tif] .n-purple .ts { fill: #AFA9EC; }
            [data-tif] .n-coral rect { fill: #4A1B0C; stroke: #F0997B; }
            [data-tif] .n-coral .th { fill: #F5C4B3; }
            [data-tif] .n-coral .ts { fill: #F0997B; }
            [data-tif] .n-gray rect { fill: #2C2C2A; stroke: #B4B2A9; }
            [data-tif] .n-gray .th { fill: #D3D1C7; }
            [data-tif] .n-gray .ts { fill: #B4B2A9; }
            [data-tif] .label-dim { fill: #6b6b65; }
            [data-tif] .grad-line { stroke: #AFA9EC; }
            [data-tif] .grad-tick { stroke: #AFA9EC; }
          }

          @media (prefers-reduced-motion: no-preference) {
            [data-tif] .gradient-flow {
              stroke-dasharray: 6 4;
              animation: tif-flow 1.2s linear infinite;
            }
          }
          @keyframes tif-flow { to { stroke-dashoffset: -20; } }
        `}</style>
        <marker
          id="tif-arrow"
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
          id="tif-arrow-purple"
          viewBox="0 0 10 10"
          refX="8"
          refY="5"
          markerWidth="6"
          markerHeight="6"
          orient="auto-start-reverse"
        >
          <path d="M2 1L8 5L2 9" fill="none" stroke="#7F77DD" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
        </marker>
      </defs>

      {/* Forward path label */}
      <text className="label-dim" x="340" y="28" textAnchor="middle" dominantBaseline="central">
        forward pass
      </text>

      {/* Node 1: Embedding lookup (purple, trainable) */}
      <g className="n-purple">
        <rect x="40" y="48" width="108" height="56" rx="8" strokeWidth="0.5" />
        <text className="th" x="94" y="68" textAnchor="middle" dominantBaseline="central">
          Embedding
        </text>
        <text className="ts" x="94" y="88" textAnchor="middle" dominantBaseline="central">
          1 row trainable
        </text>
      </g>

      {/* Forward arrow: Embedding -> CLIP */}
      <line x1="152" y1="76" x2="172" y2="76" className="arr" markerEnd="url(#tif-arrow)" />

      {/* Node 2: CLIP transformer (gray, frozen) */}
      <g className="n-gray">
        <rect x="178" y="48" width="108" height="56" rx="8" strokeWidth="0.5" />
        <text className="th" x="232" y="68" textAnchor="middle" dominantBaseline="central">
          CLIP encoder
        </text>
        <text className="ts" x="232" y="88" textAnchor="middle" dominantBaseline="central">
          frozen
        </text>
      </g>

      {/* Forward arrow: CLIP -> Cross-attn */}
      <line x1="290" y1="76" x2="310" y2="76" className="arr" markerEnd="url(#tif-arrow)" />

      {/* Tensor shape label on CLIP->Cross-attn arrow */}
      <text className="label-dim" x="300" y="62" textAnchor="middle" dominantBaseline="central">
        [77, 768]
      </text>

      {/* Node 3: Cross-attention K/V (gray, frozen) */}
      <g className="n-gray">
        <rect x="316" y="48" width="108" height="56" rx="8" strokeWidth="0.5" />
        <text className="th" x="370" y="68" textAnchor="middle" dominantBaseline="central">
          Cross-attn K/V
        </text>
        <text className="ts" x="370" y="88" textAnchor="middle" dominantBaseline="central">
          frozen
        </text>
      </g>

      {/* Forward arrow: Cross-attn -> U-Net */}
      <line x1="428" y1="76" x2="448" y2="76" className="arr" markerEnd="url(#tif-arrow)" />

      {/* Node 4: U-Net denoise (gray, frozen) */}
      <g className="n-gray">
        <rect x="454" y="48" width="108" height="56" rx="8" strokeWidth="0.5" />
        <text className="th" x="508" y="68" textAnchor="middle" dominantBaseline="central">
          U-Net denoise
        </text>
        <text className="ts" x="508" y="88" textAnchor="middle" dominantBaseline="central">
          frozen
        </text>
      </g>

      {/* Forward arrow: U-Net -> Loss */}
      <line x1="566" y1="76" x2="586" y2="76" className="arr" markerEnd="url(#tif-arrow)" />

      {/* Node 5: MSE Loss (coral) */}
      <g className="n-coral">
        <rect x="592" y="56" width="56" height="40" rx="8" strokeWidth="0.5" />
        <text className="th" x="620" y="76" textAnchor="middle" dominantBaseline="central">
          Loss
        </text>
      </g>

      {/* Backward / gradient path: curved L-path from Loss bottom, left, up to Embedding bottom */}
      <path
        d="M 620 96 L 620 148 L 94 148 L 94 108"
        fill="none"
        className="grad-line gradient-flow"
        markerEnd="url(#tif-arrow-purple)"
      />

      {/* Gradient ticks at each frozen node (upward stubs from the gradient path) */}
      <line x1="508" y1="148" x2="508" y2="108" className="grad-tick" opacity="0.5" />
      <line x1="370" y1="148" x2="370" y2="108" className="grad-tick" opacity="0.5" />
      <line x1="232" y1="148" x2="232" y2="108" className="grad-tick" opacity="0.5" />

      {/* Gradient path label */}
      <text className="label-dim" x="358" y="170" textAnchor="middle" dominantBaseline="central">
        gradient backpropagation
      </text>

      {/* Legend */}
      <line x1="180" y1="220" x2="210" y2="220" className="arr" markerEnd="url(#tif-arrow)" />
      <text className="label-dim" x="218" y="220" dominantBaseline="central">
        forward pass (all layers)
      </text>

      <line x1="180" y1="242" x2="210" y2="242" className="grad-line" strokeDasharray="6 4" markerEnd="url(#tif-arrow-purple)" />
      <text className="label-dim" x="218" y="242" dominantBaseline="central">
        gradient flows back (only embedding updates)
      </text>
    </svg>
  )
}

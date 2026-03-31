export function ControlNetArchitectureDiagram() {
  return (
    <svg
      data-cna=""
      xmlns="http://www.w3.org/2000/svg"
      width="100%"
      viewBox="0 0 680 760"
      role="img"
      aria-label="ControlNet architecture: frozen SD encoder and decoder with skip connections on the left, trainable encoder copy on the right connected via zero convolution nodes to the frozen decoder merge points"
    >
      <defs>
        <style>{`
          [data-cna] text {
            font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
          }
          [data-cna] .th { font-size: 14px; font-weight: 500; }
          [data-cna] .ts { font-size: 12px; font-weight: 400; }
          [data-cna] .arr { stroke: #9c9a92; stroke-width: 1.5; }

          [data-cna] .container-frozen { fill: none; stroke: rgba(0,0,0,0.15); }
          [data-cna] .container-cn { fill: none; stroke: #7F77DD; }
          [data-cna] .container-label { fill: #6b6b65; font-size: 12px; }
          [data-cna] .container-label-cn { fill: #7F77DD; font-size: 12px; }

          [data-cna] .n-gray rect { fill: #F1EFE8; stroke: #5F5E5A; }
          [data-cna] .n-gray .th { fill: #444441; }
          [data-cna] .n-gray .ts { fill: #5F5E5A; }

          [data-cna] .n-purple rect { fill: #EEEDFE; stroke: #534AB7; }
          [data-cna] .n-purple .th { fill: #3C3489; }
          [data-cna] .n-purple .ts { fill: #534AB7; }

          [data-cna] .n-teal rect { fill: #E1F5EE; stroke: #0F6E56; }
          [data-cna] .n-teal .th { fill: #085041; }

          [data-cna] .n-blue rect { fill: #E6F1FB; stroke: #185FA5; }
          [data-cna] .n-blue .th { fill: #0C447C; }

          [data-cna] .merge-circle { fill: #ffffff; stroke: #1D9E75; }
          [data-cna] .merge-text { fill: #0F6E56; font-size: 14px; font-weight: 600; }
          [data-cna] .merge-link { stroke: #1D9E75; stroke-width: 1.5; }

          [data-cna] .skip { stroke: #888780; stroke-width: 1; stroke-dasharray: 4 3; fill: none; }
          [data-cna] .zc-line { stroke: #1D9E75; stroke-width: 1.5; fill: none; }
          [data-cna] .zc-stub { stroke: #1D9E75; stroke-width: 1.5; }

          @media (prefers-color-scheme: dark) {
            [data-cna] .arr { stroke: #6b6b65; }

            [data-cna] .container-frozen { stroke: rgba(255,255,255,0.15); }
            [data-cna] .container-cn { stroke: #AFA9EC; }
            [data-cna] .container-label { fill: #9c9a92; }
            [data-cna] .container-label-cn { fill: #AFA9EC; }

            [data-cna] .n-gray rect { fill: #2C2C2A; stroke: #B4B2A9; }
            [data-cna] .n-gray .th { fill: #D3D1C7; }
            [data-cna] .n-gray .ts { fill: #B4B2A9; }

            [data-cna] .n-purple rect { fill: #26215C; stroke: #AFA9EC; }
            [data-cna] .n-purple .th { fill: #CECBF6; }
            [data-cna] .n-purple .ts { fill: #AFA9EC; }

            [data-cna] .n-teal rect { fill: #04342C; stroke: #5DCAA5; }
            [data-cna] .n-teal .th { fill: #9FE1CB; }

            [data-cna] .n-blue rect { fill: #042C53; stroke: #85B7EB; }
            [data-cna] .n-blue .th { fill: #B5D4F4; }

            [data-cna] .merge-circle { fill: #1a1a1a; stroke: #5DCAA5; }
            [data-cna] .merge-text { fill: #5DCAA5; }
            [data-cna] .merge-link { stroke: #5DCAA5; }

            [data-cna] .skip { stroke: #6b6b65; }
            [data-cna] .zc-line { stroke: #5DCAA5; }
            [data-cna] .zc-stub { stroke: #5DCAA5; }
          }
        `}</style>
        <marker
          id="cna-arrow"
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
        <marker
          id="cna-arrow-skip"
          viewBox="0 0 10 10"
          refX="8"
          refY="5"
          markerWidth="5"
          markerHeight="5"
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

      {/* ===== Container: Frozen SD Model ===== */}
      <rect
        x="30" y="44" width="346" height="700" rx="16"
        className="container-frozen" strokeWidth="0.5"
        strokeDasharray="6 4" opacity="0.4"
      />
      <text
        className="container-label" x="203" y="62"
        textAnchor="middle" opacity="0.6"
      >
        Frozen SD model
      </text>

      {/* ===== Container: ControlNet copy ===== */}
      <rect
        x="390" y="44" width="252" height="346" rx="16"
        className="container-cn" strokeWidth="0.5"
        strokeDasharray="6 4" opacity="0.4"
      />
      <text
        className="container-label-cn" x="516" y="62"
        textAnchor="middle" opacity="0.6"
      >
        Trainable ControlNet copy
      </text>

      {/* ===== Input nodes ===== */}
      <g className="n-blue">
        <rect x="40" y="76" width="130" height="34" rx="8" strokeWidth="0.5" />
        <text className="th" x="105" y="93" textAnchor="middle" dominantBaseline="central">
          Noisy latent
        </text>
      </g>
      <g className="n-blue">
        <rect x="500" y="76" width="130" height="34" rx="8" strokeWidth="0.5" />
        <text className="th" x="565" y="93" textAnchor="middle" dominantBaseline="central">
          Spatial map
        </text>
      </g>

      {/* Input arrows */}
      <line x1="105" y1="110" x2="105" y2="126" className="arr" markerEnd="url(#cna-arrow)" />
      <line x1="565" y1="110" x2="565" y2="126" className="arr" markerEnd="url(#cna-arrow)" />

      {/* ===== Frozen encoder (Col A, gray) ===== */}
      <g className="n-gray">
        <rect x="40" y="130" width="130" height="44" rx="8" strokeWidth="0.5" />
        <text className="th" x="105" y="145" textAnchor="middle" dominantBaseline="central">Enc 64x64</text>
        <text className="ts" x="105" y="163" textAnchor="middle" dominantBaseline="central">frozen</text>
      </g>
      <g className="n-gray">
        <rect x="40" y="200" width="130" height="44" rx="8" strokeWidth="0.5" />
        <text className="th" x="105" y="215" textAnchor="middle" dominantBaseline="central">Enc 32x32</text>
        <text className="ts" x="105" y="233" textAnchor="middle" dominantBaseline="central">frozen</text>
      </g>
      <g className="n-gray">
        <rect x="40" y="270" width="130" height="44" rx="8" strokeWidth="0.5" />
        <text className="th" x="105" y="285" textAnchor="middle" dominantBaseline="central">Enc 16x16</text>
        <text className="ts" x="105" y="303" textAnchor="middle" dominantBaseline="central">frozen</text>
      </g>
      <g className="n-gray">
        <rect x="40" y="340" width="130" height="44" rx="8" strokeWidth="0.5" />
        <text className="th" x="105" y="355" textAnchor="middle" dominantBaseline="central">Enc 8x8</text>
        <text className="ts" x="105" y="373" textAnchor="middle" dominantBaseline="central">frozen</text>
      </g>

      {/* Encoder vertical arrows */}
      <line x1="105" y1="174" x2="105" y2="196" className="arr" markerEnd="url(#cna-arrow)" />
      <line x1="105" y1="244" x2="105" y2="266" className="arr" markerEnd="url(#cna-arrow)" />
      <line x1="105" y1="314" x2="105" y2="336" className="arr" markerEnd="url(#cna-arrow)" />

      {/* ===== Bottleneck ===== */}
      <g className="n-gray">
        <rect x="40" y="410" width="130" height="44" rx="8" strokeWidth="0.5" />
        <text className="th" x="105" y="432" textAnchor="middle" dominantBaseline="central">Bottleneck</text>
      </g>
      <line x1="105" y1="384" x2="105" y2="406" className="arr" markerEnd="url(#cna-arrow)" />

      {/* Bottleneck to decoder: L-shaped path */}
      <path
        d="M 170 432 L 230 432 L 230 502 L 235 502"
        fill="none" className="arr" markerEnd="url(#cna-arrow)"
      />

      {/* ===== Frozen decoder (Col B, gray) ===== */}
      <g className="n-gray">
        <rect x="235" y="480" width="130" height="44" rx="8" strokeWidth="0.5" />
        <text className="th" x="300" y="495" textAnchor="middle" dominantBaseline="central">Dec 8x8</text>
        <text className="ts" x="300" y="513" textAnchor="middle" dominantBaseline="central">frozen</text>
      </g>
      <g className="n-gray">
        <rect x="235" y="550" width="130" height="44" rx="8" strokeWidth="0.5" />
        <text className="th" x="300" y="565" textAnchor="middle" dominantBaseline="central">Dec 16x16</text>
        <text className="ts" x="300" y="583" textAnchor="middle" dominantBaseline="central">frozen</text>
      </g>
      <g className="n-gray">
        <rect x="235" y="620" width="130" height="44" rx="8" strokeWidth="0.5" />
        <text className="th" x="300" y="635" textAnchor="middle" dominantBaseline="central">Dec 32x32</text>
        <text className="ts" x="300" y="653" textAnchor="middle" dominantBaseline="central">frozen</text>
      </g>
      <g className="n-gray">
        <rect x="235" y="690" width="130" height="44" rx="8" strokeWidth="0.5" />
        <text className="th" x="300" y="705" textAnchor="middle" dominantBaseline="central">Dec 64x64</text>
        <text className="ts" x="300" y="723" textAnchor="middle" dominantBaseline="central">frozen</text>
      </g>

      {/* Decoder vertical arrows */}
      <line x1="300" y1="524" x2="300" y2="546" className="arr" markerEnd="url(#cna-arrow)" />
      <line x1="300" y1="594" x2="300" y2="616" className="arr" markerEnd="url(#cna-arrow)" />
      <line x1="300" y1="664" x2="300" y2="686" className="arr" markerEnd="url(#cna-arrow)" />

      {/* ===== Merge (+) circles ===== */}
      {/* Merge at Dec 16x16 level */}
      <circle cx="218" cy="572" r="12" className="merge-circle" strokeWidth="1.5" />
      <text className="merge-text" x="218" y="572" textAnchor="middle" dominantBaseline="central">+</text>
      <line x1="230" y1="572" x2="233" y2="572" className="merge-link" />

      {/* Merge at Dec 32x32 level */}
      <circle cx="218" cy="642" r="12" className="merge-circle" strokeWidth="1.5" />
      <text className="merge-text" x="218" y="642" textAnchor="middle" dominantBaseline="central">+</text>
      <line x1="230" y1="642" x2="233" y2="642" className="merge-link" />

      {/* Merge at Dec 64x64 level */}
      <circle cx="218" cy="712" r="12" className="merge-circle" strokeWidth="1.5" />
      <text className="merge-text" x="218" y="712" textAnchor="middle" dominantBaseline="central">+</text>
      <line x1="230" y1="712" x2="233" y2="712" className="merge-link" />

      {/* ===== Skip connections (dashed, encoder to merge nodes) ===== */}
      <path
        d="M 40 300 L 20 300 L 20 572 L 206 572"
        className="skip" markerEnd="url(#cna-arrow-skip)"
      />
      <path
        d="M 40 230 L 10 230 L 10 642 L 206 642"
        className="skip" markerEnd="url(#cna-arrow-skip)"
      />
      <path
        d="M 40 160 L 0 160 L 0 712 L 206 712"
        className="skip" markerEnd="url(#cna-arrow-skip)" opacity="0.7"
      />

      {/* ===== ControlNet encoder copy (Col C, purple) ===== */}
      <g className="n-purple">
        <rect x="500" y="130" width="130" height="44" rx="8" strokeWidth="0.5" />
        <text className="th" x="565" y="145" textAnchor="middle" dominantBaseline="central">Copy 64x64</text>
        <text className="ts" x="565" y="163" textAnchor="middle" dominantBaseline="central">trainable</text>
      </g>
      <g className="n-purple">
        <rect x="500" y="200" width="130" height="44" rx="8" strokeWidth="0.5" />
        <text className="th" x="565" y="215" textAnchor="middle" dominantBaseline="central">Copy 32x32</text>
        <text className="ts" x="565" y="233" textAnchor="middle" dominantBaseline="central">trainable</text>
      </g>
      <g className="n-purple">
        <rect x="500" y="270" width="130" height="44" rx="8" strokeWidth="0.5" />
        <text className="th" x="565" y="285" textAnchor="middle" dominantBaseline="central">Copy 16x16</text>
        <text className="ts" x="565" y="303" textAnchor="middle" dominantBaseline="central">trainable</text>
      </g>
      <g className="n-purple">
        <rect x="500" y="340" width="130" height="44" rx="8" strokeWidth="0.5" />
        <text className="th" x="565" y="355" textAnchor="middle" dominantBaseline="central">Copy 8x8</text>
        <text className="ts" x="565" y="373" textAnchor="middle" dominantBaseline="central">trainable</text>
      </g>

      {/* ControlNet vertical arrows */}
      <line x1="565" y1="174" x2="565" y2="196" className="arr" markerEnd="url(#cna-arrow)" />
      <line x1="565" y1="244" x2="565" y2="266" className="arr" markerEnd="url(#cna-arrow)" />
      <line x1="565" y1="314" x2="565" y2="336" className="arr" markerEnd="url(#cna-arrow)" />

      {/* ===== Zero convolution nodes (teal) ===== */}
      {/* ZC: Copy 8x8 -> Dec 8x8 */}
      <g className="n-teal">
        <rect x="410" y="352" width="80" height="30" rx="6" strokeWidth="0.5" />
        <text className="th" x="450" y="367" textAnchor="middle" dominantBaseline="central" style={{ fontSize: 12 }}>zero conv</text>
      </g>
      <line x1="500" y1="362" x2="494" y2="362" className="zc-stub" markerStart="url(#cna-arrow)" />
      <path d="M 410 367 L 390 367 L 390 502 L 367 502" className="zc-line" markerEnd="url(#cna-arrow)" />

      {/* ZC: Copy 16x16 -> merge at Dec 16x16 */}
      <g className="n-teal">
        <rect x="410" y="282" width="80" height="30" rx="6" strokeWidth="0.5" />
        <text className="th" x="450" y="297" textAnchor="middle" dominantBaseline="central" style={{ fontSize: 12 }}>zero conv</text>
      </g>
      <line x1="500" y1="292" x2="494" y2="292" className="zc-stub" markerStart="url(#cna-arrow)" />
      <path d="M 410 297 L 380 297 L 380 572 L 232 572" className="zc-line" markerEnd="url(#cna-arrow)" />

      {/* ZC: Copy 32x32 -> merge at Dec 32x32 */}
      <g className="n-teal">
        <rect x="410" y="212" width="80" height="30" rx="6" strokeWidth="0.5" />
        <text className="th" x="450" y="227" textAnchor="middle" dominantBaseline="central" style={{ fontSize: 12 }}>zero conv</text>
      </g>
      <line x1="500" y1="222" x2="494" y2="222" className="zc-stub" markerStart="url(#cna-arrow)" />
      <path d="M 410 227 L 370 227 L 370 642 L 232 642" className="zc-line" markerEnd="url(#cna-arrow)" />

      {/* ZC: Copy 64x64 -> merge at Dec 64x64 */}
      <g className="n-teal">
        <rect x="410" y="142" width="80" height="30" rx="6" strokeWidth="0.5" />
        <text className="th" x="450" y="157" textAnchor="middle" dominantBaseline="central" style={{ fontSize: 12 }}>zero conv</text>
      </g>
      <line x1="500" y1="152" x2="494" y2="152" className="zc-stub" markerStart="url(#cna-arrow)" />
      <path d="M 410 157 L 360 157 L 360 712 L 232 712" className="zc-line" markerEnd="url(#cna-arrow)" />
    </svg>
  )
}

export function DdpmTrainingFlowDiagram() {
  return (
    <svg
      data-dtfd=""
      xmlns="http://www.w3.org/2000/svg"
      width="100%"
      viewBox="0 0 680 724"
      role="img"
      aria-label="DDPM training flow: three inputs (clean image, random timestep, noise) converge through a closed-form formula into a noisy image, which the neural network denoises. MSE loss compares predicted and true noise, then backpropagation updates weights."
    >
      <defs>
        <style>{`
          [data-dtfd] text {
            font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
          }
          [data-dtfd] .th { font-size: 14px; font-weight: 500; }
          [data-dtfd] .ts { font-size: 12px; font-weight: 400; }
          [data-dtfd] .arr { stroke: #9c9a92; stroke-width: 1.5; }
          [data-dtfd] .ah { stroke: #9c9a92; stroke-width: 1.5; stroke-linecap: round; stroke-linejoin: round; }

          [data-dtfd] .n-blue rect { fill: #E6F1FB; stroke: #185FA5; }
          [data-dtfd] .n-blue .th { fill: #0C447C; }
          [data-dtfd] .n-blue .ts { fill: #185FA5; }

          [data-dtfd] .n-amber rect { fill: #FAEEDA; stroke: #854F0B; }
          [data-dtfd] .n-amber .th { fill: #633806; }
          [data-dtfd] .n-amber .ts { fill: #854F0B; }

          [data-dtfd] .n-teal rect { fill: #E1F5EE; stroke: #0F6E56; }
          [data-dtfd] .n-teal .th { fill: #085041; }
          [data-dtfd] .n-teal .ts { fill: #0F6E56; }

          [data-dtfd] .n-gray rect { fill: #F1EFE8; stroke: #5F5E5A; }
          [data-dtfd] .n-gray .th { fill: #444441; }
          [data-dtfd] .n-gray .ts { fill: #5F5E5A; }

          [data-dtfd] .n-purple rect { fill: #EEEDFE; stroke: #534AB7; }
          [data-dtfd] .n-purple .th { fill: #3C3489; }
          [data-dtfd] .n-purple .ts { fill: #534AB7; }

          [data-dtfd] .n-coral rect { fill: #FAECE7; stroke: #993C1D; }
          [data-dtfd] .n-coral .th { fill: #712B13; }
          [data-dtfd] .n-coral .ts { fill: #993C1D; }

          [data-dtfd] .side-amber { stroke: #BA7517; stroke-width: 1.5; }
          [data-dtfd] .side-teal { stroke: #1D9E75; stroke-width: 1.5; }

          @media (prefers-color-scheme: dark) {
            [data-dtfd] .arr { stroke: #6b6b65; }
            [data-dtfd] .ah { stroke: #6b6b65; }

            [data-dtfd] .n-blue rect { fill: #042C53; stroke: #85B7EB; }
            [data-dtfd] .n-blue .th { fill: #B5D4F4; }
            [data-dtfd] .n-blue .ts { fill: #85B7EB; }

            [data-dtfd] .n-amber rect { fill: #412402; stroke: #EF9F27; }
            [data-dtfd] .n-amber .th { fill: #FAC775; }
            [data-dtfd] .n-amber .ts { fill: #EF9F27; }

            [data-dtfd] .n-teal rect { fill: #04342C; stroke: #5DCAA5; }
            [data-dtfd] .n-teal .th { fill: #9FE1CB; }
            [data-dtfd] .n-teal .ts { fill: #5DCAA5; }

            [data-dtfd] .n-gray rect { fill: #2C2C2A; stroke: #B4B2A9; }
            [data-dtfd] .n-gray .th { fill: #D3D1C7; }
            [data-dtfd] .n-gray .ts { fill: #B4B2A9; }

            [data-dtfd] .n-purple rect { fill: #26215C; stroke: #AFA9EC; }
            [data-dtfd] .n-purple .th { fill: #CECBF6; }
            [data-dtfd] .n-purple .ts { fill: #AFA9EC; }

            [data-dtfd] .n-coral rect { fill: #4A1B0C; stroke: #F0997B; }
            [data-dtfd] .n-coral .th { fill: #F5C4B3; }
            [data-dtfd] .n-coral .ts { fill: #F0997B; }

            [data-dtfd] .side-amber { stroke: #EF9F27; }
            [data-dtfd] .side-teal { stroke: #5DCAA5; }
          }
        `}</style>
        <marker
          id="dtfd-arrow"
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
          id="dtfd-arrow-amber"
          viewBox="0 0 10 10"
          refX="8"
          refY="5"
          markerWidth="6"
          markerHeight="6"
          orient="auto-start-reverse"
        >
          <path d="M2 1L8 5L2 9" fill="none" stroke="#BA7517" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
        </marker>
        <marker
          id="dtfd-arrow-teal"
          viewBox="0 0 10 10"
          refX="8"
          refY="5"
          markerWidth="6"
          markerHeight="6"
          orient="auto-start-reverse"
        >
          <path d="M2 1L8 5L2 9" fill="none" stroke="#1D9E75" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
        </marker>
      </defs>

      {/* ===== ROW 1 (y=40, h=56): Three inputs ===== */}

      {/* x₀ clean image (blue) */}
      <g className="n-blue">
        <rect x="50" y="40" width="170" height="56" rx="8" strokeWidth="0.5" />
        <text className="th" x="135" y="58" textAnchor="middle" dominantBaseline="central">
          x&#x2080; clean image
        </text>
        <text className="ts" x="135" y="78" textAnchor="middle" dominantBaseline="central">
          Training sample
        </text>
      </g>

      {/* t ~ Uniform(1,T) (amber) */}
      <g className="n-amber">
        <rect x="248" y="40" width="184" height="56" rx="8" strokeWidth="0.5" />
        <text className="th" x="340" y="58" textAnchor="middle" dominantBaseline="central">
          t ~ Uniform(1,T)
        </text>
        <text className="ts" x="340" y="78" textAnchor="middle" dominantBaseline="central">
          Random timestep
        </text>
      </g>

      {/* ε ~ N(0,I) (teal) */}
      <g className="n-teal">
        <rect x="460" y="40" width="170" height="56" rx="8" strokeWidth="0.5" />
        <text className="th" x="545" y="58" textAnchor="middle" dominantBaseline="central">
          &#x03B5; ~ N(0,I)
        </text>
        <text className="ts" x="545" y="78" textAnchor="middle" dominantBaseline="central">
          Pure noise sample
        </text>
      </g>

      {/* ===== ROW 2 (y=148, h=44): Closed-form formula ===== */}

      <g className="n-gray">
        <rect x="190" y="148" width="260" height="44" rx="8" strokeWidth="0.5" />
        <text className="th" x="320" y="170" textAnchor="middle" dominantBaseline="central">
          Closed-form formula
        </text>
      </g>

      {/* ===== ROW 3 (y=244, h=44): Noisy image ===== */}

      <g className="n-gray">
        <rect x="220" y="244" width="200" height="44" rx="8" strokeWidth="0.5" />
        <text className="th" x="320" y="266" textAnchor="middle" dominantBaseline="central">
          x&#x209C; noisy image
        </text>
      </g>

      {/* ===== ROW 4 (y=340, h=56): Neural network ===== */}

      <g className="n-purple">
        <rect x="210" y="340" width="220" height="56" rx="8" strokeWidth="0.5" />
        <text className="th" x="320" y="358" textAnchor="middle" dominantBaseline="central">
          Neural network
        </text>
        <text className="ts" x="320" y="378" textAnchor="middle" dominantBaseline="central">
          Predicts the noise
        </text>
      </g>

      {/* ===== ROW 5 (y=448, h=44): Predicted noise εθ ===== */}

      <g className="n-purple">
        <rect x="195" y="448" width="250" height="44" rx="8" strokeWidth="0.5" />
        <text className="th" x="320" y="470" textAnchor="middle" dominantBaseline="central">
          Predicted noise &#x03B5;&#x03B8;
        </text>
      </g>

      {/* ===== ROW 6 (y=544, h=44): MSE loss ===== */}

      <g className="n-coral">
        <rect x="250" y="544" width="140" height="44" rx="8" strokeWidth="0.5" />
        <text className="th" x="320" y="566" textAnchor="middle" dominantBaseline="central">
          MSE loss
        </text>
      </g>

      {/* ===== ROW 7 (y=640, h=44): Backprop + update ===== */}

      <g className="n-coral">
        <rect x="218" y="640" width="204" height="44" rx="8" strokeWidth="0.5" />
        <text className="th" x="320" y="662" textAnchor="middle" dominantBaseline="central">
          Backprop + update
        </text>
      </g>

      {/* ===== ARROWS: Main vertical flow ===== */}

      {/* x₀ (cx=135, bottom=96) → closed-form (top=148): L-path via y=122 */}
      <path d="M135 96 L135 122 L260 122 L260 148" fill="none" className="arr" markerEnd="url(#dtfd-arrow)" />

      {/* t (cx=340, bottom=96) → closed-form (cx=320, top=148) */}
      <line x1="340" y1="96" x2="320" y2="138" className="arr" markerEnd="url(#dtfd-arrow)" />

      {/* ε (cx=545, bottom=96) → closed-form (top=148): L-path via y=122 */}
      <path d="M545 96 L545 122 L380 122 L380 148" fill="none" className="arr" markerEnd="url(#dtfd-arrow)" />

      {/* closed-form (cx=320, bottom=192) → xₜ (cx=320, top=244) */}
      <line x1="320" y1="192" x2="320" y2="234" className="arr" markerEnd="url(#dtfd-arrow)" />

      {/* xₜ (cx=320, bottom=288) → neural network (cx=320, top=340) */}
      <line x1="320" y1="288" x2="320" y2="330" className="arr" markerEnd="url(#dtfd-arrow)" />

      {/* neural network (cx=320, bottom=396) → εθ (cx=320, top=448) */}
      <line x1="320" y1="396" x2="320" y2="438" className="arr" markerEnd="url(#dtfd-arrow)" />

      {/* εθ (cx=320, bottom=492) → MSE loss (cx=320, top=544) */}
      <line x1="320" y1="492" x2="320" y2="534" className="arr" markerEnd="url(#dtfd-arrow)" />

      {/* MSE loss (cx=320, bottom=588) → backprop (cx=320, top=640) */}
      <line x1="320" y1="588" x2="320" y2="630" className="arr" markerEnd="url(#dtfd-arrow)" />

      {/* ===== ARROWS: Side routes (secondary connections) ===== */}

      {/* t → neural network: right side route */}
      {/* From t right edge (x=432, y=68) → right to x=596 → down to y=368 → left to neural net right edge (x=430, y=368) */}
      <path
        d="M432 68 L596 68 L596 368 L440 368"
        fill="none"
        className="side-amber"
        markerEnd="url(#dtfd-arrow-amber)"
        opacity="0.6"
      />

      {/* ε → MSE loss: right side route */}
      {/* From ε right edge (x=630, y=68) → right to x=636 → down to y=566 → left to MSE right edge (x=390, y=566) */}
      <path
        d="M630 68 L636 68 L636 566 L400 566"
        fill="none"
        className="side-teal"
        markerEnd="url(#dtfd-arrow-teal)"
        opacity="0.6"
      />
    </svg>
  )
}

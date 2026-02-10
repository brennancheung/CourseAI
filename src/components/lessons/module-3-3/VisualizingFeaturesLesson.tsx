'use client'

import { LessonLayout } from '@/components/lessons/LessonLayout'
import { Row } from '@/components/layout/Row'
import {
  LessonHeader,
  ObjectiveBlock,
  ConstraintBlock,
  SectionHeader,
  InsightBlock,
  TipBlock,
  WarningBlock,
  ConceptBlock,
  SummaryBlock,
  GradientCard,
  ComparisonRow,
  PhaseCard,
  ReferencesBlock,
} from '@/components/lessons'
import { CodeBlock } from '@/components/common/CodeBlock'
import { ExternalLink } from 'lucide-react'
import 'katex/dist/katex.min.css'
import { InlineMath, BlockMath } from 'react-katex'

/**
 * Visualizing Features
 *
 * First lesson in Module 3.3 (Seeing What CNNs See).
 * Teaches the student to visualize and interpret what a trained CNN
 * has learned at three levels: raw filter weights, activation maps,
 * and Grad-CAM heatmaps.
 *
 * Core concepts at DEVELOPED:
 * - Filter visualization (viewing conv1 kernels as images)
 * - Activation map capture and interpretation via hooks
 * - Grad-CAM (gradient-weighted class activation mapping)
 *
 * Concepts at INTRODUCED:
 * - PyTorch hooks (register_forward_hook)
 *
 * Previous: Transfer Learning (module 3.2, lesson 3)
 * Next: Lesson 2 of Module 3.3 (fine-tuning + visualization as debugging)
 */

export function VisualizingFeaturesLesson() {
  return (
    <LessonLayout>
      {/* ================================================================
          1. Header
          ================================================================ */}
      <Row>
        <Row.Content>
          <LessonHeader
            title="Visualizing Features"
            description="Open the black box&mdash;see the filters a CNN learned, watch activations flow through layers, and use Grad-CAM to ask why the model made a specific prediction."
            category="Seeing What CNNs See"
          />
        </Row.Content>
      </Row>

      {/* ================================================================
          1. Context + Constraints
          ================================================================ */}
      <Row>
        <Row.Content>
          <ObjectiveBlock>
            Visualize and interpret what a trained CNN has learned using three
            techniques: filter visualization (what patterns does a layer look
            for?), activation maps (what did the layer find in this image?), and
            Grad-CAM (what in this image mattered for this prediction?).
          </ObjectiveBlock>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="What You Already Know">
            You have loaded pretrained ResNets, inspected their layers, and
            understand the feature hierarchy (edges, textures, parts, objects).
            You know{' '}
            <code className="text-xs bg-muted px-1 rounded">requires_grad</code>{' '}
            and <code className="text-xs bg-muted px-1 rounded">.backward()</code>{' '}
            from autograd. This lesson puts all of that together to make the
            invisible visible.
          </TipBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <ConstraintBlock
            title="Scope for This Lesson"
            items={[
              'Three visualization techniques: filter viz, activation maps, Grad-CAM',
              'PyTorch hooks for capturing intermediate activations',
              'Interpreting what each technique shows and what it does NOT show',
              'Limitations of visualization as a debugging tool',
              'NOT: Deep dream, activation maximization, or image generation',
              'NOT: Saliency maps via input gradients (pixel-level attribution)',
              'NOT: LIME, SHAP, or model-agnostic interpretability methods',
              'NOT: Adversarial examples or feature inversion',
            ]}
          />
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="Three Questions, Three Tools">
            Each technique answers a different question. By the end, you will
            know which tool to reach for depending on what you want to
            understand about a model.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          2. Hook: "You've been taking my word for it"
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title={`"You've Been Taking My Word for It"`}
            subtitle="Time to see the evidence"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Since What Convolutions Compute, you have been building a mental
              model on trust. Early layers detect edges. Middle layers detect
              textures and parts. Later layers detect high-level object
              features. You accepted this because it made theoretical
              sense&mdash;small receptive fields see local patterns, larger
              receptive fields see global structure.
            </p>
            <p className="text-muted-foreground">
              But you have never actually <strong>looked</strong>.
            </p>
            <p className="text-muted-foreground">
              In Transfer Learning, you froze a pretrained backbone and trusted
              that its features would transfer. In Architecture Evolution, you
              learned that deeper networks build richer hierarchies. In every
              case, the feature hierarchy was an assertion, not an observation.
            </p>
            <p className="text-muted-foreground">
              Today you will load the same ResNet-18 you used for transfer
              learning, crack it open, and see what it actually learned. You
              might find that the mental model was right. You might find
              surprises. Either way, you will have{' '}
              <strong>evidence instead of faith</strong>.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Why This Matters">
            Visualization is not just academic curiosity. When a production
            model makes a wrong prediction, these techniques are how you
            figure out <em>why</em>. They turn debugging from guesswork into
            investigation.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          3. Explain 1: Filter Visualization
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Technique 1: Filter Visualization"
            subtitle="What does this layer look for?"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The simplest way to peer inside a CNN: look directly at the
              learned filter weights. You already know that{' '}
              <code className="text-xs bg-muted px-1 rounded">model.conv1</code>{' '}
              is a <code className="text-xs bg-muted px-1 rounded">Conv2d(3, 64, kernel_size=7)</code>.
              That means 64 filters, each 7x7x3&mdash;three channels for RGB.
              Each filter is small enough to visualize as a tiny image.
            </p>
            <CodeBlock
              code={`import torchvision.models as models
from torchvision.models import ResNet18_Weights

model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.eval()

# Access first-layer filters: shape [64, 3, 7, 7]
filters = model.conv1.weight.data.clone()

# Normalize to [0, 1] for display (simplified: global min/max)
# The notebook normalizes per-filter for better contrast
filters -= filters.min()
filters /= filters.max()

# Display as an 8x8 grid of 7x7 RGB images
# (see notebook for the matplotlib code)`}
              language="python"
              filename="filter_viz.py"
            />
            <p className="text-muted-foreground">
              When you run this in the notebook, you will see 64 tiny images.
              They will not look like cats, dogs, or any recognizable
              object. Instead, you will see:
            </p>
            <ul className="list-disc list-inside text-muted-foreground space-y-1 ml-4">
              <li>
                <strong>Oriented edges</strong> at different angles&mdash;horizontal,
                vertical, diagonal
              </li>
              <li>
                <strong>Color gradients</strong>&mdash;transitions between
                complementary colors
              </li>
              <li>
                <strong>High-frequency patterns</strong>&mdash;alternating
                light/dark stripes
              </li>
            </ul>

            {/* Schematic illustration: typical conv1 filter patterns */}
            <div className="rounded-lg border bg-muted/30 p-4">
              <p className="text-xs text-muted-foreground mb-3 text-center font-medium">
                Schematic: Typical conv1 filter patterns (each cell represents a 7x7 filter)
              </p>
              <svg viewBox="0 0 400 110" className="w-full" style={{ maxHeight: 130 }}>
                {/* Row of 8 schematic filters */}
                {/* Filter 1: Vertical edge */}
                <rect x="8" y="8" width="40" height="40" rx="2" fill="#1a1a2e" stroke="#4a4a6a" strokeWidth="0.5" />
                <rect x="8" y="8" width="20" height="40" fill="#2a2a4e" />
                <rect x="28" y="8" width="20" height="40" fill="#6366f1" opacity="0.6" />
                <text x="28" y="60" textAnchor="middle" fill="#888" fontSize="7">vertical</text>

                {/* Filter 2: Horizontal edge */}
                <rect x="58" y="8" width="40" height="40" rx="2" fill="#1a1a2e" stroke="#4a4a6a" strokeWidth="0.5" />
                <rect x="58" y="8" width="40" height="20" fill="#2a2a4e" />
                <rect x="58" y="28" width="40" height="20" fill="#6366f1" opacity="0.6" />
                <text x="78" y="60" textAnchor="middle" fill="#888" fontSize="7">horizontal</text>

                {/* Filter 3: Diagonal (top-left to bottom-right) */}
                <rect x="108" y="8" width="40" height="40" rx="2" fill="#1a1a2e" stroke="#4a4a6a" strokeWidth="0.5" />
                <polygon points="108,8 148,8 108,48" fill="#2a2a4e" />
                <polygon points="148,8 148,48 108,48" fill="#6366f1" opacity="0.6" />
                <text x="128" y="60" textAnchor="middle" fill="#888" fontSize="7">diagonal</text>

                {/* Filter 4: Opposite diagonal */}
                <rect x="158" y="8" width="40" height="40" rx="2" fill="#1a1a2e" stroke="#4a4a6a" strokeWidth="0.5" />
                <polygon points="158,8 198,8 198,48" fill="#2a2a4e" />
                <polygon points="158,8 158,48 198,48" fill="#6366f1" opacity="0.6" />
                <text x="178" y="60" textAnchor="middle" fill="#888" fontSize="7">diagonal</text>

                {/* Filter 5: Color gradient (warm to cool) */}
                <rect x="208" y="8" width="40" height="40" rx="2" fill="#1a1a2e" stroke="#4a4a6a" strokeWidth="0.5" />
                <defs>
                  <linearGradient id="colorGrad1" x1="0" y1="0" x2="1" y2="0">
                    <stop offset="0%" stopColor="#f97316" stopOpacity="0.7" />
                    <stop offset="100%" stopColor="#3b82f6" stopOpacity="0.7" />
                  </linearGradient>
                </defs>
                <rect x="208" y="8" width="40" height="40" fill="url(#colorGrad1)" />
                <text x="228" y="60" textAnchor="middle" fill="#888" fontSize="7">color</text>

                {/* Filter 6: Horizontal stripes */}
                <rect x="258" y="8" width="40" height="40" rx="2" fill="#1a1a2e" stroke="#4a4a6a" strokeWidth="0.5" />
                {[0, 2, 4].map((i) => (
                  <rect key={`stripe-${i}`} x={258} y={8 + i * 7} width={40} height={4} fill="#6366f1" opacity="0.5" />
                ))}
                {[1, 3, 5].map((i) => (
                  <rect key={`dark-${i}`} x={258} y={8 + i * 7} width={40} height={4} fill="#2a2a4e" />
                ))}
                <text x="278" y="60" textAnchor="middle" fill="#888" fontSize="7">stripes</text>

                {/* Filter 7: Center-surround */}
                <rect x="308" y="8" width="40" height="40" rx="2" fill="#2a2a4e" stroke="#4a4a6a" strokeWidth="0.5" />
                <circle cx="328" cy="28" r="12" fill="#6366f1" opacity="0.5" />
                <text x="328" y="60" textAnchor="middle" fill="#888" fontSize="7">blob</text>

                {/* Filter 8: Color gradient vertical */}
                <rect x="358" y="8" width="40" height="40" rx="2" fill="#1a1a2e" stroke="#4a4a6a" strokeWidth="0.5" />
                <defs>
                  <linearGradient id="colorGrad2" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#22c55e" stopOpacity="0.7" />
                    <stop offset="100%" stopColor="#a855f7" stopOpacity="0.7" />
                  </linearGradient>
                </defs>
                <rect x="358" y="8" width="40" height="40" fill="url(#colorGrad2)" />
                <text x="378" y="60" textAnchor="middle" fill="#888" fontSize="7">color</text>

                {/* Label row */}
                <text x="200" y="80" textAnchor="middle" fill="#aaa" fontSize="9">
                  8 of 64 filters &mdash; actual conv1 kernels are 7x7 RGB
                </text>
                <text x="200" y="95" textAnchor="middle" fill="#666" fontSize="8">
                  No objects, no faces &mdash; just edges, gradients, and patterns
                </text>
              </svg>
            </div>

            <p className="text-muted-foreground">
              These are the &ldquo;questions&rdquo; the first layer asks at
              every 7x7 patch of the input image. &ldquo;Is there a vertical
              edge here? A horizontal gradient? A color boundary?&rdquo; This
              is exactly what you were told in What Convolutions
              Compute&mdash;but now you can see it.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Not Objects&mdash;Edges">
            A common expectation: if the network classifies objects, its
            filters should look like objects. But a 7x7 pixel patch cannot
            contain a dog. Object representations emerge only through many
            layers of composition. First-layer filters are always low-level
            feature detectors.
          </WarningBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              This confirms what you learned in Transfer Learning about feature
              transferability: conv1 filters are{' '}
              <strong>universal</strong>. Edge detectors and color gradient
              filters are useful whether the network classifies dogs, medical
              images, or satellite photos. A vertical edge is a vertical edge
              regardless of what it belongs to.
            </p>
            <GradientCard title="Limitation" color="amber">
              <p className="text-sm">
                Filter visualization only works well for{' '}
                <strong>conv1</strong>. Deeper layers operate on
                multi-channel feature maps, not RGB images. A filter in
                layer3 has shape [256, 128, 3, 3]&mdash;128 input channels.
                You cannot display 128 channels as a meaningful image. For
                deeper layers, we need different techniques.
              </p>
            </GradientCard>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Connection to Transfer Learning">
            In Transfer Learning, you froze conv1 because its features are
            &ldquo;universal.&rdquo; Now you can see why: these filters detect
            the same edges and gradients regardless of the task.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          4. Check 1: Predict and Verify
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Check: Predict Before You Look"
            subtitle="What would edge-detection filters produce?"
          />
          <GradientCard title="Predict and Verify" color="emerald">
            <div className="space-y-3 text-sm">
              <p>
                <strong>Question:</strong> You feed a photo of a cat through
                conv1 with these edge-detection filters. What would the
                activation maps look like? Where would they be bright? Where
                would they be dark?
              </p>
              <details>
                <summary className="font-medium cursor-pointer text-primary">
                  Reveal
                </summary>
                <div className="mt-2 space-y-2">
                  <p>
                    <strong>Bright</strong> at edges and boundaries in the
                    image&mdash;the outline of the cat, whisker edges, the
                    boundary between fur and background.{' '}
                    <strong>Dark</strong> in uniform regions&mdash;flat patches
                    of fur or solid-color background where there are no edges
                    to detect.
                  </p>
                  <p>
                    Each of the 64 activation maps will look different because
                    each filter detects a different type of edge or gradient.
                    A horizontal-edge filter lights up along horizontal
                    boundaries; a vertical-edge filter lights up along vertical
                    boundaries.
                  </p>
                </div>
              </details>
            </div>
          </GradientCard>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Activation Map = Feature Map">
            You already know this concept by another name. An activation map is
            a feature map for a specific input image. Same thing&mdash;the
            output of a convolutional layer when a particular image passes
            through.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          5. Brief Section: PyTorch Hooks
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="PyTorch Hooks"
            subtitle="Capturing intermediate outputs without modifying the model"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              You want to see what layer2 or layer4 produces for a specific
              image. But when you call{' '}
              <code className="text-xs bg-muted px-1 rounded">model(image)</code>,
              you only get the final output&mdash;the 1000 class scores. The
              intermediate activations are computed and then discarded.
            </p>
            <p className="text-muted-foreground">
              How do you capture them without rewriting the model? PyTorch
              provides <strong>hooks</strong>&mdash;callbacks you attach to any
              layer. A forward hook runs every time that layer completes its
              forward pass, and it receives the layer&apos;s output.
            </p>
            <CodeBlock
              code={`# Storage for the captured activation
activation = {}

def hook_fn(module, input, output):
    activation['layer2'] = output.detach()

# Attach the hook to layer2
hook = model.layer2.register_forward_hook(hook_fn)

# Run a forward pass — the hook fires automatically
output = model(image)

# Now activation['layer2'] contains the feature maps!
print(activation['layer2'].shape)  # [1, 128, 28, 28]

# Clean up — remove the hook when done
hook.remove()`}
              language="python"
              filename="hooks.py"
            />
            <p className="text-muted-foreground">
              Think of it as placing a sensor on a layer&mdash;it records what
              passes through without changing anything. The model is
              completely unchanged. Four lines: define the hook function,
              register it, run the forward pass, remove it when done.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="No Architecture Changes Needed">
            You might assume that inspecting intermediate layers requires
            modifying the model&mdash;splitting it, adding return statements,
            or building a new class. Hooks let you inspect any layer of any
            model without touching its code.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          6. Explain 2: Activation Maps
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Technique 2: Activation Maps"
            subtitle="What did this layer find in this image?"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Filter visualization answers &ldquo;what does this layer look
              for?&rdquo; Activation maps answer the complementary
              question: <strong>&ldquo;what did this layer find in THIS
              image?&rdquo;</strong> You feed a specific image through the
              network, capture the output of a layer using a hook, and display
              the resulting feature maps.
            </p>
            <p className="text-muted-foreground">
              In the notebook, you will capture activations at three depths and
              see the feature hierarchy unfold:
            </p>

            <PhaseCard number={1} title="conv1 Activations" subtitle="Spatial resolution: 112x112" color="cyan">
              <p className="text-sm">
                Sharp, spatially detailed responses. You can see exactly which
                edges and boundaries the layer detected. A horizontal-edge
                filter produces a bright activation everywhere there is a
                horizontal boundary in the input.
              </p>
            </PhaseCard>

            <PhaseCard number={2} title="layer2 Activations" subtitle="Spatial resolution: 28x28" color="blue">
              <p className="text-sm">
                Less spatially precise, more abstract. Individual activations
                respond to textures and local patterns rather than simple
                edges. The representation is beginning to encode
                &ldquo;what&rdquo; rather than &ldquo;where.&rdquo;
              </p>
            </PhaseCard>

            <PhaseCard number={3} title="layer4 Activations" subtitle="Spatial resolution: 7x7" color="violet">
              <p className="text-sm">
                Abstract spatial patterns. Individual channels are{' '}
                <strong>not recognizable as objects</strong>. The
                representation is distributed across many channels&mdash;no
                single channel encodes &ldquo;dog&rdquo; or &ldquo;car.&rdquo;
                But the pattern across all 512 channels is what the
                classification head uses to make its prediction.
              </p>
            </PhaseCard>

            {/* Schematic: activation maps at three depths */}
            <div className="rounded-lg border bg-muted/30 p-4">
              <p className="text-xs text-muted-foreground mb-3 text-center font-medium">
                Schematic: Activation maps at three depths for an image with a diagonal edge
              </p>
              <svg viewBox="0 0 440 150" className="w-full" style={{ maxHeight: 170 }}>
                {/* Input image schematic — a simple scene with a diagonal edge */}
                <rect x="10" y="10" width="80" height="80" rx="2" fill="#1a1a2e" stroke="#4a4a6a" strokeWidth="0.5" />
                <polygon points="10,10 90,10 90,90" fill="#3b3b5e" />
                <polygon points="10,10 10,90 90,90" fill="#1a1a2e" />
                <line x1="10" y1="90" x2="90" y2="10" stroke="#6366f1" strokeWidth="1.5" opacity="0.8" />
                <text x="50" y="105" textAnchor="middle" fill="#aaa" fontSize="8">Input</text>

                {/* Arrow */}
                <line x1="100" y1="50" x2="125" y2="50" stroke="#666" strokeWidth="1" markerEnd="url(#arrowGray)" />
                <defs>
                  <marker id="arrowGray" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="6" markerHeight="6" orient="auto">
                    <path d="M 0 0 L 10 5 L 0 10 z" fill="#666" />
                  </marker>
                </defs>

                {/* conv1 activation: sharp edge detection */}
                <rect x="135" y="10" width="80" height="80" rx="2" fill="#0f172a" stroke="#4a4a6a" strokeWidth="0.5" />
                {/* Bright pixels along the diagonal where the edge is */}
                {[0, 1, 2, 3, 4, 5, 6, 7].map((i) => (
                  <rect
                    key={`conv1-edge-${i}`}
                    x={135 + (i * 10)}
                    y={10 + (70 - i * 10)}
                    width={10}
                    height={10}
                    fill="#22d3ee"
                    opacity={0.7}
                  />
                ))}
                <text x="175" y="105" textAnchor="middle" fill="#22d3ee" fontSize="8" fontWeight="bold">conv1</text>
                <text x="175" y="116" textAnchor="middle" fill="#888" fontSize="7">112x112 &mdash; sharp edges</text>

                {/* Arrow */}
                <line x1="225" y1="50" x2="250" y2="50" stroke="#666" strokeWidth="1" markerEnd="url(#arrowGray)" />

                {/* layer2 activation: blurrier, textural */}
                <rect x="260" y="15" width="60" height="60" rx="2" fill="#0f172a" stroke="#4a4a6a" strokeWidth="0.5" />
                {/* Broader, fuzzier blobs */}
                <ellipse cx="280" cy="35" rx="15" ry="12" fill="#3b82f6" opacity="0.5" />
                <ellipse cx="300" cy="55" rx="12" ry="10" fill="#3b82f6" opacity="0.35" />
                <text x="290" y="93" textAnchor="middle" fill="#3b82f6" fontSize="8" fontWeight="bold">layer2</text>
                <text x="290" y="104" textAnchor="middle" fill="#888" fontSize="7">28x28 &mdash; textures</text>

                {/* Arrow */}
                <line x1="330" y1="45" x2="355" y2="45" stroke="#666" strokeWidth="1" markerEnd="url(#arrowGray)" />

                {/* layer4 activation: abstract blobs, very low resolution */}
                <rect x="365" y="20" width="50" height="50" rx="2" fill="#0f172a" stroke="#4a4a6a" strokeWidth="0.5" />
                {/* Abstract scattered blobs */}
                <rect x="370" y="25" width="14" height="14" fill="#8b5cf6" opacity="0.5" rx="2" />
                <rect x="395" y="40" width="14" height="14" fill="#8b5cf6" opacity="0.35" rx="2" />
                <rect x="380" y="48" width="10" height="10" fill="#8b5cf6" opacity="0.2" rx="2" />
                <text x="390" y="86" textAnchor="middle" fill="#8b5cf6" fontSize="8" fontWeight="bold">layer4</text>
                <text x="390" y="97" textAnchor="middle" fill="#888" fontSize="7">7x7 &mdash; abstract</text>

                {/* Bottom label */}
                <text x="220" y="140" textAnchor="middle" fill="#666" fontSize="8">
                  Spatial resolution shrinks, representations go from concrete edges to abstract patterns
                </text>
              </svg>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Deep Layers Are Abstract">
            You might expect that if early layers show edges, deeper layers
            should show increasingly recognizable objects&mdash;faces, wheels,
            paws. They do not. Layer4 activations are useful for classification
            but look like abstract blobs to human eyes. The object-level
            meaning is encoded in the <em>pattern</em> across channels, not in
            any single channel.
          </WarningBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              This is the &ldquo;zoom out&rdquo; progression you learned about
              in Building a CNN&mdash;made visible. Spatial resolution shrinks
              (112 to 28 to 7), channel count grows (64 to 128 to 512), and
              the nature of the representation shifts from concrete (edges) to
              abstract (distributed patterns). The feature hierarchy is real.
            </p>
            <ComparisonRow
              left={{
                title: 'Filter Visualization',
                color: 'cyan',
                items: [
                  'Shows what the layer LOOKS FOR',
                  'Model-level (same for all images)',
                  'Only useful for conv1 (RGB input)',
                ],
              }}
              right={{
                title: 'Activation Maps',
                color: 'blue',
                items: [
                  'Shows what the layer FOUND',
                  'Input-specific (different per image)',
                  'Works at any depth via hooks',
                ],
              }}
            />
          </div>
        </Row.Content>
      </Row>

      {/* ================================================================
          7. Check 2: Distinguish the Techniques
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Check: Activation Maps vs Classification"
            subtitle="Is finding features the same as knowing the class?"
          />
          <GradientCard title="Think About It" color="emerald">
            <div className="space-y-3 text-sm">
              <p>
                <strong>Question:</strong> An activation map for a
                horizontal-edge filter shows all horizontal edges in the
                image&mdash;the top of a fence, the horizon line, the edge of
                a table. Is this the same as knowing what class the image
                belongs to?
              </p>
              <details>
                <summary className="font-medium cursor-pointer text-primary">
                  Reveal
                </summary>
                <div className="mt-2 space-y-2">
                  <p>
                    No. Activation maps show <strong>feature
                    detection</strong>&mdash;what the layer found. They are
                    class-agnostic. Horizontal edges appear whether the image
                    is a dog, a cat, or a landscape. The activation map does
                    not know or care about the class label.
                  </p>
                  <p>
                    To understand what mattered for a{' '}
                    <em>specific prediction</em>, we need a technique that is
                    class-specific. That is Grad-CAM.
                  </p>
                </div>
              </details>
            </div>
          </GradientCard>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The Missing Piece">
            Filter visualization is model-level. Activation maps are
            input-specific. Neither is class-specific. We need one more tool
            to answer: &ldquo;what in this image mattered for{' '}
            <em>this</em> prediction?&rdquo;
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          8. Explain 3: Grad-CAM
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Technique 3: Grad-CAM"
            subtitle="What in this image mattered for this prediction?"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Activation maps tell you what each filter detected, but they are
              the same regardless of which class the model predicts. Grad-CAM
              (Gradient-weighted Class Activation Mapping) answers the most
              useful question in applied deep learning:{' '}
              <strong>
                which spatial regions of the input mattered for a specific
                class prediction?
              </strong>
            </p>
            <p className="text-muted-foreground">
              The intuition: ask the network to retrace its reasoning. &ldquo;You
              predicted golden retriever&mdash;which parts of the last feature
              maps were most important for that prediction?&rdquo; Grad-CAM
              answers this by flowing gradients backward from the class score
              to the final convolutional layer.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="Same Mechanism, Different Purpose">
            The backward pass in Grad-CAM is the same mechanism that trains the
            network. During training, gradients flow back to update weights.
            Here, gradients flow back to <em>explain</em> predictions. Same
            math, different use.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Here is how Grad-CAM works, step by step. Concrete first, then
              the formula:
            </p>
            <div className="py-4 px-6 bg-muted/50 rounded-lg space-y-3">
              <p className="text-sm font-medium text-muted-foreground">
                Grad-CAM in six steps:
              </p>
              <ol className="list-decimal list-inside text-sm text-muted-foreground space-y-2 ml-2">
                <li>
                  <strong>Forward pass:</strong> Run the image through the model.
                  Capture the last conv layer&apos;s activations (using a hook)
                  and the class score for the predicted class.
                </li>
                <li>
                  <strong>Backward pass:</strong> Compute gradients of that class
                  score with respect to the last conv layer&apos;s activation maps.
                  &ldquo;How much does each spatial position in each channel
                  affect the class score?&rdquo;
                </li>
                <li>
                  <strong>Global average pool the gradients:</strong> For each
                  channel, average its gradient over all spatial positions. This
                  gives one weight per channel&mdash;how important that channel
                  is for the predicted class.
                </li>
                <li>
                  <strong>Weighted sum:</strong> Multiply each activation map by
                  its weight and sum across channels. This produces a single
                  spatial heatmap.
                </li>
                <li>
                  <strong>ReLU:</strong> Keep only positive values. We want
                  regions that <em>support</em> the prediction, not regions that
                  argue against it.
                </li>
                <li>
                  <strong>Upsample and overlay:</strong> The heatmap is small
                  (7x7 for ResNet-18). Upsample to input size and overlay on the
                  original image.
                </li>
              </ol>
            </div>

            {/* Concrete worked example with small numbers */}
            <GradientCard title="Worked Example: Grad-CAM with 3 Channels" color="blue">
              <div className="text-sm space-y-3">
                <p>
                  Imagine a toy case: 3 channels at 2x2 spatial resolution
                  (instead of 512 channels at 7x7).
                </p>
                <p>
                  <strong>Step 3 &mdash; Gradient weights:</strong> Global average
                  pool the gradients for each channel:
                </p>
                <div className="font-mono text-xs bg-background/50 rounded px-3 py-2 space-y-1">
                  <p>Channel 1: avg gradient = <strong>0.8</strong> (important for this class)</p>
                  <p>Channel 2: avg gradient = <strong>0.1</strong> (barely matters)</p>
                  <p>Channel 3: avg gradient = <strong>0.5</strong> (moderately important)</p>
                </div>
                <p>
                  <strong>Step 4 &mdash; Weighted sum:</strong> Each channel&apos;s
                  activation map gets multiplied by its weight and summed:
                </p>
                <div className="font-mono text-xs bg-background/50 rounded px-3 py-2 space-y-1">
                  <p>Ch1 activations (top-left bright):  [[<strong>0.9</strong>, 0.1], [0.2, 0.0]]  x 0.8</p>
                  <p>Ch2 activations (uniform):           [[0.3, 0.3], [0.3, 0.3]]  x 0.1</p>
                  <p>Ch3 activations (bottom-right bright): [[0.0, 0.1], [0.2, <strong>0.8</strong>]]  x 0.5</p>
                  <p className="pt-1 border-t border-border/50">Weighted sum: [[<strong>0.75</strong>, 0.16], [0.29, <strong>0.40</strong>]]</p>
                </div>
                <p>
                  <strong>Step 5 &mdash; ReLU:</strong> All values are already
                  positive, so nothing changes. The heatmap says: &ldquo;top-left
                  matters most, bottom-right matters some.&rdquo;
                </p>
                <p className="text-muted-foreground">
                  In real Grad-CAM the same logic applies across 512 channels at
                  7x7 spatial resolution&mdash;but the principle is identical.
                </p>
              </div>
            </GradientCard>

            <p className="text-muted-foreground">
              The formula captures steps 3&ndash;5 concisely. The weight for
              channel <InlineMath math="k" /> is the global average of the
              gradient:
            </p>
            <div className="py-4 px-6 bg-muted/50 rounded-lg">
              <BlockMath math="\alpha_k = \frac{1}{Z} \sum_i \sum_j \frac{\partial y^c}{\partial A^k_{ij}}" />
            </div>
            <p className="text-muted-foreground">
              Where <InlineMath math="y^c" /> is the score for class{' '}
              <InlineMath math="c" />, <InlineMath math="A^k_{ij}" /> is the
              activation at spatial position <InlineMath math="(i,j)" /> in
              channel <InlineMath math="k" />, and{' '}
              <InlineMath math="Z" /> is the number of spatial positions. Then
              the Grad-CAM heatmap is:
            </p>
            <div className="py-4 px-6 bg-muted/50 rounded-lg">
              <BlockMath math="L_{\text{Grad-CAM}} = \text{ReLU}\!\left(\sum_k \alpha_k \cdot A^k\right)" />
            </div>
            <p className="text-muted-foreground">
              Notice the connection to global average pooling from ResNets and
              Skip Connections. The{' '}
              <InlineMath math="\alpha_k" /> computation is the same operation
              used in ResNet&apos;s classification head&mdash;but applied to
              gradients instead of activations.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Why the Last Conv Layer?">
            Grad-CAM uses the last convolutional layer because it has the best
            balance: rich semantic information (it encodes high-level concepts)
            combined with spatial information (it still has a 7x7 grid). The
            fully connected layers after it have lost all spatial structure.
          </TipBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              In the notebook, you will implement Grad-CAM step by step. When
              you overlay the heatmap on a photo of a dog in a park, the result
              is striking: the heatmap highlights the dog, not the trees or
              grass. The model is focusing on the right region for the
              prediction.
            </p>

            {/* Schematic: Grad-CAM heatmap overlay */}
            <div className="rounded-lg border bg-muted/30 p-4">
              <p className="text-xs text-muted-foreground mb-3 text-center font-medium">
                Schematic: Grad-CAM heatmap highlights the class-relevant region
              </p>
              <svg viewBox="0 0 420 140" className="w-full" style={{ maxHeight: 160 }}>
                {/* Scene: dog in a park (simplified) */}
                <rect x="10" y="10" width="110" height="90" rx="3" fill="#1a3a1a" stroke="#4a4a6a" strokeWidth="0.5" />
                {/* Sky */}
                <rect x="10" y="10" width="110" height="30" rx="3" fill="#1a2a4a" />
                {/* Ground */}
                <rect x="10" y="70" width="110" height="30" rx="0" fill="#2a4a2a" />
                {/* Dog silhouette */}
                <ellipse cx="65" cy="55" rx="22" ry="18" fill="#8b6914" opacity="0.8" />
                <circle cx="55" cy="42" r="8" fill="#8b6914" opacity="0.8" />
                {/* Bench */}
                <rect x="90" y="55" width="25" height="4" fill="#5a3a1a" />
                <rect x="92" y="59" width="3" height="12" fill="#5a3a1a" />
                <rect x="110" y="59" width="3" height="12" fill="#5a3a1a" />
                <text x="65" y="115" textAnchor="middle" fill="#aaa" fontSize="8">Original image</text>

                {/* Arrow */}
                <text x="150" y="55" textAnchor="middle" fill="#888" fontSize="9">Grad-CAM</text>
                <line x1="135" y1="60" x2="175" y2="60" stroke="#666" strokeWidth="1" markerEnd="url(#arrowGray)" />
                <text x="155" y="73" textAnchor="middle" fill="#888" fontSize="7">&ldquo;golden retriever&rdquo;</text>

                {/* Grad-CAM result: dog highlighted */}
                <rect x="185" y="10" width="110" height="90" rx="3" fill="#1a3a1a" stroke="#4a4a6a" strokeWidth="0.5" />
                <rect x="185" y="10" width="110" height="30" rx="3" fill="#1a2a4a" />
                <rect x="185" y="70" width="110" height="30" fill="#2a4a2a" />
                {/* Dog silhouette */}
                <ellipse cx="240" cy="55" rx="22" ry="18" fill="#8b6914" opacity="0.8" />
                <circle cx="230" cy="42" r="8" fill="#8b6914" opacity="0.8" />
                {/* Bench */}
                <rect x="265" y="55" width="25" height="4" fill="#5a3a1a" />
                <rect x="267" y="59" width="3" height="12" fill="#5a3a1a" />
                <rect x="285" y="59" width="3" height="12" fill="#5a3a1a" />
                {/* Heatmap overlay on dog region — warm red/orange glow */}
                <ellipse cx="237" cy="52" rx="30" ry="25" fill="#ef4444" opacity="0.35" />
                <ellipse cx="237" cy="52" rx="20" ry="17" fill="#f97316" opacity="0.35" />
                <ellipse cx="237" cy="52" rx="10" ry="9" fill="#fbbf24" opacity="0.3" />
                <text x="240" y="115" textAnchor="middle" fill="#aaa" fontSize="8">Highlights the dog</text>

                {/* Second Grad-CAM with different class */}
                <text x="325" y="55" textAnchor="middle" fill="#888" fontSize="9">Grad-CAM</text>
                <text x="330" y="73" textAnchor="middle" fill="#888" fontSize="7">&ldquo;park bench&rdquo;</text>

                {/* Result: bench highlighted */}
                <rect x="345" y="30" width="65" height="60" rx="3" fill="#0f172a" stroke="#4a4a6a" strokeWidth="0.5" />
                {/* Simplified scene */}
                <ellipse cx="365" cy="55" rx="10" ry="8" fill="#8b6914" opacity="0.4" />
                {/* Bench */}
                <rect x="380" y="55" width="20" height="3" fill="#5a3a1a" />
                <rect x="382" y="58" width="2" height="8" fill="#5a3a1a" />
                <rect x="396" y="58" width="2" height="8" fill="#5a3a1a" />
                {/* Heatmap on bench region */}
                <ellipse cx="390" cy="57" rx="18" ry="12" fill="#ef4444" opacity="0.35" />
                <ellipse cx="390" cy="57" rx="10" ry="7" fill="#fbbf24" opacity="0.3" />
                <text x="378" y="115" textAnchor="middle" fill="#aaa" fontSize="8">Highlights the bench</text>

                {/* Bottom label */}
                <text x="210" y="135" textAnchor="middle" fill="#666" fontSize="8">
                  Same image, different class target &rarr; different spatial focus
                </text>
              </svg>
            </div>

            <CodeBlock
              code={`# Simplified Grad-CAM (you'll implement this in the notebook)

# 1. Hook to capture activations AND gradients
activations = {}
gradients = {}

def forward_hook(module, input, output):
    activations['value'] = output

def backward_hook(module, grad_input, grad_output):
    gradients['value'] = grad_output[0]

# 2. Register hooks on the last conv layer
fhook = model.layer4.register_forward_hook(forward_hook)
# register_full_backward_hook is the modern API (register_backward_hook is deprecated)
bhook = model.layer4.register_full_backward_hook(backward_hook)

# 3. Forward pass
output = model(image)
class_score = output[0, predicted_class]

# 4. Backward pass
model.zero_grad()
class_score.backward()

# 5. Compute Grad-CAM
weights = gradients['value'].mean(dim=[2, 3])  # global avg pool
cam = (weights.unsqueeze(-1).unsqueeze(-1) * activations['value']).sum(dim=1)
cam = torch.relu(cam)  # keep positive only
cam = cam / cam.max()  # normalize to [0, 1]

# 6. Upsample and overlay (see notebook for display code)
fhook.remove()
bhook.remove()`}
              language="python"
              filename="gradcam.py"
            />
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Class-Specific Attention">
            For the same image, Grad-CAM produces different heatmaps for
            different classes. Ask for &ldquo;golden retriever&rdquo; and it
            highlights the dog. Ask for &ldquo;park bench&rdquo; and it shifts
            to the bench. Same image, different focus&mdash;because the
            gradients change.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          9. Check 3: Transfer Question
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Check: Class-Specific Heatmaps"
            subtitle="Same image, different question"
          />
          <GradientCard title="Transfer Question" color="emerald">
            <div className="space-y-3 text-sm">
              <p>
                <strong>Question:</strong> You run Grad-CAM on an image of a
                dog sitting on a park bench. The model&apos;s top prediction is
                &ldquo;golden retriever&rdquo; and the second prediction is
                &ldquo;park bench.&rdquo; What would the Grad-CAM heatmap
                look like for &ldquo;park bench&rdquo; instead of &ldquo;golden
                retriever&rdquo;?
              </p>
              <details>
                <summary className="font-medium cursor-pointer text-primary">
                  Reveal
                </summary>
                <div className="mt-2 space-y-2">
                  <p>
                    The heatmap shifts from the dog to the bench. Grad-CAM is
                    class-specific&mdash;different class scores produce different
                    gradients, which produce different channel weights, which
                    highlight different spatial regions. Same image, different
                    focus.
                  </p>
                  <p>
                    You will try this in the notebook and see the shift
                    yourself.
                  </p>
                </div>
              </details>
            </div>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* Activation Maps vs Grad-CAM comparison */}
      <Row>
        <Row.Content>
          <ComparisonRow
            left={{
              title: 'Activation Maps',
              color: 'blue',
              items: [
                'Shows what the layer FOUND in this image',
                'Same for all classes (class-agnostic)',
                'Highlights all detected features (edges, textures)',
                'Answers: "what did the layer find?"',
              ],
            }}
            right={{
              title: 'Grad-CAM',
              color: 'violet',
              items: [
                'Shows what mattered for THIS prediction',
                'Different for each class (class-specific)',
                'Highlights only features relevant to one class',
                'Answers: "what mattered for this prediction?"',
              ],
            }}
          />
        </Row.Content>
      </Row>

      {/* ================================================================
          10. Elaborate: Limitations and Shortcut Learning
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Limitations and Shortcut Learning"
            subtitle="When correct predictions hide broken reasoning"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Grad-CAM is powerful, but it has important limitations:
            </p>
            <ul className="list-disc list-inside text-muted-foreground space-y-2 ml-4">
              <li>
                <strong>Resolution:</strong> The heatmap is limited by the
                spatial resolution of the last conv layer. For ResNet-18, that
                is 7x7&mdash;the heatmap is coarse. It tells you the general
                region, not the exact pixels.
              </li>
              <li>
                <strong>Positive evidence only:</strong> The ReLU step keeps
                only regions that <em>support</em> the prediction. Grad-CAM
                does not show regions that argue <em>against</em> the predicted
                class.
              </li>
              <li>
                <strong>Not causal:</strong> Grad-CAM shows correlation between
                spatial regions and the class score. It does not prove that the
                model &ldquo;understands&rdquo; the object in any human sense.
              </li>
            </ul>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Resolution Limit">
            Grad-CAM highlights a region, not individual pixels. For
            ResNet-18, the 7x7 spatial resolution means each cell in the
            heatmap covers a 32x32 patch of the 224x224 input. Good enough
            to identify which object matters, too coarse for pixel-level
            attribution.
          </TipBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <div className="space-y-4">
            <GradientCard title="The Shortcut Learning Trap" color="rose">
              <div className="text-sm space-y-3">
                <p>
                  A model trained to classify &ldquo;wolf&rdquo; vs
                  &ldquo;husky&rdquo; achieves 90% accuracy on the test set.
                  Impressive&mdash;until you apply Grad-CAM.
                </p>
                <p>
                  The heatmap reveals that the model is not looking at the
                  animal at all. It is focusing on the{' '}
                  <strong>background</strong>: snow for &ldquo;husky&rdquo;
                  and forest for &ldquo;wolf.&rdquo; The training data
                  happened to contain huskies mostly in snowy scenes and
                  wolves mostly in forests. The model learned the easiest
                  shortcut, not the concept.
                </p>
                <p>
                  The accuracy was real. The reasoning was broken. This is
                  a known failure mode called <strong>shortcut learning</strong>,
                  and it is far more common than you might think.
                </p>
              </div>
            </GradientCard>

            <p className="text-muted-foreground">
              This is the most practically important insight of this lesson:{' '}
              <strong>
                correct prediction does not mean correct reasoning
              </strong>
              . A model can get the right answer for the wrong reasons.
              Visualization is not just a pretty picture&mdash;it is a
              debugging tool that reveals whether the model learned what you
              intended.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Right Answer, Wrong Reason">
            High accuracy on a biased dataset can mask broken reasoning. The
            husky/wolf example is real. Without visualization, you would
            deploy a &ldquo;snow detector&rdquo; thinking it was an animal
            classifier. Grad-CAM catches this before deployment.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          11. Practice: Notebook
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Practice: See It Yourself"
            subtitle="Run every visualization on real images"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The notebook walks you through all three techniques on a
              pretrained ResNet-18 with real images. Boilerplate for image
              loading, preprocessing, and display is provided&mdash;your job
              is to write the visualization code.
            </p>
            <div className="rounded-lg border-2 border-primary/50 bg-primary/5 p-6">
              <div className="space-y-4">
                <p className="text-muted-foreground">
                  Open the scaffolded notebook in Google Colab. You will:
                </p>
                <ul className="list-disc list-inside text-muted-foreground space-y-1 ml-4 text-sm">
                  <li>Visualize all 64 conv1 filters as a grid</li>
                  <li>Register hooks and capture activation maps at three depths</li>
                  <li>Implement Grad-CAM step by step</li>
                  <li>Compare Grad-CAM heatmaps for different classes on the same image</li>
                  <li>Investigate a shortcut learning example</li>
                </ul>
                <a
                  href="https://colab.research.google.com/github/brennancheung/CourseAI/blob/main/notebooks/3-3-1-visualizing-features.ipynb"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-2 px-4 py-2 rounded-md bg-primary text-primary-foreground font-medium text-sm hover:bg-primary/90 transition-colors"
                >
                  <ExternalLink className="w-4 h-4" />
                  Open in Google Colab
                </a>
                <p className="text-xs text-muted-foreground">
                  The notebook includes image loading, preprocessing, and display
                  utilities. Expected time: 20&ndash;30 minutes on a Colab GPU.
                </p>
              </div>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="What&apos;s Provided">
            <ul className="space-y-1 text-sm">
              <li>Pretrained model loading and image preprocessing</li>
              <li>Display utilities (grid plots, heatmap overlay)</li>
              <li>Scaffolded TODO sections for each technique</li>
              <li>Guided experiments with specific images</li>
            </ul>
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          12. Summary: Three Questions, Three Tools
          ================================================================ */}
      <Row>
        <Row.Content>
          <SummaryBlock
            title="Three Questions, Three Tools"
            items={[
              {
                headline:
                  'Filter visualization: "What does this layer look for?"',
                description:
                  'Display the learned weights as images. Works for conv1 (RGB input). Shows edge detectors, color gradients, and frequency patterns—universal features that confirm why transfer learning works.',
              },
              {
                headline:
                  'Activation maps: "What did this layer find in this image?"',
                description:
                  'Capture intermediate outputs using hooks. Input-specific but class-agnostic. Shows the feature hierarchy in action—from concrete edges to abstract distributed patterns.',
              },
              {
                headline:
                  'Grad-CAM: "What in this image mattered for this prediction?"',
                description:
                  'Flow gradients from the class score back to the last conv layer. Input-specific AND class-specific. The most practical tool for debugging model behavior.',
              },
              {
                headline:
                  'Visualization is a debugging tool, not just a pretty picture.',
                description:
                  'Correct predictions can hide broken reasoning. Shortcut learning is real and common. Grad-CAM reveals whether the model learned what you intended—or found a shortcut.',
              },
            ]}
          />
        </Row.Content>
      </Row>

      {/* ================================================================
          13. Next Step
          ================================================================ */}
      <Row>
        <Row.Content>
          <InsightBlock title="The Feature Hierarchy Is Real">
            You have now seen it with your own eyes: early layers detect edges
            and gradients, middle layers detect textures and parts, and deep
            layers encode abstract representations. The mental model you built
            across Building a CNN, Architecture Evolution, and Transfer
            Learning is confirmed by direct observation.
          </InsightBlock>
        </Row.Content>
      </Row>

      <Row>
        <Row.Content>
          <GradientCard title="Up Next" color="cyan">
            <p className="text-sm">
              You now know how to <em>see</em> what a CNN learned. In the next
              lesson, you will fine-tune a pretrained model on your own small
              dataset and use Grad-CAM to verify that it learned the right
              features&mdash;not shortcuts. Visualization becomes a debugging
              tool for your own models.
            </p>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* References */}
      <Row>
        <Row.Content>
          <ReferencesBlock
            references={[
              {
                title: 'Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization',
                authors: 'Selvaraju, Cogswell, Das, Vedantam, Parikh & Batra, 2017',
                url: 'https://arxiv.org/abs/1610.02391',
                note: 'Introduced gradient-weighted class activation mapping, the visualization technique taught in this lesson.',
              },
            ]}
          />
        </Row.Content>
      </Row>
    </LessonLayout>
  )
}

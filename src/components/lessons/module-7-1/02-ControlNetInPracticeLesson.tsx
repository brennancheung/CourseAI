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
  GradientCard,
  ComparisonRow,
  SummaryBlock,
  NextStepBlock,
  ReferencesBlock,
  LessonLink,
} from '@/components/lessons'
import { CodeBlock } from '@/components/common/CodeBlock'
import { MermaidDiagram } from '@/components/widgets/MermaidDiagram'
import { ExternalLink } from 'lucide-react'

const NOTEBOOK_URL =
  'https://colab.research.google.com/github/brennancheung/CourseAI/blob/main/notebooks/7-1-2-controlnet-in-practice.ipynb'

/**
 * ControlNet in Practice
 *
 * Lesson 2 in Module 7.1 (Controllable Generation). Lesson 2 overall in Series 7.
 * Cognitive load: CONSOLIDATE (0 new concepts).
 *
 * Practical hands-on lesson covering:
 * 1. Preprocessors (Canny edge detection, MiDaS depth estimation, OpenPose skeleton)
 * 2. Conditioning scale (control-creativity tradeoff)
 * 3. Multi-ControlNet stacking
 * 4. Failure modes and practical guidelines
 *
 * No new theory. This lesson elevates two concepts from INTRODUCED to DEVELOPED:
 * - ControlNet coexistence with text conditioning
 * - ControlNet as modular, swappable component
 *
 * Previous: ControlNet (Module 7.1, Lesson 1 / STRETCH)
 * Next: IP-Adapter (Module 7.1, Lesson 3)
 */

export function ControlNetInPracticeLesson() {
  return (
    <LessonLayout>
      {/* Section 1: Header */}
      <Row>
        <Row.Content>
          <LessonHeader
            title="ControlNet in Practice"
            description="You know the architecture. Now drive it—real preprocessors, real images, the conditioning scale dial, and multi-ControlNet stacking."
            category="Controllable Generation"
          />
        </Row.Content>
      </Row>

      {/* Section 2: Objective + Constraints */}
      <Row>
        <Row.Content>
          <ObjectiveBlock>
            Turn any photograph into a spatial control signal using Canny edge
            detection, MiDaS depth estimation, or OpenPose skeleton extraction.
            Tune the conditioning scale to balance spatial precision against
            creative freedom. Stack multiple ControlNets for combined structural
            control.
          </ObjectiveBlock>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Practical Session">
            No new theory today. Everything here builds on the architecture you
            learned in <LessonLink slug="controlnet">ControlNet</LessonLink>. The emphasis is entirely on
            using the tool: which preprocessor, what settings, how to combine.
          </TipBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <ConstraintBlock
            title="Scope for This Lesson"
            items={[
              'How to preprocess images into spatial maps (Canny, MiDaS depth, OpenPose)',
              'The conditioning_scale parameter and the control-creativity tradeoff',
              'Stacking multiple ControlNets for combined spatial control',
              'Practical guidelines: when to use which preprocessor, how to tune, failure modes',
              'NOT: ControlNet architecture internals (covered in ControlNet—brief callbacks only)',
              'NOT: how preprocessors work internally (Canny algorithm, MiDaS architecture, OpenPose architecture)—used as black-box tools',
              'NOT: training a ControlNet from scratch',
              'NOT: IP-Adapter or image-based conditioning—next lesson',
              'NOT: every preprocessor type (lineart, scribble, normal maps, segmentation)—mentioned for breadth only',
            ]}
          />
        </Row.Content>
      </Row>

      {/* Section 3: Hook — The Missing Piece */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Missing Piece"
            subtitle="You have seen spatial maps—but where do they come from?"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              In <LessonLink slug="controlnet">ControlNet</LessonLink>, you saw Canny edge maps as the
              primary example of spatial conditioning. Edge map goes in,
              ControlNet produces features, those features get added to the
              frozen decoder via zero convolutions. The architecture made
              sense—you could trace the forward pass and explain why it works.
            </p>
            <p className="text-muted-foreground">
              But where did that edge map come from? You have been looking at
              preprocessed spatial maps without knowing how they were made. The
              answer is almost anticlimactic:
            </p>
            <CodeBlock
              code={`import cv2

# One function call. That is it.
edges = cv2.Canny(image, low_threshold=100, high_threshold=200)`}
              language="python"
              filename="preprocess.py"
            />
            <p className="text-muted-foreground">
              One function call. But the quality of this preprocessing step
              determines the quality of everything that follows. Garbage in,
              garbage out—ControlNet faithfully follows whatever spatial map you
              give it.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Three Practical Skills">
            This lesson covers the three things you need to actually use
            ControlNet: (1) extract spatial maps with different preprocessors,
            (2) tune how strongly the model follows them, (3) stack multiple
            types of spatial control.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Section 4: Explain — Preprocessors overview */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Three Preprocessors, Three Types of Control"
            subtitle="Edges, depth, and pose—each captures different spatial information"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Each preprocessor extracts a different kind of spatial information
              from a photograph. The choice of preprocessor determines what kind
              of structural control you get—contours, 3D layering, or body
              pose. The ControlNet architecture is identical for all three; only
              the preprocessing and the checkpoint differ.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="Black-Box Tools">
            We use these preprocessors as tools—pass an image in, get a spatial
            map out. How they work internally (Canny&rsquo;s gradient
            computation, MiDaS&rsquo;s depth network, OpenPose&rsquo;s keypoint
            detection) is outside our scope. The practical skill is knowing
            what each extracts and when to use it.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* Preprocessor pipeline diagram */}
      <Row>
        <Row.Content>
          <MermaidDiagram chart={`
            graph LR
              P["Photograph"]:::input --> PRE["Preprocessor"]:::process
              PRE --> SM["Spatial Map"]:::map
              SM --> CN["ControlNet"]:::controlnet
              CN --> SD["Stable Diffusion Pipeline"]:::pipeline
              T["Text Prompt"]:::input --> SD
              SD --> OUT["Generated Image"]:::output

              classDef input fill:#1e3a5f,stroke:#3b82f6,color:#dbeafe
              classDef process fill:#5b21b6,stroke:#8b5cf6,color:#f5f3ff
              classDef map fill:#92400e,stroke:#f59e0b,color:#fef3c7
              classDef controlnet fill:#065f46,stroke:#10b981,color:#d1fae5
              classDef pipeline fill:#374151,stroke:#6b7280,color:#d1d5db
              classDef output fill:#9f1239,stroke:#f43f5e,color:#ffe4e6
          `} />
          <p className="mt-3 text-sm text-muted-foreground">
            The full workflow: photograph goes through a preprocessor to produce
            a spatial map. The spatial map feeds ControlNet. ControlNet&rsquo;s
            features merge into the frozen SD pipeline alongside the text
            prompt. The choice of preprocessor and ControlNet checkpoint
            determines the type of spatial control.
          </p>
        </Row.Content>
      </Row>

      {/* Canny Edge Detection */}
      <Row>
        <Row.Content>
          <GradientCard title="Canny Edge Detection" color="emerald">
            <div className="space-y-3 text-sm">
              <p>
                <strong>What it extracts:</strong> A binary edge map from
                intensity gradients. Two thresholds (low, high) control edge
                sensitivity—edges below the low threshold are discarded, edges
                above the high threshold are kept, and edges in between are
                kept only if connected to strong edges.
              </p>
              <p>
                <strong>What it controls:</strong> 2D contours and silhouettes.
                The model follows the drawn edges, preserving the composition
                and object boundaries from the source image.
              </p>
              <p>
                <strong>When to use:</strong> Architectural scenes, product
                design, any case where you want precise contour control. Best
                when the source image has clear, well-defined edges.
              </p>
            </div>
          </GradientCard>
          <div className="mt-4">
            <CodeBlock
              code={`import cv2
import numpy as np
from PIL import Image

# Load your photograph
image = np.array(Image.open("photo.jpg"))

# Extract Canny edges — the two thresholds are the key parameter
edges = cv2.Canny(image, low_threshold=100, high_threshold=200)

# Convert to PIL Image for the pipeline
canny_image = Image.fromarray(edges)`}
              language="python"
              filename="canny_preprocessing.py"
            />
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Threshold Tuning Matters">
            The Canny thresholds are the most impactful practical decision you
            will make. Too low (e.g., 50/100): noisy edges everywhere, the
            model tries to follow every spurious contour. Too high (e.g.,
            200/300): only the strongest edges survive, losing important
            structural detail. The sweet spot depends on the image.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* Canny threshold comparison */}
      <Row>
        <Row.Content>
          <p className="text-sm text-muted-foreground mb-3">
            Three threshold settings on the same photograph. These are the
            predictions you should carry into the notebook&mdash;Exercise 1
            will let you verify each one firsthand:
          </p>
          <div className="grid gap-4 md:grid-cols-3">
            <GradientCard title="Too Few Edges" color="rose">
              <div className="space-y-2 text-sm">
                <p><strong>Thresholds: (200, 300)</strong></p>
                <p>
                  Only the boldest contours survive&mdash;the person&rsquo;s
                  silhouette, maybe the strongest architectural lines. Everything
                  else vanishes: facial features, clothing folds, window frames.
                  The edge map looks almost empty.
                </p>
                <p>
                  <strong>What ControlNet produces:</strong> The model gets so
                  little spatial guidance that it improvises. The overall
                  composition lands somewhere near the original, but details are
                  the model&rsquo;s invention. Walls shift position, proportions
                  drift, the face may not align with the silhouette. It feels
                  like a rough sketch, not a controlled generation.
                </p>
              </div>
            </GradientCard>
            <GradientCard title="Good Edges" color="emerald">
              <div className="space-y-2 text-sm">
                <p><strong>Thresholds: (100, 200)</strong></p>
                <p>
                  Clean structural edges capture object boundaries, the
                  person&rsquo;s outline, architectural lines, and major fabric
                  folds&mdash;without picking up surface texture or noise. The
                  edge map reads like a clean line drawing of the scene.
                </p>
                <p>
                  <strong>What ControlNet produces:</strong> The model follows
                  the composition precisely&mdash;objects sit where the edges
                  say, proportions are correct, the figure&rsquo;s pose matches.
                  But within those contours, textures look natural: skin has
                  variation, fabric has shading, the background has depth. The
                  structure is controlled; the details are alive.
                </p>
              </div>
            </GradientCard>
            <GradientCard title="Too Many Edges" color="rose">
              <div className="space-y-2 text-sm">
                <p><strong>Thresholds: (50, 100)</strong></p>
                <p>
                  Every surface texture becomes an edge&mdash;fabric weave,
                  skin pores, wall grain, JPEG compression artifacts. The edge
                  map looks like a dense, scratchy mess rather than a clean
                  structural drawing. There are edges everywhere, and the model
                  will try to follow all of them.
                </p>
                <p>
                  <strong>What ControlNet produces:</strong> The model
                  over-constrains to noise. Skin looks like cracked plaster
                  because the model is faithfully following the texture edges.
                  Flat surfaces (walls, sky) develop strange patterns. Fabric
                  becomes rigid and grid-like instead of flowing. The output has
                  a harsh, mechanical quality&mdash;the model is tracing rather
                  than generating.
                </p>
              </div>
            </GradientCard>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Garbage In, Garbage Out">
            ControlNet faithfully follows whatever spatial map you give it. If
            the map is noisy, the output is noisy. If the map is sparse, the
            output is loosely controlled. Preprocessing quality is the single
            most impactful practical decision.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* MiDaS Depth Estimation */}
      <Row>
        <Row.Content>
          <GradientCard title="MiDaS Depth Estimation" color="orange">
            <div className="space-y-3 text-sm">
              <p>
                <strong>What it extracts:</strong> A monocular depth map from a
                single photograph. Lighter pixels are closer, darker pixels are
                farther. Uses a pre-trained depth estimation model (MiDaS or
                DPT) to infer 3D structure from a 2D image.
              </p>
              <p>
                <strong>What it controls:</strong> 3D structure, perspective,
                and spatial layering. Objects at different depths maintain
                their relative positioning. Foreground/background separation is
                preserved.
              </p>
              <p>
                <strong>When to use:</strong> Landscape composition, scene
                depth control, any case where you care about spatial
                arrangement and perspective more than exact contours. Great for
                preserving the &ldquo;feel&rdquo; of a scene&rsquo;s depth
                without locking down specific edges.
              </p>
            </div>
          </GradientCard>
          <div className="mt-4">
            <CodeBlock
              code={`from transformers import pipeline
from PIL import Image

# Load depth estimation model
depth_estimator = pipeline("depth-estimation", model="Intel/dpt-large")

# Extract depth map — one call, no threshold tuning
image = Image.open("photo.jpg")
depth_result = depth_estimator(image)
depth_map = depth_result["depth"]  # PIL Image, grayscale`}
              language="python"
              filename="depth_preprocessing.py"
            />
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="No Thresholds to Tune">
            Unlike Canny, depth estimation has no user-facing threshold
            parameter. The model produces a depth map directly. This makes
            depth preprocessing simpler—but also means you have less control
            over what the map captures. The quality depends entirely on the
            depth model&rsquo;s performance on your specific image.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* OpenPose Skeleton Detection */}
      <Row>
        <Row.Content>
          <GradientCard title="OpenPose Skeleton Detection" color="violet">
            <div className="space-y-3 text-sm">
              <p>
                <strong>What it extracts:</strong> Body keypoints connected
                into a stick-figure skeleton. Identifies joints (shoulders,
                elbows, wrists, hips, knees, ankles) and draws the connections
                between them. Optionally includes hand and face keypoints for
                finer control.
              </p>
              <p>
                <strong>What it controls:</strong> Human body pose. The model
                generates a figure matching the skeleton&rsquo;s pose—same arm
                positions, same leg angles, same body orientation. The content
                (who the person is, what they wear, the background) comes from
                the text prompt.
              </p>
              <p>
                <strong>When to use:</strong> Character positioning, pose
                transfer (extract pose from one photo, generate a completely
                different person in the same pose), any case where human body
                structure matters. The most &ldquo;magical&rdquo; ControlNet
                application—a stick figure in, a photorealistic person out.
              </p>
            </div>
          </GradientCard>
          <div className="mt-4">
            <CodeBlock
              code={`from controlnet_aux import OpenposeDetector

# Load OpenPose detector
openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")

# Extract skeleton — requires a photo with a visible person
image = Image.open("person_photo.jpg")
pose_map = openpose(image)  # PIL Image, skeleton on black background`}
              language="python"
              filename="openpose_preprocessing.py"
            />
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Pose Transfer">
            The real power: extract a skeleton from a dance photo, then prompt
            &ldquo;an astronaut on Mars.&rdquo; Same pose, completely different
            person and scene. The skeleton carries the WHERE, the text prompt
            carries the WHAT.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* "Notice what stays the same" moment */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Notice What Stays the Same"
            subtitle="The pipeline does not know or care what kind of map it receives"
          />
          <CodeBlock
            code={`from diffusers import StableDiffusionControlNetPipeline, ControlNetModel

# Load ControlNet checkpoint — THIS is the only thing that changes
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny"   # swap for -depth or -openpose
)

# Pipeline setup — identical for all map types
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    controlnet=controlnet,
)

# Generation — identical API regardless of map type
result = pipe(
    prompt="a watercolor painting of a mountain village",
    image=spatial_map,        # Canny, depth, or pose — pipeline does not care
    num_inference_steps=30,
)`}
            language="python"
            filename="controlnet_pipeline.py"
          />
          <div className="space-y-4 mt-4">
            <p className="text-muted-foreground">
              The pipeline call is <strong>identical</strong> for all three
              preprocessor types. The only things that change are: (1) which
              preprocessor extracts the map from the image, and (2) which
              ControlNet checkpoint you load. The architecture you learned in{' '}
              <LessonLink slug="controlnet">ControlNet</LessonLink> is genuinely map-agnostic—the
              trainable encoder copy processes whatever spatial input you give
              it.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Modularity in Action">
            Remember &ldquo;four translators, one pipeline&rdquo;? Swapping
            ControlNet checkpoints is the same pattern as swapping any modular
            component: load a different checkpoint, get a different behavior,
            no retraining. The pipeline is a socket; the ControlNet checkpoint
            is the plug.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Other preprocessor types — brief mention */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              <strong>Other preprocessor types</strong> exist beyond these
              three: lineart (clean line drawings), scribble (hand-drawn
              sketches), normal maps (surface orientation), segmentation maps
              (semantic regions), and more. Each has its own ControlNet
              checkpoint. The pattern is always the same: extract a spatial
              map, load the matching checkpoint, run the pipeline.
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* Section 5: Check — Preprocessor Selection */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Which Preprocessor?"
            subtitle="Match the control type to your creative intent"
          />
          <div className="space-y-4">
            <GradientCard title="Scenario 1" color="cyan">
              <p className="text-sm">
                You want to generate a fantasy castle that follows the skyline
                of a real city photograph. Which preprocessor?
              </p>
              <details className="mt-2">
                <summary className="font-medium cursor-pointer text-primary text-sm">
                  Reveal
                </summary>
                <p className="mt-2 text-sm text-muted-foreground">
                  <strong>Canny edges</strong> for the silhouette, or{' '}
                  <strong>depth</strong> for the layering. If you want the
                  exact skyline contour, Canny gives precise edge control. If
                  you care more about the foreground/background arrangement,
                  depth preserves the spatial layering without locking down
                  specific contours.
                </p>
              </details>
            </GradientCard>
            <GradientCard title="Scenario 2" color="cyan">
              <p className="text-sm">
                You want to generate a character in the same pose as a photo
                of a dancer. Which preprocessor?
              </p>
              <details className="mt-2">
                <summary className="font-medium cursor-pointer text-primary text-sm">
                  Reveal
                </summary>
                <p className="mt-2 text-sm text-muted-foreground">
                  <strong>OpenPose.</strong> You want body pose—joint positions
                  and limb angles—not edges or depth. The skeleton captures
                  exactly the structural information you need: the dancer&rsquo;s
                  pose transfers to whatever character the text prompt
                  describes.
                </p>
              </details>
            </GradientCard>
            <GradientCard title="Scenario 3" color="cyan">
              <p className="text-sm">
                You want to generate a landscape that preserves the
                foreground/background layering of a reference photo but changes
                the content entirely. Which preprocessor?
              </p>
              <details className="mt-2">
                <summary className="font-medium cursor-pointer text-primary text-sm">
                  Reveal
                </summary>
                <p className="mt-2 text-sm text-muted-foreground">
                  <strong>Depth.</strong> You care about spatial arrangement
                  and perspective, not specific contours. The depth map
                  preserves where things are in 3D space—near objects stay
                  near, far objects stay far—while the text prompt fills in
                  entirely new content.
                </p>
              </details>
            </GradientCard>
          </div>
        </Row.Content>
      </Row>

      {/* Section 6: Explain — Conditioning Scale */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Conditioning Scale"
            subtitle="The volume knob for spatial control"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              You already know the CFG guidance scale: higher values mean
              &ldquo;follow the text prompt more strongly&rdquo; at the cost of
              diversity. Conditioning scale is the <strong>same idea</strong>{' '}
              for spatial control: higher values mean &ldquo;follow the
              spatial map more strictly&rdquo; at the cost of natural
              variation. Two knobs on the same mixing board.
            </p>
            <p className="text-muted-foreground">
              Remember &ldquo;Nothing, Then a Whisper&rdquo; from{' '}
              <LessonLink slug="controlnet">ControlNet</LessonLink>? During training, zero convolutions
              gradually turn up the volume of spatial control from zero. At
              inference time, <code className="text-xs">conditioning_scale</code>{' '}
              is your <strong>manual volume knob</strong> for the same signal.
              You choose how loud the spatial control is. If the tradeoff feels
              familiar, it should&mdash;in <LessonLink slug="img2img-and-inpainting">Img2Img and Inpainting</LessonLink>,
              the strength parameter controlled how much of the original image
              survived versus how much the model generated freely. Conditioning
              scale is the same idea applied to spatial maps.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Two Knobs, One Board">
            CFG guidance scale controls text influence. Conditioning scale
            controls spatial influence. Both trade precision for creativity.
            Both have sweet spots where the model balances following the signal
            against generating natural-looking output.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Conditioning scale sweep */}
      <Row>
        <Row.Content>
          <p className="text-sm text-muted-foreground mb-3">
            Imagine the same Canny edge map of a person in front of architecture,
            the same text prompt, the same seed&mdash;only the conditioning scale
            changes. Here is what you will see when you run the sweep in the
            notebook:
          </p>
          <div className="grid gap-3 md:grid-cols-3 lg:grid-cols-6">
            <GradientCard title="Scale 0.3" color="blue">
              <p className="text-xs">
                The edge map is barely audible. The person&rsquo;s figure might
                land in roughly the right area, but contours drift and the
                composition is the model&rsquo;s own interpretation of the text
                prompt. You would struggle to tell this was edge-conditioned.
              </p>
            </GradientCard>
            <GradientCard title="Scale 0.5" color="blue">
              <p className="text-xs">
                The spatial map starts to show. Large shapes&mdash;the
                person&rsquo;s silhouette, major architectural lines&mdash;land
                in approximately the right places. But fine contours are loose:
                windows might shift, the face outline is approximate.
              </p>
            </GradientCard>
            <GradientCard title="Scale 0.7" color="emerald">
              <p className="text-xs">
                Clear structural adherence. Object boundaries follow the edges,
                the figure&rsquo;s pose matches, proportions are correct. Yet
                textures remain natural&mdash;skin has variation, fabric
                flows, the background has depth. The sweet spot.
              </p>
            </GradientCard>
            <GradientCard title="Scale 1.0" color="emerald">
              <p className="text-xs">
                Strong spatial control. The composition closely matches the edge
                map at every contour. Textures are still plausible but slightly
                less varied&mdash;surfaces that should be smooth start hinting
                at the underlying edges. Typical default.
              </p>
            </GradientCard>
            <GradientCard title="Scale 1.5" color="amber">
              <p className="text-xs">
                Over-constraining begins. Textures flatten&mdash;skin looks
                painted-on, fabric loses its folds. You can almost see ghost
                lines from the edge map bleeding through surfaces that should
                be smooth. A stiff, mechanical quality creeps in.
              </p>
            </GradientCard>
            <GradientCard title="Scale 2.0" color="rose">
              <p className="text-xs">
                Heavily over-constrained. The model is tracing the edge map
                pixel-by-pixel. Flat areas develop strange banding artifacts.
                Textures are gone&mdash;everything looks like a rigid cutout.
                The image is technically &ldquo;controlled&rdquo; but no longer
                looks like a natural photograph or painting.
              </p>
            </GradientCard>
          </div>
          <div className="space-y-4 mt-4">
            <p className="text-muted-foreground">
              The sweet spot is typically <strong>0.7&ndash;1.0</strong> for
              most use cases. Below 0.5, the spatial control is too weak to
              meaningfully guide composition. Above 1.5, the model
              over-constrains: fine details become rigid, textures flatten, and
              the image looks mechanical.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="The Sweet Spot">
            Start at 1.0 and adjust. If the output feels too rigid or
            artifacted, decrease to 0.7&ndash;0.8. If the spatial structure is
            too loose, increase to 1.1&ndash;1.2. Rarely go above 1.5.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Misconception: scale 1.0 does not disable text */}
      <Row>
        <Row.Content>
          <WarningBlock title="Scale 1.0 Does Not Disable Text Conditioning">
            A natural assumption: conditioning_scale=1.0 means &ldquo;100%
            follow the spatial map, ignore the text.&rdquo; This is wrong.
            Generate with scale=1.0 and two different text prompts using the
            same edge map—the structure is identical but the content and style
            differ clearly. Both conditioning dimensions remain active.
            Conditioning scale modulates spatial strength; it does not disable
            text conditioning. Remember WHEN/WHAT/WHERE: the scale controls
            how loud the WHERE signal is, not whether the WHAT signal is on.
          </WarningBlock>
        </Row.Content>
      </Row>

      {/* Code example showing conditioning_scale */}
      <Row>
        <Row.Content>
          <CodeBlock
            code={`# The conditioning_scale parameter — your volume knob
result = pipe(
    prompt="a watercolor painting of a cozy village",
    image=canny_edges,
    controlnet_conditioning_scale=0.8,   # <-- the dial
    num_inference_steps=30,
)

# Same edges, different prompt — structure stays, content changes
result_alt = pipe(
    prompt="a cyberpunk city at night, neon lights",
    image=canny_edges,                    # same spatial map
    controlnet_conditioning_scale=0.8,    # same strength
    num_inference_steps=30,
)
# Both images follow the same edges. Content is completely different.`}
            language="python"
            filename="conditioning_scale.py"
          />
        </Row.Content>
      </Row>

      {/* Section 7: Check — Conditioning Scale Prediction */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Predict Before You Run"
            subtitle="Build intuition for the conditioning scale dial"
          />
          <div className="space-y-4">
            <GradientCard title="Prediction 1: High Scale on Detailed Edges" color="cyan">
              <p className="text-sm">
                You set conditioning_scale=2.0 on a Canny edge map of a
                detailed architectural drawing with hundreds of fine edges.
                What do you expect to see?
              </p>
              <details className="mt-2">
                <summary className="font-medium cursor-pointer text-primary text-sm">
                  Reveal
                </summary>
                <p className="mt-2 text-sm text-muted-foreground">
                  Over-constrained output. The model tries to follow every edge
                  pixel, producing rigid textures and artifacts where edges are
                  dense or ambiguous. Fine details that should be smooth
                  textures (walls, sky, water) become locked to the edge map,
                  losing natural variation. The image looks mechanical—like the
                  model is tracing rather than generating.
                </p>
              </details>
            </GradientCard>
            <GradientCard title="Prediction 2: Low Scale" color="cyan">
              <p className="text-sm">
                You set conditioning_scale=0.3 on the same edge map. What do
                you expect?
              </p>
              <details className="mt-2">
                <summary className="font-medium cursor-pointer text-primary text-sm">
                  Reveal
                </summary>
                <p className="mt-2 text-sm text-muted-foreground">
                  The model mostly ignores the edges. The composition might
                  roughly follow the spatial layout (large shapes in
                  approximately the right places) but fine details are the
                  model&rsquo;s own. The text prompt dominates. The spatial map
                  is a faint suggestion, not a constraint.
                </p>
              </details>
            </GradientCard>
          </div>
        </Row.Content>
      </Row>

      {/* Section 8: Explain — Multi-ControlNet Stacking */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Multi-ControlNet Stacking"
            subtitle="Combining multiple types of spatial control"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Remember the forward pass from <LessonLink slug="controlnet">ControlNet</LessonLink>: each
              ControlNet independently contributes additive features to the
              skip connections. Stacking multiple ControlNets means the decoder
              receives the sum of all their contributions:
            </p>
            <div className="py-3 px-5 bg-muted/50 rounded-lg font-mono text-sm text-center">
              skip_connection = e_i + z_i_canny + z_i_depth
            </div>
            <p className="text-muted-foreground">
              Each ControlNet provides a different type of spatial
              information—edges enforce contours while depth enforces
              layering—and they compose by summation. The decoder receives the
              combined structural constraints and generates accordingly.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="More Translators">
            &ldquo;Four translators, one pipeline&rdquo; extends naturally.
            Stacking adds a fifth translator. Each ControlNet independently
            contributes additive features. They compose by summation, just
            like the single-ControlNet case.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Stacking code pattern */}
      <Row>
        <Row.Content>
          <CodeBlock
            code={`from diffusers import StableDiffusionControlNetPipeline, ControlNetModel

# Load multiple ControlNet checkpoints
controlnet_canny = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny"
)
controlnet_depth = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-depth"
)

# Pipeline takes a LIST of ControlNets
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    controlnet=[controlnet_canny, controlnet_depth],  # list!
)

# Generation: list of images, list of scales
result = pipe(
    prompt="a fantasy landscape, digital art",
    image=[canny_map, depth_map],                          # one per ControlNet
    controlnet_conditioning_scale=[0.8, 0.6],              # one per ControlNet
    num_inference_steps=30,
)`}
            language="python"
            filename="multi_controlnet.py"
          />
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Per-ControlNet Scale">
            Each ControlNet gets its own conditioning_scale. You can weight
            edges more heavily than depth, or vice versa. This gives you fine
            control over which type of spatial information dominates.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Complementary vs conflicting stacking */}
      <Row>
        <Row.Content>
          <ComparisonRow
            left={{
              title: 'Complementary Stacking',
              color: 'emerald',
              items: [
                'Both maps from the same source image',
                'Edges enforce contours, depth enforces layering',
                'Information is complementary, not redundant',
                'Combined output is more precisely controlled',
                'Use moderate scales (0.5–0.8) for each',
              ],
            }}
            right={{
              title: 'Conflicting Stacking',
              color: 'rose',
              items: [
                'Maps from different source images',
                'Edges say "boundary here" but depth says "smooth here"',
                'Contradictory spatial signals fight each other',
                'Produces artifacts, especially at high scales',
                'Lower scales to reduce conflict, or pick one map',
              ],
            }}
          />
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Stacking Is Not Doubling">
            A common misconception: stacking two ControlNets doubles the
            control strength. It does not. Stacking provides two different
            types of control simultaneously—edges enforce contours while depth
            enforces layering. If both scales are too high, they can conflict,
            producing artifacts rather than stronger control.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* Practical stacking guidelines */}
      <Row>
        <Row.Content>
          <GradientCard title="Practical Stacking Guidelines" color="blue">
            <ul className="space-y-2 text-sm">
              <li>
                <strong>Start with one.</strong> Get a single ControlNet
                working well before adding a second. Debugging two at once is
                harder than debugging one.
              </li>
              <li>
                <strong>Use moderate scales.</strong> When stacking, use
                0.5&ndash;0.8 per ControlNet rather than 1.0+. The combined
                effect is stronger than either alone.
              </li>
              <li>
                <strong>Same source image.</strong> Keep both maps derived
                from the same photograph for complementary control. Maps from
                different images are more likely to conflict.
              </li>
              <li>
                <strong>Add complexity gradually.</strong> One ControlNet,
                then two. Tune each scale separately. The goal is
                complementary constraints, not maximum constraint.
              </li>
            </ul>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* Section 9: Check — Stacking Transfer Question */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Debugging a Friend's Stack"
            subtitle="Apply what you know to a practical problem"
          />
          <div className="space-y-4">
            <GradientCard title="Your Friend's Problem" color="cyan">
              <p className="text-sm">
                Your friend stacks three ControlNets (Canny + depth + pose)
                all at conditioning_scale=1.5 and gets muddy, artifacted
                results. What would you suggest?
              </p>
              <details className="mt-2">
                <summary className="font-medium cursor-pointer text-primary text-sm">
                  Reveal
                </summary>
                <div className="mt-2 text-sm text-muted-foreground space-y-2">
                  <p>
                    <strong>Lower the conditioning scales.</strong> Three
                    ControlNets at 1.5 each massively over-constrains the
                    model. Start with 0.5&ndash;0.7 for each and increase
                    selectively.
                  </p>
                  <p>
                    <strong>Check source consistency.</strong> Are all three
                    spatial maps from the same source image? If not, they may
                    provide contradictory spatial information.
                  </p>
                  <p>
                    <strong>Simplify first.</strong> Test each ControlNet
                    individually to verify each one produces good results
                    alone. Then add them one at a time.
                  </p>
                </div>
              </details>
            </GradientCard>
          </div>
        </Row.Content>
      </Row>

      {/* Section 10: Practice — Notebook Exercises */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Practice: Hands-On ControlNet"
            subtitle="Five exercises from guided to independent"
          />
          <div className="rounded-lg border-2 border-primary/50 bg-primary/5 p-6">
            <div className="space-y-4">
              <p className="text-muted-foreground">
                This is the core of the lesson. The notebook is where you
                build practical intuition—running real inference, varying
                parameters, and seeing the results. The concepts above provide
                the framework; the notebook provides the experience.
              </p>
              <a
                href={NOTEBOOK_URL}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 px-4 py-2 rounded-md bg-primary text-primary-foreground font-medium text-sm hover:bg-primary/90 transition-colors"
              >
                <ExternalLink className="w-4 h-4" />
                Open in Google Colab
              </a>
              <ul className="text-sm text-muted-foreground space-y-3">
                <li>
                  <strong>Exercise 1 (Guided): Canny Edge Preprocessing.</strong>{' '}
                  Load a photograph and extract Canny edges with three threshold
                  pairs: (50, 100), (100, 200), (200, 300). Generate an image
                  with each and observe how edge quality affects output.
                  Predict before running: which threshold will produce the best
                  result?
                </li>
                <li>
                  <strong>Exercise 2 (Guided): Three Preprocessors, One
                  Pipeline.</strong>{' '}
                  Take the same source photo and extract Canny edges, a depth
                  map, and an OpenPose skeleton. Generate with each and compare
                  side-by-side. Notice what stays the same in the API code
                  across all three.
                </li>
                <li>
                  <strong>Exercise 3 (Supported): Conditioning Scale Sweep.</strong>{' '}
                  Use the best Canny map from Exercise 1 and generate at
                  scales 0.3, 0.5, 0.7, 1.0, 1.5, 2.0. Display as a
                  comparison grid. Then: same edges, scale=1.0, two different
                  prompts—verify text conditioning is still active.
                </li>
                <li>
                  <strong>Exercise 4 (Supported): Multi-ControlNet Stacking.</strong>{' '}
                  Load Canny and depth checkpoints. Extract both maps from the
                  same image. Generate with edges only, depth only, and both
                  stacked. Compare in a 3-column grid.
                </li>
                <li>
                  <strong>Exercise 5 (Independent): Your Composition.</strong>{' '}
                  Choose your own source image. Select the best preprocessor(s)
                  for your creative intent. Tune conditioning scale(s) to
                  achieve a specific visual goal. No scaffolding—you decide
                  the full workflow.
                </li>
              </ul>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Exercise Progression">
            <ol className="space-y-1 text-sm list-decimal list-inside">
              <li>Guided: preprocessing fundamentals</li>
              <li>Guided: preprocessor comparison</li>
              <li>Supported: conditioning scale exploration</li>
              <li>Supported: multi-ControlNet stacking</li>
              <li>Independent: integrate everything</li>
            </ol>
            <p className="text-sm mt-2">
              Exercises are cumulative: Exercise 1&rsquo;s best edge map feeds
              Exercise 3. Exercise 2&rsquo;s depth map feeds Exercise 4.
            </p>
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Section 11: Summary */}
      <Row>
        <Row.Content>
          <SummaryBlock
            title="Key Takeaways"
            items={[
              {
                headline:
                  'Three preprocessors, three types of spatial control.',
                description:
                  'Canny edges control 2D contours. MiDaS depth controls 3D structure and layering. OpenPose controls body pose. Many more exist—these are the three most common. The pipeline API is identical for all of them.',
              },
              {
                headline:
                  'Preprocessing quality is the most impactful practical decision.',
                description:
                  'ControlNet faithfully follows whatever spatial map you give it. Bad Canny thresholds produce noisy edges; noisy edges produce artifacted output. Garbage in, garbage out. Tune the preprocessing before touching anything else.',
              },
              {
                headline:
                  'Conditioning scale is your volume knob for spatial control.',
                description:
                  'Low scale: spatial map is a suggestion, model generates freely. High scale: model rigidly follows the map, losing natural variation. Sweet spot is typically 0.7–1.0. Same tradeoff as CFG guidance scale—precision vs creativity.',
              },
              {
                headline:
                  'Multiple ControlNets stack by summing their additive features.',
                description:
                  'Use complementary maps from the same source image, moderate scales (0.5–0.8 each), and add complexity gradually. Conflicting maps or excessive scales produce artifacts, not stronger control.',
              },
            ]}
          />
        </Row.Content>
      </Row>

      {/* Mental model echo */}
      <Row>
        <Row.Content>
          <InsightBlock title="Mental Model">
            The WHEN/WHAT/WHERE framework now has a volume knob for WHERE. And
            you can layer multiple WHERE signals from different spatial
            translators—each contributing its own type of structural
            control, composing by summation at the skip connections.
          </InsightBlock>
        </Row.Content>
      </Row>

      {/* References */}
      <Row>
        <Row.Content>
          <ReferencesBlock
            references={[
              {
                title: 'Adding Conditional Control to Text-to-Image Diffusion Models',
                authors: 'Zhang, Rao & Agrawala, 2023',
                url: 'https://arxiv.org/abs/2302.05543',
                note: 'The original ControlNet paper. Section 4 covers results across spatial map types. Appendix has preprocessor details.',
              },
              {
                title: 'controlnet_aux: Preprocessors for ControlNet',
                authors: 'Hugging Face community',
                url: 'https://github.com/huggingface/controlnet_aux',
                note: 'Library providing all ControlNet preprocessors (Canny, depth, OpenPose, lineart, etc.) in a unified API.',
              },
              {
                title: 'Towards Robust Monocular Depth Estimation (MiDaS)',
                authors: 'Ranftl et al., 2020',
                url: 'https://arxiv.org/abs/1907.01341',
                note: 'The depth estimation model used for depth-conditioned ControlNet. Background reading only—we use it as a black box.',
              },
            ]}
          />
        </Row.Content>
      </Row>

      {/* Section 12: Next Step */}
      <Row>
        <Row.Content>
          <NextStepBlock
            href="/app"
            title="Up Next: IP-Adapter"
            description="So far, all conditioning has been text (WHAT) or spatial maps (WHERE). But what if you want to say 'generate something that looks like this reference image'—not its edges or depth, but its semantic content and style? IP-Adapter adds a new conditioning dimension by injecting image embeddings into cross-attention."
          />
        </Row.Content>
      </Row>
    </LessonLayout>
  )
}

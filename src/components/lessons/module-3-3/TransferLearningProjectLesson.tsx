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
  PhaseCard,
  ModuleCompleteBlock,
} from '@/components/lessons'
import { ExternalLink } from 'lucide-react'

/**
 * Project: Transfer Learning
 *
 * Second and final lesson in Module 3.3 (Seeing What CNNs See).
 * FINAL lesson in Series 3 (CNNs).
 *
 * This is a CONSOLIDATE capstone project. No new concepts are taught.
 * The student combines transfer learning (3.2.3) and Grad-CAM (3.3.1)
 * into a real practitioner workflow: train, visualize, diagnose.
 *
 * Depth upgrades:
 * - Transfer learning workflow: DEVELOPED -> APPLIED
 * - Grad-CAM as debugging tool: DEVELOPED -> APPLIED
 * - Fine-tuning with differential LR: INTRODUCED -> DEVELOPED
 * - Shortcut learning detection: INTRODUCED -> DEVELOPED
 *
 * Previous: Visualizing Features (module 3.3, lesson 1)
 * Next: Series 4 (LLMs)
 */

export function TransferLearningProjectLesson() {
  return (
    <LessonLayout>
      {/* ================================================================
          1. Header
          ================================================================ */}
      <Row>
        <Row.Content>
          <LessonHeader
            title="Project: Transfer Learning"
            description="Fine-tune a pretrained CNN on a small flower dataset and use Grad-CAM to verify the model learned the right features&mdash;not shortcuts."
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
            Execute the complete transfer learning practitioner workflow:
            fine-tune a pretrained model on a small dataset, evaluate accuracy,
            then use Grad-CAM to answer the question that matters
            most&mdash;is the model right for the right reasons?
          </ObjectiveBlock>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="What You Own">
            <ul className="space-y-1 text-sm">
              <li>Decisions (freeze/unfreeze, learning rates)</li>
              <li>Interpretation (Grad-CAM analysis)</li>
              <li>Reflection (what the model learned)</li>
            </ul>
          </TipBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <ConstraintBlock
            title="Project Scope"
            items={[
              'This is a capstone project: no new concepts, all integration',
              'The notebook provides: dataset, data loading, display utilities, training loop skeleton, Grad-CAM utilities',
              'Your job: make decisions, interpret results, diagnose model behavior',
              'NOT: hyperparameter tuning, architecture search, or advanced augmentation',
              'NOT: deployment, multi-GPU training, or production optimization',
              'NOT: fixing or iterating on a broken model (diagnose, not debug)',
              'Estimated time: 30–45 minutes on a Colab GPU',
            ]}
          />
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="No New Concepts">
            Every technique in this project was taught in a prior lesson.
            Transfer learning from Transfer Learning, Grad-CAM from
            Visualizing Features. The challenge is putting them together into
            a coherent workflow.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          2. Hook: "You have all the pieces"
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="You Have All the Pieces"
            subtitle="Time to use them together"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Throughout this series, you built a complete mental model of
              convolutional neural networks. You learned what convolutions
              compute, how architectures evolved from LeNet to ResNet, how to
              reuse pretrained models for new tasks, and how to see what your
              model actually learned with Grad-CAM. Each of those skills was
              practiced in isolation.
            </p>
            <p className="text-muted-foreground">
              This project puts them together for the first time. You will
              take a small dataset of flower images the pretrained model has
              never seen, fine-tune a ResNet to classify them, and then use
              Grad-CAM to answer the question that matters most in
              practice: <strong>is the model right for the right reasons?</strong>
            </p>
            <p className="text-muted-foreground">
              High accuracy is not enough. In Visualizing Features, you saw
              how a wolf/husky classifier could achieve 90% accuracy by
              learning the wrong features&mdash;snow vs forest backgrounds
              instead of animal characteristics. Grad-CAM is how you check.
              This is the workflow that professional practitioners use every
              day&mdash;train, visualize, diagnose&mdash;and by the end of
              this project, you will have done it yourself.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The Practitioner Workflow">
            Train. Evaluate. Visualize. Diagnose. This four-step loop is how
            professionals validate CNN models. Accuracy is step two. The real
            work starts at step three.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          3. Project Brief: The Dataset and Task
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Dataset and the Task"
            subtitle="Flower classification on a small dataset"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              You will work with a subset of the{' '}
              <strong>Oxford Flowers102</strong> dataset&mdash;a collection
              of flower photographs from 102 species. The notebook filters
              this down to 8 visually distinct species with roughly 50&ndash;80
              images each. This is a <em>realistic</em> small-dataset
              scenario: enough data for transfer learning to work, but far
              too little to train from scratch.
            </p>
            <p className="text-muted-foreground">
              Your task: classify flower images into the correct species using
              a pretrained ResNet-18. Then verify the model is actually
              looking at the flowers&mdash;not the background, the pot, or the
              photographer&apos;s hand.
            </p>
            <GradientCard title="Decision Framework" color="blue">
              <div className="text-sm space-y-2">
                <p>
                  Apply the 2x2 matrix from Transfer Learning to this
                  dataset:
                </p>
                <ul className="space-y-1 ml-2">
                  <li>
                    <strong>Dataset size:</strong> Small (~400&ndash;600
                    images total)
                  </li>
                  <li>
                    <strong>Domain similarity:</strong> Moderate&mdash;ImageNet
                    contains flowers, but these specific species were not
                    individual classes
                  </li>
                  <li>
                    <strong>Recommendation:</strong> Start with feature
                    extraction. Add fine-tuning only if needed.
                  </li>
                </ul>
                <p className="text-muted-foreground">
                  Remember: &ldquo;Start with the simplest strategy, add
                  complexity only if needed.&rdquo;
                </p>
              </div>
            </GradientCard>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Why Flowers?">
            Flowers are ideal for this project: visually distinct species,
            clear discriminative features (petal shape, color, arrangement),
            and Grad-CAM heatmaps are easy to interpret. You can tell at a
            glance whether the model is focusing on the flower or the
            background.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          4. Project Guidance: The Workflow
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Workflow"
            subtitle="Step by step through the notebook"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The notebook mirrors the practitioner workflow. Each phase builds
              on the last. Here is what to expect and what to look for at each
              step.
            </p>

            <PhaseCard
              number={1}
              title="Explore the Data"
              subtitle="Understand what you are working with"
              color="cyan"
            >
              <div className="text-sm space-y-2">
                <p>
                  Load the dataset, examine class distribution, and look at
                  sample images from each species. Apply data augmentation
                  transforms (random crops, horizontal flips, color jitter)
                  for the training set.
                </p>
                <p className="text-muted-foreground">
                  <strong>Look for:</strong> Are classes roughly balanced? Are
                  the images varied (different backgrounds, lighting, angles)?
                  Imbalanced classes or uniform backgrounds could bias the
                  model.
                </p>
              </div>
            </PhaseCard>

            <PhaseCard
              number={2}
              title="Feature Extraction"
              subtitle="Freeze the backbone, train the head"
              color="blue"
            >
              <div className="text-sm space-y-2">
                <p>
                  Load a pretrained ResNet-18, freeze all backbone parameters,
                  replace the classification head for your number of classes,
                  and train. This is the three-step pattern from Transfer
                  Learning.
                </p>
                <p className="text-muted-foreground">
                  <strong>Look for:</strong> Training accuracy climbing
                  quickly (the pretrained features should be strong).
                  Validation accuracy should be close to training
                  accuracy&mdash;overfitting is less likely with a frozen
                  backbone.
                </p>
              </div>
            </PhaseCard>

            <PhaseCard
              number={3}
              title="Grad-CAM Validation"
              subtitle="Is the model right for the right reasons?"
              color="violet"
            >
              <div className="text-sm space-y-2">
                <p>
                  <strong>This is the most important step.</strong> Run
                  Grad-CAM on correctly classified test images from each
                  class. Overlay the heatmap on the original image and ask:
                  does the model focus on the flower, or on something else?
                </p>
                <p className="text-muted-foreground">
                  <strong>Look for:</strong> Heatmap highlighting petals,
                  stamens, or flower structure. Be suspicious if the heatmap
                  highlights image borders, background foliage, or the pot.
                </p>
              </div>
            </PhaseCard>

            <PhaseCard
              number={4}
              title="Fine-Tuning (Optional Extension)"
              subtitle="Now that you have a baseline, decide if it is worth adapting the backbone"
              color="orange"
            >
              <div className="text-sm space-y-2">
                <p>
                  You have feature extraction results and Grad-CAM heatmaps.
                  Now try fine-tuning to see if it helps: unfreeze layer4 and
                  retrain with differential learning rates (low for backbone,
                  higher for head). Compare accuracy <em>and</em> Grad-CAM
                  focus to the feature extraction baseline.
                </p>
                <p className="text-muted-foreground">
                  <strong>Look for:</strong> Does fine-tuning improve accuracy?
                  On a small dataset, the improvement may be modest or even
                  negative. This is a realistic outcome, not a
                  failure&mdash;it means feature extraction was already
                  sufficient.
                </p>
              </div>
            </PhaseCard>

            <PhaseCard
              number={5}
              title="Final Comparison"
              subtitle="Side-by-side results and Grad-CAM heatmaps"
              color="emerald"
            >
              <div className="text-sm space-y-2">
                <p>
                  Build an accuracy comparison table (feature extraction vs
                  fine-tuning). Then run Grad-CAM on the same test images for
                  both models. Compare: did fine-tuning change what the model
                  focuses on? Are the heatmaps tighter, more focused on the
                  flower?
                </p>
                <p className="text-muted-foreground">
                  <strong>Look for:</strong> Differences in spatial focus
                  between the two approaches. Fine-tuning often produces
                  tighter, more object-focused heatmaps because the backbone
                  adapts to the specific task.
                </p>
              </div>
            </PhaseCard>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Accuracy Is Not Enough">
            A model that achieves 90% accuracy by looking at the background
            will fail on new images with different backgrounds. Grad-CAM is
            your insurance policy.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          5. Grad-CAM Interpretation Guide
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Reading Grad-CAM Heatmaps"
            subtitle="What to look for in your results"
          />
          <div className="space-y-4">
            <GradientCard title="Good Signs" color="emerald">
              <ul className="space-y-1 text-sm">
                <li>
                  Heatmap highlights the flower or its discriminative
                  features (petals, center, shape)
                </li>
                <li>
                  Focus area matches what a human would use to identify the
                  species
                </li>
                <li>
                  Different classes produce different spatial focus (the model
                  learned species-specific features)
                </li>
              </ul>
            </GradientCard>

            <GradientCard title="Warning Signs" color="rose">
              <ul className="space-y-1 text-sm">
                <li>
                  Heatmap highlights background, image borders, or
                  non-flower regions
                </li>
                <li>
                  Same spatial focus for all classes (the model found a
                  shortcut, not species-specific features)
                </li>
                <li>
                  Focus on artifacts: watermarks, image corners, or the pot
                  the flower sits in
                </li>
              </ul>
            </GradientCard>

            <GradientCard title="Ambiguous Cases" color="amber">
              <ul className="space-y-1 text-sm">
                <li>
                  Heatmap highlights the flower plus some surrounding context
                  (leaves, stem)&mdash;this can be legitimate
                </li>
                <li>
                  Focus is on the correct region but diffuse rather than
                  tight&mdash;remember, Grad-CAM resolution is limited to 7x7
                </li>
              </ul>
            </GradientCard>

            <p className="text-muted-foreground">
              Remember the husky/wolf example from Visualizing Features. Your
              flower model might have its own version of this. The key question
              for each heatmap: <strong>&ldquo;If I showed this to someone
              who does not know ML, would they agree that the model is
              focusing on the right thing?&rdquo;</strong>
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The Litmus Test">
            A non-technical person should be able to look at your Grad-CAM
            overlay and say &ldquo;Yes, the model is looking at the
            flower.&rdquo; If the heatmap requires ML knowledge to explain
            away, that is a red flag.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          6. Notebook Link + Getting Started
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Open the Notebook"
            subtitle="Everything you need is scaffolded"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The notebook provides all the boilerplate: dataset loading,
              preprocessing, display utilities, training loop skeleton, and
              Grad-CAM implementation. Your job is to fill in the TODO
              sections&mdash;making the decisions about freezing, learning
              rates, and interpreting what Grad-CAM reveals.
            </p>
            <p className="text-muted-foreground">
              You have done every piece of this before. In Transfer Learning
              you froze a backbone and trained a head. In Visualizing Features
              you implemented Grad-CAM from scratch. The notebook guides you
              through combining them on a new dataset.
            </p>
            <div className="rounded-lg border-2 border-primary/50 bg-primary/5 p-6">
              <div className="space-y-4">
                <p className="text-muted-foreground">
                  Open the capstone project notebook in Google Colab. You
                  will:
                </p>
                <ul className="list-disc list-inside text-muted-foreground space-y-1 ml-4 text-sm">
                  <li>
                    Explore the Oxford Flowers dataset (8 species, ~50&ndash;80
                    images each)
                  </li>
                  <li>
                    Fine-tune a pretrained ResNet-18 via feature extraction
                  </li>
                  <li>
                    Validate model reasoning with Grad-CAM heatmaps
                  </li>
                  <li>
                    Optionally fine-tune with unfrozen layer4 and compare
                    results
                  </li>
                  <li>
                    Build a side-by-side comparison of both approaches
                  </li>
                </ul>
                <a
                  href="https://colab.research.google.com/github/brennancheung/CourseAI/blob/main/notebooks/3-3-2-transfer-learning-project.ipynb"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-2 px-4 py-2 rounded-md bg-primary text-primary-foreground font-medium text-sm hover:bg-primary/90 transition-colors"
                >
                  <ExternalLink className="w-4 h-4" />
                  Open in Google Colab
                </a>
                <p className="text-xs text-muted-foreground">
                  Make sure to select a GPU runtime (Runtime &rarr; Change
                  runtime type &rarr; T4 GPU). Expected time: 30&ndash;45
                  minutes.
                </p>
              </div>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="What&apos;s Provided">
            <ul className="space-y-1 text-sm">
              <li>Dataset download and filtering</li>
              <li>Image preprocessing and augmentation</li>
              <li>Training loop with loss/accuracy tracking</li>
              <li>Grad-CAM utility function</li>
              <li>Heatmap overlay display function</li>
            </ul>
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          7. Reflection Prompts (post-notebook)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Reflect on Your Results"
            subtitle="After completing the notebook"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Come back here after finishing the notebook. Take a few minutes
              to reflect on what you observed. These prompts are not graded
              and have no right answers&mdash;they are for your own
              consolidation.
            </p>
            <GradientCard title="Reflection Prompts" color="violet">
              <ol className="list-decimal list-inside space-y-3 text-sm">
                <li>
                  What accuracy did feature extraction achieve? Was it
                  sufficient for this task?
                </li>
                <li>
                  Did fine-tuning improve accuracy? By how much? Was the
                  added complexity worth it?
                </li>
                <li>
                  What did Grad-CAM reveal about what your model learned? Any
                  surprises?
                </li>
                <li>
                  Was the model right for the right reasons, or did you find
                  evidence of shortcut learning?
                </li>
                <li>
                  If you had 10x more data, what would you do differently?
                  Would you still start with feature extraction?
                </li>
                <li>
                  If you ran the optional filter viz and activation maps
                  section, how did those compare to Grad-CAM for
                  understanding your model? If you skipped it, would those
                  tools have added useful information here? Why or why not?
                </li>
              </ol>
            </GradientCard>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Why Reflect?">
            Executing the workflow is step one. Making sense of the
            results&mdash;connecting observations back to concepts&mdash;is
            where deep understanding forms. This is the difference between
            &ldquo;I ran the code&rdquo; and &ldquo;I understand the
            model.&rdquo;
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          8. Summary
          ================================================================ */}
      <Row>
        <Row.Content>
          <SummaryBlock
            title="The Practitioner Workflow"
            items={[
              {
                headline: 'Train with the simplest strategy first',
                description:
                  'Feature extraction—freeze the backbone, train the head. Start simple and measure before adding complexity.',
              },
              {
                headline:
                  'Evaluate with metrics, then validate with visualization',
                description:
                  'Accuracy tells you how often the model is right. Grad-CAM tells you whether it is right for the right reasons.',
              },
              {
                headline: 'Fine-tune only when the data justifies it',
                description:
                  'On small datasets, feature extraction may match or beat fine-tuning. The comparison is not about which is "better"—it is about which is appropriate for your data.',
              },
              {
                headline:
                  'Visualization is a debugging tool, not a victory lap',
                description:
                  'Correct prediction does not mean correct reasoning. The husky/wolf lesson is real and common. Always check what your model learned.',
              },
            ]}
          />
        </Row.Content>
      </Row>

      {/* ================================================================
          9. Series 3 Completion
          ================================================================ */}
      <Row>
        <Row.Content>
          <ModuleCompleteBlock
            module="3.3"
            title="Seeing What CNNs See"
            achievements={[
              'Visualized conv1 filters and confirmed they detect edges and color gradients',
              'Captured activation maps at multiple depths to observe the feature hierarchy',
              'Implemented Grad-CAM to produce class-specific spatial heatmaps',
              'Used visualization to diagnose model behavior on your own fine-tuned model',
              'Identified shortcut learning and understood why accuracy alone is insufficient',
            ]}
            nextModule="4.1"
            nextTitle="LLMs and Transformers"
          />
        </Row.Content>
      </Row>

      <Row>
        <Row.Content>
          <GradientCard title="Series 3 Complete: CNNs" color="emerald">
            <div className="text-sm space-y-3">
              <p>
                You started Series 3 asking &ldquo;what is a
                convolution?&rdquo; You ended it by fine-tuning a pretrained
                CNN on a custom dataset and using Grad-CAM to verify the
                model&apos;s reasoning.
              </p>
              <p>
                Along the way, you learned the convolution operation, built a
                CNN architecture from scratch, traced the evolution from LeNet
                to ResNet, discovered how skip connections make deep networks
                trainable, transferred knowledge from ImageNet to new tasks,
                and opened the black box to see what your models actually
                learned.
              </p>
              <p>
                The practical superpower you built is not &ldquo;can I get
                high accuracy?&rdquo;&mdash;it is <strong>&ldquo;can I
                understand what my model learned and trust its
                reasoning?&rdquo;</strong>
              </p>
            </div>
          </GradientCard>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The Real Skill">
            Any tutorial can get you to 90% accuracy with copy-paste code. The
            skill you built is knowing <em>why</em> the model works, seeing
            <em>what</em> it learned, and recognizing <em>when</em> it is
            fooling you.
          </InsightBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <GradientCard title="What Comes Next: Series 4" color="cyan">
            <div className="text-sm space-y-3">
              <p>
                Series 4 moves to a different architecture, a different data
                modality, and a different scale: <strong>Large Language
                Models</strong>. You will learn how transformers work, what
                attention mechanisms compute, and how models like GPT generate
                text.
              </p>
              <p>
                The practitioner mindset you built in this series carries
                forward. Different architecture, same principle:{' '}
                <strong>understand the model, do not just use it.</strong>
              </p>
            </div>
          </GradientCard>
        </Row.Content>
      </Row>
    </LessonLayout>
  )
}

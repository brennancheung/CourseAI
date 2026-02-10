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
  ModuleCompleteBlock,
} from '@/components/lessons'
import { CodeBlock } from '@/components/common/CodeBlock'
import { ExternalLink } from 'lucide-react'

/**
 * Transfer Learning
 *
 * Third and final lesson in Module 3.2 (Modern Architectures).
 * Teaches the student to use pretrained models (torchvision.models)
 * for image classification on small datasets via feature extraction
 * and fine-tuning.
 *
 * Core concepts at DEVELOPED:
 * - Pretrained models / torchvision.models API
 * - Feature extraction (freeze backbone, replace head)
 *
 * Concepts at INTRODUCED:
 * - Fine-tuning (unfreeze some layers + differential LR)
 * - Data augmentation transforms (RandomHorizontalFlip, RandomResizedCrop, ColorJitter)
 * - Differential learning rates (parameter groups in optimizer)
 *
 * Previous: ResNets and Skip Connections (module 3.2, lesson 2)
 * Next: Module 3.3 (visualization / interpretability)
 */

// ---------------------------------------------------------------------------
// Inline SVG: Frozen backbone + trainable head diagram
// ---------------------------------------------------------------------------

function TransferDiagram() {
  const w = 480
  const h = 280

  return (
    <div className="flex justify-center py-2">
      <svg
        width={w}
        height={h}
        viewBox={`0 0 ${w} ${h}`}
        className="text-muted-foreground"
      >
        {/* Frozen backbone region */}
        <rect
          x={20}
          y={30}
          width={280}
          height={220}
          rx={8}
          fill="rgba(56, 189, 248, 0.06)"
          stroke="rgba(56, 189, 248, 0.3)"
          strokeWidth={1.5}
          strokeDasharray="6 3"
        />
        <text
          x={160}
          y={20}
          textAnchor="middle"
          fontSize={11}
          fill="rgba(56, 189, 248, 0.8)"
          fontWeight={600}
        >
          Frozen Backbone (requires_grad=False)
        </text>

        {/* Layer boxes inside backbone */}
        {[
          { label: 'conv1', y: 45, sublabel: 'edges, gradients' },
          { label: 'layer1', y: 85, sublabel: 'textures, patterns' },
          { label: 'layer2', y: 125, sublabel: 'parts, shapes' },
          { label: 'layer3', y: 165, sublabel: 'object parts' },
          { label: 'layer4', y: 205, sublabel: 'high-level features' },
        ].map((layer) => (
          <g key={layer.label}>
            <rect
              x={40}
              y={layer.y}
              width={120}
              height={28}
              rx={4}
              fill="rgba(56, 189, 248, 0.12)"
              stroke="rgba(56, 189, 248, 0.4)"
              strokeWidth={1}
            />
            <text
              x={100}
              y={layer.y + 18}
              textAnchor="middle"
              fontSize={10}
              fill="currentColor"
              fontWeight={500}
            >
              {layer.label}
            </text>
            <text
              x={180}
              y={layer.y + 18}
              fontSize={9}
              fill="currentColor"
              opacity={0.5}
            >
              {sublabelArrow} {layer.sublabel}
            </text>
          </g>
        ))}

        {/* Arrows between layers */}
        {[75, 115, 155, 195].map((y) => (
          <line
            key={y}
            x1={100}
            y1={y}
            x2={100}
            y2={y + 8}
            stroke="currentColor"
            strokeWidth={1}
            opacity={0.3}
          />
        ))}

        {/* Arrow from backbone to head */}
        <line
          x1={160}
          y1={235}
          x2={160}
          y2={240}
          stroke="currentColor"
          strokeWidth={1}
          opacity={0.3}
        />
        <line
          x1={160}
          y1={240}
          x2={380}
          y2={240}
          stroke="currentColor"
          strokeWidth={1}
          opacity={0.3}
        />
        <line
          x1={380}
          y1={240}
          x2={380}
          y2={140}
          stroke="currentColor"
          strokeWidth={1}
          opacity={0.3}
        />

        {/* Trainable head region */}
        <rect
          x={330}
          y={50}
          width={130}
          height={90}
          rx={8}
          fill="rgba(34, 197, 94, 0.08)"
          stroke="rgba(34, 197, 94, 0.4)"
          strokeWidth={1.5}
        />
        <text
          x={395}
          y={42}
          textAnchor="middle"
          fontSize={11}
          fill="rgba(34, 197, 94, 0.8)"
          fontWeight={600}
        >
          New Head (trainable)
        </text>

        {/* Head layers */}
        <rect
          x={345}
          y={60}
          width={100}
          height={28}
          rx={4}
          fill="rgba(34, 197, 94, 0.15)"
          stroke="rgba(34, 197, 94, 0.5)"
          strokeWidth={1}
        />
        <text
          x={395}
          y={78}
          textAnchor="middle"
          fontSize={10}
          fill="currentColor"
          fontWeight={500}
        >
          AdaptiveAvgPool
        </text>

        <line
          x1={395}
          y1={90}
          x2={395}
          y2={100}
          stroke="currentColor"
          strokeWidth={1}
          opacity={0.3}
        />

        <rect
          x={345}
          y={100}
          width={100}
          height={28}
          rx={4}
          fill="rgba(34, 197, 94, 0.15)"
          stroke="rgba(34, 197, 94, 0.5)"
          strokeWidth={1}
        />
        <text
          x={395}
          y={118}
          textAnchor="middle"
          fontSize={10}
          fill="currentColor"
          fontWeight={500}
        >
          Linear(512, N)
        </text>

        {/* Output label */}
        <text
          x={395}
          y={155}
          textAnchor="middle"
          fontSize={10}
          fill="currentColor"
          opacity={0.6}
        >
          N = your classes
        </text>
      </svg>
    </div>
  )
}

const sublabelArrow = '→'

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export function TransferLearningLesson() {
  return (
    <LessonLayout>
      {/* ================================================================
          1. Header
          ================================================================ */}
      <Row>
        <Row.Content>
          <LessonHeader
            title="Transfer Learning"
            description="Take a pretrained ResNet and adapt it to new tasks in minutes&mdash;the practical payoff of everything you have learned about CNN architectures."
            category="Modern Architectures"
          />
        </Row.Content>
      </Row>

      {/* ================================================================
          1. Context + Constraints
          ================================================================ */}
      <Row>
        <Row.Content>
          <ObjectiveBlock>
            Use a pretrained ResNet to classify images on a small dataset by
            freezing the backbone and replacing the classification head. Understand
            when to use feature extraction vs fine-tuning, and why transfer
            learning works.
          </ObjectiveBlock>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="What You Already Know">
            You built a ResNet from scratch in the previous lesson and trained it
            on CIFAR-10. You understand hierarchical features (edges, textures,
            parts, objects) and the{' '}
            <code className="text-xs bg-muted px-1 rounded">requires_grad</code>{' '}
            flag from autograd. This lesson puts all of that to practical use.
          </TipBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <ConstraintBlock
            title="Scope for This Lesson"
            items={[
              'Loading pretrained models with torchvision.models',
              'Feature extraction: freeze backbone, replace and train the classification head',
              'Fine-tuning: selectively unfreeze layers with differential learning rates',
              'Data augmentation basics for small datasets',
              'Decision framework: when to use feature extraction vs fine-tuning',
              'NOT: Training on ImageNet from scratch',
              'NOT: Object detection, segmentation, or tasks beyond classification',
              'NOT: EfficientNet, ViT, or architectures beyond ResNet',
              'NOT: Knowledge distillation, self-supervised pretraining',
            ]}
          />
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="Module Arc">
            Architecture Evolution traced how CNNs got deeper. ResNets solved the
            depth limit. This lesson completes the arc: use those pretrained
            architectures for real tasks without massive datasets.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          2. Hook: The Small Dataset Problem
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Small Dataset Problem"
            subtitle="You have 1,500 images and 11 million parameters"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              You have built a ResNet from scratch and trained it on 50,000
              CIFAR-10 images. Now imagine a realistic scenario: you want to
              classify just 3 types of animals&mdash;cats, dogs, and
              horses&mdash;and you have about 500 images per class. That is 1,500
              images total.
            </p>
            <p className="text-muted-foreground">
              You try training a ResNet-18 from scratch on this small dataset.
              The result is predictable:
            </p>
            <ComparisonRow
              left={{
                title: 'Training from Scratch',
                color: 'rose',
                items: [
                  'Training accuracy: 99%',
                  'Validation accuracy: ~35%',
                  '11M parameters, 1,500 images',
                  'Massive overfitting',
                ],
              }}
              right={{
                title: 'With Transfer Learning',
                color: 'emerald',
                items: [
                  'Training accuracy: 95%',
                  'Validation accuracy: 85%+',
                  'Same architecture, same data',
                  'A few lines of code changed',
                ],
              }}
            />
            <p className="text-muted-foreground">
              Same model. Same data. The only difference is the{' '}
              <strong>starting point</strong>. Training from random weights fails
              because 1,500 images are nowhere near enough to learn general visual
              features. But what if those features were already learned?
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The Key Question">
            You already know that early CNN layers learn general features
            (edges, textures) while later layers learn task-specific features
            (dog breeds, car models). If those general features are
            universal&mdash;useful for ANY image task&mdash;why learn them again?
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          3. Explain: Why Transfer Learning Works
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Why Transfer Learning Works"
            subtitle="General features are universal"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              In Building a CNN and Architecture Evolution, you learned that CNNs
              build a feature hierarchy: early layers detect edges and gradients,
              middle layers detect textures and patterns, and later layers detect
              object parts and whole objects.
            </p>
            <p className="text-muted-foreground">
              Here is the insight that makes transfer learning possible:{' '}
              <strong>
                those early and middle features are not specific to ImageNet
              </strong>
              . Edge detectors, texture recognizers, and gradient filters are useful
              whether you are classifying dogs, flowers, medical images, or
              satellite photos. A vertical edge is a vertical edge regardless of
              what it belongs to.
            </p>

            <GradientCard title="Analogy: Hiring an Experienced Employee" color="blue">
              <div className="text-sm space-y-2">
                <p>
                  You do not train a new hire from scratch to recognize edges,
                  textures, and shapes. You <strong>hire someone who already has
                  those skills</strong> (the pretrained backbone) and teach them the
                  specifics of YOUR job (the new classification head).
                </p>
                <p>
                  <strong>Feature extraction</strong> = &ldquo;Just learn our
                  product catalog.&rdquo; The employee keeps all their existing
                  skills intact.
                </p>
                <p>
                  <strong>Fine-tuning</strong> = &ldquo;Also adjust your general
                  skills slightly for our industry.&rdquo; Some existing skills
                  get refined, but you do not start over.
                </p>
              </div>
            </GradientCard>

            <TransferDiagram />
            <p className="text-xs text-muted-foreground text-center">
              Left: the pretrained backbone (frozen) extracts general features.
              Right: the new classification head (trainable) maps those features
              to your specific classes.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The Core Realization">
            A model trained on 1.2 million ImageNet images has already done the
            hardest work&mdash;learning to see. You do not need to re-learn
            edge detection. You just need to teach the model what YOUR task cares
            about.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          4. Explain: The torchvision.models API
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The torchvision.models API"
            subtitle="Loading a pretrained ResNet in one line"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              You have used{' '}
              <code className="text-xs bg-muted px-1 rounded">torchvision.datasets</code>{' '}
              to load MNIST and CIFAR-10. The same library provides pretrained
              models:
            </p>
            <CodeBlock
              code={`import torchvision.models as models
from torchvision.models import ResNet18_Weights

# Load ResNet-18 with pretrained ImageNet weights
model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

# This is just an nn.Module — print it!
print(model)`}
              language="python"
              filename="load_pretrained.py"
            />
            <p className="text-muted-foreground">
              You might assume a pretrained model is a sealed unit&mdash;load it
              and use it as-is. Not so. The model you get back is the same{' '}
              <code className="text-xs bg-muted px-1 rounded">nn.Module</code>{' '}
              you have been building with&mdash;the same LEGO bricks. You can
              print it, access its layers by name, and replace any component:
            </p>
            <CodeBlock
              code={`# Inspect specific layers — same attribute access as your own Modules
print(model.conv1)   # Conv2d(3, 64, kernel_size=7, ...)
print(model.layer1)  # Sequential of ResidualBlocks
print(model.fc)      # Linear(in_features=512, out_features=1000)`}
              language="python"
              filename="inspect_model.py"
            />
            <p className="text-muted-foreground">
              Notice <code className="text-xs bg-muted px-1 rounded">model.fc</code>:
              it is a <code className="text-xs bg-muted px-1 rounded">Linear(512, 1000)</code>.
              The 512 comes from global average pooling (you learned this in
              ResNets and Skip Connections), and 1000 is the number of ImageNet
              classes. If your task has a different number of classes, you need to
              replace this layer.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="ImageNet Requirements">
            Pretrained models expect specific input:
            <ul className="space-y-1 text-sm mt-2">
              <li>
                <strong>Size:</strong> 224x224 pixels
              </li>
              <li>
                <strong>Normalization:</strong> mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
              </li>
              <li>
                <strong>Channels:</strong> 3 (RGB)
              </li>
            </ul>
            These are fixed by how the model was trained on ImageNet.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          5. Check 1: Predict and Verify
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Check: What Is model.fc?"
            subtitle="Predict before you peek"
          />
          <GradientCard title="Predict and Verify" color="emerald">
            <div className="space-y-3 text-sm">
              <p>
                <strong>Question 1:</strong> You load ResNet-18 with pretrained
                weights and access{' '}
                <code className="text-xs bg-muted px-1 rounded">model.fc</code>.
                What do you expect to see? How many input features? How many
                output classes?
              </p>
              <details>
                <summary className="font-medium cursor-pointer text-primary">
                  Reveal
                </summary>
                <div className="mt-2 space-y-1">
                  <p>
                    <code className="text-xs bg-muted px-1 rounded">
                      Linear(in_features=512, out_features=1000)
                    </code>
                    &mdash;512 features from global average pooling over the final
                    feature maps, and 1000 ImageNet classes.
                  </p>
                </div>
              </details>
              <p className="pt-2">
                <strong>Question 2:</strong> If your task has 3 classes (like
                our cats, dogs, and horses scenario), what needs to change?
              </p>
              <details>
                <summary className="font-medium cursor-pointer text-primary">
                  Reveal
                </summary>
                <div className="mt-2 space-y-1">
                  <p>
                    Replace <code className="text-xs bg-muted px-1 rounded">model.fc</code>{' '}
                    with <code className="text-xs bg-muted px-1 rounded">nn.Linear(512, 3)</code>.
                    The input features stay at 512 (the backbone output does not
                    change). Only the number of output classes changes.
                  </p>
                </div>
              </details>
            </div>
          </GradientCard>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Same Bricks, Different Building">
            A pretrained ResNet is not a sealed black box. It is the same{' '}
            <code className="text-xs bg-muted px-1 rounded">nn.Module</code>{' '}
            pattern you know. You can print it, access its layers, and replace
            parts&mdash;just like your own models.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          6. Explain: Feature Extraction (Strategy 1)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Feature Extraction"
            subtitle="Freeze everything, train only the head"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Feature extraction is the simplest transfer learning strategy. Three
              steps:
            </p>
            <div className="py-4 px-6 bg-muted/50 rounded-lg space-y-2">
              <p className="text-sm font-medium text-muted-foreground">
                The Three Steps of Feature Extraction:
              </p>
              <ol className="list-decimal list-inside text-sm text-muted-foreground space-y-2 ml-2">
                <li>
                  <strong>Load the pretrained model</strong>
                </li>
                <li>
                  <strong>Freeze all parameters:</strong>{' '}
                  <code className="text-xs bg-muted px-1 rounded">
                    for param in model.parameters(): param.requires_grad = False
                  </code>
                </li>
                <li>
                  <strong>Replace the head:</strong>{' '}
                  <code className="text-xs bg-muted px-1 rounded">
                    model.fc = nn.Linear(512, num_classes)
                  </code>
                  &mdash;this new layer has{' '}
                  <code className="text-xs bg-muted px-1 rounded">requires_grad=True</code>{' '}
                  by default
                </li>
              </ol>
            </div>
            <CodeBlock
              code={`import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights

# Step 1: Load pretrained
model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

# Step 2: Freeze all parameters
for param in model.parameters():
    param.requires_grad = False

# Step 3: Replace the classification head
num_classes = 3  # cats, dogs, horses
model.fc = nn.Linear(512, num_classes)
# model.fc.weight and model.fc.bias have requires_grad=True

# Only the head's parameters are optimized
optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)`}
              language="python"
              filename="feature_extraction.py"
            />
            <p className="text-muted-foreground">
              You already know{' '}
              <code className="text-xs bg-muted px-1 rounded">requires_grad</code>{' '}
              from the autograd lesson&mdash;it controls whether PyTorch tracks
              gradients for a tensor. Here you are using the same mechanism at a
              larger scale: freezing entire layers instead of individual tensors.
            </p>
            <p className="text-muted-foreground">
              Since this is a <strong>classification</strong> task (not regression),
              we use{' '}
              <code className="text-xs bg-muted px-1 rounded">nn.CrossEntropyLoss</code>{' '}
              instead of{' '}
              <code className="text-xs bg-muted px-1 rounded">nn.MSELoss</code>.
              It combines log-softmax and negative log-likelihood into one
              operation and is the standard loss for multi-class classification.
              The training loop itself is unchanged&mdash;just swap the loss
              function.
            </p>
            <p className="text-muted-foreground">
              <strong>Why this works:</strong> Only the head&apos;s parameters need
              gradients, so training is fast and memory-efficient. On a small
              dataset, this takes minutes, not hours. The frozen backbone acts as a
              fixed feature extractor&mdash;it transforms your images into
              512-dimensional feature vectors, and the new head learns which
              features correspond to which class.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="BatchNorm Subtlety">
            Setting{' '}
            <code className="text-xs bg-muted px-1 rounded">requires_grad=False</code>{' '}
            stops gradient computation but does <strong>not</strong> stop
            BatchNorm from updating its running statistics in train mode. It
            is{' '}
            <code className="text-xs bg-muted px-1 rounded">model.eval()</code>{' '}
            that switches BN to use stored running averages. In practice, calling{' '}
            <code className="text-xs bg-muted px-1 rounded">model.train()</code>{' '}
            during the training loop works fine for feature extraction&mdash;but
            know that eval mode is what truly freezes BN behavior.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          7. Data Augmentation for Small Datasets
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Data Augmentation for Small Datasets"
            subtitle="Making 1,500 images act like 15,000"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Before we practice in the notebook, one more technique for small
              datasets. You already know{' '}
              <code className="text-xs bg-muted px-1 rounded">transforms.Compose</code>,{' '}
              <code className="text-xs bg-muted px-1 rounded">ToTensor()</code>, and{' '}
              <code className="text-xs bg-muted px-1 rounded">Normalize()</code>{' '}
              from Datasets and DataLoaders. For small datasets, add random
              augmentations to prevent overfitting:
            </p>
            <CodeBlock
              code={`from torchvision import transforms

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),       # random crop + resize
    transforms.RandomHorizontalFlip(p=0.5),  # mirror with 50% chance
    transforms.ColorJitter(                  # vary lighting
        brightness=0.2, contrast=0.2
    ),
    transforms.ToTensor(),
    transforms.Normalize(                    # ImageNet statistics
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# Validation: NO random augmentation (deterministic evaluation)
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])`}
              language="python"
              filename="augmentation.py"
            />
            <p className="text-muted-foreground">
              Each augmented image is slightly different&mdash;a different crop, a
              random flip, slightly varied brightness. The model cannot memorize
              exact pixel patterns because it never sees the same image twice.
              This connects directly to what you learned about overfitting in
              Overfitting and Regularization: augmentation is a form of data-level
              regularization.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Train vs Val Transforms">
            Random augmentation is only for training. Validation must be
            deterministic (always the same crop, no random flips) so you can
            compare accuracy consistently across epochs. This is the same
            train/eval distinction you know from batch normalization.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          8. Notebook: Feature Extraction
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Practice: Feature Extraction in Code"
            subtitle="Try it yourself in a Colab notebook"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The notebook walks you through the full process: load a pretrained
              ResNet-18, freeze the backbone, replace the head, set up data
              augmentation, and train on a small dataset. You also train from
              scratch on the same data to see the difference firsthand.
            </p>
            <div className="rounded-lg border-2 border-primary/50 bg-primary/5 p-6">
              <div className="space-y-4">
                <p className="text-muted-foreground">
                  Open the scaffolded notebook in Google Colab. Data loading and
                  training loops are provided&mdash;your job is to set up the
                  pretrained model and compare approaches.
                </p>
                <a
                  href="https://colab.research.google.com/github/brennancheung/CourseAI/blob/main/notebooks/3-2-3-transfer-learning.ipynb"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-2 px-4 py-2 rounded-md bg-primary text-primary-foreground font-medium text-sm hover:bg-primary/90 transition-colors"
                >
                  <ExternalLink className="w-4 h-4" />
                  Open in Google Colab
                </a>
                <p className="text-xs text-muted-foreground">
                  The notebook includes: training from scratch (baseline),
                  feature extraction, fine-tuning, and a comparison of all three.
                  Expected training time: ~5-10 minutes on a Colab GPU.
                </p>
              </div>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="What&apos;s Provided">
            <ul className="space-y-1 text-sm">
              <li>Small dataset setup (CIFAR-10 subset, 3 classes)</li>
              <li>Data augmentation transforms</li>
              <li>Complete training + evaluation loop</li>
              <li>Accuracy comparison and plotting code</li>
            </ul>
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          8. Elaborate: Why Early Layers Transfer
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Why Early Layers Transfer Across Domains"
            subtitle='Addressing the "only works on similar datasets" misconception'
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              You might wonder: can a model trained on ImageNet (dogs, cars,
              birds) really help with medical images or satellite photos?
              The domains seem completely different.
            </p>
            <p className="text-muted-foreground">
              Consider what the early layers actually detect. A Gabor-like edge
              detector in <code className="text-xs bg-muted px-1 rounded">conv1</code>{' '}
              responds to vertical edges&mdash;and a vertical edge is a vertical
              edge whether it belongs to a cat ear, a tumor boundary, or a
              building roof. Texture detectors in{' '}
              <code className="text-xs bg-muted px-1 rounded">layer1</code> and{' '}
              <code className="text-xs bg-muted px-1 rounded">layer2</code>{' '}
              recognize patterns like stripes, dots, and smooth gradients&mdash;patterns
              that appear in every visual domain.
            </p>
            <div className="py-4 px-6 bg-muted/50 rounded-lg space-y-2">
              <p className="text-sm font-medium text-muted-foreground">
                The transferability spectrum:
              </p>
              <ul className="list-disc list-inside text-sm text-muted-foreground space-y-1 ml-2">
                <li>
                  <strong>conv1, layer1:</strong> Highly universal (edges,
                  gradients, textures). Transfer everywhere.
                </li>
                <li>
                  <strong>layer2, layer3:</strong> Moderately universal (shapes,
                  patterns, parts). Transfer well to most image domains.
                </li>
                <li>
                  <strong>layer4:</strong> Increasingly task-specific (object
                  parts composed for ImageNet categories). May need adaptation
                  for very different domains.
                </li>
                <li>
                  <strong>fc:</strong> Fully task-specific (1000 ImageNet classes).
                  Always replaced.
                </li>
              </ul>
            </div>
            <p className="text-muted-foreground">
              This is why fine-tuning later layers (but not early ones) makes
              sense for different domains. The deeper you go, the more
              task-specific the features become&mdash;and the more likely they
              need adaptation.
            </p>
            <p className="text-muted-foreground">
              ImageNet-pretrained models have been successfully used for medical
              imaging (X-rays, pathology), satellite imagery, art
              classification, and industrial defect detection. The features that
              matter most&mdash;low-level visual features&mdash;are truly
              domain-agnostic.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Common Misconception">
            &ldquo;Transfer learning only works when the new task looks like
            ImageNet.&rdquo; Not true. Early-layer features (edges, textures,
            color gradients) are universal to all natural images. Even for
            domains that look nothing like ImageNet, the pretrained backbone
            provides a massive head start.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          9. Explain: Fine-Tuning (Strategy 2)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Fine-Tuning"
            subtitle="When feature extraction is not enough"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Feature extraction treats the backbone as a fixed feature extractor.
              But what if the frozen features are not quite right for your task?
              For example, if your domain is very different from natural photos
              (medical imaging, electron microscopy), the later layers might need
              adjustment.
            </p>
            <p className="text-muted-foreground">
              <strong>Fine-tuning</strong> means unfreezing some of the pretrained
              layers and training them alongside the new head. The key practical
              skill is <strong>differential learning rates</strong>: pretrained
              layers get a much lower learning rate than the new head.
            </p>
            <CodeBlock
              code={`import torch.optim as optim

# Unfreeze the last ResNet stage
for param in model.layer4.parameters():
    param.requires_grad = True

# Differential learning rates via parameter groups
optimizer = optim.Adam([
    {'params': model.fc.parameters(), 'lr': 1e-3},      # new head: fast
    {'params': model.layer4.parameters(), 'lr': 1e-5},   # pretrained: slow
])`}
              language="python"
              filename="fine_tuning.py"
            />
            <p className="text-muted-foreground">
              Same optimizer, same{' '}
              <code className="text-xs bg-muted px-1 rounded">.step()</code> and{' '}
              <code className="text-xs bg-muted px-1 rounded">.zero_grad()</code>.
              The only new thing is <strong>parameter groups</strong>&mdash;you
              pass a list of dictionaries, each with its own learning rate. The
              training loop is unchanged.
            </p>
            <p className="text-muted-foreground">
              Why the low learning rate for pretrained layers? These weights
              represent features learned from 1.2 million images. A high learning
              rate would <strong>destroy</strong> those features in a few batches,
              undoing millions of images worth of training. A tiny learning rate
              allows gentle adaptation without catastrophic forgetting.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The Spectrum">
            Transfer learning is a spectrum, not a binary choice:
            <ul className="space-y-1 text-sm mt-2">
              <li>
                <strong>Freeze all</strong> + train head = feature extraction
              </li>
              <li>
                <strong>Freeze early</strong> + fine-tune later layers + head
              </li>
              <li>
                <strong>Fine-tune all</strong> with very low LR
              </li>
            </ul>
            Start with the simplest (freeze all). Only unfreeze more if accuracy
            is not good enough.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          10. Check 2: Transfer Question
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Check: Debugging a Transfer Attempt"
            subtitle="Apply what you just learned"
          />
          <GradientCard title="Transfer Question" color="emerald">
            <div className="space-y-3 text-sm">
              <p>
                A colleague has 200 images of 3 types of manufacturing defects
                (very different from ImageNet). They fine-tune ALL layers of
                ResNet-50 with <code className="text-xs bg-muted px-1 rounded">lr=0.01</code>{' '}
                and get terrible results. What went wrong? What would you
                recommend?
              </p>
              <details>
                <summary className="font-medium cursor-pointer text-primary">
                  Reveal answer
                </summary>
                <div className="mt-2 space-y-2">
                  <p>
                    <strong>Problem:</strong>{' '}
                    <code className="text-xs bg-muted px-1 rounded">lr=0.01</code>{' '}
                    is far too high for pretrained layers. It destroys the learned
                    features that took millions of images to learn. With only 200
                    images, there is not enough data to recover.
                  </p>
                  <p>
                    <strong>Recommendation:</strong> Start with feature extraction
                    (freeze everything, train only the new head with{' '}
                    <code className="text-xs bg-muted px-1 rounded">lr=1e-3</code>).
                    If that is not enough, try fine-tuning only{' '}
                    <code className="text-xs bg-muted px-1 rounded">layer4</code>{' '}
                    with <code className="text-xs bg-muted px-1 rounded">lr=1e-5</code>.
                    The domain is different from ImageNet, but the low-level
                    features (edges, textures) still transfer.
                  </p>
                </div>
              </details>
            </div>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* ================================================================
          11. Notebook: Fine-Tuning
          ================================================================ */}
      <Row>
        <Row.Content>
          <TipBlock title="Try It: Fine-Tuning in the Notebook">
            The notebook you opened earlier also includes fine-tuning exercises
            (sections 5&ndash;6). After completing the feature extraction section,
            try the fine-tuning TODOs to see differential learning rates in action.
            Then compare all three approaches side by side in section 6.
          </TipBlock>
        </Row.Content>
      </Row>

      {/* ================================================================
          12. Elaborate: Decision Framework
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Decision Framework"
            subtitle="When to use which strategy"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Here is a practical decision guide based on two factors: how much
              data you have and how similar your domain is to the pretrained
              data (ImageNet = natural photos).
            </p>
            <div className="overflow-x-auto">
              <table className="w-full text-sm text-muted-foreground">
                <thead>
                  <tr className="border-b border-muted/30">
                    <th className="text-left py-2 pr-4 font-medium">Your Dataset</th>
                    <th className="text-left py-2 pr-4 font-medium">Strategy</th>
                    <th className="text-left py-2 font-medium">Why</th>
                  </tr>
                </thead>
                <tbody>
                  <tr className="border-b border-muted/10">
                    <td className="py-2 pr-4">Small + similar domain</td>
                    <td className="py-2 pr-4">Feature extraction</td>
                    <td className="py-2">
                      Pretrained features already match; not enough data to safely fine-tune
                    </td>
                  </tr>
                  <tr className="border-b border-muted/10">
                    <td className="py-2 pr-4">Small + different domain</td>
                    <td className="py-2 pr-4">
                      Feature extraction (possibly fine-tune last stage)
                    </td>
                    <td className="py-2">
                      Early features still useful; too little data for deep fine-tuning
                    </td>
                  </tr>
                  <tr className="border-b border-muted/10">
                    <td className="py-2 pr-4">Large + similar domain</td>
                    <td className="py-2 pr-4">Fine-tune everything (low LR)</td>
                    <td className="py-2">
                      Enough data to safely refine all layers
                    </td>
                  </tr>
                  <tr>
                    <td className="py-2 pr-4">Large + different domain</td>
                    <td className="py-2 pr-4">
                      Fine-tune everything (possibly from scratch)
                    </td>
                    <td className="py-2">
                      May need to adapt early features too
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
            <div className="py-4 px-6 bg-muted/50 rounded-lg space-y-2">
              <p className="text-sm font-medium text-muted-foreground">
                Practical rules of thumb:
              </p>
              <ul className="list-disc list-inside text-sm text-muted-foreground space-y-1 ml-2">
                <li>
                  <strong>Always start with feature extraction.</strong> It is
                  fast, hard to mess up, and gives a strong baseline.
                </li>
                <li>
                  Only fine-tune if feature extraction is not good enough AND you
                  have enough data.
                </li>
                <li>
                  Data augmentation is your friend on small datasets&mdash;it
                  artificially increases diversity.
                </li>
              </ul>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="More Data Does Not Always Win">
            You might think transfer learning is just a crutch for small
            datasets. But even with 50K images, starting from pretrained weights
            converges faster and often achieves better accuracy than training from
            scratch. Transfer learning gives you a head start, not just a
            workaround.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          13. Summary
          ================================================================ */}
      <Row>
        <Row.Content>
          <SummaryBlock
            title="Key Takeaways"
            items={[
              {
                headline:
                  'Pretrained models carry general visual knowledge that transfers across tasks and domains.',
                description:
                  'Early CNN layers learn universal features (edges, textures) that are useful for nearly any image task. You do not need to re-learn them.',
              },
              {
                headline:
                  'Feature extraction is the simple, safe default—start here.',
                description:
                  'Freeze the backbone, replace the head, train only the new layer. A few lines of code separate "overfitting on a small dataset" from "85%+ accuracy."',
              },
              {
                headline:
                  'Fine-tuning gives more flexibility when you have enough data.',
                description:
                  'Unfreeze later layers with a low learning rate (differential LR). The training loop is unchanged—just parameter groups in the optimizer.',
              },
              {
                headline:
                  'Mental model: "Hire experienced, train specific."',
                description:
                  'The pretrained backbone is the experienced employee. The new head is the job-specific training. Feature extraction = "learn our catalog." Fine-tuning = "adjust your skills for our industry."',
              },
            ]}
          />
        </Row.Content>
      </Row>

      {/* Mental model */}
      <Row>
        <Row.Content>
          <InsightBlock title="Looking Ahead">
            Understanding pretrained models is not just useful for image
            classification. The same principle&mdash;start from a model that has
            already learned general features and adapt it to your task&mdash;is
            the foundation of how we use large language models (fine-tuning GPT)
            and diffusion models (fine-tuning Stable Diffusion). Transfer
            learning is the default way practitioners use deep learning.
          </InsightBlock>
        </Row.Content>
      </Row>

      {/* ================================================================
          14. Module Complete
          ================================================================ */}
      <Row>
        <Row.Content>
          <ModuleCompleteBlock
            module="3.2"
            title="Modern Architectures"
            achievements={[
              'Architecture evolution from LeNet to AlexNet to VGG',
              'ResNets: residual connections and the degradation problem',
              'Transfer learning: feature extraction, fine-tuning, and data augmentation for small datasets',
              'The decision framework for choosing a transfer learning strategy',
            ]}
            nextModule="3.3"
            nextTitle="Visualizing What CNNs Learn"
          />
        </Row.Content>
      </Row>
    </LessonLayout>
  )
}

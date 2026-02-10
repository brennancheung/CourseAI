'use client'

import { LessonLayout } from '@/components/lessons/LessonLayout'
import { Row } from '@/components/layout/Row'
import {
  LessonHeader,
  ObjectiveBlock,
  SectionHeader,
  InsightBlock,
  TipBlock,
  WarningBlock,
  ConceptBlock,
  SummaryBlock,
  ConstraintBlock,
  ComparisonRow,
  NextStepBlock,
} from '@/components/lessons'
import { CodeBlock } from '@/components/common/CodeBlock'
import { ExternalLink } from 'lucide-react'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine,
  ResponsiveContainer,
} from 'recharts'

/**
 * Saving, Loading, and Checkpoints -- Lesson 1 of Module 2.3 (Practical Patterns)
 *
 * BUILD lesson: teaches the student to persist and restore model state
 * using state_dict, and implement checkpointing to resume interrupted training.
 *
 * New concepts (2, no new ML theory):
 *   1. state_dict (model and optimizer) as the canonical persistence format (DEVELOPED)
 *   2. Checkpoint pattern (bundling model + optimizer + epoch + loss) (DEVELOPED)
 *
 * Also covers (lighter touch):
 *   - map_location for cross-device loading (INTRODUCED)
 *   - torch.save(model) pitfalls (INTRODUCED)
 *
 * Central insight: state_dict is just a dictionary of tensors (a snapshot of
 * all the knobs). Saving state_dict is preferred over saving the full model
 * because it decouples learned values from code structure.
 */

// Data for the "resume with vs without optimizer state" loss curve comparison.
// With optimizer state: smooth continuation. Without: loss spike then recovery.
const resumeComparisonData = [
  { epoch: 1, withOpt: 0.82, withoutOpt: 0.82 },
  { epoch: 2, withOpt: 0.61, withoutOpt: 0.61 },
  { epoch: 3, withOpt: 0.45, withoutOpt: 0.45 },
  { epoch: 4, withOpt: 0.34, withoutOpt: 0.34 },
  { epoch: 5, withOpt: 0.27, withoutOpt: 0.27 },
  // -- checkpoint saved at epoch 5, training resumes --
  { epoch: 6, withOpt: 0.22, withoutOpt: 0.58 },
  { epoch: 7, withOpt: 0.18, withoutOpt: 0.49 },
  { epoch: 8, withOpt: 0.15, withoutOpt: 0.41 },
  { epoch: 9, withOpt: 0.13, withoutOpt: 0.35 },
  { epoch: 10, withOpt: 0.11, withoutOpt: 0.30 },
  { epoch: 11, withOpt: 0.10, withoutOpt: 0.26 },
  { epoch: 12, withOpt: 0.09, withoutOpt: 0.23 },
]

function ResumeComparisonChart() {
  return (
    <div className="rounded-lg border bg-[#1a1a2e] p-4 space-y-3">
      <div className="flex items-center gap-3 border-b border-white/10 pb-2">
        <span className="text-xs font-mono text-white/70">
          Training Loss: Resume With vs Without Optimizer State
        </span>
      </div>

      <ResponsiveContainer width="100%" height={220}>
        <LineChart
          data={resumeComparisonData}
          margin={{ top: 5, right: 20, bottom: 5, left: 10 }}
        >
          <CartesianGrid
            strokeDasharray="3 3"
            stroke="rgba(255,255,255,0.07)"
          />
          <XAxis
            dataKey="epoch"
            stroke="rgba(255,255,255,0.3)"
            tick={{ fontSize: 10, fill: 'rgba(255,255,255,0.4)' }}
            label={{
              value: 'epoch',
              position: 'insideBottom',
              offset: -2,
              fontSize: 10,
              fill: 'rgba(255,255,255,0.3)',
            }}
          />
          <YAxis
            stroke="rgba(255,255,255,0.3)"
            tick={{ fontSize: 10, fill: 'rgba(255,255,255,0.4)' }}
            domain={[0, 1.0]}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: '#1e1e3a',
              border: '1px solid rgba(255,255,255,0.15)',
              borderRadius: '6px',
              fontSize: 11,
            }}
            labelStyle={{ color: 'rgba(255,255,255,0.6)' }}
          />
          <Legend
            wrapperStyle={{
              fontSize: 10,
              color: 'rgba(255,255,255,0.6)',
            }}
          />
          <ReferenceLine
            x={5.5}
            stroke="rgba(255,255,255,0.4)"
            strokeDasharray="4 4"
            label={{
              value: 'Resume',
              position: 'top',
              fill: 'rgba(255,255,255,0.5)',
              fontSize: 10,
            }}
          />
          <Line
            type="monotone"
            dataKey="withOpt"
            name="With optimizer state"
            stroke="#34d399"
            strokeWidth={2}
            dot={false}
          />
          <Line
            type="monotone"
            dataKey="withoutOpt"
            name="Without optimizer state"
            stroke="#fb7185"
            strokeWidth={2}
            dot={false}
          />
        </LineChart>
      </ResponsiveContainer>

      <div className="flex items-center gap-4 border-t border-white/10 pt-2">
        <span className="text-[10px] font-mono text-white/40">
          Checkpoint saved at epoch 5. Training resumed at epoch 6.
        </span>
      </div>
    </div>
  )
}

export function SavingAndLoadingLesson() {
  return (
    <LessonLayout>
      {/* 1. Header */}
      <Row>
        <Row.Content>
          <LessonHeader
            title="Saving, Loading, and Checkpoints"
            description="Make your trained models durable—save state, resume interrupted training, and never lose progress."
            category="Practical Patterns"
          />
        </Row.Content>
      </Row>

      {/* Objective */}
      <Row>
        <Row.Content>
          <ObjectiveBlock>
            Learn to save a trained model using state_dict, load it later for
            inference, and implement checkpointing so you can resume interrupted
            training without losing progress. By the end, you will have the
            complete save/load/checkpoint pattern that every PyTorch workflow
            depends on.
          </ObjectiveBlock>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Code, Not Theory">
            This is a BUILD lesson. No new ML theory&mdash;just the PyTorch API
            patterns for persistence. The cognitive demand is &ldquo;learn new
            API calls,&rdquo; not &ldquo;understand a new idea.&rdquo;
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* 1. Context + Constraints */}
      <Row>
        <Row.Content>
          <ConstraintBlock
            items={[
              'state_dict save/load and checkpoint patterns only',
              'No TorchScript, ONNX export, or model compilation',
              'No distributed checkpointing or multi-GPU saving',
              'No experiment management tools (MLflow, Weights & Biases)',
              'No model versioning or production deployment',
              'GPU-specific loading covered briefly (map_location)—full GPU training is next lesson',
            ]}
          />
        </Row.Content>
      </Row>

      {/* 2. Hook -- The Durability Problem */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Durability Problem"
            subtitle="Close the notebook and the model is gone"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              You just spent 45 minutes training your MNIST model to 97%
              accuracy. You close the notebook. The model is gone. Every weight,
              every bias, every learned feature&mdash;erased. You would need to
              retrain from scratch.
            </p>

            <p className="text-muted-foreground">
              Now imagine a more realistic scenario: you are training a larger
              model on a real dataset. It takes an hour per run. At epoch 47 of
              50, your process crashes&mdash;a kernel restart, a power blip, an
              out-of-memory error. Three hours of training, gone. No way to
              recover. No way to pick up where you left off.
            </p>

            <p className="text-muted-foreground">
              Or imagine wanting to share your best model with a friend. Or
              loading it next week for inference on new data. Right now, you
              cannot do any of this. Every model you have trained exists only in
              your notebook&rsquo;s memory.
            </p>

            <p className="text-muted-foreground">
              This lesson solves the <strong>durability problem</strong>: how to
              save a trained model so it survives beyond the current session, and
              how to checkpoint during training so you never lose progress.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Not Optional">
            Every production ML workflow depends on saving and loading models.
            It is not a nice-to-have&mdash;it is the bridge between
            &ldquo;training works&rdquo; and &ldquo;training is useful.&rdquo;
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* 3. Explain: state_dict */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="What Is a state_dict?"
            subtitle="Just a dictionary of tensors—nothing hidden"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Before any explanation, look at what a state_dict actually is.
              Print it:
            </p>

            <CodeBlock
              language="python"
              filename="print_state_dict.py"
              code={`model = MNISTClassifier()

# Print the model's state_dict
for name, tensor in model.state_dict().items():
    print(f"{name:30s}  shape: {tensor.shape}")`}
            />

            <div className="rounded-lg border bg-muted/30 p-4 font-mono text-xs space-y-0.5 overflow-x-auto">
              <p className="text-muted-foreground">fc1.weight                      shape: torch.Size([256, 784])</p>
              <p className="text-muted-foreground">fc1.bias                        shape: torch.Size([256])</p>
              <p className="text-muted-foreground">fc2.weight                      shape: torch.Size([128, 256])</p>
              <p className="text-muted-foreground">fc2.bias                        shape: torch.Size([128])</p>
              <p className="text-muted-foreground">fc3.weight                      shape: torch.Size([10, 128])</p>
              <p className="text-muted-foreground">fc3.bias                        shape: torch.Size([10])</p>
            </div>

            <p className="text-muted-foreground">
              That is it. A state_dict is just a Python dictionary that maps
              layer names to tensors. Those are the same layer names from your
              MNIST model in nn.Module&mdash;<code className="text-sm bg-muted px-1.5 py-0.5 rounded">fc1</code>,{' '}
              <code className="text-sm bg-muted px-1.5 py-0.5 rounded">fc2</code>,{' '}
              <code className="text-sm bg-muted px-1.5 py-0.5 rounded">fc3</code>&mdash;with{' '}
              <code className="text-sm bg-muted px-1.5 py-0.5 rounded">.weight</code> and{' '}
              <code className="text-sm bg-muted px-1.5 py-0.5 rounded">.bias</code> appended.
              Nothing opaque, nothing hidden.
            </p>

            <p className="text-muted-foreground">
              Remember the &ldquo;parameters are knobs&rdquo; metaphor from
              Series 1? <strong>state_dict is a snapshot of all the
              knobs.</strong> Saving it is like photographing the position of
              every knob. Loading it is setting them all back to where they were.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="Snapshot of the Knobs">
            <code className="text-xs bg-muted px-1 py-0.5 rounded">model.state_dict()</code>{' '}
            returns a dictionary: layer names mapped to tensors. That is all
            it is&mdash;the current values of every learnable parameter.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* state_dict: The Complete Save/Load Pattern */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Now the complete save and load pattern. Two steps to save, three
              to load:
            </p>

            <CodeBlock
              language="python"
              filename="save_state_dict.py"
              code={`# SAVE: extract state_dict, write to disk
torch.save(model.state_dict(), 'mnist_model.pth')`}
            />

            <CodeBlock
              language="python"
              filename="load_state_dict.py"
              code={`# LOAD: create a fresh model, load state_dict into it
loaded_model = MNISTClassifier()                       # 1. Create model (random weights)
state_dict = torch.load('mnist_model.pth',             # 2. Load state_dict from disk
                        weights_only=True)
loaded_model.load_state_dict(state_dict)               # 3. Copy saved weights into model
loaded_model.eval()                                    # 4. Set to eval mode for inference`}
            />

            <p className="text-muted-foreground">
              Step 4 matters because your MNIST model uses batch norm and
              dropout. In eval mode, dropout is disabled and batch norm uses
              its saved running statistics instead of per-batch
              statistics&mdash;the same lesson from the MNIST project. Without
              it, your predictions would be noisy and non-reproducible. If you
              are loading a checkpoint to <em>resume training</em> instead of
              inference, use{' '}
              <code className="text-sm bg-muted px-1.5 py-0.5 rounded">model.train()</code>{' '}
              instead.
            </p>

            <p className="text-muted-foreground">
              Notice the load pattern: you must create the model first, then
              load the state_dict into it.{' '}
              <strong>The state_dict does not contain the architecture</strong>&mdash;just
              the parameter values. The architecture lives in your code (the
              class definition), and the state_dict provides the learned weights.
            </p>

            <p className="text-muted-foreground">
              If you have used Keras or TensorFlow, you might expect a single{' '}
              <code className="text-sm bg-muted px-1.5 py-0.5 rounded">model.save()</code>{' '}
              call to capture both the architecture and the weights. PyTorch
              deliberately separates them. This is a feature, not a limitation:
              the architecture lives in your version-controlled code where you
              can refactor, rename layers, and reorganize files. The state_dict
              stores only the learned values. As long as the parameter names and
              shapes match, the state_dict loads into any compatible model
              instance.
            </p>

            <p className="text-muted-foreground">
              Verify it works&mdash;the loaded model should produce identical
              predictions:
            </p>

            <CodeBlock
              language="python"
              filename="verify_predictions.py"
              code={`# Verify: predictions match
test_input = torch.randn(1, 1, 28, 28)

original_output = model(test_input)          # the trained model from above
loaded_output = loaded_model(test_input)     # the freshly loaded model

print(torch.allclose(original_output, loaded_output))  # True`}
            />

            <p className="text-muted-foreground">
              The two model objects are completely separate in memory&mdash;different
              Python objects, created at different times. But they produce
              identical output because their state_dicts are the same.
              The numbers were copied faithfully.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="The .pth Convention">
            PyTorch files typically use{' '}
            <code className="text-xs bg-muted px-1 py-0.5 rounded">.pth</code>{' '}
            or{' '}
            <code className="text-xs bg-muted px-1 py-0.5 rounded">.pt</code>{' '}
            extensions. This is just a convention&mdash;the file is a
            serialized Python dictionary. Use whichever you prefer.
          </TipBlock>
          <TipBlock title="weights_only=True">
            Since PyTorch 2.6,{' '}
            <code className="text-xs bg-muted px-1 py-0.5 rounded">torch.load()</code>{' '}
            warns if you do not specify{' '}
            <code className="text-xs bg-muted px-1 py-0.5 rounded">weights_only</code>.
            Use{' '}
            <code className="text-xs bg-muted px-1 py-0.5 rounded">weights_only=True</code>{' '}
            for state_dicts (tensors only). Checkpoints with metadata like
            epoch or loss need{' '}
            <code className="text-xs bg-muted px-1 py-0.5 rounded">weights_only=False</code>&mdash;only
            load files you trust.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* 4. Check 1 -- Predict-and-Verify */}
      <Row>
        <Row.Content>
          <div className="rounded-lg border-2 border-primary/30 bg-primary/5 p-5">
            <h3 className="text-sm font-bold text-primary mb-2">
              Check 1: Predict-and-Verify
            </h3>
            <div className="space-y-3">
              <p className="text-muted-foreground text-sm">
                You train a model, save its state_dict. Then you change the
                number of hidden units in the model class (say, from 256 to
                512 in layer 1) and try to load the saved state_dict. What
                happens?
              </p>

              <details className="group">
                <summary className="cursor-pointer text-sm font-medium text-primary hover:text-primary/80 transition-colors">
                  Show answer
                </summary>
                <div className="mt-3 text-sm border-t border-primary/20 pt-3">
                  <div className="space-y-2 text-muted-foreground">
                    <p>
                      <strong>Size mismatch error.</strong> The saved state_dict
                      has{' '}
                      <code className="text-xs bg-muted px-1 py-0.5 rounded">fc1.weight</code>{' '}
                      with shape [256, 784], but the new model expects [512, 784].
                      PyTorch refuses to load because the shapes do not match.
                    </p>
                    <p>
                      This is a feature, not a bug. The state_dict only stores
                      parameter <strong>values</strong>, not the architecture.
                      If the architecture changes, the values no longer fit.
                      You must define the same model class you trained with
                      before loading the state_dict.
                    </p>
                  </div>
                </div>
              </details>
            </div>
          </div>
        </Row.Content>
      </Row>

      {/* 5. Explain: Optimizer state_dict */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Optimizer State Matters Too"
            subtitle="Adam remembers more than just your weights"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              You might think saving the model weights is enough to resume
              training. It is not. Remember from the Optimizers lesson&mdash;Adam
              tracks <strong>momentum buffers</strong> and{' '}
              <strong>adaptive learning rates</strong> for every parameter. Those
              are separate from the model weights, and they live inside the
              optimizer.
            </p>

            <p className="text-muted-foreground">
              The optimizer has its own state_dict:
            </p>

            <CodeBlock
              language="python"
              filename="optimizer_state_dict.py"
              code={`optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# After a few training steps...
for name, value in optimizer.state_dict().items():
    if name == 'param_groups':
        print(f"param_groups: lr={value[0]['lr']}, betas={value[0]['betas']}")
    if name == 'state':
        for param_id, state in value.items():
            print(f"  param {param_id}: exp_avg shape={state['exp_avg'].shape}, "
                  f"exp_avg_sq shape={state['exp_avg_sq'].shape}")`}
            />

            <div className="rounded-lg border bg-muted/30 p-4 font-mono text-xs space-y-0.5 overflow-x-auto">
              <p className="text-muted-foreground">param_groups: lr=0.001, betas=(0.9, 0.999)</p>
              <p className="text-muted-foreground">  param 0: exp_avg shape=torch.Size([256, 784]), exp_avg_sq shape=torch.Size([256, 784])</p>
              <p className="text-muted-foreground">  param 1: exp_avg shape=torch.Size([256]), exp_avg_sq shape=torch.Size([256])</p>
              <p className="text-muted-foreground">  param 2: exp_avg shape=torch.Size([128, 256]), exp_avg_sq shape=torch.Size([128, 256])</p>
              <p className="text-muted-foreground">  ...</p>
            </div>

            <p className="text-muted-foreground">
              Those{' '}
              <code className="text-sm bg-muted px-1.5 py-0.5 rounded">exp_avg</code>{' '}
              and{' '}
              <code className="text-sm bg-muted px-1.5 py-0.5 rounded">exp_avg_sq</code>{' '}
              tensors are Adam&rsquo;s running averages from the Optimizers
              lesson:{' '}
              <code className="text-sm bg-muted px-1.5 py-0.5 rounded">exp_avg</code>{' '}
              is the momentum term (running average of gradients, so Adam
              remembers which direction it has been moving), and{' '}
              <code className="text-sm bg-muted px-1.5 py-0.5 rounded">exp_avg_sq</code>{' '}
              is the adaptive rate (running average of squared gradients, so
              Adam can scale its step size per-parameter). Each one has the same
              shape as the corresponding parameter. For a model with 235K
              parameters, Adam tracks 470K additional values.
            </p>

            <p className="text-muted-foreground">
              <strong>What happens if you resume training without this
              state?</strong> Adam starts from scratch&mdash;zero momentum, no
              history. It is like giving an experienced pilot amnesia and asking
              them to land the plane. The results are not pretty:
            </p>

            <ResumeComparisonChart />

            <p className="text-muted-foreground">
              The green line (with optimizer state) continues smoothly from
              where it left off. The red line (without optimizer state) spikes
              at epoch 6&mdash;Adam lost its momentum buffers and adaptive
              rates, so it stumbles before gradually recovering. The spike is
              wasted computation. With proper checkpointing, you avoid it
              entirely.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Model Weights Alone Are Not Enough">
            Restoring model weights without optimizer state causes a loss
            spike when you resume training. Adam forgets its momentum and
            adaptive rates. Always save both.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* 6. Explain: Checkpoint Pattern */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Checkpoint Pattern"
            subtitle="Bundle everything needed to resume into one file"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              A checkpoint bundles everything you need to resume training into a
              single dictionary: model weights, optimizer state, the current
              epoch, and the current loss. Save that dictionary, and you can
              resume from exactly where you left off.
            </p>

            <CodeBlock
              language="python"
              filename="save_checkpoint.py"
              code={`# Save a checkpoint
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss.item(),
}
torch.save(checkpoint, 'checkpoint.pth')`}
            />

            <CodeBlock
              language="python"
              filename="load_checkpoint.py"
              code={`# Resume from a checkpoint
# weights_only=False because the checkpoint dict contains non-tensor
# metadata (epoch, loss). Only use this with files you trust.
checkpoint = torch.load('checkpoint.pth', weights_only=False)

model = MNISTClassifier()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1

model.train()  # Set back to training mode
print(f"Resuming from epoch {start_epoch}, loss was {checkpoint['loss']:.4f}")`}
            />

            <p className="text-muted-foreground">
              Now integrate this into a training loop. Two common strategies:
              save every N epochs (insurance against crashes), and save the best
              model by validation loss (for early stopping):
            </p>

            <CodeBlock
              language="python"
              filename="checkpoint_in_loop.py"
              code={`best_val_loss = float('inf')

for epoch in range(start_epoch, num_epochs):
    # --- Training (the same heartbeat as always) ---
    model.train()
    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # --- Validation ---
    model.eval()
    val_loss = evaluate(model, val_loader, criterion)

    # --- Checkpoint every 5 epochs ---
    if (epoch + 1) % 5 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss,
        }, f'checkpoint_epoch_{epoch+1}.pth')

    # --- Save best model ---
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pth')
        print(f"New best model saved (val_loss={val_loss:.4f})")`}
            />

            <p className="text-muted-foreground">
              Notice two separate saves: the periodic checkpoint saves
              everything (for crash recovery), while the best model save stores
              only the state_dict (for inference). This is the pattern that
              implements the early stopping concept you learned in Overfitting
              and Regularization&mdash;&ldquo;save best model weights&rdquo; is
              now concrete code.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Same Heartbeat, New Pattern">
            The training loop did not change. Forward, loss, backward,
            update&mdash;still there. The checkpoint logic wraps around the
            loop, not inside it. Same heartbeat, new durability.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* 7. Check 2 -- Transfer Question */}
      <Row>
        <Row.Content>
          <div className="rounded-lg border-2 border-primary/30 bg-primary/5 p-5">
            <h3 className="text-sm font-bold text-primary mb-2">
              Check 2: Transfer Question
            </h3>
            <div className="space-y-3">
              <p className="text-muted-foreground text-sm">
                A colleague trained a model overnight. The process crashed at
                epoch 80 of 100. They restart training from epoch 0. What
                would you tell them?
              </p>

              <details className="group">
                <summary className="cursor-pointer text-sm font-medium text-primary hover:text-primary/80 transition-colors">
                  Show answer
                </summary>
                <div className="mt-3 text-sm border-t border-primary/20 pt-3">
                  <div className="space-y-2 text-muted-foreground">
                    <p>
                      <strong>Implement checkpointing.</strong> Save a checkpoint
                      every N epochs during training. When the process crashes,
                      load the last checkpoint and resume from there&mdash;no need
                      to retrain from scratch.
                    </p>
                    <p>
                      The checkpoint must include both model and optimizer state.
                      If they only save model weights, Adam will lose its momentum
                      buffers and training will spike before recovering (as we saw
                      in the loss curve above).
                    </p>
                    <p>
                      Going forward: always train with periodic checkpoints.
                      The cost is negligible (one file write every few epochs),
                      and the insurance is invaluable.
                    </p>
                  </div>
                </div>
              </details>
            </div>
          </div>
        </Row.Content>
      </Row>

      {/* 8. Elaborate: torch.save(model) and Why Not */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Why Not torch.save(model)?"
            subtitle="The shortcut that creates fragile files"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              You might wonder: why not just save the entire model object?
              PyTorch does support this:
            </p>

            <CodeBlock
              language="python"
              filename="save_whole_model.py"
              code={`# This works, but don't do it
torch.save(model, 'model_full.pth')

# Load the full model
model = torch.load('model_full.pth')`}
            />

            <p className="text-muted-foreground">
              Simpler! One line to save, one to load, no need to create the
              model class first. But here is the trap: try renaming or moving
              the class file.
            </p>

            <CodeBlock
              language="python"
              filename="fragile_model.py"
              code={`# Step 1: Save with torch.save(model)
# (model defined in models/mnist.py as class MNISTClassifier)
torch.save(model, 'model_full.pth')

# Step 2: Rename models/mnist.py to models/classifiers.py

# Step 3: Try to load
model = torch.load('model_full.pth')
# ModuleNotFoundError: No module named 'models.mnist'`}
            />

            <p className="text-muted-foreground">
              <strong>The file depends on the exact class definition and file
              path.</strong>{' '}
              <code className="text-sm bg-muted px-1.5 py-0.5 rounded">torch.save(model)</code>{' '}
              uses Python&rsquo;s pickle, which stores a reference to the class
              (including the module path where it lives). Move the file,
              rename the class, refactor your code&mdash;the saved model breaks.
            </p>

            <ComparisonRow
              left={{
                title: 'state_dict (Recommended)',
                color: 'emerald',
                items: [
                  'Saves only parameter values (numbers)',
                  'Survives code refactoring and file moves',
                  'Architecture lives in version-controlled code',
                  'PyTorch community standard',
                ],
              }}
              right={{
                title: 'torch.save(model) (Fragile)',
                color: 'rose',
                items: [
                  'Pickles the entire object + class reference',
                  'Breaks if you rename or move the class file',
                  'Ties binary file to exact code structure',
                  'Common in tutorials, discouraged in practice',
                ],
              }}
            />

            <p className="text-muted-foreground">
              With state_dict, the architecture lives in your code (version
              controlled, readable, editable) and the learned values live in the
              file. They are decoupled. This is a feature&mdash;not a limitation.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Pickle Fragility">
            <code className="text-xs bg-muted px-1 py-0.5 rounded">torch.save(model)</code>{' '}
            creates a pickle file that depends on the exact module path and
            class name. Refactor your code and the file breaks. Always save
            state_dict instead.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* 9. Elaborate: map_location */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Loading Across Devices"
            subtitle="A brief preview for when you start using GPUs"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              What if you saved a model on GPU but want to load it on CPU? Or
              vice versa? By default, torch.load tries to load tensors onto the
              same device they were saved on. If that device is not available,
              it fails.
            </p>

            <p className="text-muted-foreground">
              The fix is{' '}
              <code className="text-sm bg-muted px-1.5 py-0.5 rounded">map_location</code>:
            </p>

            <CodeBlock
              language="python"
              filename="map_location.py"
              code={`# Saved on GPU, loading on CPU
state_dict = torch.load('model.pth', map_location='cpu',
                        weights_only=True)
model.load_state_dict(state_dict)

# Saved on CPU, loading on a specific GPU
state_dict = torch.load('model.pth', map_location='cuda:0',
                        weights_only=True)
model.load_state_dict(state_dict)

# Portable pattern: load to whatever device is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
state_dict = torch.load('model.pth', map_location=device,
                        weights_only=True)
model.load_state_dict(state_dict)
model.to(device)`}
            />

            <p className="text-muted-foreground">
              This is a preview. Full GPU training comes in the next lesson.
              For now, know that{' '}
              <code className="text-sm bg-muted px-1.5 py-0.5 rounded">map_location</code>{' '}
              exists and handles the device mismatch for you.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="The Portable Pattern">
            Use{' '}
            <code className="text-xs bg-muted px-1 py-0.5 rounded">map_location=device</code>{' '}
            with the standard device detection pattern. Your loading code
            works on any machine regardless of what device the model was saved
            on.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* 10. Practice -- Colab Notebook */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Build It Yourself"
            subtitle="Practice saving, loading, and checkpointing hands-on"
          />
          <div className="rounded-lg border-2 border-primary/50 bg-primary/5 p-6">
            <div className="space-y-4">
              <p className="text-muted-foreground">
                The Colab notebook walks you through each pattern using your
                MNIST model from previous lessons:
              </p>
              <ol className="list-decimal list-inside text-muted-foreground space-y-2 ml-2 text-sm">
                <li>
                  <strong>(Guided)</strong> Save and load a trained MNIST model.
                  Verify predictions match between original and loaded model
                  using{' '}
                  <code className="text-xs bg-muted px-1 py-0.5 rounded">torch.allclose()</code>.
                </li>
                <li>
                  <strong>(Supported)</strong> Add checkpointing to an existing
                  training loop&mdash;save every 5 epochs and save the best model
                  by validation loss.
                </li>
                <li>
                  <strong>(Supported)</strong> Simulate a training crash: train
                  for 10 epochs, save a checkpoint, create a fresh model +
                  optimizer, load the checkpoint, resume for 10 more epochs.
                  Verify the loss curve is continuous.
                </li>
                <li>
                  <strong>(Independent)</strong> Implement the full early stopping
                  pattern from Overfitting and Regularization using
                  checkpoints&mdash;patience counter, save best model, restore
                  the best model at the end of training.
                </li>
              </ol>
              <a
                href="https://colab.research.google.com/github/brennancheung/CourseAI/blob/main/notebooks/2-3-1-saving-and-loading.ipynb"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 px-4 py-2 rounded-md bg-primary text-primary-foreground font-medium text-sm hover:bg-primary/90 transition-colors"
              >
                <ExternalLink className="w-4 h-4" />
                Open in Google Colab
              </a>
              <p className="text-xs text-muted-foreground">
                Exercise 1 is guided with complete starter code. Exercises
                2&ndash;3 are supported with templates. Exercise 4 is
                independent&mdash;you implement early stopping from scratch
                using the checkpoint pattern.
              </p>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Scaffolding">
            The exercises go from guided (save and verify) to independent
            (full early stopping). If you get stuck on the independent
            exercise, re-read the checkpoint pattern section above.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* 11. Summary */}
      <Row>
        <Row.Content>
          <SummaryBlock
            items={[
              {
                headline:
                  'state_dict = a snapshot of all the knobs',
                description:
                  'model.state_dict() returns a dictionary mapping layer names to tensors. Nothing opaque, nothing hidden—just the learned parameter values.',
              },
              {
                headline: 'Always save state_dict, not the model object',
                description:
                  'torch.save(model) creates a fragile pickle tied to your code structure. state_dict stores just the numbers and survives refactoring.',
              },
              {
                headline:
                  'Checkpoints bundle model + optimizer + metadata',
                description:
                  'Save model state_dict, optimizer state_dict, epoch, and loss in one dictionary. Resume training exactly where you left off.',
              },
              {
                headline:
                  'Optimizer state matters—without it, Adam forgets',
                description:
                  "Restoring model weights alone causes a loss spike on resume. Adam's momentum buffers and adaptive rates need saving too.",
              },
            ]}
          />
        </Row.Content>
      </Row>

      {/* 12. Next step */}
      <Row>
        <Row.Content>
          <NextStepBlock
            href="/app/lesson/gpu-training"
            title="GPU Training"
            description="Your models are now durable. But they are still training on CPU. Next: how to put them on a GPU and make training fast."
          />
        </Row.Content>
      </Row>
    </LessonLayout>
  )
}

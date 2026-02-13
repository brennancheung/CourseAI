# Architecture Diagram Component Research

Research findings for building a reusable architecture diagram component on top of react-konva.

## Problem

Mermaid diagrams have fundamental limitations:
- `\n` escape issue (literal `\n` in node labels instead of line breaks)
- No interactivity (hover/click on nodes, highlight connections)
- No zoom control for complex diagrams
- No fullscreen mode
- Opaque auto-layout, limited styling

We already have a `ZoomableCanvas` (react-konva) with pan/zoom/pinch. Build on that.

## Visual Element Taxonomy

Analyzed 6 reference diagrams from ML papers (transformer architecture, U-Net conditioning, Z-Image, data pipelines, RLHF verification loops).

### Nodes

| Node Type | Examples | Visual Properties |
|-----------|----------|-------------------|
| Processing block | "Conv + AdaGN(t)", "Feed Forward" | Rounded rect, solid fill, label centered |
| Data/tensor node | "t_emb (512-dim)", "64×64×64" | Dimension annotation, possibly colored by type |
| Operator node | ⊕ (add), ⊙ (multiply) | Small circle with symbol, at merge points |
| Input/output node | "Input: noisy image", "Output Probabilities" | At diagram edges, distinct styling |
| Compound block | Stacked sub-labels within one container | Multiple text nodes in one group |

**Variation axes:** fill color (semantic), border style (solid/dashed), corner radius, size, label position, multi-line text, sub-content (dimensions, annotations).

### Connections

| Connection Type | Examples | Visual Properties |
|-----------------|----------|-------------------|
| Primary data flow | Main forward pass | Solid arrow, heavier stroke |
| Conditioning/modulation | t_emb → AdaGN blocks | Dashed arrow, labeled (γ(t), β(t)) |
| Skip connection | Encoder → decoder skip | Dashed/curved, spans large distance |
| Residual connection | Add loops in transformer | Curved arrow bypassing blocks |
| Branching/forking | One source → multiple targets | Fan-out from single point |
| Cycle/loop | RLHF verification | Arrows forming a cycle |
| Labeled edge | "downsample", "skip", "fail ✗" | Text annotation on/near arrow |

**Variation axes:** stroke style (solid/dashed/dotted), weight, direction, label text, curvature (straight/bezier/orthogonal), color.

### Groups / Containers

| Group Type | Examples | Visual Properties |
|------------|----------|-------------------|
| Named subsystem | "Encoder (Downsampling)" | Large rounded rect, title at top, child nodes |
| Repeat group | "Nx" on transformer blocks | Group border + repeat annotation |
| Semantic region | Blue-tinted pipeline stages | Background color fill, soft border |
| Dashed boundary | "# Single-Stream Attention Block" | Dashed border vs solid groups |

**Variation axes:** background fill + opacity, border style, title position, nesting depth (2-3 levels), repeat annotation ("×N").

### Layout Patterns

| Pattern | Use Case |
|---------|----------|
| Vertical stack (top→bottom or bottom→top) | Layer-by-layer processing |
| Horizontal pipeline (left→right) | Sequential data processing |
| U-shape | Encoder-bottleneck-decoder |
| Parallel columns | Comparing subsystems |
| Cyclic | Iterative processes, feedback |
| Mixed | Complex architectures |

### Color Usage

1. **Semantic category**: Each layer type gets a distinct color (blue=attention, orange=norm, etc.)
2. **Grouping**: Light background tints define regions (lower opacity than node fills)
3. **Emphasis**: Amber/gold for conditioning, purple for bottleneck
4. **State**: Green=pass, red=fail
5. **Data type**: Different colors for different modalities

## What Konva Gives Us For Free

### No abstraction needed

| Capability | Konva Primitive | Notes |
|---|---|---|
| Grouping / nesting | `Group` | Relative positioning, transforms, opacity |
| Rectangular clipping | `Group` with `clipX/Y/Width/Height` | |
| Text rendering | `Text` | Multiline (`\n` works natively), wrapping, centering, padding |
| Arrows (straight + curved) | `Arrow` | `tension` for curves, `dash` for dashed |
| Hit detection on thin shapes | `hitStrokeWidth` | Set to ~20px on arrows, renders thin |
| Hover/click/drag events | `onMouseEnter`, `onClick`, etc. | On any shape or Group |
| Event delegation | Handler on parent Group, `e.target` for child | |
| Tooltips/callouts | `Label` + `Tag` | Auto-sizing, pointer direction |
| Rect with rounded corners | `Rect` with `cornerRadius` | Single value or per-corner array |
| Z-order | Render order = z-order | Manage through data array |

### Key Konva details

**Group**: Container with no visual representation. No fill, no stroke. Children positioned relative to Group's x,y. Supports clipping. Nest arbitrarily deep.

**Text**: `\n` creates line breaks. `wrap="word"` for wrapping (needs `width`). `align="center"` + `verticalAlign="middle"` for centering. `padding` is uniform (single number). No rich text — use multiple Text nodes for mixed formatting.

**Arrow**: `points={[x1, y1, x2, y2, ...]}` flat array. `tension` for curves through points. `bezier={true}` for bezier control points. `pointerAtBeginning`/`pointerAtEnding`.

**Events**: `onMouseEnter`/`onMouseLeave` on any shape. Bubbling via `e.cancelBubble = true`. `e.target` for delegation.

**Label + Tag**: `pointerDirection="down"` etc. for tooltip arrows. Auto-sizes Tag to Text. Good for hover info.

**Hit detection**: `hitStrokeWidth={20}` on Arrow/Line — renders at visual width, clickable at 20px. `listening={false}` on non-interactive shapes.

## Architecture: Semantic Data Model + Konva Rendering

Separate the **semantic data model** (what nodes/edges/groups exist, their relationships, descriptions) from the **Konva rendering** (positions, sizes, visual props).

The semantic model holds node IDs, edge relationships, descriptions, and metadata. Konva components receive visual props and fire callbacks with IDs. When a node/edge is hovered or selected, the callback looks up the semantic model and updates visual props accordingly (highlight connections, dim unrelated nodes, show info panel).

```
Semantic Model (our data)     Konva Layer (rendering)
┌─────────────────────┐       ┌─────────────────────┐
│ nodes: id, label,   │       │ Group + Rect + Text  │
│   description,      │──────>│   (positioned, styled)│
│   group membership  │       │                      │
│                     │       │ Arrow components      │
│ edges: from, to,    │──────>│   (computed endpoints)│
│   label, style      │       │                      │
│                     │       │ Group containers      │
│ groups: id, title,  │──────>│   (background rects)  │
│   color, children   │       │                      │
└─────────────────────┘       └─────────────────────┘
         │                              │
         │  callbacks (hover, click)    │
         │<─────────────────────────────│
         │                              │
         ▼                              │
   Update visual props ────────────────>│
   (highlight, dim, show panel)
```

This means:
- Konva nodes can have `id` props matching our semantic model
- Hover/click handlers fire with the Konva node ID
- We look up relationships in the semantic model
- We derive updated visual props (highlighted edges, dimmed nodes) and pass them down
- Konva re-renders only what changed (React.memo)

## Thin Wrappers to Build

| Component | What | Why |
|---|---|---|
| `DiagramNode` | `Group + Rect + Text(s)` | No "box with label" primitive in Konva |
| `DiagramEdge` | `Arrow` with computed endpoints | No connector/anchor concept |
| Responsive + fullscreen container | `ResizeObserver` + zoom-to-fit | Stage doesn't auto-resize |

### Edge Endpoint Calculation

The one real gap. Konva's `Arrow` needs explicit `points`. We need:

```ts
// Compute where an arrow exits a rectangle aimed at a target point
function getEdgePoint(
  rect: { x: number; y: number; width: number; height: number },
  target: { x: number; y: number }
): { x: number; y: number }
```

Basic ray-rectangle intersection geometry. Not a framework, just a utility function.

## Performance Notes

- 50-100 nodes is trivial for Konva. Single `Layer` is fine.
- `React.memo` on node/edge components
- Stable references for constant props (dash arrays, colors)
- `listening={false}` on non-interactive shapes
- Event delegation on parent Group rather than per-node handlers
- Each `Layer` = 2 canvas elements (scene + hit). Keep to 1-2 layers.

## Interactive Features

1. **Pan + zoom**: Already in `ZoomableCanvas`
2. **Hover highlight**: `onMouseEnter` → look up edges in semantic model → update styles
3. **Click for details**: `onClick` → show description panel (HTML overlay or Konva Label)
4. **Fullscreen**: Expand container to viewport, recalculate stage size
5. **Zoom to fit**: `group.getClientRect()` for bounding box, compute scale
6. **Double-click reset**: Already in `ZoomableCanvas`

## What NOT to Build

- Generic graph layout engine (manual positioning is fine for educational diagrams)
- Connector abstraction with anchor points / snapping / routing
- Custom event system (Konva's is excellent)
- Custom hit detection (hitStrokeWidth solves it)
- Layer-per-node architecture
- Abstraction over Group (just use Group)

# Design Spec: Dot Product ManimCE Animation

**Date**: 2026-07-23  
**Topic**: ManimCE Animation for Dot Product Intuition  
**Target File**: `phase01-math-foundations/lesson01-linear-algebra-intuition/code/dot_product_animation.py`  
**Reference Document**: `phase01-math-foundations/lesson01-linear-algebra-intuition/README.md`  

---

## 1. Overview & Goals

Create a high-quality, visually stunning Manim Community Edition (v0.20.1) animation explaining the **Dot Product** concept based on Lesson 1 of Linear Algebra Intuition.

The video bridges geometric intuition, algebraic calculation, and practical applications in AI (Similarity Search & Attention Mechanism).

---

## 2. Animation Structure & Scenes

The animation consists of a single continuous scene `DotProductIntuition(Scene)` divided into 3 key sections:

### Section 1: Algebraic & Geometric Projection (0:00 - 0:30)
- **Grid Setup**: Display a `NumberPlane` with coordinate axes.
- **Vectors**: Render vector $\vec{a} = [3, 2]$ (Yellow) and vector $\vec{b} = [4, 0]$ (Blue).
- **Algebraic Formula**: Show MathTex formula $\vec{a} \cdot \vec{b} = a_1 b_1 + a_2 b_2 = (3)(4) + (2)(0) = 12$.
- **Projection Highlight**: Lower a dashed line from the tip of $\vec{a}$ perpendicular to $\vec{b}$, revealing the projection vector $\text{proj}_{\vec{b}}(\vec{a})$ in Green.
- **Formula Connection**: Highlight $\vec{a} \cdot \vec{b} = \|\vec{b}\| \times \|\text{proj}_{\vec{b}}(\vec{a})\|$.

### Section 2: Angle & Similarity Dynamics (0:30 - 1:00)
- **Dynamic Rotation**: Rotate vector $\vec{a}$ around the origin to demonstrate 3 critical states:
  1. **Acute Angle ($\theta < 90^\circ$)**: $\vec{a} \cdot \vec{b} > 0$ (Vectors share direction / Similar).
  2. **Right Angle ($\theta = 90^\circ$)**: $\vec{a} \cdot \vec{b} = 0$ (Perpendicular / Orthogonal / Unrelated).
  3. **Obtuse Angle ($\theta > 90^\circ$)**: $\vec{a} \cdot \vec{b} < 0$ (Opposite direction / Dissimilar).
- **Angle Indicator & Value Tracker**: Display dynamic arc angle $\theta$ and real-time dot product scalar value updating via `ValueTracker` / `always_redraw`.

### Section 3: AI Applications (1:00 - 1:30)
- **Text & Embedding Context**: Transition plane away or fade out coordinate grid.
- **Query & Key Vectors**: Label vector $\vec{a}$ as `Query Vector (Q)` and vector $\vec{b}$ as `Key Vector (K)`.
- **Cosine Similarity / Attention Weight**: Show how $\text{Attention Score} = \text{softmax}(Q \cdot K^T / \sqrt{d_k})$, emphasizing that Dot Product determines how much attention a model pays to a concept or word embedding.

---

## 3. Visual Styling & Best Practices (ManimCE)

- **Color Palette**:
  - Background: `#0f172a` (Dark slate navy)
  - Vector A / Query: `#f59e0b` (Amber Yellow)
  - Vector B / Key: `#3b82f6` (Royal Blue)
  - Projection / Result: `#10b981` (Emerald Green)
  - Angle Arc / Math Accents: `#ec4899` (Pink)
- **Code Rules (ManimCE)**:
  - Imports: `from manim import *`
  - Math rendering: `MathTex` with proper raw strings `r"..."` and color matching.
  - Positioning: Clean layout using `.next_to()`, `.to_corner()`, and `.move_to()`.
  - Updaters: Use `always_redraw` or `add_updater` cleanly for smooth rotation tracking.

---

## 4. Verification & Output Criteria

- Render script using command:
  ```bash
  manim -pql phase01-math-foundations/lesson01-linear-algebra-intuition/code/dot_product_animation.py DotProductIntuition
  ```
- Verify video output exists in `media/videos/dot_product_animation/480p15/DotProductIntuition.mp4` without errors.

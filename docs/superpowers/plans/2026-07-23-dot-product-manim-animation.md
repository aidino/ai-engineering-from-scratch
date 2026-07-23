# Dot Product ManimCE Animation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a Manim Community Edition animation script (`dot_product_animation.py`) that visually explains Dot Product through 3 key phases: Algebraic/Geometric Projection, Angle & Similarity dynamics, and AI applications (Attention / Embeddings), then render and verify the generated MP4.

**Architecture:** Implement a single `DotProductIntuition(Scene)` in ManimCE with smooth transitions between coordinate system vector projections, rotational dynamic updaters, and Transformer Attention query/key visualization.

**Tech Stack:** Python 3.12, Manim Community Edition v0.20.1

## Global Constraints

- Use Manim Community Edition imports: `from manim import *`
- Use LaTeX formula formatting with `MathTex(r"...")`
- Output target path: `phase01-math-foundations/lesson01-linear-algebra-intuition/code/dot_product_animation.py`
- Follow color conventions: Vector A/Query (Amber Yellow `#f59e0b`), Vector B/Key (Royal Blue `#3b82f6`), Projection/Result (Emerald Green `#10b981`).

---

### Task 1: Create ManimCE Script for Dot Product Intuition

**Files:**
- Create: `phase01-math-foundations/lesson01-linear-algebra-intuition/code/dot_product_animation.py`

**Interfaces:**
- Consumes: ManimCE library (`Scene`, `NumberPlane`, `Vector`, `MathTex`, `ValueTracker`, `always_redraw`, `Create`, `Transform`, `FadeOut`, `Write`)
- Produces: `DotProductIntuition` Scene class ready for Manim rendering.

- [ ] **Step 1: Write `dot_product_animation.py`**

Write the complete code for `DotProductIntuition` class covering Section 1 (Projection), Section 2 (Angles & Rotation), and Section 3 (AI Applications).

```python
from manim import *

class DotProductIntuition(Scene):
    def construct(self):
        # Set dark background color
        self.camera.background_color = "#0f172a"

        # ---------------------------------------------------------
        # SECTION 1: Algebraic & Geometric Projection
        # ---------------------------------------------------------
        plane = NumberPlane(
            x_range=[-4, 5, 1],
            y_range=[-3, 4, 1],
            background_line_style={
                "stroke_color": TEAL,
                "stroke_width": 1,
                "stroke_opacity": 0.3,
            }
        )
        
        # Vectors
        vec_a_coords = np.array([3, 2, 0])
        vec_b_coords = np.array([4, 0, 0])
        
        vec_a = Arrow(plane.c2p(0, 0), plane.c2p(*vec_a_coords[:2]), buff=0, color="#f59e0b", stroke_width=4)
        vec_b = Arrow(plane.c2p(0, 0), plane.c2p(*vec_b_coords[:2]), buff=0, color="#3b82f6", stroke_width=4)
        
        label_a = MathTex(r"\vec{a} = [3, 2]", color="#f59e0b").next_to(vec_a.get_end(), UP + RIGHT, buff=0.1)
        label_b = MathTex(r"\vec{b} = [4, 0]", color="#3b82f6").next_to(vec_b.get_end(), DOWN, buff=0.1)
        
        title = Text("Dot Product: Trực Giác & Phép Chiếu", font_size=32, color=WHITE).to_corner(UL)
        
        self.play(FadeIn(title), Create(plane))
        self.play(Create(vec_b), Write(label_b))
        self.play(Create(vec_a), Write(label_a))
        self.wait(1)

        # Formula text
        formula = MathTex(
            r"\vec{a} \cdot \vec{b} = a_1 b_1 + a_2 b_2 = (3)(4) + (2)(0) = 12",
            font_size=28
        ).to_corner(UR)
        formula.set_color_by_tex("a_1 b_1", "#f59e0b")
        
        self.play(Write(formula))
        self.wait(1)

        # Perpendicular line for projection
        proj_point = plane.c2p(3, 0)
        dashed_line = DashedLine(vec_a.get_end(), proj_point, color=GRAY)
        proj_vec = Arrow(plane.c2p(0, 0), proj_point, buff=0, color="#10b981", stroke_width=5)
        proj_label = MathTex(r"\text{proj}_{\vec{b}}(\vec{a}) = [3, 0]", color="#10b981", font_size=24).next_to(proj_vec, DOWN, buff=0.4)

        self.play(Create(dashed_line))
        self.play(Create(proj_vec), Write(proj_label))
        
        geom_formula = MathTex(
            r"\vec{a} \cdot \vec{b} = \|\vec{b}\| \times \|\text{proj}_{\vec{b}}(\vec{a})\| = 4 \times 3 = 12",
            font_size=26,
            color="#10b981"
        ).next_to(formula, DOWN, aligned_edge=RIGHT)
        
        self.play(Write(geom_formula))
        self.wait(2)

        # Cleanup section 1 text
        self.play(
            FadeOut(formula),
            FadeOut(geom_formula),
            FadeOut(dashed_line),
            FadeOut(proj_vec),
            FadeOut(proj_label),
            FadeOut(label_a),
            FadeOut(label_b),
            FadeOut(title)
        )

        # ---------------------------------------------------------
        # SECTION 2: Dynamic Angle & Similarity
        # ---------------------------------------------------------
        title_sec2 = Text("Độ Tương Tự & Góc Theta", font_size=32, color=WHITE).to_corner(UL)
        self.play(FadeIn(title_sec2))

        # Angle tracking
        angle_tracker = ValueTracker(np.arctan2(2, 3))
        
        # Redraw dynamic vector a
        dynamic_vec_a = always_redraw(lambda: Arrow(
            plane.c2p(0, 0),
            plane.c2p(np.sqrt(13) * np.cos(angle_tracker.get_value()), np.sqrt(13) * np.sin(angle_tracker.get_value())),
            buff=0, color="#f59e0b", stroke_width=4
        ))
        
        dynamic_dot = always_redraw(lambda: MathTex(
            rf"\vec{{a}} \cdot \vec{{b}} = {4 * np.sqrt(13) * np.cos(angle_tracker.get_value()):.1f}",
            font_size=30,
            color="#f59e0b"
        ).to_corner(UR))
        
        similarity_text = always_redraw(lambda: (
            Text("Cùng hướng: Dot Product > 0 (Tương tự)", color=GREEN, font_size=24).next_to(dynamic_dot, DOWN, aligned_edge=RIGHT)
            if np.cos(angle_tracker.get_value()) > 0.05 else (
                Text("Vuông góc: Dot Product = 0 (Độc lập)", color=YELLOW, font_size=24).next_to(dynamic_dot, DOWN, aligned_edge=RIGHT)
                if abs(np.cos(angle_tracker.get_value())) <= 0.05 else
                Text("Ngược hướng: Dot Product < 0 (Trái ngược)", color=RED, font_size=24).next_to(dynamic_dot, DOWN, aligned_edge=RIGHT)
            )
        ))

        self.play(ReplacementTransform(vec_a, dynamic_vec_a))
        self.add(dynamic_dot, similarity_text)
        self.wait(1)

        # Animate rotation to 90 deg (orthogonal)
        self.play(angle_tracker.animate.set_value(np.pi / 2), run_time=2.5)
        self.wait(1)

        # Animate rotation to 150 deg (opposite)
        self.play(angle_tracker.animate.set_value(5 * np.pi / 6), run_time=2.5)
        self.wait(1)

        # Rotate back to acute angle
        self.play(angle_tracker.animate.set_value(np.pi / 6), run_time=2)
        self.wait(1)

        self.play(
            FadeOut(title_sec2),
            FadeOut(plane),
            FadeOut(dynamic_vec_a),
            FadeOut(vec_b),
            FadeOut(dynamic_dot),
            FadeOut(similarity_text)
        )

        # ---------------------------------------------------------
        # SECTION 3: AI Applications (Attention & Embeddings)
        # ---------------------------------------------------------
        title_ai = Text("Ứng dụng trong AI: Attention & Similarity", font_size=32, color=WHITE).to_corner(UL)
        self.play(FadeIn(title_ai))

        ai_box = Rectangle(height=4, width=8, color=BLUE, fill_opacity=0.1).move_to(ORIGIN)
        
        q_text = MathTex(r"\text{Query (Q): Vector truy vấn}", color="#f59e0b", font_size=28)
        k_text = MathTex(r"\text{Key (K): Vector khóa/dữ liệu}", color="#3b82f6", font_size=28)
        att_formula = MathTex(
            r"\text{Attention Score} = \text{softmax}\left(\frac{Q \cdot K^T}{\sqrt{d_k}}\right)",
            color=WHITE, font_size=32
        )
        
        VGroup(q_text, k_text, att_formula).arrange(DOWN, buff=0.4).move_to(ai_box.get_center())

        self.play(Create(ai_box))
        self.play(Write(q_text), Write(k_text))
        self.wait(1)
        self.play(Write(att_formula))
        self.wait(2)

        summary = Text("Dot Product chính là thước đo độ tương đồng trong AI!", font_size=24, color=YELLOW).next_to(ai_box, DOWN, buff=0.5)
        self.play(Write(summary))
        self.wait(2)
```

- [ ] **Step 2: Commit file**

```bash
git add phase01-math-foundations/lesson01-linear-algebra-intuition/code/dot_product_animation.py
git commit -m "feat: implement Dot Product ManimCE animation script"
```

---

### Task 2: Render and Verify ManimCE Animation Video

**Files:**
- Output: `media/videos/dot_product_animation/480p15/DotProductIntuition.mp4`

- [ ] **Step 1: Execute Manim rendering**

Run: `manim -pql phase01-math-foundations/lesson01-linear-algebra-intuition/code/dot_product_animation.py DotProductIntuition`
Expected: Successful render with no errors, producing `.mp4` video.

- [ ] **Step 2: Verify video creation**

Run: `ls -la media/videos/dot_product_animation/480p15/DotProductIntuition.mp4`
Expected: File exists and size > 0.

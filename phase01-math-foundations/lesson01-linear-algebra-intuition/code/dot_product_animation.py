from manim import *
import numpy as np

class DotProductIntuition(Scene):
    def construct(self):
        # 1. Background setup
        self.camera.background_color = "#0f172a"

        # Hex Color Palette
        COLOR_A = "#f59e0b"     # Amber Yellow
        COLOR_B = "#3b82f6"     # Royal Blue
        COLOR_PROJ = "#10b981"  # Emerald Green

        # ==========================================
        # SECTION 1: Projection & Formulae
        # ==========================================
        title_sec1 = Text("1. Phép Chiếu Vuông Góc & Công Thức", font_size=36, color=WHITE)
        title_sec1.to_edge(UP, buff=0.4)
        self.play(Write(title_sec1))

        # NumberPlane coordinate system
        plane = NumberPlane(
            x_range=[-1, 6, 1],
            y_range=[-1, 4, 1],
            x_length=7,
            y_length=4.5,
            axis_config={"color": BLUE_D},
            background_line_style={"stroke_color": TEAL, "stroke_width": 1, "stroke_opacity": 0.25}
        ).shift(DOWN * 0.5 + LEFT * 1.8)

        self.play(Create(plane))

        # Define points in plane coordinates
        start_pt = plane.c2p(0, 0, 0)
        a_end = plane.c2p(3, 2, 0)
        b_end = plane.c2p(4, 0, 0)
        proj_end = plane.c2p(3, 0, 0)

        # Vector a = [3, 2] & Vector b = [4, 0]
        vec_a = Arrow(start_pt, a_end, buff=0, color=COLOR_A, stroke_width=4)
        vec_b = Arrow(start_pt, b_end, buff=0, color=COLOR_B, stroke_width=4)

        label_a = MathTex(r"\vec{a} = [3, 2]", color=COLOR_A, font_size=30).next_to(vec_a.get_end(), UP + RIGHT, buff=0.1)
        label_b = MathTex(r"\vec{b} = [4, 0]", color=COLOR_B, font_size=30).next_to(vec_b.get_end(), DOWN, buff=0.15)

        self.play(GrowArrow(vec_b), Write(label_b))
        self.play(GrowArrow(vec_a), Write(label_a))
        self.wait(0.5)

        # Formula 1: Algebraic definition
        alg_formula = MathTex(
            r"\vec{a} \cdot \vec{b} = a_1 b_1 + a_2 b_2 = (3)(4) + (2)(0) = 12",
            font_size=30
        ).to_corner(UR, buff=0.4).shift(DOWN * 0.8)

        self.play(Write(alg_formula))
        self.wait(0.5)

        # Dashed projection line & Projection vector
        dash_line = DashedLine(a_end, proj_end, color=GRAY_B, stroke_width=2)
        proj_vec = Arrow(start_pt, proj_end, buff=0, color=COLOR_PROJ, stroke_width=5)
        label_proj = MathTex(r"\text{proj}_{\vec{b}}(\vec{a}) = [3, 0]", color=COLOR_PROJ, font_size=26).next_to(proj_vec, DOWN, buff=0.45)

        self.play(Create(dash_line))
        self.play(GrowArrow(proj_vec), Write(label_proj))
        self.wait(0.5)

        # Formula 2: Geometric definition
        geo_formula = MathTex(
            r"\vec{a} \cdot \vec{b} = \|\vec{b}\| \times \|\text{proj}_{\vec{b}}(\vec{a})\| = 4 \times 3 = 12",
            font_size=30
        ).next_to(alg_formula, DOWN, aligned_edge=LEFT, buff=0.4)

        self.play(Write(geo_formula))
        self.wait(2)

        # Clean up section 1 elements
        self.play(
            FadeOut(title_sec1),
            FadeOut(alg_formula),
            FadeOut(geo_formula),
            FadeOut(label_a),
            FadeOut(label_b),
            FadeOut(label_proj),
            FadeOut(dash_line),
            FadeOut(proj_vec),
            FadeOut(vec_a),
            FadeOut(vec_b)
        )

        # ==========================================
        # SECTION 2: Angles & Rotation
        # ==========================================
        title_sec2 = Text("2. Độ Tương Tự & Góc Theta", font_size=36, color=WHITE)
        title_sec2.to_edge(UP, buff=0.4)
        self.play(Write(title_sec2))

        # Static Vector B
        vec_b_static = Arrow(start_pt, b_end, buff=0, color=COLOR_B, stroke_width=4)
        label_b_static = MathTex(r"\vec{b}", color=COLOR_B, font_size=32).next_to(vec_b_static.get_end(), RIGHT, buff=0.1)
        self.play(Create(vec_b_static), Write(label_b_static))

        # ValueTracker for vector a angle (in radians)
        init_angle = np.arctan2(2, 3)
        angle_tracker = ValueTracker(init_angle)
        r = np.sqrt(13)  # magnitude of vector a

        # Dynamic Vector A
        dyn_vec_a = always_redraw(lambda: Arrow(
            start_pt,
            plane.c2p(r * np.cos(angle_tracker.get_value()), r * np.sin(angle_tracker.get_value()), 0),
            buff=0, color=COLOR_A, stroke_width=4
        ))

        # Dynamic Dashed Line
        dyn_dash = always_redraw(lambda: DashedLine(
            plane.c2p(r * np.cos(angle_tracker.get_value()), r * np.sin(angle_tracker.get_value()), 0),
            plane.c2p(r * np.cos(angle_tracker.get_value()), 0, 0),
            color=GRAY_B, stroke_width=2
        ))

        # Dynamic Projection Vector
        def get_dyn_proj():
            x_proj = r * np.cos(angle_tracker.get_value())
            if abs(x_proj) < 0.05:
                return Line(start_pt, start_pt, color=COLOR_PROJ)
            return Arrow(
                start_pt,
                plane.c2p(x_proj, 0, 0),
                buff=0, color=COLOR_PROJ, stroke_width=5
            )

        dyn_proj = always_redraw(get_dyn_proj)

        # Dynamic Info Card
        def get_info_panel():
            ang_rad = angle_tracker.get_value()
            ang_deg = np.degrees(ang_rad) % 360
            if ang_deg > 180:
                ang_deg -= 360

            dot_val = 4 * r * np.cos(ang_rad)

            if abs(abs(ang_deg) - 90) < 1.5:
                status_str = "Trực giao (= 0): Vuông góc"
                status_color = YELLOW
            elif abs(ang_deg) < 90:
                status_str = "Góc Nhọn (> 0): Cùng hướng"
                status_color = GREEN
            else:
                status_str = "Góc Tù (< 0): Ngược hướng"
                status_color = RED

            t1 = MathTex(rf"\theta = {ang_deg:.1f}^\circ", font_size=30)
            t2 = MathTex(rf"\vec{{a}} \cdot \vec{{b}} = {dot_val:.2f}", font_size=32, color=COLOR_PROJ)
            t3 = Text(status_str, font_size=22, color=status_color)

            group = VGroup(t1, t2, t3).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
            group.to_corner(UR, buff=0.4).shift(DOWN * 0.8)
            return group

        info_panel = always_redraw(get_info_panel)

        self.play(Create(dyn_vec_a), Create(dyn_dash), Create(dyn_proj), FadeIn(info_panel))
        self.wait(1)

        # Animate through acute (>0), 90 deg (=0), and obtuse (<0)
        # 1. Acute angle (30 deg)
        self.play(angle_tracker.animate.set_value(np.radians(30)), run_time=1.5)
        self.wait(1)

        # 2. Right angle (90 deg -> orthogonal = 0)
        self.play(angle_tracker.animate.set_value(np.radians(90)), run_time=2)
        self.wait(1.5)

        # 3. Obtuse angle (150 deg -> negative)
        self.play(angle_tracker.animate.set_value(np.radians(150)), run_time=2)
        self.wait(1.5)

        # Return to initial angle
        self.play(angle_tracker.animate.set_value(init_angle), run_time=1.5)
        self.wait(1)

        # Clean up section 2
        self.play(
            FadeOut(plane),
            FadeOut(vec_b_static),
            FadeOut(label_b_static),
            FadeOut(dyn_vec_a),
            FadeOut(dyn_dash),
            FadeOut(dyn_proj),
            FadeOut(info_panel),
            FadeOut(title_sec2)
        )

        # ==========================================
        # SECTION 3: AI Applications (Attention Mechanism)
        # ==========================================
        title_sec3 = Text("3. Ứng Dụng Trong AI & Attention Mechanism", font_size=36, color=WHITE)
        title_sec3.to_edge(UP, buff=0.4)
        self.play(Write(title_sec3))

        # Query (Q) and Key (K) vectors diagram
        q_label = Text("Query (Q)", font_size=24, color=COLOR_A)
        k_label = Text("Key (K)", font_size=24, color=COLOR_B)

        q_arrow = Arrow(ORIGIN, RIGHT * 2.5 + UP * 1.5, buff=0, color=COLOR_A, stroke_width=4)
        k_arrow = Arrow(ORIGIN, RIGHT * 3 + UP * 0.3, buff=0, color=COLOR_B, stroke_width=4)

        q_group = VGroup(q_arrow, q_label.next_to(q_arrow.get_end(), UP, buff=0.1))
        k_group = VGroup(k_arrow, k_label.next_to(k_arrow.get_end(), DOWN, buff=0.1))

        vectors_group = VGroup(q_group, k_group).center().shift(LEFT * 3.5 + UP * 0.3)

        self.play(GrowArrow(q_arrow), Write(q_label))
        self.play(GrowArrow(k_arrow), Write(k_label))
        self.wait(0.5)

        # Attention Formula
        attn_formula = MathTex(
            r"\text{Attention Score} = \text{softmax}\left(\frac{Q \cdot K^T}{\sqrt{d_k}}\right)",
            font_size=34
        ).shift(RIGHT * 2 + UP * 0.5)

        attn_box = SurroundingRectangle(attn_formula, color=COLOR_PROJ, buff=0.25, corner_radius=0.1)

        self.play(Write(attn_formula))
        self.play(Create(attn_box))
        self.wait(1)

        # Explanatory Text
        explanation = Text(
            "Tích vô hướng Q · Kᵀ đo lường mức độ tương đồng/liên quan giữa các Token",
            font_size=20,
            color=GRAY_A
        ).next_to(attn_box, DOWN, buff=0.4)
        self.play(FadeIn(explanation))
        self.wait(1)

        # Summary Banner at the bottom
        summary_box = RoundedRectangle(
            corner_radius=0.15,
            width=11, height=1.1,
            color=COLOR_A, fill_color="#1e293b", fill_opacity=0.95
        ).to_edge(DOWN, buff=0.5)

        summary_text = Text(
            "Dot Product chính là thước đo độ tương đồng trong AI!",
            font_size=26,
            color=YELLOW
        ).move_to(summary_box.get_center())

        self.play(Create(summary_box), Write(summary_text))
        self.wait(3)

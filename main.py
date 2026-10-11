import kivy
kivy.require('2.1.0')

from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.uix.popup import Popup
from kivy.uix.slider import Slider
from kivy.graphics import Color, Ellipse, Line
from kivy.clock import Clock

import numpy as np
import random

# --- Genetic Programming Utilities ---

FUNC_MAP = {'+': '+', '-': '-', '*': '*', '/': '/'}

def eval_expr(expr, x):
    if isinstance(expr, tuple):
        func, left, right = expr
        l_val = eval_expr(left, x)
        r_val = eval_expr(right, x)
        try:
            if func == '+': return l_val + r_val
            if func == '-': return l_val - r_val
            if func == '*': return l_val * r_val
            if func == '/': return l_val / r_val if r_val != 0 else l_val
        except Exception:
            return 0
    else:
        return x if expr == 'x' else expr

def subtree_score(expr):
    score = 0
    for i in range(-5,6):
        target = i**2 + i
        guess = eval_expr(expr, i)
        score += abs(target - guess)
    return -score

def get_active_path(expr):
    path = set()
    def helper(node):
        if isinstance(node, tuple):
            left_score = subtree_score(node[1])
            right_score = subtree_score(node[2])
            if left_score >= right_score:
                path.add(id(node))
                helper(node[1])
            else:
                path.add(id(node))
                helper(node[2])
        else:
            path.add(id(node))
    helper(expr)
    return path

def generate_random_tree(depth=3):
    if depth == 0:
        return random.choice(['x', random.randint(-5,5)])
    func = random.choice(list(FUNC_MAP.keys()))
    return (func, generate_random_tree(depth-1), generate_random_tree(depth-1))

# --- Kivy Widget for Tree Visualization ---
class TreeArea(Widget):
    pass

class GPVisualization(App):
    def build(self):
        self.root_layout = BoxLayout(orientation='vertical')
        self.tree_area = TreeArea(size_hint=(1,0.7))
        self.root_layout.add_widget(self.tree_area)

        control_layout = BoxLayout(size_hint=(1,0.1))
        self.pause_btn = Button(text="Pause Evolution")
        self.resume_btn = Button(text="Resume Evolution")
        control_layout.add_widget(self.pause_btn)
        control_layout.add_widget(self.resume_btn)
        self.root_layout.add_widget(control_layout)

        self.best_expr = generate_random_tree()
        self.node_positions = {}
        self.paused = False
        self.evolution_interval = 0.5
        self.evolution_event = Clock.schedule_interval(self.step, self.evolution_interval)

        self.tree_area.bind(on_touch_down=self.on_tree_touch)
        self.pause_btn.bind(on_press=self.pause_evolution)
        self.resume_btn.bind(on_press=self.resume_evolution)

        self.draw_tree(self.best_expr)
        return self.root_layout

    # --- Evolution step ---
    def step(self, dt):
        if self.paused: return
        def mutate(expr):
            if isinstance(expr, tuple):
                if random.random() < 0.2:
                    return generate_random_tree(2)
                return (expr[0], mutate(expr[1]), mutate(expr[2]))
            else:
                if random.random() < 0.2:
                    return random.choice(['x', random.randint(-5,5)])
                return expr
        self.best_expr = mutate(self.best_expr)
        self.draw_tree(self.best_expr)

    def pause_evolution(self, instance): self.paused = True
    def resume_evolution(self, instance): self.paused = False

    # --- Tree drawing ---
    def draw_tree(self, expr, x=None, y=None, dx=None):
        self.tree_area.canvas.clear()
        self.node_positions = {}
        if expr is None: return
        if x is None: x = self.tree_area.width / 2
        if y is None: y = self.tree_area.height - 40
        if dx is None: dx = self.tree_area.width / 4
        active_path = get_active_path(expr)
        with self.tree_area.canvas:
            self._draw_node(expr, x, y, dx, active_path)

    def _draw_node(self, expr, x, y, dx, active_path=None):
        node_id = id(expr)
        is_active = active_path and node_id in active_path
        if isinstance(expr, tuple):
            func, left, right = expr
            Color(1,0.5,0) if is_active else Color(0.2,0.4,1)
            Ellipse(pos=(x-18,y-18), size=(36,36))
            self._draw_text(FUNC_MAP.get(func,'?'), x, y)
            new_y = y - 70
            if left: Line(points=[x,y,x-dx,new_y]); self._draw_node(left, x-dx, new_y, dx/2, active_path)
            if right: Line(points=[x,y,x+dx,new_y]); self._draw_node(right, x+dx, new_y, dx/2, active_path)
        else:
            Color(1,0.5,0) if is_active else Color(0,0.8,0.2)
            Ellipse(pos=(x-14,y-14), size=(28,28))
            self._draw_text(str(expr), x, y)
        self.node_positions[node_id] = (x,y,18 if isinstance(expr, tuple) else 14, expr)

    def _draw_text(self, text, x, y):
        label = Label(text=text, pos=(x-10,y-10), size=(20,20))
        self.tree_area.add_widget(label)

    # --- Node interaction ---
    def on_tree_touch(self, instance, touch):
        for node_id, (x, y, radius, expr) in self.node_positions.items():
            if (touch.x-x)**2 + (touch.y-y)**2 <= radius**2:
                self.show_node_info(expr)
                return True
        return False

    def show_node_info(self, expr):
        content = BoxLayout(orientation='vertical')
        score = subtree_score(expr)
        content.add_widget(Label(text=f"Subtree Score: {score:.2f}", size_hint=(1,0.1)))
        content.add_widget(TextInput(text=str(expr), readonly=True, size_hint=(1,0.2)))

        plot_widget = Widget(size_hint=(1,0.6))
        plot_widget.popup_expr_ref = {'expr': expr}
        plot_widget.min_x = -5
        plot_widget.max_x = 5
        plot_widget.selected_nodes = [expr]
        plot_widget.active_path = get_active_path(expr)
        plot_widget.zoom = 1.0
        plot_widget.offset_x = 0.0
        plot_widget.offset_y = 0.0
        plot_widget.pan_start = None
        content.add_widget(plot_widget)

        # Controls
        control_layout = BoxLayout(size_hint=(1,0.1))
        pause_btn = Button(text="Pause Evolution")
        resume_btn = Button(text="Resume Evolution")
        control_layout.add_widget(pause_btn)
        control_layout.add_widget(resume_btn)
        pause_btn.bind(on_press=lambda inst: self.pause_evolution(inst))
        resume_btn.bind(on_press=lambda inst: self.resume_evolution(inst))
        content.add_widget(control_layout)

        # Sliders
        range_layout = BoxLayout(size_hint=(1,0.1))
        min_label = Label(text="Min X: -5", size_hint=(0.25,1))
        max_label = Label(text="Max X: 5", size_hint=(0.25,1))
        min_slider = Slider(min=-10, max=0, value=-5, step=0.1, size_hint=(0.25,1))
        max_slider = Slider(min=0, max=10, value=5, step=0.1, size_hint=(0.25,1))
        range_layout.add_widget(min_label); range_layout.add_widget(min_slider)
        range_layout.add_widget(max_label); range_layout.add_widget(max_slider)
        content.add_widget(range_layout)

        min_slider.bind(value=lambda inst,val: self.update_plot_range(plot_widget, val, True, min_label))
        max_slider.bind(value=lambda inst,val: self.update_plot_range(plot_widget, val, False, max_label))

        # Export Button
        export_btn = Button(text="Export Plot")
        content.add_widget(export_btn)
        export_btn.bind(on_press=lambda inst: self.export_plot(plot_widget))

        # Bind hover, zoom, pan
        plot_widget.bind(on_touch_move=lambda inst, touch: self.on_plot_hover(inst, touch))
        plot_widget.bind(on_touch_down=lambda inst, touch: self.on_plot_touch_down(inst, touch))
        plot_widget.bind(on_touch_move=lambda inst, touch: self.on_plot_touch_move(inst, touch))
        plot_widget.bind(on_touch_up=lambda inst, touch: self.on_plot_touch_up(inst, touch))

        popup = Popup(title="Node Info & Subtree Plot", content=content, size_hint=(0.9,0.9))
        popup.open()
        Clock.schedule_interval(lambda dt: self.draw_multi_subtree_plot(plot_widget), 0.5)

    def update_plot_range(self, plot_widget, val, is_min, label):
        if is_min: plot_widget.min_x = val; label.text=f"Min X: {val:.2f}"
        else: plot_widget.max_x = val; label.text=f"Max X: {val:.2f}"
        self.draw_multi_subtree_plot(plot_widget)

    # --- Tooltip ---
    def on_plot_hover(self, plot_widget, touch):
        if not (0 <= touch.x <= plot_widget.width and 0 <= touch.y <= plot_widget.height): return
        width, height = plot_widget.width, plot_widget.height
        min_x, max_x = plot_widget.min_x, plot_widget.max_x
        xs = np.linspace(min_x, max_x, 100)
        idx = int((touch.x / width) * (len(xs)-1))
        x_val = xs[idx]
        tooltip_text = f"x={x_val:.2f}\n"
        for i,node_expr in enumerate(plot_widget.selected_nodes):
            y_val = eval_expr(node_expr, x_val)
            target_val = x_val**2 + x_val
            error = abs(y_val-target_val)
            tooltip_text += f"Node {i+1}: y={y_val:.2f}, error={error:.2f}\n"
        if hasattr(plot_widget, 'tooltip_label'):
            plot_widget.tooltip_label.text = tooltip_text
            plot_widget.tooltip_label.pos = (touch.x+10, touch.y+10)
        else:
            from kivy.uix.label import Label
            tooltip_label = Label(text=tooltip_text, size_hint=(None,None), size=(180,90), color=(0,0,0,1))
            plot_widget.tooltip_label = tooltip_label
            plot_widget.add_widget(tooltip_label)
    def remove_tooltip(self, plot_widget):
        if hasattr(plot_widget, 'tooltip_label'):
            plot_widget.remove_widget(plot_widget.tooltip_label)
            del plot_widget.tooltip_label

    # --- Zoom & Pan ---
    def on_plot_touch_down(self, plot_widget, touch):
        if 'button' in touch.profile:
            if touch.button=='scrollup': plot_widget.zoom*=1.2; self.draw_multi_subtree_plot(plot_widget)
            elif touch.button=='scrolldown': plot_widget.zoom/=1.2; self.draw_multi_subtree_plot(plot_widget)
        if touch.is_double_tap: plot_widget.zoom=1.0; plot_widget.offset_x=0; plot_widget.offset_y=0; self.draw_multi_subtree_plot(plot_widget)
        else: plot_widget.pan_start=(touch.x, touch.y)

    def on_plot_touch_move(self, plot_widget, touch):
        if plot_widget.pan_start:
            dx=(touch.x - plot_widget.pan_start[0])/plot_widget.zoom
            dy=(touch.y - plot_widget.pan_start[1])/plot_widget.zoom
            plot_widget.offset_x+=dx; plot_widget.offset_y+=dy
            plot_widget.pan_start=(touch.x, touch.y)
            self.draw_multi_subtree_plot(plot_widget)

    def on_plot_touch_up(self, plot_widget, touch): plot_widget.pan_start=None

    # --- Draw multi-subtree plot ---
    def draw_multi_subtree_plot(self, plot_widget):
        plot_widget.canvas.clear()
        width, height = plot_widget.width, plot_widget.height
        min_x, max_x = plot_widget.min_x, plot_widget.max_x
        xs = np.linspace(min_x, max_x, 100)
        ys_target = [x**2 + x for x in xs]
        min_y=min(ys_target); max_y=max(ys_target)
        all_ys=[]; errors=[]
        for node_expr in plot_widget.selected_nodes:
            ys=[eval_expr(node_expr,x) for x in xs]
            all_ys.append(ys)
            min_y=min(min_y,min(ys)); max_y=max(max_y,max(ys))
            errors.append(sum(abs(np.array(ys)-np.array(ys_target))))
        rng_y=max_y-min_y+1e-5
        colors=[(1,0,0),(0,0,1),(1,0.5,0),(0,1,1),(0.5,0,0.5)]
        best_idx=int(np.argmin(errors))
        with plot_widget.canvas:
            # Target
            Color(0,1,0)
            points_target=[]
            for i,y in enumerate(ys_target):
                x_pos=((i*(width/99)+plot_widget.offset_x)*plot_widget.zoom)
                y_pos=(((y-min_y)/rng_y)*height + plot_widget.offset_y)*plot_widget.zoom
                points_target.extend([x_pos,y_pos])
            Line(points=points_target,width=2)
            # Subtrees
            for idx, ys in enumerate(all_ys):
                Color(*colors[idx%len(colors)])
                points=[]
                for i,y in enumerate(ys):
                    x_pos=((i*(width/99)+plot_widget.offset_x)*plot_widget.zoom)
                    y_pos=(((y-min_y)/rng_y)*height + plot_widget.offset_y)*plot_widget.zoom
                    points.extend([x_pos,y_pos])
                Line(points=points,width=3 if idx==best_idx else 2)

    # --- Export ---
    def export_plot(self, plot_widget):
        from kivy.graphics import Fbo, ClearColor, ClearBuffers
        fbo = Fbo(size=plot_widget.size)
        with fbo:
            ClearColor(1,1,1,1); ClearBuffers()
            self.draw_multi_subtree_plot(fbo)
        fbo.draw()
        fbo.texture.save("multi_subtree_plot.png")
        print("Plot exported as multi_subtree_plot.png")

if __name__=="__main__":
    GPVisualization().run()

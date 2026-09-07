"""Paused-by-default window for autonomous same-destination route cues.

The UI can reset a trial or emit PCM, but has no human muscle control. Only the
restricted sensor packet enters the controller. The frozen evaluation checkpoint
is read-only; explicitly saved and automatically saved life uses another file.
"""
from __future__ import annotations

import argparse
from collections import deque
import copy
import hashlib
import importlib.util
import json
import logging
from pathlib import Path
import tempfile
import time

import numpy as np

from route_switch_controller import RouteSwitchController
from route_switch_environment import RouteSwitchEnv


HERE = Path(__file__).resolve().parent
# environment.py imports the physical world from another folder that also has
# a viewer.py. Resolve the intended renderer by its exact file, not sys.path.
_draw_spec = importlib.util.spec_from_file_location("_beacon_drawing_only", HERE/"viewer.py")
_draw_module = importlib.util.module_from_spec(_draw_spec)
_draw_spec.loader.exec_module(_draw_module)
DrawingOnly = _draw_module.Viewer
RESULTS = HERE / "results"
TRAINED_PATH = RESULTS / "route_switch_evaluation.npz"
LIVE_PATH = RESULTS / "living_route_switch.npz"
TEXTURE_TRAINED_PATH = RESULTS / "route_texture_evaluation.npz"
TEXTURE_LIVE_PATH = RESULTS / "living_route_texture.npz"
DT = .1
ROUTE_LABELS = {"bottom": "下方通道", "top": "上方通道"}
STARTS = {"左侧中部": (1.3, 3.4, 0.), "左侧下部": (1.3, 1.4, 0.),
          "左侧上部": (1.3, 6.6, 0.)}


class RouteSwitchSession:
    """Sensor -> independent prediction -> physical consequence -> scalar reward."""

    def __init__(self, environment, controller, *, learning=True):
        self.environment, self.controller = environment, controller
        self.learning = bool(learning)
        self.info = {}
        self.frames = 0
        self.elapsed_ms = 0.
        self.reward_given = False
        self.dirty = False
        self.trail = deque(maxlen=2400)
        self.refresh_view()

    def refresh_view(self):
        self.view = self.environment.get_view_state()
        self.trail.append(tuple(self.view["position"]))
        return self.view

    def tick(self):
        if self.environment.private_metrics()["terminated"]:
            self.info.update(ui_terminated=True, ui_reward=0.)
            return self.info
        begin = time.perf_counter()
        packet = self.environment.sensor_packet()
        muscles, info = self.controller.observe_then_act(packet, learning=self.learning)
        muscles = np.asarray(muscles, dtype=float)
        if muscles.shape != (4,) or not np.isfinite(muscles).all():
            raise ValueError("模型必须输出四个有限肌肉活动值")
        if info.get("teaching"):
            raise RuntimeError("本窗口只允许自主预测")
        self.environment.step(muscles, DT)
        self.frames += 1
        self.dirty |= self.learning
        metrics = self.environment.private_metrics()
        reward = 0.
        # Never reward merely arriving at red through the wrong passage.
        if self.learning and not self.reward_given and metrics["terminated"] and metrics["correct_arrival"]:
            self.controller.receive_reward(1.)
            self.reward_given = True
            reward = 1.
        self.info = {**info, "ui_muscles": muscles.tolist(), "ui_reward": reward,
                     "ui_terminated": metrics["terminated"]}
        self.elapsed_ms = (time.perf_counter()-begin)*1000
        self.refresh_view()
        return self.info

    def new_episode(self, route, *, start=STARTS["左侧中部"]):
        self.environment.reset(target="red", route=route, start=start)
        self.controller.reset_activity()
        self.frames = 0
        self.info = {}
        self.reward_given = False
        self.trail.clear()
        self.refresh_view()

    def emit_route_cue(self, route):
        # Does not call reset_activity, modify neural state or supply an action.
        self.environment.emit_route_cue(route)
        self.refresh_view()

    def save(self, path=LIVE_PATH):
        destination = Path(path).resolve()
        if destination in (TRAINED_PATH.resolve(), TEXTURE_TRAINED_PATH.resolve()):
            raise ValueError("正式评估存档只读；请选择生活存档路径")
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.controller.save(destination)
        self.dirty = False


class RouteSwitchViewer:
    # Reuse only drawing routines. No old session, keyboard handlers, manual
    # controls, reward logic, timers or old checkpoint loading is inherited.
    color = staticmethod(DrawingOnly.color)
    draw_sensors = DrawingOnly.draw_sensors
    draw_activity = DrawingOnly.draw_activity

    def __init__(self, root, session, message, *, save_path=LIVE_PATH):
        import tkinter as tk
        from tkinter import ttk
        self.root, self.session, self.save_path = root, session, Path(save_path)
        self.running = False
        self.closed = False
        self.after_id = None
        self.last_save = time.monotonic()
        self.performance = deque(maxlen=300)
        root.title("关联神经元 · 同一目的地，途中声音改道")
        root.geometry("1240x850")
        root.minsize(1080, 760)
        root.protocol("WM_DELETE_WINDOW", self.close)
        style = ttk.Style(root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure(".", font=("Microsoft YaHei UI", 10), background="#edf2f7", foreground="#263d54")
        style.configure("TButton", padding=(6, 5))
        shell = ttk.Frame(root, padding=15)
        shell.pack(fill="both", expand=True)
        ttk.Label(shell, text="目的地保持，途中改变路线", font=("Microsoft YaHei UI", 20, "bold")).pack(anchor="w")
        ttk.Label(shell, text="两种任意短音 · 自主四肌肉控制 · 尚非自然语言指令或通用规划",
                  foreground="#667d92").pack(anchor="w", pady=(4, 12))
        content = ttk.Frame(shell)
        content.pack(fill="both", expand=True)
        content.columnconfigure(0, weight=1)
        content.rowconfigure(0, weight=1)
        left = ttk.Frame(content)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        scroll_shell = ttk.Frame(content)
        scroll_shell.grid(row=0, column=1, sticky="nsew")
        self.right_canvas = tk.Canvas(scroll_shell, width=360, height=450, bg="#edf2f7", highlightthickness=0)
        scrollbar = ttk.Scrollbar(scroll_shell, orient="vertical", command=self.right_canvas.yview)
        self.right_canvas.configure(yscrollcommand=scrollbar.set)
        self.right_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        right = ttk.Frame(self.right_canvas)
        window = self.right_canvas.create_window((0, 0), window=right, anchor="nw")
        right.bind("<Configure>", lambda event: self.right_canvas.configure(scrollregion=self.right_canvas.bbox("all")))
        self.right_canvas.bind("<Configure>", lambda event: self.right_canvas.itemconfigure(window, width=event.width))
        self.canvas = tk.Canvas(left, width=760, height=470, bg="#112237", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda event: self.draw())
        self.sensor_canvas = tk.Canvas(left, height=72, bg="white", highlightthickness=0)
        self.sensor_canvas.pack(fill="x", pady=7)
        self.target_var = tk.StringVar()
        ttk.Label(left, textvariable=self.target_var, wraplength=720, foreground="#3b607f").pack(anchor="w", pady=(0, 8))
        self.log_box = tk.Text(left, height=5, state="disabled", wrap="word", relief="flat",
                               bg="white", font=("Microsoft YaHei UI", 9))
        self.log_box.pack(fill="x")

        control = ttk.LabelFrame(right, text="运行与生活存档", padding=9)
        control.pack(fill="x", pady=(0, 9))
        self.run_button = ttk.Button(control, text="开始", command=self.toggle)
        self.run_button.grid(row=0, column=0, sticky="ew", padx=(0, 5))
        ttk.Button(control, text="保存生活", command=self.save).grid(row=0, column=1, sticky="ew")
        self.learning_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(control, text="在线学习（真实成功才给奖励）", variable=self.learning_var,
                        command=self.set_learning).grid(row=1, column=0, columnspan=2, sticky="w", pady=7)
        ttk.Button(control, text="恢复生活存档并暂停", command=self.resume_living).grid(row=2, column=0, columnspan=2, sticky="ew")
        control.columnconfigure(0, weight=1)
        control.columnconfigure(1, weight=1)

        trials = ttk.LabelFrame(right, text="新回合 · 红色目的地不变", padding=9)
        trials.pack(fill="x", pady=(0, 9))
        self.start_var = tk.StringVar(value="左侧中部")
        ttk.Combobox(trials, textvariable=self.start_var, values=list(STARTS), state="readonly", width=14).grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        for index, route in enumerate(("bottom", "top")):
            ttk.Button(trials, text=f"新回合：{ROUTE_LABELS[route]}", command=lambda r=route: self.new_episode(r)).grid(row=1, column=index, sticky="ew", padx=2)
            trials.columnconfigure(index, weight=1)
        ttk.Label(trials, text="重新放置身体、清空短期活动；保留学过的连接。", wraplength=320,
                  foreground="#667d92", font=("Microsoft YaHei UI", 9)).grid(row=2, column=0, columnspan=2, sticky="w", pady=(6, 0))

        cues = ttk.LabelFrame(right, text="途中发声 · 身体与模型继续", padding=9)
        cues.pack(fill="x", pady=(0, 9))
        self.cue_buttons = []
        for index, route in enumerate(("bottom", "top")):
            frequency = session.environment._route_tone_map[route]
            button = ttk.Button(cues, text=f"{int(frequency)} Hz\n{ROUTE_LABELS[route]}", command=lambda r=route: self.emit_cue(r))
            button.grid(row=0, column=index, sticky="ew", padx=2)
            cues.columnconfigure(index, weight=1)
            self.cue_buttons.append(button)
        ttk.Label(cues, text="每次响0.6秒。脑只收到模拟PCM；本窗口不播放电脑扬声器。",
                  wraplength=320, foreground="#667d92", font=("Microsoft YaHei UI", 9)).grid(row=1, column=0, columnspan=2, sticky="w", pady=(6, 0))

        activity = ttk.LabelFrame(right, text="当前神经与身体活动", padding=9)
        activity.pack(fill="x")
        self.phase_var = tk.StringVar(value="已暂停")
        ttk.Label(activity, textvariable=self.phase_var, foreground="#1e6fac").pack(anchor="w")
        self.stats_var = tk.StringVar()
        ttk.Label(activity, textvariable=self.stats_var, wraplength=325, justify="left",
                  font=("Microsoft YaHei UI", 9)).pack(anchor="w", pady=7)
        self.activity_canvas = tk.Canvas(activity, height=75, bg="white", highlightthickness=0)
        self.activity_canvas.pack(fill="x")
        self.status_var = tk.StringVar(value=message)
        ttk.Label(shell, textvariable=self.status_var, wraplength=1170, foreground="#60768c").pack(anchor="w", pady=(10, 0))
        self.log(message)
        self.log("首次到红即暂停。路线不合规不给奖励；换路音不保证模型能完成。")
        self.draw()
        self.update_stats()
        self.after_id = root.after(100, self.loop)

    def log(self, text):
        self.log_box.configure(state="normal")
        self.log_box.insert("end", f"{time.strftime('%H:%M:%S')}  {text}\n")
        if int(self.log_box.index("end-1c").split(".")[0]) > 150:
            self.log_box.delete("1.0", "21.0")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")
        logging.info(text)

    def draw(self):
        DrawingOnly.draw(self)
        if not hasattr(self, "canvas"):
            return
        state, canvas = self.session.view, self.canvas
        surfaces = state.get("wall_texture_surfaces", [])
        if not state.get("texture_enabled") or not surfaces:
            return
        width, height = float(state["width"]), float(state["height"])
        scale = min((max(canvas.winfo_width(), 100)-40)/width,
                    (max(canvas.winfo_height(), 100)-40)/height)
        if scale <= 0:
            return
        ox, oy = (canvas.winfo_width()-width*scale)/2, (canvas.winfo_height()+height*scale)/2
        xy = lambda p: (ox+p[0]*scale, oy-p[1]*scale)
        # In this top-down 2-D view the visible vertical wall faces are their
        # boundary lines. Their sampled pigment is the same function used by
        # the actual camera ray hits, not a separate decorative image.
        foreground = next((item for item in canvas.find_all() if canvas.type(item) == "oval"), None)
        for surface in surfaces:
            points, colors = surface["points"], surface["colors"]
            for a, b, rgb in zip(points[:-1], points[1:], colors):
                canvas.create_line(*xy(a), *xy(b), fill=self.color(rgb),
                                   width=max(2., min(5., .06*scale)), tags=("wall_texture",))
        # Paint belongs above the solid wall geometry and below beacons, rays
        # and the moving body. It never hides the live body contact feedback.
        if foreground is not None:
            canvas.tag_lower("wall_texture", foreground)

    def error(self, text, exc):
        self.running = False
        self.run_button.configure(text="继续")
        self.status_var.set(f"{text}：{type(exc).__name__}: {exc}")
        self.log(self.status_var.get())
        logging.exception(text)

    def set_learning(self):
        self.session.learning = bool(self.learning_var.get())
        self.log("在线学习已开启。" if self.session.learning else "已冻结学习；身体仍可运行。")

    def toggle(self):
        if self.session.environment.private_metrics()["terminated"]:
            self.log("本回合已结束，请选择新回合。")
            return
        self.running = not self.running
        self.run_button.configure(text="暂停" if self.running else "继续")
        self.update_stats()

    def new_episode(self, route):
        try:
            self.session.new_episode(route, start=STARTS[self.start_var.get()])
            self.running = False
            self.run_button.configure(text="开始")
            self.log(f"新回合：红色目的地、{ROUTE_LABELS[route]}；按开始运行。")
            self.draw()
            self.update_stats()
        except Exception as exc:
            self.error("新回合失败", exc)

    def emit_cue(self, route):
        try:
            self.session.emit_route_cue(route)
            self.log(f"发出{int(self.session.environment._route_tone_map[route])} Hz短音，要求{ROUTE_LABELS[route]}；未重置身体或脑。")
            self.draw()
            self.update_stats()
        except Exception as exc:
            self.error("发声失败", exc)

    def save(self):
        try:
            self.session.save(self.save_path)
            self.last_save = time.monotonic()
            self.status_var.set(f"生活已保存：{self.save_path}")
            self.log("生活模型已保存；正式评估存档保持只读。")
            return True
        except Exception as exc:
            self.error("保存失败", exc)
            return False

    def resume_living(self):
        try:
            controller = RouteSwitchController.load(self.save_path)
            route = self.session.environment.private_metrics()["requested_route"] or "bottom"
            self.session.controller = controller
            self.session.dirty = False
            self.new_episode(route)
            self.status_var.set(f"已恢复生活连接：{self.save_path}；新回合暂停。")
        except Exception as exc:
            self.error("恢复生活失败", exc)

    def update_stats(self):
        info, state = self.session.info, self.session.view
        metrics = state["private_metrics"]
        terminal = metrics["terminated"]
        self.phase_var.set("已结束，请开始新回合" if terminal else "自主运行" if self.running else "已暂停")
        for button in self.cue_buttons:
            button.configure(state="disabled" if terminal else "normal")
        show = lambda values: " / ".join(f"{float(v):.2f}" for v in values)
        controller = self.session.controller
        goal = np.asarray(controller.context[:3])
        rule = np.asarray(controller.rule_context)
        muscles = info.get("ui_muscles", (0, 0, 0, 0))
        p95 = float(np.percentile(self.performance, 95)) if self.performance else 0.
        self.stats_var.set(
            f"模拟：{state['time']:.1f}秒 / {self.session.frames}帧\n"
            f"目标群（220/440/660）：{show(goal)}\n"
            f"规则群（静息/990/1320）：{show(rule)}\n"
            f"混合细胞活动：{info.get('pfc_active_count', 0)}\n"
            f"共同回忆：{info.get('recall_count', 0)}个事件\n"
            f"顺序支撑事件：{info.get('sequence_supported_events', 0)}\n"
            f"动作来源：{info.get('source', '尚未运行')}\n"
            f"四肌肉（左前/左后/右前/右后）：\n{show(muscles)}\n"
            f"接触帧：{metrics['contact_steps']}；已给奖励：{int(self.session.reward_given)}\n"
            f"帧耗时：{self.session.elapsed_ms:.1f}ms；P95：{p95:.1f}ms"
        )
        ids = np.flatnonzero(controller.pfc_activity)
        groups = np.histogram(ids, bins=16, range=(0, controller.mixed_neurons))[0]
        self.draw_activity(groups)
        route = ROUTE_LABELS.get(metrics["requested_route"], "尚无")
        first = ROUTE_LABELS.get(metrics["chosen_route"], "尚未完整通过")
        cue = "短音输入中" if state["cue_active"] or state["route_cue_active"] else "已静音"
        outcome = ("正确抵达" if metrics["correct_arrival"] else "首次抵达路线不合规") if terminal else "进行中"
        self.target_var.set(f"环境评分：红色目的地 · 要求{route} · 首次通道：{first}\n{cue}；{outcome}。地图、要求和评分均不传给脑。")

    def loop(self):
        if self.closed:
            return
        begin = time.perf_counter()
        if self.running:
            try:
                info = self.session.tick()
                self.performance.append(self.session.elapsed_ms)
                if info["ui_terminated"]:
                    self.running = False
                    self.run_button.configure(text="回合结束")
                    metrics = self.session.view["private_metrics"]
                    self.log("正确抵达，已给奖励+1并暂停。" if info["ui_reward"] else
                             "正确抵达，学习已关闭；暂停。" if metrics["correct_arrival"] else
                             "首次到达红色但路线不合规：无奖励，已暂停。")
                self.draw()
                self.update_stats()
            except Exception as exc:
                self.error("逐帧失败，已暂停", exc)
        if self.session.dirty and time.monotonic()-self.last_save >= 300:
            self.save()
        self.after_id = self.root.after(max(1, int(100-(time.perf_counter()-begin)*1000)), self.loop)

    def close(self, *, save=True):
        self.closed = True
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
        if save and self.session.dirty:
            self.save()
        self.root.destroy()


def build_session(*, seed=2026, resume_living=False, model=None, texture_seed=None):
    textured = texture_seed is not None
    default_model = TEXTURE_TRAINED_PATH if textured else TRAINED_PATH
    living_model = TEXTURE_LIVE_PATH if textured else LIVE_PATH
    source = living_model if resume_living else Path(model) if model is not None else default_model
    if not source.exists():
        raise FileNotFoundError(f"未找到存档 {source}；先完成对应训练或保存生活")
    controller = RouteSwitchController.load(source)
    controller.reset_activity()
    if textured:
        from textured_route_environment import TexturedRouteEnv
        environment = TexturedRouteEnv(seed=seed, texture_seed=texture_seed)
    else:
        environment = RouteSwitchEnv(seed=seed)
    environment.reset(target="red", route="bottom", start=STARTS["左侧中部"])
    appearance = f"墙面纹理种子{texture_seed}" if textured else "原灰墙环境"
    return RouteSwitchSession(environment, controller), f"已载入 {source.name}；{appearance}；初始暂停，在线学习开启。"


def smoke_test():
    """Short real physics/learning loop and withdrawn Tk; temporary saves only."""
    from route_state_digest import digest
    tracked = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in (TRAINED_PATH, LIVE_PATH, TEXTURE_LIVE_PATH) if p.exists()}
    environment = RouteSwitchEnv(seed=201)
    environment.reset(route="bottom")
    controller = RouteSwitchController(seed=2026, mixed_neurons=512, max_events=512)
    session = RouteSwitchSession(environment, controller)
    position = environment.world.position.copy()
    heading = environment.world.heading
    for _ in range(20):
        session.tick()
    assert session.frames == 20 and (not np.array_equal(position, environment.world.position)
                                    or heading != environment.world.heading)
    assert not session.reward_given
    before_brain = digest(controller)
    before_world = copy.deepcopy(environment.world.get_state())
    session.emit_route_cue("top")
    assert digest(controller) == before_brain and environment.world.get_state() == before_world
    for _ in range(8):
        session.tick()
    assert np.array_equal(controller.context[:3], [1., 0., 0.])
    assert np.array_equal(controller.rule_context, [0., 0., 1.])
    assert np.count_nonzero(environment.sensor_packet()["waveform"]) == 0
    # Actual no-passage initial arrival: even repeated session ticks cannot pay.
    session.new_episode("bottom", start=(8.5, 1.5, 0.))
    frames = session.frames
    session.tick()
    session.tick()
    assert session.frames == frames and not session.reward_given
    assert session.info["ui_terminated"]
    session.new_episode("top")
    for protected in (TRAINED_PATH, TEXTURE_TRAINED_PATH):
        try:
            session.save(protected)
            raise AssertionError("A protected evaluation path was writable")
        except ValueError:
            pass
    with tempfile.TemporaryDirectory(prefix="route_viewer_smoke_") as directory:
        saved = Path(directory)/"temporary_life.npz"
        session.save(saved)
        restored = RouteSwitchController.load(saved)
        assert np.array_equal(restored.rule_context, controller.rule_context)
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        viewer = RouteSwitchViewer(root, session, "隐藏布局烟测", save_path=saved)
        root.update_idletasks()
        root.update()
        assert not viewer.running and viewer.learning_var.get()
        assert len(viewer.cue_buttons) == 2
        viewer.emit_cue("bottom")
        viewer.draw()
        viewer.update_stats()
        assert viewer.canvas.find_all() and viewer.sensor_canvas.find_all()
        labels = {"root_requested_width": root.winfo_reqwidth(), "root_requested_height": root.winfo_reqheight(),
                  "canvas_items": len(viewer.canvas.find_all()), "right_scroll_region": viewer.right_canvas.cget("scrollregion")}
        viewer.close(save=False)
        from textured_route_environment import TexturedRouteEnv
        textured_session = RouteSwitchSession(TexturedRouteEnv(seed=201, texture_seed=44017),
            RouteSwitchController(seed=2026, mixed_neurons=512, max_events=512))
        for _ in range(5):
            textured_session.tick()
        root = tk.Tk()
        root.withdraw()
        texture_viewer = RouteSwitchViewer(root, textured_session, "隐藏纹理布局烟测", save_path=saved)
        root.update_idletasks()
        root.update()
        texture_viewer.draw()
        texture_item_count = len(texture_viewer.canvas.find_withtag("wall_texture"))
        expected_segments = sum(len(surface["colors"]) for surface in textured_session.view["wall_texture_surfaces"])
        assert texture_item_count == expected_segments and texture_item_count > 0
        assert np.array_equal(textured_session.environment.sensor_packet()["observation"]["ray_colors"],
                              textured_session.view["ray_colors"])
        texture_viewer.close(save=False)
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == h for p, h in tracked.items())
    return {"real_physics_28_frames": True, "same_world_and_brain_at_mid_cue": True,
            "silent_goal_and_changed_rule_retained": True,
            "wrong_route_arrival_never_rewarded_and_no_late_steps": True,
            "temporary_checkpoint_roundtrip": True, "withdrawn_tk_layout": labels,
            "textured_real_physics_frames": 5,
            "withdrawn_texture_viewer_surface_segments": texture_item_count,
            "texture_viewer_and_brain_ray_colors_identical": True,
            "both_evaluation_checkpoint_paths_reject_save": True,
            "official_and_living_checkpoints_unchanged": True,
            "limitation": "Interface smoke test; not evidence of learned navigation or switching"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--resume-living", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--model", type=Path, help="Explicit checkpoint; default stays the gray-wall model")
    parser.add_argument("--texture-seed", type=int, help="Opt in to visible wall pigment; omitted means original gray walls")
    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=RESULTS/"route_switch_viewer.log", level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", encoding="utf-8")
    if args.smoke_test:
        report = smoke_test()
        (RESULTS/"route_switch_viewer_smoke.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    import tkinter as tk
    from tkinter import messagebox
    root = tk.Tk()
    try:
        session, message = build_session(seed=args.seed, resume_living=args.resume_living,
                                         model=args.model, texture_seed=args.texture_seed)
        living_path = LIVE_PATH if args.texture_seed is None else TEXTURE_LIVE_PATH
        RouteSwitchViewer(root, session, message, save_path=living_path)
        root.mainloop()
    except Exception as exc:
        logging.exception("改道窗口启动失败")
        messagebox.showerror("改道窗口启动失败", f"{type(exc).__name__}: {exc}")
        root.destroy()
        raise


if __name__ == "__main__":
    main()

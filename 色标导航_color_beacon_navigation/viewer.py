"""Tk view for sound-cued beacon navigation; rendering never controls the brain.

Only sensor_packet() output reaches observe_then_act(). Geometry and the scoring
target remain in the environment/UI. After the actually executed action first
reaches the target, the external scorer delivers one scalar receive_reward(1).
W/A/S/D/Space are explicit human muscle demonstrations, not an automatic expert.
The controller predicts BEFORE receiving a demonstration update; while a key is
held the world executes the human demonstration, and the UI displays both values.
"""

from __future__ import annotations

import argparse
from collections import deque
import logging
import math
from pathlib import Path
import sys
import tempfile
import time

import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
RESULTS = HERE / "results"
LIVE_PATH = RESULTS / "living_navigation.npz"
TRAINED_PATH = RESULTS / "navigation_brain.npz"
DT = .1
MANUAL_MUSCLES = {
    "w": np.array([.6, 0, .6, 0]),
    "s": np.array([0, .5, 0, .5]),
    "a": np.array([0, .3, .3, 0]),
    "d": np.array([.3, 0, 0, .3]),
    "space": np.zeros(4),
}
MANUAL_LABELS = {"w": "前进", "s": "后退", "a": "左转", "d": "右转", "space": "静息"}


def _numbers(value, default=()):
    try:
        array = np.asarray(value, dtype=float).ravel()
        return array if np.isfinite(array).all() else np.asarray(default, dtype=float)
    except (TypeError, ValueError):
        return np.asarray(default, dtype=float)


class NavigationSession:
    """The entire integration boundary: packet -> model -> muscles -> world."""

    def __init__(self, environment, controller, *, seed=2026):
        self.environment = environment
        self.controller = controller
        self.rng = np.random.default_rng(seed)
        self.selected_target = 0
        self.info = {}
        self.view = {}
        self.elapsed_ms = 0.
        self.frames = 0
        self.manual_frames = 0
        self.reward_given = False
        self.trail = deque(maxlen=1600)
        self.refresh_view()

    def refresh_view(self):
        self.view = self.environment.get_view_state()
        if "position" in self.view:
            self.trail.append(tuple(self.view["position"]))
        return self.view

    def tick(self, teacher_muscles=None):
        started = time.perf_counter()
        # No UI geometry/target/trial identifier is added to this packet.
        packet = self.environment.sensor_packet()
        teacher = None if teacher_muscles is None else np.asarray(teacher_muscles, dtype=float).copy()
        predicted, model_info = self.controller.observe_then_act(packet, teacher_muscles=teacher, learning=True)
        predicted = np.asarray(predicted, dtype=float)
        if predicted.shape != (4,) or not np.isfinite(predicted).all():
            raise ValueError("controller must return four finite muscle activations")
        applied = predicted if teacher is None else teacher
        self.environment.step(applied, dt=DT)
        self.frames += 1
        self.manual_frames += int(teacher is not None)
        self.info = dict(model_info)
        self.info["ui_predicted_muscles"] = predicted.tolist()
        self.info["ui_applied_muscles"] = applied.tolist()
        self.info["ui_manual_demonstration"] = teacher is not None
        self.refresh_view()
        # A real consequence is scored only AFTER the applied muscles moved the
        # world. No target identity or coordinate reaches the neural controller.
        self.info["ui_reward"] = 0.
        if not self.reward_given and self.view.get("private_metrics", {}).get("target_reached"):
            self.controller.receive_reward(1.0)
            self.reward_given = True
            self.info["ui_reward"] = 1.0
        self.elapsed_ms = (time.perf_counter() - started) * 1000
        return self.info

    def new_episode(self, target=None, *, change_start=False):
        if target is not None:
            self.selected_target = int(target)
        start = (self._random_start() if change_start else
                 self.view.get("private_metrics", {}).get("initial_pose"))
        self.environment.reset(start=start, target=self.selected_target)
        self.controller.reset_activity()
        self.frames = 0
        self.manual_frames = 0
        self.reward_given = False
        self.info = {}
        self.trail.clear()
        self.refresh_view()

    def _random_start(self):
        """Scene placement for a human-requested reset; never an action policy."""
        current_pose = self.view.get("private_metrics", {}).get("initial_pose", ())
        choices = [name for name, pose in self.environment.scene.starts.items()
                   if not np.array_equal(np.asarray(pose), np.asarray(current_pose))]
        return str(self.rng.choice(choices)) if choices else None

    def save(self, path=LIVE_PATH):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.controller.save(path)


class Viewer:
    def __init__(self, root, session, startup_message):
        import tkinter as tk
        from tkinter import ttk
        self.root, self.tk, self.ttk = root, tk, ttk
        self.session = session
        self.running = False
        self.closed = False
        self.manual_key = None
        self.after_id = None
        self.last_autosave = time.monotonic()
        self.performance = deque(maxlen=300)
        self.last_outcome = None
        self.last_error = None
        root.title("关联神经元 · 声音与色标导航实验")
        root.geometry("1280x900")
        root.minsize(1080, 800)
        root.configure(bg="#edf2f7")
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.bind("<KeyPress>", self.key_down)
        root.bind("<KeyRelease>", self.key_up)
        root.bind("<FocusOut>", self.focus_out)
        style = ttk.Style(root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure(".", font=("Microsoft YaHei UI", 10), background="#edf2f7", foreground="#23374e")
        style.configure("TButton", padding=(6, 4))
        style.configure("TLabelframe", padding=7)
        shell = ttk.Frame(root, padding=16)
        shell.pack(fill="both", expand=True)
        ttk.Label(shell, text="声音 → 保持上下文 → 色标导航", font=("Microsoft YaHei UI", 20, "bold")).pack(anchor="w")
        ttk.Label(shell, text="纯自主探索 · 首次到达给真实奖励 · 短音后保持上下文 · 尚非自然语言或通用规划",
                  foreground="#677c91").pack(anchor="w", pady=(5, 12))
        content = ttk.Frame(shell)
        content.pack(fill="both", expand=True)
        content.columnconfigure(0, weight=1)
        content.columnconfigure(1, weight=0)
        content.rowconfigure(0, weight=1)
        left = ttk.Frame(content)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 14))
        right_shell = ttk.Frame(content)
        right_shell.grid(row=0, column=1, sticky="nsew")
        right_canvas = tk.Canvas(right_shell, width=355, height=470, bg="#edf2f7", highlightthickness=0)
        right_scroll = ttk.Scrollbar(right_shell, orient="vertical", command=right_canvas.yview)
        right_canvas.configure(yscrollcommand=right_scroll.set)
        right_canvas.pack(side="left", fill="both", expand=True)
        right_scroll.pack(side="right", fill="y")
        right = ttk.Frame(right_canvas)
        right_window = right_canvas.create_window((0, 0), window=right, anchor="nw")
        right.bind("<Configure>", lambda event: right_canvas.configure(scrollregion=right_canvas.bbox("all")))
        right_canvas.bind("<Configure>", lambda event: right_canvas.itemconfigure(right_window, width=event.width))
        self.canvas = tk.Canvas(left, bg="#112237", width=760, height=470, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda event: self.draw())
        self.canvas.bind("<Button-1>", lambda event: self.canvas.focus_set())
        self.sensor_canvas = tk.Canvas(left, bg="#fff", height=72, highlightthickness=0)
        self.sensor_canvas.pack(fill="x", pady=(8, 8))
        self.target_var = tk.StringVar(value="环境目标：等待")
        ttk.Label(left, textvariable=self.target_var, foreground="#385672", wraplength=760).pack(anchor="w", pady=(0, 6))
        log_frame = ttk.LabelFrame(left, text="经历与操作记录")
        log_frame.pack(fill="x")
        self.log_box = tk.Text(log_frame, height=4, wrap="word", state="disabled", relief="flat", bg="white", font=("Microsoft YaHei UI", 9))
        self.log_box.pack(fill="x")
        controls = ttk.LabelFrame(right, text="运行与模型")
        controls.pack(fill="x", pady=(0, 9))
        self.run_button = ttk.Button(controls, text="开始", command=self.toggle)
        self.run_button.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        ttk.Button(controls, text="保存学习", command=lambda: self.save("手动保存")).grid(row=0, column=1, sticky="ew")
        ttk.Button(controls, text="换一个起点（保留学习）", command=self.change_start).grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        controls.columnconfigure(0, weight=1)
        controls.columnconfigure(1, weight=1)
        sounds = ttk.LabelFrame(right, text="开始新任务 · 仅起始短音")
        sounds.pack(fill="x", pady=(0, 9))
        tone_map = self.session.view.get("tone_map", {})
        for index in range(3):
            name = ("red", "green", "blue")[index]
            frequency = tone_map.get(name)
            label = f"{'①②③'[index]} {int(frequency)}Hz" if frequency else f"音调{'①②③'[index]}"
            ttk.Button(sounds, text=label, command=lambda i=index: self.start_sound(i)).grid(row=0, column=index, sticky="ew", padx=2)
            sounds.columnconfigure(index, weight=1)
        ttk.Label(sounds, text="每回合仅一次短音。目标只给环境评分与界面，脑收到受体和波形。",
                  foreground="#61768c", wraplength=320, font=("Microsoft YaHei UI", 9)).grid(row=1, column=0, columnspan=3, sticky="w", pady=(5, 0))
        manual = ttk.LabelFrame(right, text="人工肌肉示范 · 按住有效")
        manual.pack(fill="x", pady=(0, 9))
        for column, key in enumerate(("w", "a", "s", "d")):
            button = ttk.Button(manual, text=f"{key.upper()} {MANUAL_LABELS[key]}")
            button.grid(row=0, column=column, padx=2, sticky="ew")
            button.bind("<ButtonPress-1>", lambda event, k=key: self.press_manual(k))
            button.bind("<ButtonRelease-1>", self.release_manual)
            button.bind("<Leave>", self.release_manual)
        ttk.Label(manual, text="空格：静息。先预测再学，释放后自主控制。\n人工轨迹不计为自主能力。",
                  foreground="#61768c", wraplength=320, font=("Microsoft YaHei UI", 9)).grid(row=1, column=0, columnspan=4, sticky="w", pady=5)
        activity = ttk.LabelFrame(right, text="模型状态 · 多簇活动")
        activity.pack(fill="both", expand=True)
        self.phase_var = tk.StringVar(value="已暂停")
        ttk.Label(activity, textvariable=self.phase_var, foreground="#1f67a7").pack(anchor="w")
        self.stats_var = tk.StringVar()
        ttk.Label(activity, textvariable=self.stats_var, justify="left", wraplength=330, font=("Microsoft YaHei UI", 9)).pack(anchor="w", pady=7)
        self.activity_canvas = tk.Canvas(activity, bg="white", height=75, highlightthickness=0)
        self.activity_canvas.pack(fill="x", pady=3)
        self.status_var = tk.StringVar(value=startup_message)
        ttk.Label(shell, textvariable=self.status_var, foreground="#5e748a", wraplength=1220).pack(anchor="w", pady=(10, 0))
        self.log(startup_message)
        self.log("生活存档：living_navigation.npz；正式评估模型保持原样。")
        self.log("音调通过模拟PCM送入脑；此窗口未连接真实麦克风或扬声器。")
        self.draw()
        self.update_stats()
        self.after_id = root.after(100, self.loop)

    def log(self, message):
        self.log_box.configure(state="normal")
        self.log_box.insert("end", f"{time.strftime('%H:%M:%S')}  {message}\n")
        if int(self.log_box.index("end-1c").split(".")[0]) > 200:
            self.log_box.delete("1.0", "21.0")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")
        logging.info(message)

    def toggle(self):
        self.running = not self.running
        self.run_button.configure(text="暂停" if self.running else "继续")
        if not self.running:
            self.release_manual()
        self.log("同一感知与学习循环继续。" if self.running else "已暂停物理与神经循环。")
        self.update_stats()

    def start_sound(self, index):
        try:
            self.release_manual()
            self.session.new_episode(index)
            self.last_outcome = None
            self.running = True
            self.run_button.configure(text="暂停")
            self.log(f"新任务音调{'①②③'[index]}；短音结束后不再提示目标。")
            self.draw()
            self.update_stats()
        except Exception as exc:
            self.error("无法开始任务", exc)

    def change_start(self):
        try:
            self.release_manual()
            self.session.new_episode(change_start=True)
            self.last_outcome = None
            self.log("换起点并重发本次起始短音；保留已学习连接。")
            self.draw()
            self.update_stats()
        except Exception as exc:
            self.error("换起点失败", exc)

    def press_manual(self, key):
        if self.manual_key != key:
            self.log(f"人工示范：{MANUAL_LABELS[key]}。")
        self.manual_key = key
        self.update_stats()

    def release_manual(self, event=None):
        self.manual_key = None
        if hasattr(self, "phase_var"):
            self.update_stats()

    def focus_out(self, event):
        def check_focus():
            if not self.closed and self.root.focus_displayof() is None:
                self.release_manual()
        self.root.after_idle(check_focus)

    def key_down(self, event):
        key = event.keysym.lower()
        if key in MANUAL_MUSCLES and not (event.state & 0x4):
            self.press_manual(key)
            return "break"

    def key_up(self, event):
        if event.keysym.lower() == self.manual_key:
            self.release_manual()
            return "break"

    def save(self, reason):
        try:
            self.session.save(LIVE_PATH)
            self.status_var.set(f"{reason}完成：{LIVE_PATH}")
            self.log(f"{reason}成功；保存脑连接与支持的活动状态。")
            return True
        except Exception as exc:
            self.error(reason + "失败", exc)
            return False

    def error(self, prefix, exc):
        logging.exception(prefix)
        self.status_var.set(f"{prefix}：{exc}；详见 results/viewer.log")
        message = f"{prefix}：{type(exc).__name__}: {exc}"
        if message != self.last_error:
            self.log(message)
            self.last_error = message

    def loop(self):
        if self.closed:
            return
        start = time.perf_counter()
        if self.running:
            try:
                teacher = MANUAL_MUSCLES[self.manual_key] if self.manual_key else None
                self.session.tick(teacher)
                self.performance.append(self.session.elapsed_ms)
                self.last_error = None
                self.draw()
                self.update_stats()
                if self.session.info.get("ui_reward", 0) > 0:
                    origin = "含人工示范的回合" if self.session.manual_frames else "自主探索回合"
                    self.log(f"{origin}首次到达目标，真实奖励 +1 已送达；已暂停，可开始新音调任务。")
                    self.running = False
                    self.run_button.configure(text="继续")
                    self.release_manual()
            except Exception as exc:
                self.running = False
                self.run_button.configure(text="继续")
                self.release_manual()
                self.error("逐帧失败，已暂停", exc)
        if time.monotonic() - self.last_autosave >= 300:
            self.save("自动保存")
            self.last_autosave = time.monotonic()
        delay = max(1, int(DT * 1000 - (time.perf_counter() - start) * 1000))
        self.after_id = self.root.after(delay, self.loop)

    def update_stats(self):
        info, state = self.session.info, self.session.view
        phase = "已暂停" if not self.running else ("人工示范 · " + MANUAL_LABELS[self.manual_key] if self.manual_key else "自主探索 · 到达奖励")
        self.phase_var.set(phase)
        heard = _numbers(info.get("heard", info.get("audio", [])))
        predicted = _numbers(info.get("ui_predicted_muscles", []))
        applied = _numbers(info.get("ui_applied_muscles", []))
        p95 = float(np.percentile(self.performance, 95)) if self.performance else 0
        active_ids = _numbers(info.get("pfc_active", [])).astype(int)
        total_neurons = int(getattr(self.session.controller, "mixed_neurons", 4096))
        groups = (np.histogram(active_ids, bins=16, range=(0, total_neurons))[0]
                  if len(active_ids) else np.array([]))
        context = _numbers(info.get("context_currents", []))
        metrics = state.get("private_metrics", {})
        def show(values):
            return " / ".join(f"{x:.2f}" for x in values) if len(values) else "尚无数据"
        self.stats_var.set(
            f"模拟帧：{self.session.frames}   时间：{state.get('time', 0):.1f}s\n"
            f"模型听到：{show(heard)}\n"
            f"上下文活动：{show(context)}\n"
            f"PFC混合细胞：{info.get('pfc_active_count', '未提供')} 个（下图16区段）\n"
            f"回忆事件：{info.get('recall_count', '未提供')} 个\n"
            f"记忆：{info.get('memory_events', info.get('memory_count', '未提供'))}\n"
            f"动作：{info.get('action_name', '等待')}（{'记忆' if info.get('source') == 'associative_memory' else '自发'}）\n"
            f"自主预测四肌肉：{show(predicted)}\n"
            f"实际施力四肌肉：{show(applied)}\n"
            f"触痛：{float(state.get('pain', 0)):.2f}\n"
            f"错误色标进入：{metrics.get('wrong_beacon_entries', 0)} 次\n"
            f"已获到达奖励：{int(self.session.reward_given)}  人工帧：{self.session.manual_frames}\n"
            f"闭环：{self.session.elapsed_ms:.1f}ms   P95：{p95:.1f}ms"
        )
        target = metrics.get("target", "未提供")
        target = {"red": "红色色标", "green": "绿色色标", "blue": "蓝色色标"}.get(target, target)
        cue = "起始短音输入中" if state.get("cue_active") else "已静音，模型需保持上下文"
        self.target_var.set(f"环境任务目标：{target}（仅评分与界面使用）　{cue}。")
        self.draw_activity(groups)

    @staticmethod
    def color(rgb, minimum=0):
        values = np.clip(np.asarray(rgb, dtype=float) * 255, minimum, 255).astype(int)
        return "#" + "".join(f"{value:02x}" for value in values[:3])

    def draw(self):
        if not hasattr(self, "canvas"):
            return
        canvas, state = self.canvas, self.session.view
        canvas.delete("all")
        width, height = float(state.get("width", 8)), float(state.get("height", 6))
        scale = min((max(canvas.winfo_width(), 100) - 40) / width, (max(canvas.winfo_height(), 100) - 40) / height)
        if scale <= 0:
            return
        ox = (canvas.winfo_width() - width * scale) / 2
        oy = (canvas.winfo_height() + height * scale) / 2
        def xy(point):
            return ox + point[0] * scale, oy - point[1] * scale
        canvas.create_rectangle(*xy((0, height)), *xy((width, 0)), fill="#142b43", outline="#8199b0", width=2)
        for x in range(1, int(width)):
            canvas.create_line(*xy((x, 0)), *xy((x, height)), fill="#203b55")
        for y in range(1, int(height)):
            canvas.create_line(*xy((0, y)), *xy((width, y)), fill="#203b55")
        for wall in state.get("walls", []):
            if isinstance(wall, dict):
                a, b, c, d = wall["bounds"]
                rgb = wall.get("color", (.4, .4, .4))
            else:
                a, b, c, d = wall
                rgb = (.4, .4, .4)
            canvas.create_rectangle(*xy((a, d)), *xy((c, b)), fill=self.color(rgb, 65), outline="#b1bac3")
        for index, item in enumerate(state.get("objects", [])):
            x, y = xy(item["position"])
            radius = float(item.get("radius", .3)) * scale
            canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill=self.color(item.get("color", (1, 0, 0))), outline="#e6f3ff")
            canvas.create_text(x, y + radius + 12, text=item.get("name", f"色标{index + 1}"), fill="#bed3e7", font=("Microsoft YaHei UI", 9))
        trail = list(self.session.trail)
        if len(trail) > 1:
            canvas.create_line(*[v for point in trail for v in xy(point)], fill="#6596bf", width=1.5)
        position = state.get("position", (width / 2, height / 2))
        x, y = xy(position)
        endpoints, colors = state.get("ray_endpoints", []), state.get("ray_colors", [])
        for point, rgb in zip(endpoints, colors):
            canvas.create_line(x, y, *xy(point), fill=self.color(rgb, 45), dash=(3, 5))
        radius = max(7, float(state.get("radius", .18)) * scale)
        canvas.create_oval(x - radius, y - radius, x + radius, y + radius,
                           fill="#e88070" if state.get("pain", 0) > .01 else "#6cd1ee", outline="white", width=2)
        angle = float(state.get("heading", 0))
        tip = (x + radius * 1.6 * math.cos(angle), y - radius * 1.6 * math.sin(angle))
        canvas.create_line(x, y, *tip, fill="white", width=3, arrow="last")
        self.draw_sensors()

    def draw_sensors(self):
        canvas, state = self.sensor_canvas, self.session.view
        canvas.delete("all")
        distances = _numbers(state.get("ray_distances", []))
        colors = state.get("ray_colors", [])
        count = len(distances)
        if not count:
            canvas.create_text(10, 15, text="等待真实受体数据", anchor="w")
            return
        cell = max(canvas.winfo_width(), 100) / count
        for index, (distance, rgb) in enumerate(zip(distances, colors)):
            x = index * cell
            canvas.create_rectangle(x + 1, 8, x + cell - 1, 28, fill=self.color(rgb), outline="")
            bar = min(1., distance / 5.) * 25
            canvas.create_rectangle(x + 2, 60 - bar, x + cell - 2, 60, fill="#639ac0", outline="")
        canvas.create_text(8, 67, text=f"{count} 束真实RGB/距离受体 · 右→左", anchor="sw", fill="#536f86", font=("Microsoft YaHei UI", 8))

    def draw_activity(self, groups):
        if not hasattr(self, "activity_canvas"):
            return
        canvas = self.activity_canvas
        canvas.delete("all")
        if not len(groups):
            canvas.create_text(8, 12, text="模型尚未提供PFC簇活动", anchor="nw", fill="#6d8091", font=("Microsoft YaHei UI", 9))
            return
        # Display reduction only: this never feeds back into the neural model.
        indices = np.arange(len(groups)) if len(groups) <= 20 else np.argsort(np.abs(groups))[-20:]
        peak = max(float(np.max(np.abs(groups))), 1e-9)
        cell = max(canvas.winfo_width(), 120) / len(indices)
        for column, index in enumerate(indices):
            value = abs(float(groups[index])) / peak
            x = column * cell
            canvas.create_rectangle(x + 2, 55 - 40 * value, x + cell - 2, 55, fill="#427cae" if groups[index] >= 0 else "#c67c70", outline="")
            canvas.create_text(x + cell / 2, 66, text=str(int(index)), fill="#647b90", font=("Microsoft YaHei UI", 8))

    def close(self):
        self.closed = True
        self.release_manual()
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
        try:
            self.save("退出保存")
        finally:
            self.root.destroy()


def build_session(seed=2026, scenario="barrier"):
    from environment import BeaconNavigationEnv
    # The final controller class is resolved here, after the modules are loaded.
    from associative_controller import AssociativeController
    controller = None
    source = "出生模型"
    for path in (LIVE_PATH, TRAINED_PATH):
        if path.exists():
            try:
                controller = AssociativeController.load(path)
                source = str(path.name)
                break
            except Exception:
                logging.exception("无法加载导航模型 %s", path)
    if controller is None:
        controller = AssociativeController()
    environment = BeaconNavigationEnv(scenario=scenario, ray_count=31, seed=seed)
    environment.reset(start="left_middle", target=0)
    controller.reset_activity()
    session = NavigationSession(environment, controller, seed=seed)
    return session, f"已载入{source}；按开始或选择音调。"


def smoke_test():
    session, message = build_session()
    assert np.asarray(session.view["position"]).shape == (2,)
    for _ in range(3):
        session.tick()
    predicted = session.tick(MANUAL_MUSCLES["w"])
    assert predicted["ui_manual_demonstration"]
    np.testing.assert_array_equal(predicted["ui_applied_muscles"], MANUAL_MUSCLES["w"])
    session.new_episode(1, change_start=True)
    session.tick()
    with tempfile.TemporaryDirectory(prefix="navigation_viewer_smoke_") as directory:
        session.save(Path(directory) / "model.npz")
    # A display/scoring-only placement starts inside a beacon to exercise the
    # real post-action reward boundary. It is not a navigation capability test.
    from environment import BeaconNavigationEnv
    from associative_controller import AssociativeController
    environment = BeaconNavigationEnv(seed=991)
    beacon = environment.scene.beacon_positions[0]
    environment.reset(start=(*beacon, 0.), target=0)
    controller = AssociativeController(exploration=False)
    rewards = []
    receive = controller.receive_reward
    def capture_reward(value):
        rewards.append(value)
        receive(value)
    controller.receive_reward = capture_reward
    scoring = NavigationSession(environment, controller)
    for _ in range(3):
        scoring.tick()
    assert rewards == [1.0], "arrival reward must be delivered once per episode"
    print("PASS: real sensor/controller/physics adapter, explicit manual demonstration, reset, temporary save, once-only reward.")
    print(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--scenario", default="barrier")
    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=RESULTS / "viewer.log", level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", encoding="utf-8")
    if args.smoke_test:
        smoke_test()
        return
    import tkinter as tk
    root = tk.Tk()
    try:
        session, message = build_session(seed=args.seed, scenario=args.scenario)
        Viewer(root, session, message)
        root.mainloop()
    except Exception as exc:
        logging.exception("导航窗口启动失败")
        from tkinter import messagebox
        messagebox.showerror("导航实验未能启动", f"{type(exc).__name__}: {exc}\n详见 {RESULTS / 'viewer.log'}")
        root.destroy()
        raise


if __name__ == "__main__":
    main()

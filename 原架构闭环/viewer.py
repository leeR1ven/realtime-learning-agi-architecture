"""Tk display of the original shared-time brain in a real physical loop.

The UI can present ordinary objects and PCM; only the environment's sensor
packet reaches the brain. There are no teacher muscles or navigation rewards.
Saving writes one atomic living snapshot, including the exact physical world.
"""
from __future__ import annotations

import argparse
from collections import deque
import copy
from dataclasses import asdict, fields
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
import time
import tkinter as tk
from tkinter import messagebox, ttk
import wave

import numpy as np

from brain import OriginalBrain, BrainConfig, packet_digest
from checkpoint import load_tree, save_tree
from experience_environment import ExperienceEnvironment, Scene


HERE = Path(__file__).resolve().parent
LIVE_PATH = HERE / "results" / "living_original.npz"
VIEWER_KIND = "original_brain_and_physical_life_v1"
PAIRINGS = (("red", "红", 220.), ("green", "绿", 440.), ("blue", "蓝", 660.))


def read_config(path=None):
    path = Path(path) if path else HERE / "results" / "continuous_default_config.json"
    document = json.loads(path.read_text(encoding="utf-8-sig"))
    allowed = {field.name for field in fields(BrainConfig)}
    if "adapter_kwargs" in document:
        result = {}
        for name, value in document["adapter_kwargs"].items():
            name = "sensory_threshold" if name == "threshold" else name
            if name in allowed:
                result[name] = value
            elif name == "motor_once" and value is True:
                pass  # Current original BrainConfig uses this original default.
            elif name == "distance_scale" and value == 16:
                pass
            else:
                raise ValueError(f"Current BrainConfig does not expose adapter parameter {name}")
        result.update(document.get("pfc_kwargs", {}))
    else:
        result = document
    if set(result) - allowed:
        raise ValueError("Unknown BrainConfig fields: " + ", ".join(sorted(set(result) - allowed)))
    return BrainConfig(**result)


def _safe_value(value):
    if isinstance(value, deque):
        return {"_life_deque": [_safe_value(v) for v in value], "maxlen": value.maxlen}
    if isinstance(value, set):
        return {"_life_set": [_safe_value(v) for v in sorted(value)]}
    if isinstance(value, dict):
        return {str(k): _safe_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.copy()
    if value is None or isinstance(value, (str, bool, int, float, np.generic)):
        return value
    raise TypeError("Unsupported physical snapshot field: " + type(value).__name__)


def _restore_value(value):
    if isinstance(value, dict):
        if set(value) == {"_life_deque", "maxlen"}:
            return deque((_restore_value(v) for v in value["_life_deque"]), maxlen=value["maxlen"])
        if set(value) == {"_life_set"}:
            return set(_restore_value(v) for v in value["_life_set"])
        return {k: _restore_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_restore_value(v) for v in value]
    return value.copy() if isinstance(value, np.ndarray) else value


def environment_snapshot(env):
    state = dict(scene=asdict(env.scene),
        attributes=_safe_value({k: v for k, v in vars(env).items() if k not in ("scene", "world", "rng")}),
        rng=copy.deepcopy(env.rng.bit_generator.state), world_config=asdict(env.world.config),
        world_attributes=_safe_value({k: v for k, v in vars(env.world).items() if k not in ("config", "wall_pigment")}))
    if hasattr(env.world, "wall_pigment"):
        state["wall_pigment"] = _safe_value(vars(env.world.wall_pigment))
    return state


def restore_environment(snapshot):
    scene_data = snapshot["scene"]
    scene = Scene(scene_data["name"], scene_data["width"], scene_data["height"],
        tuple(tuple(v) for v in scene_data["walls"]),
        tuple(tuple(v) for v in scene_data["beacon_positions"]),
        {k: tuple(v) for k, v in scene_data["starts"].items()})
    attributes = _restore_value(snapshot["attributes"])
    env = ExperienceEnvironment(scene, seed=0, texture_seed=attributes.get("texture_seed"),
        ray_count=attributes["ray_count"], cue_seconds=attributes["cue_seconds"],
        cue_amplitude=attributes["cue_amplitude"], beacon_radius=attributes["beacon_radius"],
        arrival_tolerance=attributes["arrival_tolerance"],
        fast_internal_observe=attributes["fast_internal_observe"])
    if set(attributes) != set(vars(env)) - {"scene", "world", "rng"}:
        raise ValueError("Environment fields differ from the saved implementation")
    world_attributes = _restore_value(snapshot["world_attributes"])
    if set(world_attributes) != set(vars(env.world)) - {"config", "wall_pigment"}:
        raise ValueError("Physical world fields differ from the saved implementation")
    env.__dict__.update(attributes)
    env.rng.bit_generator.state = snapshot["rng"]
    env.world.config = type(env.world.config)(**snapshot["world_config"])
    env.world.__dict__.update(world_attributes)
    if "wall_pigment" in snapshot:
        env.world.wall_pigment.__dict__.update(_restore_value(snapshot["wall_pigment"]))
    env.world._validate_config()
    packet_digest(env.sensor_packet())
    return env


class LifeSession:
    """One act -> actual physics -> outcome, with no extra brain calls to draw."""
    def __init__(self, brain, environment=None, *, origin="初生脑 · 尚未经历当前呈现"):
        self.brain = brain
        self.environment = environment or ExperienceEnvironment(seed=20260909)
        self.packet = self.environment.sensor_packet()
        self.origin = origin
        self.info = copy.deepcopy(brain.last_info)
        self.decision = {}
        self.executed = np.asarray(self.environment.world.last_action, float).copy()
        self.trail = deque([self.environment.world.position.copy()], maxlen=4000)
        self.catalog = {}
        self.control_steps = 0
        self.last_frame_ms = 0.
        self.presentation = None

    def _catalog(self, info):
        if "index_time" not in info:
            return
        self.catalog[str(info["index_time"])] = dict(
            physical_time=float(self.environment.world.time),
            visible_rays=self.environment.visible_beacon_rays(),
            external_presentation=copy.deepcopy(self.presentation))

    def tick(self, next_presentation=None):
        started = time.perf_counter()
        muscles, decision = self.brain.observe_then_act(self.packet, learning=True)
        self._catalog(decision)
        result = self.environment.step(muscles, dt=.1)
        if next_presentation is not None:
            # The experimenter changes the real stimuli at a phase boundary;
            # the resulting actual packet is what the brain observes next.
            self.pair(*next_presentation)
            result = self.packet
        outcome = self.brain.observe_outcome(result, muscles)
        self.packet = result
        self.executed = np.asarray(muscles, float).copy()
        self.decision, self.info = decision, outcome
        self._catalog(outcome)
        self.control_steps += 1
        self.trail.append(self.environment.world.position.copy())
        self.last_frame_ms = (time.perf_counter() - started) * 1000
        return outcome

    def pair(self, beacon, frequency, duration=.8):
        env = self.environment
        heading = np.array([math.cos(env.world.heading), math.sin(env.world.heading)])
        # Exactly the external presentation geometry used by run_closed_loop.
        position = np.clip(env.world.position + 1.4 * heading,
                           (.31, .31), (env.scene.width - .31, env.scene.height - .31))
        self.packet = env.setup_pairing(beacon, frequency, duration=duration,
                                        presentation_position=position)
        self.presentation = dict(color=beacon, frequency=float(frequency))

    def tone(self, frequency, duration=.8):
        self.packet = self.environment.emit_tone(frequency, duration)
        self.presentation = dict(color=None, frequency=float(frequency))

    def hide(self):
        self.packet = self.environment.hide_beacons()
        self.presentation = None

    def stop_sound(self):
        self.packet = self.environment.stop_tone()

    def restore_beacons(self):
        self.packet = self.environment.restore_beacons()
        self.presentation = None

    def snapshot(self):
        if self.brain.pending is not None:
            raise RuntimeError("Save only after the actual outcome has been recorded")
        return dict(kind=VIEWER_KIND, brain=self.brain.snapshot(),
            environment=environment_snapshot(self.environment), packet_digest=packet_digest(self.packet),
            presentation=copy.deepcopy(self.presentation), origin=self.origin,
            catalog=copy.deepcopy(self.catalog), executed=self.executed.copy(),
            trail=[p.copy() for p in self.trail], control_steps=self.control_steps,
            info=copy.deepcopy(self.info), decision=copy.deepcopy(self.decision))

    def save(self, path=LIVE_PATH):
        save_tree(path, self.snapshot())

    @classmethod
    def load(cls, path):
        state = load_tree(path)
        if state.get("kind") != VIEWER_KIND:
            brain = OriginalBrain.from_snapshot(state)
            # A standalone brain archive contains no evidence of a body state.
            env = ExperienceEnvironment(seed=20260909)
            brain.pending = None
            brain.prepared = None
            brain.last_executed = env.world.last_action.copy()
            return cls(brain, env, origin="仅载入脑存档 · 身体从新场景开始；未执行动作已清除")
        brain = OriginalBrain.from_snapshot(state["brain"])
        env = restore_environment(state["environment"])
        obj = cls(brain, env, origin="生活存档 · 脑、身体、时间与声音状态完整恢复")
        if packet_digest(obj.packet) != state["packet_digest"]:
            raise ValueError("Restored physical sensors differ from the saved packet")
        obj.presentation = state["presentation"]
        obj.catalog = state["catalog"]
        obj.executed = np.asarray(state["executed"], float)
        obj.trail = deque((np.asarray(p, float) for p in state["trail"]), maxlen=4000)
        obj.control_steps = int(state["control_steps"])
        obj.info, obj.decision = state["info"], state["decision"]
        return obj


def rgb_hex(rgb, brightness=1.):
    values = np.clip(np.asarray(rgb) * brightness, 0, 1)
    return "#" + "".join(f"{int(round(c * 255)):02x}" for c in values)


def memory_record_note(catalog, time_id):
    """Offline display annotation; never read by the brain or sensor adapter."""
    if time_id is None:
        return "无回忆时间"
    record = catalog.get(str(time_id))
    if record is None:
        return f"t={time_id}（无该时刻实验记录）"
    visible = record.get("visible_rays", {})
    colors = [label for color, label, _ in PAIRINGS if len(visible.get(color, [])) > 0]
    return f"t={time_id} 当时可见[{ '、'.join(colors) if colors else '无色标' }]"


class Viewer:
    def __init__(self, root, session, *, save_path=LIVE_PATH):
        self.root, self.session = root, session
        self.save_path = Path(save_path)
        self.warmup_failed = False
        self.running = False
        self.warming = False
        self.closed = False
        self.after_id = None
        self.speaker_directory = tempfile.TemporaryDirectory(prefix="original-brain-tone-")
        self.speaker_enabled = tk.BooleanVar(value=False)
        self.frequency = tk.StringVar(value="220")
        self.duration = tk.StringVar(value="0.8")
        self.status = tk.StringVar(value="已暂停。设置呈现后，点单步或运行，脑才会实际经历。")
        self.stats = tk.StringVar()
        self.origin = tk.StringVar(value=session.origin)
        self.root.title("原架构闭环 · 连续感知与记忆实验")
        self.root.geometry(f"{min(1280, root.winfo_screenwidth()-50)}x{min(900, root.winfo_screenheight()-80)}")
        self.root.minsize(1040, 730)
        style = ttk.Style(root)
        style.configure("TLabel", font=("Microsoft YaHei UI", 10))
        style.configure("TButton", font=("Microsoft YaHei UI", 10), padding=5)
        shell = ttk.Frame(root, padding=14)
        shell.pack(fill="both", expand=True)
        ttk.Label(shell, text="连续感知 · 共享时间记忆 · 自主肌肉", font=("Microsoft YaHei UI", 18, "bold")).pack(anchor="w")
        ttk.Label(shell, text="原架构闭环实验：声音联想与运动；寻路尚未验收。",
                  foreground="#875b32").pack(anchor="w", pady=(4, 10))
        body = ttk.Frame(shell)
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=1)
        body.rowconfigure(0, weight=1)
        left = ttk.Frame(body)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 14))
        self.canvas = tk.Canvas(left, width=740, height=420, background="#122234", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.sensors = tk.Canvas(left, height=86, background="#f3f6f8", highlightthickness=0)
        self.sensors.pack(fill="x", pady=(8, 4))
        self.muscles = tk.Canvas(left, height=96, background="#f3f6f8", highlightthickness=0)
        self.muscles.pack(fill="x")
        right = ttk.Frame(body, width=350)
        right.grid(row=0, column=1, sticky="ns")
        ttk.Label(right, textvariable=self.origin, wraplength=340, foreground="#465e70").pack(fill="x")
        controls = ttk.Frame(right)
        controls.pack(fill="x", pady=8)
        self.run_button = ttk.Button(controls, text="运行", command=self.toggle)
        self.run_button.pack(side="left", fill="x", expand=True)
        ttk.Button(controls, text="单步 0.1 秒", command=self.single_step).pack(side="left", fill="x", expand=True, padx=(5, 0))
        ttk.Button(right, text="安全另存当前生活状态", command=self.save).pack(fill="x")
        stimuli = ttk.LabelFrame(right, text="外界呈现（不提供动作答案）", padding=8)
        stimuli.pack(fill="x", pady=10)
        for color, label, hz in PAIRINGS:
            ttk.Button(stimuli, text=f"同时呈现 {label}色色标 ＋ {int(hz)} Hz",
                       command=lambda c=color, f=hz: self.present(c, f)).pack(fill="x", pady=2)
        entry = ttk.Frame(stimuli)
        entry.pack(fill="x", pady=(8, 3))
        ttk.Label(entry, text="频率 Hz").pack(side="left")
        ttk.Entry(entry, textvariable=self.frequency, width=8).pack(side="left", padx=4)
        ttk.Label(entry, text="时长 s").pack(side="left")
        ttk.Entry(entry, textvariable=self.duration, width=6).pack(side="left", padx=4)
        ttk.Button(stimuli, text="仅发出此声音（保持当前画面）", command=self.emit).pack(fill="x", pady=2)
        visual_controls = ttk.Frame(stimuli)
        visual_controls.pack(fill="x", pady=2)
        ttk.Button(visual_controls, text="隐藏色标", command=lambda: self.action(self.session.hide, "色标已隐藏，身体和时间不变。")).pack(side="left", fill="x", expand=True)
        ttk.Button(visual_controls, text="恢复原色标", command=lambda: self.action(self.session.restore_beacons, "已恢复场景原色标位置。")).pack(side="left", fill="x", expand=True, padx=(4, 0))
        ttk.Button(stimuli, text="停止声音", command=self.stop_sound).pack(fill="x", pady=2)
        ttk.Checkbutton(stimuli, text="扬声器试听（仅展示，无麦克风）", variable=self.speaker_enabled).pack(anchor="w")
        readings = ttk.LabelFrame(right, text="当前神经状态与实际运动", padding=8)
        readings.pack(fill="both", expand=True)
        ttk.Label(readings, textvariable=self.stats, justify="left", wraplength=326).pack(anchor="w")
        self.activity = tk.Canvas(readings, height=94, width=318, background="#172637", highlightthickness=0)
        self.activity.pack(fill="x", pady=7)
        ttk.Label(readings, text="前额叶活动：每格 8 个细胞，仅作显示汇总。", foreground="#607080", wraplength=320).pack(anchor="w")
        ttk.Label(shell, textvariable=self.status, wraplength=1200, foreground="#405f70").pack(anchor="w", pady=(10, 0))
        self.canvas.bind("<Configure>", lambda _: self.draw())
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.bind("<space>", lambda event: None if isinstance(event.widget, ttk.Entry) else self.toggle())
        self.draw()

    def action(self, function, message):
        if self.warming:
            self.status.set("自动共现进行中；可点暂停保留已经真实发生的经历。")
            return
        try:
            function()
            self.status.set(message + (" 点单步或运行让脑经历。" if not self.running else ""))
            self.draw()
        except Exception as error:
            self.pause()
            self.status.set(str(error))
            messagebox.showerror("操作未完成", str(error), parent=self.root)

    def speaker(self, frequency, duration):
        if not self.speaker_enabled.get():
            return
        try:
            import winsound
            duration = min(float(duration), 8.)
            samples = .45*np.sin(2*np.pi*frequency*np.arange(int(8000*duration))/8000)
            path = Path(self.speaker_directory.name) / "presentation.wav"
            with wave.open(str(path), "wb") as stream:
                stream.setnchannels(1); stream.setsampwidth(2); stream.setframerate(8000)
                stream.writeframes((samples*32767).astype("<i2").tobytes())
            winsound.PlaySound(str(path), winsound.SND_FILENAME | winsound.SND_ASYNC)
        except (ImportError, OSError, RuntimeError) as error:
            self.status.set("脑内 PCM 已设置；扬声器试听不可用：" + str(error))

    def present(self, color, frequency):
        def operation():
            duration = float(self.duration.get())
            self.session.pair(color, frequency, duration)
            self.frequency.set(str(int(frequency)))
            self.speaker(frequency, duration)
        self.action(operation, "已在身体前方 1.4 米呈现实物与声音；未移动身体。")

    def emit(self):
        def operation():
            frequency, duration = float(self.frequency.get()), float(self.duration.get())
            self.session.tone(frequency, duration)
            self.speaker(frequency, duration)
        self.action(operation, "已发出外界声音；当前画面保持。")

    def stop_sound(self):
        def operation():
            self.session.stop_sound()
            try:
                import winsound
                winsound.PlaySound(None, 0)
            except ImportError:
                pass
        self.action(operation, "声音已停止。")

    def pause(self):
        self.running = False
        self.warming = False
        self.run_button.configure(text="运行")
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None

    def toggle(self):
        if self.running or self.warming:
            self.pause()
            self.status.set("已暂停；脑、身体与声音的仿真时间保持。")
        else:
            self.running = True
            self.run_button.configure(text="暂停")
            self._advance()

    def _advance(self):
        self.after_id = None
        if not self.running or self.closed:
            return
        started = time.perf_counter()
        try:
            self.session.tick()
            self.draw()
            self.status.set(f"在线学习中 · 实际执行 {self.session.control_steps} 步 · 本步 {self.session.last_frame_ms:.1f} ms · 暂停可检查或保存")
        except Exception as error:
            self.pause()
            self.status.set("已暂停：" + str(error))
            messagebox.showerror("闭环执行中止", str(error), parent=self.root)
            return
        delay = max(1, int(100-(time.perf_counter()-started)*1000))
        self.after_id = self.root.after(delay, self._advance)

    def single_step(self):
        self.pause()
        self.action(self.session.tick, "已完成预测 → 真实运动 → 结果记忆，一步 0.1 秒。")

    def save(self):
        self.pause()
        def operation():
            self.session.save(self.save_path)
            self.warmup_failed = False
        self.action(operation, "已原子保存脑与完整身体状态：" + str(self.save_path))

    def start_warmup(self, steps, save_path=None):
        self.pause()
        if save_path is not None:
            self.save_path = Path(save_path)
        save_path = self.save_path
        self.warmup_failed = False
        self.warming = True
        self.run_button.configure(text="暂停")
        self.status.set(f"自动共现 0/{steps}：真实物体与声音，动作由脑自主产生。")
        self.after_id = self.root.after(1, lambda: self._warmup_frame(0, steps, save_path))

    def _warmup_frame(self, frame, steps, save_path):
        self.after_id = None
        if not self.warming or self.closed:
            return
        try:
            warmup_frame(self.session, frame, steps)
            self.draw()
            self.status.set(f"自动共现 {frame+1}/{steps} · 真实自主动作 · 本步 {self.session.last_frame_ms:.1f} ms")
            if frame+1 == steps:
                self.session.origin=f"实验候选参数 · 已真实共现 {steps} 步 · 当前暂停"
                self.session.save(save_path)
                self.origin.set(self.session.origin)
                self.pause()
                self.status.set("共现已完成，脑与身体已一起保存。隐藏色标后发声音、点单步，可检查回忆。")
            else:
                self.after_id = self.root.after(1, lambda: self._warmup_frame(frame+1, steps, save_path))
        except Exception as error:
            self.pause()
            self.warmup_failed = True
            self.origin.set(f"自动共现中止 · 已完成 {self.session.control_steps} 步")
            self.status.set("共现未完成，未覆盖生活存档；关窗也不会覆盖："+str(error))
            messagebox.showerror("自动共现未完成", str(error), parent=self.root)

    def draw(self):
        if not hasattr(self, "activity") or self.closed:
            return
        session, canvas = self.session, self.canvas
        state = session.environment.world.get_state()
        canvas.delete("all")
        width, height = state["width"], state["height"]
        cw, ch = max(canvas.winfo_width(), 300), max(canvas.winfo_height(), 240)
        scale = min((cw-36)/width, (ch-36)/height)
        ox, oy = (cw-width*scale)/2, (ch+height*scale)/2
        def xy(point):
            return ox+point[0]*scale, oy-point[1]*scale
        canvas.create_rectangle(*xy((0,height)), *xy((width,0)), fill="#192e43", outline="#93a9ba", width=2)
        for x in range(1, int(width)):
            canvas.create_line(*xy((x,0)), *xy((x,height)), fill="#223b50")
        for y in range(1, int(height)):
            canvas.create_line(*xy((0,y)), *xy((width,y)), fill="#223b50")
        for wall in state["walls"]:
            a,b,c,d = wall["bounds"]
            canvas.create_rectangle(*xy((a,d)), *xy((c,b)), fill=rgb_hex(wall["color"]), outline="#97a5ad")
        for obj in state["objects"]:
            x,y = xy(obj["position"]); r = obj["radius"]*scale
            canvas.create_oval(x-r,y-r,x+r,y+r, fill=rgb_hex(obj["color"]), outline="#edf5fa", width=2)
            name = {"red":"红", "green":"绿", "blue":"蓝"}.get(obj["name"], obj["name"])
            canvas.create_text(x,y+r+12,text=name,fill="#e2edf5")
        if len(session.trail)>1:
            canvas.create_line(*[v for point in session.trail for v in xy(point)], fill="#71a0bf", width=1.5)
        x,y = xy(state["position"])
        for point,color in zip(state["ray_endpoints"], state["ray_colors"]):
            canvas.create_line(x,y,*xy(point),fill=rgb_hex(color,.8),dash=(2,5))
        r = max(7,state["radius"]*scale)
        canvas.create_oval(x-r,y-r,x+r,y+r,fill="#ef846f" if state["pain"]>.01 else "#76d5e8",outline="white",width=2)
        canvas.create_line(x,y,x+1.6*r*math.cos(state["heading"]),y-1.6*r*math.sin(state["heading"]),fill="white",width=3,arrow="last")
        self.sensors.delete("all")
        sw = max(self.sensors.winfo_width(), 300)
        self.sensors.create_text(8,10,text="31 条真实 RGB（右侧视野 → 正前 → 左侧视野）",anchor="w",fill="#405368")
        for i,(color,distance) in enumerate(zip(state["ray_colors"],state["ray_distances"])):
            a,b = 8+i*(sw-16)/31,8+(i+1)*(sw-16)/31
            self.sensors.create_rectangle(a,24,b,48,fill=rgb_hex(color),outline="#d9e2e7")
            bar = min(distance/16,1)*22
            self.sensors.create_rectangle(a+1,76-bar,b-1,76,fill="#688fac",outline="")
        self.sensors.create_text(sw-8,82,text="深度 0–16 m",anchor="e",fill="#526a7a")
        self.muscles.delete("all")
        mw=max(self.muscles.winfo_width(),300)
        for i,label in enumerate(("左前","左后","右前","右后")):
            left=i*mw/4+8; right=(i+1)*mw/4-8
            self.muscles.create_text(left,12,text=label,anchor="w",fill="#344d62")
            for y0,value,color,name in ((27,float(session.executed[i]),"#3c97b0","执行"),
                                      (56,float(state["muscle_activations"][i]),"#7eab68","实际激活")):
                self.muscles.create_rectangle(left,y0,right,y0+12,fill="#e0e7eb",outline="")
                self.muscles.create_rectangle(left,y0,left+(right-left)*value,y0+12,fill=color,outline="")
                self.muscles.create_text(left,y0+19,text=f"{name} {value:.3f}",anchor="w",fill="#52697b")
        info=session.info
        n=lambda name:len(info.get(name, []))
        source={"spontaneous_muscles":"自发肌肉", "body_reflex":"身体反射", "pfc_time_motor_recall":"前额叶→时间→运动回忆", "rest":"静息"}.get(session.decision.get("source"), "尚未执行")
        clock=session.brain.runtime.clock
        tone=session.environment._tone
        active_tone=tone if tone is not None and state["time"]<tone["end_time"]-1e-10 else None
        sound="静音" if active_tone is None else f"{active_tone['frequency_hz']:g} Hz（外界）"
        self.stats.set(f"物理时间 {state['time']:.2f} s · {sound}\n"
            f"统一时间单元 {clock.当前时间} · 活跃 {int(clock.时间激活.sum())}\n"
            f"外部前额叶 {n('external_pfc_cells')} · 当前念头 {int(session.brain.thought.sum())}\n"
            f"声音提示细胞 {n('sound_cue_cells')}\n"
            f"实验记录注释（脑无颜色标签）：\n"
            f"声音索引 {memory_record_note(session.catalog, info.get('auditory_recall_time'))}\n"
            f"联想索引 {memory_record_note(session.catalog, info.get('final_recall_time'))}\n"
            f"回忆视觉 {n('final_visual_cells')} · 运动码 {n('motor_command_cells')}\n"
            f"已执行来源：{source}\n"
            f"运动反向连接 {session.brain.adapter.motor_reverse.条数():,}\n"
            f"最近一步 {session.last_frame_ms:.1f} ms · 在线学习开启")
        self.activity.delete("all")
        cells=session.brain.thought
        grouped=np.pad(cells,(0,(-len(cells))%8)).reshape(-1,8).sum(axis=1)
        cols=64; aw=max(self.activity.winfo_width(),318); rows=math.ceil(len(grouped)/cols)
        h=94/max(rows,1); w=aw/cols
        for i,count in enumerate(grouped):
            if count:
                a,b=(i%cols)*w,(i//cols)*h
                self.activity.create_rectangle(a,b,a+w-1,b+h-1,fill=rgb_hex((.18+.55*count/8,.4+.55*count/8,.55+.4*count/8)),outline="")

    def close(self):
        self.pause()
        if not self.warmup_failed:
            try:
                self.session.save(self.save_path)
            except Exception as error:
                self.status.set("关窗保存失败，窗口保持暂停，生活状态仍在内存："+str(error))
                messagebox.showerror("生活状态未能保存", str(error), parent=self.root)
                return False
        self.closed = True
        try:
            import winsound
            winsound.PlaySound(None,0)
        except ImportError:
            pass
        self.speaker_directory.cleanup()
        self.root.destroy()
        return True


def warmup_pair(phase):
    # Same rotation as run_closed_loop.train: RGB, GBR, BRG, ...
    color,_,frequency=PAIRINGS[(phase//3+phase%3)%3]
    return color,frequency,.8


def warmup_frame(session, frame, steps):
    if frame == 0:
        session.pair(*warmup_pair(0))
    next_pair = warmup_pair((frame+1)//8) if (frame+1)%8 == 0 and frame+1<steps else None
    session.tick(next_presentation=next_pair)


def warmup(session, steps):
    for frame in range(steps):
        warmup_frame(session, frame, steps)


def smoke(config):
    """Hidden Tk, actual autonomous steps, complete save/load and continuation."""
    from run_closed_loop import train

    def same_tree(a, b, location="snapshot"):
        if isinstance(a, np.ndarray):
            assert isinstance(b, np.ndarray) and a.dtype == b.dtype and np.array_equal(a, b), location
        elif isinstance(a, dict):
            assert set(a) == set(b), location
            for key in a:
                same_tree(a[key], b[key], location+"/"+str(key))
        elif isinstance(a, (tuple, list)):
            assert len(a) == len(b), location
            for i, (x, y) in enumerate(zip(a, b)):
                same_tree(x, y, location+"/"+str(i))
        else:
            assert a == b, location

    source_paths=[HERE/name for name in ("brain.py","experience_environment.py","sensory_adapter.py")]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    living_hash=hashlib.sha256(LIVE_PATH.read_bytes()).hexdigest() if LIVE_PATH.exists() else None
    session=LifeSession(OriginalBrain(config))
    before=copy.deepcopy(session.environment.world.get_state())
    session.pair("red",220,.8)
    assert session.environment.world.time==before["time"]
    assert np.array_equal(session.environment.world.position,before["position"])
    for _ in range(3): session.tick()
    session.hide(); session.tone(777,.8)
    for _ in range(2): session.tick()
    assert session.brain.pending is None and session.control_steps==5
    with tempfile.TemporaryDirectory(prefix="original-viewer-smoke-") as directory:
        path=Path(directory)/"living_original.npz"
        session.save(path)
        restored=LifeSession.load(path)
        assert packet_digest(session.packet)==packet_digest(restored.packet)
        same_tree(session.brain.snapshot(), restored.brain.snapshot())
        same_tree(environment_snapshot(session.environment), environment_snapshot(restored.environment))
        for _ in range(3):
            session.tick(); restored.tick()
            assert np.array_equal(session.executed,restored.executed)
            assert packet_digest(session.packet)==packet_digest(restored.packet)
            assert np.array_equal(session.brain.thought,restored.brain.thought)
        brain_before_drawing=copy.deepcopy(restored.brain.snapshot())
        world_before_drawing=environment_snapshot(restored.environment)
        packet_before_drawing=packet_digest(restored.packet)
        catalog_before_drawing=copy.deepcopy(restored.catalog)
        root=tk.Tk(); root.withdraw()
        viewer=Viewer(root,restored,save_path=path)
        root.update_idletasks(); viewer.draw()
        assert not viewer.running and len(viewer.canvas.find_all())>31
        assert len(viewer.sensors.find_all())>=64
        assert not viewer.speaker_enabled.get()
        same_tree(restored.brain.snapshot(),brain_before_drawing)
        same_tree(environment_snapshot(restored.environment),world_before_drawing)
        same_tree(restored.catalog,catalog_before_drawing)
        assert packet_digest(restored.packet)==packet_before_drawing
        assert "实验记录注释（脑无颜色标签）" in viewer.stats.get()
        note_fixture={'10':{'visible_rays':{'red':[],'green':[],'blue':[3,4]},
                            'external_presentation':{'color':'red','frequency':220}}}
        assert memory_record_note(note_fixture,10)=="t=10 当时可见[蓝]"
        assert "无该时刻实验记录" in memory_record_note(note_fixture,11)
        assert memory_record_note(note_fixture,None)=="无回忆时间"
        viewer.close()

        # The optional GUI course must be exactly the formal two-round loop,
        # including the actual post-action scene at each presentation boundary.
        course=LifeSession(OriginalBrain(config))
        reference=LifeSession(OriginalBrain(config))
        formal=train(reference.brain, reference.environment, repeats=2)
        completed_path=Path(directory)/"completed_life.npz"
        root=tk.Tk(); root.withdraw(); viewer=Viewer(root, course,save_path=completed_path)
        errors=[]
        original_showerror=messagebox.showerror
        messagebox.showerror=lambda *args, **kwargs: errors.append(args)
        try:
            viewer.start_warmup(48, completed_path)
            deadline=time.perf_counter()+60
            while viewer.warming and time.perf_counter()<deadline:
                root.update()
            assert not viewer.warming and not viewer.running and not errors
            assert course.control_steps == 48 and completed_path.exists()
            same_tree(course.brain.snapshot(), reference.brain.snapshot())
            same_tree(environment_snapshot(course.environment), environment_snapshot(reference.environment))
            continued=LifeSession.load(completed_path)
            same_tree(course.brain.snapshot(), continued.brain.snapshot())
            same_tree(environment_snapshot(course.environment), environment_snapshot(continued.environment))
            for _ in range(3):
                course.tick(); continued.tick()
                same_tree(course.brain.snapshot(), continued.brain.snapshot())
                same_tree(environment_snapshot(course.environment), environment_snapshot(continued.environment))
            viewer.close()
            closed_life=LifeSession.load(completed_path)
            same_tree(course.brain.snapshot(), closed_life.brain.snapshot())
            same_tree(environment_snapshot(course.environment), environment_snapshot(closed_life.environment))

            broken=LifeSession(OriginalBrain(config))
            def fail(*args, **kwargs):
                raise RuntimeError("Injected warmup verification failure")
            broken.tick=fail
            failed_path=Path(directory)/"failed_life.npz"
            root=tk.Tk(); root.withdraw(); viewer=Viewer(root, broken,save_path=failed_path)
            viewer.start_warmup(48, failed_path)
            while viewer.warming:
                root.update()
            assert errors and not failed_path.exists() and "Injected" in viewer.status.get()
            viewer.close()
            assert not failed_path.exists()

            # An existing life must also survive failed new warmup and close.
            preserved_bytes=completed_path.read_bytes()
            root=tk.Tk(); root.withdraw(); viewer=Viewer(root, broken,save_path=completed_path)
            viewer.start_warmup(48)
            while viewer.warming:
                root.update()
            assert viewer.close()
            assert completed_path.read_bytes() == preserved_bytes

            root=tk.Tk(); root.withdraw(); viewer=Viewer(root, continued,save_path=completed_path)
            original_save=continued.save
            continued.save=fail
            assert viewer.close() is False
            assert not viewer.closed and not viewer.running and root.winfo_exists()
            assert completed_path.read_bytes() == preserved_bytes
            continued.save=original_save
            assert viewer.close()
        finally:
            messagebox.showerror=original_showerror
    assert all(hashlib.sha256(p.read_bytes()).hexdigest()==hashes[str(p)] for p in source_paths)
    assert (hashlib.sha256(LIVE_PATH.read_bytes()).hexdigest() if LIVE_PATH.exists() else None) == living_hash
    report=dict(passed=True, actual_first_session_steps=8, restored_continuation_steps=3,
        teacher_actions=0,reward=0,default_paused=True,hidden_tk_draw=True,
        full_life_save_load_exact=True, original_sources_unchanged=True,
        warmup_actual_actions=formal['frames'], warmup_exactly_matches_formal_training=True,
        warmup_save_then_three_online_steps_exact=True,
        warmup_failure_displayed_without_saving=True,
        normal_close_saves_latest_brain_and_world=True,
        failed_warmup_close_preserves_existing_life=True,
        failed_close_save_keeps_paused_window_open=True,
        smoke_never_changes_real_living_checkpoint=True,
        gui_draw_preserves_brain_world_packet_and_catalog=True,
        memory_annotation_uses_actual_visible_rays_not_presentation_label=True,
        config=asdict(config),source_hashes=hashes,
        viewer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    report_path=HERE / 'results' / 'viewer_smoke.json'
    report_path.parent.mkdir(parents=True,exist_ok=True)
    report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path)
    parser.add_argument("--checkpoint",type=Path)
    parser.add_argument("--save-path",type=Path,help="Continue saving the chosen life here; defaults to the loaded checkpoint")
    warmups=parser.add_mutually_exclusive_group()
    warmups.add_argument("--warmup",type=int,default=0,help="Optional actual co-exposure steps, then save a living snapshot and pause")
    warmups.add_argument("--warmup2",dest="warmup",action="store_const",const=48,help="Two real RGB co-exposure rounds (48 autonomous actions), save and pause")
    parser.add_argument("--smoke",action="store_true",help="Hidden disposable verification; never writes the living file")
    args=parser.parse_args()
    if args.warmup<0: parser.error("--warmup must be nonnegative")
    if args.smoke:
        smoke(read_config(args.config)); return
    if args.checkpoint:
        session=LifeSession.load(args.checkpoint)
    else:
        session=LifeSession(OriginalBrain(read_config(args.config)),origin="实验候选参数 · 初生脑 · 当前暂停")
    root=tk.Tk()
    viewer=Viewer(root,session,save_path=args.save_path or args.checkpoint or LIVE_PATH)
    if args.warmup:
        viewer.start_warmup(args.warmup)
    root.mainloop()


if __name__=="__main__":
    main()

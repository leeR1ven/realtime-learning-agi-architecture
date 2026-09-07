"""本地Tk界面；展示同一闭环，不在界面内代替神经元做行为决策。

运行：.venv/Scripts/pythonw.exe 闭环仿真/viewer.py
无窗口接口冒烟：python -X utf8 闭环仿真\viewer.py --smoke-test
"""
from __future__ import annotations

from collections import deque
from pathlib import Path
import argparse
import json
import wave
import logging
import math
import sys
import tempfile
import time
import traceback
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from brain import Brain, ACTION_NAMES, FREQUENCIES, tone, DT
from world import World

RESULTS = HERE / 'results'
LIVE_PATH = RESULTS / 'living_brain.npz'
DEMO_PATH = RESULTS / 'demo_brain.npz'
TONE_LABELS = tuple(f'音调{number} · {int(frequency)} Hz'
                    for number, frequency in zip('①②③④', FREQUENCIES))


def make_world():
    return World(walls=[(5.4, 1.6, 5.65, 4.2), (1.5, 1.5, 2.8, 1.7)], objects=[
        {'position': (2.0, 4.5), 'radius': .30, 'color': (1.0, .4, .12), 'name': '橙色标'},
        {'position': (6.8, 4.8), 'radius': .32, 'color': (.2, .85, .5), 'name': '绿色标'},
        {'position': (6.7, 1.0), 'radius': .28, 'color': (.22, .55, 1.0), 'name': '蓝色标'},
    ])


class SimulationSession:
    """每tick执行一次观测→脑/学习→肌肉→物理，口部PCM延迟一帧听回。"""
    def __init__(self, brain=None, world=None):
        self.brain = brain if brain is not None else Brain()
        self.world = world if world is not None else make_world()
        self.external_tone = None
        self.tone_frames = 0
        self.teaching_frames = 0
        self.teaching_action = 0
        self.teach_mouth = False
        self.delayed_voice = np.zeros(800)
        self.info = {}
        self.elapsed_ms = 0.
        self.trail = deque(maxlen=500)
        self.state = self.world.get_state()
        self.trail.append(tuple(self.state['position']))

    def play_tone(self, index, frames=10):
        if int(index) not in range(4):
            raise ValueError('音调编号需在0~3之间')
        self.external_tone = int(index)
        self.tone_frames = int(frames)
        self.teaching_frames = 0
        self.teach_mouth = False

    def teach(self, index, action, mouth=False, frames=20):
        self.play_tone(index, frames)
        if int(action) not in range(5):
            raise ValueError('动作编号需在0~4之间')
        self.teaching_action = int(action)
        self.teaching_frames = int(frames)
        self.teach_mouth = bool(mouth)

    def reset_scene(self):
        self.world = make_world()
        self.brain.reset_activity()
        self.delayed_voice[:] = 0
        self.external_tone = None
        self.tone_frames = self.teaching_frames = 0
        self.teach_mouth = False
        self.trail.clear()
        self.state = self.world.get_state()
        self.trail.append(tuple(self.state['position']))
        self.info = {}

    def tick(self):
        started = time.perf_counter()
        observation = self.world.observe()
        waveform = self.delayed_voice.copy()
        teaching = self.teaching_frames > 0
        if self.tone_frames > 0:
            waveform += tone(self.external_tone)
        stimulation = np.eye(5)[self.teaching_action] if teaching else None
        mouth_stimulation = (np.eye(4)[self.external_tone]
                             if teaching and self.teach_mouth else None)
        muscles, emitted, info = self.brain.step(
            observation, waveform, stimulation=stimulation,
            pleasant=1.0 if teaching else 0.0, mouth_stimulation=mouth_stimulation,
        )
        self.world.step(muscles, dt=DT)
        self.delayed_voice = np.asarray(emitted, dtype=float).copy()
        self.tone_frames = max(0, self.tone_frames - 1)
        self.teaching_frames = max(0, self.teaching_frames - 1)
        self.info = info
        self.state = self.world.get_state()
        self.trail.append(tuple(self.state['position']))
        self.elapsed_ms = (time.perf_counter() - started) * 1000
        return info


def load_starting_brain():
    for path in (LIVE_PATH, DEMO_PATH):
        if path.exists():
            try:
                return Brain.load(path), f'已加载 {path.name}，继续在线学习'
            except Exception:
                logging.exception('无法载入检查点 %s', path)
    return Brain(), '从出生连接开始，在线学习已开启'


def write_speaker_tones(directory):
    """预生成展示用0.1秒单声道WAV；不改变神经元收到的PCM。"""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for index in range(4):
        pcm = tone(index, amplitude=.22, samples=800)
        envelope = np.ones(800)
        envelope[:40] = np.linspace(0, 1, 40)
        envelope[-40:] = np.linspace(1, 0, 40)
        data = np.rint(pcm * envelope * 32767).astype('<i2').tobytes()
        path = directory / f'tone_{index + 1}.wav'
        with wave.open(str(path), 'wb') as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(8000)
            stream.writeframes(data)
        paths.append(path)
    return paths


class Viewer:
    def __init__(self, root, session, startup_message):
        import tkinter as tk
        from tkinter import ttk
        self.tk, self.ttk, self.root = tk, ttk, root
        self.session = session
        self.running = True
        self.closed = False
        self.after_id = None
        self.last_autosave = time.monotonic()
        self.last_error = ''
        self.performance = deque(maxlen=200)
        self.scene_frames = 0
        self.mapping = [1, 3, 4, 0]
        self.speaker_paths = []
        self.speaker_module = None
        self.root.title('关联神经元 · 实时闭环仿真')
        self.root.geometry('1220x860')
        self.root.minsize(980, 730)
        self.root.configure(bg='#eef2f6')
        self.root.protocol('WM_DELETE_WINDOW', self.close)
        style = ttk.Style(root)
        if 'clam' in style.theme_names():
            style.theme_use('clam')
        style.configure('.', font=('Microsoft YaHei UI', 10), background='#eef2f6', foreground='#1c2d41')
        style.configure('TButton', padding=(9, 4))
        style.configure('TLabelframe', padding=6)
        style.configure('TLabelframe.Label', font=('Microsoft YaHei UI', 10, 'bold'))
        style.configure('Accent.TButton', foreground='white', background='#245ccd')
        style.map('Accent.TButton', background=[('active', '#1746aa')])

        shell = ttk.Frame(root, padding=18)
        shell.pack(fill='both', expand=True)
        top = ttk.Frame(shell)
        top.pack(fill='x', pady=(0, 12))
        ttk.Label(top, text='关联神经元 / 感知、行动与在线学习', font=('Microsoft YaHei UI', 20, 'bold')).pack(side='left')
        self.run_badge = ttk.Label(top, text='● 运行中 · 每秒10帧 · 在线学习', foreground='#147d53')
        self.run_badge.pack(side='right', pady=8)
        ttk.Label(shell, text='二维推进身体（不是双足机器人） · 五束RGB/距离感官 · 声音信号关联（不是自然语言）',
                  foreground='#576980').pack(anchor='w', pady=(0, 12))
        content = ttk.Frame(shell)
        content.pack(fill='both', expand=True)
        content.columnconfigure(0, weight=1)
        content.columnconfigure(1, weight=0)
        content.rowconfigure(0, weight=1)
        left = ttk.Frame(content)
        left.grid(row=0, column=0, sticky='nsew', padx=(0, 14))
        right = ttk.Frame(content, width=335)
        right.grid(row=0, column=1, sticky='nsew')
        self.canvas = tk.Canvas(left, bg='#101d30', highlightthickness=0, width=790, height=510)
        self.canvas.pack(fill='both', expand=True)
        self.canvas.bind('<Configure>', lambda event: self.draw())
        self.sensor_canvas = tk.Canvas(left, bg='#fff', highlightthickness=0, height=62)
        self.sensor_canvas.pack(fill='x', pady=(8, 0))
        ttk.Label(left, text='线段是实际受体射线；灰色墙有碰撞，彩色色标只影响视觉。细线记录最近轨迹。',
                  foreground='#576980', wraplength=750).pack(anchor='w', pady=(5, 10))
        logs = ttk.LabelFrame(left, text='交互记录与口部输出')
        logs.pack(fill='x')
        self.log_box = tk.Text(logs, height=5, wrap='word', state='disabled', bg='white',
                               fg='#31445b', relief='flat', font=('Microsoft YaHei UI', 9))
        self.log_box.pack(fill='x')

        control = ttk.Frame(right)
        control.pack(fill='x', pady=(0, 10))
        self.pause_button = ttk.Button(control, text='暂停', command=self.toggle_pause)
        self.pause_button.pack(side='left', expand=True, fill='x', padx=(0, 5))
        ttk.Button(control, text='保存检查点', command=lambda: self.save('手动保存')).pack(side='left', expand=True, fill='x')
        ttk.Button(right, text='重新开始场景（保留已学权重）', command=self.reset_scene).pack(fill='x', pady=(0, 10))

        sounds = ttk.LabelFrame(right, text='向外界发声音 · 不施加教学动作')
        sounds.pack(fill='x', pady=(0, 10))
        for i in range(4):
            ttk.Button(sounds, text=TONE_LABELS[i], command=lambda index=i: self.play(index)).grid(
                row=i // 2, column=i % 2, padx=2, pady=3, sticky='ew')
        sounds.columnconfigure(0, weight=1)
        sounds.columnconfigure(1, weight=1)
        self.speaker_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(sounds, text='播放音调到扬声器（仅展示）', variable=self.speaker_var,
                        command=self.toggle_speaker).grid(row=2, column=0, columnspan=2, sticky='w', pady=(5, 0))

        teach = ttk.LabelFrame(right, text='现场教学 · 感知、行动、学习仍在同一循环')
        teach.pack(fill='x', pady=(0, 10))
        self.tone_var = tk.StringVar(value=TONE_LABELS[0])
        self.action_var = tk.StringVar(value=ACTION_NAMES[self.mapping[0]])
        ttk.Label(teach, text='声音').grid(row=0, column=0, sticky='w', pady=2)
        self.tone_combo = ttk.Combobox(teach, textvariable=self.tone_var, values=TONE_LABELS, state='readonly', width=21)
        self.tone_combo.grid(row=0, column=1, sticky='ew', padx=(8, 0))
        self.tone_combo.bind('<<ComboboxSelected>>', self.select_tone)
        ttk.Label(teach, text='配对动作').grid(row=1, column=0, sticky='w', pady=2)
        self.action_combo = ttk.Combobox(teach, textvariable=self.action_var, values=ACTION_NAMES, state='readonly', width=21)
        self.action_combo.grid(row=1, column=1, sticky='ew', padx=(8, 0))
        self.action_combo.bind('<<ComboboxSelected>>', self.select_action)
        self.mouth_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(teach, text='同时示范口部发出同一音调', variable=self.mouth_var).grid(
            row=2, column=0, columnspan=2, sticky='w', pady=(4, 3))
        ttk.Button(teach, text='配对教学 2 秒', style='Accent.TButton', command=self.start_teaching).grid(
            row=3, column=0, columnspan=2, sticky='ew', pady=2)
        ttk.Button(teach, text='撤去教学，只发声音', command=self.sound_only).grid(
            row=4, column=0, columnspan=2, sticky='ew', pady=2)
        ttk.Label(teach, text='教学含运动刺激与正强化；之后可仅发声音检验。',
                  foreground='#66758a', wraplength=310, font=('Microsoft YaHei UI', 9)).grid(
            row=5, column=0, columnspan=2, sticky='w', pady=(4, 0))
        teach.columnconfigure(1, weight=1)

        activity = ttk.LabelFrame(right, text='神经活动与物理反馈')
        activity.pack(fill='both', expand=True)
        self.phase_var = tk.StringVar(value='自主活动 · 无教学刺激')
        ttk.Label(activity, textvariable=self.phase_var, foreground='#245ccd', wraplength=315,
                  font=('Microsoft YaHei UI', 9)).pack(anchor='w', pady=(0, 2))
        self.stats_var = tk.StringVar()
        ttk.Label(activity, textvariable=self.stats_var, justify='left', wraplength=320,
                  font=('Microsoft YaHei UI', 9)).pack(anchor='w')
        self.status_var = tk.StringVar(value=startup_message)
        ttk.Label(shell, textvariable=self.status_var, foreground='#576980', wraplength=1140).pack(
            side='bottom', before=content, fill='x', pady=(8, 0))
        self.log(startup_message)
        self.log('每5分钟及退出时保存 living_brain.npz；demo基线保持原样。')
        self.update_stats()
        self.after_id = root.after(100, self.loop)

    def log(self, message):
        stamp = time.strftime('%H:%M:%S')
        self.log_box.configure(state='normal')
        self.log_box.insert('end', f'{stamp}  {message}\n')
        if int(self.log_box.index('end-1c').split('.')[0]) > 200:
            self.log_box.delete('1.0', '21.0')
        self.log_box.see('end')
        self.log_box.configure(state='disabled')
        logging.info(message)

    def selected_tone(self):
        return TONE_LABELS.index(self.tone_var.get())

    def select_tone(self, event=None):
        self.action_var.set(ACTION_NAMES[self.mapping[self.selected_tone()]])

    def select_action(self, event=None):
        self.mapping[self.selected_tone()] = ACTION_NAMES.index(self.action_var.get())

    def play(self, index):
        self.session.play_tone(index)
        self.play_speaker(index)
        self.log(f'外界播放{TONE_LABELS[index]}，1秒；教学刺激和正强化均已撤去。')
        self.update_stats()

    def sound_only(self):
        self.play(self.selected_tone())

    def start_teaching(self):
        index, action = self.selected_tone(), ACTION_NAMES.index(self.action_var.get())
        self.mapping[index] = action
        self.session.teach(index, action, self.mouth_var.get())
        self.play_speaker(index)
        suffix = '，同时示范同一口部音调' if self.mouth_var.get() else ''
        self.log(f'开始20帧配对：{TONE_LABELS[index]} → {ACTION_NAMES[action]}{suffix}。')
        self.update_stats()

    def toggle_speaker(self):
        if not self.speaker_var.get():
            self.log('扬声器展示已关闭；脑内PCM感知与口部反馈继续运行。')
            return
        try:
            import winsound
            self.speaker_module = winsound
            self.speaker_paths = write_speaker_tones(RESULTS / 'audio')
            self.log('扬声器展示已开启：按钮声音及实际口部输出播放短音，不采集麦克风。')
        except Exception as exc:
            self.speaker_var.set(False)
            logging.exception('扬声器展示准备失败')
            self.log(f'扬声器展示不可用：{exc}；在线学习继续。')

    def play_speaker(self, index):
        if not self.speaker_var.get() or self.speaker_module is None:
            return
        try:
            sound = self.speaker_module
            sound.PlaySound(str(self.speaker_paths[int(index)]),
                            sound.SND_FILENAME | sound.SND_ASYNC | sound.SND_NODEFAULT)
        except Exception as exc:
            self.speaker_var.set(False)
            logging.exception('异步播放音调失败')
            self.log(f'声音展示失败：{exc}；在线学习继续。')

    def toggle_pause(self):
        self.running = not self.running
        self.pause_button.configure(text='继续运行' if not self.running else '暂停')
        self.run_badge.configure(text='Ⅱ 已暂停' if not self.running else '● 运行中 · 每秒10帧 · 在线学习',
                                 foreground='#a06815' if not self.running else '#147d53')
        self.log('暂停物理与神经循环。' if not self.running else '恢复同一在线学习循环。')

    def reset_scene(self):
        self.session.reset_scene()
        self.scene_frames = 0
        self.log('场景已重置；保留已学连接，清除短时活动与待发声音。')
        self.draw()
        self.update_stats()

    def save(self, reason):
        try:
            RESULTS.mkdir(parents=True, exist_ok=True)
            self.session.brain.save(LIVE_PATH)
            self.status_var.set(f'{reason}完成 · {LIVE_PATH} · demo基线未改动')
            self.log(f'{reason}：living_brain.npz')
            return True
        except Exception as exc:
            logging.exception('%s失败', reason)
            self.status_var.set(f'{reason}失败：{exc}；仿真可继续，详情见 results/viewer.log')
            self.log(f'{reason}失败：{type(exc).__name__}: {exc}')
            return False

    def loop(self):
        started = time.perf_counter()
        if self.closed:
            return
        if self.running:
            try:
                teaching_before = self.session.teaching_frames
                info = self.session.tick()
                self.scene_frames += 1
                self.performance.append(self.session.elapsed_ms)
                if teaching_before and not self.session.teaching_frames:
                    self.log('配对教学结束，恢复自主活动。可点击“撤去教学，只发声音”检验。')
                if info.get('emitted'):
                    self.play_speaker(info['emitted'][0])
                    tones = '、'.join(f'音调{"①②③④"[i]}' for i in info['emitted'])
                    context = '教学刺激中的口部活动' if teaching_before and self.session.teach_mouth else '口部神经输出'
                    self.log(f'{context}：{tones}；PCM将在下一帧由它自己听到。')
                self.last_error = ''
                self.draw()
                self.update_stats()
            except Exception as exc:
                logging.exception('逐帧循环失败，保持界面可用')
                key = f'{type(exc).__name__}: {exc}'
                if key != self.last_error:
                    self.log(f'本帧失败：{key}；可暂停或重置场景。')
                    self.last_error = key
                self.status_var.set(f'本帧失败：{key}；详情见 results/viewer.log')
        if time.monotonic() - self.last_autosave >= 300:
            self.save('自动保存')
            self.last_autosave = time.monotonic()
        delay = max(1, int(DT * 1000 - (time.perf_counter() - started) * 1000))
        self.after_id = self.root.after(delay, self.loop)

    def update_stats(self):
        s, info = self.session, self.session.info
        if s.teaching_frames:
            phase = f'教学中 · {ACTION_NAMES[s.teaching_action]} · 剩{s.teaching_frames / 10:.1f}秒'
        elif s.tone_frames:
            phase = f'外界音调{"①②③④"[s.external_tone]} · 剩{s.tone_frames / 10:.1f}秒 · 无教学刺激'
        else:
            phase = '自主活动 · 无教学刺激'
        self.phase_var.set(phase)
        dangers = np.asarray(info.get('danger_weights', s.brain.danger_weights))
        muscles = np.asarray(info.get('muscles', [0, 0, 0, 0]))
        heard = [str(i + 1) for i, v in enumerate(info.get('heard', [])) if v > .15]
        p95 = np.percentile(self.performance, 95) if self.performance else 0
        self.stats_var.set(
            f'行动：{info.get("action_name", "等待首帧")}  听觉：{", ".join(heard) or "静音"}\n'
            f'场景时间：{s.state["time"]:.1f}秒   生涯帧：{s.brain.frame:,}\n'
            f'感官 / 念头：{info.get("sensory_active", 0)} / {info.get("thought_active", 0)}\n'
            f'时间记忆连接：{info.get("memory_edges", 0):,}\n'
            f'前额联想连接：{info.get("pfc_edges", 0):,}\n'
            f'回忆：{info.get("recall_events", 0)} 个时刻  疼痛：{s.state["pain"]:.2f}\n'
            f'危险（右→左）：{"  ".join(f"{x:.2f}" for x in dangers)}\n'
            f'肌肉 左前/后 右前/后：{"  ".join(f"{x:.1f}" for x in muscles)}\n'
            f'闭环耗时：{s.elapsed_ms:.1f}ms  P95 {p95:.1f}ms'
        )

    @staticmethod
    def color(rgb, minimum=0):
        values = np.clip(np.asarray(rgb) * 255, minimum, 255).astype(int)
        return '#' + ''.join(f'{c:02x}' for c in values)

    def draw(self):
        if not hasattr(self, 'canvas'):
            return
        c = self.canvas
        c.delete('all')
        s = self.session.state
        width, height = max(c.winfo_width(), 10), max(c.winfo_height(), 10)
        scale = min((width - 48) / s['width'], (height - 48) / s['height'])
        if scale <= 0:
            return
        ox, oy = (width - s['width'] * scale) / 2, (height + s['height'] * scale) / 2
        def xy(p):
            return ox + p[0] * scale, oy - p[1] * scale
        x0, y0 = xy((0, s['height']))
        x1, y1 = xy((s['width'], 0))
        c.create_rectangle(x0, y0, x1, y1, fill='#14253a', outline='#66839e', width=3)
        for x in range(1, int(s['width'])):
            c.create_line(*xy((x, 0)), *xy((x, s['height'])), fill='#1e344b')
        for y in range(1, int(s['height'])):
            c.create_line(*xy((0, y)), *xy((s['width'], y)), fill='#1e344b')
        trail = list(self.session.trail)
        if len(trail) > 1:
            c.create_line(*[v for p in trail for v in xy(p)], fill='#46799b', width=1.5)
        for wall in s['walls']:
            a, b, d, e = wall['bounds']
            c.create_rectangle(*xy((a, e)), *xy((d, b)), fill=self.color(wall['color'], 60), outline='#9aaabe')
        for obj in s['objects']:
            px, py = xy(obj['position']); r = obj['radius'] * scale
            c.create_oval(px - r, py - r, px + r, py + r, fill=self.color(obj['color']), outline='')
            c.create_text(px, py + r + 13, text=obj['name'], fill='#b9cbdf', font=('Microsoft YaHei UI', 9))
        px, py = xy(s['position'])
        for i, (end, rgb) in enumerate(zip(s['ray_endpoints'], s['ray_colors'])):
            ex, ey = xy(end)
            col = self.color(rgb, 72)
            c.create_line(px, py, ex, ey, fill=col, dash=(4, 3), width=2)
            c.create_oval(ex - 3, ey - 3, ex + 3, ey + 3, fill=col, outline='')
            c.create_text(ex + 7, ey - 8, text=str(i + 1), fill='#c3d4e7', font=('Microsoft YaHei UI', 8))
        r = max(9, s['radius'] * scale)
        c.create_oval(px - r, py - r, px + r, py + r, fill='#df786c' if s['pain'] > .01 else '#66c6ed', outline='#d8f1ff', width=2)
        angle = s['heading']
        direction = np.array([math.cos(angle), -math.sin(angle)])
        lateral = np.array([-direction[1], direction[0]])
        tip = np.array([px, py]) + 1.45 * r * direction
        rear = np.array([px, py]) + .1 * r * direction
        coords = [*tip, *(rear + .45 * r * lateral), *(rear - .45 * r * lateral)]
        c.create_polygon(*coords, fill='#f2fbff', outline='')
        c.create_text(x0 + 12, y0 + 14, text='8 × 6 m  /  俯视', anchor='w', fill='#a9bed4', font=('Microsoft YaHei UI', 9))
        self.draw_sensors()

    def draw_sensors(self):
        c, s = self.sensor_canvas, self.session.state
        c.delete('all')
        width = max(c.winfo_width(), 100)
        cell = width / 5
        for i, (rgb, distance) in enumerate(zip(s['ray_colors'], s['ray_distances'])):
            x = i * cell
            c.create_rectangle(x + 8, 9, x + 33, 34, fill=self.color(rgb), outline='#cbd4e0')
            c.create_text(x + 41, 17, anchor='w', text=f'受体 {i + 1}  {distance:.2f}m', fill='#2f455e', font=('Microsoft YaHei UI', 9))
            c.create_text(x + 10, 48, anchor='w', text='RGB ' + '/'.join(f'{v:.1f}' for v in rgb), fill='#6b7d91', font=('Microsoft YaHei UI', 8))

    def close(self):
        self.closed = True
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
        if self.speaker_module is not None:
            try:
                self.speaker_module.PlaySound(None, self.speaker_module.SND_PURGE)
            except Exception:
                logging.exception('停止展示声音失败，仍继续保存与退出')
        try:
            self.save('退出保存')
        finally:
            self.root.destroy()


def smoke_test():
    """不创建Tk窗口，不读写用户检查点，验证真实API及教学撤除。"""
    session = SimulationSession()
    initial_position = np.array(session.state['position'])
    for _ in range(5):
        session.tick()
    assert np.linalg.norm(np.array(session.state['position']) - initial_position) > .01
    session.teach(0, 3, mouth=True)
    for _ in range(20):
        info = session.tick()
        assert info['action'] == 3
    assert session.teaching_frames == session.tone_frames == 0
    assert session.brain.audio_weights[0, 3] > 0
    session.teach(1, 2)
    session.play_tone(0)
    assert session.teaching_frames == 0 and not session.teach_mouth
    for _ in range(10):
        session.tick()
    audio_before = session.brain.audio_weights.copy()
    frame_before = session.brain.frame
    session.reset_scene()
    np.testing.assert_array_equal(session.brain.audio_weights, audio_before)
    assert session.brain.frame == frame_before and session.state['time'] == 0
    with tempfile.TemporaryDirectory(prefix='agi_viewer_smoke_') as directory:
        path = Path(directory) / 'test_brain.npz'
        session.brain.save(path)
        recovered = Brain.load(path)
        for sound_path in write_speaker_tones(Path(directory) / 'audio'):
            with wave.open(str(sound_path), 'rb') as stream:
                assert (stream.getnchannels(), stream.getsampwidth(), stream.getframerate(), stream.getnframes()) == (1, 2, 8000, 800)
        np.testing.assert_array_equal(recovered.audio_weights, session.brain.audio_weights)
    print('PASS: autonomous movement, one-loop teaching, voice feedback, teaching withdrawal, scene reset, save/load and four valid WAV files')
    print('No Tk window was created; no existing model or checkpoint was modified.')


def layout_check():
    """只计算自身Tk控件几何，不显示窗口、不运行after、不保存生活模型。"""
    import tkinter as tk
    directory = Path(tempfile.mkdtemp(prefix='agi_viewer_layout_'))
    root = tk.Tk()
    root.withdraw()
    viewer = None
    try:
        viewer = Viewer(root, SimulationSession(), '布局检查：新生临时脑，不读取或保存生活检查点')
        root.geometry('1220x860')
        root.after_cancel(viewer.after_id)
        viewer.after_id = None
        root.attributes('-alpha', 0.0)
        if float(root.attributes('-alpha')) != 0.0:
            raise RuntimeError('Tk does not support a fully transparent layout window')
        root.deiconify()
        root.update()  # 透明映射后分配真实几何；逐帧after已取消。
        root.update_idletasks()
        width, height = root.winfo_width(), root.winfo_height()
        x0, y0 = root.winfo_rootx(), root.winfo_rooty()
        rows = []
        def inspect(widget):
            if widget is not root:
                try:
                    label = str(widget.cget('text'))
                except tk.TclError:
                    label = ''
                x, y = widget.winfo_rootx() - x0, widget.winfo_rooty() - y0
                w, h = widget.winfo_width(), widget.winfo_height()
                rows.append({'widget': str(widget), 'class': widget.winfo_class(), 'label': label,
                             'x': x, 'y': y, 'width': w, 'height': h,
                             'requested_width': widget.winfo_reqwidth(), 'requested_height': widget.winfo_reqheight(),
                             'inside_window': bool(x >= 0 and y >= 0 and x + w <= width and y + h <= height),
                             'content_fits': bool(w > 1 and h > 1 and (widget.winfo_class() not in
                                 ('TButton', 'TCheckbutton', 'TCombobox', 'TLabel') or
                                 (w >= widget.winfo_reqwidth() and h >= widget.winfo_reqheight())))})
            for child in widget.winfo_children():
                inspect(child)
        inspect(root)
        overflow = [row for row in rows if not row['inside_window'] or not row['content_fits']]
        report = {'window': [width, height], 'transparent_mapped': root.state() == 'normal' and float(root.attributes('-alpha')) == 0.0,
                  'widget_count': len(rows), 'overflow': overflow, 'widgets': rows,
                  'existing_models_loaded': False, 'living_checkpoint_saved': False}
        path = directory / 'layout_report.json'
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
        print(f'Layout report: {path}')
        print(f'Window {width}x{height}; transparent_mapped={report["transparent_mapped"]}; {len(rows)} widgets; overflow={len(overflow)}')
        for row in overflow:
            print('OVERFLOW:', row['class'], row['label'], row['x'], row['y'], row['width'], row['height'])
        assert (width, height) == (1220, 860), 'unexpected test-window size'
        assert report['transparent_mapped'], 'layout-check must never show visible content'
        assert not overflow, 'some controls extend outside the window'
        print('PASS: all application controls inside window; no screenshot, OS input, live loop, or model save')
    finally:
        if viewer is not None:
            viewer.closed = True
            if viewer.after_id is not None:
                root.after_cancel(viewer.after_id)
        root.withdraw()
        root.destroy()


def main():
    parser = argparse.ArgumentParser(description='Tkinter local embodied neural simulation')
    parser.add_argument('--smoke-test', action='store_true')
    parser.add_argument('--layout-check', action='store_true')
    args = parser.parse_args()
    if args.smoke_test:
        smoke_test()
        return
    if args.layout_check:
        layout_check()
        return
    RESULTS.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=RESULTS / 'viewer.log', level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s', encoding='utf8')
    import tkinter as tk
    root = tk.Tk()
    brain, message = load_starting_brain()
    Viewer(root, SimulationSession(brain), message)
    root.mainloop()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        RESULTS.mkdir(parents=True, exist_ok=True)
        text = traceback.format_exc()
        with (RESULTS / 'viewer.log').open('a', encoding='utf8') as stream:
            stream.write(text + '\n')
        if '--smoke-test' in sys.argv or '--layout-check' in sys.argv:
            raise
        try:
            import tkinter.messagebox as messagebox
            messagebox.showerror('仿真启动失败', f'错误已写入 {RESULTS / "viewer.log"}\n\n{text[-1700:]}')
        except Exception:
            pass


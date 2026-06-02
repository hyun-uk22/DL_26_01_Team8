import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import queue
import time
import cv2
import numpy as np
import tempfile
from pathlib import Path
from PIL import Image, ImageTk

from app.video_loader import VideoLoader
from app.pose_detector import PoseDetector
from app.similarity import SimilarityCalculator
from app.visualizer import Visualizer


class RangeSlider(tk.Canvas):
    """시작/끝 값을 한 줄에서 조정하는 간단한 range slider."""

    def __init__(
        self,
        parent,
        start_var: tk.IntVar,
        end_var: tk.IntVar,
        command=None,
        bg: str = "#111827",
        track_bg: str = "#0f172a",
        active_bg: str = "#0ea5e9",
        thumb_bg: str = "#e5e7eb",
        **kwargs,
    ):
        super().__init__(
            parent,
            height=34,
            bg=bg,
            highlightthickness=0,
            bd=0,
            **kwargs,
        )
        self.start_var = start_var
        self.end_var = end_var
        self.command = command
        self.track_bg = track_bg
        self.active_bg = active_bg
        self.thumb_bg = thumb_bg
        self.min_value = 0
        self.max_value = 100
        self.dragging: str | None = None
        self.pad = 14
        self.thumb_radius = 7

        self.bind("<Configure>", lambda _event: self._draw())
        self.bind("<ButtonPress-1>", self._on_press)
        self.bind("<B1-Motion>", self._on_drag)
        self.bind("<ButtonRelease-1>", self._on_release)
        self.bind("<MouseWheel>", self._on_mousewheel)
        self.bind("<Button-4>", lambda event: self._nudge_nearest_thumb(event.x, 1))
        self.bind("<Button-5>", lambda event: self._nudge_nearest_thumb(event.x, -1))

    def set_bounds(self, min_value: int, max_value: int):
        self.min_value = min_value
        self.max_value = max(min_value + 1, max_value)
        self.start_var.set(max(self.min_value, min(self.start_var.get(), self.max_value - 1)))
        self.end_var.set(max(self.start_var.get() + 1, min(self.end_var.get(), self.max_value)))
        self._draw()

    def _on_press(self, event):
        sx = self._value_to_x(self.start_var.get())
        ex = self._value_to_x(self.end_var.get())
        self.dragging = "start" if abs(event.x - sx) <= abs(event.x - ex) else "end"
        self._update_from_x(event.x)

    def _on_drag(self, event):
        self._update_from_x(event.x)

    def _on_release(self, _event):
        self.dragging = None

    def _on_mousewheel(self, event):
        step = 1 if event.delta > 0 else -1
        self._nudge_nearest_thumb(event.x, step)

    def _nudge_nearest_thumb(self, x: int, step: int):
        sx = self._value_to_x(self.start_var.get())
        ex = self._value_to_x(self.end_var.get())
        target = self.dragging or ("start" if abs(x - sx) <= abs(x - ex) else "end")
        if target == "start":
            value = max(self.min_value, min(self.start_var.get() + step, self.end_var.get() - 1))
            self.start_var.set(value)
        else:
            value = min(self.max_value, max(self.end_var.get() + step, self.start_var.get() + 1))
            self.end_var.set(value)
        self._draw()
        if self.command is not None:
            self.command()

    def _update_from_x(self, x: int):
        value = self._x_to_value(x)
        if self.dragging == "start":
            value = min(value, self.end_var.get() - 1)
            self.start_var.set(max(self.min_value, value))
        elif self.dragging == "end":
            value = max(value, self.start_var.get() + 1)
            self.end_var.set(min(self.max_value, value))
        self._draw()
        if self.command is not None:
            self.command()

    def _value_to_x(self, value: int) -> int:
        width = max(1, self.winfo_width() - self.pad * 2)
        ratio = (value - self.min_value) / (self.max_value - self.min_value)
        return int(self.pad + ratio * width)

    def _x_to_value(self, x: int) -> int:
        width = max(1, self.winfo_width() - self.pad * 2)
        ratio = (x - self.pad) / width
        ratio = max(0.0, min(1.0, ratio))
        return int(round(self.min_value + ratio * (self.max_value - self.min_value)))

    def _draw(self):
        self.delete("all")
        width = self.winfo_width()
        y = self.winfo_height() // 2
        if width <= 1:
            return

        sx = self._value_to_x(self.start_var.get())
        ex = self._value_to_x(self.end_var.get())
        self.create_line(self.pad, y, width - self.pad, y, fill=self.track_bg, width=6)
        self.create_line(sx, y, ex, y, fill=self.active_bg, width=6)
        for x in (sx, ex):
            self.create_oval(
                x - self.thumb_radius,
                y - self.thumb_radius,
                x + self.thumb_radius,
                y + self.thumb_radius,
                fill=self.thumb_bg,
                outline=self.active_bg,
                width=2,
            )


class SeekBar(tk.Canvas):
    """YouTube처럼 클릭한 위치로 이동하는 단일 progress bar."""

    def __init__(
        self,
        parent,
        command=None,
        bg: str = "#111827",
        track_bg: str = "#64748b",
        active_bg: str = "#0ea5e9",
        thumb_bg: str = "#f8fafc",
        **kwargs,
    ):
        super().__init__(
            parent,
            height=18,
            bg=bg,
            highlightthickness=0,
            bd=0,
            cursor="hand2",
            **kwargs,
        )
        self.command = command
        self.track_bg = track_bg
        self.active_bg = active_bg
        self.thumb_bg = thumb_bg
        self.max_value = 1
        self.value = 0
        self.pad = 6

        self.bind("<Configure>", lambda _event: self._draw())
        self.bind("<Button-1>", self._on_seek)
        self.bind("<B1-Motion>", self._on_seek)

    def set_max(self, max_value: int):
        self.max_value = max(1, max_value)
        self.value = max(0, min(self.value, self.max_value - 1))
        self._draw()

    def set_value(self, value: int):
        self.value = max(0, min(value, self.max_value - 1))
        self._draw()

    def _on_seek(self, event):
        width = max(1, self.winfo_width() - self.pad * 2)
        ratio = max(0.0, min(1.0, (event.x - self.pad) / width))
        value = int(round(ratio * (self.max_value - 1)))
        self.set_value(value)
        if self.command is not None:
            self.command(value)

    def _draw(self):
        self.delete("all")
        width = self.winfo_width()
        height = self.winfo_height()
        if width <= 1:
            return

        y = height // 2
        ratio = self.value / max(self.max_value - 1, 1)
        x = int(self.pad + ratio * (width - self.pad * 2))
        self.create_line(self.pad, y, width - self.pad, y, fill=self.track_bg, width=4)
        self.create_line(self.pad, y, x, y, fill=self.active_bg, width=4)
        self.create_oval(x - 5, y - 5, x + 5, y + 5, fill=self.thumb_bg, outline=self.active_bg, width=2)


class PoseCoachApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Pose Coach - 홈트레이닝 자세 교정")
        self.root.configure(bg="#0b1120")
        self.root.geometry("1280x760")
        self.root.resizable(True, True)

        # 핵심 모듈
        self.loader = VideoLoader()
        self.user_loader = VideoLoader()
        self.detector = PoseDetector()          # 첫 실행 시 모델 자동 다운로드
        self.similarity_calc = SimilarityCalculator()
        self.visualizer = Visualizer()

        # 상태 변수
        self.ref_frames: list[np.ndarray] = []
        self.ref_poses: list[dict] = []         # 전처리된 레퍼런스 포즈
        self.current_ref_idx = 0
        self.is_running = False
        self.is_recording = False
        self.is_playing_analysis = False
        self.is_playing_reference_preview = False
        self.playback_target = "reference"
        self.recorded_video_path: str | None = None
        self.analysis_playback_frames: list[np.ndarray] = []
        self.analysis_playhead_idx = 0
        self.selection_playhead_frame = 0
        self.crop_start: int | None = None
        self.crop_end: int | None = None
        self.user_crop_start_var = tk.IntVar(value=0)
        self.user_crop_end_var = tk.IntVar(value=0)
        self.crop_roi: tuple[float, float, float, float] | None = None
        self.crop_roi_history: list[tuple[float, float, float, float] | None] = []
        self.drag_start: tuple[int, int] | None = None
        self.preview_frame: np.ndarray | None = None
        self.display_origin = (0, 0)
        self.display_size = (0, 0)
        self.video_area_size = (960, 540)
        self.ref_sample_step = 2  # 빠른 동작(점프 등) 감지를 위해 샘플링 증가
        self.ref_playback_fps = 6.0
        self.last_ref_advance = 0.0
        self.preview_after_id: str | None = None
        self.user_preview_after_id: str | None = None
        self.preview_debounce_ms = 50
        self.selection_playback_generation = 0
        self.video_controls_visible = True
        self.video_controls_y = -2
        self.video_controls_hide_after_id: str | None = None
        self.video_controls_animation_after_id: str | None = None
        self.video_controls_hide_delay_ms = 1800

        self._build_ui()

    # ── UI 구성 ────────────────────────────────────────────

    def _disable_buttons_recursive(self, widget):
        """모든 버튼을 재귀적으로 비활성화"""
        if isinstance(widget, (ttk.Button, tk.Button)):
            widget.config(state=tk.DISABLED)
        for child in widget.winfo_children():
            self._disable_buttons_recursive(child)

    def _enable_buttons_recursive(self, widget):
        """모든 버튼을 재귀적으로 활성화 (특정 버튼은 상태에 따라 제외)"""
        if isinstance(widget, (ttk.Button, tk.Button)):
            # 실행 중 상태에 따라 일부 버튼만 활성화
            if widget == self.btn_stop and not self.is_running:
                return
            if widget == self.btn_record_stop and not self.is_recording:
                return
            if widget == self.btn_record_analyze and self.user_loader.container is None:
                return
            widget.config(state=tk.NORMAL)
        for child in widget.winfo_children():
            self._enable_buttons_recursive(child)

    def _build_ui(self):
        bg = "#0b1120"
        panel_bg = "#111827"
        section_bg = "#172033"
        text = "#e5e7eb"
        muted = "#94a3b8"
        accent = "#38bdf8"
        border = "#243247"
        video_controls_bg = "#020617"
        video_controls_active_bg = "#0f172a"

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TLabel", background=panel_bg, foreground=text, font=("Malgun Gothic", 10))
        style.configure(
            "TButton",
            background="#1f2937",
            foreground=text,
            borderwidth=0,
            focusthickness=0,
            relief="flat",
            padding=(10, 7),
            font=("Malgun Gothic", 10),
        )
        style.map(
            "TButton",
            background=[("disabled", "#1f2937"), ("active", "#334155")],
            foreground=[("disabled", "#64748b")],
        )
        style.configure(
            "Primary.TButton",
            background="#0ea5e9",
            foreground="#f8fafc",
            font=("Malgun Gothic", 10, "bold"),
        )
        style.map("Primary.TButton", background=[("active", "#0284c7"), ("disabled", "#1f2937")])
        style.configure("Danger.TButton", background="#7f1d1d", foreground="#fee2e2")
        style.map("Danger.TButton", background=[("active", "#991b1b"), ("disabled", "#1f2937")])
        style.configure(
            "TEntry",
            fieldbackground="#0f172a",
            foreground=text,
            insertbackground=text,
            bordercolor=border,
            lightcolor=border,
            darkcolor=border,
            padding=(8, 6),
            font=("Malgun Gothic", 10),
        )
        style.configure(
            "Horizontal.TScale",
            background=section_bg,
            troughcolor="#0f172a",
            bordercolor=section_bg,
            lightcolor=section_bg,
            darkcolor=section_bg,
        )

        self.root.configure(bg=bg)

        def section(parent, title: str) -> tk.Frame:
            wrap = tk.Frame(parent, bg=section_bg, highlightbackground=border, highlightthickness=1)
            wrap.pack(fill=tk.X, padx=14, pady=(0, 12))
            tk.Label(
                wrap,
                text=title,
                bg=section_bg,
                fg=accent,
                anchor="w",
                font=("Malgun Gothic", 11, "bold"),
            ).pack(fill=tk.X, padx=14, pady=(12, 8))
            return wrap

        def body_label(parent, label: str):
            tk.Label(
                parent,
                text=label,
                bg=section_bg,
                fg=muted,
                anchor="w",
                font=("Malgun Gothic", 9),
            ).pack(fill=tk.X, padx=14, pady=(2, 3))

        # ── 좌측 패널 (설정) ──
        left = tk.Frame(self.root, bg=panel_bg, width=320)
        left.pack(side=tk.LEFT, fill=tk.Y)
        left.pack_propagate(False)

        tk.Label(
            left,
            text="Pose Coach",
            bg=panel_bg,
            fg="#f8fafc",
            anchor="w",
            font=("Malgun Gothic", 20, "bold"),
        ).pack(fill=tk.X, padx=18, pady=(18, 0))
        tk.Label(
            left,
            text="Keypoint 기반 홈트레이닝 자세 분석",
            bg=panel_bg,
            fg=muted,
            anchor="w",
            font=("Malgun Gothic", 9),
        ).pack(fill=tk.X, padx=18, pady=(2, 18))

        # 파일 / URL 입력
        ref_section = section(left, "레퍼런스 영상")
        body_label(ref_section, "동영상 파일 또는 유튜브 링크")
        url_frame = tk.Frame(ref_section, bg=section_bg)
        url_frame.pack(fill=tk.X, padx=14, pady=(0, 8))
        self.url_var = tk.StringVar()
        ttk.Entry(url_frame, textvariable=self.url_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(url_frame, text="로드", command=self._load_video, style="Primary.TButton").pack(
            side=tk.LEFT, padx=(8, 0)
        )

        ttk.Button(ref_section, text="파일 선택", command=self._browse_file).pack(
            fill=tk.X, padx=14, pady=(0, 14)
        )

        self.crop_start_var = tk.IntVar(value=0)
        self.crop_end_var = tk.IntVar(value=0)

        # 실행 제어
        live_section = section(left, "실시간 분석")

        self.btn_start = ttk.Button(live_section, text="분석 시작", command=self._start_live,
                                    style="Primary.TButton")
        self.btn_start.pack(pady=(0, 8), fill=tk.X, padx=14)
        self.btn_stop = ttk.Button(live_section, text="중지", command=self._stop_live,
                                   state=tk.DISABLED, style="Danger.TButton")
        self.btn_stop.pack(pady=(0, 14), fill=tk.X, padx=14)

        record_section = section(left, "녹화 분석")
        self.btn_record_start = ttk.Button(
            record_section,
            text="카메라 녹화 시작",
            command=self._start_recording,
            style="Primary.TButton",
        )
        self.btn_record_start.pack(pady=(0, 8), fill=tk.X, padx=14)
        self.btn_record_stop = ttk.Button(
            record_section,
            text="녹화 종료",
            command=self._stop_recording,
            state=tk.DISABLED,
            style="Danger.TButton",
        )
        self.btn_record_stop.pack(pady=(0, 8), fill=tk.X, padx=14)
        self.btn_record_analyze = ttk.Button(
            record_section,
            text="녹화 구간 분석",
            command=self._analyze_recorded_video,
            state=tk.DISABLED,
            style="Primary.TButton",
        )
        self.btn_record_analyze.pack(pady=(0, 14), fill=tk.X, padx=14)

        # 유사도 표시
        score_section = section(left, "유사도")
        self.sim_label = tk.Label(score_section, text="--.--%", bg=section_bg,
                                  fg="#4ade80", font=("Malgun Gothic", 24, "bold"),
                                  anchor="center", width=8)
        self.sim_label.pack(fill=tk.X, padx=10, pady=(0, 2))
        self.status_label = tk.Label(score_section, text="대기 중", bg=section_bg,
                                     fg=muted, font=("Malgun Gothic", 10))
        self.status_label.pack(fill=tk.X, padx=14, pady=(0, 14))

        # ── 우측 영상 캔버스 ──
        right = tk.Frame(self.root, bg=bg)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=14, pady=14)

        tk.Label(
            right,
            text="Live Comparison",
            bg=bg,
            fg="#f8fafc",
            anchor="w",
            font=("Malgun Gothic", 15, "bold"),
        ).pack(fill=tk.X, pady=(0, 8))

        self.video_area = tk.Frame(
            right,
            bg="#020617",
            highlightbackground=border,
            highlightthickness=1,
        )
        self.video_area.pack(fill=tk.BOTH, expand=True)
        self.video_area.pack_propagate(False)
        self.video_area.bind("<Configure>", self._on_video_area_resize)
        self.video_area.bind("<Motion>", self._on_video_area_motion)

        self.canvas = tk.Label(
            self.video_area,
            bg="#020617",
            cursor="crosshair",
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Motion>", self._on_video_area_motion)
        self.canvas.bind("<ButtonPress-1>", self._on_roi_start)
        self.canvas.bind("<B1-Motion>", self._on_roi_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_roi_end)

        self.analysis_controls = tk.Frame(self.video_area, bg=video_controls_bg)
        self.analysis_controls.place(relx=0.0, rely=1.0, relwidth=1.0, y=-2, anchor="sw")
        self.btn_video_play = tk.Button(
            self.analysis_controls,
            text="▶",
            command=self._play_video_from_controls,
            state=tk.NORMAL,
            bg=video_controls_bg,
            fg="#f8fafc",
            activebackground=video_controls_active_bg,
            activeforeground="#f8fafc",
            disabledforeground="#64748b",
            bd=0,
            padx=10,
            pady=4,
            font=("Malgun Gothic", 13, "bold"),
            cursor="hand2",
        )
        self.btn_video_play.pack(side=tk.LEFT, padx=(12, 4), pady=(4, 8))
        self.analysis_seek_bar = SeekBar(
            self.analysis_controls,
            command=self._seek_playback,
            bg=video_controls_bg,
            track_bg="#64748b",
            active_bg="#0ea5e9",
            thumb_bg="#f8fafc",
        )
        self.analysis_seek_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 10), pady=(4, 8))
        self.analysis_time_label = tk.Label(
            self.analysis_controls,
            text="0:00.00 / 0:00.00",
            bg=video_controls_bg,
            fg="#e5e7eb",
            font=("Malgun Gothic", 9),
            width=18,
            anchor="e",
        )
        self.analysis_time_label.pack(side=tk.LEFT, padx=(6, 12), pady=(4, 8))
        self._bind_video_controls_motion()
        self.root.after(self.video_controls_hide_delay_ms, self._hide_video_controls)

        timeline = tk.Frame(
            right,
            bg="#111827",
            highlightbackground=border,
            highlightthickness=1,
        )
        timeline.pack(fill=tk.X, pady=(8, 0))

        tk.Label(timeline, text="레퍼런스", bg="#111827", fg=muted,
                 font=("Malgun Gothic", 9)).pack(side=tk.LEFT, padx=(12, 8), pady=10)
        self.range_slider = RangeSlider(
            timeline,
            self.crop_start_var,
            self.crop_end_var,
            command=self._on_slider_change,
            bg="#111827",
            track_bg="#0f172a",
            active_bg="#0ea5e9",
            thumb_bg="#e5e7eb",
        )
        self.range_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, pady=10)

        ttk.Button(
            timeline,
            text="포즈 추출",
            command=self._extract_ref_poses,
            style="Primary.TButton",
            width=9,
        ).pack(side=tk.LEFT, padx=(0, 4), pady=8)
        ttk.Button(timeline, text="영역 초기화", command=self._reset_roi, width=10).pack(
            side=tk.LEFT, padx=(0, 4), pady=8
        )
        self.btn_roi_undo = ttk.Button(timeline, text="↶ 되돌리기", command=self._undo_roi, width=10, state=tk.DISABLED)
        self.btn_roi_undo.pack(side=tk.LEFT, padx=(0, 8), pady=8)

        user_timeline = tk.Frame(
            right,
            bg="#111827",
            highlightbackground=border,
            highlightthickness=1,
        )
        user_timeline.pack(fill=tk.X, pady=(8, 0))

        tk.Label(user_timeline, text="녹화본", bg="#111827", fg=muted,
                 font=("Malgun Gothic", 9)).pack(side=tk.LEFT, padx=(12, 8), pady=10)
        self.user_range_slider = RangeSlider(
            user_timeline,
            self.user_crop_start_var,
            self.user_crop_end_var,
            command=self._on_user_slider_change,
            bg="#111827",
            track_bg="#0f172a",
            active_bg="#22c55e",
            thumb_bg="#e5e7eb",
        )
        self.user_range_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, pady=10)

        self.user_crop_label = tk.Label(
            user_timeline,
            text="구간: -",
            bg="#111827",
            fg=text,
            font=("Malgun Gothic", 10, "bold"),
            width=24,
        )
        self.user_crop_label.pack(side=tk.LEFT, padx=10, pady=10)
        ttk.Button(
            user_timeline,
            text="녹화 미리보기",
            command=self._show_user_selected_start,
        ).pack(side=tk.LEFT, padx=(10, 12), pady=8)

        self.info_bar = tk.Label(right, text="영상 또는 유튜브 URL을 입력하세요.",
                                 bg="#111827", fg=muted, anchor="w",
                                 padx=12, pady=8, font=("Malgun Gothic", 9))
        self.info_bar.pack(fill=tk.X, pady=(8, 0))

        # 카메라 프레임 큐
        self.frame_queue: queue.Queue = queue.Queue(maxsize=2)

    # ── 이벤트 핸들러 ───────────────────────────────────────

    def _browse_file(self):
        path = filedialog.askopenfilename(
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv"), ("All", "*.*")]
        )
        if path:
            self.url_var.set(path)
            self._load_video()

    def _load_video(self):
        src = self.url_var.get().strip()
        if not src:
            messagebox.showwarning("입력 오류", "URL 또는 파일 경로를 입력하세요.")
            return

        self.info_bar.config(text="영상 로딩 중...")
        self.root.update()

        if VideoLoader.is_youtube_url(src):
            ok = self.loader.load_youtube(src)
        else:
            ok = self.loader.load_local(src)

        if not ok:
            messagebox.showerror("로드 실패", "영상을 불러올 수 없습니다.")
            self.info_bar.config(text="로드 실패")
            return

        total_frames = self._loader_total_frames(self.loader)
        self.playback_target = "reference"
        self.crop_start_var.set(0)
        self.crop_end_var.set(total_frames)
        self.range_slider.set_bounds(0, total_frames)
        self._on_slider_change(update_preview=False)
        self.info_bar.config(
            text=f"로드 완료 | {self.loader.width}×{self.loader.height} | "
                 f"{self._format_seconds(self.loader.duration)} | {self.loader.fps:.1f}fps"
        )
        # 첫 프레임 미리보기
        frame = self.loader.get_frame_at(0)
        if frame is not None:
            self._show_preview_frame(frame)

    def _on_slider_change(self, *_, update_preview: bool = True):
        was_playing_reference = self.is_playing_reference_preview and self.playback_target == "reference"
        self.playback_target = "reference"
        s = self.crop_start_var.get()
        e = self.crop_end_var.get()
        if e <= s:
            self.crop_end_var.set(s + 1)
            e = s + 1
        if was_playing_reference:
            self._restart_reference_preview_from_range(s, e)
            return
        if update_preview and self.analysis_playback_frames:
            self.analysis_playback_frames = []
            self._reset_analysis_playback()
            self.btn_video_play.config(text="▶", state=tk.NORMAL)
        if self.playback_target == "reference" and not self.analysis_playback_frames:
            self.selection_playhead_frame = s
            self._sync_selection_seek_ui()
        if update_preview:
            self._schedule_preview_update(s)

    def _restart_reference_preview_from_range(self, start_frame: int, end_frame: int):
        if self.loader.container is None or end_frame <= start_frame:
            return

        self.selection_playback_generation += 1
        generation = self.selection_playback_generation
        self.selection_playhead_frame = start_frame
        self._sync_selection_seek_ui()
        self._clear_frame_queue()
        self.btn_video_play.config(text="■", state=tk.NORMAL)
        self.info_bar.config(
            text=f"레퍼런스 선택 구간 재생 중 | "
                 f"{self._format_frame_time(self.loader, start_frame)} ~ {self._format_frame_time(self.loader, end_frame)}"
        )
        threading.Thread(
            target=self._play_selection_video_worker,
            args=(self.loader, start_frame, end_frame, True, generation),
            daemon=True,
        ).start()

    def _schedule_preview_update(self, frame_idx: int):
        if self.preview_after_id is not None:
            self.root.after_cancel(self.preview_after_id)
        self.preview_after_id = self.root.after(
            self.preview_debounce_ms,
            self._update_preview_at_second,
            frame_idx,
        )

    def _update_preview_at_second(self, frame_idx: int):
        self.preview_after_id = None
        if (
            self.loader.container is None
            or self.is_running
            or self.is_recording
            or self.is_playing_analysis
            or self.is_playing_reference_preview
        ):
            return
        frame = self.loader.get_frame_at(frame_idx)
        if frame is not None:
            self._show_preview_frame(frame)

    def _play_selection_video(self, target: str):
        if target == "user":
            loader = self.user_loader
            start_var = self.user_crop_start_var
            end_var = self.user_crop_end_var
            label = "녹화본"
            crop = False
        else:
            loader = self.loader
            start_var = self.crop_start_var
            end_var = self.crop_end_var
            label = "레퍼런스"
            crop = True

        if loader.container is None:
            messagebox.showwarning(f"{label} 없음", f"먼저 {label} 영상을 준비하세요.")
            return
        if self.is_running or self.is_recording or self.is_playing_analysis:
            messagebox.showwarning("실행 중", "현재 실행 중인 분석 또는 녹화를 중지하세요.")
            return
        if self.is_playing_reference_preview:
            self.is_playing_reference_preview = False
            self.selection_playback_generation += 1
            self.btn_video_play.config(text="▶")
            self.info_bar.config(text=f"{label} 구간 재생 중지")
            return

        selection_start = start_var.get()
        end_frame = end_var.get()
        if end_frame <= selection_start:
            messagebox.showwarning("구간 오류", "끝 지점은 시작 지점보다 커야 합니다.")
            return

        start_frame = self.selection_playhead_frame
        if start_frame < selection_start or start_frame >= end_frame - 1:
            start_frame = selection_start
        self.selection_playhead_frame = start_frame
        self._sync_selection_seek_ui()

        self.is_playing_reference_preview = True
        self.selection_playback_generation += 1
        generation = self.selection_playback_generation
        self._clear_frame_queue()
        self.btn_video_play.config(text="■")
        self.info_bar.config(
            text=f"{label} 선택 구간 재생 중 | "
                 f"{self._format_frame_time(loader, start_frame)} ~ {self._format_frame_time(loader, end_frame)}"
        )
        threading.Thread(
            target=self._play_selection_video_worker,
            args=(loader, start_frame, end_frame, crop, generation),
            daemon=True,
        ).start()
        self._update_canvas()

    def _play_selection_video_worker(
        self,
        loader: VideoLoader,
        start_frame: int,
        end_frame: int,
        crop: bool,
        generation: int,
    ):
        frame_interval = 1.0 / max(loader.fps, 1.0)

        for offset, frame in enumerate(loader.get_frame_range(start_frame, end_frame)):
            if not self.is_playing_reference_preview or generation != self.selection_playback_generation:
                break
            frame_idx = start_frame + offset
            display = self._crop_frame_to_roi(frame) if crop else frame
            self.selection_playhead_frame = frame_idx
            self.root.after(0, self._sync_selection_seek_ui)
            self._enqueue_frame(display)
            time.sleep(frame_interval)

        if generation == self.selection_playback_generation:
            self.root.after(0, self._finish_reference_preview)

    def _finish_reference_preview(self):
        self.is_playing_reference_preview = False
        self.btn_video_play.config(text="▶", state=tk.NORMAL)
        self.info_bar.config(text="선택 구간 재생 완료")

    def _extract_ref_poses(self):
        """크롭된 구간에서 레퍼런스 포즈를 추출합니다."""
        if self.loader.container is None:
            messagebox.showwarning("오류", "먼저 영상을 로드하세요.")
            return
        if self.is_recording:
            messagebox.showwarning("녹화 중", "녹화를 종료한 뒤 레퍼런스 포즈를 추출하세요.")
            return
        if self.is_playing_reference_preview:
            self.is_playing_reference_preview = False
            self.selection_playback_generation += 1
            self.btn_video_play.config(text="▶", state=tk.NORMAL)
            self._clear_frame_queue()
        if not self._validate_reference_roi_for_pose_extraction():
            return

        start_f = self.crop_start_var.get()
        end_f = self.crop_end_var.get()

        self.ref_frames.clear()
        self.ref_poses.clear()

        # 버튼 비활성화로 중복 실행 방지
        for widget in self.root.winfo_children():
            self._disable_buttons_recursive(widget)

        self.detector.reset_filter("reference")
        self.info_bar.config(text="포즈 추출 중... (잠시 기다려 주세요)")
        self.root.update()

        threading.Thread(
            target=self._extract_ref_poses_worker,
            args=(start_f, end_f),
            daemon=True,
        ).start()

    def _extract_ref_poses_worker(self, start_f: int, end_f: int):
        ref_frames = []
        ref_poses = []

        for i, frame in enumerate(self.loader.get_frame_range(start_f, end_f)):
            if i % self.ref_sample_step != 0:
                continue
            frame = self._crop_frame_to_roi(frame)
            timestamp = (start_f + i) / max(self.loader.fps, 1.0)
            pose = self.detector.detect(frame, filter_stream="reference", timestamp=timestamp)
            if pose is not None:
                ref_frames.append(frame.copy())
                ref_poses.append(pose)

        self.root.after(0, self._finish_ref_pose_extraction, ref_frames, ref_poses)

    def _finish_ref_pose_extraction(self, ref_frames: list[np.ndarray], ref_poses: list[dict]):
        # 버튼 재활성화
        for widget in self.root.winfo_children():
            self._enable_buttons_recursive(widget)

        if not ref_poses:
            messagebox.showwarning("감지 실패", "해당 구간에서 포즈를 감지하지 못했습니다.")
            self.info_bar.config(text="포즈 감지 실패")
            return

        self.ref_frames = ref_frames
        self.ref_poses = ref_poses
        self.current_ref_idx = 0
        self.ref_playback_fps = max(1.0, self.loader.fps / self.ref_sample_step)
        self.info_bar.config(
            text=f"레퍼런스 포즈 {len(self.ref_poses)}개 추출 완료. 실시간 분석을 시작하세요."
        )

    def _validate_reference_roi_for_pose_extraction(self) -> bool:
        if self.preview_frame is None:
            return True

        target_frame = self._crop_frame_to_roi(self.preview_frame)
        person_count = self.detector.count_people(target_frame)

        if person_count <= 1:
            return True

        if self.crop_roi is None:
            messagebox.showwarning(
                "여러 사람 감지",
                "레퍼런스 영상에서 여러 사람이 감지되었습니다.\n"
                "원하는 사람만 포함되도록 영상 위에서 드래그해 영역을 선택한 뒤 포즈를 추출하세요.",
            )
            self.info_bar.config(text="여러 사람 감지: 원하는 사람만 ROI crop으로 선택하세요.")
        else:
            messagebox.showwarning(
                "영역 재선택 필요",
                "선택한 crop 영역 안에 아직 여러 사람이 감지되었습니다.\n"
                "분석할 사람 한 명만 들어오도록 영역을 더 좁게 선택하세요.",
            )
            self.info_bar.config(text="ROI 안에 여러 사람 감지: 영역을 더 좁게 선택하세요.")
        return False

    def _start_live(self):
        if not self.ref_poses:
            messagebox.showwarning("준비 안됨", "레퍼런스 포즈를 먼저 추출하세요.")
            return
        if self.is_recording:
            messagebox.showwarning("녹화 중", "녹화를 종료한 뒤 실시간 분석을 시작하세요.")
            return
        if self.is_playing_reference_preview:
            messagebox.showwarning("재생 중", "레퍼런스 구간 재생을 중지한 뒤 실시간 분석을 시작하세요.")
            return
        self.is_running = True
        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.last_ref_advance = time.monotonic()
        self.detector.reset_filter("live")
        threading.Thread(target=self._live_loop, daemon=True).start()
        self._update_canvas()

    def _stop_live(self):
        self.is_running = False
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)

    def _start_recording(self):
        if self.is_running:
            messagebox.showwarning("실행 중", "실시간 분석을 중지한 뒤 녹화를 시작하세요.")
            return
        if self.is_playing_reference_preview:
            messagebox.showwarning("재생 중", "레퍼런스 구간 재생을 중지한 뒤 녹화를 시작하세요.")
            return
        if self.loader.container is None:
            messagebox.showwarning("레퍼런스 없음", "먼저 유튜브 또는 레퍼런스 영상을 로드하세요.")
            return
        if not self.ref_frames or not self.ref_poses:
            messagebox.showwarning("준비 안됨", "레퍼런스 구간의 포즈를 먼저 추출하세요.")
            return
        if self.is_recording:
            return

        output_dir = Path(tempfile.gettempdir()) / "posecoach_recordings"
        output_dir.mkdir(parents=True, exist_ok=True)
        self.recorded_video_path = str(output_dir / f"user_recording_{int(time.time())}.mp4")

        self.is_recording = True
        self.btn_record_start.config(state=tk.DISABLED)
        self.btn_record_stop.config(state=tk.NORMAL)
        self.btn_record_analyze.config(state=tk.DISABLED)
        self.btn_video_play.config(state=tk.DISABLED)
        self.analysis_playback_frames = []
        self._reset_analysis_playback()
        self._clear_frame_queue()
        self.info_bar.config(text="카메라 준비 중...")
        threading.Thread(target=self._record_loop, args=(self.recorded_video_path,), daemon=True).start()
        self._update_canvas()

    def _stop_recording(self):
        self.is_recording = False
        self.btn_record_stop.config(state=tk.DISABLED)
        self.info_bar.config(text="녹화 종료 처리 중...")

    def _record_loop(self, output_path: str):
        cap = self._open_camera()
        if not cap.isOpened():
            self.is_recording = False
            self.root.after(0, messagebox.showerror, "카메라 오류", "카메라를 열 수 없습니다.")
            self.root.after(0, self._finish_recording, None)
            return

        ret, first_frame = cap.read()
        if not ret:
            cap.release()
            self.is_recording = False
            self.root.after(0, messagebox.showerror, "카메라 오류", "카메라 프레임을 읽을 수 없습니다.")
            self.root.after(0, self._finish_recording, None)
            return

        first_frame = cv2.flip(first_frame, 1)
        self._enqueue_frame(self._decorate_recording_frame(first_frame.copy(), 0.0))
        self.root.after(
            0,
            lambda: self.info_bar.config(text="카메라 녹화 중입니다. 종료 버튼을 누르면 녹화본을 불러옵니다."),
        )

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if fps <= 1:
            fps = 30.0
        height, width = first_frame.shape[:2]
        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            cap.release()
            self.is_recording = False
            self.root.after(0, messagebox.showerror, "녹화 오류", "녹화 파일을 생성할 수 없습니다.")
            self.root.after(0, self._finish_recording, None)
            return

        started_at = time.monotonic()
        writer.write(first_frame)

        while self.is_recording:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            writer.write(frame)
            display = self._decorate_recording_frame(frame.copy(), time.monotonic() - started_at)
            self._enqueue_frame(display)

        writer.release()
        cap.release()
        self.root.after(0, self._finish_recording, output_path)

    def _open_camera(self):
        """플랫폼별 최적 카메라 백엔드를 선택합니다."""
        import platform
        system = platform.system()

        if system == "Windows":
            backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, 0]
        elif system == "Darwin":  # macOS
            backends = [cv2.CAP_AVFOUNDATION, 0]
        else:  # Linux
            backends = [cv2.CAP_V4L2, 0]

        for backend in backends:
            cap = cv2.VideoCapture(0, backend) if backend else cv2.VideoCapture(0)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                return cap
            cap.release()
        return cv2.VideoCapture(0)

    def _decorate_recording_frame(self, frame: np.ndarray, elapsed: float) -> np.ndarray:
        cv2.circle(frame, (24, 24), 8, (0, 0, 255), -1)
        cv2.putText(frame, "REC", (40, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        ref_display = self._get_reference_display_for_elapsed(
            elapsed,
            overlay_pose=bool(self.ref_frames and self.ref_poses),
        )
        if ref_display is not None:
            return self._compose_split_screen(frame, ref_display)
        return frame

    def _clear_frame_queue(self):
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break

    def _enqueue_frame(self, frame: np.ndarray):
        if self.frame_queue.full():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                pass
        self.frame_queue.put(frame)

    def _get_reference_display_for_elapsed(self, elapsed: float, overlay_pose: bool = False):
        if self.ref_frames:
            idx = int(elapsed * self.ref_playback_fps) % len(self.ref_frames)
            frame = self.ref_frames[idx].copy()
            if overlay_pose and idx < len(self.ref_poses):
                frame = self.visualizer.draw_pose(frame, self.ref_poses[idx])
            return frame

        if self.loader.container is None:
            return None
        start_f = self.crop_start_var.get()
        end_f = self.crop_end_var.get()
        duration_frames = max(1, end_f - start_f)
        frame_idx = start_f + int(elapsed * self.loader.fps) % duration_frames
        frame = self.loader.get_frame_at(frame_idx)
        if frame is None:
            return None
        return self._crop_frame_to_roi(frame)

    def _finish_recording(self, output_path: str | None):
        self.is_recording = False
        self.btn_record_start.config(state=tk.NORMAL)
        self.btn_record_stop.config(state=tk.DISABLED)
        if not output_path:
            self.info_bar.config(text="녹화에 실패했습니다.")
            self.btn_video_play.config(text="▶", state=tk.NORMAL)
            return

        ok = self.user_loader.load_local(output_path)
        if not ok:
            self.info_bar.config(text="녹화본을 불러올 수 없습니다.")
            self.btn_video_play.config(text="▶", state=tk.NORMAL)
            return

        total_frames = self._loader_total_frames(self.user_loader)
        self.user_crop_start_var.set(0)
        self.user_crop_end_var.set(total_frames)
        self.user_range_slider.set_bounds(0, total_frames)
        self._on_user_slider_change(update_preview=False)
        self.btn_record_analyze.config(state=tk.NORMAL)
        self.info_bar.config(
            text=f"녹화 완료 | {self.user_loader.width}×{self.user_loader.height} | "
                 f"{self._format_seconds(self.user_loader.duration)} | {self.user_loader.fps:.1f}fps"
        )
        self.btn_video_play.config(text="▶", state=tk.NORMAL)
        self.playback_target = "user"
        self._show_user_selected_start()

    def _on_user_slider_change(self, *_, update_preview: bool = True):
        self.playback_target = "user"
        if update_preview and self.analysis_playback_frames:
            self.analysis_playback_frames = []
            self._reset_analysis_playback()
            self.btn_video_play.config(text="▶", state=tk.NORMAL)
        s = self.user_crop_start_var.get()
        e = self.user_crop_end_var.get()
        if e <= s:
            self.user_crop_end_var.set(s + 1)
            e = s + 1
        self.user_crop_label.config(
            text=f"구간: {self._format_frame_time(self.user_loader, s)} ~ "
                 f"{self._format_frame_time(self.user_loader, e)}"
        )
        if self.playback_target == "user" and not self.analysis_playback_frames:
            self.selection_playhead_frame = s
            self._sync_selection_seek_ui()
        if update_preview:
            self._schedule_user_preview_update(s)

    def _schedule_user_preview_update(self, frame_idx: int):
        if self.user_preview_after_id is not None:
            self.root.after_cancel(self.user_preview_after_id)
        self.user_preview_after_id = self.root.after(
            self.preview_debounce_ms,
            self._update_user_preview_at_second,
            frame_idx,
        )

    def _update_user_preview_at_second(self, frame_idx: int):
        self.user_preview_after_id = None
        if self.user_loader.container is None or self.is_running or self.is_recording or self.is_playing_analysis:
            return
        frame = self.user_loader.get_frame_at(frame_idx)
        if frame is not None:
            self._show_user_preview_frame(frame)

    def _show_user_selected_start(self):
        if self.user_loader.container is None:
            messagebox.showwarning("녹화 없음", "먼저 카메라 녹화를 완료하세요.")
            return
        self.playback_target = "user"
        self.selection_playhead_frame = self.user_crop_start_var.get()
        self._sync_selection_seek_ui()
        self._update_user_preview_at_second(self.user_crop_start_var.get())

    def _reset_analysis_playback(self):
        self.analysis_playhead_idx = 0
        self.analysis_seek_bar.set_max(1)
        self.analysis_seek_bar.set_value(0)
        self._update_analysis_time_label()

    def _set_analysis_playback_frames(self, frames: list[np.ndarray]):
        self.analysis_playback_frames = frames
        self.analysis_playhead_idx = 0
        self.analysis_seek_bar.set_max(max(1, len(frames)))
        self.analysis_seek_bar.set_value(0)
        self._update_analysis_time_label()

    def _seek_playback(self, frame_idx: int):
        if self.playback_target == "analysis" and self.analysis_playback_frames:
            self._seek_analysis(frame_idx)
            return
        self._seek_selection_playback(frame_idx)

    def _seek_analysis(self, frame_idx: int):
        if not self.analysis_playback_frames:
            return
        self.analysis_playhead_idx = max(0, min(frame_idx, len(self.analysis_playback_frames) - 1))
        self.analysis_seek_bar.set_value(self.analysis_playhead_idx)
        self._update_analysis_time_label()
        if self.is_playing_analysis:
            self.is_playing_analysis = False
        if not (self.is_running or self.is_recording):
            self._show_frame(self.analysis_playback_frames[self.analysis_playhead_idx].copy())

    def _seek_selection_playback(self, offset: int):
        context = self._selection_playback_context()
        if context is None:
            return
        loader, start_frame, end_frame, crop, label = context
        if end_frame <= start_frame:
            return

        if self.is_playing_reference_preview:
            self.is_playing_reference_preview = False
            self.selection_playback_generation += 1
            self.btn_video_play.config(text="▶")

        length = max(1, end_frame - start_frame)
        offset = max(0, min(offset, length - 1))
        frame_idx = start_frame + offset
        self.selection_playhead_frame = frame_idx
        self._sync_selection_seek_ui()

        frame = loader.get_frame_at(frame_idx)
        if frame is not None and not (self.is_running or self.is_recording or self.is_playing_analysis):
            display = self._crop_frame_to_roi(frame) if crop else frame
            self._show_frame(display)
        self.info_bar.config(text=f"{label} 위치 이동 | {self._format_frame_time(loader, frame_idx)}")

    def _selection_playback_context(self):
        if self.playback_target == "user":
            if self.user_loader.container is None:
                return None
            return (
                self.user_loader,
                self.user_crop_start_var.get(),
                self.user_crop_end_var.get(),
                False,
                "녹화본",
            )
        if self.loader.container is None:
            return None
        return (
            self.loader,
            self.crop_start_var.get(),
            self.crop_end_var.get(),
            True,
            "레퍼런스",
        )

    def _sync_selection_seek_ui(self):
        context = self._selection_playback_context()
        if context is None:
            return
        loader, start_frame, end_frame, _crop, _label = context
        length = max(1, end_frame - start_frame)
        offset = max(0, min(self.selection_playhead_frame - start_frame, length - 1))
        self.analysis_seek_bar.set_max(length)
        self.analysis_seek_bar.set_value(offset)
        self.analysis_time_label.config(
            text=f"{self._format_seconds(offset / max(loader.fps, 1e-6))} / "
                 f"{self._format_seconds((length - 1) / max(loader.fps, 1e-6))}"
        )

    def _update_analysis_time_label(self):
        total = len(self.analysis_playback_frames)
        if total <= 0:
            self.analysis_time_label.config(text="0:00 / 0:00")
            return
        fps = max(self.ref_playback_fps, 1.0)
        self.analysis_time_label.config(
            text=f"{self._format_seconds(self.analysis_playhead_idx / fps)} / "
                 f"{self._format_seconds((total - 1) / fps)}"
        )

    @staticmethod
    def _format_seconds(seconds: float) -> str:
        seconds = max(0.0, float(seconds))
        minutes = int(seconds // 60)
        remaining = seconds - minutes * 60
        return f"{minutes}:{remaining:05.2f}"

    @classmethod
    def _format_frame_time(cls, loader: VideoLoader, frame_idx: int) -> str:
        return cls._format_seconds(frame_idx / max(loader.fps, 1e-6))

    @staticmethod
    def _loader_total_frames(loader: VideoLoader) -> int:
        if loader.total_frames > 0:
            return max(1, int(loader.total_frames))
        return max(1, int(np.ceil(loader.duration * max(loader.fps, 1.0))))

    def _analyze_recorded_video(self):
        if not self.ref_poses:
            messagebox.showwarning("준비 안됨", "레퍼런스 포즈를 먼저 추출하세요.")
            return
        if self.user_loader.container is None:
            messagebox.showwarning("녹화 없음", "먼저 카메라 녹화를 완료하세요.")
            return
        if self.is_playing_analysis:
            messagebox.showwarning("재생 중", "분석 결과 재생이 끝난 뒤 다시 분석하세요.")
            return

        start_f = self.user_crop_start_var.get()
        end_f = self.user_crop_end_var.get()

        # 버튼 비활성화로 중복 실행 방지
        for widget in self.root.winfo_children():
            self._disable_buttons_recursive(widget)

        self.analysis_playback_frames = []
        self._reset_analysis_playback()
        self.info_bar.config(text="녹화 구간 분석 중... (시간이 걸릴 수 있습니다)")
        self.root.update()

        threading.Thread(
            target=self._analyze_recorded_video_worker,
            args=(start_f, end_f),
            daemon=True,
        ).start()

    def _analyze_recorded_video_worker(self, start_f: int, end_f: int):
        user_frames = []
        user_poses = []
        sample_step = max(1, int(round(self.user_loader.fps / self.ref_playback_fps)))
        total_frames = end_f - start_f
        self.detector.reset_filter("recorded")

        # 1단계: 포즈 추출
        for i, frame in enumerate(self.user_loader.get_frame_range(start_f, end_f)):
            if i % sample_step != 0:
                continue

            # 진행률 업데이트
            if i % (sample_step * 5) == 0:  # 5프레임마다 업데이트
                self.root.after(
                    0,
                    lambda idx=i: self.info_bar.config(
                        text=f"1단계: 포즈 추출 중... {idx}/{total_frames} 프레임"
                    )
                )

            timestamp = (start_f + i) / max(self.user_loader.fps, 1.0)
            pose = self.detector.detect(frame, filter_stream="recorded", timestamp=timestamp)
            if pose is not None:
                user_frames.append(frame.copy())
                user_poses.append(pose)

        if not user_poses:
            self.root.after(0, self._finish_recorded_analysis, None, None, None, [])
            return

        # 2단계: DTW 매칭
        self.root.after(0, lambda: self.info_bar.config(text="2단계: 동작 정렬 중... (DTW)"))
        ref_poses, ref_frames, repeat_count = self._repeat_reference_for_analysis(len(user_poses))
        scores = []
        last_display = None
        playback_frames = []
        dtw_path, dtw_cost = self.similarity_calc.dtw_match(ref_poses, user_poses)
        if not dtw_path:
            self.root.after(0, self._finish_recorded_analysis, None, None, None, [])
            return

        # 3단계: 유사도 계산 및 시각화
        total_pairs = len(dtw_path)
        for pair_idx, (ref_idx, user_idx) in enumerate(dtw_path):
            # 진행률 업데이트
            if pair_idx % 5 == 0:
                self.root.after(
                    0,
                    lambda idx=pair_idx: self.info_bar.config(
                        text=f"3단계: 유사도 계산 중... {idx}/{total_pairs} 쌍"
                    )
                )
            sim = self.similarity_calc.compute(ref_poses[ref_idx], user_poses[user_idx])
            scores.append(sim["overall"])
            user_display = self.visualizer.draw_pose(
                user_frames[user_idx].copy(), user_poses[user_idx], sim["keypoint_status"]
            )
            user_display = self.visualizer.draw_similarity_hud(user_display, sim)
            cv2.putText(
                user_display,
                f"DTW cost: {dtw_cost:.3f}",
                (10, 132),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (220, 220, 220),
                1,
            )
            ref_display = self.visualizer.draw_pose(
                ref_frames[ref_idx].copy(), ref_poses[ref_idx]
            )
            last_display = self._compose_split_screen(user_display, ref_display)
            playback_frames.append(last_display)

        avg_score = float(np.mean(scores)) if scores else 0.0
        min_score = float(np.min(scores)) if scores else 0.0
        max_score = float(np.max(scores)) if scores else 0.0

        # 나쁜 구간 찾기 (유사도 70% 미만)
        bad_indices = [i for i, s in enumerate(scores) if s < 0.70]
        bad_count = len(bad_indices)

        self.root.after(
            0,
            self._finish_recorded_analysis,
            avg_score,
            min_score,
            max_score,
            bad_count,
            total_pairs,
            f"{len(dtw_path)} / ref x{repeat_count}",
            last_display,
            playback_frames,
        )

    def _repeat_reference_for_analysis(self, target_len: int) -> tuple[list[dict], list[np.ndarray], int]:
        base_len = min(len(self.ref_poses), len(self.ref_frames))
        if base_len <= 0:
            return [], [], 0
        repeat_count = max(1, target_len // base_len)
        repeated_len = repeat_count * base_len
        ref_poses = [self.ref_poses[idx % base_len] for idx in range(repeated_len)]
        ref_frames = [self.ref_frames[idx % base_len] for idx in range(repeated_len)]
        return ref_poses, ref_frames, repeat_count

    def _finish_recorded_analysis(
        self,
        avg_score: float | None,
        min_score: float | None,
        max_score: float | None,
        bad_count: int | None,
        total_count: int | None,
        pair_count: int | str | None,
        display_frame: np.ndarray | None,
        playback_frames: list[np.ndarray],
    ):
        # 버튼 재활성화
        for widget in self.root.winfo_children():
            self._enable_buttons_recursive(widget)

        if avg_score is None or pair_count is None:
            messagebox.showwarning("감지 실패", "녹화 구간에서 포즈를 감지하지 못했습니다.")
            self.info_bar.config(text="녹화 구간 포즈 감지 실패")
            return
        self._set_analysis_playback_frames(playback_frames)
        if self.analysis_playback_frames:
            self.playback_target = "analysis"
        self._update_sim_ui(avg_score)

        # 상세 정보 표시
        info_text = (
            f"분석 완료 | 평균: {avg_score*100:.1f}% | "
            f"최소: {min_score*100:.1f}% | 최대: {max_score*100:.1f}% | "
            f"교정 필요: {bad_count}/{total_count} 구간"
        )
        self.info_bar.config(text=info_text)

        if display_frame is not None:
            self._show_frame(display_frame)

    def _play_video_from_controls(self):
        if self.is_playing_reference_preview:
            self._play_selection_video(self.playback_target)
            return
        if self.playback_target == "analysis" and self.analysis_playback_frames:
            self._play_analysis_result()
            return
        if self.playback_target == "user":
            self._play_selection_video("user")
            return
        self._play_selection_video("reference")

    def _play_analysis_result(self):
        if not self.analysis_playback_frames:
            messagebox.showwarning("재생 없음", "먼저 녹화 구간 분석을 완료하세요.")
            return
        if self.is_recording or self.is_running:
            messagebox.showwarning("실행 중", "녹화 또는 실시간 분석을 중지한 뒤 재생하세요.")
            return
        if self.is_playing_analysis:
            return

        last_idx = len(self.analysis_playback_frames) - 1
        start_idx = 0 if self.analysis_playhead_idx >= last_idx else self.analysis_playhead_idx
        start_idx = max(0, min(start_idx, last_idx))

        self.is_playing_analysis = True
        self._clear_frame_queue()
        self.btn_record_analyze.config(state=tk.DISABLED)
        self.btn_video_play.config(text="■", state=tk.DISABLED)
        self.info_bar.config(text="분석 결과 영상 재생 중...")
        threading.Thread(
            target=self._play_analysis_result_worker,
            args=(start_idx,),
            daemon=True,
        ).start()
        self._update_canvas()

    def _play_analysis_result_worker(self, start_idx: int):
        interval = 1.0 / max(self.ref_playback_fps, 1.0)
        total = len(self.analysis_playback_frames)
        for frame_idx in range(start_idx, total):
            if not self.is_playing_analysis:
                break
            display = self.analysis_playback_frames[frame_idx].copy()
            cv2.putText(
                display,
                f"PLAY {frame_idx + 1}/{total}",
                (16, display.shape[0] - 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            self.analysis_playhead_idx = frame_idx
            self.root.after(0, self._sync_analysis_seek_ui, frame_idx)
            self._enqueue_frame(display)
            time.sleep(interval)
        self.root.after(0, self._finish_analysis_playback)

    def _sync_analysis_seek_ui(self, frame_idx: int):
        self.analysis_playhead_idx = max(0, min(frame_idx, len(self.analysis_playback_frames) - 1))
        self.analysis_seek_bar.set_value(self.analysis_playhead_idx)
        self._update_analysis_time_label()

    def _finish_analysis_playback(self):
        self.is_playing_analysis = False
        self.btn_record_analyze.config(state=tk.NORMAL)
        self.btn_video_play.config(text="▶", state=tk.NORMAL)
        self.info_bar.config(text="분석 결과 영상 재생 완료")

    # ── 실시간 카메라 루프 (별도 스레드) ──────────────────────

    def _live_loop(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.is_running = False
            self.root.after(0, messagebox.showerror, "카메라 오류", "카메라를 열 수 없습니다.")
            self.root.after(0, self._stop_live)
            return

        ref_pose = self._get_ref_pose()
        frame_count = 0
        last_process_time = time.monotonic()

        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)  # 좌우 반전 (거울 모드)

            frame_count += 1
            current_time = time.monotonic()

            # 프레임 스킵: 처리 시간이 오래 걸리면 2프레임마다 처리
            process_interval = current_time - last_process_time
            skip_frame = (process_interval < 0.1 and frame_count % 2 == 0)

            if skip_frame:
                # 프레임 스킵 - 이전 결과 재사용
                if not self.frame_queue.full():
                    self.frame_queue.put(frame)
                continue

            last_process_time = current_time
            user_pose = self.detector.detect(
                frame,
                filter_stream="live",
                timestamp=current_time,
            )

            if user_pose is not None and ref_pose is not None:
                sim = self.similarity_calc.compute(ref_pose, user_pose)
                user_frame = self.visualizer.draw_pose(
                    frame, user_pose, sim["keypoint_status"]
                )
                user_frame = self.visualizer.draw_similarity_hud(user_frame, sim)

                ref_frame = self.ref_frames[self.current_ref_idx].copy()
                ref_frame = self.visualizer.draw_pose(ref_frame, ref_pose)
                frame = self._compose_split_screen(user_frame, ref_frame)

                # 유사도 수치 UI 업데이트 (메인 스레드에 위임)
                overall = sim["overall"]
                self.root.after(0, self._update_sim_ui, overall)

                ref_pose = self._advance_ref_pose_by_time()
            else:
                cv2.putText(frame, "No pose detected", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 180, 255), 2)
                if ref_pose is not None and self.ref_frames:
                    ref_frame = self.ref_frames[self.current_ref_idx].copy()
                    ref_frame = self.visualizer.draw_pose(ref_frame, ref_pose)
                    frame = self._compose_split_screen(frame, ref_frame)
                    ref_pose = self._advance_ref_pose_by_time()

            # 큐에 프레임 전달
            if not self.frame_queue.full():
                self.frame_queue.put(frame)

        cap.release()

    def _get_ref_pose(self) -> dict | None:
        if self.ref_poses:
            return self.ref_poses[self.current_ref_idx]
        return None

    def _advance_ref_pose_by_time(self) -> dict | None:
        if not self.ref_poses:
            return None

        now = time.monotonic()
        interval = 1.0 / self.ref_playback_fps
        steps = int((now - self.last_ref_advance) / interval)
        if steps > 0:
            self.current_ref_idx = (self.current_ref_idx + steps) % len(self.ref_poses)
            self.last_ref_advance += steps * interval
        return self.ref_poses[self.current_ref_idx]

    def _compose_split_screen(
        self,
        user_frame: np.ndarray,
        ref_frame: np.ndarray,
    ) -> np.ndarray:
        """사용자 카메라와 레퍼런스 영상을 같은 비율의 좌우 패널로 합성합니다."""
        panel_h, panel_w = user_frame.shape[:2]
        user_panel = self._fit_frame_to_panel(user_frame, panel_w, panel_h)
        ref_panel = self._fit_frame_to_panel(ref_frame, panel_w, panel_h)

        self._draw_panel_label(user_panel, "USER")
        self._draw_panel_label(ref_panel, "REFERENCE")

        return np.hstack((user_panel, ref_panel))

    @staticmethod
    def _fit_frame_to_panel(frame: np.ndarray, panel_w: int, panel_h: int) -> np.ndarray:
        h, w = frame.shape[:2]
        scale = min(panel_w / w, panel_h / h)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
        x = (panel_w - new_w) // 2
        y = (panel_h - new_h) // 2
        panel[y:y + new_h, x:x + new_w] = resized
        return panel

    @staticmethod
    def _draw_panel_label(frame: np.ndarray, text: str):
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (150, 34), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
        cv2.putText(
            frame,
            text,
            (12, 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )

    # ── UI 업데이트 ───────────────────────────────────────

    def _on_video_area_resize(self, event):
        self.video_area_size = (max(1, event.width), max(1, event.height))
        self._place_video_controls(self.video_controls_y)

    def _bind_video_controls_motion(self):
        widgets = (
            self.analysis_controls,
            self.btn_video_play,
            self.analysis_seek_bar,
            self.analysis_time_label,
        )
        for widget in widgets:
            widget.bind("<Motion>", self._on_video_area_motion, add="+")
            widget.bind("<ButtonPress-1>", self._on_video_area_motion, add="+")

    def _on_video_area_motion(self, _event=None):
        self._show_video_controls()
        self._schedule_video_controls_hide()

    def _schedule_video_controls_hide(self):
        if self.video_controls_hide_after_id is not None:
            self.root.after_cancel(self.video_controls_hide_after_id)
        self.video_controls_hide_after_id = self.root.after(
            self.video_controls_hide_delay_ms,
            self._hide_video_controls,
        )

    def _show_video_controls(self):
        self.video_controls_visible = True
        self._animate_video_controls(-2)

    def _hide_video_controls(self):
        self.video_controls_hide_after_id = None
        self.video_controls_visible = False
        self._animate_video_controls(self._hidden_video_controls_y())

    def _hidden_video_controls_y(self) -> int:
        return max(40, self.analysis_controls.winfo_reqheight() + 8)

    def _animate_video_controls(self, target_y: int):
        if self.video_controls_animation_after_id is not None:
            self.root.after_cancel(self.video_controls_animation_after_id)
            self.video_controls_animation_after_id = None

        current_y = self.video_controls_y
        if current_y == target_y:
            self._place_video_controls(target_y)
            return

        direction = 1 if target_y > current_y else -1
        next_y = current_y + direction * min(8, abs(target_y - current_y))
        self._place_video_controls(next_y)
        self.video_controls_animation_after_id = self.root.after(
            12,
            self._animate_video_controls,
            target_y,
        )

    def _place_video_controls(self, y: int):
        self.video_controls_y = y
        self.analysis_controls.place(relx=0.0, rely=1.0, relwidth=1.0, y=y, anchor="sw")

    def _update_canvas(self):
        """Tkinter 메인 루프에서 주기적으로 캔버스 갱신."""
        if not self.frame_queue.empty():
            frame = self.frame_queue.get()
            self._show_frame(frame)
        if self.is_running or self.is_recording or self.is_playing_analysis or self.is_playing_reference_preview:
            self.root.after(30, self._update_canvas)

    def _show_frame(self, frame: np.ndarray):
        """OpenCV BGR 프레임을 Tkinter 캔버스에 표시."""
        cw, ch = self.video_area_size
        cw = self.video_area.winfo_width() or cw or 960
        ch = self.video_area.winfo_height() or ch or 540

        h, w = frame.shape[:2]
        scale = min(cw / max(w, 1), ch / max(h, 1))
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        canvas_frame = np.zeros((ch, cw, 3), dtype=np.uint8)
        x = (cw - new_w) // 2
        y = (ch - new_h) // 2
        canvas_frame[y:y + new_h, x:x + new_w] = resized
        self.display_size = (new_w, new_h)
        self.display_origin = (x, y)

        rgb = cv2.cvtColor(canvas_frame, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.canvas.config(image=photo)
        self.canvas.image = photo  # 참조 유지

    def _show_preview_frame(self, frame: np.ndarray, drag_box: tuple[int, int, int, int] | None = None):
        self.preview_frame = frame.copy()
        display = frame.copy()
        if self.crop_roi is not None:
            x1, y1, x2, y2 = self._roi_to_pixels(display, self.crop_roi)
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 220, 255), 2)
        if drag_box is not None:
            x1, y1, x2, y2 = drag_box
            overlay = display.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 220, 255), -1)
            cv2.addWeighted(overlay, 0.18, display, 0.82, 0, display)
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 220, 255), 2)
        self._show_frame(display)

    def _show_user_preview_frame(self, frame: np.ndarray):
        display = frame.copy()
        cv2.putText(display, "RECORDED USER", (16, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (34, 197, 94), 2)
        self._show_frame(display)

    def _on_roi_start(self, event):
        if self.preview_frame is None or self.is_running or self.is_recording or self.is_playing_analysis:
            return
        canvas_x, canvas_y = self._event_to_canvas_point(event)
        if self._canvas_to_frame_point(canvas_x, canvas_y) is None:
            return
        self.drag_start = (canvas_x, canvas_y)
        self.canvas.bind_all("<B1-Motion>", self._on_roi_drag, add="+")
        self.canvas.bind_all("<ButtonRelease-1>", self._on_roi_end, add="+")
        self.info_bar.config(text="ROI 영역 선택 중...")

    def _on_roi_drag(self, event):
        if self.preview_frame is None or self.drag_start is None or self.is_running or self.is_recording or self.is_playing_analysis:
            return

        canvas_x, canvas_y = self._event_to_canvas_point(event)
        start = self._canvas_to_frame_point(*self.drag_start)
        end = self._canvas_to_frame_point(canvas_x, canvas_y, clamp=True)
        if start is None or end is None:
            return

        x1, x2 = sorted((start[0], end[0]))
        y1, y2 = sorted((start[1], end[1]))
        self._show_preview_frame(self.preview_frame, (x1, y1, x2, y2))

    def _on_roi_end(self, event):
        if self.preview_frame is None or self.drag_start is None or self.is_running or self.is_recording or self.is_playing_analysis:
            return

        canvas_x, canvas_y = self._event_to_canvas_point(event)
        start = self._canvas_to_frame_point(*self.drag_start)
        end = self._canvas_to_frame_point(canvas_x, canvas_y, clamp=True)
        self.canvas.unbind_all("<B1-Motion>")
        self.canvas.unbind_all("<ButtonRelease-1>")
        self.drag_start = None
        if start is None or end is None:
            self._show_preview_frame(self.preview_frame)
            return

        h, w = self.preview_frame.shape[:2]
        x1, x2 = sorted((start[0], end[0]))
        y1, y2 = sorted((start[1], end[1]))
        if (x2 - x1) < 20 or (y2 - y1) < 20:
            self._show_preview_frame(self.preview_frame)
            self.info_bar.config(text="ROI 영역이 너무 작습니다. 분석할 사람 전체가 들어오도록 다시 드래그하세요.")
            return

        # 이전 ROI를 히스토리에 저장
        self.crop_roi_history.append(self.crop_roi)
        if len(self.crop_roi_history) > 10:  # 최대 10개까지만 저장
            self.crop_roi_history.pop(0)

        self.crop_roi = (x1 / w, y1 / h, x2 / w, y2 / h)
        self.btn_roi_undo.config(state=tk.NORMAL)
        self.info_bar.config(
            text=f"ROI crop 설정 완료: ({x1}, {y1})~({x2}, {y2}). 포즈 추출을 누르세요."
        )
        self._show_preview_frame(self.preview_frame)

    def _undo_roi(self):
        """ROI 설정을 이전 상태로 되돌림"""
        if not self.crop_roi_history:
            return

        self.crop_roi = self.crop_roi_history.pop()
        if not self.crop_roi_history:
            self.btn_roi_undo.config(state=tk.DISABLED)

        if self.preview_frame is not None:
            self._show_preview_frame(self.preview_frame)
        self.info_bar.config(text="ROI가 이전 상태로 되돌려졌습니다.")

    def _reset_roi(self):
        if self.crop_roi is not None:
            # ROI가 설정되어 있는 경우: 히스토리에 저장
            self.crop_roi_history.append(self.crop_roi)
            if len(self.crop_roi_history) > 10:
                self.crop_roi_history.pop(0)
            self.btn_roi_undo.config(state=tk.NORMAL)
            self.crop_roi = None
            if self.preview_frame is not None:
                self._show_preview_frame(self.preview_frame)
            self.info_bar.config(text="영역 crop이 초기화되었습니다.")
        else:
            # ROI가 없는 경우: 이미 초기화된 상태
            self.info_bar.config(text="설정된 영역이 없습니다. 영상 위에서 드래그하여 영역을 선택하세요.")

    def _canvas_to_frame_point(self, x: int, y: int, clamp: bool = False) -> tuple[int, int] | None:
        if self.preview_frame is None:
            return None

        ox, oy = self.display_origin
        dw, dh = self.display_size
        if dw <= 0 or dh <= 0:
            return None
        if clamp:
            x = max(ox, min(ox + dw, x))
            y = max(oy, min(oy + dh, y))
        elif x < ox or y < oy or x > ox + dw or y > oy + dh:
            return None

        h, w = self.preview_frame.shape[:2]
        fx = int((x - ox) * w / dw)
        fy = int((y - oy) * h / dh)
        return max(0, min(w - 1, fx)), max(0, min(h - 1, fy))

    def _event_to_canvas_point(self, event) -> tuple[int, int]:
        return event.x_root - self.canvas.winfo_rootx(), event.y_root - self.canvas.winfo_rooty()

    def _crop_frame_to_roi(self, frame: np.ndarray) -> np.ndarray:
        if self.crop_roi is None:
            return frame
        x1, y1, x2, y2 = self._roi_to_pixels(frame, self.crop_roi)
        return frame[y1:y2, x1:x2]

    @staticmethod
    def _roi_to_pixels(frame: np.ndarray, roi: tuple[float, float, float, float]):
        h, w = frame.shape[:2]
        x1 = max(0, min(w - 1, int(roi[0] * w)))
        y1 = max(0, min(h - 1, int(roi[1] * h)))
        x2 = max(x1 + 1, min(w, int(roi[2] * w)))
        y2 = max(y1 + 1, min(h, int(roi[3] * h)))
        return x1, y1, x2, y2

    def _update_sim_ui(self, overall: float):
        pct = overall * 100
        self.sim_label.config(
            text=f"{pct:.1f}%",
            fg="#4ade80" if overall >= 0.80 else "#f87171",
        )
        self.status_label.config(
            text="GOOD POSE" if overall >= 0.80 else "자세를 교정하세요",
            fg="#4ade80" if overall >= 0.80 else "#f87171",
        )

    def run(self):
        self.root.mainloop()

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
        self.recorded_video_path: str | None = None
        self.analysis_playback_frames: list[np.ndarray] = []
        self.crop_start: int | None = None
        self.crop_end: int | None = None
        self.user_crop_start_var = tk.IntVar(value=0)
        self.user_crop_end_var = tk.IntVar(value=0)
        self.crop_roi: tuple[float, float, float, float] | None = None
        self.drag_start: tuple[int, int] | None = None
        self.preview_frame: np.ndarray | None = None
        self.display_origin = (0, 0)
        self.display_size = (0, 0)
        self.ref_sample_step = 5
        self.ref_playback_fps = 6.0
        self.last_ref_advance = 0.0
        self.preview_after_id: str | None = None
        self.user_preview_after_id: str | None = None
        self.preview_debounce_ms = 180

        self._build_ui()

    # ── UI 구성 ────────────────────────────────────────────

    def _build_ui(self):
        bg = "#0b1120"
        panel_bg = "#111827"
        section_bg = "#172033"
        text = "#e5e7eb"
        muted = "#94a3b8"
        accent = "#38bdf8"
        border = "#243247"

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
        self.btn_play_analysis = ttk.Button(
            record_section,
            text="분석 결과 재생",
            command=self._play_analysis_result,
            state=tk.DISABLED,
        )
        self.btn_play_analysis.pack(pady=(0, 14), fill=tk.X, padx=14)

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

        self.canvas = tk.Label(
            right,
            bg="#020617",
            cursor="crosshair",
            highlightbackground=border,
            highlightthickness=1,
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<ButtonPress-1>", self._on_roi_start)
        self.canvas.bind("<ButtonRelease-1>", self._on_roi_end)

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

        self.crop_label = tk.Label(
            timeline,
            text="구간: -",
            bg="#111827",
            fg=text,
            font=("Malgun Gothic", 10, "bold"),
            width=18,
        )
        self.crop_label.pack(side=tk.LEFT, padx=10, pady=10)

        ttk.Button(
            timeline,
            text="포즈 추출",
            command=self._extract_ref_poses,
            style="Primary.TButton",
        ).pack(side=tk.LEFT, padx=(10, 6), pady=8)
        ttk.Button(timeline, text="영역 초기화", command=self._reset_roi).pack(
            side=tk.LEFT, padx=(0, 12), pady=8
        )

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
            width=18,
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

        total_sec = max(1, int(self.loader.duration))
        self.crop_start_var.set(0)
        self.crop_end_var.set(total_sec)
        self.range_slider.set_bounds(0, total_sec)
        self._on_slider_change(update_preview=False)
        self.info_bar.config(
            text=f"로드 완료 | {self.loader.width}×{self.loader.height} | "
                 f"{total_sec}초 | {self.loader.fps:.1f}fps"
        )
        # 첫 프레임 미리보기
        frame = self.loader.get_frame_at(0)
        if frame is not None:
            self._show_preview_frame(frame)

    def _on_slider_change(self, *_, update_preview: bool = True):
        s = self.crop_start_var.get()
        e = self.crop_end_var.get()
        if e <= s:
            self.crop_end_var.set(s + 1)
            e = s + 1
        self.crop_label.config(text=f"구간: {s}초 ~ {e}초")
        if update_preview:
            self._schedule_preview_update(s)

    def _schedule_preview_update(self, second: int):
        if self.preview_after_id is not None:
            self.root.after_cancel(self.preview_after_id)
        self.preview_after_id = self.root.after(
            self.preview_debounce_ms,
            self._update_preview_at_second,
            second,
        )

    def _update_preview_at_second(self, second: int):
        self.preview_after_id = None
        if self.loader.container is None or self.is_running or self.is_recording or self.is_playing_analysis:
            return
        f_idx = int(second * self.loader.fps)
        frame = self.loader.get_frame_at(f_idx)
        if frame is not None:
            self._show_preview_frame(frame)

    def _extract_ref_poses(self):
        """크롭된 구간에서 레퍼런스 포즈를 추출합니다."""
        if self.loader.container is None:
            messagebox.showwarning("오류", "먼저 영상을 로드하세요.")
            return
        if self.is_recording:
            messagebox.showwarning("녹화 중", "녹화를 종료한 뒤 레퍼런스 포즈를 추출하세요.")
            return

        start_f = int(self.crop_start_var.get() * self.loader.fps)
        end_f = int(self.crop_end_var.get() * self.loader.fps)

        self.ref_frames.clear()
        self.ref_poses.clear()
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
            pose = self.detector.detect(frame)
            if pose is not None:
                ref_frames.append(frame.copy())
                ref_poses.append(pose)

        self.root.after(0, self._finish_ref_pose_extraction, ref_frames, ref_poses)

    def _finish_ref_pose_extraction(self, ref_frames: list[np.ndarray], ref_poses: list[dict]):
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

    def _start_live(self):
        if not self.ref_poses:
            messagebox.showwarning("준비 안됨", "레퍼런스 포즈를 먼저 추출하세요.")
            return
        if self.is_recording:
            messagebox.showwarning("녹화 중", "녹화를 종료한 뒤 실시간 분석을 시작하세요.")
            return
        self.is_running = True
        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.last_ref_advance = time.monotonic()
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
        self.btn_play_analysis.config(state=tk.DISABLED)
        self.analysis_playback_frames = []
        self.info_bar.config(text="카메라 녹화 중입니다. 종료 버튼을 누르면 녹화본을 불러옵니다.")
        threading.Thread(target=self._record_loop, args=(self.recorded_video_path,), daemon=True).start()
        self._update_canvas()

    def _stop_recording(self):
        self.is_recording = False
        self.btn_record_stop.config(state=tk.DISABLED)
        self.info_bar.config(text="녹화 종료 처리 중...")

    def _record_loop(self, output_path: str):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.is_recording = False
            self.root.after(0, messagebox.showerror, "카메라 오류", "카메라를 열 수 없습니다.")
            self.root.after(0, self._finish_recording, None)
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if fps <= 1:
            fps = 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
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

        while self.is_recording:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            writer.write(frame)
            display = frame.copy()
            cv2.circle(display, (24, 24), 8, (0, 0, 255), -1)
            cv2.putText(display, "REC", (40, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            ref_display = self._get_reference_display_for_elapsed(
                time.monotonic() - started_at,
                overlay_pose=bool(self.ref_frames and self.ref_poses),
            )
            if ref_display is not None:
                display = self._compose_split_screen(display, ref_display)
            if not self.frame_queue.full():
                self.frame_queue.put(display)

        writer.release()
        cap.release()
        self.root.after(0, self._finish_recording, output_path)

    def _get_reference_display_for_elapsed(self, elapsed: float, overlay_pose: bool = False):
        if self.ref_frames:
            idx = int(elapsed * self.ref_playback_fps) % len(self.ref_frames)
            frame = self.ref_frames[idx].copy()
            if overlay_pose and idx < len(self.ref_poses):
                frame = self.visualizer.draw_pose(frame, self.ref_poses[idx])
            return frame

        if self.loader.container is None:
            return None
        start_sec = self.crop_start_var.get()
        end_sec = self.crop_end_var.get()
        duration = max(1, end_sec - start_sec)
        second = start_sec + (elapsed % duration)
        frame = self.loader.get_frame_at(int(second * self.loader.fps))
        if frame is None:
            return None
        return self._crop_frame_to_roi(frame)

    def _finish_recording(self, output_path: str | None):
        self.is_recording = False
        self.btn_record_start.config(state=tk.NORMAL)
        self.btn_record_stop.config(state=tk.DISABLED)
        if not output_path:
            self.info_bar.config(text="녹화에 실패했습니다.")
            return

        ok = self.user_loader.load_local(output_path)
        if not ok:
            self.info_bar.config(text="녹화본을 불러올 수 없습니다.")
            return

        total_sec = max(1, int(self.user_loader.duration))
        self.user_crop_start_var.set(0)
        self.user_crop_end_var.set(total_sec)
        self.user_range_slider.set_bounds(0, total_sec)
        self._on_user_slider_change(update_preview=False)
        self.btn_record_analyze.config(state=tk.NORMAL)
        self.info_bar.config(
            text=f"녹화 완료 | {self.user_loader.width}×{self.user_loader.height} | "
                 f"{total_sec}초 | {self.user_loader.fps:.1f}fps"
        )
        self._show_user_selected_start()

    def _on_user_slider_change(self, *_, update_preview: bool = True):
        s = self.user_crop_start_var.get()
        e = self.user_crop_end_var.get()
        if e <= s:
            self.user_crop_end_var.set(s + 1)
            e = s + 1
        self.user_crop_label.config(text=f"구간: {s}초 ~ {e}초")
        if update_preview:
            self._schedule_user_preview_update(s)

    def _schedule_user_preview_update(self, second: int):
        if self.user_preview_after_id is not None:
            self.root.after_cancel(self.user_preview_after_id)
        self.user_preview_after_id = self.root.after(
            self.preview_debounce_ms,
            self._update_user_preview_at_second,
            second,
        )

    def _update_user_preview_at_second(self, second: int):
        self.user_preview_after_id = None
        if self.user_loader.container is None or self.is_running or self.is_recording or self.is_playing_analysis:
            return
        frame = self.user_loader.get_frame_at(int(second * self.user_loader.fps))
        if frame is not None:
            self._show_user_preview_frame(frame)

    def _show_user_selected_start(self):
        if self.user_loader.container is None:
            messagebox.showwarning("녹화 없음", "먼저 카메라 녹화를 완료하세요.")
            return
        self._update_user_preview_at_second(self.user_crop_start_var.get())

    def _analyze_recorded_video(self):
        if not self.ref_poses:
            messagebox.showwarning("준비 안됨", "레퍼런스 포즈를 먼저 추출하세요.")
            return
        if self.user_loader.container is None:
            messagebox.showwarning("녹화 없음", "먼저 카메라 녹화를 완료하세요.")
            return

        start_f = int(self.user_crop_start_var.get() * self.user_loader.fps)
        end_f = int(self.user_crop_end_var.get() * self.user_loader.fps)
        self.btn_record_analyze.config(state=tk.DISABLED)
        self.btn_play_analysis.config(state=tk.DISABLED)
        self.analysis_playback_frames = []
        self.info_bar.config(text="녹화 구간 분석 중...")
        threading.Thread(
            target=self._analyze_recorded_video_worker,
            args=(start_f, end_f),
            daemon=True,
        ).start()

    def _analyze_recorded_video_worker(self, start_f: int, end_f: int):
        user_frames = []
        user_poses = []
        sample_step = max(1, int(round(self.user_loader.fps / self.ref_playback_fps)))

        for i, frame in enumerate(self.user_loader.get_frame_range(start_f, end_f)):
            if i % sample_step != 0:
                continue
            pose = self.detector.detect(frame)
            if pose is not None:
                user_frames.append(frame.copy())
                user_poses.append(pose)

        if not user_poses:
            self.root.after(0, self._finish_recorded_analysis, None, None, None, [])
            return

        scores = []
        last_display = None
        playback_frames = []
        pair_count = min(len(self.ref_poses), len(user_poses))
        for idx in range(pair_count):
            ref_idx = int(idx * len(self.ref_poses) / pair_count)
            user_idx = int(idx * len(user_poses) / pair_count)
            sim = self.similarity_calc.compute(self.ref_poses[ref_idx], user_poses[user_idx])
            scores.append(sim["overall"])
            user_display = self.visualizer.draw_pose(
                user_frames[user_idx].copy(), user_poses[user_idx], sim["keypoint_status"]
            )
            user_display = self.visualizer.draw_similarity_hud(user_display, sim)
            ref_display = self.visualizer.draw_pose(
                self.ref_frames[ref_idx].copy(), self.ref_poses[ref_idx]
            )
            last_display = self._compose_split_screen(user_display, ref_display)
            playback_frames.append(last_display)

        avg_score = float(np.mean(scores)) if scores else 0.0
        self.root.after(
            0,
            self._finish_recorded_analysis,
            avg_score,
            pair_count,
            last_display,
            playback_frames,
        )

    def _finish_recorded_analysis(
        self,
        avg_score: float | None,
        pair_count: int | None,
        display_frame: np.ndarray | None,
        playback_frames: list[np.ndarray],
    ):
        self.btn_record_analyze.config(state=tk.NORMAL)
        if avg_score is None or pair_count is None:
            messagebox.showwarning("감지 실패", "녹화 구간에서 포즈를 감지하지 못했습니다.")
            self.info_bar.config(text="녹화 구간 포즈 감지 실패")
            return
        self.analysis_playback_frames = playback_frames
        if self.analysis_playback_frames:
            self.btn_play_analysis.config(state=tk.NORMAL)
        self._update_sim_ui(avg_score)
        self.info_bar.config(text=f"녹화 구간 분석 완료 | 비교 프레임 {pair_count}개 | 평균 유사도 {avg_score*100:.1f}%")
        if display_frame is not None:
            self._show_frame(display_frame)

    def _play_analysis_result(self):
        if not self.analysis_playback_frames:
            messagebox.showwarning("재생 없음", "먼저 녹화 구간 분석을 완료하세요.")
            return
        if self.is_recording or self.is_running:
            messagebox.showwarning("실행 중", "녹화 또는 실시간 분석을 중지한 뒤 재생하세요.")
            return
        if self.is_playing_analysis:
            return

        self.is_playing_analysis = True
        self.btn_play_analysis.config(state=tk.DISABLED)
        self.info_bar.config(text="분석 결과 재생 중...")
        threading.Thread(target=self._play_analysis_result_worker, daemon=True).start()
        self._update_canvas()

    def _play_analysis_result_worker(self):
        interval = 1.0 / max(self.ref_playback_fps, 1.0)
        for frame in self.analysis_playback_frames:
            if not self.is_playing_analysis:
                break
            if not self.frame_queue.full():
                self.frame_queue.put(frame.copy())
            time.sleep(interval)
        self.root.after(0, self._finish_analysis_playback)

    def _finish_analysis_playback(self):
        self.is_playing_analysis = False
        self.btn_play_analysis.config(state=tk.NORMAL)
        self.info_bar.config(text="분석 결과 재생 완료")

    # ── 실시간 카메라 루프 (별도 스레드) ──────────────────────

    def _live_loop(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.is_running = False
            self.root.after(0, messagebox.showerror, "카메라 오류", "카메라를 열 수 없습니다.")
            self.root.after(0, self._stop_live)
            return

        ref_pose = self._get_ref_pose()

        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)  # 좌우 반전 (거울 모드)

            user_pose = self.detector.detect(frame)

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

    def _update_canvas(self):
        """Tkinter 메인 루프에서 주기적으로 캔버스 갱신."""
        if not self.frame_queue.empty():
            frame = self.frame_queue.get()
            self._show_frame(frame)
        if self.is_running or self.is_recording or self.is_playing_analysis:
            self.root.after(30, self._update_canvas)

    def _show_frame(self, frame: np.ndarray):
        """OpenCV BGR 프레임을 Tkinter 캔버스에 표시."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)

        # 캔버스 크기에 맞게 리사이즈
        cw = self.canvas.winfo_width() or 960
        ch = self.canvas.winfo_height() or 720
        img.thumbnail((cw, ch), Image.LANCZOS)
        self.display_size = img.size
        self.display_origin = ((cw - img.size[0]) // 2, (ch - img.size[1]) // 2)

        photo = ImageTk.PhotoImage(img)
        self.canvas.config(image=photo)
        self.canvas.image = photo  # 참조 유지

    def _show_preview_frame(self, frame: np.ndarray):
        self.preview_frame = frame.copy()
        display = frame.copy()
        if self.crop_roi is not None:
            x1, y1, x2, y2 = self._roi_to_pixels(display, self.crop_roi)
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
        self.drag_start = (event.x, event.y)

    def _on_roi_end(self, event):
        if self.preview_frame is None or self.drag_start is None or self.is_running or self.is_recording or self.is_playing_analysis:
            return

        start = self._canvas_to_frame_point(*self.drag_start)
        end = self._canvas_to_frame_point(event.x, event.y)
        self.drag_start = None
        if start is None or end is None:
            return

        h, w = self.preview_frame.shape[:2]
        x1, x2 = sorted((start[0], end[0]))
        y1, y2 = sorted((start[1], end[1]))
        if (x2 - x1) < 20 or (y2 - y1) < 20:
            return

        self.crop_roi = (x1 / w, y1 / h, x2 / w, y2 / h)
        self.info_bar.config(text="영역 crop 설정 완료. 구간 확정 & 포즈 추출을 누르세요.")
        self._show_preview_frame(self.preview_frame)

    def _reset_roi(self):
        self.crop_roi = None
        if self.preview_frame is not None:
            self._show_preview_frame(self.preview_frame)
        self.info_bar.config(text="영역 crop이 초기화되었습니다.")

    def _canvas_to_frame_point(self, x: int, y: int) -> tuple[int, int] | None:
        if self.preview_frame is None:
            return None

        ox, oy = self.display_origin
        dw, dh = self.display_size
        if dw <= 0 or dh <= 0:
            return None
        if x < ox or y < oy or x > ox + dw or y > oy + dh:
            return None

        h, w = self.preview_frame.shape[:2]
        fx = int((x - ox) * w / dw)
        fy = int((y - oy) * h / dh)
        return max(0, min(w - 1, fx)), max(0, min(h - 1, fy))

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

import re
import shutil
import threading
from fractions import Fraction

import av
import yt_dlp


class VideoLoader:
    """로컬 영상 또는 유튜브 URL에서 PyAV로 프레임을 로드합니다."""

    def __init__(self):
        self.container: av.container.InputContainer | None = None
        self.video_stream = None
        self.lock = threading.RLock()
        self.total_frames = 0
        self.fps = 30.0
        self.width = 0
        self.height = 0
        self.duration = 0.0
        self.source_duration = 0.0
        self.source = ""
        self.is_stream = False

    def load_local(self, path: str) -> bool:
        """로컬 파일 로드."""
        self.release()
        self.source_duration = 0.0
        self.source = path
        self.is_stream = False
        return self._open_source()

    def load_youtube(self, url: str) -> bool:
        """yt-dlp로 유튜브 스트리밍 URL을 추출해 PyAV로 로드."""
        self.release()
        stream_url = self._get_youtube_stream(url)
        if stream_url is None:
            return False
        self.source = stream_url
        self.is_stream = True
        return self._open_source()

    def _get_youtube_stream(self, url: str) -> str | None:
        ydl_opts = {
            "format": (
                "best[height<=720][ext=mp4][protocol^=http][vcodec!=none][acodec!=none]/"
                "best[height<=720][protocol^=http][vcodec!=none]/"
                "best[height<=720]/best"
            ),
            "js_runtimes": self._get_js_runtimes(),
            "quiet": True,
            "noplaylist": True,
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                self.source_duration = float(info.get("duration") or 0.0)
                stream_url = info.get("url")
                if stream_url:
                    return stream_url
                for fmt in reversed(info.get("formats", [])):
                    if fmt.get("url") and fmt.get("vcodec") != "none":
                        return fmt["url"]
        except Exception as e:
            print(f"[VideoLoader] 유튜브 로드 실패: {e}")
        return None

    @staticmethod
    def _get_js_runtimes() -> dict:
        node_path = shutil.which("node")
        if node_path:
            return {"node": {"path": node_path}}
        return {"deno": {}}

    def _open_source(self) -> bool:
        if not self.source:
            return False
        with self.lock:
            try:
                self.container = av.open(
                    self.source,
                    options={
                        "reconnect": "1",
                        "reconnect_streamed": "1",
                        "reconnect_delay_max": "2",
                    },
                )
                self.video_stream = next(
                    stream for stream in self.container.streams if stream.type == "video"
                )
                self._init_metadata()
                return True
            except Exception as e:
                print(f"[VideoLoader] 영상 열기 실패: {e}")
                self.release()
                return False

    def _init_metadata(self):
        stream = self.video_stream
        self.width = int(stream.codec_context.width or 0)
        self.height = int(stream.codec_context.height or 0)
        self.fps = self._fraction_to_float(stream.average_rate) or 30.0
        self.total_frames = int(stream.frames or 0)

        duration_sec = 0.0
        if stream.duration and stream.time_base:
            duration_sec = float(stream.duration * stream.time_base)
        elif self.container and self.container.duration:
            duration_sec = self.container.duration / av.time_base
        elif self.total_frames > 0 and self.fps > 0:
            duration_sec = self.total_frames / self.fps
        else:
            duration_sec = self.source_duration

        self.duration = max(1.0, float(duration_sec or 1.0))
        if self.total_frames <= 0 and self.fps > 0:
            self.total_frames = int(self.duration * self.fps)

    @staticmethod
    def _fraction_to_float(value: Fraction | None) -> float:
        if value is None:
            return 0.0
        return float(value) if value.denominator else 0.0

    def get_frame_at(self, frame_idx: int):
        """특정 프레임 인덱스의 이미지를 BGR ndarray로 반환."""
        if self.container is None:
            return None
        frame_idx = self._clamp_frame_idx(frame_idx)
        with self.lock:
            frame = self._read_frame_at(frame_idx)
            if frame is None and self.is_stream:
                self._reopen_source()
                frame = self._read_frame_at(frame_idx)
        return frame

    def get_frame_range(self, start_frame: int, end_frame: int):
        """시작~끝 프레임 구간을 BGR ndarray로 반환하는 제너레이터."""
        if self.container is None:
            return
        start_frame = self._clamp_frame_idx(start_frame)
        if self.total_frames > 0:
            end_frame = min(end_frame, self.total_frames)
        if end_frame <= start_frame:
            return

        with self.lock:
            try:
                self._seek_to_frame(start_frame)
                start_time = start_frame / max(self.fps, 1e-6)
                end_time = end_frame / max(self.fps, 1e-6)
                for packet in self.container.demux(self.video_stream):
                    for frame in packet.decode():
                        frame_time = self._frame_time(frame)
                        if frame_time is not None and frame_time < start_time:
                            continue
                        if frame_time is not None and frame_time >= end_time:
                            return
                        yield frame.to_ndarray(format="bgr24")
            except Exception as e:
                print(f"[VideoLoader] 구간 프레임 읽기 실패: {e}")

    def _read_frame_at(self, frame_idx: int):
        try:
            self._seek_to_frame(frame_idx)
            target_time = frame_idx / max(self.fps, 1e-6)
            for packet in self.container.demux(self.video_stream):
                for frame in packet.decode():
                    frame_time = self._frame_time(frame)
                    if frame_time is not None and frame_time < target_time:
                        continue
                    return frame.to_ndarray(format="bgr24")
        except Exception as e:
            print(f"[VideoLoader] 프레임 읽기 실패: {e}")
        return None

    @staticmethod
    def _frame_time(frame) -> float | None:
        if frame.pts is None or frame.time_base is None:
            return None
        return float(frame.pts * frame.time_base)

    def _seek_to_frame(self, frame_idx: int):
        if self.container is None or self.video_stream is None:
            return
        seconds = frame_idx / max(self.fps, 1e-6)
        timestamp = int(seconds / self.video_stream.time_base)
        self.container.seek(timestamp, stream=self.video_stream, backward=True, any_frame=False)

    def _clamp_frame_idx(self, frame_idx: int) -> int:
        frame_idx = max(0, frame_idx)
        if self.total_frames > 0:
            return min(frame_idx, self.total_frames - 1)
        return frame_idx

    def _reopen_source(self):
        if self.container:
            self.container.close()
        self.container = None
        self.video_stream = None
        self._open_source()

    def release(self):
        with self.lock:
            if self.container:
                self.container.close()
            self.container = None
            self.video_stream = None
            self.source = ""
            self.is_stream = False

    @staticmethod
    def is_youtube_url(text: str) -> bool:
        pattern = r"(youtube\.com/watch|youtu\.be/)"
        return bool(re.search(pattern, text))

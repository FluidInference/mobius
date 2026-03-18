from __future__ import annotations
import sys
import os
import warnings
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore', category=UserWarning, module='coremltools')

import argparse
import queue
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import matplotlib

matplotlib.use("TkAgg")

import librosa
import numpy as np
import sounddevice as sd
import soundfile as sf
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from scipy.signal import medfilt

from torch_inference.ls_eend_runtime import (
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIG,
    LSEENDInferenceEngine,
    StreamingUpdate,
    ensure_mono,
    save_json,
    write_rttm,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Live microphone GUI for raw LS-EEND inference.")
    parser.add_argument("--coreml-model", type=Path, default=None, help="Optional fixed-shape CoreML package to use instead of the PyTorch checkpoint.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent.parent / "artifacts" / "mic_gui")
    parser.add_argument("--device", type=str, default=None, help="Torch device. Defaults to cuda, mps, or cpu.")
    parser.add_argument(
        "--compute-units",
        choices=("all", "cpu_only", "cpu_and_gpu", "cpu_and_ne"),
        default="cpu_only",
        help="CoreML compute units when --coreml-model is used.",
    )
    parser.add_argument("--num-speakers", type=int, default=None, help="Optional number of displayed speaker tracks.")
    parser.add_argument("--input-device", type=str, default=None, help="Sounddevice input device id or name.")
    parser.add_argument("--input-sample-rate", type=int, default=None, help="Optional microphone sample rate override.")
    parser.add_argument("--block-seconds", type=float, default=0.1, help="Audio callback block size.")
    parser.add_argument("--refresh-seconds", type=float, default=0.1, help="Minimum queued audio delta before dispatching a streaming step.")
    parser.add_argument(
        "--analysis-seconds",
        type=float,
        default=60.0,
        help="Deprecated compatibility flag. The true stateful streaming path ignores it.",
    )
    parser.add_argument("--display-seconds", type=float, default=120.0, help="Timeline window length shown in the GUI.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Binary activity threshold.")
    parser.add_argument("--median", type=int, default=11, help="Median filter width for binary activity display.")
    parser.add_argument("--simulate-audio", type=Path, default=None, help="Optional audio file to stream into the GUI instead of the microphone.")
    parser.add_argument("--simulate-speed", type=float, default=1.0, help="Playback speed for --simulate-audio.")
    parser.add_argument("--auto-start", action="store_true", help="Start capture immediately on launch.")
    parser.add_argument("--list-devices", action="store_true", help="Print input devices and exit.")
    return parser


@dataclass
class InferenceUpdate:
    update: StreamingUpdate | None = None
    finalized: bool = False
    error: str | None = None


class MicrophoneAudioSource:
    def __init__(
        self,
        audio_queue: queue.Queue[np.ndarray],
        status_queue: queue.Queue[str],
        input_device: str | None,
        input_sample_rate: int | None,
        target_sample_rate: int,
        block_seconds: float,
    ) -> None:
        self.audio_queue = audio_queue
        self.status_queue = status_queue
        self.input_device = input_device
        device_info = sd.query_devices(input_device, "input")
        self.device_name = device_info["name"]
        self.sample_rate = int(input_sample_rate or target_sample_rate)
        self.blocksize = max(1, int(round(block_seconds * self.sample_rate)))
        self.stream: sd.InputStream | None = None

    def start(self) -> None:
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            blocksize=self.blocksize,
            device=self.input_device,
            channels=1,
            dtype="float32",
            callback=self._callback,
        )
        self.stream.start()
        self.status_queue.put(f"Capturing microphone: {self.device_name} @ {self.sample_rate} Hz")

    def stop(self) -> None:
        if self.stream is None:
            return
        self.stream.stop()
        self.stream.close()
        self.stream = None

    def _callback(self, indata, frames, callback_time, status) -> None:  # type: ignore[no-untyped-def]
        if status:
            self.status_queue.put(f"Audio callback status: {status}")
        self.audio_queue.put(indata[:, 0].copy())


class SimulatedAudioSource:
    def __init__(
        self,
        audio_queue: queue.Queue[np.ndarray],
        status_queue: queue.Queue[str],
        audio_path: Path,
        target_sample_rate: int,
        block_seconds: float,
        speed: float,
    ) -> None:
        self.audio_queue = audio_queue
        self.status_queue = status_queue
        self.audio_path = audio_path
        self.block_seconds = block_seconds
        self.speed = max(speed, 1e-3)
        audio, sample_rate = sf.read(audio_path)
        audio = ensure_mono(audio).astype(np.float32, copy=False)
        if sample_rate != target_sample_rate:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=target_sample_rate).astype(np.float32, copy=False)
            sample_rate = target_sample_rate
        self.audio = audio
        self.sample_rate = int(sample_rate)
        self.device_name = f"Simulated: {audio_path.name}"
        self.blocksize = max(1, int(round(block_seconds * self.sample_rate)))
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self.status_queue.put(f"Streaming simulation: {self.audio_path.name} @ {self.sample_rate} Hz")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None

    def _run(self) -> None:
        for start in range(0, len(self.audio), self.blocksize):
            if self._stop_event.is_set():
                return
            stop = min(len(self.audio), start + self.blocksize)
            self.audio_queue.put(self.audio[start:stop].copy())
            time.sleep(((stop - start) / self.sample_rate) / self.speed)
        self.status_queue.put("Simulation finished.")


class LSEENDMicGUI:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        if args.coreml_model is not None:
            from ls_eend_coreml_runtime import CoreMLLSEENDInferenceEngine

            self.engine = CoreMLLSEENDInferenceEngine(
                coreml_model_path=args.coreml_model,
                config_path=args.config,
                compute_units=args.compute_units,
            )
        else:
            self.engine = LSEENDInferenceEngine(
                checkpoint_path=args.checkpoint,
                config_path=args.config,
                device=args.device,
                actual_num_speakers=args.num_speakers,
            )
        self.output_dir = args.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.root = tk.Tk()
        backend_label = "CoreML" if args.coreml_model is not None else "PyTorch"
        self.root.title(f"LS-EEND Live Microphone GUI ({backend_label})")
        self.root.geometry("1500x920")

        self.audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        self.status_queue: queue.Queue[str] = queue.Queue()
        self.result_queue: queue.Queue[InferenceUpdate] = queue.Queue()
        self.audio_lock = threading.Lock()
        self.pending_audio = np.zeros(0, dtype=np.float32)
        self.total_samples_received = 0
        self.timeline_probabilities = np.zeros((0, 0), dtype=np.float32)
        self.preview_probabilities = np.zeros((0, 0), dtype=np.float32)
        self.preview_start_frame = 0
        self.latest_result = None
        self.session = None
        self.inference_thread: threading.Thread | None = None
        self.inference_in_flight = False
        self.finalize_requested = False
        self.source = None
        self.sample_rate = 0
        self.session_index = 0
        self.display_order: list[int] = []
        self.track_labels: dict[int, tk.StringVar] = {}
        self.swap_left_var = tk.StringVar()
        self.swap_right_var = tk.StringVar()

        self.status_var = tk.StringVar(value="Loading model...")
        self.source_var = tk.StringVar(value="Not started")
        self.buffer_var = tk.StringVar(value="Buffered: 0.0 s")
        self.inference_var = tk.StringVar(value="Inference: idle")
        self.window_seconds_var = tk.DoubleVar(value=float(args.display_seconds))
        self.threshold_var = tk.DoubleVar(value=float(args.threshold))
        self.median_var = tk.IntVar(value=int(args.median))

        self._build_ui()
        self.status_var.set("Ready.")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(50, self._poll_audio)
        self.root.after(75, self._poll_status)
        self.root.after(100, self._poll_results)

        if args.auto_start or args.simulate_audio is not None:
            self.root.after(150, self.start_capture)

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        container = ttk.Frame(self.root, padding=8)
        container.grid(row=0, column=0, sticky="nsew")
        container.columnconfigure(0, weight=3)
        container.columnconfigure(1, weight=1)
        container.rowconfigure(0, weight=1)

        plot_frame = ttk.Frame(container)
        plot_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        plot_frame.columnconfigure(0, weight=1)
        plot_frame.rowconfigure(0, weight=1)

        self.figure = Figure(figsize=(12, 8), dpi=100, constrained_layout=True)
        self.binary_axis = self.figure.add_subplot(211)
        self.prob_axis = self.figure.add_subplot(212, sharex=self.binary_axis)
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self._draw_placeholder()

        side = ttk.Frame(container)
        side.grid(row=0, column=1, sticky="nsew")
        side.columnconfigure(0, weight=1)

        control_frame = ttk.LabelFrame(side, text="Session", padding=8)
        control_frame.grid(row=0, column=0, sticky="ew")
        for column in range(2):
            control_frame.columnconfigure(column, weight=1)

        ttk.Button(control_frame, text="Start", command=self.start_capture).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        ttk.Button(control_frame, text="Stop", command=self.stop_capture).grid(row=0, column=1, sticky="ew", padx=(4, 0))
        ttk.Button(control_frame, text="Reset Timeline", command=self.reset_timeline).grid(row=1, column=0, sticky="ew", pady=(6, 0), padx=(0, 4))
        ttk.Button(control_frame, text="Save RTTM", command=self.save_current_rttm).grid(row=1, column=1, sticky="ew", pady=(6, 0), padx=(4, 0))
        ttk.Button(control_frame, text="Save Heatmap", command=self.save_current_heatmap).grid(row=2, column=0, sticky="ew", pady=(6, 0), padx=(0, 4))
        ttk.Button(control_frame, text="Save Session", command=self.save_session_metadata).grid(row=2, column=1, sticky="ew", pady=(6, 0), padx=(4, 0))

        status_frame = ttk.LabelFrame(side, text="Status", padding=8)
        status_frame.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        ttk.Label(status_frame, textvariable=self.status_var, wraplength=340, justify="left").grid(row=0, column=0, sticky="w")
        ttk.Label(status_frame, textvariable=self.source_var, wraplength=340, justify="left").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Label(status_frame, textvariable=self.buffer_var, wraplength=340, justify="left").grid(row=2, column=0, sticky="w", pady=(6, 0))
        ttk.Label(status_frame, textvariable=self.inference_var, wraplength=340, justify="left").grid(row=3, column=0, sticky="w", pady=(6, 0))

        display_frame = ttk.LabelFrame(side, text="Display", padding=8)
        display_frame.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        ttk.Label(display_frame, text="Window (s)").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(display_frame, from_=10, to=3600, increment=10, textvariable=self.window_seconds_var, width=10, command=self.refresh_plot).grid(row=0, column=1, sticky="w")
        ttk.Label(display_frame, text="Threshold").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Spinbox(display_frame, from_=0.05, to=0.95, increment=0.05, textvariable=self.threshold_var, width=10, command=self.refresh_plot).grid(row=1, column=1, sticky="w", pady=(6, 0))
        ttk.Label(display_frame, text="Median").grid(row=2, column=0, sticky="w", pady=(6, 0))
        ttk.Spinbox(display_frame, from_=1, to=51, increment=2, textvariable=self.median_var, width=10, command=self.refresh_plot).grid(row=2, column=1, sticky="w", pady=(6, 0))

        swap_frame = ttk.LabelFrame(side, text="Swap Rows", padding=8)
        swap_frame.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        ttk.Label(swap_frame, text="Row A").grid(row=0, column=0, sticky="w")
        ttk.Label(swap_frame, text="Row B").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.swap_left = ttk.Combobox(swap_frame, state="readonly", textvariable=self.swap_left_var)
        self.swap_right = ttk.Combobox(swap_frame, state="readonly", textvariable=self.swap_right_var)
        self.swap_left.grid(row=0, column=1, sticky="ew")
        self.swap_right.grid(row=1, column=1, sticky="ew", pady=(6, 0))
        ttk.Button(swap_frame, text="Swap", command=self.swap_selected_rows).grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Button(swap_frame, text="Reset Order", command=self.reset_order).grid(row=3, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        swap_frame.columnconfigure(1, weight=1)

        self.speaker_frame = ttk.LabelFrame(side, text="Speakers", padding=8)
        self.speaker_frame.grid(row=4, column=0, sticky="nsew", pady=(8, 0))
        side.rowconfigure(4, weight=1)

    def _draw_placeholder(self) -> None:
        for axis, title in ((self.binary_axis, "Binary Activity"), (self.prob_axis, "Speaker Probability")):
            axis.clear()
            axis.set_title(title)
            axis.text(0.5, 0.5, "No inference yet", ha="center", va="center", transform=axis.transAxes)
            axis.set_yticks([])
        self.prob_axis.set_xlabel("Time (seconds)")
        self.canvas.draw_idle()

    def _create_source(self):
        if self.args.simulate_audio is not None:
            return SimulatedAudioSource(
                audio_queue=self.audio_queue,
                status_queue=self.status_queue,
                audio_path=self.args.simulate_audio,
                target_sample_rate=self.engine.target_sample_rate,
                block_seconds=self.args.block_seconds,
                speed=self.args.simulate_speed,
            )
        return MicrophoneAudioSource(
            audio_queue=self.audio_queue,
            status_queue=self.status_queue,
            input_device=self.args.input_device,
            input_sample_rate=self.args.input_sample_rate,
            target_sample_rate=self.engine.target_sample_rate,
            block_seconds=self.args.block_seconds,
        )

    def start_capture(self) -> None:
        if self.source is not None:
            self.status_var.set("Capture is already running.")
            return
        if self.session is not None and self.session.finalized:
            self.reset_timeline()
        try:
            self.source = self._create_source()
            self.sample_rate = int(self.source.sample_rate)
            self.session = self.engine.create_session(self.sample_rate)
            self.source.start()
        except Exception as exc:
            self.source = None
            self.session = None
            messagebox.showerror("Audio Source Error", str(exc))
            return
        self.finalize_requested = False
        self.source_var.set(f"Source: {self.source.device_name} @ {self.sample_rate} Hz")
        self.status_var.set(
            f"Capture started. Stateful streaming latency is about {self.engine.streaming_latency_seconds:.2f} s."
        )
        self._update_buffer_label()

    def stop_capture(self) -> None:
        if self.source is None and self.session is None:
            return
        if self.source is not None:
            self.source.stop()
            self.source = None
        self.finalize_requested = True
        self.status_var.set("Capture stopped. Flushing the delayed tail...")
        self._maybe_schedule_inference(force=True)

    def reset_timeline(self) -> None:
        if self.source is not None:
            self.source.stop()
            self.source = None
        with self.audio_lock:
            self.pending_audio = np.zeros(0, dtype=np.float32)
            self.total_samples_received = 0
        self.latest_result = None
        self.session = None
        self.timeline_probabilities = np.zeros((0, 0), dtype=np.float32)
        self.preview_probabilities = np.zeros((0, 0), dtype=np.float32)
        self.preview_start_frame = 0
        self.finalize_requested = False
        self.display_order = []
        self._rebuild_speaker_controls()
        self.buffer_var.set("Buffered: 0.0 s")
        self.inference_var.set("Inference: idle")
        self.status_var.set("Timeline reset.")
        self._draw_placeholder()

    def _poll_audio(self) -> None:
        received = False
        while True:
            try:
                chunk = self.audio_queue.get_nowait()
            except queue.Empty:
                break
            received = True
            with self.audio_lock:
                self.pending_audio = np.concatenate([self.pending_audio, chunk], axis=0)
                self.total_samples_received += len(chunk)
            self._update_buffer_label()
        if received:
            self._maybe_schedule_inference(force=False)
        self.root.after(50, self._poll_audio)

    def _poll_status(self) -> None:
        message = None
        while True:
            try:
                message = self.status_queue.get_nowait()
            except queue.Empty:
                break
        if message is not None:
            if message == "Simulation finished.":
                self.stop_capture()
            else:
                self.status_var.set(message)
        self.root.after(100, self._poll_status)

    def _poll_results(self) -> None:
        updated = False
        while True:
            try:
                update = self.result_queue.get_nowait()
            except queue.Empty:
                break
            self.inference_in_flight = False
            if update.error is not None:
                self.status_var.set(update.error)
            else:
                if update.update is not None:
                    self.latest_result = update.update
                    self._merge_result_into_timeline(update.update.start_frame, update.update.probabilities)
                    self._set_preview(update.update.preview_start_frame, update.update.preview_probabilities)
                    self.inference_var.set(
                        f"Inference: {update.update.total_emitted_frames} committed + {update.update.preview_probabilities.shape[0]} preview frames"
                    )
                    self._ensure_track_state(self._combined_probabilities().shape[1])
                    self.refresh_plot()
                    self._update_buffer_label()
                    updated = True
                if update.finalized:
                    self.preview_probabilities = np.zeros((0, self.timeline_probabilities.shape[1] if self.timeline_probabilities.ndim == 2 else 0), dtype=np.float32)
                    self.preview_start_frame = self.timeline_probabilities.shape[0]
                    self._update_buffer_label()
                    self.status_var.set("Streaming tail flushed.")
        if self.session is not None and not self.inference_in_flight:
            with self.audio_lock:
                has_pending_audio = self.pending_audio.size > 0
            if has_pending_audio:
                self._maybe_schedule_inference(force=True)
            elif self.finalize_requested and not self.session.finalized:
                self._start_background_step(finalize=True)
        elif updated:
            self.status_var.set("Inference updated.")
        self.root.after(100, self._poll_results)

    def _maybe_schedule_inference(self, force: bool) -> None:
        if self.session is None or self.sample_rate <= 0:
            return
        minimum_delta = max(1, int(round(self.args.refresh_seconds * self.sample_rate)))
        if self.inference_in_flight:
            return
        with self.audio_lock:
            queued_samples = len(self.pending_audio)
        if queued_samples == 0:
            if force and self.finalize_requested and not self.session.finalized:
                self._start_background_step(finalize=True)
            return
        if not force and queued_samples < minimum_delta:
            return
        self._start_background_step(finalize=False)

    def _start_background_step(self, finalize: bool) -> None:
        self.inference_in_flight = True
        self.inference_var.set("Inference: running...")
        self.inference_thread = threading.Thread(target=self._run_inference, args=(finalize,), daemon=True)
        self.inference_thread.start()

    def _run_inference(self, finalize: bool) -> None:
        try:
            if self.session is None:
                raise RuntimeError("Streaming session is not initialized.")
            if finalize:
                update = self.session.finalize()
                self.result_queue.put(InferenceUpdate(update=update, finalized=True))
                return
            with self.audio_lock:
                chunk = self.pending_audio.copy()
                self.pending_audio = np.zeros(0, dtype=np.float32)
            update = self.session.push_audio(chunk)
            if update is not None and self.args.num_speakers is not None:
                clip = max(0, min(int(self.args.num_speakers), update.probabilities.shape[1]))
                update.logits = update.logits[:, :clip]
                update.probabilities = update.probabilities[:, :clip]
            self.result_queue.put(InferenceUpdate(update=update))
        except Exception as exc:
            self.result_queue.put(InferenceUpdate(error=f"Inference failed: {exc}"))

    def _merge_result_into_timeline(self, start_frame: int, probabilities: np.ndarray) -> None:
        if probabilities.size == 0:
            return
        end_frame = start_frame + probabilities.shape[0]
        current_frames, current_tracks = self.timeline_probabilities.shape if self.timeline_probabilities.size else (0, 0)
        target_tracks = max(current_tracks, probabilities.shape[1])
        if current_frames < end_frame or current_tracks < target_tracks:
            expanded = np.zeros((max(current_frames, end_frame), target_tracks), dtype=np.float32)
            if current_frames > 0 and current_tracks > 0:
                expanded[:current_frames, :current_tracks] = self.timeline_probabilities
            self.timeline_probabilities = expanded
        self.timeline_probabilities[start_frame:end_frame, : probabilities.shape[1]] = probabilities

    def _set_preview(self, start_frame: int, probabilities: np.ndarray) -> None:
        self.preview_start_frame = int(start_frame)
        self.preview_probabilities = probabilities.astype(np.float32, copy=False)

    def _update_buffer_label(self) -> None:
        received_seconds = self.total_samples_received / max(self.sample_rate, 1)
        committed_seconds = self.timeline_probabilities.shape[0] / self.engine.model_frame_hz
        preview_seconds = (
            (self.preview_start_frame + self.preview_probabilities.shape[0]) / self.engine.model_frame_hz
            if self.preview_probabilities.size
            else committed_seconds
        )
        self.buffer_var.set(
            f"Buffered: {received_seconds:.1f} s received, {committed_seconds:.1f} s committed, {preview_seconds:.1f} s incl preview"
        )

    def _selected_probabilities(self) -> np.ndarray:
        probabilities = self._combined_probabilities()
        if probabilities.size == 0:
            return probabilities
        if not self.display_order:
            return probabilities
        return probabilities[:, self.display_order]

    def _selected_preview_range(self) -> tuple[int | None, int | None]:
        if self.preview_probabilities.size == 0:
            return None, None
        return self.preview_start_frame, self.preview_start_frame + self.preview_probabilities.shape[0]

    def _combined_probabilities(self) -> np.ndarray:
        current_frames, current_tracks = self.timeline_probabilities.shape if self.timeline_probabilities.size else (0, 0)
        preview_frames, preview_tracks = self.preview_probabilities.shape if self.preview_probabilities.size else (0, 0)
        total_tracks = max(current_tracks, preview_tracks)
        total_frames = max(current_frames, self.preview_start_frame + preview_frames)
        if total_frames == 0 or total_tracks == 0:
            return np.zeros((0, 0), dtype=np.float32)
        combined = np.zeros((total_frames, total_tracks), dtype=np.float32)
        if current_frames and current_tracks:
            combined[:current_frames, :current_tracks] = self.timeline_probabilities
        if preview_frames and preview_tracks:
            combined[self.preview_start_frame : self.preview_start_frame + preview_frames, :preview_tracks] = self.preview_probabilities
        return combined

    def _speaker_labels_for_display(self) -> list[str]:
        labels = []
        for track_index in self.display_order:
            label_var = self.track_labels.get(track_index)
            label = label_var.get().strip() if label_var is not None else ""
            labels.append(label or f"Speaker {track_index + 1}")
        return labels

    def _ensure_track_state(self, track_count: int) -> None:
        if track_count <= 0:
            return
        if len(self.display_order) != track_count or set(self.display_order) != set(range(track_count)):
            self.display_order = list(range(track_count))
        for track_index in range(track_count):
            if track_index not in self.track_labels:
                self.track_labels[track_index] = tk.StringVar(value=f"Speaker {track_index + 1}")
        self._rebuild_speaker_controls()

    def _rebuild_speaker_controls(self) -> None:
        for child in self.speaker_frame.winfo_children():
            child.destroy()
        if not self.display_order:
            ttk.Label(self.speaker_frame, text="Run inference to populate speaker tracks.").grid(row=0, column=0, sticky="w")
            self.swap_left["values"] = ()
            self.swap_right["values"] = ()
            self.swap_left_var.set("")
            self.swap_right_var.set("")
            return
        row_options = [str(index + 1) for index in range(len(self.display_order))]
        self.swap_left["values"] = row_options
        self.swap_right["values"] = row_options
        if not self.swap_left_var.get():
            self.swap_left_var.set(row_options[0])
        if len(row_options) > 1 and not self.swap_right_var.get():
            self.swap_right_var.set(row_options[1])
        elif row_options:
            self.swap_right_var.set(row_options[0])

        for row_index, track_index in enumerate(self.display_order):
            ttk.Label(self.speaker_frame, text=f"Row {row_index + 1}").grid(row=row_index, column=0, sticky="w")
            ttk.Entry(self.speaker_frame, textvariable=self.track_labels[track_index], width=18).grid(row=row_index, column=1, sticky="ew", padx=(6, 6))
            ttk.Button(self.speaker_frame, text="Up", command=lambda idx=row_index: self.move_row(idx, -1)).grid(row=row_index, column=2, sticky="ew")
            ttk.Button(self.speaker_frame, text="Down", command=lambda idx=row_index: self.move_row(idx, 1)).grid(row=row_index, column=3, sticky="ew", padx=(4, 0))
        self.speaker_frame.columnconfigure(1, weight=1)

    def move_row(self, row_index: int, delta: int) -> None:
        swap_index = row_index + delta
        if swap_index < 0 or swap_index >= len(self.display_order):
            return
        self.display_order[row_index], self.display_order[swap_index] = self.display_order[swap_index], self.display_order[row_index]
        self._rebuild_speaker_controls()
        self.refresh_plot()

    def swap_selected_rows(self) -> None:
        if not self.display_order:
            return
        try:
            left = int(self.swap_left_var.get()) - 1
            right = int(self.swap_right_var.get()) - 1
        except ValueError:
            return
        if left < 0 or right < 0 or left >= len(self.display_order) or right >= len(self.display_order) or left == right:
            return
        self.display_order[left], self.display_order[right] = self.display_order[right], self.display_order[left]
        self._rebuild_speaker_controls()
        self.refresh_plot()

    def reset_order(self) -> None:
        if not self.display_order:
            return
        self.display_order = list(range(len(self.display_order)))
        self._rebuild_speaker_controls()
        self.refresh_plot()

    def _current_binary(self) -> np.ndarray:
        probabilities = self._selected_probabilities()
        if probabilities.size == 0:
            return probabilities
        binary = (probabilities >= float(self.threshold_var.get())).astype(np.float32)
        median_width = max(1, int(self.median_var.get()))
        if median_width > 1:
            if median_width % 2 == 0:
                median_width += 1
            binary = medfilt(binary, kernel_size=(median_width, 1)).astype(np.float32)
        return binary

    def refresh_plot(self) -> None:
        if self.timeline_probabilities.size == 0:
            self._draw_placeholder()
            return
        probabilities = self._selected_probabilities()
        if probabilities.size == 0:
            self._draw_placeholder()
            return
        binary = self._current_binary()
        frame_hz = self.engine.model_frame_hz
        window_frames = max(1, int(round(float(self.window_seconds_var.get()) * frame_hz)))
        start_frame = max(0, probabilities.shape[0] - window_frames)
        shown_probs = probabilities[start_frame:]
        shown_binary = binary[start_frame:]
        start_seconds = start_frame / frame_hz
        end_seconds = probabilities.shape[0] / frame_hz
        speaker_labels = self._speaker_labels_for_display()
        preview_start_frame, preview_end_frame = self._selected_preview_range()
        preview_start_seconds = None if preview_start_frame is None else preview_start_frame / frame_hz
        preview_end_seconds = None if preview_end_frame is None else preview_end_frame / frame_hz

        self.binary_axis.clear()
        self.binary_axis.imshow(
            shown_binary.T,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            extent=[start_seconds, end_seconds, -0.5, shown_binary.shape[1] - 0.5],
            cmap="Greys",
            vmin=0.0,
            vmax=1.0,
        )
        self.binary_axis.set_title("Binary Activity")
        self.binary_axis.set_yticks(range(len(speaker_labels)))
        self.binary_axis.set_yticklabels(speaker_labels)
        if preview_start_seconds is not None and preview_end_seconds is not None and preview_end_seconds > start_seconds:
            self.binary_axis.axvspan(
                max(preview_start_seconds, start_seconds),
                min(preview_end_seconds, end_seconds),
                color="orange",
                alpha=0.08,
            )
            self.binary_axis.axvline(preview_start_seconds, color="orange", linestyle="--", linewidth=1.0)

        self.prob_axis.clear()
        image = self.prob_axis.imshow(
            shown_probs.T,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            extent=[start_seconds, end_seconds, -0.5, shown_probs.shape[1] - 0.5],
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
        )
        self.prob_axis.set_title("Speaker Probability")
        self.prob_axis.set_yticks(range(len(speaker_labels)))
        self.prob_axis.set_yticklabels(speaker_labels)
        self.prob_axis.set_xlabel("Time (seconds)")
        if preview_start_seconds is not None and preview_end_seconds is not None and preview_end_seconds > start_seconds:
            self.prob_axis.axvspan(
                max(preview_start_seconds, start_seconds),
                min(preview_end_seconds, end_seconds),
                color="orange",
                alpha=0.08,
            )
            self.prob_axis.axvline(preview_start_seconds, color="orange", linestyle="--", linewidth=1.0)
        if len(self.figure.axes) < 3:
            self.figure.colorbar(image, ax=self.prob_axis, fraction=0.02, pad=0.01)
        self.canvas.draw_idle()

    def _next_output_stem(self) -> str:
        self.session_index += 1
        return time.strftime(f"mic_session_%Y%m%d_%H%M%S_{self.session_index:02d}")

    def save_current_rttm(self) -> None:
        if self.timeline_probabilities.size == 0:
            messagebox.showinfo("Save RTTM", "No inference available yet.")
            return
        output_path = self.output_dir / f"{self._next_output_stem()}.rttm"
        write_rttm(
            recording_id=output_path.stem,
            binary_prediction=self._current_binary(),
            output_path=output_path,
            frame_rate=self.engine.model_frame_hz,
            speaker_labels=self._speaker_labels_for_display(),
        )
        self.status_var.set(f"Saved RTTM: {output_path.name}")

    def save_current_heatmap(self) -> None:
        if self.timeline_probabilities.size == 0:
            messagebox.showinfo("Save Heatmap", "No inference available yet.")
            return
        path = filedialog.asksaveasfilename(
            title="Save heatmap",
            defaultextension=".png",
            initialdir=str(self.output_dir),
            initialfile=f"{self._next_output_stem()}.png",
            filetypes=[("PNG image", "*.png")],
        )
        if not path:
            return
        self.figure.savefig(path, dpi=200)
        self.status_var.set(f"Saved heatmap: {Path(path).name}")

    def save_session_metadata(self) -> None:
        if self.timeline_probabilities.size == 0:
            messagebox.showinfo("Save Session", "No inference available yet.")
            return
        output_path = self.output_dir / f"{self._next_output_stem()}.json"
        payload = {
            "backend": "coreml" if self.args.coreml_model is not None else "pytorch",
            "coreml_model": None if self.args.coreml_model is None else str(self.args.coreml_model),
            "checkpoint": str(self.args.checkpoint),
            "config": str(self.args.config),
            "source": self.source_var.get(),
            "duration_seconds": float(self.timeline_probabilities.shape[0] / self.engine.model_frame_hz),
            "preview_duration_seconds": float(self.preview_probabilities.shape[0] / self.engine.model_frame_hz),
            "frame_hz": float(self.engine.model_frame_hz),
            "display_order": self.display_order,
            "speaker_labels": self._speaker_labels_for_display(),
            "threshold": float(self.threshold_var.get()),
            "median": int(self.median_var.get()),
            "streaming_latency_seconds": float(self.engine.streaming_latency_seconds),
        }
        save_json(payload, output_path)
        self.status_var.set(f"Saved session metadata: {output_path.name}")

    def on_close(self) -> None:
        self.stop_capture()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def list_devices() -> None:
    devices = sd.query_devices()
    for index, device in enumerate(devices):
        if device["max_input_channels"] <= 0:
            continue
        print(f"{index}: {device['name']} | input_channels={device['max_input_channels']} | default_samplerate={device['default_samplerate']}")


def main() -> None:
    args = build_parser().parse_args()
    if args.list_devices:
        list_devices()
        return
    app = LSEENDMicGUI(args)
    app.run()


if __name__ == "__main__":
    main()

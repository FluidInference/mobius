#!/usr/bin/env python3
"""Compare Parakeet TDT v3 Torch vs CoreML components on a fixed 15s window.

Writes numeric diffs to the specified output directory (metadata.json) and
saves plots under a repo-tracked directory: plots/<script-name>/.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import coremltools as ct
import numpy as np
import soundfile as sf
import torch
import typer

import nemo.collections.asr as nemo_asr

# Optional plotting
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False


@dataclass
class ValidationSettings:
    audio_path: Optional[Path]
    seconds: float
    seed: Optional[int]
    rtol: float
    atol: float


def _compute_length(seconds: float, sample_rate: int) -> int:
    return int(round(seconds * sample_rate))


def _prepare_audio(
    validation_audio: Optional[Path],
    sample_rate: int,
    max_samples: int,
    seed: Optional[int],
) -> torch.Tensor:
    if validation_audio is None:
        if seed is not None:
            torch.manual_seed(seed)
        return torch.randn(1, max_samples, dtype=torch.float32)

    data, sr = sf.read(str(validation_audio), dtype="float32")
    if sr != sample_rate:
        raise typer.BadParameter(
            f"Validation audio sample rate {sr} does not match model rate {sample_rate}"
        )
    if data.ndim > 1:
        data = data[:, 0]
    if data.size == 0:
        raise typer.BadParameter("Validation audio is empty")
    if data.size < max_samples:
        data = np.pad(data, (0, max_samples - data.size))
    elif data.size > max_samples:
        data = data[:max_samples]
    return torch.from_numpy(data).unsqueeze(0).to(dtype=torch.float32)


def _np(x: torch.Tensor, dtype=None) -> np.ndarray:
    arr = x.detach().cpu().numpy()
    if dtype is not None:
        return arr.astype(dtype, copy=False)
    return arr


def _to_t(x) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    elif isinstance(x, np.ndarray):
        # Ensure a separate tensor (avoid shared memory weirdness)
        return torch.from_numpy(np.array(x, copy=True))
    else:
        return torch.tensor(x)


def _max_diffs(a, b, rtol: float, atol: float) -> Tuple[float, float, bool]:
    # Use NumPy for comparisons to avoid invoking the PyTorch C-API in contexts
    # where the GIL may not be held (which can trigger PyEval_SaveThread errors).
    na = np.array(a, dtype=np.float32, copy=True)
    nb = np.array(b, dtype=np.float32, copy=True)
    if na.size == 0:
        return 0.0, 0.0, True
    diff = np.abs(na - nb)
    max_abs = float(diff.max())
    denom = np.maximum(np.abs(na), np.abs(nb))
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(denom == 0.0, 0.0, diff / denom)
    max_rel = float(rel.max())
    ok = bool(np.allclose(na, nb, rtol=rtol, atol=atol))
    return max_abs, max_rel, ok


def _plot_line(x_ref: np.ndarray, x_ml: np.ndarray, title: str, path: Path, also_delta: bool = False):
    if not HAS_MPL:
        return None
    if also_delta:
        fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
        axes[0].plot(x_ref, label="torch", linewidth=1)
        axes[0].plot(x_ml, label="coreml", linewidth=1, alpha=0.8)
        axes[0].set_title(title)
        axes[0].legend()
        delta = np.asarray(x_ref) - np.asarray(x_ml)
        axes[1].plot(delta, color="C3", linewidth=1)
        axes[1].set_title("Delta (torch - coreml)")
        axes[1].set_xlabel("time/step")
        plt.tight_layout()
        plt.savefig(path)
        plt.close(fig)
    else:
        plt.figure(figsize=(8, 3))
        plt.plot(x_ref, label="torch", linewidth=1)
        plt.plot(x_ml, label="coreml", linewidth=1, alpha=0.8)
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
    return str(path.name)


def _plot_image(img: np.ndarray, title: str, path: Path, vmin=None, vmax=None):
    if not HAS_MPL:
        return None
    plt.figure(figsize=(6, 4))
    plt.imshow(img, aspect='auto', origin='lower', interpolation='nearest', vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.colorbar(shrink=0.8)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return str(path.name)


def _plot_mel_composite(
    mel_torch: np.ndarray,
    mel_coreml: np.ndarray,
    path: Path,
    vmin=None,
    vmax=None,
):
    """Create a single PNG with mel torch, mel coreml, abs diff heatmap, and mean-over-time curves with delta."""
    if not HAS_MPL:
        return None
    mel_torch = np.asarray(mel_torch)
    mel_coreml = np.asarray(mel_coreml)
    absdiff = np.abs(mel_torch - mel_coreml)
    mean_t = mel_torch.mean(axis=0)
    mean_c = mel_coreml.mean(axis=0)
    delta = mean_t - mean_c

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(mel_torch, aspect='auto', origin='lower', interpolation='nearest', vmin=vmin, vmax=vmax)
    ax1.set_title("Mel (Torch)")
    fig.colorbar(im1, ax=ax1, shrink=0.8)

    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(mel_coreml, aspect='auto', origin='lower', interpolation='nearest', vmin=vmin, vmax=vmax)
    ax2.set_title("Mel (CoreML)")
    fig.colorbar(im2, ax=ax2, shrink=0.8)

    ax3 = fig.add_subplot(gs[1, 0])
    im3 = ax3.imshow(absdiff, aspect='auto', origin='lower', interpolation='nearest')
    ax3.set_title("Mel |diff|")
    fig.colorbar(im3, ax=ax3, shrink=0.8)

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(mean_t, label="torch", linewidth=1)
    ax4.plot(mean_c, label="coreml", linewidth=1, alpha=0.8)
    ax4.plot(delta, label="delta", linewidth=1, color="C3")
    ax4.set_title("Mel mean over time + delta")
    ax4.legend()

    plt.tight_layout()
    plt.savefig(path)
    plt.close(fig)
    return str(path.name)


def _plot_latency_bars(
    labels,
    torch_means,
    torch_stds,
    coreml_means,
    coreml_stds,
    path: Path,
):
    if not HAS_MPL:
        return None
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    b1 = ax.bar(x - width/2, torch_means, width, yerr=torch_stds, label="torch", color="C0", alpha=0.9)
    b2 = ax.bar(x + width/2, coreml_means, width, yerr=coreml_stds, label="coreml", color="C1", alpha=0.9)
    ax.set_xticks(x, labels, rotation=15)
    ax.set_ylabel("latency (ms)")
    ax.set_title("Component latency (15s window inputs)")
    ax.legend()
    # Add value labels on bars
    def _annotate(bars):
        for bar in bars:
            h = bar.get_height()
            if np.isnan(h):
                continue
            ax.annotate(f"{h:.0f}",
                        xy=(bar.get_x() + bar.get_width()/2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
    _annotate(b1)
    _annotate(b2)
    plt.tight_layout()
    plt.savefig(path)
    plt.close(fig)
    return str(path.name)


def _plot_speedup_bars(labels, torch_means, coreml_means, path: Path):
    if not HAS_MPL:
        return None
    speedup = []
    for t, c in zip(torch_means, coreml_means):
        if c and c > 0:
            speedup.append(float(t) / float(c))
        else:
            speedup.append(np.nan)
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(x, speedup, color="C2")
    ax.set_xticks(x, labels, rotation=15)
    ax.set_ylabel("torch/coreml speedup")
    ax.set_title("CoreML speedup vs Torch (higher is better)")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    # Add value labels
    for bar in bars:
        h = bar.get_height()
        if np.isnan(h):
            continue
        ax.annotate(f"{h:.2f}",
                    xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    plt.savefig(path)
    plt.close(fig)
    return str(path.name)



app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


@app.command()
def compare(
    output_dir: Path = typer.Option(Path("parakeet_coreml"), help="Directory containing mlpackages + metadata.json"),
    nemo_path: Optional[Path] = typer.Option(None, "--nemo-path", exists=True, resolve_path=True, help="Path to .nemo checkpoint"),
    model_id: str = typer.Option("nvidia/parakeet-tdt-0.6b-v3", "--model-id", help="HF model id if --nemo-path omitted"),
    validation_audio: Optional[Path] = typer.Option(None, exists=True, resolve_path=True, help="15s, 16kHz wav for validation (defaults to audio/yc_first_minute_16k_15s.wav if present)"),
    seed: Optional[int] = typer.Option(None, help="Random seed for synthetic input when audio is not provided"),
    rtol: float = typer.Option(1e-3, help="Relative tolerance for comparisons"),
    atol: float = typer.Option(1e-4, help="Absolute tolerance for comparisons"),
    runs: int = typer.Option(10, help="Timed runs per model for latency measurement"),
    warmup: int = typer.Option(3, help="Warmup runs before timing (compilation, caches)"),
    symbol_steps: int = typer.Option(
        32,
        help="Number of sequential decoder steps to validate with streaming U=1 inputs",
    ),
) -> None:
    """Run Torch vs CoreML comparisons and update metadata.json with plots and diffs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if symbol_steps < 1:
        raise typer.BadParameter("symbol_steps must be >= 1")

    meta_path = output_dir / "metadata.json"
    exported_meta: Dict[str, object] = {}
    if meta_path.exists():
        try:
            exported_meta = json.loads(meta_path.read_text())
        except Exception:
            exported_meta = {}
    exported_max_u = int(exported_meta.get("max_symbol_steps", 1))
    if exported_max_u != 1:
        typer.echo(
            f"Note: CoreML export reports max_symbol_steps={exported_max_u}; "
            "comparison still drives decoder step-wise with U=1 inputs."
        )
    if nemo_path is not None:
        typer.echo(f"Loading NeMo model from {nemo_path}…")
        asr_model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(str(nemo_path), map_location="cpu")
    else:
        typer.echo(f"Downloading NeMo model via {model_id}…")
        asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(model_id, map_location="cpu")
    asr_model.eval()

    sample_rate = int(asr_model.cfg.preprocessor.sample_rate)
    max_samples = _compute_length(15.0, sample_rate)
    default_audio = (Path(__file__).parent / "audio" / "yc_first_minute_16k_15s.wav").resolve()
    chosen_audio = validation_audio if validation_audio is not None else (default_audio if default_audio.exists() else None)
    if chosen_audio is not None and validation_audio is None:
        typer.echo(f"Using default validation audio: {chosen_audio}")
        
    audio_tensor = _prepare_audio(chosen_audio, sample_rate, max_samples, seed)
    audio_length = torch.tensor([max_samples], dtype=torch.int32)

    asr_model.decoder._rnnt_export = True
    # Disable fused loss/WER computation for simpler joint inference
    asr_model.joint.set_fuse_loss_wer(False)
    # Important: ensure the joint returns raw logits (not log-softmax)
    # RNNTJoint applies log_softmax on CPU by default when `log_softmax is None`.
    # Our exported CoreML joint emits pre-softmax logits, so make the Torch
    # reference do the same to avoid systematic offsets in comparisons/plots.
    try:
        # Some versions expose this as a plain attribute
        asr_model.joint.log_softmax = False
    except Exception:
        pass

    # Generate reference outputs directly from NeMo model components
    with torch.inference_mode():
        # Preprocessor - direct NeMo call
        mel_ref, mel_length_ref = asr_model.preprocessor(
            input_signal=audio_tensor,
            length=audio_length.to(dtype=torch.long)
        )
        mel_length_ref = mel_length_ref.to(dtype=torch.int32)

        # Encoder - direct NeMo call
        encoder_ref, encoder_length_ref = asr_model.encoder(
            audio_signal=mel_ref,
            length=mel_length_ref.to(dtype=torch.long)
        )
        encoder_length_ref = encoder_length_ref.to(dtype=torch.int32)

    vocab_size = int(asr_model.tokenizer.vocab_size)
    num_extra = int(asr_model.joint.num_extra_outputs)
    decoder_hidden = int(asr_model.decoder.pred_hidden)
    decoder_layers = int(asr_model.decoder.pred_rnn_layers)
    blank_id = int(asr_model.decoder.blank_idx)

    blank_targets = torch.tensor([[blank_id]], dtype=torch.int32)
    blank_target_lengths = torch.tensor([1], dtype=torch.int32)
    blank_targets_long = blank_targets.to(dtype=torch.long)
    blank_target_lengths_long = blank_target_lengths.to(dtype=torch.long)

    def _decoder_rollout_torch(num_steps: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        outputs = []
        h_state = torch.zeros(decoder_layers, 1, decoder_hidden, dtype=torch.float32)
        c_state = torch.zeros(decoder_layers, 1, decoder_hidden, dtype=torch.float32)
        state = [h_state, c_state]
        with torch.inference_mode():
            for _ in range(num_steps):
                y, _, new_state = asr_model.decoder(
                    targets=blank_targets_long,
                    target_length=blank_target_lengths_long,
                    states=state,
                )
                outputs.append(y.detach())
                state = [new_state[0].detach(), new_state[1].detach()]
        if outputs:
            decoder_seq = torch.cat(outputs, dim=-1)
        else:
            decoder_seq = torch.zeros(1, decoder_hidden, 0, dtype=torch.float32)
        return decoder_seq, state[0], state[1]

    decoder_ref, h_ref, c_ref = _decoder_rollout_torch(symbol_steps)

    with torch.inference_mode():
        logits_ref = asr_model.joint(
            encoder_outputs=encoder_ref,
            decoder_outputs=decoder_ref,
        )

    # Convert tensors to numpy for CoreML
    def _np32(x):
        return np.array(x.detach().cpu().numpy(), dtype=np.float32, copy=True)

    # Prepare plot dir (write to repo-tracked plots/<script-name>/)
    plots_root = Path(__file__).parent / "plots"
    plots_dir = plots_root / Path(__file__).stem
    plots_dir.mkdir(parents=True, exist_ok=True)

    encoder_np = _np32(encoder_ref)
    decoder_ref_np = _np32(decoder_ref)

    summary: Dict[str, object] = {
        "requested": True,
        "status": "ok",
        "atol": atol,
        "rtol": rtol,
        "symbol_steps": int(symbol_steps),
        "audio_path": None if validation_audio is None else str(validation_audio),
        "components": {},
    }

    # Preprocessor
    pre = ct.models.MLModel(str(output_dir / "parakeet_preprocessor.mlpackage"), compute_units=ct.ComputeUnit.CPU_AND_NE)
    pre_out = pre.predict({"audio_signal": _np32(audio_tensor), "audio_length": _np32(audio_length).astype(np.int32)})
    mel_ml = np.array(pre_out["mel"], dtype=np.float32, copy=True)
    mel_len_ml = np.array(pre_out["mel_length"], dtype=np.int32, copy=True)
    pre_atol, pre_rtol = max(atol, 1.0), max(rtol, 1e-2)
    a_mel, r_mel, ok_mel = _max_diffs(_np32(mel_ref), mel_ml, pre_rtol, pre_atol)
    ok_len = int(_np32(mel_length_ref).astype(np.int32)[0]) == int(np.array(mel_len_ml).astype(np.int32)[0])
    mel_t = _np32(mel_ref)[0]
    mel_c = mel_ml[0]
    vmin = float(min(mel_t.min(), mel_c.min()))
    vmax = float(max(mel_t.max(), mel_c.max()))
    pre_plots = {
        "mel_composite.png": _plot_mel_composite(mel_t, mel_c, plots_dir / "mel_composite.png", vmin=vmin, vmax=vmax),
    }
    # Latency measurements: Torch and CoreML
    def _time_coreml(model: ct.models.MLModel, inputs: Dict[str, np.ndarray]) -> Tuple[float, float]:
        # Warmup
        for _ in range(max(0, warmup)):
            _ = model.predict(inputs)
        times = []
        for _ in range(max(1, runs)):
            t0 = time.perf_counter()
            _ = model.predict(inputs)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)
        arr = np.array(times, dtype=np.float64)
        return float(arr.mean()), float(arr.std(ddof=1) if arr.size > 1 else 0.0)

    def _time_torch(fn, *args, **kwargs) -> Tuple[float, float]:
        with torch.inference_mode():
            for _ in range(max(0, warmup)):
                _ = fn(*args, **kwargs)
            times = []
            for _ in range(max(1, runs)):
                t0 = time.perf_counter()
                _ = fn(*args, **kwargs)
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000.0)
        arr = np.array(times, dtype=np.float64)
        return float(arr.mean()), float(arr.std(ddof=1) if arr.size > 1 else 0.0)

    pre_torch_ms_mean, pre_torch_ms_std = _time_torch(
        asr_model.preprocessor, input_signal=audio_tensor, length=audio_length.to(dtype=torch.long)
    )
    pre_coreml_ms_mean, pre_coreml_ms_std = _time_coreml(
        pre,
        {"audio_signal": _np32(audio_tensor), "audio_length": _np32(audio_length).astype(np.int32)},
    )
    seconds = 15.0
    pre_coreml_rtf = float(pre_coreml_ms_mean / (seconds * 1000.0)) if pre_coreml_ms_mean > 0 else None
    pre_torch_rtf = float(pre_torch_ms_mean / (seconds * 1000.0)) if pre_torch_ms_mean > 0 else None

    summary["components"]["preprocessor"] = {
        "mel": {"max_abs": a_mel, "max_rel": r_mel, "match": bool(ok_mel)},
        "length_match": bool(ok_len),
        "latency": {
            "runs": int(runs),
            "warmup": int(warmup),
            "torch_ms": {"mean": pre_torch_ms_mean, "std": pre_torch_ms_std},
            "coreml_ms": {"mean": pre_coreml_ms_mean, "std": pre_coreml_ms_std},
            "rtf": {"torch": pre_torch_rtf, "coreml": pre_coreml_rtf},
        },
        "plots": {k: v for k, v in pre_plots.items() if v},
    }

    # Encoder
    enc = ct.models.MLModel(str(output_dir / "parakeet_encoder.mlpackage"), compute_units=ct.ComputeUnit.CPU_AND_NE)
    enc_out = enc.predict({"mel": _np32(mel_ref), "mel_length": _np32(mel_length_ref).astype(np.int32)})
    enc_ml = np.array(enc_out["encoder"], dtype=np.float32, copy=True)
    enc_len_ml = np.array(enc_out["encoder_length"], dtype=np.int32, copy=True)
    a_enc, r_enc, ok_enc = _max_diffs(_np32(encoder_ref), enc_ml, max(rtol, 5e-3), max(atol, 5e-2))
    ok_enc_len = int(_np32(encoder_length_ref).astype(np.int32)[0]) == int(np.array(enc_len_ml).astype(np.int32)[0])
    enc_t = _np32(encoder_ref)[0]
    enc_c = enc_ml[0]
    enc_plots = {
        "encoder_time_l2.png": _plot_line(
            np.linalg.norm(enc_t, axis=0),  # L2 norm over features (D) for each time step
            np.linalg.norm(enc_c, axis=0),  # enc_t shape is (D, T), so axis=0 is features
            "Encoder L2 over time",
            plots_dir / "encoder_time_l2.png",
            also_delta=True,
        ),
    }
    enc_torch_ms_mean, enc_torch_ms_std = _time_torch(
        asr_model.encoder, audio_signal=mel_ref, length=mel_length_ref.to(dtype=torch.long)
    )
    enc_coreml_ms_mean, enc_coreml_ms_std = _time_coreml(
        enc, {"mel": _np32(mel_ref), "mel_length": _np32(mel_length_ref).astype(np.int32)}
    )
    enc_coreml_rtf = float(enc_coreml_ms_mean / (seconds * 1000.0)) if enc_coreml_ms_mean > 0 else None
    enc_torch_rtf = float(enc_torch_ms_mean / (seconds * 1000.0)) if enc_torch_ms_mean > 0 else None

    summary["components"]["encoder"] = {
        "encoder": {"max_abs": a_enc, "max_rel": r_enc, "match": bool(ok_enc)},
        "length_match": bool(ok_enc_len),
        "latency": {
            "runs": int(runs),
            "warmup": int(warmup),
            "torch_ms": {"mean": enc_torch_ms_mean, "std": enc_torch_ms_std},
            "coreml_ms": {"mean": enc_coreml_ms_mean, "std": enc_coreml_ms_std},
            "rtf": {"torch": enc_torch_rtf, "coreml": enc_coreml_rtf},
        },
        "plots": {k: v for k, v in enc_plots.items() if v},
    }

    # Decoder (sequential U=1 rollout)
    dec = ct.models.MLModel(str(output_dir / "parakeet_decoder.mlpackage"), compute_units=ct.ComputeUnit.CPU_AND_NE)

    zero_state_np = np.zeros((decoder_layers, 1, decoder_hidden), dtype=np.float32)
    blank_targets_np = np.array(blank_targets.detach().cpu().numpy(), dtype=np.int32, copy=True)
    blank_target_lengths_np = np.array(blank_target_lengths.detach().cpu().numpy(), dtype=np.int32, copy=True)

    def _decoder_rollout_coreml(num_steps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        outputs = []
        h_np = zero_state_np.copy()
        c_np = zero_state_np.copy()
        for _ in range(num_steps):
            res = dec.predict(
                {
                    "targets": blank_targets_np,
                    "target_length": blank_target_lengths_np,
                    "h_in": h_np,
                    "c_in": c_np,
                }
            )
            outputs.append(np.array(res["decoder"], dtype=np.float32, copy=True))
            h_np = np.array(res["h_out"], dtype=np.float32, copy=True)
            c_np = np.array(res["c_out"], dtype=np.float32, copy=True)
        if outputs:
            decoder_seq = np.concatenate(outputs, axis=-1)
        else:
            decoder_seq = np.zeros((1, decoder_hidden, 0), dtype=np.float32)
        return decoder_seq, h_np, c_np

    dec_ml, h_ml, c_ml = _decoder_rollout_coreml(symbol_steps)
    h_ref_np = _np32(h_ref)
    c_ref_np = _np32(c_ref)

    a_dec, r_dec, ok_dec = _max_diffs(decoder_ref_np, dec_ml, max(rtol, 1e-2), max(atol, 1e-1))
    a_h, r_h, ok_h = _max_diffs(h_ref_np, h_ml, max(rtol, 1e-2), max(atol, 2.5e-1))
    a_c, r_c, ok_c = _max_diffs(c_ref_np, c_ml, max(rtol, 5e-2), max(atol, 1.5e0))

    dec_t = decoder_ref_np[0]
    dec_c = dec_ml[0]
    dec_plots = {
        "decoder_steps_l2.png": _plot_line(
            np.linalg.norm(dec_t, axis=0),
            np.linalg.norm(dec_c, axis=0),
            "Decoder L2 over steps",
            plots_dir / "decoder_steps_l2.png",
            also_delta=True,
        ),
    }

    def _time_decoder_coreml() -> Tuple[float, float]:
        for _ in range(max(0, warmup)):
            _decoder_rollout_coreml(symbol_steps)
        times = []
        for _ in range(max(1, runs)):
            t0 = time.perf_counter()
            _decoder_rollout_coreml(symbol_steps)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)
        arr = np.array(times, dtype=np.float64)
        return float(arr.mean()), float(arr.std(ddof=1) if arr.size > 1 else 0.0)

    dec_torch_ms_mean, dec_torch_ms_std = _time_torch(lambda: _decoder_rollout_torch(symbol_steps))
    dec_coreml_ms_mean, dec_coreml_ms_std = _time_decoder_coreml()
    dec_coreml_rtf = float(dec_coreml_ms_mean / (seconds * 1000.0)) if dec_coreml_ms_mean > 0 else None
    dec_torch_rtf = float(dec_torch_ms_mean / (seconds * 1000.0)) if dec_torch_ms_mean > 0 else None

    summary["components"]["decoder"] = {
        "decoder": {"max_abs": a_dec, "max_rel": r_dec, "match": bool(ok_dec)},
        "h_out": {"max_abs": a_h, "max_rel": r_h, "match": bool(ok_h)},
        "c_out": {"max_abs": a_c, "max_rel": r_c, "match": bool(ok_c)},
        "latency": {
            "runs": int(runs),
            "warmup": int(warmup),
            "torch_ms": {"mean": dec_torch_ms_mean, "std": dec_torch_ms_std},
            "coreml_ms": {"mean": dec_coreml_ms_mean, "std": dec_coreml_ms_std},
            "rtf": {"torch": dec_torch_rtf, "coreml": dec_coreml_rtf},
        },
        "plots": {k: v for k, v in dec_plots.items() if v},
    }

    # Joint (sequential U=1 rollouts)
    j = ct.models.MLModel(str(output_dir / "parakeet_joint.mlpackage"), compute_units=ct.ComputeUnit.CPU_AND_NE)

    def _joint_rollout_coreml(decoder_seq_np: np.ndarray) -> np.ndarray:
        logits_steps = []
        for u in range(decoder_seq_np.shape[2]):
            dec_slice = decoder_seq_np[:, :, u : u + 1]
            res = j.predict({"encoder": encoder_np, "decoder": dec_slice})
            logits_steps.append(np.array(res["logits"], dtype=np.float32, copy=True))
        if not logits_steps:
            raise RuntimeError("No decoder steps provided for joint rollout")
        return np.concatenate(logits_steps, axis=2)

    logits_ml = _joint_rollout_coreml(decoder_ref_np)
    logits_ref_np = _np32(logits_ref)
    a_j, r_j, ok_j = _max_diffs(logits_ref_np, logits_ml, max(rtol, 1e-2), max(atol, 1e-1))
    joint_plots = {}
    if HAS_MPL:
        lt = logits_ref_np[0, 0, 0, :]
        lc = logits_ml[0, 0, 0, :]
        top_idx = np.argsort(-np.abs(lt))[:50]
        path = plots_dir / "joint_top50.png"
        plt.figure(figsize=(8, 3))
        plt.plot(lt[top_idx], label="torch")
        plt.plot(lc[top_idx], label="coreml", alpha=0.8)
        plt.title("Joint logits (t=0,u=0) top-50 |torch|")
        plt.legend(); plt.tight_layout(); plt.savefig(path); plt.close()
        joint_plots["joint_top50.png"] = str(path.name)

        # Delta-over-time visualization (fix u=0; summarize over vocab)
        jt = logits_ref_np[0, :, 0, :]
        jc = logits_ml[0, :, 0, :]
        l2_t = np.linalg.norm(jt, axis=1)
        l2_c = np.linalg.norm(jc, axis=1)
        path2 = plots_dir / "joint_time_l2.png"
        _plot_line(l2_t, l2_c, "Joint L2 over time (u=0)", path2, also_delta=True)
        joint_plots["joint_time_l2.png"] = str(path2.name)

    def _time_joint_coreml() -> Tuple[float, float]:
        for _ in range(max(0, warmup)):
            _joint_rollout_coreml(decoder_ref_np)
        times = []
        for _ in range(max(1, runs)):
            t0 = time.perf_counter()
            _joint_rollout_coreml(decoder_ref_np)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)
        arr = np.array(times, dtype=np.float64)
        return float(arr.mean()), float(arr.std(ddof=1) if arr.size > 1 else 0.0)

    joint_torch_ms_mean, joint_torch_ms_std = _time_torch(
        asr_model.joint, encoder_outputs=encoder_ref, decoder_outputs=decoder_ref
    )
    joint_coreml_ms_mean, joint_coreml_ms_std = _time_joint_coreml()
    joint_coreml_rtf = float(joint_coreml_ms_mean / (seconds * 1000.0)) if joint_coreml_ms_mean > 0 else None
    joint_torch_rtf = float(joint_torch_ms_mean / (seconds * 1000.0)) if joint_torch_ms_mean > 0 else None

    summary["components"]["joint"] = {
        "logits": {"max_abs": a_j, "max_rel": r_j, "match": bool(ok_j)},
        "latency": {
            "runs": int(runs),
            "warmup": int(warmup),
            "torch_ms": {"mean": joint_torch_ms_mean, "std": joint_torch_ms_std},
            "coreml_ms": {"mean": joint_coreml_ms_mean, "std": joint_coreml_ms_std},
            "rtf": {"torch": joint_torch_rtf, "coreml": joint_coreml_rtf},
        },
        "plots": joint_plots,
    }

    # Fused components
    # 1) Mel+Encoder fused vs separate
    mel_enc_plots = {}
    try:
        mel_enc = ct.models.MLModel(str(output_dir / "parakeet_mel_encoder.mlpackage"), compute_units=ct.ComputeUnit.CPU_AND_NE)
        mel_enc_out = mel_enc.predict({
            "audio_signal": _np32(audio_tensor),
            "audio_length": _np32(audio_length).astype(np.int32),
        })
        mel_enc_ml = np.array(mel_enc_out["encoder"], dtype=np.float32, copy=True)
        mel_enc_len_ml = np.array(mel_enc_out["encoder_length"], dtype=np.int32, copy=True)
        # Compare fused output vs Torch reference encoder
        a_melenc, r_melenc, ok_melenc = _max_diffs(_np32(encoder_ref), mel_enc_ml, max(rtol, 5e-3), max(atol, 5e-2))
        ok_melenc_len = int(_np32(encoder_length_ref).astype(np.int32)[0]) == int(mel_enc_len_ml.astype(np.int32)[0])
        # Also compare fused vs separate CoreML pipeline (pre -> enc)
        a_melenc_vs_sep, r_melenc_vs_sep, ok_melenc_vs_sep = _max_diffs(enc_ml, mel_enc_ml, max(rtol, 5e-3), max(atol, 5e-2))

        # Plots: L2 over time (fused vs torch)
        enc_t_ref = _np32(encoder_ref)[0]
        enc_c_fused = mel_enc_ml[0]
        mel_enc_plots["mel_encoder_time_l2.png"] = _plot_line(
            np.linalg.norm(enc_t_ref, axis=0),
            np.linalg.norm(enc_c_fused, axis=0),
            "Mel+Encoder (fused) L2 over time",
            plots_dir / "mel_encoder_time_l2.png",
            also_delta=True,
        )

        # Latency: fused CoreML vs separate (CoreML pre + CoreML enc)
        mel_enc_coreml_ms_mean, mel_enc_coreml_ms_std = _time_coreml(
            mel_enc,
            {"audio_signal": _np32(audio_tensor), "audio_length": _np32(audio_length).astype(np.int32)},
        )
        sep_coreml_ms_mean = float(pre_coreml_ms_mean + enc_coreml_ms_mean)
        sep_coreml_ms_std = float((pre_coreml_ms_std ** 2 + enc_coreml_ms_std ** 2) ** 0.5)
        # Torch baseline (separate torch pre + enc)
        sep_torch_ms_mean = float(pre_torch_ms_mean + enc_torch_ms_mean)
        sep_torch_ms_std = float((pre_torch_ms_std ** 2 + enc_torch_ms_std ** 2) ** 0.5)

        mel_enc_coreml_rtf = float(mel_enc_coreml_ms_mean / (seconds * 1000.0)) if mel_enc_coreml_ms_mean > 0 else None
        sep_coreml_rtf = float(sep_coreml_ms_mean / (seconds * 1000.0)) if sep_coreml_ms_mean > 0 else None
        sep_torch_rtf = float(sep_torch_ms_mean / (seconds * 1000.0)) if sep_torch_ms_mean > 0 else None

        summary["components"]["mel_encoder"] = {
            "encoder": {"max_abs": a_melenc, "max_rel": r_melenc, "match": bool(ok_melenc)},
            "length_match": bool(ok_melenc_len),
            "vs_separate_coreml": {"max_abs": a_melenc_vs_sep, "max_rel": r_melenc_vs_sep, "match": bool(ok_melenc_vs_sep)},
            "latency": {
                "runs": int(runs),
                "warmup": int(warmup),
                "fused_coreml_ms": {"mean": mel_enc_coreml_ms_mean, "std": mel_enc_coreml_ms_std},
                "separate_coreml_ms": {"mean": sep_coreml_ms_mean, "std": sep_coreml_ms_std},
                "separate_torch_ms": {"mean": sep_torch_ms_mean, "std": sep_torch_ms_std},
                "rtf": {"fused_coreml": mel_enc_coreml_rtf, "separate_coreml": sep_coreml_rtf, "separate_torch": sep_torch_rtf},
            },
            "plots": {k: v for k, v in mel_enc_plots.items() if v},
        }
    except Exception as e:
        summary["components"]["mel_encoder_error"] = str(e)

    # 2) JointDecision fused vs CPU PyTorch post-processing
    jd_plots = {}
    try:
        # Fused CoreML joint decision
        jd = ct.models.MLModel(str(output_dir / "parakeet_joint_decision.mlpackage"), compute_units=ct.ComputeUnit.CPU_AND_NE)
        def _joint_decision_rollout_coreml(decoder_seq_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            token_ids = []
            token_probs = []
            durations = []
            for u in range(decoder_seq_np.shape[2]):
                dec_slice = decoder_seq_np[:, :, u : u + 1]
                res = jd.predict({"encoder": encoder_np, "decoder": dec_slice})
                token_ids.append(np.array(res["token_id"], dtype=np.int32, copy=True))
                token_probs.append(np.array(res["token_prob"], dtype=np.float32, copy=True))
                durations.append(np.array(res["duration"], dtype=np.int32, copy=True))
            if not token_ids:
                raise RuntimeError("No decoder steps provided for joint decision rollout")
            return (
                np.concatenate(token_ids, axis=2),
                np.concatenate(token_probs, axis=2),
                np.concatenate(durations, axis=2),
            )

        token_id_ml, token_prob_ml, duration_ml = _joint_decision_rollout_coreml(decoder_ref_np)

        # CPU PyTorch decision using Torch logits
        vocab_with_blank = int(vocab_size) + 1
        with torch.inference_mode():
            logits_t = logits_ref
            token_logits_t = logits_t[..., :vocab_with_blank]
            duration_logits_t = logits_t[..., -num_extra:] if num_extra > 0 else None
            token_ids_t = torch.argmax(token_logits_t, dim=-1).to(dtype=torch.int32)
            token_probs_all_t = torch.softmax(token_logits_t, dim=-1)
            token_prob_t = torch.gather(
                token_probs_all_t, dim=-1, index=token_ids_t.long().unsqueeze(-1)
            ).squeeze(-1)
            if duration_logits_t is not None and duration_logits_t.numel() > 0:
                duration_t = torch.argmax(duration_logits_t, dim=-1).to(dtype=torch.int32)
            else:
                duration_t = torch.zeros_like(token_ids_t, dtype=torch.int32)

        # Also derive CPU decision from CoreML joint logits for "separate" path
        token_logits_c = _to_t(logits_ml)[..., :vocab_with_blank]
        duration_logits_c = _to_t(logits_ml)[..., -num_extra:] if num_extra > 0 else None
        token_ids_c = torch.argmax(token_logits_c, dim=-1).to(dtype=torch.int32)
        token_probs_all_c = torch.softmax(token_logits_c, dim=-1)
        token_prob_c = torch.gather(
            token_probs_all_c, dim=-1, index=token_ids_c.long().unsqueeze(-1)
        ).squeeze(-1)
        if duration_logits_c is not None and duration_logits_c.numel() > 0:
            duration_c = torch.argmax(duration_logits_c, dim=-1).to(dtype=torch.int32)
        else:
            duration_c = torch.zeros_like(token_ids_c, dtype=torch.int32)

        # Compare fused outputs to CPU PyTorch decisions
        a_tid_t, r_tid_t, ok_tid_t = _max_diffs(_np(token_ids_t), token_id_ml, 0.0, 0.0)
        a_tprob_t, r_tprob_t, ok_tprob_t = _max_diffs(_np(token_prob_t), token_prob_ml, max(rtol, 1e-2), max(atol, 1e-1))
        a_dur_t, r_dur_t, ok_dur_t = _max_diffs(_np(duration_t), duration_ml, 0.0, 0.0)

        a_tid_c, r_tid_c, ok_tid_c = _max_diffs(_np(token_ids_c), token_id_ml, 0.0, 0.0)
        a_tprob_c, r_tprob_c, ok_tprob_c = _max_diffs(_np(token_prob_c), token_prob_ml, max(rtol, 1e-2), max(atol, 1e-1))
        a_dur_c, r_dur_c, ok_dur_c = _max_diffs(_np(duration_c), duration_ml, 0.0, 0.0)

        # Plots: token_prob over time for u=0 (fused vs torch CPU)
        if HAS_MPL:
            prob_t = _np(token_prob_t)[0, :, 0]
            prob_ml = token_prob_ml[0, :, 0]
            jd_plots["joint_decision_prob_u0.png"] = _plot_line(
                prob_t,
                prob_ml,
                "JointDecision token_prob (u=0)",
                plots_dir / "joint_decision_prob_u0.png",
                also_delta=True,
            )

            # Agreement heatmap for token_id
            agree = (_np(token_ids_t)[0] == token_id_ml[0]).astype(np.float32)
            jd_plots["joint_decision_token_agree.png"] = _plot_image(
                agree,
                "token_id agreement (torch CPU vs fused)",
                plots_dir / "joint_decision_token_agree.png",
                vmin=0.0,
                vmax=1.0,
            )

        # Latency: fused CoreML vs separate (CoreML joint + CPU PyTorch decision)
        def _time_joint_decision_coreml() -> Tuple[float, float]:
            for _ in range(max(0, warmup)):
                _joint_decision_rollout_coreml(decoder_ref_np)
            times = []
            for _ in range(max(1, runs)):
                t0 = time.perf_counter()
                _joint_decision_rollout_coreml(decoder_ref_np)
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000.0)
            arr = np.array(times, dtype=np.float64)
            return float(arr.mean()), float(arr.std(ddof=1) if arr.size > 1 else 0.0)

        jd_coreml_ms_mean, jd_coreml_ms_std = _time_joint_decision_coreml()

        # Time CPU post-processing only (Torch) on top of CoreML or Torch logits. Use Torch logits.
        def _decision_torch_call():
            with torch.inference_mode():
                tl = logits_ref
                tl_token = tl[..., :vocab_with_blank]
                tl_ids = torch.argmax(tl_token, dim=-1)
                tl_probs = torch.softmax(tl_token, dim=-1)
                _ = torch.gather(tl_probs, -1, tl_ids.long().unsqueeze(-1)).squeeze(-1)
                if num_extra > 0:
                    _ = torch.argmax(tl[..., -num_extra:], dim=-1)
            return None

        jd_decision_torch_ms_mean, jd_decision_torch_ms_std = _time_torch(lambda: _decision_torch_call())
        sep_joint_plus_cpu_ms_mean = float(joint_coreml_ms_mean + jd_decision_torch_ms_mean)
        sep_joint_plus_cpu_ms_std = float((joint_coreml_ms_std ** 2 + jd_decision_torch_ms_std ** 2) ** 0.5)
        jd_coreml_rtf = float(jd_coreml_ms_mean / (seconds * 1000.0)) if jd_coreml_ms_mean > 0 else None
        sep_joint_cpu_rtf = float(sep_joint_plus_cpu_ms_mean / (seconds * 1000.0)) if sep_joint_plus_cpu_ms_mean > 0 else None

        summary["components"]["joint_decision"] = {
            "vs_torch_cpu": {
                "token_id": {"max_abs": a_tid_t, "max_rel": r_tid_t, "match": bool(ok_tid_t)},
                "token_prob": {"max_abs": a_tprob_t, "max_rel": r_tprob_t, "match": bool(ok_tprob_t)},
                "duration": {"max_abs": a_dur_t, "max_rel": r_dur_t, "match": bool(ok_dur_t)},
            },
            "vs_coreml_joint_cpu": {
                "token_id": {"max_abs": a_tid_c, "max_rel": r_tid_c, "match": bool(ok_tid_c)},
                "token_prob": {"max_abs": a_tprob_c, "max_rel": r_tprob_c, "match": bool(ok_tprob_c)},
                "duration": {"max_abs": a_dur_c, "max_rel": r_dur_c, "match": bool(ok_dur_c)},
            },
            "latency": {
                "runs": int(runs),
                "warmup": int(warmup),
                "fused_coreml_ms": {"mean": jd_coreml_ms_mean, "std": jd_coreml_ms_std},
                "separate_joint_coreml_plus_cpu_ms": {"mean": sep_joint_plus_cpu_ms_mean, "std": sep_joint_plus_cpu_ms_std},
                "rtf": {"fused_coreml": jd_coreml_rtf, "separate_joint_coreml_plus_cpu": sep_joint_cpu_rtf},
            },
            "plots": {k: v for k, v in jd_plots.items() if v},
        }
    except Exception as e:
        summary["components"]["joint_decision_error"] = str(e)

    # Latency overview plots (saved alongside component plots)
    latency_plots = {}
    labels = ["preprocessor", "encoder", "decoder", "joint"]
    torch_means = [pre_torch_ms_mean, enc_torch_ms_mean, dec_torch_ms_mean, joint_torch_ms_mean]
    torch_stds = [pre_torch_ms_std, enc_torch_ms_std, dec_torch_ms_std, joint_torch_ms_std]
    coreml_means = [pre_coreml_ms_mean, enc_coreml_ms_mean, dec_coreml_ms_mean, joint_coreml_ms_mean]
    coreml_stds = [pre_coreml_ms_std, enc_coreml_ms_std, dec_coreml_ms_std, joint_coreml_ms_std]
    lat_path = plots_dir / "latency_summary.png"
    spd_path = plots_dir / "latency_speedup.png"
    latency_plots["latency_summary.png"] = _plot_latency_bars(
        labels, torch_means, torch_stds, coreml_means, coreml_stds, lat_path
    )
    latency_plots["latency_speedup.png"] = _plot_speedup_bars(
        labels, torch_means, coreml_means, spd_path
    )

    # Fused vs separate latency summary
    fused_labels = ["mel+encoder", "joint_decision"]
    fused_baseline_means = [
        float(pre_torch_ms_mean + enc_torch_ms_mean),
        float(joint_coreml_ms_mean + jd_decision_torch_ms_mean if 'jd_coreml_ms_mean' in locals() else joint_coreml_ms_mean),
    ]
    fused_coreml_means = [
        float(mel_enc_coreml_ms_mean if 'mel_enc_coreml_ms_mean' in locals() else np.nan),
        float(jd_coreml_ms_mean if 'jd_coreml_ms_mean' in locals() else np.nan),
    ]
    fused_latency_path = plots_dir / "latency_fused_vs_separate.png"
    fused_speedup_path = plots_dir / "latency_fused_speedup.png"
    latency_plots["latency_fused_vs_separate.png"] = _plot_latency_bars(
        fused_labels, fused_baseline_means, [0, 0], fused_coreml_means, [0, 0], fused_latency_path
    )
    latency_plots["latency_fused_speedup.png"] = _plot_speedup_bars(
        fused_labels, fused_baseline_means, fused_coreml_means, fused_speedup_path
    )

    all_ok = (
        summary["components"]["preprocessor"]["mel"]["match"]
        and summary["components"]["preprocessor"]["length_match"]
        and summary["components"]["encoder"]["encoder"]["match"]
        and summary["components"]["encoder"]["length_match"]
        and summary["components"]["decoder"]["decoder"]["match"]
        and summary["components"]["decoder"]["h_out"]["match"]
        and summary["components"]["decoder"]["c_out"]["match"]
        and summary["components"]["joint"]["logits"]["match"]
    )
    summary["status"] = "ok" if all_ok else "mismatch"

    # Update metadata.json
    meta_path = output_dir / "metadata.json"
    try:
        meta = json.loads(meta_path.read_text())
    except Exception:
        meta = {}
    meta["validation"] = summary
    meta_path.write_text(json.dumps(meta, indent=2))

    typer.echo(f"Validation {'passed' if all_ok else 'mismatched'}. Updated {meta_path}")
    if HAS_MPL:
        typer.echo(f"Saved plots to {plots_dir}")


if __name__ == "__main__":
    app()

"""Whisper load + transcribe probe for the skeleton cloud deploy.

Goal: prove that the Whisper model folder is present and that a real encode/decode +
transcribe pass works and how long/how much memory it takes. Uses a synthetic tone
(no speech), so vad_filter is disabled to force the encoder to actually run.

Usage:
    backend\\venv\\Scripts\\python.exe -u backend\\scripts\\whisper_probe.py --model small
    backend\\venv\\Scripts\\python.exe -u backend\\scripts\\whisper_probe.py --model base

Never prints environment values (no secrets). The temp .ogg is written inside the repo
(*.ogg is git-ignored) and deleted afterwards.
"""
import argparse
import asyncio
import os
import sys
import tempfile
import time

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from backend.config.settings import settings
from backend.services.audio import decode_audio, encode_ogg_opus

TRANSCRIBE_CAP_SECONDS = 120.0


def synth_tone(seconds: float = 8.0, rate: int = 16000, freq: float = 440.0,
               amp: float = 0.2, noise: float = 0.01):
    """8 s of 16 kHz mono synthetic audio: a sine tone plus low noise, in memory."""
    import numpy as np

    n = int(seconds * rate)
    t = np.arange(n, dtype=np.float64) / rate
    rng = np.random.default_rng(1234)
    sig = amp * np.sin(2.0 * np.pi * freq * t) + noise * rng.standard_normal(n)
    return np.clip(sig, -1.0, 1.0).astype(np.float32)


def peak_rss_mb():
    """Peak RSS: resource.getrusage on Linux, else psutil, else None ('n/a')."""
    try:
        import resource

        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return ru / (1024.0 * 1024.0) if sys.platform == "darwin" else ru / 1024.0
    except Exception:
        pass
    try:
        import psutil

        return psutil.Process().memory_info().rss / (1024.0 * 1024.0)
    except Exception:
        return None


def cgroup_report():
    """On Linux, read cgroup memory/cpu limits if readable; otherwise return []."""
    if not sys.platform.startswith("linux"):
        return []
    lines = []
    for path in ("/sys/fs/cgroup/memory.peak", "/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/cpu.max"):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                lines.append(f"    {path} = {fh.read().strip()}")
        except Exception:
            lines.append(f"    {path} = (unreadable)")
    return lines


def _transcribe_sync(model, audio):
    segments, info = model.transcribe(
        audio,
        beam_size=1,
        vad_filter=False,  # force the encoder to run on synthetic audio
        condition_on_previous_text=False,
        temperature=0.0,  # no fallback retries; a tone otherwise triggers repeated decoding
    )
    text = " ".join(s.text.strip() for s in segments).strip()
    return text, info


async def run(model_name: str) -> int:
    print("=" * 78)
    print("WHISPER PROBE")
    print("=" * 78)

    model_dir = settings.whisper_model_dir if settings.WHISPER_MODEL_NAME == model_name else \
        (Path(settings.WHISPER_MODEL_ROOT) if Path(settings.WHISPER_MODEL_ROOT).is_absolute()
         else REPO_ROOT / settings.WHISPER_MODEL_ROOT) / f"whisper-{model_name}"

    print(f"model name        : {model_name}")
    print(f"model dir         : {model_dir}")
    print(f"dir exists        : {model_dir.is_dir()}")
    print(f"cpu_count         : {os.cpu_count()}")
    print("note              : synthetic audio, vad_filter=False (real speech later uses vad_filter=True)")
    print("-" * 78)

    if not model_dir.is_dir():
        print(f"FAIL: model directory not found: {model_dir}")
        return 1

    # 1) Build the Ogg/Opus file in memory, write inside the repo, delete afterwards.
    samples = synth_tone()
    fd, tmp_path = tempfile.mkstemp(suffix=".ogg", dir=str(REPO_ROOT))
    os.close(fd)
    try:
        t0 = time.perf_counter()
        encode_ogg_opus(samples, tmp_path, rate=16000)
        enc_s = time.perf_counter() - t0
        decoded = decode_audio(tmp_path)
        print(f"synthetic audio   : {samples.size} samples ({samples.size/16000:.1f}s) "
              f"-> ogg {os.path.getsize(tmp_path)} bytes in {enc_s:.3f}s")
        print(f"decoded           : {decoded.size} samples ({decoded.size/16000:.1f}s)")
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    # 2) Load the model.
    from faster_whisper import WhisperModel

    rss_before = peak_rss_mb()
    t0 = time.perf_counter()
    model = WhisperModel(str(model_dir), device="cpu", compute_type="int8")
    load_s = time.perf_counter() - t0

    # 3) Transcribe with a hard 120 s cap.
    t0 = time.perf_counter()
    try:
        text, info = await asyncio.wait_for(asyncio.to_thread(_transcribe_sync, model, decoded),
                                            timeout=TRANSCRIBE_CAP_SECONDS)
        tx_s = time.perf_counter() - t0
    except asyncio.TimeoutError:
        print(f"FAIL: transcription exceeded the {TRANSCRIBE_CAP_SECONDS:.0f}s cap")
        return 1

    rss_after = peak_rss_mb()

    print("-" * 78)
    print(f"load time         : {load_s:.2f}s")
    print(f"transcribe time   : {tx_s:.2f}s")
    print(f"peak RSS          : {('n/a' if rss_after is None else f'{rss_after:.1f} MB')}"
          f"{'' if rss_before is None else f' (before load {rss_before:.1f} MB)'}")
    print(f"detected language : {getattr(info, 'language', 'n/a')}")
    print(f"transcript chars  : {len(text)} (expected ~0 for a tone)")
    cg = cgroup_report()
    if cg:
        print("cgroup            :")
        for line in cg:
            print(line)
    elif sys.platform.startswith("linux"):
        print("cgroup            : (none readable)")
    else:
        print("cgroup            : n/a (not Linux)")
    print("=" * 78)
    print("WHISPER PROBE OK")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["small", "base"], default="small")
    args = parser.parse_args()
    try:
        return asyncio.run(run(args.model))
    except Exception as exc:  # noqa: BLE001 - probe reports its own failures
        print(f"FAIL: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

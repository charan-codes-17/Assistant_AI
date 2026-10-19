"""Shared audio helpers (PyAV): decode OGG/Opus -> 16 kHz mono float32, and encode.

The decode function is copied verbatim from the Phase 1 spike
(backend/scripts/spike_telegram.py) so the Telegram bot and the Whisper probe share
the exact same decode path. spike_telegram.py itself is intentionally not modified.

Phase 4b adds the Whisper transcription side: a lazy, process-wide int8 model singleton
and a bytes -> text helper used by the bot voice handler. whisper_probe.py stays independent.
"""
from __future__ import annotations

import os
import tempfile
import threading

_model = None
_model_lock = threading.Lock()


def decode_audio(path: str):
    """Decode OGG/Opus to 16 kHz mono float32 using PyAV directly."""
    import av
    import numpy as np

    resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
    chunks = []

    def collect(frames) -> None:
        if frames is None:
            return
        if not isinstance(frames, (list, tuple)):
            frames = [frames]
        for fr in frames:
            chunks.append(fr.to_ndarray().reshape(-1))

    with av.open(path) as container:
        stream = container.streams.audio[0]
        for frame in container.decode(stream):
            collect(resampler.resample(frame))
        collect(resampler.resample(None))

    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks).astype(np.float32) / 32768.0


def encode_ogg_opus(samples, path: str, rate: int = 16000) -> None:
    """Encode a mono float32 waveform ([-1, 1]) to an OGG/Opus file with PyAV."""
    import av
    import numpy as np

    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)

    container = av.open(path, mode="w", format="ogg")
    stream = container.add_stream("libopus", rate=rate)
    stream.layout = "mono"
    stream.format = "s16"
    frame = av.AudioFrame.from_ndarray(pcm.reshape(1, -1), format="s16", layout="mono")
    frame.sample_rate = rate
    for packet in stream.encode(frame):
        container.mux(packet)
    for packet in stream.encode(None):
        container.mux(packet)
    container.close()


def get_whisper_model():
    """Return the process-wide WhisperModel, loading it once (thread-safe, int8, CPU)."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from faster_whisper import WhisperModel

                from backend.config.settings import settings

                _model = WhisperModel(
                    str(settings.whisper_model_dir),
                    device="cpu",
                    compute_type="int8",
                    cpu_threads=settings.WHISPER_CPU_THREADS,
                )
    return _model


def warmup() -> None:
    """Blocking model load, for startup warm-up via asyncio.to_thread."""
    get_whisper_model()


def transcribe_samples(model, samples, vad_filter: bool = True) -> str:
    """Transcribe a 16 kHz mono float32 waveform. Speech only, so vad_filter defaults on."""
    segments, _info = model.transcribe(
        samples,
        beam_size=1,
        vad_filter=vad_filter,
        condition_on_previous_text=False,
        temperature=0.0,
    )
    return " ".join(segment.text.strip() for segment in segments).strip()


def transcribe_bytes(data: bytes) -> str:
    """Decode OGG/Opus bytes and transcribe them. The temp file is always deleted."""
    fd, path = tempfile.mkstemp(suffix=".ogg")
    os.close(fd)
    try:
        with open(path, "wb") as handle:
            handle.write(data)
        samples = decode_audio(path)
        return transcribe_samples(get_whisper_model(), samples)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


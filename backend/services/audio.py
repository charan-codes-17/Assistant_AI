"""Shared audio helpers (PyAV): decode OGG/Opus -> 16 kHz mono float32, and encode.

The decode function is copied verbatim from the Phase 1 spike
(backend/scripts/spike_telegram.py) so the Telegram bot and the Whisper probe share
the exact same decode path. spike_telegram.py itself is intentionally not modified.
"""
from __future__ import annotations


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

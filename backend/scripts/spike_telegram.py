"""Throwaway Phase 1 spike: Telegram echo + inline button + voice transcription."""
import asyncio
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import psutil
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from backend.config.settings import settings

PROC = psutil.Process()
WHISPER = None


def ram_mb() -> float:
    return PROC.memory_info().rss / (1024 * 1024)


def load_whisper() -> None:
    global WHISPER
    from faster_whisper import WhisperModel

    r0, t0 = ram_mb(), time.perf_counter()
    WHISPER = WhisperModel(str(Path(__file__).resolve().parents[1] / "whisper_models" / "whisper-small"), device="cpu", compute_type="int8")
    print(f"[whisper] loaded in {time.perf_counter() - t0:.1f}s, RAM {r0:.0f} -> {ram_mb():.0f} MB")


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


def transcribe(path: str) -> str:
    audio = decode_audio(path)
    if audio.size == 0:
        return ""
    segments, _info = WHISPER.transcribe(audio, beam_size=1, vad_filter=True)
    return " ".join(s.text.strip() for s in segments).strip()


async def on_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    kb = InlineKeyboardMarkup([[InlineKeyboardButton("Press me", callback_data="btn_1")]])
    await update.message.reply_text("Spike bot is up. Tap the button, or send text or a voice note.", reply_markup=kb)


async def on_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    q = update.callback_query
    await q.answer()
    await q.edit_message_text(f"You pressed: {q.data}")


async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(f"Echo: {update.message.text}")


async def on_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.message
    tmp_path = None
    try:
        tg_file = await context.bot.get_file(msg.voice.file_id)
        fd, tmp_path = tempfile.mkstemp(suffix=".ogg")
        os.close(fd)
        await tg_file.download_to_drive(tmp_path)
        t0 = time.perf_counter()
        text = await asyncio.to_thread(transcribe, tmp_path)
        dt = time.perf_counter() - t0
        print(f"[voice] {msg.voice.duration}s audio -> transcribed in {dt:.2f}s, RAM {ram_mb():.0f} MB, chars={len(text)}")
        if not text:
            await msg.reply_text("I couldn't make out any speech in that. Could you try again?")
        else:
            await msg.reply_text(f"Heard: {text}")
    except Exception as exc:
        print(f"[voice] ERROR {type(exc).__name__}: {exc}")
        await msg.reply_text("Sorry, I couldn't process that voice note.")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


async def amain() -> None:
    app = Application.builder().token(settings.TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", on_start))
    app.add_handler(CallbackQueryHandler(on_button))
    app.add_handler(MessageHandler(filters.VOICE, on_voice))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    await app.initialize()
    await app.start()
    await app.updater.start_polling()
    print("Bot running. Press Ctrl+C to stop.")
    try:
        await asyncio.Event().wait()
    finally:
        await app.updater.stop()
        await app.stop()
        await app.shutdown()


if __name__ == "__main__":
    settings.require("TELEGRAM_BOT_TOKEN")
    load_whisper()
    try:
        asyncio.run(amain())
    except KeyboardInterrupt:
        print("stopped")
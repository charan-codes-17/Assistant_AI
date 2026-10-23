"""backend/bot/__main__.py
Entry point: `python -m backend.bot`.

Runs the Telegram bot with long polling (text only, Phase 4a). telegram is imported inside main()
so that importing this module (e.g. from tests, to verify the logging setup) has no side effects
and does not require a token.
"""

from __future__ import annotations

import asyncio
import logging

from backend.config.settings import settings


def configure_logging() -> None:
    """Configure root logging and silence the libraries that would log the bot token.

    The token appears in the request URL, so httpx / httpx2 / httpcore / telegram are pinned to
    WARNING to keep it out of the logs.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    for name in ("httpx", "httpx2", "httpcore", "telegram"):
        logging.getLogger(name).setLevel(logging.WARNING)


def main() -> None:
    configure_logging()
    settings.require("TELEGRAM_BOT_TOKEN")

    from backend.bot import scheduler as scheduler_module
    from backend.bot.app import build_application

    if not settings.INVITE_CODES.strip():
        logging.getLogger(__name__).warning(
            "INVITE_CODES is empty: nobody can sign up via /start <code>."
        )

    async def post_init(application) -> None:
        """Start the reminder scheduler FIRST, then warm up Whisper without blocking.

        The scheduler is deterministic (no LLM / embedding calls). Starting it before the
        warm-up means a Whisper failure can never stop reminders from being delivered.
        """
        task = asyncio.create_task(scheduler_module.scheduler_loop(application.bot))
        application.bot_data["reminder_task"] = task
        logging.getLogger(__name__).info("Reminder scheduler started")

        from backend.services import audio as audio_service

        try:
            await asyncio.to_thread(audio_service.warmup)
            logging.getLogger(__name__).info("Whisper model warm-up complete.")
        except Exception:  # noqa: BLE001 - voice notes will fail friendly until it works
            logging.getLogger(__name__).warning(
                "Whisper warm-up failed; voice notes will return an error until it succeeds."
            )

    async def post_shutdown(application) -> None:
        """Cancel the reminder scheduler task cleanly."""
        task = application.bot_data.get("reminder_task")
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    application = build_application(post_init=post_init, post_shutdown=post_shutdown)

    application.run_polling(
        drop_pending_updates=True,
        allowed_updates=["message", "callback_query"],
    )


if __name__ == "__main__":
    main()

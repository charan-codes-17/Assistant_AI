"""backend/bot/__main__.py
Entry point: `python -m backend.bot`.

Runs the Telegram bot with long polling (text only, Phase 4a). telegram is imported inside main()
so that importing this module (e.g. from tests, to verify the logging setup) has no side effects
and does not require a token.
"""

from __future__ import annotations

import logging

from backend.config.settings import settings


def configure_logging() -> None:
    """Configure root logging and silence the libraries that would log the bot token.

    The token appears in the request URL, so httpx / httpcore / telegram are pinned to WARNING
    to keep it out of the logs.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    for name in ("httpx", "httpcore", "telegram"):
        logging.getLogger(name).setLevel(logging.WARNING)


def main() -> None:
    configure_logging()
    settings.require("TELEGRAM_BOT_TOKEN")

    from telegram.ext import (
        Application,
        CallbackQueryHandler,
        CommandHandler,
        MessageHandler,
        filters,
    )

    from backend.bot.handlers import (
        cmd_start,
        on_callback,
        on_error,
        on_newsession,
        on_other_command,
        on_text,
        on_voice,
    )

    application = Application.builder().token(settings.TELEGRAM_BOT_TOKEN).build()
    application.add_handler(CommandHandler("start", cmd_start))
    application.add_handler(CommandHandler("newsession", on_newsession))
    application.add_handler(CallbackQueryHandler(on_callback))
    application.add_handler(MessageHandler(filters.VOICE, on_voice))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    application.add_handler(MessageHandler(filters.COMMAND, on_other_command))
    application.add_error_handler(on_error)

    application.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()

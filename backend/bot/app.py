"""backend/bot/app.py
Builder for the Telegram Application, shared by the two entry points:

- the standalone worker (`python -m backend.bot`), which passes its own post_init /
  post_shutdown callbacks and then calls `run_polling`;
- single-process mode, where the API lifespan starts the same Application manually
  (`initialize` / `start` / `updater.start_polling`) and NEVER calls `run_polling`.

The handlers, their order and the builder options are identical to the ones the worker
used before this module existed, so the two entry points cannot drift apart.
"""

from __future__ import annotations

from typing import Any, Callable, Optional


def build_application(
    post_init: Optional[Callable[[Any], Any]] = None,
    post_shutdown: Optional[Callable[[Any], Any]] = None,
):
    """Return the Application built exactly as the standalone worker builds it.

    `telegram` is imported here (not at module import) so that importing this module in
    an API process has no library side effects until a bot is actually wanted. The caller
    is responsible for having a token in settings.
    """
    from telegram.ext import (
        Application,
        CallbackQueryHandler,
        CommandHandler,
        MessageHandler,
        filters,
    )

    from backend.bot.handlers import (
        cmd_forget,
        cmd_help,
        cmd_memory,
        cmd_quiet,
        cmd_skill,
        cmd_skills,
        cmd_start,
        on_callback,
        on_error,
        on_newsession,
        on_other_command,
        on_text,
        on_voice,
    )
    from backend.config.settings import settings

    builder = (
        Application.builder()
        .token(settings.TELEGRAM_BOT_TOKEN)
        .concurrent_updates(True)
    )
    if post_init is not None:
        builder = builder.post_init(post_init)
    if post_shutdown is not None:
        builder = builder.post_shutdown(post_shutdown)

    application = builder.build()
    application.add_handler(CommandHandler("start", cmd_start))
    application.add_handler(CommandHandler("newsession", on_newsession))
    application.add_handler(CommandHandler("quiet", cmd_quiet))
    application.add_handler(CommandHandler("help", cmd_help))
    application.add_handler(CommandHandler("memory", cmd_memory))
    application.add_handler(CommandHandler("skills", cmd_skills))
    application.add_handler(CommandHandler("skill", cmd_skill))
    application.add_handler(CommandHandler("forget", cmd_forget))
    application.add_handler(CallbackQueryHandler(on_callback))
    application.add_handler(MessageHandler(filters.VOICE, on_voice))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    application.add_handler(MessageHandler(filters.COMMAND, on_other_command))
    application.add_error_handler(on_error)
    return application

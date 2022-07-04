import asyncio
import logging
from aiogram import executor
from aiogram.utils.executor import start_webhook
from data.config import WEBHOOK_URL, WEBHOOK_PATH, WEBAPP_HOST, WEBAPP_PORT
from handlers import dp
from loader import  bot


async def on_startapp(dispatcher):
    from utils.notify_admins import on_startup_notify
    await on_startup_notify(dispatcher)

    from utils.set_bot_commands import set_default_commands
    await set_default_commands(dispatcher)

    print('Bot works!')

async def on_startup(dispatcher):
    await bot.set_webhook(WEBHOOK_URL, drop_pending_updates=True)
    await on_startapp(dispatcher)


async def on_shutdown(dispatcher):
    await bot.delete_webhook()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    start_webhook(
        dispatcher=dp,
        webhook_path=WEBHOOK_PATH,
        skip_updates=True,
        on_startup=on_startup,
        on_shutdown=on_shutdown,
        host=WEBAPP_HOST,
        port=WEBAPP_PORT,
    )
    #executor.start_polling(dp, on_startup=on_startapp)
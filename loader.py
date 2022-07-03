from aiogram import Bot, Dispatcher
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from data import config


bot = Bot(token=config.BOT_TOKEN)

dp = Dispatcher(bot, storage=MemoryStorage())
from aiogram import types
from loader import dp
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters.state import State, StatesGroup
import nmodel.StyleModel as sm
from data.config import admins_id
import os


class UserData(StatesGroup):
    waiting_for_content_photo = State()
    waiting_for_style_photo = State()
    content_photo_id = 0

async  def style_transfer(content_path, style_path, user_id):
    model = sm.StyleModel(content_path, style_path)
    await model.load_images()
    await model.run_style_transfer()
    image = await model.image_output()
    output_path = f'images/output{user_id}.jpg'
    image.save(output_path)
    return output_path

@dp.message_handler(commands=['start'], state="*")
async def content_request(message: types.Message, state: FSMContext):
    await state.finish()
    await message.answer(f'Приветствую, {message.from_user.first_name}!\n'
                         'Пришлите фото-контент, которое хотите изменить')
    for admin in admins_id:
        try:
            text = f'User {message.from_user.full_name} started bot'
            await dp.bot.send_message(chat_id=admin, text=text)
        except Exception as err:
            print(err)
    await UserData.waiting_for_content_photo.set()

@dp.message_handler(content_types=["photo"], state=UserData.waiting_for_content_photo)
async def style_request(message: types.Message, state: FSMContext):
    if message.content_type!='photo':
        await message.answer("Пожалуйста, пришлите фото-контент")
        return
    await message.photo[-1].download(destination="photos/")
    await state.update_data(content_photo_id=message.photo[-1].file_id)
    await message.answer("Пришлите фото-стиль, в котором хотите увидеть контент")
    await UserData.next()

@dp.message_handler(content_types=["photo"], state=UserData.waiting_for_style_photo)
async def photo_loaded(message: types.Message, state: FSMContext):
    if message.content_type != 'photo':
        await message.answer("Пожалуйста, пришлите фото-стиль")
        return
    await message.photo[-1].download(destination="photos/")
    user_data = await state.get_data()
    content_file = await dp.bot.get_file(user_data['content_photo_id'])
    content_path = "photos/" + content_file['file_path']
    style_file = await dp.bot.get_file(message.photo[-1].file_id)
    style_path = "photos/" + style_file['file_path']
    await message.answer("Ждите, фото обрабатываются. Время обработки зависит от размера изображения...")
    output_path = await style_transfer(content_path, style_path, message.from_user.id)
    photo = open(output_path, 'rb')
    await dp.bot.send_photo(chat_id=message.from_user.id, photo=photo)
    await message.answer("Готово!")
    for admin in admins_id:
        try:
            text = f'User {message.from_user.full_name} finished bot'
            await dp.bot.send_message(chat_id=admin, text=text)
        except Exception as err:
            print(err)
    if os.path.isfile(content_path):
        os.remove(content_path)
    if os.path.isfile(style_path):
        os.remove(style_path)
    if os.path.isfile(output_path):
        os.remove(output_path)
    await state.finish()


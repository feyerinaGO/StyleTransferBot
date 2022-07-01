## **StyleTransferBot**

Телеграм бот по переносу стилей изображений.

Библиотеки для запуска: aiogram, python-dotenv

Запускается из консоли командой python app.py

Папка photos - папка, в которую сохраняются изображения от пользователя

Папка images - папка, в которую сохраняется полученное изображение.

Папка nmodel: всё, что относится к нейронной сети и, собственно, переносу стиля.

В папке nmodel>model лежит предобученная модель vgg19. В коде прописан выбор: если папка пуста, то загружать из torchvision (из папки быстрее)

Запуск бота - файл app.py

Основыне команды бота - файл handlers>users>start.py
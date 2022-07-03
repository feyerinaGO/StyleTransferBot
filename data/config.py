import os
from dotenv import load_dotenv


load_dotenv()

BOT_TOKEN = str(os.getenv("BOT_TOKEN"))
HEROKU_APP_NAME = str(os.getenv("HEROKU_APP_NAME"))

admins_id = [
    567790058
]

# webhook settings
WEBHOOK_HOST = f'https://{HEROKU_APP_NAME}.herokuapp.com'
WEBHOOK_PATH = f'/webhook/{BOT_TOKEN}'
WEBHOOK_URL = f'{WEBHOOK_HOST}{WEBHOOK_PATH}'

# webserver settings
WEBAPP_HOST = '0.0.0.0'
WEBAPP_PORT = int(os.getenv('PORT', default=8000))
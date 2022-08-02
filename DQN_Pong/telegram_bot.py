import keys 
import time
from telegram import Bot
import datetime

# Define a bot class
class AtariBot:
    '''
        Bot de Telegram
    '''
    def __init__(self):
        self.token = keys.TOKEN
        self.last_message = 0
        self.bot = Bot(token=keys.TOKEN)

    def send_msg(self, text):
        self.bot.send_message(chat_id=keys.CHAT_ID, text=text)
        return

    def send_photo(self,photo):
        
        self.bot.send_photo(keys.CHAT_ID, photo=photo)
        return
    def clear_chat(self):
       for i in range(10000,0):
        try:
            self.bot.delete_message(chat_id=keys.CHAT_ID, message_id=i)
        except:
            pass
        return
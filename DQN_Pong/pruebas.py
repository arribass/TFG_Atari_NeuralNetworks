import telegram_bot as tb
import keys

aaa = tb.AtariBot()
for i in range(10000):
    try:
        aaa.bot.delete_message(chat_id=keys.CHAT_ID, message_id=i)
    except:
        pass
# PongBot.send_msg("Hola, soy un bot de Telegram")
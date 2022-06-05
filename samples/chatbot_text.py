"""
Chatterbot
	Installation:
		$ pip install chatterbot==1.0.2
		$ pip install chatterbot-corpus

		Check language:
			/home/omarjg/anaconda3/envs/cursopln/lib/python3.8/site-packages/chatterbot_corpus/data/spanish

	Reference:
		https://github.com/gunthercox/chatterbot-corpus/tree/master/chatterbot_corpus/data/spanish
		https://inloop.github.io/sqlite-viewer/
"""
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer
import logging
logging.basicConfig(filename='chatbotlog.log', level=logging.DEBUG)

#Instancia de chatbot
#~ chatbot = ChatBot('miniBot')

chatbot = ChatBot(
    'miniBot',
    storage_adapter='chatterbot.storage.SQLStorageAdapter',
    logic_adapters=[
        'chatterbot.logic.MathematicalEvaluation',
        'chatterbot.logic.TimeLogicAdapter',
        'chatterbot.logic.BestMatch'
    ],
    database_uri='sqlite:///database.db'
)

#Entrenamiento
trainer = ChatterBotCorpusTrainer(chatbot)
trainer.train('chatterbot.corpus.spanish')

#Flujo de conversación
while True:
	request = input('Yo: ')
	response = chatbot.get_response(request)
	print('miniBot: ', response)
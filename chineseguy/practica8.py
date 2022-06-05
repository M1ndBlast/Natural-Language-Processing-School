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
import logging
import os

import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer, SnowballStemmer
import unidecode
from chatterbot import ChatBot, filters, preprocessors
from chatterbot.conversation import Statement
from chatterbot.trainers import ChatterBotCorpusTrainer
from MyLogicAdapter import MyLogicAdapter
from Scrapper import transleDeepl

nltk.download('averaged_perceptron_tagger') 
nltk.download('omw-1.4')

logging.basicConfig(filename='logs/chatbotlog.log', level=logging.DEBUG)

#Function that convert all of text to lower case and change the accented letters
def lemmatize(statement: Statement):
	#print(f"lemmatize {statement.text} ->", end=" ")
	accents={'á':'a','é':'e','í':'i','ó':'o', 'ú':'u','ä':'a','ë':'e','ï':'i','ö':'o','ü':'u','â':'a','ê':'e','î':'i','ô':'o','û':'u','à':'a','è':'e','ì':'i','ò':'o','ù':'u'}
	sentence=statement.text
	sentence=sentence.lower()
	for accent in accents:
		if accent in sentence:
			statement.text=sentence.replace(accent,accents[accent])
	
	#print(f"{statement.text}")
	return statement


chatbot = ChatBot(
	'Chinesse Guy',
	storage_adapter='chatterbot.storage.SQLStorageAdapter',
	logic_adapters=[{
		'import_path':'chatterbot.logic.BestMatch',
	}, {
		'import_path':'ScrapringLogicAdapter.ScrapringLogicAdapter',
	}],
	filters=[filters.get_recent_repeated_responses],

	database_uri='sqlite:///database/chinesse_foreign.db'
)

chatbot.preprocessors = [ preprocessors.clean_whitespace, lemmatize ]

#Entrenamiento
trainer = ChatterBotCorpusTrainer(chatbot)
trainer.train('chatterbot.corpus.spanish')

#Flujo de conversación
os.system("clear")
print("""
¡Felicidades! Acabas de llegar a China, pero...
¿Cómo que no sabes Chino?
Bueno, ¡Mira! por ahí se ve un chico, se ve muy amable, puede que sepa algo de español
¡Hablalé!
""")
while True:
	request = input('Yo: ')
	print("Chinito...", end="")
	response = chatbot.get_response(request)
	chinse_response = transleDeepl(response.text)
	chinse_response = "\rChinito: "+(" "*int(len(chinse_response)*1/2))+chinse_response
	print(chinse_response)
	print(" "*(len(chinse_response)+2)+f"\033[A ({response})")
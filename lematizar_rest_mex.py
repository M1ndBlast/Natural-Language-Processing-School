from Preprocesamiento.lematizador import lematizar as lematyze

import pandas as pd
from sklearn.model_selection import train_test_split
#~ from sklearn.model_selection import KFold
import numpy as np
import sys
import pickle
import re

"""
A - Adjective
C - Conjunction
D - Determiner
N - Noun
	C - Common
	P - Proper
P - Pronoun
R - Adverb
S - Adposition
V - Verb
Z - Number
W - Date
I - Interjection
F - Symbols
"""
TAGS = ['NC', 'V']
STOPWORDS = ['do', 'año']

def preprocessor(slist:list):
	result = []
	preprocessing = lematyze("\n".join(slist))
	for ln in preprocessing:
		for w in ln:
			if w.get_lemma() not in STOPWORDS:
				#print(f"{w.get_form()} [{w.get_tag()}] -> {w.get_lemma()}")
				for tag in TAGS:
					if w.get_tag().startswith(tag):
						result.append(w.get_lemma())
	return " ".join(result)

class data_set_polarity:
	def __init__(self, X_train, y_train, X_test, y_test):
		self.X_train = X_train
		self.y_train = y_train
		self.X_test = X_test
		self.y_test = y_test

class data_set_attraction:
	def __init__(self, X_train, y_train, X_test, y_test):
		self.X_train = X_train
		self.y_train = y_train
		self.X_test = X_test
		self.y_test = y_test
		
		
def generate_train_test(file_name):
	pd.options.display.max_colwidth = 200				

	#Lee el corpus original del archivo de entrada y lo pasa a una DataFrame
	df = pd.read_excel(file_name, dtype=str)
	df=df.replace(to_replace=np.NaN,value="")
	#~ print (df)
	X = df.drop(['Polarity', 'Attraction'],axis=1).values   
	y_polarity = df['Polarity'].values
	y_attraction = df['Attraction'].values

	X = np.array(list(map(preprocessor, X)))
	
	#~ #Separa el corpus cargado en el DataFrame en el 90% para entrenamiento y el 10% para pruebas
	X_train, X_test, y_train_polarity, y_test_polarity = train_test_split(X, y_polarity, test_size=0.2, random_state=0)
	#~ print (X_train)
	#~ print (X_train.shape)
	#~ print (y_train_polarity)
	#~ print (y_train_polarity.shape)
	#~ print(X_test.shape)
	X_train, X_test, y_train_attraction, y_test_attraction = train_test_split(X, y_attraction, test_size=0.2, random_state=0)
	#~ print (X_train)
	#~ print (y_train_attraction)
	
	X_train_lematizado = []
	for index in range(len(X_train)):
		line = X_train[index]
	
	#~ for line in corpus_polarity.X_train:
		string = ''
		title = str(line[0])
		opinion = str(line[1])
		opinion = re.sub('\n', ' ', opinion)
		
		#~ print (line)
		
		string = title + ' ' + opinion
		string = string.lower()
		#~ print (string)
		cadena_lematizada = lematizar(string)
		
		if (len(cadena_lematizada) == 0):
			cadena_lematizada = string
		
		#~ print (cadena_lematizada)
		print ('------------Entrenamiento---------------------') 
		X_train_lematizado.append(cadena_lematizada)
	
	X_test_lematizado = []
	for index in range(len(X_test)):
		line = X_test[index]
	
	#~ for line in corpus_polarity.X_train:
		string = ''
		title = str(line[0])
		opinion = str(line[1])
		opinion = re.sub('\n', ' ', opinion)
		
		#~ print (line)
		
		string = title + ' ' + opinion
		string = string.lower()
		#~ print (string)
		cadena_lematizada = lematizar(string)
		
		if (len(cadena_lematizada) == 0):
			cadena_lematizada = string
		
		#~ print (cadena_lematizada)
		print ('--------------Prueba-------------------') 
		X_test_lematizado.append(cadena_lematizada)
	
	
	#~ return (data_set_polarity(X_train, y_train_polarity, X_test, y_test_polarity), data_set_attraction(X_train, y_train_attraction, X_test, y_test_attraction))
	return (data_set_polarity(X_train_lematizado, y_train_polarity, X_test_lematizado, y_test_polarity), data_set_attraction(X_train_lematizado, y_train_attraction, X_test_lematizado, y_test_attraction))
	
	#~ print (X_train.shape)
	#~ print (X_train)
	#~ print (y_train.shape)
	#~ print (y_train)
	#~ print (X_test.shape)
	#~ print (X_test)
	#~ print (y_test.shape)
	#~ print (y_test)
	
	
if __name__=='__main__':
	corpus_polarity, corpus_attraction = generate_train_test('Corpus/Rest_Mex_2022_Sentiment_Analysis_Track_Train.xlsx')
	
	#Guarda el dataset en pickle
	
	dataset_file = open ('Corpus/Rest_Mex_2022_Sentiment_Analysis_Track_Train-Polarity-Preprocessed.pkl','wb')
	pickle.dump(corpus_polarity, dataset_file)
	dataset_file.close()
	
	print (corpus_attraction.X_train[0])
	dataset_file = open ('Corpus/Rest_Mex_2022_Sentiment_Analysis_Track_Train-Attraction-Preprocessed','wb')
	pickle.dump(corpus_attraction, dataset_file)
	dataset_file.close()








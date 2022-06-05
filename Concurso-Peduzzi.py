import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

from Preprocesamiento.lematizador import lematizar as lematyze


class data_set:
	def __init__(self, X_train:np.ndarray, y_train:np.ndarray, X_test:np.ndarray, y_test:np.ndarray):
		self.X_train = X_train.tolist()
		self.X_test = X_test.tolist()
		self.y_train = y_train.tolist()
		self.y_test = y_test.tolist()

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

def generate_train_test(file_name, test_size=0.1):
	pd.options.display.max_colwidth = 200				

	#Lee el corpus original del archivo de entrada y lo pasa a una DataFrame
	df = pd.read_excel(file_name, dtype=str, nrows=5)
	
	X = df.drop(['Polarity', 'Attraction'],axis=1).values   
	y_polarity = df['Polarity'].values
	y_attraction = df['Attraction'].values

	X = np.array(list(map(preprocessor, X)))
	
	#~ #Separa el corpus cargado en el DataFrame en el 80% para entrenamiento y el 20% para pruebas
	X_train, X_test, y_train_polarity, y_test_polarity = train_test_split(X, y_polarity, test_size=test_size, random_state=0)
	X_train, X_test, y_train_attraction, y_test_attraction = train_test_split(X, y_attraction, test_size=test_size, random_state=0)
	
	#	Polarity,	Attraction
	return (data_set(X_train, y_train_polarity, X_test, y_test_polarity), data_set(X_train, y_train_attraction, X_test, y_test_attraction))

if __name__=='__main__':
	corpus_polarity, corpus_attraction = generate_train_test(
		file_name='Corpus/Rest_Mex_2022_Sentiment_Analysis_Track_Train.xlsx', 
		test_size=0.2
	)

	# Entrenamiento
	vectorize_train = CountVectorizer()
	X_train_df = pd.DataFrame(
		vectorize_train.fit_transform(corpus_attraction.X_train).toarray(), 
		columns=vectorize_train.get_feature_names_out()
	)
	X_train_df['_Polarity'] = corpus_polarity.y_train
	X_train_df['_Attraction'] = corpus_attraction.y_train


	X_train_Hotel_df = X_train_df.loc[X_train_df['_Attraction']=='Hotel']
	X_train_Restaurant_df = X_train_df.loc[X_train_df['_Attraction']=='Restaurant']
	X_train_Attractive_df = X_train_df.loc[X_train_df['_Attraction']=='Attractive']

	print(f"Hotel DF\n{X_train_Hotel_df}")
	print(f"Restaurant DF\n{X_train_Restaurant_df}")
	print(f"Attractive DF\n{X_train_Attractive_df}")
	
	
	X_train_Diff = pd.DataFrame(
		np.subtract(
			np.array(X_train_Hotel_df.drop(['_Polarity','_Attraction'],axis=1)).astype(int), 
			np.array(X_train_Restaurant_df.drop(['_Polarity','_Attraction'],axis=1)).astype(int)), 
		X_train_Hotel_df.columns().to_list()
	)
	print(f"Diff DF\n{X_train_Diff}")


	# Prueba
	vectorize_test = CountVectorizer()
	X_test_df = pd.DataFrame(
		vectorize_test.fit_transform(corpus_attraction.X_test).toarray(), 
		columns=vectorize_test.get_feature_names_out()
	)
	X_test_df['_Polarity'] = corpus_polarity.y_test
	X_test_df['_Attraction'] = corpus_attraction.y_test

"""
	Concurso Rest-Mex-2022 Sentiment Analysis
"""
import re, os, pickle, numpy as np, pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix)
from Preprocesamiento.lematizador import lematizar as lematyze

class data_set:
	def __init__(self, X_train, y_polarity_train, y_attraction_train, X_test, id_test):
		self.X_train 			= X_train
		self.y_polarity_train 	= y_polarity_train
		self.y_attraction_train = y_attraction_train
		self.X_test 			= X_test
		self.id_test 			= id_test

#~ Preprocesamiento
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
#TAGS = ['NC', 'V']
#STOPWORDS = ['do', 'año']

def preprocessor(slist:list, i, sz):
	print(f"\r{i}/{sz}", end="")
	result = []
	preprocessing = lematyze("\n".join(slist))
	for ln in preprocessing:
		for w in ln:
			#if w.get_lemma() not in STOPWORDS:
			#	print(f"{w.get_form()} [{w.get_tag()}] -> {w.get_lemma()}")
			#	for tag in TAGS:
			#		if w.get_tag().startswith(tag):
			result.append(w.get_lemma())
	return " ".join(result)

def generate_train_test(filename_train:str, filename_test:str):
	#pd.options.display.max_colwidth = 200				
	"""
	Structure of Rest_Mex_2022_Sentiment_Analysis_Track_Train.xlsx
		Title
		Opinion
		Polarity
		Attraction

	Structure of Rest_Mex_2022_Sentiment_Analysis_Track_Test.xlsx
		Id
		Title
		Opinion
	"""

	#~ Reading the Training Corpus
	df_train = pd.read_excel(filename_train, dtype=str).replace(to_replace=np.NaN,value="")
	X_train = df_train.drop(['Polarity', 'Attraction'],axis=1).values   
	y_polarity_train = df_train['Polarity'].values
	y_attraction_train = df_train['Attraction'].values

	if not (os.path.exists('tmp_Xtrain_preprocessed.pkl')):
		print("\nPreprocessing Train Data")
		X_train = np.array(list(map(lambda y: preprocessor(y[1], y[0], len(X_train)), enumerate(X_train))))
#		with open('tmp_Xtrain_preprocessed.pkl', 'w') as tmpXtrain:
#			pickle.dump(X_train, tmpXtrain)
		print("\nsaved!")
	else:
		print("\nPreprocessed Train Data Loaded")
#		with open('tmp_Xtrain_preprocessed.pkl', 'r') as tmpXtrain:
#			X_train = pickle.load(tmpXtrain)

	#~Reading the Test Corpus
	df_test = pd.read_excel(filename_test, dtype=str).replace(to_replace=np.NaN,value="")
	X_test = df_test.drop(['Id'],axis=1).values   
	id_test = df_test['Id'].values


	if not (os.path.exists('tmp_Xtest_preprocessed.pkl')):
		print("\nPreprocessing Test Data")
		X_test = np.array(list(map(lambda y: preprocessor(y[1], y[0], len(X_test)), enumerate(X_test))))
#		with open('tmp_Xtest_preprocessed.pkl', 'w') as tmpXtest:
#			pickle.dump(X_test, tmpXtest)
		print("\nsaved!")
	else:
		print("\nPreprocessed Test Data Loaded")
#		with open('tmp_Xtest_preprocessed.pkl', 'r') as tmpXtest:
#			X_test = pickle.load(tmpXtest)


	return data_set(X_train, y_polarity_train, y_attraction_train, X_test, id_test)
	

"""
	2. Save as .pkl each
	3. Limpiar Codigo
	4. Limpiar impresiones
	5. In
"""

if __name__=='__main__':
	#~ Generación/Carga Corpus Rest_Mex_2022_Sentiment_Analysis_Track
	if not(os.path.exists('Corpus/Rest_Mex-2022-FULL.pkl')):
		corpus_RestMex2022 = generate_train_test(
			filename_train = 'Corpus/Rest_Mex_2022_Sentiment_Analysis_Track_Train.xlsx',
			filename_test  = 'Corpus/Rest_Mex_2022_Sentiment_Analysis_Track_Test.xlsx'
		)
		with open ('Corpus/Rest_Mex-2022-FULL.pkl','wb') as dataset_file:
			pickle.dump(corpus_RestMex2022, dataset_file)
	else:
		with open ('Corpus/Rest_Mex-2022-FULL.pkl','rb') as dataset_file:
			corpus_RestMex2022 = pickle.load(dataset_file)

	#~ Attraction Process
	print("Attraction Process")
	target_names = ['Restaurant','Hotel','Attractive']

	vectorizador_binario = CountVectorizer(binary=True)
	vectorizador_binario_fit = vectorizador_binario.fit(corpus_RestMex2022.X_train)
	X_train = vectorizador_binario_fit.transform(corpus_RestMex2022.X_train)
	y_attraction_train = corpus_RestMex2022.y_attraction_train
	#print (vectorizador_binario.get_feature_names_out())
	#print (X_train.shape)#sparse matrix
	clf = LogisticRegression(max_iter=10000)
	clf.fit(X_train, y_attraction_train)

	X_test = vectorizador_binario_fit.transform(corpus_RestMex2022.X_test)

	y_attraction_test = clf.predict(X_test)
	
	#Stadistics [Useless]
	print(f"y_pred:\n{y_pred}")
	print(accuracy_score(y_test, y_pred))
	print(confusion_matrix(y_test, y_pred,labels=['Restaurant','Hotel','Attractive']))
	print(classification_report(y_test, y_pred, target_names=target_names))
	
		
	#~ Polarity Process
	print("Polarity Process")
	target_names = ['1','2','3','4','5']

	vectorizador_frecuencia = CountVectorizer()
	vectorizador_frecuencia_fit = vectorizador_frecuencia.fit(corpus_RestMex2022.X_train)
	X_train = vectorizador_frecuencia_fit.transform(corpus_RestMex2022.X_train)
	y_polarity_train = corpus_RestMex2022.y_polarity_train
	#print(vectorizador_binario_fit.get_feature_names_out()) #Este es de atraction
	#print(X_test.shape)#sparse matrix
	
	clf = LogisticRegression(max_iter=10000)
	clf.fit(X_train, y_polarity_train)
	
	X_test = vectorizador_frecuencia_fit.transform(corpus_RestMex2022.X_test)

	y_polarity_test = clf.predict(X_test)
	
	# Stadistics [Useless]
	print(y_pred)
	print(accuracy_score(y_test, y_pred))
	print(confusion_matrix(y_test, y_pred,labels=target_names))
	print(classification_report(y_test, y_pred, target_names=target_names))
	
	

	#~ Writting Results
	print("Writting Results")
	with open('Result-Rest-Mex-2022-Sentiment-Analysis.txt', 'w') as fResult:
		sz = len(corpus_RestMex2022.id_test)
		for i in corpus_RestMex2022.id_test:
			print(f"i{re.split('.', i)[0]}")
			i = int(re.split('.', i)[1])
			print(f"\r{i}/{sz}", end="")
			fResult.write(f'"sentiment"\t"{i}"\t"{y_polarity_test[i]}"\t"{y_attraction_test[i]}"\n')

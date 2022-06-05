import pandas as pd
from sklearn.model_selection import train_test_split
#~ from sklearn.model_selection import KFold
import numpy as np
import sys
import pickle


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
	df = pd.read_excel(file_name)
	#~ print (df)
	X = df.drop(['Polarity', 'Attraction'],axis=1).values   
	y_polarity = df['Polarity'].values
	y_attraction = df['Attraction'].values
	
	#~ #Separa el corpus cargado en el DataFrame en el 90% para entrenamiento y el 10% para pruebas
	X_train, X_test, y_train_polarity, y_test_polarity = train_test_split(X, y_polarity, test_size=0.1, random_state=0)
	#~ print (X_train)
	#~ print (X_train.shape)
	#~ print (y_train_polarity)
	#~ print (y_train_polarity.shape)
	#~ print(X_test.shape)
	X_train, X_test, y_train_attraction, y_test_attraction = train_test_split(X, y_attraction, test_size=0.1, random_state=0)
	#~ print (X_train)
	#~ print (y_train_attraction)
	
	return (data_set_polarity(X_train, y_train_polarity, X_test, y_test_polarity), data_set_attraction(X_train, y_train_attraction, X_test, y_test_attraction))
	
	#~ print (X_train.shape)
	#~ print (X_train)
	#~ print (y_train.shape)
	#~ print (y_train)
	#~ print (X_test.shape)
	#~ print (X_test)
	#~ print (y_test.shape)
	#~ print (y_test)
	
	
if __name__=='__main__':
	corpus_polarity, corpus_attraction = generate_train_test('Rest_Mex_2022_Sentiment_Analysis_Track_Train.xlsx')
	
	#~ print (corpus_polarity.X_train)
	#~ print (corpus_polarity.y_train)
	#~ print (corpus_attraction.X_train)
	#~ print (corpus_attraction.y_train)
	
	#Guarda el dataset en pickle
	dataset_file = open ('corpus_polarity.pkl','wb')
	pickle.dump(corpus_polarity, dataset_file)
	dataset_file.close()
	
	dataset_file = open ('corpus_polarity.pkl','rb')
	my_corpus_polarity = pickle.load(dataset_file)
	print ("-----------------------------------------------")
	print (my_corpus_polarity.X_train)













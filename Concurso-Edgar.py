from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
import os, pickle
from sklearn.linear_model import LogisticRegression
from  sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import svm
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.naive_bayes import MultinomialNB

def vectorial(X_train,X_test,y_train):
	clf=svm.SVC(kernel='linear',C=100)
	clf.fit(X_train,y_train)
	y_predict=clf.predict(X_test)
	return y_predict

def MAE(y_test_polarity:list,y_predict_polarity:list):
	y_test_polarity=np.array(y_test_polarity)
	y_predict_polarity=np.array(y_predict_polarity)
#	print(f"polaridad verdadera: {y_test_polarity}")
#	print(f"polaridad predicha: {y_predict_polarity}")
	mae=(1/len(y_test_polarity))*(sum(abs(np.subtract(y_test_polarity,y_predict_polarity))))
	#print(f"valor absoluto de la diferencia: {abs(np.subtract(y_test_polarity,y_predict_polarity))}")
	#print(f"suma de la diferencia: {sum(abs(np.subtract(y_test_polarity,y_predict_polarity)))}")
#	print(f"El valor de MAE es: {mae}")
	return mae

def clfKnn_bin(X_train,y_train,X_test,y_test,lenx):
	
	#dataframe = pd.read_csv(r"reviews_sentiment.csv",sep=';')

	#X = dataframe[['wordcount','sentimentValue']].values
	#y = dataframe['Star Rating'].values

	#X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
	
	n_neighbors = 100
	print("llegó aquí")
	knn = KNeighborsClassifier(n_neighbors)
	print("llegó aquí x2")
	knn.fit(X_train, y_train)
	print("llegó aquí x3")
	y_predict=knn.predict(X_test)
	print("llegó aquí x4")
	print('Accuracy of K-NN classifier on training set: {:.2f}'
		.format(knn.score(X_train, y_train)))
	print('Accuracy of K-NN classifier on test set: {:.2f}'
		.format(knn.score(X_test, y_test)))
	return y_predict

def clfKnn(X_train,y_train,X_test,y_test,lenx):
	
	#dataframe = pd.read_csv(r"reviews_sentiment.csv",sep=';')

	#X = dataframe[['wordcount','sentimentValue']].values
	#y = dataframe['Star Rating'].values

	#X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
	scaler = MinMaxScaler()

	X_train = scaler.fit_transform(X_train)
	X_test = scaler.transform(X_test)

	n_neighbors = 500
	print("llegó aquí ")
	knn = KNeighborsClassifier(n_neighbors)
	print("llegó aquí x2")
	knn.fit(X_train, y_train)
	print(X_test)
	print("llegó aquí x3")
	y_predict=knn.predict(X_test)

	print('Accuracy of K-NN classifier on training set: {:.2f}'
		.format(knn.score(X_train, y_train)))
	print('Accuracy of K-NN classifier on test set: {:.2f}'
		.format(knn.score(X_test, y_test)))
	return y_predict

class data_set_polarity:
	def __init__(self, X_train, y_train, X_test, y_test):
		self.X_train = X_train
		self.y_train = y_train
		self.X_test = X_test
		self.y_test = y_test
        
if not (os.path.exists('Corpus/corpus_polarity.pkl')):
	print ('no se ha generado el corpus lematizado')
else:
	corpus_file = open ('Corpus/corpus_polarity.pkl','rb')
	corpus_polarity = pickle.load(corpus_file)
	corpus_file.close()
#Binario
vectorizador_binario = CountVectorizer(binary=True)
vectorizador_binario_fit = vectorizador_binario.fit(corpus_polarity.X_train)
X_train_bin = vectorizador_binario_fit.transform(corpus_polarity.X_train)
X_test_bin = vectorizador_binario_fit.transform(corpus_polarity.X_test)

#Frecuencia
vectorizador_frecuence = CountVectorizer()
vectorizador_frecuence_fit = vectorizador_frecuence.fit(corpus_polarity.X_train)
X_train_frec = vectorizador_frecuence_fit.transform(corpus_polarity.X_train)
X_test_frec = vectorizador_frecuence_fit.transform(corpus_polarity.X_test)

#Tfidf
vectorizador_tfidf=TfidfVectorizer()
vectorizador_tfidf_fit = vectorizador_tfidf.fit(corpus_polarity.X_train)
X_train_tfidf = vectorizador_tfidf_fit.transform(corpus_polarity.X_train)
X_test_tfidf = vectorizador_tfidf_fit.transform(corpus_polarity.X_test)

y_train_polarity = corpus_polarity.y_train
y_test = corpus_polarity.y_test

#Bayes
print("Bayes binarizado")
clf = MultinomialNB()
clf.fit(X_train_bin, y_train_polarity)
y_pred=clf.predict(X_test_bin)
print(accuracy_score(y_test, y_pred))
target_names = [1,2,3,4,5]
print(confusion_matrix(y_test, y_pred,labels=target_names))
print(classification_report(y_test, y_pred, target_names=target_names))
print(f"El valor de MAE es: {MAE(y_test, y_pred)}")

print("Bayes Frecuenciado")
clf = MultinomialNB()
clf.fit(X_train_frec, y_train_polarity)
y_pred=clf.predict(X_test_frec)
print(accuracy_score(y_test, y_pred))
#target_names = ['1','2','3','4','5']
print(confusion_matrix(y_test, y_pred,labels=target_names))
print(classification_report(y_test, y_pred, target_names=target_names))
print(f"El valor de MAE es: {MAE(y_test, y_pred)}")

print("Bayes tfidf")
clf = MultinomialNB()
clf.fit(X_train_tfidf, y_train_polarity)
y_pred=clf.predict(X_test_tfidf)
print(accuracy_score(y_test, y_pred))
#target_names = ['1','2','3','4','5']
print(confusion_matrix(y_test, y_pred,labels=target_names))
print(classification_report(y_test, y_pred, target_names=target_names))
print(f"El valor de MAE es: {MAE(y_test, y_pred)}")


"""
print("Regresión en binario")
clf_polarity = LogisticRegression(max_iter=10000)
clf_polarity.fit(X_train_bin, y_train_polarity)
y_pred = clf_polarity.predict(X_test_bin)
print (y_pred)
print(accuracy_score(y_test, y_pred)))
target_names = ['1','2','3','4','5']
print(confusion_matrix(y_test, y_pred,labels=target_names))
print(classification_report(y_test, y_pred, target_names=target_names))
print(f"El valor de MAE es: {MAE(y_test, y_pred)}")

print("regresión en frecuencia")
clf_polarity_frec = LogisticRegression(max_iter=10000)
clf_polarity_frec.fit(X_train_frec, y_train_polarity)
y_pred = clf_polarity_frec.predict(X_test_frec)
#print(accuracy_score(y_test, y_pred))
print (y_pred)
print(accuracy_score(y_test, y_pred))
target_names = ['1','2','3','4','5']
print(confusion_matrix(y_test, y_pred,labels=target_names))
print(classification_report(y_test, y_pred, target_names=target_names))
print(f"El valor de MAE es: {MAE(y_test, y_pred)}")

print("regresion en tfidf")
clf_polarity_tfidf = LogisticRegression(max_iter=10000)
clf_polarity_tfidf.fit(X_train_tfidf, y_train_polarity)
y_pred = clf_polarity_tfidf.predict(X_test_frec)
print (y_pred)
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred,labels=target_names))
target_names = ['1','2','3','4','5']
print(classification_report(y_test, y_pred, target_names=target_names))
print(f"El valor de MAE es: {MAE(y_test, y_pred)}")


print("Soporte vectorial")
print("binario")
y_pred_support=vectorial(X_train_bin,X_test_bin,y_train_polarity)
print(accuracy_score(y_test, y_pred_support))
print(confusion_matrix(y_test, y_pred_support,labels=[1,2,3,4,5]))
target_names = ['1','2','3','4','5']
print(classification_report(y_test, y_pred_support, target_names=target_names))
print("frecuencia")
y_pred_support=vectorial(X_train_frec,X_test_frec,y_train_polarity)
print(accuracy_score(y_test, y_pred_support))
print(confusion_matrix(y_test, y_pred_support,labels=[1,2,3,4,5]))
target_names = ['1','2','3','4','5']
print(classification_report(y_test, y_pred_support, target_names=target_names))
print("tfidf")
y_pred_support=vectorial(X_train_tfidf,X_test_tfidf,y_train_polarity)
print(accuracy_score(y_test, y_pred_support))
print(confusion_matrix(y_test, y_pred_support,labels=[1,2,3,4,5]))
target_names = ['1','2','3','4','5']
print(classification_report(y_test, y_pred_support, target_names=target_names))
"""
"""
print("Knn")
print("binario")
y_pred_knn=clfKnn_bin(X_train_bin,y_train_polarity,X_test_bin,y_test)
print(accuracy_score(y_test, y_pred_knn))
print(confusion_matrix(y_test, y_pred_knn,labels=[1,2,3,4,5]))
target_names = ['1','2','3','4','5']
print(classification_report(y_test, y_pred_knn, target_names=target_names))

print("frecuencia")
y_pred_knn=clfKnn(X_train_frec,y_train_polarity,X_test_frec,y_test)
print(accuracy_score(y_test, y_pred_knn))
print(confusion_matrix(y_test, y_pred_knn,labels=[1,2,3,4,5]))
target_names = ['1','2','3','4','5']
print(classification_report(y_test, y_pred_knn, target_names=target_names))

print("tfidf")
y_pred_knn=clfKnn_bin(X_train_tfidf,y_train_polarity,X_test_tfidf,y_test)
print(accuracy_score(y_test, y_pred_knn))
print(confusion_matrix(y_test, y_pred_knn,labels=[1,2,3,4,5]))
target_names = ['1','2','3','4','5']
print(classification_report(y_test, y_pred_knn, target_names=target_names))
"""

import os
import sys
import numpy as np

from lxml import etree
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
							 confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

"""
	Taeas a resolver:
		- Clasificación de genere
		- Clasificación de edad
	Cada tarea debe de hacer:
		- Aplique el algoritmo de LSA para crear representaciones vectoriales del texto con distinto número de tópicos
		- Proponga distintos conjuntos de características como las utilizadas en el ejemplo base de la práctica (debe ser almenos uno distinto)
		- Una la representación creada por LSA y los conjuntos de características propuestos
		- Utilice las características unidas para entrenar un modelo de regresiónlogística
		- Utilice el modelo entrenado para clasificar las instancias del conjunto de pruebas (20% de los datos)
		- Calcule la exactitud del modelo
"""
"""
	Evidencias:
		- Código fuente
		- PDF con:
			1. Una tabla con los números de tópicos y características probadas en el conjunto de desarrollo y los valores de exactitud (accuracy),
				precisión, recall y F-measure obtenidos por prueba en cada tarea de clasificación
			2. La configuración de tópicos y características que mejores resultados obtuvo en el conjunto de desarrollo en cada tarea de clasificación
			3. Los valores de exactitud (accuracy), precisión, recall y F-measure obtenidos por los modelos en cada tarea de clasificación en el conjunto de prueba
			4. Matriz de confusión de los valores predichos vs los reales en el conjunto de prueba de cada tarea de clasificación
"""


class Capitals(BaseEstimator, TransformerMixin):
	# feature that counts capitalized characters in a tweet
	def fit(self, X, Y=None):
		return self

	def transform(self, X):
		return [[sum(1 for ch in doc if ch.isupper())] for doc in X]


class Patterns(BaseEstimator, TransformerMixin):
	# feature that counts occurences for a range of patterns
	def __init__(self, patterns):
		self.patterns = patterns

	def fit(self, X, Y=None):
		return self

	def transform(self, X):
		return [[doc.lower().count(pattern)/len(doc) for pattern in self.patterns] for doc in X]

# El vector se crea fuera, antes de feature union, se pasa como param dentro de feature union
class LSA():
	def __init__(self, vec_tfidf):
		self.vec_tfidf = vec_tfidf
		self.svd = TruncatedSVD(500)
		self.vectorizer = TfidfVectorizer()
		
		
	def fit(self, X, Y=None):
		self.svd = self.svd.fit(self.vectorizer.fit_transform(X))
		return self
		
		
	def transform(self, X):
		return self.svd.transform(self.vectorizer.transform(X))


def main(argv):
	trainDir = argv[1]
	testDir = argv[2]

	target_age_names = ['18-24', '25-34', '35-49', '50-XX']
	target_gender_names = ['F', 'M']

	# Load Dataset
	X_train, y_gender_train, y_age_train = loadCorpus(trainDir)
	X_train, X_test, y_age_train, y_age_test, y_gender_train, y_gender_test = train_test_split(X_train, y_age_train, y_gender_train, test_size=0.2, shuffle=True, random_state=0)
	"""
	X_train, y_gender_train, y_age_train = loadCorpus(trainDir)
	X_test, y_gender_test, y_age_test = loadCorpus(testDir)
	
	"""
	print(f"X_train:\t{len(X_train)}\tX_test:\t\t{len(X_test)}\ny_gender_train:\t{len(y_gender_train)}\ty_gender_test:\t{len(y_gender_test)}\ny_age_train:\t{len(y_age_train)}\ty_age_test:\t{len(y_age_test)}")

	modelPipeline(X_train, X_test, y_age_train, y_age_test, target_age_names)
	print("*"*40)

	"""
	# Clasification LSA
	vectorizer = TfidfVectorizer()
	vectors_train = vectorizer.fit_transform(X_train)
	vectors_test = vectorizer.transform(X_test)

	print("Age")
	print("*"*40)
	classificationLSA(vectors_train, vectors_test,
					  y_age_train, y_age_test, target_age_names)
	print("*"*40)
	print("Gender")
	print("*"*40)
	classificationLSA(vectors_train, vectors_test,
					  y_gender_train, y_gender_test, target_gender_names)
	"""


def loadCorpus(directory):
	X, Ygender, Yage = [], [], []
	tweets = loadData(directory)
	truths = loadTruth(directory)
	# print(f"tweets {len(tweets)}")
	# print(f"truths {len(truths)}")
	for author, tweet in tweets.items():
		X.extend(tweet)
		# 015f2a45-47f5-48bf-904c-264acc3475df:::F:::18-24:::0.1:::0.1:::0.4:::0.2:::0.1
		# truths[split[0]]=(split[1],split[2])
		Ygender.extend([truths[author][0]] * len(tweet))
		Yage.extend([truths[author][1]] * len(tweet))

	return X, Ygender, Yage


def loadData(directory):
	# returns a dictionary of tweets per author
	tweets = {}
	for filename in os.listdir(directory):
		if filename.endswith('xml'):
			handle = open(os.path.join(directory, filename), 'rb')
			tree = etree.fromstring(handle.read())
			documents = tree.xpath('//document')
			tweets[filename[:-4]] = [doc.text.rstrip() for doc in documents]
	return tweets


def loadTruth(directory):
	# 015f2a45-47f5-48bf-904c-264acc3475df:::F:::18-24:::0.1:::0.1:::0.4:::0.2:::0.1
	# returns a dictionary of truth values per author
	truths = {}
	filepath = os.path.join(directory, 'truth.txt')
	handle = open(filepath, 'r')
	for line in handle:
		split = line.split(':::')
		truths[split[0]] = (split[1], split[2])

	return truths


def metrics(y_test, y_pred, target_name=None):
	print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
	print(confusion_matrix(y_test, y_pred, labels=target_name))
	print(classification_report(y_test, y_pred, target_names=target_name))


def modelPipeline(X_train, X_test, y_train, y_test, target_names):

	vectorizer = TfidfVectorizer()
	X_vect_train = vectorizer.fit_transform(X_train)

	features = FeatureUnion([('patterns', Patterns(['.', '!', '?', 'rt', '#', '@', 'http'])),
							 ('svd', LSA(X_vect_train)),
							 ]
							# ~ ,transformer_weights={'tfidf': 3})
							)


	# Train model
	clf = LogisticRegression(max_iter=10000)
	pipeline = Pipeline([('features', features), ('classifier', clf)])
	pipeline.fit(X_train, y_train)

	# Test model
	y_pred = pipeline.predict(X_test)
	print(f"\nPipeline - LogisticRegresion - LSA")
	metrics(y_test, y_pred, target_names)


def classificationLSA(X_train, X_test, y_train, y_test, target_names):

	clf = LogisticRegression(max_iter=10000)
	clf.fit(X_train, y_train)

	y_pred = clf.predict(X_test)
	# print(np.array(y_pred))
	print('vectors_train.shape {}'.format(X_train.shape))
	print('vectors_test.shape {}'.format(X_test.shape))
	print(f"\nLogisticRegresion")
	metrics(y_test, y_pred, target_names)

	svd = TruncatedSVD(500)
	vectors_train_lsa = svd.fit_transform(X_train)
	##~ print (vectors_train_lsa)
	print('vectors_train_lsa.shape {}'.format(vectors_train_lsa.shape))

	clf.fit(vectors_train_lsa, y_train)

	vectors_test_lsa = svd.transform(X_test)
	print('vectors_test_lsa.shape {}'.format(vectors_test_lsa.shape))

	y_pred = clf.predict(vectors_test_lsa)
	# print(np.array(y_pred))

	print(f"\nTruncatedSVD")
	metrics(y_test, y_pred, target_names)


"""
	Execute with:
		python practica7.py .\pan15-author-profiling-training-dataset-spanish-2015-04-23 .\pan15-author-profiling-test-dataset2-spanish-2015-04-23 
"""

if __name__ == '__main__':
	main(sys.argv)

import numpy as np
import pandas as pd
import sys, os, re, pickle, pygad
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer


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

def stopwords(opinions):
	opinions_normalizate=[]
	tags = ['V','N']
	for opinion in opinions:
		preproceced_words = lematyze(opinion)
		opinion_normalizate=[]
		for ln in preproceced_words:
			for w in ln:
				w_tag= w.get_tag()
				if w_tag[0] in tags:
					opinion_normalizate+=w.get_lemma()+" "
					print(f"word: {w.get_form()}\tlemma: {w.get_lemma()}\ttag: {w.get_tag()}")
		opinion_normalizate=opinion_normalizate[:-1]
		opinions_normalizate.append(opinion_normalizate)
	return opinions_normalizate


def vect_repres(X):
	vectorizador_frecuencia = CountVectorizer()
	arr_frec = vectorizador_frecuencia.fit_transform(X).toarray()
	return arr_frec

def index_attraction(list_attraction:list, type_attraction: str):
	index_att=[]
	for i in list_attraction:
		if i == type_attraction:
			index_att.append(i)
	return index_att


def knn(X_train,X_test,y_train_attraction, y_test_attraction):
	
	Hotel=index_attraction(y_train_attraction, "Hotel")
	Restaurant=index_attraction(y_train_attraction, "Restaurant")
	Attractive=index_attraction(y_train_attraction, "Attractive")

	#Normalizar datos con minmax()

	#Llamar función clasificador=KNeighborsClassifier(numero de vecinos)
	#clasificador.fit(datos,clases)

	#Pasar datos de prueba y listo :D

	X_train_process=stopwords(X_train)
	X_test_process=stopwords(X_test)
	X_train_frec=vect_repres(X_train_process)
	X_test_frec=vect_repres(X_test_process)



def generate_train_test(file_name, test_size=0.1):
	pd.options.display.max_colwidth = 200				

	#Lee el corpus original del archivo de entrada y lo pasa a una DataFrame
	df = pd.read_excel(file_name, dtype=str)
	
	X = df.drop(['Polarity', 'Attraction'],axis=1).values   
	y_polarity = df['Polarity'].values
	y_attraction = df['Attraction'].values
	
	#~ #Separa el corpus cargado en el DataFrame en el 80% para entrenamiento y el 20% para pruebas
	X_train, X_test, y_train_polarity, y_test_polarity = train_test_split(X, y_polarity, test_size=test_size, random_state=0)
	X_train, X_test, y_train_attraction, y_test_attraction = train_test_split(X, y_attraction, test_size=test_size, random_state=0)
	
	return (data_set(X_train, y_train_polarity, X_test, y_test_polarity), data_set(X_train, y_train_attraction, X_test, y_test_attraction))
	

def getSELFeatures(cadenas, lexicon_sel):
	#"hastiar": [("Enojo\n', '0.629"), ("Repulsi\xf3n\n', '0.596")]
	features  = []
	polaridad = []
	for cadena in cadenas:
		valor_alegria = 0.0
		valor_enojo = 0.0
		valor_miedo = 0.0
		valor_repulsion = 0.0
		valor_sorpresa = 0.0
		valor_tristeza = 0.0
		
		cadena_palabras = re.split("\s+", cadena)
		dic = {}
		for palabra in cadena_palabras:
			if palabra in lexicon_sel:
				caracteristicas = lexicon_sel[palabra]
				for emocion, valor in caracteristicas:
					if emocion == "Alegría":
						valor_alegria = valor_alegria + float(valor)
					elif emocion == "Tristeza":
						valor_tristeza = valor_tristeza + float(valor)
					elif emocion == "Enojo":
						valor_enojo = valor_enojo + float(valor)
					elif emocion == "Repulsión":
						valor_repulsion = valor_repulsion + float(valor)
					elif emocion == "Miedo":
						valor_miedo = valor_miedo + float(valor)
					elif emocion == "Sorpresa":
						valor_sorpresa = valor_sorpresa + float(valor)
		dic["__alegria__"] = valor_alegria
		dic["__tristeza__"] = valor_tristeza
		dic["__enojo__"] = valor_enojo
		dic["__repulsion__"] = valor_repulsion
		dic["__miedo__"] = valor_miedo
		dic["__sorpresa__"] = valor_sorpresa
		
		#Esto es para los valores acumulados del mapeo a positivo (alegría + sorpresa) y negativo (enojo + miedo + repulsión + tristeza)
		dic["acumuladopositivo"] = dic["__alegria__"] + dic["__sorpresa__"]
		dic["acumuladonegativo"] = dic["__enojo__"] + dic["__miedo__"] + dic["__repulsion__"] + dic["__tristeza__"]
		
		#print(dic)
		"""
		if dif_polarity<1:
		dif_polarity = 1
		elif dif_polarity>5:
		dif_polarity = 5
		"""
		dif_polarity = dic["acumuladopositivo"]-dic["acumuladonegativo"]

		polaridad.append(dif_polarity)
		features.append(dic)	
	return polaridad


def optimization(fitness_func, params=1, gene_space=None, num_generations=50):
	ga_instance = pygad.GA(num_generations=num_generations,
					   num_parents_mating=4,
					   fitness_func=fitness_func,
					   sol_per_pop=8,
					   num_genes= params,
					   gene_space= gene_space, #np.linspace(0,1,1000).tolist(),
					   init_range_low=-1, 
					   init_range_high=1,
					   parent_selection_type="sss",
					   keep_parents=1,
					   crossover_type="single_point",
					   mutation_type="random",
					   suppress_warnings=True,
					   #mutation_percent_genes=10
					   )
	ga_instance.run()

	solution, solution_fitness, solution_idx = ga_instance.best_solution()
	print(f"Parameters of the best solution : {solution}")
	print(f"Fitness value of the best solution = {solution_fitness}")
	return solution

def metricas(pred, true):

	target_names = ['Muy Negativo', 'Negativo', 'Neutro', 'Positivo', 'Muy Positivo']

	print(classification_report(pred, true, target_names=target_names))

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

if __name__=='__main__':
	if not (os.path.exists('Corpus/Rest_Mex_2022_Sentiment_Analysis_Track_Train-Polarity-Preprocessed.pkl')):
		raise Exception("***No se ha generado el corpus lematizado para Polarity***")
	else:
		with open ('Corpus/Rest_Mex_2022_Sentiment_Analysis_Track_Train-Polarity-Preprocessed.pkl','rb') as corpus_file:
			corpus_polarity = pickle.load(corpus_file)

	corpus_polarity.X_train = corpus_polarity.X_train.tolist()
	corpus_polarity.y_train = corpus_polarity.y_train.tolist()
	corpus_polarity.X_test = corpus_polarity.X_test.tolist()
	corpus_polarity.y_test = corpus_polarity.y_test.tolist()

	corpus_polarity.y_test = np.array(corpus_polarity.y_test).astype(int).tolist()
	corpus_polarity.y_train = np.array(corpus_polarity.y_train).astype(int).tolist()

	#~ Load lexicons
	if (os.path.exists("Lexicon/SEL_full.pkl")):
		print("*** 'Lexicon/SEL_full.pkl' already exists ***") 

		lexicon_sel_file = open ("Lexicon/SEL_full.pkl","rb")
		lexicon_sel = pickle.load(lexicon_sel_file)
	else:
		print("*** 'Lexicon/SEL_full.pkl' don't found ***") 
		raise Exception("*** 'Lexicon/SEL_full.pkl' don't found ***")

	polarity_pred = getSELFeatures(corpus_polarity.X_train, lexicon_sel)

	def polarity_classification(umbral: list, polarity: list)-> list: 
		polarity_class = []
		for i in range(len(polarity)):
			if   polarity[i]<= umbral[0]:
				polarity_class.append(1)

			elif polarity[i]> umbral[0] and polarity[i]<= umbral[1]:
				polarity_class.append(2)

			elif polarity[i]> umbral[1] and polarity[i]<= umbral[2]:
				polarity_class.append(3)

			elif polarity[i]> umbral[2] and polarity[i]<= umbral[3]:
				polarity_class.append(4)
				
			elif polarity[i]> umbral[3]:
				polarity_class.append(5)		
		return polarity_class

	# solution - Lista de umbrales de mañao 4
	def fitness(solution: np.ndarray, index_solution:int):
		if solution[0]<solution[1] and solution[1]<solution[2] and solution[2]<solution[3]: #and solution[2]>0 and solution[3]>0:

			polarity_class = polarity_classification(solution.tolist(), polarity_pred)
			mae = MAE(corpus_polarity.y_train, polarity_class)
			
			return 1/mae
			#accuracy = accuracy_score(corpus_polarity.y_train, polarity_class)
			#return accuracy
		else:
			return 0

	umbrals = optimization(fitness, 4, None, 1000)
	
	polarity_class = polarity_classification(umbrals, polarity_pred)
	print(f"final umbral: {umbrals}\t\taccuracy: {accuracy_score(corpus_polarity.y_train, polarity_class)}")

	print(f"confusion:\n{confusion_matrix(corpus_polarity.y_train, polarity_class)}")

	polarity_pred = getSELFeatures(corpus_polarity.X_test, lexicon_sel)
	polarity_test = polarity_classification(umbrals, polarity_pred)

	#								true					pred
	accuracy_test = accuracy_score(corpus_polarity.y_test, polarity_test)

	print(f"accuracy test: {accuracy_test}")
	
	metricas(corpus_polarity.y_test, polarity_test)

	#Best [-3.37094824 -2.68145911 -1.09535897 -0.10111522]
	
	print(f"El valor de MAE es: {MAE(corpus_polarity.y_test, polarity_test)}")
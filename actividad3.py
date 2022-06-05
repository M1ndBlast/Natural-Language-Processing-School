"""
Lo voy a ejecutar
vamos empezando tambien practicca 4 suponiendo que la 3 está bien?
si alv, si alv x2
"""
from operator import le
import os, re, pickle
import numpy as np
import pandas as pd
from fpdf import FPDF
from sklearn.metrics import confusion_matrix, accuracy_score
from Preprocesamiento.lematizador import lematizar

def load_sel():
	#~ global lexicon_sel
	lexicon_sel = {}
	input_file = open("Lexicon/SEL_full.txt", "r")
	for line in input_file:
		#Las líneas del lexicon tienen el siguiente formato:
		#abundancia	0	0	50	50	0.83	Alegría
		
		palabras = line.split("\t")
		palabras[6]= re.sub("\n", "", palabras[6])
		pair = (palabras[6], palabras[5])
		if lexicon_sel:
			if palabras[0] not in lexicon_sel:
				lista = [pair]
				lexicon_sel[palabras[0]] = lista
			else:
				lexicon_sel[palabras[0]].append (pair)
		else:
			lista = [pair]
			lexicon_sel[palabras[0]] = lista
	input_file.close()
	del lexicon_sel["Palabra"]; #Esta llave se inserta porque es parte del encabezado del diccionario, por lo que se requiere eliminar
	#Estructura resultante
		#"hastiar": [("Enojo\n', '0.629"), ("Repulsi\xf3n\n', '0.596")]
	
	return lexicon_sel

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
		features.append (dic)


	umbral = [0.0005, 0.1]
	
	for i in range(len(polaridad)):
		if polaridad[i]<= -umbral[1]:
			polaridad[i] = 1
		elif polaridad[i]> -umbral[1] and polaridad[i]<= -umbral[0]:
			polaridad[i] = 2
		elif polaridad[i]> -umbral[0] and polaridad[i]<= umbral[0]:
			polaridad[i] = 3
		elif polaridad[i]> umbral[0] and polaridad[i]<= umbral[1]:
			polaridad[i] = 4
		elif polaridad[i]>umbral[1]:
			polaridad[i] = 5
	return polaridad


if __name__=="__main__":
	def lematyze(s1:str, s2:str, i:int, n:int)->str:
		print(f"\r{i}/{n}", end="")
		return lematizar(s1)+". "+lematizar(s2)

	#~ Load Corpus
	if os.path.exists("Corpus/Rest_Mex_2022_Sentiment_Analysis_Track_Train-Preprocessed.xlsx"):
		print("*** 'Corpus/Rest_Mex_2022_Sentiment_Analysis_Track_Train-Preprocessed.xlsx' already exists ***") 
		df = pd.read_excel("Corpus/Rest_Mex_2022_Sentiment_Analysis_Track_Train-Preprocessed.xlsx")
		df=df.replace(to_replace=np.NaN,value="")
		opinions = df["Opinion"].to_list()
		polarity_real = df['Polarity'].astype(int).to_list()	
		
	else:
		print("*** 'Corpus/Rest_Mex_2022_Sentiment_Analysis_Track_Train-Preprocessed.xlsx' don't found ***") 
		df = pd.read_excel("Corpus/Rest_Mex_2022_Sentiment_Analysis_Track_Train.xlsx", dtype=str, nrows=None)
		df = df.replace(to_replace=np.NaN, value="")
		#print(f"\n\nnan df \n{df}")

		opinions = list(map(lambda enum:  re.sub(r"_", " ", lematyze(enum[1][0], enum[1][1], enum[0], len(df.values))), enumerate(np.array([df["Title"].values, df["Opinion"].values]).transpose().tolist(), start=1)))
		polarity_real = list(map(lambda dict: int(dict[2]), df.values)) 

		df = pd.DataFrame({"Opinion":opinions, "Polarity":polarity_real, "Attraction":df["Attraction"].values})
		df.to_excel("Corpus/Rest_Mex_2022_Sentiment_Analysis_Track_Train-Preprocessed.xlsx", index=False)
	
	print(f"\nopinions #{len(opinions)}")

	#~ Load lexicons
	if (os.path.exists("Lexicon/SEL_full.pkl")):
		print("*** 'Lexicon/SEL_full.pkl' already exists ***") 

		lexicon_sel_file = open ("Lexicon/SEL_full.pkl","rb")
		lexicon_sel = pickle.load(lexicon_sel_file)
	else:
		print("*** 'Lexicon/SEL_full.pkl' don't found ***") 

		lexicon_sel = load_sel()
		lexicon_sel_file = open ("Lexicon/SEL_full.pkl","wb")
		pickle.dump(lexicon_sel, lexicon_sel_file)
		lexicon_sel_file.close()
	
	
	polarity_pred = getSELFeatures(opinions, lexicon_sel)
	polarity_pred = np.array(polarity_pred)
	polarity_pred = np.ceil(polarity_pred).astype(int).tolist()
	
	print(f"confusion:\n{confusion_matrix(polarity_real, polarity_pred)}")
	print(f"accuracy: \n{accuracy_score(polarity_real, polarity_pred)}")
	
	pdf=FPDF()
	pdf.add_page()
	pdf.add_font('roboto', '', 'Font/Roboto-Regular.ttf')
	pdf.set_font('roboto', '', 14)
	pdf.cell(200,10,txt="Integrantes:", align="C", ln=1)
	pdf.cell(200,10,txt="Aragón González Francisco Javier", align="C", ln=1)
	pdf.cell(200,10,txt="Del Valle Pérez Juan Daniel", align="C", ln=1)
	pdf.cell(200,10,txt="Peduzzi Acevedo Gustavo Alain", align="C", ln=1)
	pdf.cell(200,10,txt="Varillas Figueroa Edgar Josue", align="C", ln=1)
	pdf.cell(200,10,txt=f"Precision obtenida {accuracy_score(polarity_real, polarity_pred)}", align="L", ln=2)
	confusion = confusion_matrix(polarity_real, polarity_pred).tolist()
	pdf.cell(200,10,txt="Matriz de Confusion", align="L", ln=1)
	for i,row in enumerate(confusion, start=1):
		#print(f"\r{i}/{len(confusion)}", end="")
		pdf.multi_cell(200,10,txt=f"#{i}  {row}", align="L", ln=1)
	pdf.output("Actividad3.pdf")



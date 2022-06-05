from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd

corpus = ['El niño corre velozmente por el camino.',
          'El coche rojo del niño es grande.',
          'El brillante coche tiene un color rojo brillante y tiene llantas nuevas.',
          '¿Las nuevas canicas del niño son color rojo?,'
]

# Representación vectorial binarizada
vectorizador_binario = CountVectorizer(binary=False)
X = vectorizador_binario.fit_transform(corpus)
print (vectorizador_binario.get_feature_names_out())
print (X)#sparse matrix
print (type(X.toarray()))#dense ndarray
print ('Representación vectorial binarizada')
print (X.toarray())#dense ndarray
print("uwu")
print(pd.DataFrame(X.toarray(), columns=vectorizador_binario.get_feature_names_out()))


#~ #Representación vectorial por frecuencia
#~ vectorizador_frecuencia = CountVectorizer()
#~ X = vectorizador_frecuencia.fit_transform(corpus)
#~ print('Representación vectorial por frecuencia')
#~ print (X.toarray())


#Representación vectorial tf-idf
#~ vectorizador_tfidf = TfidfVectorizer()
#~ X = vectorizador_tfidf.fit_transform(corpus)
#~ print ('Representación vectorial tf-idf')
#~ print (X.toarray())

#uso_pandas!!!

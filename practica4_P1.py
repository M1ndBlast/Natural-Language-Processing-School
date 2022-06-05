"""
    @autors: Aragón González Francisco Javier
             Del Valle Pérez Juan Daniel
             Peduzzi Acevedo Gustavo Alain
             Varillas Figueroa Edgar Josue
    @date: 18/03/2022
    @Description: -
"""
import re
import spacy
nlp = spacy.load("es_core_news_sm") # Importación para Language processing en españofrom gensim.models.doc2vec import Doc2Vec, TaggedDocumentl
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

with open("corpus_noticias.txt") as file:
    dataset=file.readlines()

# Separa por Ampersons y obtiene la noticia
news = list(map(lambda y: re.split("&&&&&&&&", y)[2], dataset))

tags = ['SPACE']
news_token = []
for i, new in enumerate(news, start=1):
	print(f"\r{i}/{len(news)}", end="")
	new_processed = nlp(new)
	new_token = []
	
	for word in new_processed:
		if word.pos_ not in tags:
			new_token.append(word.text)
	news_token.append(new_token)

tagged_data_token = [TaggedDocument(d, [i]) for i, d in enumerate(news_token)]

tags = ['DET','PRON', 'CCONJ', 'SPACE', 'SYM', 'AUX']
news_normalized = []
for i, new in enumerate(news, start=1):
	print(f"\r{i}/{len(news)}", end="")
	new_preprocessed = nlp(new)
	new_normalized = []
	
	for word in new_preprocessed:
		if word.pos_ not in tags:
			new_normalized.append(word.lemma_)
	news_normalized.append(new_normalized)

tagged_data_normalized = [TaggedDocument(d, [i]) for i, d in enumerate(news_normalized)]

model = Doc2Vec(tagged_data_token, vector_size=100, dm=0, window=5)
print("Doc2Vec(tagged_data_token, vector_size=100, dm=0, window=5)")
model.save("doc2vec_token-dm0-vs100-w5.model")
model = Doc2Vec.load("doc2vec_token-dm0-vs100-w5.model")

## Similitud entre documentos
for j in range(0,len(news_token)):
  most_similar = model.dv.most_similar(model.dv[j])
  for i in most_similar:
	  print(f"\rNew_{j}\t-\tNew_{i[0]} <{i[1]}>" ) 

model = Doc2Vec(tagged_data_token, vector_size=100, dm=1, window=5)
print("Doc2Vec(tagged_data_token, vector_size=100, dm=1, window=5)")
model.save("doc2vec_token-dm1-vs100-w5.model")
model= Doc2Vec.load("doc2vec_token-dm1-vs100-w5.model")

## Similitud entre documentos
for j in range(0,len(news_token)):
  most_similar = model.dv.most_similar(model.dv[j])
  for i in most_similar:
	  print(f"\rNew_{j}\t-\tNew_{i[0]} <{i[1]}>" )

model = Doc2Vec(tagged_data_token, vector_size=300, dm=0, window=5)
print("Doc2Vec(tagged_data_token, vector_size=300, dm=0, window=5)")
model.save("doc2vec_token-dm0-vs300-w5.model")
model= Doc2Vec.load("doc2vec_token-dm0-vs300-w5.model")

## Similitud entre documentos
for j in range(0,len(news_token)):
  most_similar = model.dv.most_similar(model.dv[j])
  for i in most_similar:
	  print(f"\rNew_{j}\t-\tNew_{i[0]} <{i[1]}>" )

model = Doc2Vec(tagged_data_token, vector_size=300, dm=1, window=5)
print("Doc2Vec(tagged_data_token, vector_size=300, dm=1, window=5)")
model.save("doc2vec_token-dm1-vs300-w5.model")
model= Doc2Vec.load("doc2vec_token-dm1-vs300-w5.model")

## Similitud entre documentos
for j in range(0,len(news_token)):
  most_similar = model.dv.most_similar(model.dv[j])
  for i in most_similar:
	  print(f"\rNew_{j}\t-\tNew_{i[0]} <{i[1]}>" )

model = Doc2Vec(tagged_data_token, vector_size=100, dm=0, window=10)
print("Doc2Vec(tagged_data_token, vector_size=100, dm=0, window=10)")
model.save("doc2vec_token-dm0-vs100-w10.model")
model= Doc2Vec.load("doc2vec_token-dm0-vs100-w10.model")

## Similitud entre documentos
for j in range(0,len(news_token)):
  most_similar = model.dv.most_similar(model.dv[j])
  for i in most_similar:
	  print(f"\rNew_{j}\t-\tNew_{i[0]} <{i[1]}>" )

model = Doc2Vec(tagged_data_token, vector_size=300, dm=1, window=10)
print("Doc2Vec(tagged_data_token, vector_size=300, dm=1, window=10)")
model.save("doc2vec_token-dm1-vs300-w10.model")
model= Doc2Vec.load("doc2vec_token-dm1-vs300-w10.model")

## Similitud entre documentos
for j in range(0,len(news_token)):
  most_similar = model.dv.most_similar(model.dv[j])
  for i in most_similar:
	  print(f"\rNew_{j}\t-\tNew_{i[0]} <{i[1]}>" )


model = Doc2Vec(tagged_data_normalized, vector_size=100, dm=0, window=5)
print("Doc2Vec(tagged_data_normalized, vector_size=100, dm=0, window=5)")
model.save("doc2vec_normalized-dm0-vs100-w5.model")
model= Doc2Vec.load("doc2vec_normalized-dm0-vs100-w5.model")

## Similitud entre documentos
for j in range(len(news_normalized)):
	most_similar = model.dv.most_similar(model.dv[j])
	for i in most_similar:
		print(f"\rNew_{j}\t-\tNew_{i[0]} <{i[1]}>" )

model = Doc2Vec(tagged_data_normalized, vector_size=100, dm=1, window=5)
print("Doc2Vec(tagged_data_normalized, vector_size=100, dm=1, window=5)")
model.save("doc2vec_normalized-dm1-vs100-w5.model")
model= Doc2Vec.load("doc2vec_normalized-dm1-vs100-w5.model")

## Similitud entre documentos
for j in range(len(news_normalized)):
	most_similar = model.dv.most_similar(model.dv[j])
	for i in most_similar:
		print(f"\rNew_{j}\t-\tNew_{i[0]} <{i[1]}>" )

model = Doc2Vec(tagged_data_normalized, vector_size=300, dm=0, window=5)
print("Doc2Vec(tagged_data_normalized, vector_size=300, dm=0, window=5)")
model.save("doc2vec_normalized-dm0-vs300-w5.model")
model= Doc2Vec.load("doc2vec_normalized-dm0-vs300-w5.model")

## Similitud entre documentos
for j in range(len(news_normalized)):
	most_similar = model.dv.most_similar(model.dv[j])
	for i in most_similar:
		print(f"\rNew_{j}\t-\tNew_{i[0]} <{i[1]}>" )

model = Doc2Vec(tagged_data_normalized, vector_size=300, dm=1, window=5)
print("Doc2Vec(tagged_data_normalized, vector_size=300, dm=1, window=5)")
model.save("doc2vec_normalized-dm1-vs300-w5.model")
model= Doc2Vec.load("doc2vec_normalized-dm1-vs300-w5.model")

## Similitud entre documentos
for j in range(len(news_normalized)):
	most_similar = model.dv.most_similar(model.dv[j])
	for i in most_similar:
		print(f"\rNew_{j}\t-\tNew_{i[0]} <{i[1]}>" )

model = Doc2Vec(tagged_data_normalized, vector_size=100, dm=0, window=10)
print("Doc2Vec(tagged_data_normalized, vector_size=100, dm=0, window=10)")
model.save("doc2vec_normalized-dm0-vs100-w10.model")
model= Doc2Vec.load("doc2vec_normalized-dm0-vs100-w10.model")

## Similitud entre documentos
for j in range(len(news_normalized)):
	most_similar = model.dv.most_similar(model.dv[j])
	for i in most_similar:
		print(f"\rNew_{j}\t-\tNew_{i[0]} <{i[1]}>" )

model = Doc2Vec(tagged_data_normalized, vector_size=300, dm=1, window=10)
print("Doc2Vec(tagged_data_normalized, vector_size=300, dm=1, window=10)")
model.save("doc2vec_normalized-dm1-vs300-w10.model")
model= Doc2Vec.load("doc2vec_normalized-dm1-vs300-w10.model")

## Similitud entre documentos
for j in range(len(news_normalized)):
	most_similar = model.dv.most_similar(model.dv[j])
	for i in most_similar:
		print(f"\rNew_{j}\t-\tNew_{i[0]} <{i[1]}>" )
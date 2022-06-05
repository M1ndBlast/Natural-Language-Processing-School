from sklearn.datasets import fetch_20newsgroups
from pprint import pprint
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.decomposition import TruncatedSVD


newsgroups_train = fetch_20newsgroups(subset='train')
newsgroups_test = fetch_20newsgroups(subset='test')
#~ print(newsgroups_train.target_names)
#~ print(newsgroups_train.data[0])
#~ print(newsgroups_train.target[0])

X_train = newsgroups_train.data
y_train = newsgroups_train.target
X_test = newsgroups_test.data
y_test = newsgroups_test.target

#~ print (newsgroups_train.target.shape)
#~ print (newsgroups_train.target[:10])
vectorizer = TfidfVectorizer()
vectors_train = vectorizer.fit_transform(newsgroups_train.data)
print("2")

clf = LogisticRegression()
clf.fit(vectors_train, y_train)
print("3")

vectors_test = vectorizer.transform(newsgroups_test.data)
y_pred = clf.predict(vectors_test)
#~ print (y_pred)
print ('vectors_train.shape {}'.format(vectors_train.shape))
print ('vectors_test.shape {}'.format(vectors_test.shape))



print (classification_report(y_test, y_pred))
print("*"*20)
print("*"*20)

svd = TruncatedSVD(500)
vectors_train_lsa = svd.fit_transform(vectors_train)
##~ print (vectors_train_lsa)
print ('vectors_train_lsa.shape {}'.format(vectors_train_lsa.shape))

clf.fit(vectors_train_lsa, y_train)

vectors_test_lsa = svd.transform(vectors_test)
print ('vectors_test_lsa.shape {}'.format(vectors_test_lsa.shape))


y_pred = clf.predict(vectors_test_lsa)
print (y_pred)

print (classification_report(y_test, y_pred))
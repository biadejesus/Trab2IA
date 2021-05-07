import csv
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
import json
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_predict
from sklearn.feature_selection import SelectKBest, chi2
from scipy.sparse import hstack
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn import neighbors
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import imblearn
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import TomekLinks 
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek 

#positivo= 5, neg = 1,2 neu = 3, 4

def unir_classes(strelas):
  if strelas in [5]:
    return 2
  elif strelas in [1,2]:
    return 0
  else:
    return 1  

data = pd.read_csv('tripadvisor_hotel_reviews.csv')

quantidade = data.shape[0]
print(quantidade)

qtd_classe = data["Rating"].value_counts()
print(qtd_classe)

data['Sentimento'] = data['Rating'].apply(unir_classes)

stopwords_list = set(stopwords.words("english"))
punctuations = """!()-![]{};:,+'"\,<>./?@#$%^&*_~Â""" #List of punctuation to remove

def reviewParse(review):
    splitReview = review.split() #Split the review into words
    parsedReview = " ".join([word.translate(str.maketrans('', '', punctuations)) + " " for word in splitReview]) #Takes the stubborn punctuation out
    return parsedReview #Returns the parsed review
  
def clean_review(review):
    clean_words = []
    splitReview = review.split()
    for w in splitReview:
        if w.isalpha() and w not in stopwords_list:
            clean_words.append(w.lower())
    clean_review = " ".join(clean_words)
    return clean_review

data["Review"] = data["Review"].apply(reviewParse).apply(clean_review)


ngrama = 1
k = 100
aux = []
count = 0
X_sentences = []
Y_sentences = []

# for i in range(len(data)):
#     X_sentences.append(data.values[i][0])
#     Y_sentences.append(data.values[i][2])
X_sentences = list(data['Review'])[:1400]

Y_sentences = data['Sentimento'][:1400]


vectorizer = TfidfVectorizer(ngram_range=(1, ngrama), max_features = 20000)
X_sentences = vectorizer.fit_transform(X_sentences)


print("chi-quadrado ", k)
selector = SelectKBest(chi2, k=k)
X_sentences = selector.fit_transform(X_sentences, Y_sentences)

print("\nClassificador SVM: ")
smt = SMOTETomek()
param_grid = [{'class_weight': ['balanced'],'kernel': ['rbf', 'linear', 'poly', 'sigmoid'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]}]
clf = GridSearchCV(SVC(), param_grid, cv = 5, n_jobs=-1) 

X_sentences, Y_sentences = smt.fit_resample(X_sentences, Y_sentences)
clf = clf.fit(X_sentences, Y_sentences)
print("Predicão...")
pred = cross_val_predict(clf, X_sentences, Y_sentences, cv = 5, n_jobs = -1)

print('melhores parametros: ', clf.best_params_)

print("Classification_report:")
print(classification_report(Y_sentences, pred,  zero_division = 0))
print("")
print(confusion_matrix(Y_sentences, pred))
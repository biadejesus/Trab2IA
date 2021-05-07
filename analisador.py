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

def unir_classes(avaliacao):
  if avaliacao in [5]:
    return 2
  elif avaliacao in [1,2]:
    return 0
  else:
    return 1  

def pontuacao(Avaliacao):
    splitAvaliacao = Avaliacao.split() 
    parsedAvaliacao = " ".join([word.translate(str.maketrans('', '', pontuacoes)) + " " for word in splitAvaliacao]) 
    return parsedAvaliacao 
  
def processamento(Avaliacao):
    clean_words = []
    splitAvaliacao = Avaliacao.split()
    for w in splitAvaliacao:
        if w.isalpha() and w not in stopwords:
            clean_words.append(w.lower())
    processamento = " ".join(clean_words)
    return processamento

data = pd.read_csv('tripadvisor_hotel_reviews.csv')

# quantidade = data.shape[0]
# print(quantidade)

# qtd_classe = data["Rating"].value_counts()
# print(qtd_classe)

data['Sentimento'] = data['Rating'].apply(unir_classes)

stopwords = set(stopwords.words("english"))
pontuacoes = """!()-![]{};:,+'"\,<>./?@#$%^&*_~Â"""
data["Review"] = data["Review"].apply(pontuacao).apply(processamento)


ngrama = 1
k = 100
X_sentences = []
Y_sentences = []
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


# #################### MLP ###############################
print("\nClassificador MLP:")
param_grid = [{'learning_rate' :['constant', 'invscaling', 'adaptive'], 'activation': ['identity', 'logistic', 'tanh', 'relu'], 'alpha': [0.0001, 0.05], 'hidden_layer_sizes': [(50,50), (100,50)], 'solver': ['sgd', 'adam']}]
#param_grid = [{'activation': ['relu', 'identity', 'logistic', 'tanh'], 'alpha': [0.0001], 'hidden_layer_sizes': [(100,50)], 'solver': ['adam'], 'learning_rate' :['constant', 'invscaling', 'adaptive']}]
smt = SMOTETomek()
#X_sentences, Y_sentences = smt.fit_resample(X_sentences, Y_sentences)
clf = GridSearchCV(MLPClassifier(), param_grid, cv = 5, n_jobs=-1)
clf = clf.fit(X_sentences, Y_sentences)

print("Predicão...")
pred = cross_val_predict(clf, X_sentences, Y_sentences, cv=5, n_jobs=-1)

print("Classification_report:")
print(classification_report(Y_sentences, pred,  zero_division = 0))
print("")
print(confusion_matrix(Y_sentences, pred))
print('melhores parametros: ', clf.best_params_)

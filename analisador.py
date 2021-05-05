import csv
import numpy as np
import pandas as pd
# from nltk.tokenize import sent_tokenize
# from nltk.corpus import stopwords

#positivo= 5, neg = 1,2 neu = 3, 4

def sentiment(rating):
  if rating in [5]:
    return 2
  elif rating in [1,2]:
    return 0
  else:
    return 1  

data = pd.read_csv('tripadvisor_hotel_reviews.csv')

quantidade = data.shape[0]
print(quantidade)

qtd_classe = data["Rating"].value_counts()
print(qtd_classe)

data['Sentiment'] = data['Rating'].apply(sentiment)


aux = []
count = 0
sentences = []
labels = []
for i in range(len(data)):
    sentences.append(data.values[i][0])
    labels.append(data.values[i][2])

print(sentences)
print((labels))
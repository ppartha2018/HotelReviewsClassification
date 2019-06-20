import numpy as np
import os
from sklearn.model_selection import train_test_split
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

'''
@author: prasanna
Hotel Reviews classification: NLP.
Binray classification that employs nearest neighbors to classify between truthful and deceptive reviews.
Feature extraction, pre-processing, followed by model selection and testing.
Dataset involves 1000s of reviews.
Extract of the Jupyter notebook.
'''

def dummy_tokenizer(doc):
    return doc

nltk.download('stopwords')
nltk.download('punkt')
X_dir1_truthful = "truthful" #0
X_dir1_deceptive = "deceptive" #1

X = []
y = []

for i in range(1,11):
    dir = "./Training/Fold"+str(i)+"/"+X_dir1_deceptive+"/"
    
    for filename in os.listdir(dir):
        file = open(dir+filename,mode='r')
        X.append(file.read())
        file.close()
        #X.append(np.loadtxt(dir+filename, dtype=str))
        y.append(1)
    dir = "./Training/Fold"+str(i)+"/"+X_dir1_truthful+"/"
    for filename in os.listdir(dir):
        file = open(dir+filename,mode='r')
        X.append(file.read())
        #X.append(np.loadtxt(dir+filename, dtype=str))
        y.append(0)

print("Length of the trining data: ", len(X))
print("Length of class labels: ", len(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

X = [x.lower() for x in X]

#Remove numbers
X = [re.sub(r"\d+", "", input_str) for input_str in X]
     
#remove punctuations- speculation, active speech -  can they improve accuracy? -> do they actually play important role in deceptive text?
X = [s.translate(str.maketrans('','',string.punctuation)) for s in X]

#remove trailing whitespaces
X = [s.strip() for s in X]

def remove_stop_words(doc, stop_words):
    tokens = word_tokenize(doc)
    result = [i for i in tokens if not i in stop_words]
    return result

#stop words removal
stop_words = set(stopwords.words("english"))
tokens = [remove_stop_words(doc, stop_words) for doc in X]
#print (X[0])

def lemmatize(doc_arr):
    stemmer= PorterStemmer()
    for i in range(0,len(doc_arr)):
        doc_arr[i] = stemmer.stem(doc_arr[i])
    return doc_arr
        
#Lemmatization
tokens = [lemmatize(doc_arr) for doc_arr in tokens]
print("After all pre-processing: ")
print(X[0])

vectorizer = TfidfVectorizer(analyzer='word', tokenizer=dummy_tokenizer, preprocessor=dummy_tokenizer, token_pattern=None)
features = vectorizer.fit_transform(tokens)

print("After TFIDF: Features: ")
#print(vectorizer.get_feature_names())


#pos_tagging

#X[0]

from sklearn.neighbors import KNeighborsClassifier

print("Feature Shape: ", features.shape)
print(type(features))
print(features)
#X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.25, random_state=42)
#len(X_train)
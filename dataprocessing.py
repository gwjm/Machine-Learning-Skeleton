import numpy as np
import pandas as pd 
import sklearn.preprocessing 
from sklearn.model_selection import train_test_split
import nltk # Computational linguistics library for text processing 
import string

try:
    nltk.data.find(nltk.corpus.stopwords)
except LookupError: 
    nltk.download(nltk.corpus.stopwords)


class textDataHandler:
    def __init__(self, Data, Labelidx, Textidx, Embeddings = None, Stemming = "stem", Language = "English"):
        self.Data = Data
        self.labels = Labelidx
        self.embeddings = Embeddings
        self.textidx = Textidx
        self.stemMethod = True if Stemming == "stem" else False
        self.language = Language.lower()

    def prepare_embeddings(self):
        if self.embeddings == None:
            return None

    def cleantext(self):
        text_data = self.Data[self.textidx]
        # Lower casing the text data
        text_data = text_data.str.lower()
        
        #Punctuation removal 
        
        punct_remover = lambda text: text.translate(str.maketrans('','', string.punctuation))
        text_data = text_data.apply(lambda text: punct_remover(text))

        # stopword removal
        stopwords = ", ".join(nltk.corpus.stopwords.words(self.language))
        text_data = text_data.apply(lambda text: " ".join([word for word in str(text).split() if word not in stopwords]))
        
        #Lemmatization (stemming)
        if self.stemMethod == True: 
            stemmer = nltk.stem.SnowballStemmer(self.language, ignore_stopwords = True)
            text_data = text_data.apply(lambda text: " ".join([stemmer.stem(word) for word in text.split()]))
        else:
            stemmer = nltk.wordNetLemmatizer(Language = self.language)

        


# class numericalDataHandler:
#     def __init__(self, Data):
#         pass 

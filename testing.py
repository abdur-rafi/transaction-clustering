vector_path = '/content/vectors.kv'
clustering_path = '/content/clustering_model.joblib'

import pandas as pd


def toLower(df):
  df['text'] = df['text'].apply(lambda z : str(z).lower())

def eliminateNonAlphabet(df):
  df['text'] = df['text'].str.replace(r'[^\w\s]+', ' ')
  df['text'] = df['text'].replace( r'[^A-Za-z ]+', '', regex=True)

def tokenize(df):
  import nltk
  nltk.download('punkt')
  df['text'] = df['text'].apply(nltk.word_tokenize)

def dropEmpty(df):
  df.drop(df[df['text'].map(len) == 0].index, inplace = True)

def lematize(df):
  import nltk
  from nltk.stem import WordNetLemmatizer
  nltk.download('wordnet')
  lemmatizer = WordNetLemmatizer()
  df['text'] = df['text'].apply(lambda z : [lemmatizer.lemmatize(x) for x in z])

def removeWord(df, stop):
  df['text'] = df['text'].apply(lambda z : [word for word in z if word not in stop])

def removeGeneralStopWords(df):
  import nltk
  from nltk.corpus import stopwords
  nltk.download("stopwords")
  stop = set(stopwords.words("english"))
  removeWord(df, stop)

def showAppearances(df1, df2, text):
  return df1.loc[df2[df2['text'].map(lambda z : text in z) == True].index]

def removeDup(df):
  df.drop_duplicates(inplace = True)

# def removeEmptyTokensRows(df):
#   df.drop(df[df['text'].map(len) == 0].index, inplace = True)

def removeCustomWords(df):
  stops = ['ac', 'ae', 'md', 'akter', 'hossain', 'rahman', 'ltd', 'bazar', 'unnayan', 'dol', 'polli', 'mohila', 'ad', 'mia', 'ali', 'uddin', 
         'begum', 'tk', 'islam', 'amt', 'abdul', 'hasan', 'sonali', 'alam', 'miah', 'bangladesh' , 'per', 'taka',
         'da', 'vai', 'agrani' , 'ahmed', 'hossen', 'khan', 'abu', 'janata', 'al', 'unnayon', 'b', 'amin', 'mamun', 'khatun', 'unnoyon',  'nur', 'haque']
  notSure = ['mo', 'cap','fvg', 'islami', 'br', 'id' , 'dal', 'purush', 'nid', 'purpose', 'title', 'para', 'roll']
  months = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december', 'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
  removeWord(df, set(stops + notSure + months))

def replaceWord(df):
  def func(lst):
    try:
      index = lst.index('withdrawal')
      lst[index] = 'withdraw'
    except:
      None
    return lst
  df['text'] = df['text'].apply(func)

def removeNonEnglishWord(df,keep):
  import nltk
  nltk.download('words')
  from nltk.corpus import words
  wordSet = set(words.words())
  def func(row):
    r2 = []
    for w in row:
      if (len(str(w)) > 2) and ((w in wordSet) or (w in keep)):
        r2.append(w)

    return r2 

  df['text'] = df['text'].apply(func)


def cleanEng(df, rDup = True):
  toLower(df)
  eliminateNonAlphabet(df)
  if rDup:
    removeDup(df)
  tokenize(df)
  lematize(df)
  removeGeneralStopWords(df)
  removeCustomWords(df)
  replaceWord(df)
  removeNonEnglishWord(df, set(['eftn', 'ft', 'bkash', 'nogod', 'rtgs', 'pos', 'sme', 'mst']))
  dropEmpty(df)

def loadVectors(path):
  from gensim.models import KeyedVectors
  return KeyedVectors.load(path)

def getEncodings(df,vectors, dim = 100):
  import numpy as np
  def getSentEmbedding(txt):
    s = np.zeros(dim)
    c = 0
    for x in txt:
      y = str(x)
      if y in vectors:
        s += vectors[y]
        c += 1
    return (s / c) if c != 0 else s

  return np.vstack(df['text'].apply(getSentEmbedding).values)

def normalize(encodings):
  import numpy as np
  norm = np.linalg.norm(encodings, axis = 1, keepdims = True)
  norm[norm == 0] = 1
  norm_encodings = encodings / norm
  return norm_encodings

def loadClusteringModel(path):
  import joblib
  return joblib.load(path)
  

def main():
  import sys
  sents = sys.argv[1:]
  if(len(sents) == 0):
    print([])
  else:
    df = pd.DataFrame({'text' : sents})
    cleanEng(df)
    vectors = loadVectors(vector_path)
    dim = vectors[0].shape[0]
    encodings = getEncodings(df, vectors, dim)
    norm_encodings = normalize(encodings)
    clustering_model = loadClusteringModel(clustering_path)
    print(clustering_model.predict(norm_encodings))

if __name__ == "__main__":
  main()



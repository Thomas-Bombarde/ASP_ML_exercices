import pickle
import codecs
import chardet
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import dendrogram, linkage
from pathlib import Path
from string import digits, punctuation
import nltk
import pandas as pd
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def tokenize_and_stem(text):
    """Return tokens of text deprived of numbers and punctuation"""
    d = {p: "" for p in digits + punctuation} #iterates to create dict.
    text = text.translate(str.maketrans(d))
    return [_stemmer.stem(t) for t in nltk.word_tokenize(text.lower())]

_stemmer = nltk.snowball.SnowballStemmer("english")
_stopwords = nltk.corpus.stopwords.words("english")


#Read Data
data = sorted(Path('./data/speeches').glob('*R0*'))
corpus = []
for i in range(0,len(data)):
    try:
        corpus.append((data[i].open('r', encoding='utf-8').read()))
    except:
        print(data[i])
#Vectorise speeches
Tfidf = TfidfVectorizer(stop_words=_stopwords, tokenizer=tokenize_and_stem, ngram_range=(1,3), max_features = None)
Tfidf.fit(corpus)
Tfidf_matrix_pk = Tfidf.transform(corpus)
file = open('./output/speech_matrix.pk.', 'wb')
pickle.dump(Tfidf_matrix_pk, file)
Tfidf_matrix_csv = pd.DataFrame(Tfidf_matrix_pk)
Tfidf_matrix_csv.to_csv('./output/terms.csv')
file.close()

"Exercice 3"
count_matrix = open('./output/speech_matrix.pk', 'rb')
tfidf_document = pickle.load(count_matrix)
cosine = cosine_similarity(tfidf_document)
distance_matrix = squareform(cosine,force='tovector',checks=False)

from matplotlib import pyplot as plt

z_num = linkage(tfidf_document.todense(),'ward')
df = pd.DataFrame(z_num, columns = ["1", "2", "Distance", "4"])
df["Distance"].sort_values(ascending =False)
threshold = df["Distance"].iloc[9]
plt.figure(figsize=(25, 10))
dn = dendrogram(z_num, color_threshold = threshold, no_labels = True)

"""Exercice 4"""

bad_text = open("./data/Stellenanzeigen.txt", 'r').read()

chardet.detect("./data/Stellenanzeigen.txt")

#from chardet.universaldetector import UniversalDetector
#def detect_encode(file):
#    detector = UniversalDetector()
#    detector.reset()
#    with open(file, 'rb') as f:
#        for row in f:
#            detector.feed(row)
#            if detector.done: break
#    detector.close()
#    return detector.result
#print(detect_encode('./data/Stellenanzeigen.txt'))

Newspaper = read(./data/Stellenanzeigen.txt)
chardet.detect(bad_text)
chardet.universaldetector(bad_text)

codecs.decode(
df = pickle.load(bad_text)

save"./output/speeches_dendrogram.pdf."
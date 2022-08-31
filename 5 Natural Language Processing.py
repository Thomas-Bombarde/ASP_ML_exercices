import pickle
from math import floor
import re
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from pathlib import Path
from string import digits, punctuation
import nltk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


def tokenize_and_stem(text):
    """Return tokens of text deprived of numbers and punctuation"""
    d = {p: "" for p in digits + punctuation}
    text = text.translate(str.maketrans(d))
    return [_stemmer.stem(t) for t in nltk.word_tokenize(text.lower())]


def year_to_decade(year: int) -> str:
    """ Converts year (yyyy) to a decade (yyy.s)"""
    decade = floor(int(year) / 10) * 10
    return f"{decade}s"



# Exercise 2
# 2.a.
data = sorted(Path('./data/speeches').glob('*R0*'))
corpus = []
for i in range(0, len(data)):
    try:
        corpus.append((data[i].open('r', encoding='utf-8').read()))
    except:
        print(data[i])

# 2.b.
_stemmer = nltk.snowball.SnowballStemmer("english")
_stopwords = nltk.corpus.stopwords.words("english")
Tfidf = TfidfVectorizer(stop_words=_stopwords, tokenizer=tokenize_and_stem, ngram_range=(1, 3))
Tfidf.fit(corpus)
Tfidf_matrix = Tfidf.fit_transform(corpus)

# 2.c.
file = open('./output/speech_matrix.pk', 'wb')
pickle.dump(Tfidf_matrix, file)
file.close()
speech_matrix = pd.DataFrame(Tfidf_matrix.todense().T,
                             index=Tfidf.get_feature_names_out())
terms = pd.DataFrame(speech_matrix.index)
terms.to_csv('./output/terms.csv')

# Exercise 3
# 3.a.
count_matrix = open('./output/speech_matrix.pk', 'rb')
tfidf_document = pickle.load(count_matrix)
array = tfidf_document.toarray()

# 3.b.
z_num = linkage(array, method='complete', metric='cosine')
df = pd.DataFrame(z_num, columns=["1", "2", "Distance", "4"])
df["Distance"].sort_values(ascending=False)
threshold = df["Distance"].iloc[len(df) - 3]
plt.figure(figsize=(25, 10))
dn = dendrogram(z_num, color_threshold=threshold, no_labels=True)

# 3.c.
plt.savefig('./output/speeches_dendrogram.pdf')

# Exercise 4

# 4.a.
bad_text = open("./data/Stellenanzeigen.txt", 'r', encoding='utf-8').read()

Ads = re.findall(r".*,\s\d{1,2}\.\s\w+\s\d{4}\s+(.*\n?.*)", bad_text)
Newspaper = re.findall(r"(.*),\s\d{1,2}\.\s\w+\s\d{4}", bad_text)
Date = re.findall(r".*,\s(\d{1,2}\.\s\w+\s\d{4})", bad_text)

job_ads_df = pd.DataFrame({"Newspaper": Newspaper, "Date": Date, "Job Ad": Ads})
job_ads_df["Date"] = job_ads_df["Date"].str.replace("März", "3.").astype("datetime64[ns]")

# 4.b.
job_ads_df["Words per Job Ad"] = job_ads_df["Job Ad"].apply(lambda x: len(str(x).split(" ")))
years = job_ads_df["Date"].dt.year
decades = years.apply(year_to_decade)
words_per_decade = pd.DataFrame({"Decade": decades, "Words per Job Ad": job_ads_df["Words per Job Ad"]})
df = words_per_decade.groupby(["Decade"]).agg("mean")
df.plot.bar()
plt.legend(["Average Words per Job Ad"])

# 4.c
# (i) the required dataframe
job_ads_decade = job_ads_df
job_ads_decade["Date"] = decades
pivoted = job_ads_decade.pivot(columns="Date", values="Job Ad")
agg_ads_dec = []
for i in range(0, len(pivoted.columns)):
    col = pivoted.columns[i]
    corpus = pivoted[col].dropna()
    agg_ads_dec.append(''.join(corpus))

# (ii) most common words per decade
_stemmer = nltk.snowball.SnowballStemmer("german")
_stopwords = nltk.corpus.stopwords.words("german") + nltk.corpus.stopwords.words("english")
Count = CountVectorizer(stop_words=_stopwords, tokenizer=tokenize_and_stem, max_features=None, encoding="utf-8")
Count.fit(agg_ads_dec)
Count_matrix = Count.fit_transform(agg_ads_dec)
speech_matrix = pd.DataFrame(Count_matrix.todense().T,
                             index=Count.get_feature_names_out(), columns=pivoted.columns)
ten_top_words_per_dec = pd.DataFrame()
for i in range(0, len(pivoted.columns)):
    col = speech_matrix.columns[i]
    bob = speech_matrix[col].sort_values(ascending=False).iloc[0:9]
    print(f"{col},\n {bob} \n ")
    ten_top_words_per_dec[col] = bob.index
print(f"The following is a table of the most used terms per decade in the newspaper ads, in descending order:"
      f"\n {ten_top_words_per_dec}")

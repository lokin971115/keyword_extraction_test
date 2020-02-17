from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import feedparser
import re
from rake_nltk import Rake

localnews=feedparser.parse("https://rthk.hk/rthk/news/rss/e_expressnews_elocal.xml")
chinanews=feedparser.parse("https://rthk.hk/rthk/news/rss/e_expressnews_egreaterchina.xml")
worldnews=feedparser.parse("https://rthk.hk/rthk/news/rss/e_expressnews_einternational.xml")
financenews=feedparser.parse("https://rthk.hk/rthk/news/rss/e_expressnews_efinance.xml")
sportnews=feedparser.parse("https://rthk.hk/rthk/news/rss/e_expressnews_esport.xml")
text=[]
for i in range(len(localnews.entries)):
    text.append(localnews.entries[0].title.lower()+localnews.entries[0].summary.lower())
#text = re.sub("(\\d|\\W)+"," ",text)
vectorizer=CountVectorizer()
transformer=TfidfTransformer()
tfidf=transformer.fit_transform(vectorizer.fit_transform(text))
word=vectorizer.get_feature_names()
weight=tfidf.toarray()
#for i in range(len(weight)):
print(text)
for i in range(1):
    for j in range(len(word)):
        print(word[j],weight[i][j])

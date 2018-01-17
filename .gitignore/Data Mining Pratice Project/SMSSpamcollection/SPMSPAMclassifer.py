import nltk
import sys
import pandas as pd
import numpy as np
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

messages = pd.read_csv('SMSSpamCollection', sep='\t', names=['label', 'message'])

def text_process(mess):

    nonpunc = [char for char in mess if char not in string.punctuation]

    nonpunc = ''.join(nonpunc)

    return [words for words in nonpunc if words not in stopwords.words('english')]

bow_transformer = CountVectorizer(analyzer = text_process).fit(messages['message'])

message_bow = bow_transformer.transform(messages['message'])

tfidf_transformer = TfidfTransformer().fit(message_bow)

message_tfidf = tfidf_transformer.transform(message_bow)

from sklearn.naive_bayes import MultinomialNB

spam_dect_model = MultinomialNB().fit(message_tfidf, messages['label'])

all_predictions = spam_dect_model.predict(message_tfidf)

print(all_predictions)

from sklearn.metrics import classification_report
print(classification_report(messages['label'], all_predictions))

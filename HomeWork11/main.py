import re
import nltk
import string
import numpy as np
import gensim.models.keyedvectors as w2v

from nltk.corpus import stopwords
from nltk import word_tokenize

from keras.datasets import imdb

from gensim.models import KeyedVectors


nltk.download('punkt')
nltk.download('stopwords')


def preprocess(input_text):
    input_text = input_text.lower() 
    input_text = input_text.strip()  
    input_text = re.compile('<.*?>').sub('', input_text) 
    input_text = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', input_text)  
    input_text = re.sub('\s+', ' ', input_text)  
    input_text = re.sub(r'\[[0-9]*\]', ' ', input_text) 
    input_text= re.sub(r'[^\w\s]', '', str(input_text).lower().strip())
    input_text = re.sub(r'\d', ' ', input_text) 
    input_text = re.sub(r'\s+', ' ', input_text) 
    
    return input_text


def stopword(string):
    a= [i for i in string.split() if i not in stopwords.words('english')]
    
    return ' '.join(a)


def classify_review(rev_emb, pos_emb, neg_emb):
    dist_pos = np.linalg.norm(rev_emb - pos_emb)
    dist_neg = np.linalg.norm(rev_emb - neg_emb)

    if dist_pos < dist_neg:
        return 'Positive'
    else:
        return 'Negative'


def get_emb(model_obj, input_text):
    word_vectors = [model_obj.wv[i] for i in input_text.split() if i in w2v]

    if not word_vectors:
        return np.zeros(model_obj.vector_size)

    return np.mean(np.array(word_vectors), axis=0)


start_char = 1
oov_char = 2
index_from = 3

(x_train, _), _ = imdb.load_data(
    start_char=start_char, oov_char=oov_char, index_from=index_from
)


model = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin.gz", binary=True)
model.train(x_train, workers=2)


review_text = 'Jacob stood on his tiptoes. But this car is not blue. Kelly twirled in circles.'

positive_words = ['good', 
                  'nice', 
                  'great', 
                  'super', 
                  'cool', 
                  'excellent', 
                  'amazing', 
                  'positive', 
                  'happy', 
                  'joy']

negative_words = ['bad', 
                  'worst', 
                  'terrible', 
                  'awful', 
                  'negative', 
                  'sad', 
                  'anger',
                  'poor', 
                  'worse',
                  'hate']

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
sentences = (('$%^&*(**@'.join(tokenizer.tokenize(review_text))).split('$%^&*(**@'))

pre_sentences = [preprocess(_sentence) for _sentence in sentences]
tokenized_text = [word_tokenize(_sentence) for _sentence in pre_sentences]
stem_word_list = [stopword(_sentence) for _sentence in pre_sentences]
pos_emb = [get_emb(model, i) for i in positive_words if i]
neg_emb = [get_emb(model, i) for i in negative_words if i]

text_emb = [get_emb(model, _sentence) for _sentence in pre_sentences]
classification1 = classify_review(text_emb[0], pos_emb, neg_emb)
classification2 = classify_review(text_emb[1], pos_emb, neg_emb)
classification3 = classify_review(text_emb[2], pos_emb, neg_emb)

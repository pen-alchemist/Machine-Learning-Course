import re
import nltk
import string

from nltk.corpus import stopwords
from nltk import word_tokenize

from keras.datasets import imdb

from sklearn.feature_extraction.text import CountVectorizer

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


start_char = 1
oov_char = 2
index_from = 3

(x_train, _), _ = imdb.load_data(
    start_char=start_char, oov_char=oov_char, index_from=index_from
)


model = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin.gz", binary=True)
model.train(x_train, workers=2)


review_text = 'Jacob stood on his tiptoes. The car is not blue. Kelly twirled in circles.'
positive_words = ['good', 'nice', 'great', 'super', 'cool']

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
sentences = (('$%^&*(**@'.join(tokenizer.tokenize(review_text))).split('$%^&*(**@'))

pre_sentences = [preprocess(_sentence) for _sentence in sentences]
tokenized_text = [word_tokenize(_sentence) for _sentence in pre_sentences]
stem_word_list = [stopword(_sentence) for _sentence in pre_sentences]


vectorizer = CountVectorizer()
X = vectorizer.fit_transform(stem_word_list)
positive_words = ['good', 'nice', 'great', 'super', 'cool']
y = vectorizer.transform(stem_word_list)

sentence_emb = model.wv.similarity(X)
sentence_emb = model.wv.similarity(y)

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import wsd
from nltk.corpus import wordnet as wn
#from spacy.cli import download
#from spacy import load
import warnings
import numpy as np
import nltk
from nltk.corpus import stopwords
import re
import tensorflow
from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential, load_model
#from keras_preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
#from keras import preprocessing
#from keras.preprocessing import sequence
#from keras_preprocessing.sequence import pad_sequences
from keras.utils import pad_sequences
#from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

MAX_SEQUENCE_LENGTH = 5000
model = load_model('model_3.h5')

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('wordnet2022')



def preprocess_corpus(messages):
    corpus = []
    for text in messages:
        text = str(text)  # Convert to string
        review = re.sub('[^a-zA-Z]', ' ', text)
        review = review.lower()
        review = review.split()
        review = [word for word in review if not word in stopwords.words('english')]
        review = ' '.join(review)
        corpus.append(review)
        
    return corpus

## TFidf Vectorizer


def dnn(string1):
    preprocessed_string = preprocess_corpus([string1])
    tfidf_v=TfidfVectorizer(max_features=5000,ngram_range=(1,3))
    X=tfidf_v.fit_transform(preprocessed_string).toarray()
    data = pad_sequences(X, padding = 'post', maxlen = MAX_SEQUENCE_LENGTH)
    #reshaped_vector = np.pad(X, ((0, 0), (0, 5000-592)), mode='constant')
    #reshaped_vector = np.reshape(X, (1, 5000))
    prediction = model.predict(data)
    if prediction >= 0.5:
        return 'False'
    else:
        return 'True'

# test_text = """India’s Manipur state is on the boil, with social media flooded with visuals of people killed and injured in armed attacks. Violent protests broke out after an attack on June 29, while a church was looted and burned down two days prior.
# Over the past two months, Manipur’s largely Hindu Meitei community, which constitute a little over half of the state’s population, and the Christian-majority Kuki tribal group, which makes up about 16 percent of the population, have violently attacked each other in an outpouring of recrimination and revenge. Over 100 people have been killed and nearly 40,000 displaced. Angry mobs and armed vigilantes have burned down homes, churches, and offices.
# Manipur has long faced secessionist insurgencies in which both military and state security forces have committed serious human rights abuses. Longstanding ethnic disputes have also  erupted into violence. However, instead of adopting measures that would ensure the security of all communities, the Bharatiya Janata Party government of N. Biren Singh in Manipur state has replicated the national party’s politically motivated divisive policies that promote Hindu majoritarianism.
# Many Meitei seek the same affirmative action privileges that are provided to the Kuki under their protected tribal status. Tribal groups, particularly the Kuki, have argued that this would expand Meitei economic dominance and allow them to take over land in tribal areas.
# To address this explosive issue, the government needs to be trusted by all sides to play an impartial role as mediator. Instead, the Singh government has stoked ethnic divides with policy decisions that impact Kuki forest rights, and with unfair allusions to illegal immigrants, drug trade, deforestation, and militancy that fuelled anxiety among the Meitei.
# The authorities are asking for calm and restraint on all sides, but as long as there is distrust of the government, the threat of violence will persist. Survivors and families of victims need redress and accountability. The government should ensure unhindered access to humanitarian aid and the internet, take steps to demobilize and disarm abusive groups, and order an independent investigation. Mediation efforts should include all stakeholders and should be centered around ending violence and ensuring that all communities are protected."""


# test_text = test_text.replace("\n", " ")
# laptop = dnn(test_text)
# print(laptop)

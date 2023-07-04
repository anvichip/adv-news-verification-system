import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    sentences = sent_tokenize(text)
    preprocessed_sentences = []

    for sentence in sentences:
        tokens = word_tokenize(sentence.lower())
        filtered_tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalnum() and token not in stop_words]
        preprocessed_sentences.append(filtered_tokens)

    return preprocessed_sentences

def find_common_sentences(text1, text2):
    # Join the elements of the nested list into a single string
    joined_string = ' '.join([' '.join(sublist) for sublist in text1])
    # Remove unnecessary whitespaces
    joined_string = ' '.join(joined_string.split())
    preprocessed_text1 = preprocess_text(joined_string)
    preprocessed_text2 = preprocess_text(text2)
    common_sentences = []

    for sentence1 in preprocessed_text1:
        for sentence2 in preprocessed_text2:
            if sentence1 == sentence2:
                common_sentences.append(' '.join(sentence1))

    return common_sentences

    
# text1 = "The weather is sunny today. I'm going to the beach."
# text2 = "I'm going to the beach. It's a beautiful day outside."
# common_sentences = find_common_sentences(text1, text2)
# print("Common Sentences:")
# for sentence in common_sentences:
#     print(sentence)

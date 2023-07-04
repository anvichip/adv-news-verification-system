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
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from googlesearch import search
import requests
from bs4 import BeautifulSoup
from nltk import ngrams
from model_functions import dnn

def search_google(query, num_results):
  search_results = []
  # Perform the Google search
  for result in search(query, num_results=num_results):
    search_results.append(result)
  return search_results

def extractive_summarize_text(text, n):
    sentences = text.split(".")
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(sentences)
    similarity_matrix = cosine_similarity(tfidf_matrix)
    scores = np.array(similarity_matrix.sum(axis=0)).flatten()
    top_sentence_indices = scores.argsort()[-n:]
    top_sentences = [sentences[i] for i in sorted(top_sentence_indices)]
    summarized_text = " ".join(top_sentences)
    return summarized_text


def extract_paragraphs(html):
    soup = BeautifulSoup(html, 'html.parser')
    paragraphs = soup.find_all('p')
    return paragraphs

def extract_web_content(links):
    all_content_list = []  # Initialize the list outside the loops

    for link in links:
        response = requests.get(link)
        html = response.text
        # Remove unwanted part from the HTML
        start_index = html.find('<html')
        end_index = html.rfind('</html>') + len('</html>')
        html = html[start_index:end_index]
        p_tags = extract_paragraphs(html)

        content_list = []  # Move the initialization outside the loop

        for p_tag in p_tags:
            content = p_tag.get_text()
            content_list.append(content.strip())

        all_content_list.append(content_list)  # Append the content_list to all_content_list
    return all_content_list
    # Print all the elements in all_content_list
    # for content_list in all_content_list:
    #     for content in content_list:
    #         print(content)
    #     print('-----------------------------------------------------------------')


def summarize_pages(all_content_list):
    summaries = []
    for i in all_content_list:
    #print(i)
        if i != []:
            i = [item.strip() for item in i]
            text = (' ').join(i)
        #print(i)
        #print(text)
            summary = extractive_summarize_text(text,5)
            summaries.append(summary)
            #print(summary)
            #print('------------------------------------------')
    return summaries


    
def extractive_summarize_nested_text(texts, n):
    all_sentences = []
    for webpage_content in texts:
        for text in webpage_content:
            sentences = text.split(".")
            valid_sentences = [sentence.strip() for sentence in sentences if len(sentence.strip()) > 1]
            all_sentences.extend(valid_sentences)

    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(all_sentences)
    similarity_matrix = cosine_similarity(tfidf_matrix)
    scores = np.array(similarity_matrix.sum(axis=0)).flatten()
    top_sentence_indices = scores.argsort()[-n:]
    top_sentences = [all_sentences[i] for i in sorted(top_sentence_indices)]
    summarized_text = " ".join(top_sentences)
    return summarized_text

def calculate_cosine_similarity(str1, str2):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([str1, str2])
    return cosine_similarity(tfidf_matrix)[0, 1]

def calculate_jaccard_similarity(str1, str2):
    ngram_size = 1  # Change this to the desired n-gram size
    ngrams_str1 = set(ngrams(str1.split(), n=ngram_size))
    ngrams_str2 = set(ngrams(str2.split(), n=ngram_size))
    return len(ngrams_str1.intersection(ngrams_str2)) / len(ngrams_str1.union(ngrams_str2))



def scoring(internet,input):
    cosine_similarity_score = calculate_cosine_similarity(internet,input)
    jaccard_similarity_score = calculate_jaccard_similarity(internet,input)
    if cosine_similarity_score >= 0.5 or jaccard_similarity_score >= 0.5:
        #final_article = 
        return 'True'
    else:
        dnn_result = dnn(input)
        return dnn_result
        
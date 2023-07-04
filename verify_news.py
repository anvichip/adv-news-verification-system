from helper_functions import search_google, extract_web_content, extractive_summarize_nested_text, extractive_summarize_text, scoring
from article_generation import find_common_sentences
def verify(x):
    headline = x[0]
    author = x[1]
    article = x[2]
    links_list = search_google(headline,5)
    content = extract_web_content(links_list)
    internet_news_summary = extractive_summarize_nested_text(content,5)
    input_summary = extractive_summarize_text(article,5)  
    conclusion = scoring(internet_news_summary, input_summary)
    if conclusion == True:
        final_article = find_common_sentences(content,article)
        final_article = '.'.join(final_article)
    else:
        final_article = internet_news_summary
    return conclusion,final_article


    

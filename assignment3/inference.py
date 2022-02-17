# Parth Shah NLP A3 Inference

import sys
import re
from gensim.models import Word2Vec

# Reading the input word list.
def read_words(word_list):
    inp = list()
    with open(word_list, "r") as f:
        for i in f:
            inp.append(re.sub(r"[\.*]|[\,*]|[\'*]|[ *]|[\!*]|[\?*]|[\"*]|[#]|[\$]|[%]|[&]|[\(]|[\)]|[\*]|[\+]|[/]|[:]|[;]|[<]|[=]|[>]|[@]|[\[]|\\|[\]]|[\^]|[`]|[\{]|[\|]|[\}]|[~]|[-]|[\t]|[\n]", '', str(i)))
    return inp

# Print the 20 most similar words from model for each word
def print_similar(filtered_words):
    w2v_model = Word2Vec.load('data/w2v.model')
    for word in filtered_words:
        try:
            print(f"\nTop-20 most similar words for {word} are: ")
            most_similar = w2v_model.wv.most_similar(word, topn=20)
            for similar in most_similar:
                print(similar[0], ":", similar[1])
        except:
            print(word, " was not found in dataset.")

if __name__ == '__main__':
    word_list = sys.argv[1]
    filtered_words = read_words(word_list)
    print_similar(filtered_words)

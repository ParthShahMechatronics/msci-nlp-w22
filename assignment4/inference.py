# Parth Shah NLP A4 Inference

import sys
import re
from gensim.models import Word2Vec

# Reading the input word list.
def read_sens(sentences):
    inp = list()
    with open(sentences, "r") as f:
        for i in f:
            inp.append(re.sub(r"[\.*]|[\,*]|[\'*]|[ *]|[\!*]|[\?*]|[\"*]|[#]|[\$]|[%]|[&]|[\(]|[\)]|[\*]|[\+]|[/]|[:]|[;]|[<]|[=]|[>]|[@]|[\[]|\\|[\]]|[\^]|[`]|[\{]|[\|]|[\}]|[~]|[-]|[\t]|[\n]", '', str(i)))
    return inp

# Classify based on model chosen
def classify(filtered_sens, model_type):
    # load model 
    return 

if __name__ == '__main__':
    sentences = sys.argv[0]
    classifier_type = sys.argv[1]
    filtered_sens = read_sens(sentences)
    classify(filtered_sens, classifier_type)

# Parth Shah NLP A3

import os
import re
from gensim.models import Word2Vec
import argparse

parser = argparse.ArgumentParser(description='Enter folder location for Amazon corpus')
parser.add_argument('corp', type=str, help='Path to corpus folder:', default='../assignment1/')
args = parser.parse_args()

def read_dataset(data_path):
    # reads the raw dataset and returns all the tokenized lines as an array
    with open(os.path.join(data_path, 'pos.txt')) as f:
        pos_lines = f.readlines()
    with open(os.path.join(data_path, 'neg.txt')) as f:
        neg_lines = f.readlines()
    all_lines = pos_lines + neg_lines

    return [re.sub(r"[\.*]|[\,*]|[\'*]|[ *]|[\!*]|[\?*]|[\"*]|[#]|[\$]|[%]|[&]|[\(]|[\)]|[\*]|[\+]|[/]|[:]|[;]|[<]|[=]|[>]|[@]|[\[]|\\|[\]]|[\^]|[`]|[\{]|[\|]|[\}]|[~]|[-]|[\t]|[\n]", ' ', line).strip().split() for line in all_lines]

if __name__ == '__main__':
    corp_path = args.corp
    # Train a word2vec model on the given dataset
    all_lines = read_dataset(corp_path)
    print('Training word2vec model')
    # This will take some to finish
    w2v = Word2Vec(all_lines, vector_size=100, window=5, min_count=1, workers=4)
    w2v.save('data/w2v.model')
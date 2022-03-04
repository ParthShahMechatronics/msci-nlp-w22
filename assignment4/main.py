# Parth Shah A4 NLP
from cProfile import label
import sys
import re
from gensim.models import Word2Vec
import os
import gensim
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

# from keras.models import Sequential
# from keras.layers import Input, Dense, Embedding, Dropout, BatchNormalization, Activation
# from keras.preprocessing.text import text_to_word_sequence, Tokenizer
# from keras.preprocessing.sequence import pad_sequences
import argparse

parser = argparse.ArgumentParser(description='Enter folder location from A1')
parser.add_argument('--splits', type=str, help='Path to splits folder:', default='../assignment1/data/')
args = parser.parse_args()

# read in csv files from A1
def read_csv(data_path, isLab):
    with open(data_path) as f:
        data = f.readlines()
        
    if isLab:
        return [int(label) for label in data]

    return [re.sub(r"[\.*]|[\'*]|[\,*]|[\[]|[\]]|[\n]", '', line).strip().split() for line in data]

def load_data(data_dir):
    s_train = read_csv(os.path.join(data_dir, 'train.csv'), False)
    s_val = read_csv(os.path.join(data_dir, 'val.csv'), False)
    s_test = read_csv(os.path.join(data_dir, 'test.csv'), False)
    train_l = read_csv(os.path.join(data_dir, 'train_l.csv'), True)
    val_l = read_csv(os.path.join(data_dir, 'val_l.csv'), True)
    test_l = read_csv(os.path.join(data_dir, 'test_l.csv'), True)
    # without stop words
    ns_train = read_csv(os.path.join(data_dir, 'train_ns.csv'), False)
    ns_val = read_csv(os.path.join(data_dir, 'val_ns.csv'), False)
    ns_test = read_csv(os.path.join(data_dir, 'test_ns.csv'), False)

    return s_train, s_val, s_test, train_l, val_l, test_l, ns_train, ns_val, ns_test


if __name__ == '__main__':
    splits_path = args.splits
    s_train, s_val, s_test, train_l, val_l, test_l, ns_train, ns_val, ns_test = load_data(splits_path)

    # debuggin
    # s_train = read_csv(os.path.join(splits_path, 'train.csv'), False)
    # print(s_train[5])
    # print(s_train[5][1])
    # train_l = read_csv(os.path.join(splits_path, 'train_l.csv'), True)
    # train_l = [int(label) for label in train_l]
    # print(train_l[5])
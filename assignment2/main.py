# Parth Shah NLP A2

import os
import re
import argparse
import pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

parser = argparse.ArgumentParser(description='Enter folder location for splits')
parser.add_argument('--splits', type=str, help='Path to splits folder:', default='../assignment1/data/')

args = parser.parse_args()

def read_csv(data_path, label_bool):
    final_arr =[]
    with open(data_path) as f:
        data = f.readlines()
        for line in data:
            arr = []
            # word = re.sub(r"\[|\]|\'|\n", '', line)
            # word = ' '.join(line.strip().split(','))
            words = re.split("[\.*]|[\,*]|[\'*]|[\[]|[\]]|[\n]", line)
            for word in words:
                if word != '' and word != ' ':
                    if label_bool == True:
                        arr.append(int(word))
                    else:
                        arr.append(word)
            final_arr.append(arr)

    return final_arr

    # return [' '.join(line.strip().split(',')) for line in data]


def load_data(data_dir):
    s_train = read_csv(os.path.join(data_dir, 'train.csv'),False)
    s_val = read_csv(os.path.join(data_dir, 'val.csv'),False)
    s_test = read_csv(os.path.join(data_dir, 'test.csv'), False)
    train_l = read_csv(os.path.join(data_dir, 'train_l.csv'), True)
    # train_l = [int(label) for label in train_l]
    val_l = read_csv(os.path.join(data_dir, 'val_l.csv'), True)
    # val_l = [int(label) for label in val_l]
    test_l = read_csv(os.path.join(data_dir, 'test_l.csv'), True)
    # test_l = [int(label) for label in test_l]
    # without stop words
    ns_train = read_csv(os.path.join(data_dir, 'train_ns.csv'), False)
    ns_val = read_csv(os.path.join(data_dir, 'val_ns.csv'), False)
    ns_test = read_csv(os.path.join(data_dir, 'test_ns.csv'),False)

    return s_train, s_val, s_test, train_l, val_l, test_l, ns_train, ns_val, ns_test

def train(x_train, y_train):
    print('Calling CountVectorizer')
    # ngram_range=(x,y) # 1,1 is unigram 2,2 is bigram 1,2 is unigram bigram
    count_vect = CountVectorizer()
    x_train_count = count_vect.fit_transform(x_train)
    print('Building Tf-idf vectors')
    tfidf_transformer = TfidfTransformer()
    x_train_tfidf = tfidf_transformer.fit_transform(x_train_count)
    print('Training MNB')
    clf = MultinomialNB().fit(x_train_tfidf, y_train)
    return clf, count_vect, tfidf_transformer


def evaluate(x, y, clf, count_vect, tfidf_transformer):
    x_count = count_vect.transform(x)
    x_tfidf = tfidf_transformer.transform(x_count)
    preds = clf.predict(x_tfidf)
    return {
        'accuracy': accuracy_score(y, preds),
        'precision': precision_score(y, preds),
        'recall': recall_score(y, preds),
        'f1': f1_score(y, preds),
        }

if __name__ == '__main__':
    splits_path = args.splits

    s_train, s_val, s_test, train_l, val_l, test_l, ns_train, ns_val, ns_test = load_data(splits_path)
    # s_train = read_csv(os.path.join(splits_path, 'train.csv'))
    print(val_l[0])
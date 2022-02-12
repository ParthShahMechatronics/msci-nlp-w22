# Parth Shah NLP A2

import os
import re
import argparse
import pickle
import numpy as np
from pprint import pprint
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

parser = argparse.ArgumentParser(description='Enter folder location for splits')
parser.add_argument('--splits', type=str, help='Path to splits folder:', default='../assignment1/data/')
args = parser.parse_args()

def read_csv(data_path):
    with open(data_path) as f:
        data = f.readlines()
    return [' '.join(re.sub(r"[\.*]|[\'*]|[\[]|[\]]|[\n]", '', line).strip().split(',')) for line in data]

def load_data(data_dir):
    s_train = read_csv(os.path.join(data_dir, 'train.csv'))
    s_val = read_csv(os.path.join(data_dir, 'val.csv'))
    s_test = read_csv(os.path.join(data_dir, 'test.csv'))
    train_l = read_csv(os.path.join(data_dir, 'train_l.csv'))
    train_l = [int(label) for label in train_l]
    val_l = read_csv(os.path.join(data_dir, 'val_l.csv'))
    val_l = [int(label) for label in val_l]
    test_l = read_csv(os.path.join(data_dir, 'test_l.csv'))
    test_l = [int(label) for label in test_l]
    # without stop words
    ns_train = read_csv(os.path.join(data_dir, 'train_ns.csv'))
    ns_val = read_csv(os.path.join(data_dir, 'val_ns.csv'))
    ns_test = read_csv(os.path.join(data_dir, 'test_ns.csv'))

    return s_train, s_val, s_test, train_l, val_l, test_l, ns_train, ns_val, ns_test

def train(x_train, y_train, tupl, val_set, val_l):
    print('Calling CountVectorizer')
    # ngram_range=(x,y) # 1,1 is unigram 2,2 is bigram 1,2 is unigram+bigram
    count_vect = CountVectorizer(ngram_range=tupl)
    x_train_count = count_vect.fit_transform(x_train)
    x_val_count = count_vect.fit_transform(val_set)
    print('Building Tf-idf vectors')
    tfidf_transformer = TfidfTransformer()
    x_train_tfidf = tfidf_transformer.fit_transform(x_train_count)
    x_val_tfidf = tfidf_transformer.fit_transform(x_val_count)
    print('Training MNB for' + str(tupl))
    # TODO: need to find best alpha val 
    clf = MultinomialNB().fit(x_train_tfidf, y_train)
    params = {'alpha':np.linspace(0.1, 5.0, num = 10)}
    best_clf = GridSearchCV(clf, params).fit(x_val_tfidf, val_l)

    return best_clf, count_vect, tfidf_transformer


# x - train/test data, y - labels
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

    # debuggin
    # s_train = read_csv(os.path.join(splits_path, 'train.csv'))
    # print(ns_test[0])
    # print(ns_test[0][0])

    scores={}

    for (x,y) in (1,1),(2,2),(1,2):
        if(x,y) == (1,1):
            name = 'uni'
        if(x,y) == (2,2):
            name = 'bi'
        if(x,y) == (1,2):
            name = 'uni_bi'
        # with stop words
        clf, count_vect, tfidf_transformer = train(s_train, train_l, (x,y), s_val, val_l)

        with open('assignment2/data/mnb_'+ name +'.pkl', 'wb') as f:
            pickle.dump(clf, f)

        with open('assignment2/data/count_vect_'+ name +'.pkl', 'wb') as f:
            pickle.dump(count_vect, f)

        with open('assignment2/data/tfidf_transformer_'+ name +'.pkl', 'wb') as f:
            pickle.dump(tfidf_transformer, f)     
           
        scores['val_' + name] = evaluate(s_val, val_l, clf, count_vect, tfidf_transformer) 
        scores['test_' + name] = evaluate(s_test, test_l, clf, count_vect, tfidf_transformer) 

        # without sw
        clf_ns, count_vect_ns, tfidf_transformer_ns = train(ns_train, train_l, (x,y), ns_val, val_l)

        with open('assignment2/data/mnb_'+ name +'_ns.pkl', 'wb') as f:
            pickle.dump(clf_ns, f)

        with open('assignment2/data/count_vect_'+ name +'_ns.pkl', 'wb') as f:
            pickle.dump(count_vect_ns, f)

        with open('assignment2/data/tfidf_transformer_'+ name +'_ns.pkl', 'wb') as f:
            pickle.dump(tfidf_transformer_ns, f)
        
        scores['val_ns_' + name] = evaluate(ns_val, val_l, clf_ns, count_vect_ns, tfidf_transformer_ns) 
        scores['test_ns_' + name] = evaluate(ns_test, test_l, clf_ns, count_vect_ns, tfidf_transformer_ns) 

    pprint(scores)

# Parth Shah A4 NLP
import re
from gensim.models import Word2Vec
import os
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Embedding, Dropout, BatchNormalization, Activation, Flatten
from tensorflow.keras.preprocessing.text import text_to_word_sequence, Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import argparse

# These are some hyperparameter constants that can be tuned
MAX_SENT_LEN = 20
MAX_VOCAB_SIZE = 20000
EMBEDDING_DIM = 100
BATCH_SIZE = 64
N_EPOCHS = 10

parser = argparse.ArgumentParser(description='Enter folder location from A1')
parser.add_argument('--splits', type=str, help='Path to splits folder:', default='../assignment1/data/')
args = parser.parse_args()

# read in csv files from A1
def read_csv(data_path, isLab):
    with open(data_path) as f:
        data = f.readlines()
    if isLab:
        # categorize labels
        store = []
        for label in data:
            if int(label) == 0:
                store.append([0,1])
            else:
                store.append([1,0])
        return store
        # return [int(label) for label in data]

    return [re.sub(r"[\.*]|[\'*]|[\,*]|[\[]|[\]]|[\?*]|[\n]", '', line).strip().split() for line in data]
    # return [' '.join(re.sub(r"[\.*]|[\'*]|[\,*]|[\[]|[\]]|[\?*]|[\n]", '', line).strip().split()) for line in data]

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

def train_model(activation_name, embeddings_matrix, X_train, y_train, X_test, y_test):
    # Build a sequential model by stacking neural net units
    model = Sequential()
    model.add(Embedding(input_dim= len(tokenizer.word_index)+1,
                            output_dim=EMBEDDING_DIM,
                            weights = [embeddings_matrix], trainable=False, input_length=MAX_SENT_LEN,name='word_embedding_layer', 
                            mask_zero=True))
    model.add(Flatten())
    model.add(Dense(units = 128,  activation= activation_name, name='hidden_layer'))
    model.add(Dropout(0.15))
    model.add(Dense(units=2, activation='softmax', name='output_layer'))
    model.add(Dropout(0.15))
    model.summary()    
    model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=N_EPOCHS, validation_data=(X_test, y_test))
    # model.save('assignment4/data/' + activation_name + '.model')
    
    return model


if __name__ == '__main__':
    splits_path = args.splits
    w2v_dir = 'assignment3/data/w2v.model'

    s_train, s_val, s_test, train_l, val_l, test_l, ns_train, ns_val, ns_test = load_data(splits_path)

    tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)

    tokenizer.fit_on_texts(s_train)
    s_train = tokenizer.texts_to_sequences(s_train)
    s_train = pad_sequences(s_train, maxlen = MAX_SENT_LEN, padding='post', truncating='post')

    tokenizer.fit_on_texts(s_val)
    s_val = tokenizer.texts_to_sequences(s_val)
    s_val = pad_sequences(s_val, maxlen = MAX_SENT_LEN, padding='post', truncating='post')

    tokenizer.fit_on_texts(s_test)
    s_test = tokenizer.texts_to_sequences(s_test)
    s_test = pad_sequences(s_test, maxlen = MAX_SENT_LEN, padding='post', truncating='post')

    # Load the word2vec embeddings
    embeddings = Word2Vec.load(w2v_dir)
    print('Building embedding matrix')
    embeddings_matrix = np.random.uniform(-0.05, 0.05, size=(len(tokenizer.word_index)+1, EMBEDDING_DIM)) # +1 is because the matrix indices start with 0
    for word, i in tokenizer.word_index.items(): # i=0 is the embedding for the zero padding
        try:
            embeddings_vector = embeddings.wv[word]
        except KeyError:
            embeddings_vector = None
        if embeddings_vector is not None:
            embeddings_matrix[i] = embeddings_vector

    s_train = np.array(s_train)
    s_test = np.array(s_test)
    s_val = np.array(s_val)
    train_l = np.array(train_l)
    test_l = np.array(test_l)
    val_l= np.array(val_l)

    model_sig = train_model('sigmoid', embeddings_matrix, s_train, train_l, s_test, test_l)
    model_relu = train_model('relu', embeddings_matrix, s_train, train_l, s_test, test_l)
    model_tanh = train_model('tanh', embeddings_matrix, s_train, train_l, s_test, test_l)

    eval_sig = model_sig.evaluate(s_test, test_l, batch_size=BATCH_SIZE)
    eval_relu = model_relu.evaluate(s_test, test_l, batch_size=BATCH_SIZE)
    eval_tanh = model_tanh.evaluate(s_test, test_l, batch_size=BATCH_SIZE)

    print("\n Sigmoid Evaluation (loss, accuracy): ", eval_sig)
    print("\n ReLU Evaluation (loss, accuracy): ", eval_relu)
    print("\n Tanh Evaluation (loss, accuracy): ", eval_tanh)

    # debuggin
    # s_train = read_csv(os.path.join(splits_path, 'train.csv'), False)
    # print(s_train[6])
    # print(s_train[6][1])
    # train_l = read_csv(os.path.join(splits_path, 'train_l.csv'), True)
    # train_l = [int(label) for label in train_l]
    # print(train_l[5])

    # # build vocab
    # # NOTE: this script only considers tokens in the training set to build the
    # # vocabulary object.
    # vectorizer = TextVectorization(max_tokens=MAX_VOCAB_SIZE,
    #                                output_sequence_length=MAX_SENT_LEN)
    # text_data = tf.data.Dataset.from_tensor_slices(x_train_val).batch(BATCH_SIZE)
    # print('Building vocabulary')
    # vectorizer.adapt(text_data)
    # # NOTE: in this vocab, index 0 is reserved for padding and 1 is reserved
    # # for out of vocabulary tokens
    # vocab = vectorizer.get_vocabulary()

# def build_embedding_mat(data_dir, vocab, w2v):
#     """
#     Build the embedding matrix which will be used to initialize weights of
#     the embedding layer in our seq2seq architecture
#     """
#     # we have 4 special tokens in our vocab
#     token2word = {0: '<sos>', 1: '<pad>', 2: '<eos>', 3: '<unk>'}
#     word2token = {'<sos>': 0, '<pad>': 1, '<eos>': 2, '<unk>': 3}
#     # +4 for the four vocab tokens
#     vocab_size = len(vocab) + 4
#     embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM))

#     # randomly initizlize embeddings for the special tokens
#     # you can play with different types of initializers
#     embedding_matrix[0] = np.random.random((1, EMBEDDING_DIM))
#     embedding_matrix[1] = np.random.random((1, EMBEDDING_DIM))
#     embedding_matrix[2] = np.random.random((1, EMBEDDING_DIM))
#     embedding_matrix[3] = np.random.random((1, EMBEDDING_DIM))
#     for i, word in enumerate(vocab):
#         # since a word in the vocab of our vectorizer is actually stored as
#         # byte values, we need to decode them as strings explicitly
#         word = word.decode('utf-8')
#         try:
#             # again, +4 for the four special tokens in our vocab
#             embedding_matrix[i+4] = w2v[word]
#             # build token-id -> word dict (will be used when decoding)
#             token2word[i+4] = word
#             # build word -> token-id dict (will be used when encoding)
#             word2token[word] = i+4
#         except KeyError as e:
#             # skip any oov words from the perspective of our trained w2v model
#             continue
#     # save the two dicts
#     # with open(os.path.join(data_dir, 'token2word.json'), 'w') as f:
#     #     json.dump(token2word, f)
#     # with open(os.path.join(data_dir, 'word2token.json'), 'w') as f:
#     #     json.dump(word2token, f)

#     return embedding_matrix, word2token

# def seq_pad(tokenizer, data):
#     sens = [text_to_word_sequence(line) for line in data]
#     tokenizer.fit_on_texts([' '.join(sen[:MAX_SENT_LEN]) for sen in sens])
#     data = tokenizer.texts_to_sequences([' '.join(sen[:MAX_SENT_LEN]) for sen in sens])    
#     data = pad_sequences(data, maxlen = MAX_SENT_LEN, padding='post', truncating='post')

#     return data
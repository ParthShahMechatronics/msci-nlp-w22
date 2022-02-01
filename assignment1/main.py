# Parth Shah NLP A1
# 1. Tokenize the corpus 
# 2. Remove the following special characters: !"#$%&()*+/:;<=>@[\\]^`{|}~\t\n
# 3. Create two versions of your dataset: (1) with stopwords and (2) without stopwords. 
# Stopword lists are available online. 
# 4. Randomly split your data into training (80%), validation (10%) and test (10%) sets.

import random
import re
import argparse

parser = argparse.ArgumentParser(description='Enter location for pos.tx and neg.tx')
parser.add_argument('pos_in', type=str, help='Path to pos.txt:', default='pos.txt')
parser.add_argument('neg_in', type=str, help='Path to neg.txt:', default='neg.txt')
args = parser.parse_args()

# stop words list obtained from https://gist.github.com/sebleier/554280?permalink_comment_id=3056587#gistcomment-3056587
sw = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]

final_sw = list() # final tokenized sentences with stop words
final_nsw = list() # final tokenized sentences with no stop words

def parse_into_list(path, listname, val):
    file_read = open(path, "r")
    for f in file_read:
        if val == -1:
            listname.append(f)
        else:
            listname.append([f,val])
    file_read.close

def tokenize_corpus(corp):

    for line in corp:
        phrase = line[0]
        phrase_tok = re.split("[\.*]|[\,*]|[\'*]|[ *]|[\!*]|[\"*]|[#]|[\$]|[%]|[&]|[\(]|[\)]|[\*]|[\+]|[/]|[:]|[;]|[<]|[=]|[>]|[@]|[\[]|\\|[\]]|[\^]|[`]|[\{]|[\|]|[\}]|[~]|[-]|[\t]|[\n]", phrase) #|[-]

        tok_sw = list() # tokens with stop words
        tok_nsw = list() # tokens with no stop words

        for word in phrase_tok:
            if word != '':
                tok_sw.append(word.lower())
                if word.lower() not in sw:
                    tok_nsw.append(word.lower())                    
        
        final_sw.append([tok_sw, line[1]])
        final_nsw.append([tok_nsw, line[1]])

# Expected output files:  
# 1. out.csv: tokenized sentences w/ stopwords  
# 2. train.csv: training set w/ stopwords  
# 3. val.csv: validation set w/ stopwords  
# 4. test.csv: test set w/ stopwords  
# 5. out_ns.csv: tokenized sentences w/o stopwords  
# 6. train_ns.csv: training set w/o stopwords  
# 7. val_ns.csv: validation set w/o stopwords  
# 8. test_ns.csv: test set w/o stopwords
# 9. labels: out_l.csv, train_l.csv, val_l.csv, test_l.csv

def output_csv(total_size):

    for line in range(total_size):
        open("data/out.csv", "a").write(str(final_sw[line][0])+"\n")
        open("data/out_ns.csv", "a").write(str(final_nsw[line][0])+"\n")
        open("data/out_l.csv", "a").write(str(final_sw[line][1])+"\n")

        # training sets (80%)
        if line < (total_size * 0.8):
            open("data/train.csv", "a").write(str(final_sw[line][0])+"\n")
            open("data/train_ns.csv", "a").write(str(final_nsw[line][0])+"\n")
            open("data/train_l.csv", "a").write(str(final_sw[line][1])+"\n")

        # validation sets (10%)
        elif line < (total_size * 0.9):
            open("data/val.csv", "a").write(str(final_sw[line][0])+"\n")
            open("data/val_ns.csv", "a").write(str(final_nsw[line][0])+"\n")
            open("data/val_l.csv", "a").write(str(final_sw[line][1])+"\n")
        
        # test sets (10%)
        else:
            open("data/test.csv", "a").write(str(final_sw[line][0])+"\n")
            open("data/test_ns.csv", "a").write(str(final_nsw[line][0])+"\n")
            open("data/test_l.csv", "a").write(str(final_sw[line][1])+"\n")

    open("data/out.csv", "a").close()
    open("data/out_ns.csv", "a").close()
    open("data/out_l.csv", "a").close()
    open("data/train.csv", "a").close()
    open("data/train_ns.csv", "a").close()
    open("data/train_l.csv", "a").close()
    open("data/val.csv", "a").close()
    open("data/val_ns.csv", "a").close()
    open("data/val_l.csv", "a").close()
    open("data/test.csv", "a").close()
    open("data/test_ns.csv", "a").close()
    open("data/test_l.csv", "a").close()


if __name__ == '__main__':
    pos_path = args.pos_in
    neg_path = args.neg_in
    pos_list = list()
    neg_list = list()
    combined = list()

    parse_into_list(pos_path, pos_list, 1)
    parse_into_list(neg_path, neg_list, 0)
    combined = pos_list + neg_list

    # create a mixed corpus dataset with positive and negative reviews
    random.shuffle(combined)

    tokenize_corpus(combined)

    output_csv(len(combined))

    print("done")
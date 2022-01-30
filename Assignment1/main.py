# Parth Shah NLP A1
# 1. Tokenize the corpus 
# 2. Remove the following special characters: !"#$%&()*+/:;<=>@[\\]^`{|}~\t\n 
# 3. Create two versions of your dataset: (1) with stopwords and (2) without stopwords. 
# Stopword lists are available online. 
# 4. Randomly split your data into training (80%), validation (10%) and test (10%) sets.

import random
import sys
import re
import argparse

parser = argparse.ArgumentParser(description='Enter location for pos.tx and neg.tx')
parser.add_argument('--pos_in', type=str, help='Path to pos.txt:', default='pos.txt')
parser.add_argument('--neg_in', type=str, help='Path to neg.txt:', default='neg.txt')
args = parser.parse_args()

# stop words list obtained from https://gist.github.com/sebleier/554280?permalink_comment_id=3056587#gistcomment-3056587
sw = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]

final_sw = list()
final_nsw = list()

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
        phrase_tok = re.split("[\.*]|[\,*]|[\'*]|[ *]|[\!*]|[\"*]|[#]|[\$]|[%]|[&]|[\(]|[\)]|[\*]|[\+]|[/]|[:]|[;]|[<]|[=]|[>]|[@]|[\[]|\\|[\]]|[\^]|[`]|[\{]|[\|]|[\}]|[~]|[\t]|[\n]", phrase)

        tok_sw = list() # tokens with stop words
        tok_nsw = list() # tokens with no stop words

        for word in phrase_tok:
            if word != '':
                tok_sw.append(word.lower())
                if word.lower() not in sw:
                    tok_nsw.append(word.lower())                    
        
        final_sw.append([tok_sw, line[1]])
        final_nsw.append([tok_nsw, line[1]])


if __name__ == '__main__':
    pos_path = args.pos_in
    neg_path = args.neg_in
    pos_list = list()
    neg_list = list()
    combined = list()

    parse_into_list(pos_path, pos_list, 1)
    parse_into_list(neg_path, neg_list, 0)
    combined = pos_list + neg_list

    tokenize_corpus(combined)

    # # testing
    # index = 0
    # print(final_sw[index])
    # print(final_nsw[index])  

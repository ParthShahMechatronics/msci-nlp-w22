# Parth Shah NLP A1
import random
import sys
import argparse

parser = argparse.ArgumentParser(description='Enter location for pos.tx and neg.tx')
parser.add_argument('pos_in', type=str, help='Path to pos.txt:', default='pos.tx')
parser.add_argument('neg_in', type=str, help='Path to neg.txt:', default='neg.tx')
args = parser.parse_args()

if __name__ == '__main__':
    pos_path = args.pos_in
    neg_path = args.neg_in

from nltk import Tree
import re
from stanfordcorenlp import StanfordCoreNLP
import os

nlp_server = StanfordCoreNLP('/opt/stanford-corenlp')

# The script generate .cparents file
# the phrases generated from .toks file
# each number in .cparents represents the id of phrases

def constituency_parse(sentence, nlp_server=nlp_server):
    try:
        s = nlp_server.parse(sentence)
    except ConnectionError:
        print('ConnectionError')
        nlp_server = StanfordCoreNLP('/opt/stanford-corenlp')
    w = Tree.fromstring(s)
    tr = str(w).split('\n')
    reg = re.compile('\(.*?\)', re.S)
    reg1 = re.compile('\)(.*?) ', re.S)
    tr1 = [reg.findall(line) for line in tr if reg.findall(line) != []]
    processed_tree = []
    for line in tr1:
        _tmp = []
        for elem in line:
            rev_elem = elem[::-1]
            _tmp.append(' '.join([item[::-1] for item in reg1.findall(rev_elem)]))
        processed_tree.append(_tmp)
    return processed_tree

if __name__ == '__main__':
    base_dir = os.path.dirname("/home/saliency/waterloo/Castor-data/datasets/WikiQA/")
    train_dir = os.path.join(base_dir, 'train')
    dev_dir = os.path.join(base_dir, 'dev')
    test_dir = os.path.join(base_dir, 'test')

    for dirpath in [train_dir, dev_dir, test_dir]:
        filepath = os.path.join(dirpath, "a.toks")
        cparentpath = os.path.join(dirpath, "a.cparents")
        cparent_file = open(cparentpath, "a+")
        cparent_idx = []

        lines = open(filepath).readlines()
        for line in lines:
            line = line.rstrip("\n")
            line_phrases = constituency_parse(line)
            line_idx = []
            k = 0
            for phrase in line_phrases:
                for _ in range(len(phrase)):
                    line_idx.append(str(k))
                k += 1
            cparent_idx.append(line_idx)

        for ele in cparent_idx:
            cparent_file.write(" ".join(ele) + "\n")

        cparent_file.close()
        print(dirpath)

    for dirpath in [train_dir, dev_dir, test_dir]:
        filepath = os.path.join(dirpath, "b.toks")
        cparentpath = os.path.join(dirpath, "b.cparents")
        cparent_file = open(cparentpath, "a+")
        cparent_idx = []

        lines = open(filepath).readlines()
        for line in lines:
            line = line.rstrip("\n")
            line_phrases = constituency_parse(line)
            line_idx = []
            k = 0
            for phrase in line_phrases:
                for _ in range(len(phrase)):
                    line_idx.append(str(k))
                k += 1
            cparent_idx.append(line_idx)

        for ele in cparent_idx:
            cparent_file.write(" ".join(ele) + "\n")

        cparent_file.close()
        print(dirpath)

















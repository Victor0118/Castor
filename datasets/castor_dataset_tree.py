from abc import ABCMeta, abstractmethod
import os
from sys import exit

import numpy as np
import torch
from torchtext.data.dataset import Dataset
from torchtext.data.example import Example

from datasets.idf_utils import get_pairwise_word_to_doc_freq, get_pairwise_overlap_features
from datasets.binary_tree import BinaryTree


def add_descendant(tree, index, parent_index):
    # add to the left first if possible, then to the right
    if tree.has_left_descendant_at_node(parent_index):
        if tree.has_right_descendant_at_node(parent_index):
            sys.exit("Node " + str(parent_index) + " already has two children")
        else:
            tree.add_right_descendant(index, parent_index)
    else:
        tree.add_left_descendant(index, parent_index)

def convert_ptb_to_tree(line):
    index = 0
    tree = None
    line = line.rstrip()

    stack = []
    parts = line.split()
    for p_i, p in enumerate(parts):
        # opening of a bracket, create a new node, take parent from top of stack
        if p == '(':
            if tree is None:
                tree = BinaryTree(index)
            else:
                add_descendant(tree, index, stack[-1])
            # add the newly created node to the stack and increment the index
            stack.append(index)
            index += 1
        # close of a bracket, pop node on top of the stack
        elif p == ')':
            stack.pop(-1)
        # otherwise, create a new node, take parent from top of stack, and set word
        else:
            if len(stack) == 0:
                print(line)
            add_descendant(tree, index, stack[-1])
            tree.set_word(index, p)
            index += 1
    return tree

def gen_tree(sent):
    tree = convert_ptb_to_tree(sent)
    # print(sent)
    sent, lm, rm = tree.convert_to_sequence_and_masks(tree.root)
    m = [1.0] * len(sent)
    # assert len(lm) == 1
    #if len(lm) != 1:
    #    print(sent)
    #    print(lm)
    return sent, m, lm, rm


class CastorPairTreeDataset(Dataset, metaclass=ABCMeta):

    # Child classes must define
    NAME = None
    NUM_CLASSES = None
    ID_FIELD = None
    TEXT_FIELD = None
    EXT_FEATS_FIELD = None
    LABEL_FIELD = None
    RAW_TEXT_FIELD = None
    AID_FIELD = None
    MASK1_FIELD = None
    MASK2_FIELD = None
    LEFT_MASK1_FIELD = None
    LEFT_MASK2_FIELD = None
    RIGHT_MASK1_FIELD = None
    RIGHT_MASK2_FIELD = None
    EXT_FEATS = 4
    
    @abstractmethod
    def __init__(self, path, load_ext_feats=False):
        """
        Create a Castor dataset involving pairs of texts
        """
        fields = [('id', self.ID_FIELD), ('sentence_1', self.TEXT_FIELD), ('sentence_2', self.TEXT_FIELD),
                  ('ext_feats', self.EXT_FEATS_FIELD), ('label', self.LABEL_FIELD),
                  ('aid', self.AID_FIELD), ('sentence_1_raw', self.RAW_TEXT_FIELD), ('sentence_2_raw', self.RAW_TEXT_FIELD),
                  ('mask1', self.MASK1_FIELD), ('left_mask1', self.LEFT_MASK1_FIELD), ('right_mask1', self.RIGHT_MASK1_FIELD), 
                  ('mask2', self.MASK2_FIELD), ('left_mask2', self.LEFT_MASK2_FIELD), ('right_mask2', self.RIGHT_MASK2_FIELD)]

        examples = []
        with open(os.path.join(path, 'a.btree'), 'r') as f1, open(os.path.join(path, 'b.btree'), 'r') as f2:
            sent_list_1 = [l.rstrip('.\n').split(' ') for l in f1]
            sent_list_2 = [l.rstrip('.\n').split(' ') for l in f2]

        word_to_doc_cnt = get_pairwise_word_to_doc_freq(sent_list_1, sent_list_2)
        self.word_to_doc_cnt = word_to_doc_cnt

        if not load_ext_feats:
            overlap_feats = get_pairwise_overlap_features(sent_list_1, sent_list_2, word_to_doc_cnt)
        else:
            overlap_feats = np.loadtxt(os.path.join(path, 'overlap_feats.txt'))

        with open(os.path.join(path, 'id.txt'), 'r') as id_file, open(os.path.join(path, 'sim.txt'), 'r') as label_file:
            for i, (pair_id, l1, l2, ext_feats, label) in enumerate(zip(id_file, sent_list_1, sent_list_2, overlap_feats, label_file)):
                pair_id = pair_id.rstrip('.\n')
                label = label.rstrip('.\n')
                l1_str = " ".join(l1)
                l2_str = " ".join(l2)
                l1_new, mask1, left_mask1, right_mask1 = gen_tree(l1_str)
                l2_new, mask2, left_mask2, right_mask2 = gen_tree(l2_str)
                # print(" ".join(l1_new), l1_str)
                # print(len(l1_new), len(l1), len(mask1), len(left_mask1), len(right_mask1))
                example_list = [pair_id, l1_new, l2_new, ext_feats, label, i + 1, l1_new, l2_new, mask1, left_mask1, right_mask1, mask2, left_mask2, right_mask2]
                example = Example.fromlist(example_list, fields)
                examples.append(example)

        super(CastorPairTreeDataset, self).__init__(examples, fields)

    @classmethod
    def set_vectors(cls, field, vector_path):
        if os.path.isfile(vector_path):
            stoi, vectors, dim = torch.load(vector_path)
            field.vocab.vectors = torch.Tensor(len(field.vocab), dim)

            for i, token in enumerate(field.vocab.itos):
                wv_index = stoi.get(token, None)
                if wv_index is not None:
                    field.vocab.vectors[i] = vectors[wv_index]
                else:
                    # initialize <unk> with uniform_(-0.05, 0.05) vectors
                    field.vocab.vectors[i] = torch.FloatTensor(dim).uniform_(-0.05, 0.05)
        else:
            print("Error: Need word embedding pt file")
            exit(1)
        return field

import sys
import math
import torch
import numpy as np
import torch.nn as nn
from esim.util import *
import torch.nn.functional as F
from torch.autograd import Variable
from datetime import datetime
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .model import ESIM
from .model_tree import TreeESIM 

class TreeSeqESIM(nn.Module):
    def __init__(self, num_units, num_classes, embedding_size, dropout, device=0,
                 training=True, project_input=True,
                 use_intra_attention=False, distance_biases=10, max_sentence_length=30):
        """
        Create the model based on MLP networks.

        :param num_units: size of the networks
        :param num_classes: number of classes in the problem
        :param vocab_size: size of the vocabulary
        :param embedding_size: size of each word embedding
        :param use_intra_attention: whether to use intra-attention model
        :param training: whether to create training tensors (optimizer)
        :param project_input: whether to project input embeddings to a
            different dimensionality
        :param distance_biases: number of different distances with biases used
            in the intra-attention model
        """
        super(TreeSeqESIM, self).__init__()
        self.arch = "ESIM-tree-seq"
        self.tree_esim = TreeESIM(num_units, num_classes, embedding_size, dropout, device=device, max_sentence_length=max_sentence_length)   
        self.seq_esim = ESIM(num_units, num_classes, embedding_size, dropout, device=device, max_sentence_length=max_sentence_length)   

    def forward(self, x1, x2, ext_feats=None, word_to_doc_count=None, raw_sent1=None, raw_sent2=None, x1_mask=None, x1_left_mask=None, x1_right_mask=None, x2_mask=None, x2_left_mask=None, x2_right_mask=None, visualize=False):
        
        out_tree = self.tree_esim(x1, x2, x1_mask=x1_mask, x1_left_mask=x1_left_mask, x1_right_mask=x1_right_mask, x2_mask=x2_mask, x2_left_mask=x2_left_mask, x2_right_mask=x2_left_mask, visualize=visualize)
        if visualize:
            return out_tree

        out_seq = self.seq_esim(x1, x2, x1_mask=x1_mask, x2_mask=x2_mask)
        return (out_tree + out_seq) / 2

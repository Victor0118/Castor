from abc import ABCMeta, abstractmethod
import os
from sys import exit

import numpy as np
import torch
from torchtext.data.dataset import Dataset
from torchtext.data.example import Example

from datasets.idf_utils import get_pairwise_word_to_doc_freq, get_pairwise_overlap_features


class CastorPairDataset(Dataset, metaclass=ABCMeta):

    # Child classes must define
    NAME = None
    NUM_CLASSES = None
    ID_FIELD = None
    TEXT_FIELD = None
    EXT_FEATS_FIELD = None
    LABEL_FIELD = None
    RAW_TEXT_FIELD = None
    EXT_FEATS = 4
    AID_FIELD = None

    @abstractmethod
    def __init__(self, path, load_ext_feats=False):
        """
        Create a Castor dataset involving pairs of texts
        """
        fields = [('id', self.ID_FIELD), ('sentence_1', self.TEXT_FIELD), ('query1', self.TEXT_FIELD),
                  ('query2', self.TEXT_FIELD), ('query3', self.TEXT_FIELD), ('sentence_2', self.TEXT_FIELD),
                  ('ext_feats', self.EXT_FEATS_FIELD), ('label', self.LABEL_FIELD),
                  ('aid', self.AID_FIELD), ('sentence_1_raw', self.RAW_TEXT_FIELD), ('sentence_2_raw', self.RAW_TEXT_FIELD)]

        examples = []
        with open(os.path.join(path, 'a.toks'), 'r') as f1, open(os.path.join(path, 'b.toks'), 'r') as f2, \
            open(os.path.join(path, 'a1.toks'), 'r') as query1, open(os.path.join(path, 'a2.toks'), 'r') as query2, \
            open(os.path.join(path, 'a3.toks'), 'r') as query3:
            sent_list_1 = [l.rstrip('.\n').split(' ') for l in f1]
            query1 = [l.rstrip('.\n').split(' ') for l in query1]
            query2 = [l.rstrip('.\n').split(' ') for l in query2]
            query3 = [l.rstrip('.\n').split(' ') for l in query3]
            sent_list_2 = []
            print("reading: {}".format(os.path.join(path, 'b.toks')))
            ind = 0
            while True:
                try:
                    l = f2.readline()
                    sent_list_2.append(l.rstrip('.\n').split(' ') )
                    if "" == l:
                        print("reading doc file finished")
                        break
                    ind += 1
                except Exception as e:
                    print(ind)
                    print(e)
        word_to_doc_cnt = get_pairwise_word_to_doc_freq(sent_list_1, sent_list_2)
        self.word_to_doc_cnt = word_to_doc_cnt

        if not load_ext_feats:
            overlap_feats = get_pairwise_overlap_features(sent_list_1, sent_list_2, word_to_doc_cnt)
        else:
            overlap_feats = np.loadtxt(os.path.join(path, 'overlap_feats.txt'))

        with open(os.path.join(path, 'id.txt'), 'r') as id_file, open(os.path.join(path, 'sim.txt'), 'r') as label_file:
            try:
                print("reading docid file: {}".format(os.path.join(path, 'docid.txt')))
                aids = open(os.path.join(path, 'docid.txt'), 'r').readlines()
                print("reading docid file finshed")
            except Exception as e:
                aids = list(range(len(sent_list_1)))
                print(e)
            for pair_id, l1, q1, q2, q3, l2, ext_feats, label, aid in zip(id_file, sent_list_1, query1, query2, query3, sent_list_2, overlap_feats, label_file, aids):
                pair_id = pair_id.rstrip('.\n')
                label = label.rstrip('.\n')
                example_list = [pair_id, l1, q1, q2, q3, l2, ext_feats, label, aid, ' '.join(l1), ' '.join(l2)]
                example = Example.fromlist(example_list, fields)
                examples.append(example)

        super(CastorPairDataset, self).__init__(examples, fields)

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

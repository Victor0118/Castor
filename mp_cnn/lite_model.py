import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mp_cnn.model import MPCNN


class MPCNNLite(MPCNN):

    def __init__(self, n_word_dim, n_holistic_filters, n_per_dim_filters, filter_widths, hidden_layer_units, num_classes, dropout, ext_feats, attention, wide_conv):
        super(MPCNNLite, self).__init__(n_word_dim, n_holistic_filters, n_per_dim_filters, filter_widths, hidden_layer_units, num_classes, dropout, ext_feats, attention, wide_conv)
        self.arch = 'mpcnn_lite'

    def _add_layers(self):
        holistic_conv_layers_max = []

        for ws in self.filter_widths:
            if np.isinf(ws):
                continue

            padding = ws - 1 if self.wide_conv else 0

            holistic_conv_layers_max.append(nn.Sequential(
                nn.Conv1d(self.in_channels, self.n_holistic_filters, ws, padding=padding),
                nn.Tanh()
            ))

        self.holistic_conv_layers_max = nn.ModuleList(holistic_conv_layers_max)

    def _get_n_feats(self):
        COMP_1_COMPONENTS_HOLISTIC, COMP_2_COMPONENTS = 2 + self.n_holistic_filters, 2
        n_feats_h = self.n_holistic_filters * COMP_2_COMPONENTS
        n_feats_v = (
            # comparison units from holistic conv for max pooling for non-infinite widths
            ((len(self.filter_widths) - 1) ** 2) * COMP_1_COMPONENTS_HOLISTIC +
            # comparison units from holistic conv for max pooling for infinite widths
            3
        )
        n_feats = (n_feats_h + n_feats_v) * 3 + self.ext_feats
        return n_feats

    def _get_blocks_for_sentence(self, sent):
        block_a = {}
        for ws in self.filter_widths:
            if np.isinf(ws):
                sent_flattened, sent_flattened_size = sent.contiguous().view(sent.size(0), 1, -1), sent.size(1) * sent.size(2)
                block_a[ws] = {
                    'max': F.max_pool1d(sent_flattened, sent_flattened_size).view(sent.size(0), -1)
                }
                continue

            holistic_conv_out_max = self.holistic_conv_layers_max[ws - 1](sent)
            block_a[ws] = {
                'max': F.max_pool1d(holistic_conv_out_max, holistic_conv_out_max.size(2)).contiguous().view(-1, self.n_holistic_filters)
            }

        return block_a

    def _algo_1_horiz_comp(self, sent1_block_a, sent2_block_a):
        comparison_feats = []
        regM1, regM2 = [], []
        for ws in self.filter_widths:
            x1 = sent1_block_a[ws]['max'].unsqueeze(2)
            x2 = sent2_block_a[ws]['max'].unsqueeze(2)
            if np.isinf(ws):
                x1 = x1.expand(-1, self.n_holistic_filters, -1)
                x2 = x2.expand(-1, self.n_holistic_filters, -1)
            regM1.append(x1)
            regM2.append(x2)

        regM1 = torch.cat(regM1, dim=2)
        regM2 = torch.cat(regM2, dim=2)

        # Cosine similarity
        comparison_feats.append(F.cosine_similarity(regM1, regM2, dim=2))
        # Euclidean distance
        pairwise_distances = []
        for x1, x2 in zip(regM1, regM2):
            dist = F.pairwise_distance(x1, x2).view(1, -1)
            pairwise_distances.append(dist)
        comparison_feats.append(torch.cat(pairwise_distances))

        return torch.cat(comparison_feats, dim=1)

    def _algo_2_vert_comp(self, sent1_block_a, sent2_block_a):
        comparison_feats = []
        ws_no_inf = [w for w in self.filter_widths if not np.isinf(w)]
        for ws1 in self.filter_widths:
            x1 = sent1_block_a[ws1]['max']
            for ws2 in self.filter_widths:
                x2 = sent2_block_a[ws2]['max']
                if (not np.isinf(ws1) and not np.isinf(ws2)) or (np.isinf(ws1) and np.isinf(ws2)):
                    comparison_feats.append(F.cosine_similarity(x1, x2).unsqueeze(1))
                    comparison_feats.append(F.pairwise_distance(x1, x2).unsqueeze(1))
                    comparison_feats.append(torch.abs(x1 - x2))

        return torch.cat(comparison_feats, dim=1)

    def forward(self, sent1, query1, query2, query3, sent2, ext_feats=None, word_to_doc_count=None, raw_sent1=None, raw_sent2=None):
        # Attention
        if self.attention != 'none':
            sent1, sent2 = self.concat_attention(sent1, sent2, word_to_doc_count, raw_sent1, raw_sent2)
            query1, _ = self.concat_attention(query1, sent2, word_to_doc_count, raw_sent1, raw_sent2)
            query2, _ = self.concat_attention(query2, sent2, word_to_doc_count, raw_sent1, raw_sent2)
            query3, _ = self.concat_attention(query3, sent2, word_to_doc_count, raw_sent1, raw_sent2)

        # Sentence modeling module
        sent1_block_a = self._get_blocks_for_sentence(sent1)
        query1 = self._get_blocks_for_sentence(query1)
        query2 = self._get_blocks_for_sentence(query2)
        query3 = self._get_blocks_for_sentence(query3)
        sent2_block_a = self._get_blocks_for_sentence(sent2)

        # Similarity measurement layer
        feat_h1 = self._algo_1_horiz_comp(query1, sent2_block_a)
        feat_h2 = self._algo_1_horiz_comp(query2, sent2_block_a)
        feat_h3 = self._algo_1_horiz_comp(query3, sent2_block_a)
        feat_v1 = self._algo_2_vert_comp(query1, sent2_block_a)
        feat_v2 = self._algo_2_vert_comp(query2, sent2_block_a)
        feat_v3 = self._algo_2_vert_comp(query3, sent2_block_a)
        # print("feat_h1.size(): {}".format(feat_h1.size()))
        # print("feat_v1.size(): {}".format(feat_v1.size()))
        # print("feat_h2.size(): {}".format(feat_h2.size()))
        # print("feat_v2.size(): {}".format(feat_v2.size()))
        # print("feat_h3.size(): {}".format(feat_h3.size()))
        # print("feat_v3.size(): {}".format(feat_v3.size()))

        combined_feats = [feat_h1, feat_h2, feat_h3, feat_v1, feat_v1, feat_v2, feat_v3,
                          ext_feats] if self.ext_feats else [feat_h1, feat_h2, feat_h3, feat_v1, feat_v2, feat_v3]

        feat_all = torch.cat(combined_feats, dim=1)
        # print("feat_all.size(): {}".format(feat_all.size()))

        preds = self.final_layers(feat_all)
        return preds

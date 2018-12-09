import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import hashlib
import numpy as np
import itertools

def hard_pad2d(x, pad):
    def pad_side(idx):
        pad_len = max(pad - x.size(idx), 0)
        return [0, pad_len]
    padding = pad_side(3)
    padding.extend(pad_side(2))
    x = F.pad(x, padding)
    return x[:, :, :pad, :pad]

class ResNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        n_layers = config['res_layers']
        n_maps = config['res_fmaps']
        n_labels = config['n_labels']
        self.conv0 = nn.Conv2d(12, n_maps, (3, 3), padding=1)
        self.convs = nn.ModuleList([nn.Conv2d(n_maps, n_maps, (3, 3), padding=1) for _ in range(n_layers)])
        self.output = nn.Linear(n_maps, n_labels)
        self.input_len = None

    def forward(self, x):
        x = F.relu(self.conv0(x))
        old_x = x
        for i, conv in enumerate(self.convs):
            x = F.relu(conv(x))
            if i % 2 == 1:
                x += old_x
                old_x = x
        x = torch.mean(x.view(x.size(0), x.size(1), -1), 2)
        return F.log_softmax(self.output(x), 1)

class VDPWIConvNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        def make_conv(n_in, n_out):
            conv = nn.Conv2d(n_in, n_out, 3, padding=1)
            conv.bias.data.zero_()
            nn.init.xavier_normal_(conv.weight)
            return conv
        self.conv1 = make_conv(15, 128)
        self.conv2 = make_conv(128, 164)
        self.conv3 = make_conv(164, 192)
        self.conv4 = make_conv(192, 192)
        self.conv5 = make_conv(192, 128)
        self.maxpool2 = nn.MaxPool2d(2, ceil_mode=True)
        self.dnn = nn.Linear(128, 128)
        self.output = nn.Linear(128, config['n_labels'])
        self.input_len = 32

    def forward(self, x):
        x = hard_pad2d(x, self.input_len)
        pool_final = nn.MaxPool2d(2, ceil_mode=True) if x.size(2) == 32 else nn.MaxPool2d(3, 1, ceil_mode=True)
        x = self.maxpool2(F.relu(self.conv1(x)))
        x = self.maxpool2(F.relu(self.conv2(x)))
        x = self.maxpool2(F.relu(self.conv3(x)))
        x = self.maxpool2(F.relu(self.conv4(x)))
        x = pool_final(F.relu(self.conv5(x)))
        x = F.relu(self.dnn(x.view(x.size(0), -1)))
        return F.log_softmax(self.output(x), 1)


class ChildSumTreeLSTM(nn.Module):
    def __init__(self, in_dim, mem_dim):
        super(ChildSumTreeLSTM, self).__init__()
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.ioux = nn.Linear(self.in_dim, 3 * self.mem_dim)
        self.iouh = nn.Linear(self.mem_dim, 3 * self.mem_dim)
        self.fx = nn.Linear(self.in_dim, self.mem_dim)
        self.fh = nn.Linear(self.mem_dim, self.mem_dim)
        self.hidden_unit = []
        self.idx_track = []

    def node_forward(self, inputs, child_c, child_h):
        child_h_sum = torch.sum(child_h, dim=0, keepdim=True)

        iou = self.ioux(inputs) + self.iouh(child_h_sum)
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i, o, u = F.sigmoid(i), F.sigmoid(o), F.tanh(u)

        f = F.sigmoid(
            self.fh(child_h) +
            self.fx(inputs).repeat(len(child_h), 1)
        )
        fc = torch.mul(f, child_c)

        c = torch.mul(i, u) + torch.sum(fc, dim=0, keepdim=True)
        h = torch.mul(o, F.tanh(c))
        return c, h

    def forward(self, tree, inputs, raw_text):
        for idx in range(tree.num_children):
            self.forward(tree.children[idx], inputs, raw_text)

        if tree.num_children == 0:
            child_c = inputs[0].detach().new(1, self.mem_dim).fill_(0.).requires_grad_()
            child_h = inputs[0].detach().new(1, self.mem_dim).fill_(0.).requires_grad_()
        else:
            child_c, child_h = zip(* map(lambda x: x.state, tree.children))
            child_c, child_h = torch.cat(child_c, dim=0), torch.cat(child_h, dim=0)
        tree.state = self.node_forward(inputs[tree.idx], child_c, child_h)

        self.hidden_unit.append(tree.state[1][0])
        self.idx_track.append(tree.idx)
        #return tree.state
        return self.hidden_unit, self.idx_track

class VDPWIModel(nn.Module):
    def __init__(self, dim, config, tree_file=None, tree_parent_file=None):
        super().__init__()
        self.arch = 'vdpwi'
        self.hidden_dim = config['rnn_hidden_dim']
        self.rnn = nn.LSTM(dim, self.hidden_dim, 1, batch_first=True)
        self.device = config['device']
        self.tree_file = tree_file
        self.tree_parent_file = tree_parent_file
        self.treeLSTM = ChildSumTreeLSTM(300, config['rnn_hidden_dim'])
        self.treeLSTM_scale = config['treeLSTMScale']

        if config['classifier'] == 'vdpwi':
            self.classifier_net = VDPWIConvNet(config)
        elif config['classifier'] == 'resnet':
            self.classifier_net = ResNet(config)

    def create_pad_cube(self, sent1, sent2):
        pad_cube = []
        sent1_lengths = [len(s.split()) for s in sent1]
        sent2_lengths = [len(s.split()) for s in sent2]
        max_len1 = max(sent1_lengths)
        max_len2 = max(sent2_lengths)

        for s1_length, s2_length in zip(sent1_lengths, sent2_lengths):
            pad_mask = np.ones((max_len1, max_len2))
            pad_mask[:s1_length, :s2_length] = 0
            pad_cube.append(pad_mask)

        pad_cube = np.array(pad_cube)
        return torch.from_numpy(pad_cube).float().to(self.device).unsqueeze(0)

    def compute_sim_cube(self, seq1, seq2):
        def compute_sim(prism1, prism2):
            prism1_len = prism1.norm(dim=3)
            prism2_len = prism2.norm(dim=3)

            dot_prod = torch.matmul(prism1.unsqueeze(3), prism2.unsqueeze(4))
            dot_prod = dot_prod.squeeze(3).squeeze(3)
            cos_dist = dot_prod / (prism1_len * prism2_len + 1E-8)
            l2_dist = ((prism1 - prism2).norm(dim=3))
            return torch.stack([dot_prod, cos_dist, l2_dist], 1)

        def compute_prism(seq1, seq2):
            prism1 = seq1.repeat(seq2.size(1), 1, 1, 1)
            prism2 = seq2.repeat(seq1.size(1), 1, 1, 1)
            prism1 = prism1.permute(1, 2, 0, 3).contiguous()
            prism2 = prism2.permute(1, 0, 2, 3).contiguous()
            return compute_sim(prism1, prism2)

        sim_cube = torch.Tensor(seq1.size(0), 12, seq1.size(1), seq2.size(1))
        sim_cube = sim_cube.to(self.device)
        seq1_f = seq1[:, :, :self.hidden_dim]
        seq1_b = seq1[:, :, self.hidden_dim:]
        seq2_f = seq2[:, :, :self.hidden_dim]
        seq2_b = seq2[:, :, self.hidden_dim:]
        sim_cube[:, 0:3] = compute_prism(seq1, seq2)
        sim_cube[:, 3:6] = compute_prism(seq1_f, seq2_f)
        sim_cube[:, 6:9] = compute_prism(seq1_b, seq2_b)

        sim_cube[:, 9:12] = compute_prism(seq1_f + seq1_b, seq2_f + seq2_b)
        return sim_cube

    def compute_tree_sim_cube(self, seq1, seq2):
        def compute_sim(prism1, prism2):
            prism1_len = prism1.norm(dim=3)
            prism2_len = prism2.norm(dim=3)

            dot_prod = torch.matmul(prism1.unsqueeze(3), prism2.unsqueeze(4))
            dot_prod = dot_prod.squeeze(3).squeeze(3)
            cos_dist = dot_prod / (prism1_len * prism2_len + 1E-8)
            l2_dist = ((prism1 - prism2).norm(dim=3))
            return torch.stack([dot_prod, cos_dist, l2_dist], 1)

        def compute_prism(seq1, seq2):
            prism1 = seq1.repeat(seq2.size(1), 1, 1, 1)
            prism2 = seq2.repeat(seq1.size(1), 1, 1, 1)
            prism1 = prism1.permute(1, 2, 0, 3).contiguous()
            prism2 = prism2.permute(1, 0, 2, 3).contiguous()
            return compute_sim(prism1, prism2)

        sim_cube = torch.Tensor(seq1.size(0), 3, seq1.size(1), seq2.size(1))
        sim_cube = sim_cube.to(self.device)
        sim_cube[:, 0:3] = compute_prism(seq1, seq2)
        return sim_cube



    def compute_focus_cube(self, sim_cube, pad_cube, tree_mask):
        neg_magic = -10000
        pad_cube = pad_cube.repeat(12, 1, 1, 1)
        pad_cube = pad_cube.permute(1, 0, 2, 3).contiguous()
        sim_cube = neg_magic * pad_cube + sim_cube
        mask = torch.Tensor(*sim_cube.size()).to(self.device)
        mask[:, :, :, :] = 0.1

        def build_mask(index):
            max_mask = sim_cube[:, index].clone()
            for _ in range(min(sim_cube.size(2), sim_cube.size(3))):
                # values, indices: max for each one in a batch
                values, indices = torch.max(max_mask.view(sim_cube.size(0), -1), 1)
                # sim_cube.size(3): second sentence length
                row_indices = indices / sim_cube.size(3)
                col_indices = indices % sim_cube.size(3)
                row_indices = row_indices.unsqueeze(1)
                col_indices = col_indices.unsqueeze(1).unsqueeze(1)
                for i, (row_i, col_i, val) in enumerate(zip(row_indices, col_indices, values)):
                    if val < neg_magic / 2:
                        continue
                    mask[i, :, row_i, col_i] = 1
                    max_mask[i, row_i, :] = neg_magic
                    max_mask[i, :, col_i] = neg_magic
        build_mask(9)
        build_mask(10)
        focus_cube = (mask + tree_mask) * sim_cube * (1 - pad_cube)
        return focus_cube

    def findAllIndex(self, input_list, target):
        return list(np.where(np.array(input_list) == target)[0])

    def compute_tree_focus_cube(self, sim_cube, pad_cube, l_parent_list, r_parent_list):
        neg_magic = -10000
        pad_cube = pad_cube.repeat(3, 1, 1, 1)
        pad_cube = pad_cube.permute(1, 0, 2, 3).contiguous()
        sim_cube = neg_magic * pad_cube + sim_cube
        mask = torch.Tensor(*sim_cube.size()).to(self.device)
        mask[:, :, :, :] = 0.1

        l_parent_numnode_max = max([len(list(set(ele))) for ele in l_parent_list])
        r_parent_numnode_max = max([len(list(set(ele))) for ele in r_parent_list])

        def build_mask(index):
            max_mask = sim_cube[:, index].clone()
            for _ in range(min(l_parent_numnode_max, r_parent_numnode_max)):
                # values, indices: max for each one in a batch
                values, indices = torch.max(max_mask.view(sim_cube.size(0), -1), 1)
                # sim_cube.size(3): second sentence length
                row_indices = indices / sim_cube.size(3)
                col_indices = indices % sim_cube.size(3)
                row_indices = row_indices.unsqueeze(1)
                col_indices = col_indices.unsqueeze(1).unsqueeze(1)
                for i, (row_i, col_i, val) in enumerate(zip(row_indices, col_indices, values)):
                    if val < neg_magic / 2:
                        continue
                    rows = self.findAllIndex(l_parent_list[i], row_i.cpu().numpy())
                    cols = self.findAllIndex(r_parent_list[i], col_i.cpu().numpy())
                    for r, c in itertools.product(rows, cols):
                        mask[i, :, r, c] = 1
                        max_mask[i, r, :] = neg_magic
                        max_mask[i, :, c] = neg_magic
        build_mask(1)
        build_mask(2)
        return mask
        # focus_cube = mask * sim_cube * (1 - pad_cube)
        # return focus_cube

    def forward(self, sent1, sent2, ext_feats=None, word_to_doc_count=None, raw_sent1=None, raw_sent2=None):
        # The computation of word-to-word matrix part
        # Use BiLSTM

        pad_cube = self.create_pad_cube(raw_sent1, raw_sent2)
        # previous sent1: [batch_size, embed_dim, seq_len]
        sent1 = sent1.permute(0, 2, 1).contiguous() # [batch_size, seq_len, embed_dim]
        sent2 = sent2.permute(0, 2, 1).contiguous()
        seq1f, _ = self.rnn(sent1)
        seq2f, _ = self.rnn(sent2)
        seq1b, _ = self.rnn(torch.cat(sent1.split(1, 1)[::-1], 1))
        seq2b, _ = self.rnn(torch.cat(sent2.split(1, 1)[::-1], 1))
        seq1 = torch.cat([seq1f, seq1b], 2)  # [batch_size, seq_len, rnn_hd * 2]
        seq2 = torch.cat([seq2f, seq2b], 2)

        sim_cube = self.compute_sim_cube(seq1, seq2)

        truncate = self.classifier_net.input_len
        sim_cube = sim_cube[:, :, :pad_cube.size(2), :pad_cube.size(3)].contiguous()
        if truncate is not None:
            sim_cube = sim_cube[:, :, :truncate, :truncate].contiguous()
            pad_cube = pad_cube[:, :, :sim_cube.size(2), :sim_cube.size(3)].contiguous()
        # focus_cube = self.compute_focus_cube(sim_cube, pad_cube)

        # BiLSTM to form tree matrix
        seq1_treesim = seq1.clone()
        seq2_treesim = seq2.clone()
        l_parent_list = []
        r_parent_list = []
        for i in range(len(raw_sent1)):
            _sent1 = raw_sent1[i]
            _sent2 = raw_sent2[i]
            l_parent = self.tree_parent_file[hashlib.sha224(_sent1.encode()).hexdigest()]
            r_parent = self.tree_parent_file[hashlib.sha224(_sent2.encode()).hexdigest()]
            l_parent_list.append(l_parent)
            r_parent_list.append(r_parent)

            for k in range(len(l_parent)):
                target = l_parent[k]
                all_others = self.findAllIndex(l_parent, target)
                all_others.remove(k)
                if all_others != []:
                    for ele in all_others:
                        seq1_treesim[i, k, :] += seq1_treesim[i, k, :]
                    seq1_treesim[i, k, :] = seq1_treesim[i, k, :] / (len(all_others) + 1)

            for k in range(len(r_parent)):
                target = r_parent[k]
                all_others = self.findAllIndex(r_parent, target)
                all_others.remove(k)
                if all_others != []:
                    for ele in all_others:
                        seq2_treesim[i, k, :] += seq2_treesim[i, k, :]
                    seq2_treesim[i, k, :] = seq2_treesim[i, k, :] / (len(all_others) + 1)

        sim_cube_tree = self.compute_tree_sim_cube(seq1_treesim, seq2_treesim)
        truncate = self.classifier_net.input_len
        sim_cube_tree = sim_cube_tree[:, :, :pad_cube.size(2), :pad_cube.size(3)].contiguous()
        if truncate is not None:
            sim_cube_tree = sim_cube_tree[:, :, :truncate, :truncate].contiguous()
        tree_mask = self.compute_tree_focus_cube(sim_cube_tree, pad_cube, l_parent_list, r_parent_list)
        focus_cube = self.compute_focus_cube(sim_cube, pad_cube, tree_mask)

        log_prob = self.classifier_net(focus_cube)
        return log_prob

import torch
import torch.nn.functional as F

from .evaluator import Evaluator

class PIT2015Evaluator(Evaluator):

    def get_scores(self):
        self.model.eval()
        self.data_loader.init_epoch()
        n_dev_correct = 0
        total_loss = 0
        acc_total = 0
        rel_total = 0
        pre_total = 0
        off = 1e-8
        for batch_idx, batch in enumerate(self.data_loader):
            sent1, sent2 = self.get_sentence_embeddings(batch)
            scores = self.model(sent1, sent2, batch.ext_feats, batch.dataset.word_to_doc_cnt, batch.sentence_1_raw, batch.sentence_2_raw)
            prediction = torch.max(scores, 1)[1].view(batch.label.size()).data
            gold_label = batch.label.data
            n_dev_correct += (prediction == gold_label).sum().item()
            acc_total += ((prediction == batch.label.data) * (prediction == 1)).sum().item()
            total_loss += F.nll_loss(scores, batch.label, size_average=False).item()
            rel_total += gold_label.sum().item()
            pre_total += prediction.sum().item()

            del scores

        precision = acc_total / (pre_total + off)
        recall = acc_total / (rel_total + off)
        f1 = 2 * precision * recall / (precision + recall + off)
        accuracy = 100. * n_dev_correct / len(self.data_loader.dataset.examples)
        avg_loss = total_loss / len(self.data_loader.dataset.examples)
        return [accuracy, avg_loss, precision, recall, f1], ['accuracy', 'cross_entropy_loss', 'precision', 'recall', 'f1']

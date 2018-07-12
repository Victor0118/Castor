import torch.nn.functional as F

from .evaluator import Evaluator
from utils.relevancy_metrics import get_map_mrr


class QAEvaluator(Evaluator):

    def get_scores(self):
        self.model.eval()
        test_cross_entropy_loss = 0
        qids = []
        docnos = []
        true_labels = []
        predictions = []
        # index2qid = np.array(self.data_loader.ID_FIELD.vocab.itos)
        # index2aid = np.array(self.data_loader.AID_FIELD.vocab.itos)

        for batch in self.data_loader:
            qids.extend(self.index2qid[batch.id.detach().cpu().numpy()])
            docnos.extend(self.index2aid[batch.aid.detach().cpu().numpy()])
            # Select embedding
            sent1, query1, query2, query3, sent2 = self.get_sentence_embeddings(batch)

            output = self.model(sent1, query1, query2, query3, sent2, batch.ext_feats, batch.dataset.word_to_doc_cnt, batch.sentence_1_raw, batch.sentence_2_raw)
            test_cross_entropy_loss += F.cross_entropy(output, batch.label, size_average=False).item()

            true_labels.extend(batch.label.detach().cpu().numpy())
            predictions.extend(output.detach().exp()[:, 1].cpu().numpy())

            del output

        # qids = list(map(lambda n: int(round(n * 10, 0)) / 10, qids))

        mean_average_precision, mean_reciprocal_rank = get_map_mrr(qids, predictions, true_labels,
                                                                   self.data_loader.device, docnos,
                                                                   keep_results=self.keep_results)
        test_cross_entropy_loss /= len(batch.dataset.examples)

        return [mean_average_precision, mean_reciprocal_rank, test_cross_entropy_loss], ['map', 'mrr', 'cross entropy loss']

    def get_final_prediction_and_label(self, batch_predictions, batch_labels):
        predictions = batch_predictions.exp()[:, 1]

        return predictions, batch_labels

class Evaluator(object):
    """
    Evaluates a model on a Dataset, using metrics specific to the Dataset.
    """

    def __init__(self, dataset_cls, model, embedding, data_loader, batch_size, device, keep_results=False, index2qid=None, index2aid=None):
        self.dataset_cls = dataset_cls
        self.model = model
        self.embedding = embedding
        self.data_loader = data_loader
        self.batch_size = batch_size
        self.device = device
        self.keep_results = keep_results
        assert index2qid is not None
        assert index2aid is not None
        self.index2qid = index2qid
        self.index2aid = index2aid

    def get_sentence_embeddings(self, batch):
        sent1 = self.embedding(batch.sentence_1).transpose(1, 2)
        sent2 = self.embedding(batch.sentence_2).transpose(1, 2)
        query1 = self.embedding(batch.query1).transpose(1, 2)
        query2 = self.embedding(batch.query2).transpose(1, 2)
        query3 = self.embedding(batch.query3).transpose(1, 2)
        return sent1, query1, query2, query3, sent2

    def get_scores(self):
        """
        Get the scores used to evaluate the model.
        Should return ([score1, score2, ..], [score1_name, score2_name, ...]).
        The first score is the primary score used to determine if the model has improved.
        """
        raise NotImplementedError('Evaluator subclass needs to implement get_score')

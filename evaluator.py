from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
from datetime import datetime


class Evaluator:

    def __init__(self, predictor, save_filepath):

        self.predictor = predictor
        self.save_filepath = save_filepath

    def evaluate_dataset(self, test_dataset):
        s = datetime.now()
        tokenize = lambda x: x.split()

        predictions = [self.predictor.predict_one(source, num_candidates=1)[0] for source, target in test_dataset]

        hypotheses = [tokenize(prediction) for prediction in predictions]
        list_of_references = [[tokenize(target)] for source, target in test_dataset]

        with open(self.save_filepath, 'w') as file:
            for (source, target), prediction, hypothesis, references in zip(test_dataset, predictions, hypotheses, list_of_references):
                sentence_bleu_score = sentence_bleu(references, hypothesis)
                line = "{bleu_score}\t{source}\t{target}\t|\t{prediction}".format(
                    bleu_score=sentence_bleu_score,
                    source=source,
                    target=target,
                    prediction=prediction
                )
                file.write(line + '\n')

        bleu_score = corpus_bleu(list_of_references, hypotheses)

        print('Took', datetime.now() - s)
        return bleu_score
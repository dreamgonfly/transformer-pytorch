from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
from tqdm import tqdm


class Evaluator:

    def __init__(self, predictor, save_filepath):

        self.predictor = predictor
        self.save_filepath = save_filepath

    def evaluate_dataset(self, test_dataset):
        tokenize = lambda x: x.split()

        predictions = []
        for source, target in tqdm(test_dataset):
            prediction = self.predictor.predict_one(source, num_candidates=1)[0]
            predictions.append(prediction)

        hypotheses = [tokenize(prediction) for prediction in predictions]
        list_of_references = [[tokenize(target)] for source, target in test_dataset]
        smoothing_function = SmoothingFunction()

        with open(self.save_filepath, 'w') as file:
            for (source, target), prediction, hypothesis, references in zip(test_dataset, predictions,
                                                                            hypotheses, list_of_references):
                sentence_bleu_score = sentence_bleu(references, hypothesis,
                                                    smoothing_function=smoothing_function.method3)
                line = "{bleu_score}\t{source}\t{target}\t|\t{prediction}".format(
                    bleu_score=sentence_bleu_score,
                    source=source,
                    target=target,
                    prediction=prediction
                )
                file.write(line + '\n')

        bleu_score = corpus_bleu(list_of_references, hypotheses, smoothing_function=smoothing_function.method3)

        return bleu_score

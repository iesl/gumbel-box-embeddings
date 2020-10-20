from allennlp.predictors import Predictor


@Predictor.register('kb-completion-predictor')
class KBCompletionPredictor(Predictor):
    pass

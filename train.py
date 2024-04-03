# ## Define Model running (training/fit and testing/evaluate)
import os
import random
import math
# Third-party library imports
import matplotlib.pyplot as plt
from util.timer import Timer
from util.metrics import accuracy
from util.metrics import mse
from Neural_Networks_Code import *
from data_processing import *


def get_name(obj):
    try:
        if hasattr(obj, '__name__'):
            return obj.__name__
        else:
            return obj
    except Exception as e:
        return obj


def catchthrow(e, err):
    trace = traceback.format_exc()
    print(err + f"\n{trace}")
    raise e


class RunModel():
    t1 = '\t'
    t2 = '\t\t'
    t3 = '\t\t\t'

    def __init__(self, model, model_params):
        self.model_name = model.__name__
        self.model_params = model_params
        self.model = self.build_model(model, model_params)

    def build_model(self, model, model_params):
        print("="*50)
        print(f"Building model {self.model_name}")

        try:
            model = model(**model_params)
        except Exception as e:
            err = f"Exception caught while building model for {self.model_name}:"
            catchthrow(e, err)
        return model

    def fit(self, *args, **kwargs):
        print(f"Training {self.model_name}...")
        print(f"{self.t1}Using hyperparameters: ")
        [print(f"{self.t2}{n} = {get_name(v)}")
         for n, v in self.model_params.items()]
        try:
            scores = self._fit(*args, **kwargs)
            return scores
        except Exception as e:
            err = f"Exception caught while training model for {self.model_name}:"
            catchthrow(e, err)

    def _fit(self, X, y, X_vld, y_vld, metrics=None, pass_y=True):
        if pass_y:
            self.model.fit(X, y, X_vld, y_vld)
        else:
            self.model.fit(X)
        preds = self.model.predict(X)
        scores = self.get_metrics(y, preds, metrics, prefix='Train')
        return scores

    def evaluate(self, *args, **kwargs):
        print(f"Evaluating {self.model_name}...")
        try:
            return self._evaluate(*args, **kwargs)
        except Exception as e:
            err = f"Exception caught while evaluating model for {self.model_name}:"
            catchthrow(e, err)

    def _evaluate(self, X, y, metrics, prefix=''):
        preds = self.model.predict(X)
        scores = self.get_metrics(y, preds, metrics, prefix)
        return scores

    def predict(self, X):
        try:
            preds = self.model.predict(X)
        except Exception as e:
            err = f"Exception caught while making predictions for model {self.model_name}:"
            catchthrow(e, err)

        return preds

    def get_metrics(self, y, y_hat, metrics, prefix=''):
        scores = {}
        for name, metric in metrics.items():
            score = metric(y, y_hat)
            display_score = round(score, 3)
            scores[name] = score
            print(f"{self.t2}{prefix} {name}: {display_score}")
        return scores


def run_eval(eval_stage='validation'):
    main_timer = Timer()
    main_timer.start()

    task_info = [
        dict(
            model=NeuralNetworkRegressor,
            name='NeuralNetworkRegressor',
            data=HousingDataPreparation,
            data_prep=dict(return_array=True),
            metrics=dict(mse=mse),
            eval_metric='mse',
            trn_score=9999,
            eval_score=9999,
            successful=False,
        ),
        dict(
            model=NeuralNetworkClassifier,
            name='NeuralNetworkClassifier',
            data=MNISTDataPreparation,
            data_prep=dict(return_array=True),
            metrics=dict(acc=accuracy),
            eval_metric='acc',
            trn_score=0,
            eval_score=0,
            successful=False,
        ),
    ]

    for info in task_info:
        task_timer = Timer()
        task_timer.start()
        try:
            params = HyperParametersAndTransforms.get_params(info['name'])
            model_kwargs = params.get('model_kwargs', {})
            data_prep_kwargs = params.get('data_prep_kwargs', {})

            run_model = RunModel(info['model'], model_kwargs)
            data = info['data'](**data_prep_kwargs)
            X_trn, y_trn, X_vld, y_vld = data.data_prep(**info['data_prep'])

            trn_scores = run_model.fit(
                X_trn, y_trn, X_vld, y_vld, info['metrics'], pass_y=True)
            eval_scores = run_model.evaluate(
                X_vld, y_vld, info['metrics'], prefix=eval_stage.capitalize())

            if not math.isnan(trn_scores[info['eval_metric']]):
                info['trn_score'] = trn_scores[info['eval_metric']]
            if not math.isnan(eval_scores[info['eval_metric']]):
                info['eval_score'] = eval_scores[info['eval_metric']]

            info['successful'] = True

        except Exception as e:
            track = traceback.format_exc()
            print(
                "The following exception occurred while executing this test case:\n", track)
        task_timer.stop()

        print("")

    print("="*50)
    print('')
    main_timer.stop()

    final_mse, final_acc = get_eval_scores(task_info)
    print(f"Final {eval_stage.capitalize()} MSE: {final_mse}")
    print(f"Final {eval_stage.capitalize()} Accuracy: {final_acc}")

    return main_timer.last_elapsed_time, final_mse, final_acc


def get_eval_scores(task_info):
    return [i['eval_score'] for i in task_info]


if __name__ == "__main__":
    run_eval()

import collections
import operator
import random
import os

from metrics_eval import PrequentialEvaluator

import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import pickle
import math
from typing import Optional, Tuple

from river import time_series
from river import metrics
from windowed_lof import WindowedLOF
from balanced_lazy_search import BalancedLazySearch

from pyflink.datastream.functions import MapFunction, RuntimeContext
from pyflink.common import Row
from river import (
    naive_bayes,
    preprocessing,
    tree,
    forest,
    linear_model,
    optim,
    neighbors,
    anomaly,
    feature_extraction
)

from river import anomaly


class GaussianNaiveBayes(MapFunction):
    def open(self, runtime_context: RuntimeContext):
        self.scaler = preprocessing.StandardScaler()
        self.model = naive_bayes.GaussianNB()
        self.model_save_num = 1000000
        self.counter = 1

    def _row_to_features(self, value):
        try:
            rec = value.as_dict()
        except Exception:
            rec = dict(value)

        features = {k: v for k, v in rec.items() if k not in ("Label", "Timestamp")}
        return features, rec.get("Label"), rec.get("Timestamp"), rec

    def _check_dict(self, features):
        for _, v in features.items():
            if not (isinstance(v, (int, float)) and math.isfinite(v)):
                return False
        return True

    def map(self, value):
        if self.counter % self.model_save_num == 0:
            with open("./backup/gnb.pkl", "wb") as f:
                pickle.dump(self.model, f)
            with open("./backup/gnb_scaler.pkl", "wb") as f:
                pickle.dump(self.scaler, f)
            self.counter = 0
        self.counter += 1
        features, label_orig, timestamp, orig_rec = self._row_to_features(value)
        if not self._check_dict(features):
            output = {
                "Timestamp": timestamp,
                "orig_label": label_orig,
                "binary_label": "Attack" if label_orig != "Benign" else "Benign",
                "predicted_label": "-1",
                "predicted_binary_label": "-1",
                "probability": -1,
                "class_probabilities": "-1",
            }
            return Row(**output)

        try:
            self.scaler.learn_one(features)
            x = self.scaler.transform_one(features)
        except Exception:
            x = features

        try:
            proba = self.model.predict_proba_one(x) or {}
            predicted_class = max(proba, key=proba.get)
            probability = proba.get(predicted_class, 0.0)
            predicted_binary_class = (
                "Attack" if predicted_class != "Benign" else "Benign"
            )
        except Exception as e:
            print(e)
            proba = {}
            predicted_class = "None"
            predicted_binary_class = "None"
            probability = -1

        try:
            self.model.learn_one(x, label_orig)
        except Exception as e:
            print(e)

        output = {
            "Timestamp": timestamp,
            "orig_label": label_orig,
            "binary_label": "Attack" if label_orig != "Benign" else "Benign",
            "predicted_label": predicted_class,
            "predicted_binary_label": predicted_binary_class,
            "probability": probability,
            "class_probabilities": str(proba),
        }
        return Row(**output)


class AdaptiveHoeffdingTree(MapFunction):
    def open(self, runtime_context: RuntimeContext):
        self.scaler = preprocessing.StandardScaler()
        self.model = tree.HoeffdingAdaptiveTreeClassifier()
        self.model_save_num = 1000000
        self.counter = 1

    def _row_to_features(self, value):
        try:
            rec = value.as_dict()
        except Exception:
            rec = dict(value)

        features = {k: v for k, v in rec.items() if k not in ("Label", "Timestamp")}
        return features, rec.get("Label"), rec.get("Timestamp"), rec

    def _check_dict(self, features):
        for _, v in features.items():
            if not (isinstance(v, (int, float)) and math.isfinite(v)):
                return False
        return True

    def map(self, value):
        if self.counter % self.model_save_num == 0:
            with open("./backup/aht.pkl", "wb") as f:
                pickle.dump(self.model, f)
            with open("./backup/aht_scaler.pkl", "wb") as f:
                pickle.dump(self.scaler, f)
            self.counter = 0
        self.counter += 1
        features, label_orig, timestamp, orig_rec = self._row_to_features(value)
        if not self._check_dict(features):
            output = {
                "Timestamp": timestamp,
                "orig_label": label_orig,
                "binary_label": "Attack" if label_orig != "Benign" else "Benign",
                "predicted_label": "-1",
                "predicted_binary_label": "-1",
                "probability": -1,
                "class_probabilities": "-1",
            }
            return Row(**output)

        try:
            self.scaler.learn_one(features)
            x = self.scaler.transform_one(features)
        except Exception:
            x = features

        try:
            proba = self.model.predict_proba_one(x) or {}
            predicted_class = max(proba, key=proba.get)
            probability = proba.get(predicted_class, 0.0)
            predicted_binary_class = (
                "Attack" if predicted_class != "Benign" else "Benign"
            )
        except Exception as e:
            print(e)
            proba = {}
            predicted_class = "None"
            predicted_binary_class = "None"
            probability = -1

        try:
            self.model.learn_one(x, label_orig)
        except Exception as e:
            print(e)

        output = {
            "Timestamp": timestamp,
            "orig_label": label_orig,
            "binary_label": "Attack" if label_orig != "Benign" else "Benign",
            "predicted_label": predicted_class,
            "predicted_binary_label": predicted_binary_class,
            "probability": probability,
            "class_probabilities": str(proba),
        }
        return Row(**output)


class AdaptiveRandomForest(MapFunction):
    def open(self, runtime_context: RuntimeContext):
        self.scaler = preprocessing.StandardScaler()
        self.model = forest.ARFClassifier()
        self.model_save_num = 1000000
        self.counter = 1

    def _row_to_features(self, value):
        try:
            rec = value.as_dict()
        except Exception:
            rec = dict(value)

        features = {k: v for k, v in rec.items() if k not in ("Label", "Timestamp")}
        return features, rec.get("Label"), rec.get("Timestamp"), rec

    def _check_dict(self, features):
        for _, v in features.items():
            if not (isinstance(v, (int, float)) and math.isfinite(v)):
                return False
        return True

    def map(self, value):
        if self.counter % self.model_save_num == 0:
            with open("./backup/arf.pkl", "wb") as f:
                pickle.dump(self.model, f)
            with open("./backup/arf_scaler.pkl", "wb") as f:
                pickle.dump(self.scaler, f)
            self.counter = 0
        self.counter += 1
        features, label_orig, timestamp, orig_rec = self._row_to_features(value)
        if not self._check_dict(features):
            output = {
                "Timestamp": timestamp,
                "orig_label": label_orig,
                "binary_label": "Attack" if label_orig != "Benign" else "Benign",
                "predicted_label": "-1",
                "predicted_binary_label": "-1",
                "probability": -1,
                "class_probabilities": "-1",
            }
            return Row(**output)

        try:
            self.scaler.learn_one(features)
            x = self.scaler.transform_one(features)
        except Exception:
            x = features

        try:
            proba = self.model.predict_proba_one(x) or {}
            predicted_class = max(proba, key=proba.get)
            probability = proba.get(predicted_class, 0.0)
            predicted_binary_class = (
                "Attack" if predicted_class != "Benign" else "Benign"
            )
        except Exception as e:
            print(e)
            proba = {}
            predicted_class = "None"
            predicted_binary_class = "None"
            probability = -1

        try:
            self.model.learn_one(x, label_orig)
        except Exception as e:
            print(e)

        output = {
            "Timestamp": timestamp,
            "orig_label": label_orig,
            "binary_label": "Attack" if label_orig != "Benign" else "Benign",
            "predicted_label": predicted_class,
            "predicted_binary_label": predicted_binary_class,
            "probability": probability,
            "class_probabilities": str(proba),
        }
        return Row(**output)


class AggregatedMondrianForest(MapFunction):
    def open(self, runtime_context: RuntimeContext):
        self.scaler = preprocessing.StandardScaler()
        self.model = forest.AMFClassifier()
        self.model_save_num = 1000000
        self.counter = 1

    def _row_to_features(self, value):
        try:
            rec = value.as_dict()
        except Exception:
            rec = dict(value)

        features = {k: v for k, v in rec.items() if k not in ("Label", "Timestamp")}
        return features, rec.get("Label"), rec.get("Timestamp"), rec

    def _check_dict(self, features):
        for _, v in features.items():
            if not (isinstance(v, (int, float)) and math.isfinite(v)):
                return False
        return True

    def map(self, value):
        if self.counter % self.model_save_num == 0:
            with open("./backup/amf.pkl", "wb") as f:
                pickle.dump(self.model, f)
            with open("./backup/amf_scaler.pkl", "wb") as f:
                pickle.dump(self.scaler, f)
            self.counter = 0
        self.counter += 1
        features, label_orig, timestamp, orig_rec = self._row_to_features(value)
        if not self._check_dict(features):
            output = {
                "Timestamp": timestamp,
                "orig_label": label_orig,
                "binary_label": "Attack" if label_orig != "Benign" else "Benign",
                "predicted_label": "-1",
                "predicted_binary_label": "-1",
                "probability": -1,
                "class_probabilities": "-1",
            }
            return Row(**output)

        try:
            self.scaler.learn_one(features)
            x = self.scaler.transform_one(features)
        except Exception:
            x = features

        try:
            proba = self.model.predict_proba_one(x) or {}
            predicted_class = max(proba, key=proba.get)
            probability = proba.get(predicted_class, 0.0)
            predicted_binary_class = (
                "Attack" if predicted_class != "Benign" else "Benign"
            )
        except Exception as e:
            print(e)
            proba = {}
            predicted_class = "None"
            predicted_binary_class = "None"
            probability = -1

        try:
            self.model.learn_one(x, label_orig)
        except Exception as e:
            print(e)

        output = {
            "Timestamp": timestamp,
            "orig_label": label_orig,
            "binary_label": "Attack" if label_orig != "Benign" else "Benign",
            "predicted_label": predicted_class,
            "predicted_binary_label": predicted_binary_class,
            "probability": probability,
            "class_probabilities": str(proba),
        }
        return Row(**output)


class SoftmaxRegression(MapFunction):
    def open(self, runtime_context: RuntimeContext):
        self.scaler = preprocessing.StandardScaler()
        self.model = linear_model.SoftmaxRegression(optimizer=optim.Adam(0.01), l2=0.1)
        self.model_save_num = 1000000
        self.counter = 1

    def _row_to_features(self, value):
        try:
            rec = value.as_dict()
        except Exception:
            rec = dict(value)

        features = {k: v for k, v in rec.items() if k not in ("Label", "Timestamp")}
        return features, rec.get("Label"), rec.get("Timestamp"), rec

    def _check_dict(self, features):
        for _, v in features.items():
            if not (isinstance(v, (int, float)) and math.isfinite(v)):
                return False
        return True

    def map(self, value):
        if self.counter % self.model_save_num == 0:
            with open("./backup/sr.pkl", "wb") as f:
                pickle.dump(self.model, f)
            with open("./backup/sr_scaler.pkl", "wb") as f:
                pickle.dump(self.scaler, f)
            self.counter = 0
        self.counter += 1
        features, label_orig, timestamp, orig_rec = self._row_to_features(value)
        if not self._check_dict(features):
            output = {
                "Timestamp": timestamp,
                "orig_label": label_orig,
                "binary_label": "Attack" if label_orig != "Benign" else "Benign",
                "predicted_label": "-1",
                "predicted_binary_label": "-1",
                "probability": -1,
                "class_probabilities": "-1",
            }
            return Row(**output)

        try:
            self.scaler.learn_one(features)
            x = self.scaler.transform_one(features)
        except Exception:
            x = features

        try:
            proba = self.model.predict_proba_one(x) or {}
            predicted_class = max(proba, key=proba.get)
            probability = proba.get(predicted_class, 0.0)
            predicted_binary_class = (
                "Attack" if predicted_class != "Benign" else "Benign"
            )
        except Exception as e:
            print(e)
            proba = {}
            predicted_class = "None"
            predicted_binary_class = "None"
            probability = -1

        try:
            self.model.learn_one(x, label_orig)
        except Exception as e:
            print(e)

        output = {
            "Timestamp": timestamp,
            "orig_label": label_orig,
            "binary_label": "Attack" if label_orig != "Benign" else "Benign",
            "predicted_label": predicted_class,
            "predicted_binary_label": predicted_binary_class,
            "probability": probability,
            "class_probabilities": str(proba),
        }
        return Row(**output)


class HoeffdingTree(MapFunction):
    def open(self, runtime_context: RuntimeContext):
        self.scaler = preprocessing.StandardScaler()
        self.model = tree.HoeffdingTreeClassifier()
        self.model_save_num = 1000000
        self.counter = 1

    def _row_to_features(self, value):
        try:
            rec = value.as_dict()
        except Exception:
            rec = dict(value)

        features = {k: v for k, v in rec.items() if k not in ("Label", "Timestamp")}
        return features, rec.get("Label"), rec.get("Timestamp"), rec

    def _check_dict(self, features):
        for _, v in features.items():
            if not (isinstance(v, (int, float)) and math.isfinite(v)):
                return False
        return True

    def map(self, value):
        if self.counter % self.model_save_num == 0:
            with open("./backup/ht.pkl", "wb") as f:
                pickle.dump(self.model, f)
            with open("./backup/ht_scaler.pkl", "wb") as f:
                pickle.dump(self.scaler, f)
            self.counter = 0
        self.counter += 1
        features, label_orig, timestamp, orig_rec = self._row_to_features(value)
        if not self._check_dict(features):
            output = {
                "Timestamp": timestamp,
                "orig_label": label_orig,
                "binary_label": "Attack" if label_orig != "Benign" else "Benign",
                "predicted_label": "-1",
                "predicted_binary_label": "-1",
                "probability": -1,
                "class_probabilities": "-1",
            }
            return Row(**output)

        try:
            self.scaler.learn_one(features)
            x = self.scaler.transform_one(features)
        except Exception:
            x = features

        try:
            proba = self.model.predict_proba_one(x) or {}
            predicted_class = max(proba, key=proba.get)
            probability = proba.get(predicted_class, 0.0)
            predicted_binary_class = (
                "Attack" if predicted_class != "Benign" else "Benign"
            )
        except Exception as e:
            print(e)
            proba = {}
            predicted_class = "None"
            predicted_binary_class = "None"
            probability = -1

        try:
            self.model.learn_one(x, label_orig)
        except Exception as e:
            print(e)

        output = {
            "Timestamp": timestamp,
            "orig_label": label_orig,
            "binary_label": "Attack" if label_orig != "Benign" else "Benign",
            "predicted_label": predicted_class,
            "predicted_binary_label": predicted_binary_class,
            "probability": probability,
            "class_probabilities": str(proba),
        }
        return Row(**output)


class HalfSpaceTrees(MapFunction):
    def open(self, runtime_context: RuntimeContext):
        self.scaler = preprocessing.MinMaxScaler()
        self.model = anomaly.HalfSpaceTrees()
        self.model_save_num = 1000000
        self.counter = 1

    def _row_to_features(self, value):
        try:
            rec = value.as_dict()
        except Exception:
            rec = dict(value)

        features = {k: v for k, v in rec.items() if k not in ("Label", "Timestamp")}
        return features, rec.get("Label"), rec.get("Timestamp"), rec

    def _check_dict(self, features):
        for _, v in features.items():
            if not (isinstance(v, (int, float)) and math.isfinite(v)):
                return False
        return True

    def map(self, value):
        if self.counter % self.model_save_num == 0:
            with open("./backup/hst.pkl", "wb") as f:
                pickle.dump(self.model, f)
            with open("./backup/hst_scaler.pkl", "wb") as f:
                pickle.dump(self.scaler, f)
            self.counter = 0
        self.counter += 1
        features, label_orig, timestamp, orig_rec = self._row_to_features(value)
        if not self._check_dict(features):
            output = {
                "Timestamp": timestamp,
                "orig_label": label_orig,
                "binary_label": "Attack" if label_orig != "Benign" else "Benign",
                "predicted_label": "-1",
                "predicted_binary_label": "-1",
                "probability": -1,
                "class_probabilities": "-1",
            }
            return Row(**output)

        try:
            self.scaler.learn_one(features)
            x = self.scaler.transform_one(features)
        except Exception:
            x = features

        try:
            anomaly_score = self.model.score_one(x)
        except Exception as e:
            print(e)
            anomaly_score = -1
        predicted_class = "Ignore"
        predicted_binary_class = "Ignore"
        proba = {}

        try:
            if label_orig == "Benign":
                self.model.learn_one(x)
        except Exception as e:
            print(e)

        output = {
            "Timestamp": timestamp,
            "orig_label": label_orig,
            "binary_label": "Attack" if label_orig != "Benign" else "Benign",
            "predicted_label": predicted_class,
            "predicted_binary_label": predicted_binary_class,
            "probability": anomaly_score,
            "class_probabilities": str(proba),
        }
        return Row(**output)

class LocalOutlierFactor(MapFunction):
    def open(self, runtime_context: RuntimeContext):
        self.scaler = preprocessing.StandardScaler()
        # self.model = WindowedLOF(
        #     n_neighbors=10,
        #     window_size=2000,
        #     rebuild_every=1000,
        #     min_dist=1e-9,
        # )
        self.model = anomaly.LocalOutlierFactor(n_neighbors=10, window_size=2000, min_dist=1e-9)
        self.model_save_num = 1000000
        self.counter = 1

    def _row_to_features(self, value):
        try:
            rec = value.as_dict()
        except Exception:
            rec = dict(value)

        features = {k: v for k, v in rec.items() if k not in ("Label", "Timestamp")}
        return features, rec.get("Label"), rec.get("Timestamp"), rec

    def _check_dict(self, features):
        for _, v in features.items():
            if not (isinstance(v, (int, float)) and math.isfinite(v)):
                return False
        return True

    def map(self, value):
        if self.counter % self.model_save_num == 0:
            os.makedirs("D:/backup", exist_ok=True)  #
            with open("D:/backup/lof.pkl", "wb") as f:  #
                pickle.dump(self.model, f)
            with open("D:/backup/lof_scaler.pkl", "wb") as f:  #
                pickle.dump(self.scaler, f)
            self.counter = 0
        self.counter += 1
        features, label_orig, timestamp, orig_rec = self._row_to_features(value)
        if not self._check_dict(features):
            output = {
                "Timestamp": timestamp,
                "orig_label": label_orig,
                "binary_label": "Attack" if label_orig != "Benign" else "Benign",
                "predicted_label": "-1",
                "predicted_binary_label": "-1",
                "probability": -1,
                "class_probabilities": "-1",
            }
            return Row(**output)

        try:
            self.scaler.learn_one(features)
            x = self.scaler.transform_one(features)
        except Exception:
            x = features

        try:
            anomaly_score = self.model.score_one(x)
        except Exception as e:
            print(e)
            anomaly_score = -1
        predicted_class = "Ignore"
        predicted_binary_class = "Ignore"
        proba = {}

        try:
            if label_orig == "Benign":
                self.model.learn_one(x)
        except Exception as e:
            print(e)

        output = {
            "Timestamp": timestamp,
            "orig_label": label_orig,
            "binary_label": "Attack" if label_orig != "Benign" else "Benign",
            "predicted_label": predicted_class,
            "predicted_binary_label": predicted_binary_class,
            "probability": anomaly_score,
            "class_probabilities": str(proba),
        }
        return Row(**output)


class OCSVM(MapFunction):
    def open(self, runtime_context: RuntimeContext):
        self.scaler = preprocessing.StandardScaler()
        self.model = anomaly.OneClassSVM(nu=0.001)
        self.model_save_num = 1000000
        self.counter = 1

    def _row_to_features(self, value):
        try:
            rec = value.as_dict()
        except Exception:
            rec = dict(value)

        features = {k: v for k, v in rec.items() if k not in ("Label", "Timestamp")}
        return features, rec.get("Label"), rec.get("Timestamp"), rec

    def _check_dict(self, features):
        for _, v in features.items():
            if not (isinstance(v, (int, float)) and math.isfinite(v)):
                return False
        return True

    def map(self, value):
        if self.counter % self.model_save_num == 0:
            with open("./backup/ocsvm_sgd_norbf.pkl", "wb") as f:
                pickle.dump(self.model, f)
            with open("./backup/ocsvm_sgd_norbd_scaler.pkl", "wb") as f:
                pickle.dump(self.scaler, f)
            self.counter = 0
        self.counter += 1
        features, label_orig, timestamp, orig_rec = self._row_to_features(value)
        if not self._check_dict(features):
            output = {
                "Timestamp": timestamp,
                "orig_label": label_orig,
                "binary_label": "Attack" if label_orig != "Benign" else "Benign",
                "predicted_label": "-1",
                "predicted_binary_label": "-1",
                "probability": -1,
                "class_probabilities": "-1",
            }
            return Row(**output)

        try:
            self.scaler.learn_one(features)
            x = self.scaler.transform_one(features)
        except Exception:
            x = features

        try:
            anomaly_score = self.model.score_one(x)
        except Exception as e:
            print(e)
            anomaly_score = -1
        predicted_class = "Ignore"
        predicted_binary_class = "Ignore"
        proba = {}

        try:
            if label_orig == "Benign":
                self.model.learn_one(x)
        except Exception as e:
            print(e)

        output = {
            "Timestamp": timestamp,
            "orig_label": label_orig,
            "binary_label": "Attack" if label_orig != "Benign" else "Benign",
            "predicted_label": predicted_class,
            "predicted_binary_label": predicted_binary_class,
            "probability": anomaly_score,
            "class_probabilities": str(proba),
        }
        return Row(**output)


class OCSVM_RBF(MapFunction):
    def open(self, runtime_context: RuntimeContext):
        self.scaler = preprocessing.StandardScaler()
        self.model = feature_extraction.RBFSampler() | anomaly.OneClassSVM(
            nu=0.001, optimizer=optim.Adam(0.01)
        )
        self.model_save_num = 1000000
        self.counter = 1

    def _row_to_features(self, value):
        try:
            rec = value.as_dict()
        except Exception:
            rec = dict(value)

        features = {k: v for k, v in rec.items() if k not in ("Label", "Timestamp")}
        return features, rec.get("Label"), rec.get("Timestamp"), rec

    def _check_dict(self, features):
        for _, v in features.items():
            if not (isinstance(v, (int, float)) and math.isfinite(v)):
                return False
        return True

    def map(self, value):
        if self.counter % self.model_save_num == 0:
            with open("./backup/ocsvm_adam_rbf.pkl", "wb") as f:
                pickle.dump(self.model, f)
            with open("./backup/ocsvm_adam_rbf_scaler.pkl", "wb") as f:
                pickle.dump(self.scaler, f)
            self.counter = 0
        self.counter += 1
        features, label_orig, timestamp, orig_rec = self._row_to_features(value)
        if not self._check_dict(features):
            output = {
                "Timestamp": timestamp,
                "orig_label": label_orig,
                "binary_label": "Attack" if label_orig != "Benign" else "Benign",
                "predicted_label": "-1",
                "predicted_binary_label": "-1",
                "probability": -1,
                "class_probabilities": "-1",
            }
            return Row(**output)

        try:
            self.scaler.learn_one(features)
            x = self.scaler.transform_one(features)
        except Exception:
            x = features

        try:
            anomaly_score = self.model.score_one(x)
        except Exception as e:
            print(e)
            anomaly_score = -1
        predicted_class = "Ignore"
        predicted_binary_class = "Ignore"
        proba = {}

        try:
            if label_orig == "Benign":
                self.model.learn_one(x)
        except Exception as e:
            print(e)

        output = {
            "Timestamp": timestamp,
            "orig_label": label_orig,
            "binary_label": "Attack" if label_orig != "Benign" else "Benign",
            "predicted_label": predicted_class,
            "predicted_binary_label": predicted_binary_class,
            "probability": anomaly_score,
            "class_probabilities": str(proba),
        }
        return Row(**output)


from pyflink.datastream.functions import MapFunction
from pyflink.common import Row

class SNARIMAXAnomalyDetector(MapFunction):

    def open(self, runtime_context: RuntimeContext):

        self.targets = [
            "Flow Byts/s",
            "Flow Pkts/s",
            "Pkt Len Mean",
            "Pkt Len Std",
            "Flow IAT Mean",
            "Flow IAT Std",
            "Fwd Pkt Len Mean",
            "Bwd Pkt Len Mean",
            "Fwd Pkts/s",
            "Bwd Pkts/s",
            "Active Mean",
            "Idle Mean",
        ]

        self.models = {}
        for target in self.targets:
            self.models[target] = time_series.SNARIMAX(
                p=5, # 5 vrednosti unazad
                d=1, # diferenciram jednom
                q=2, # 2 prethodne greske
                m=1, # sezonalnost, nemam??
                sp=0,
                sd=0,
                sq=0
            )

        self.metrics = {}
        for target in self.targets:
            self.metrics[target] = metrics.MAE()

    def map(self, value):

        try:
            rec = value.as_dict()
        except Exception:
            rec = dict(value)

        errors = {}
        predictions = {}
        label = rec.get("Label", "")

        for target in self.targets:

            try:
                y = float(rec[target])
            except Exception:
                continue

            try:
                # IZMENA 1: klampuj na 0, fizicki ne moze biti negativno
                prediction = max(0.0, self.models[target].forecast(horizon=1)[0])
            except Exception:
                prediction = 0.0

            error = abs(y - prediction)
            predictions[target] = prediction
            errors[target] = error

            self.metrics[target].update(y, prediction)

            try:
                # za sada ovako, moze na prvih n ako se smatra da su benigni
                if label == "Benign":
                    self.models[target].learn_one(y)
            except Exception:
                pass
        # MAE mean absolute error, prosecna istorijska greska za taj target
        # IZMENA 2: normalizovani anomaly score umesto apsolutnog
        normalized_errors = []
        for target in self.targets:
            if target in errors:
                mae = self.metrics[target].get()
                if mae > 0:
                    normalized_errors.append(errors[target] / mae)
                else:
                    # MAE jos nije zagrejana (pocetni koraci), vrati 0
                    normalized_errors.append(0.0)

        anomaly_score = max(normalized_errors)

        return Row(
            timestamp=rec.get("Timestamp", ""),
            label=label,

            flow_byts_prediction=predictions.get("Flow Byts/s", 0.0),
            flow_pkts_prediction=predictions.get("Flow Pkts/s", 0.0),
            pkt_len_prediction=predictions.get("Pkt Len Mean", 0.0),
            flow_byts_error=errors.get("Flow Byts/s", 0.0),
            flow_pkts_error=errors.get("Flow Pkts/s", 0.0),
            pkt_len_error=errors.get("Pkt Len Mean", 0.0),

            pkt_len_std_prediction=predictions.get("Pkt Len Std", 0.0),
            flow_iat_mean_prediction=predictions.get("Flow IAT Mean", 0.0),
            flow_iat_std_prediction=predictions.get("Flow IAT Std", 0.0),
            fwd_pkt_len_mean_prediction=predictions.get("Fwd Pkt Len Mean", 0.0),
            bwd_pkt_len_mean_prediction=predictions.get("Bwd Pkt Len Mean", 0.0),
            fwd_pkts_s_prediction=predictions.get("Fwd Pkts/s", 0.0),
            bwd_pkts_s_prediction=predictions.get("Bwd Pkts/s", 0.0),
            active_mean_prediction=predictions.get("Active Mean", 0.0),
            idle_mean_prediction=predictions.get("Idle Mean", 0.0),

            pkt_len_std_error=errors.get("Pkt Len Std", 0.0),
            flow_iat_mean_error=errors.get("Flow IAT Mean", 0.0),
            flow_iat_std_error=errors.get("Flow IAT Std", 0.0),
            fwd_pkt_len_mean_error=errors.get("Fwd Pkt Len Mean", 0.0),
            bwd_pkt_len_mean_error=errors.get("Bwd Pkt Len Mean", 0.0),
            fwd_pkts_s_error=errors.get("Fwd Pkts/s", 0.0),
            bwd_pkts_s_error=errors.get("Bwd Pkts/s", 0.0),
            active_mean_error=errors.get("Active Mean", 0.0),
            idle_mean_error=errors.get("Idle Mean", 0.0),

            anomaly_score=anomaly_score
        )


class PADAnomalyDetector(MapFunction):
    """
    Isti interfejs kao SNARIMAXAnomalyDetector, ali umesto rucnog
    racunanja greske + normalizacije, koristi river.anomaly.PredictiveAnomalyDetection.

    Po jedan PAD wrapper za svaki target. PAD interno drzi predictive model
    (SNARIMAX) + running mean/std greske + dinamicki prag. score_one vraca
    vec normalizovan anomaly score u [0, 1]. Tokom warmup_period vraca 0.0.
    """

    def open(self, runtime_context: RuntimeContext):

        self.targets = [
            "Flow Byts/s",
            "Flow Pkts/s",
            "Pkt Len Mean",
            "Pkt Len Std",
            "Flow IAT Mean",
            "Flow IAT Std",
            "Fwd Pkt Len Mean",
            "Bwd Pkt Len Mean",
            "Fwd Pkts/s",
            "Bwd Pkts/s",
            "Active Mean",
            "Idle Mean",
        ]

        # warmup: koliko prvih instanci se "proguta" pre nego sto PAD pocne
        # da vraca smislen score. Tokom warmup-a score_one vraca 0.0.
        self.warmup_period = 50

        self.models = {}
        for target in self.targets:
            self.models[target] = anomaly.PredictiveAnomalyDetection(
                predictive_model=time_series.SNARIMAX(
                    p=5,
                    d=1,
                    q=2,
                    m=1,
                    sp=0,
                    sd=0,
                    sq=0,
                ),
                horizon=1,  # predvidja 1 korak unapred
                n_std=3.0,  # prag = 3 sigma greske; vece = manje osetljivo
                warmup_period=self.warmup_period,
            )

        self.metrics = {}
        for target in self.targets:
            self.metrics[target] = metrics.MAE()

    def map(self, value):

        try:
            rec = value.as_dict()
        except Exception:
            rec = dict(value)

        errors = {}
        predictions = {}
        scores = {}
        label = rec.get("Label", "")

        for target in self.targets:

            try:
                y = float(rec[target])
            except Exception:
                continue

            # predikcija samo za logovanje (PAD je interno racuna ponovo)
            try:
                prediction = max(0.0, self.models[target].predictive_model.forecast(horizon=1)[0])
            except Exception:
                prediction = 0.0

            # PAD score: x=None jer nemamo egzogene varijable, y je ciljna vrednost
            try:
                score = self.models[target].score_one(None, y)
            except Exception:
                score = 0.0

            error = abs(y - prediction)
            predictions[target] = prediction
            errors[target] = error
            scores[target] = score

            self.metrics[target].update(y, prediction)

            # uci samo na benign saobracaju, isto kao u SNARIMAX verziji
            try:
                if label == "Benign":
                    self.models[target].learn_one(None, y)
            except Exception:
                pass

        # PAD vec daje normalizovan score po targetu -> agregiramo sa max
        # (isti princip kao u SNARIMAX verziji: napad retko pogadja sve metrike)
        anomaly_score = max(scores.values()) if scores else 0.0

        return Row(
            timestamp=rec.get("Timestamp", ""),
            label=label,

            flow_byts_prediction=predictions.get("Flow Byts/s", 0.0),
            flow_pkts_prediction=predictions.get("Flow Pkts/s", 0.0),
            pkt_len_prediction=predictions.get("Pkt Len Mean", 0.0),
            flow_byts_error=errors.get("Flow Byts/s", 0.0),
            flow_pkts_error=errors.get("Flow Pkts/s", 0.0),
            pkt_len_error=errors.get("Pkt Len Mean", 0.0),

            pkt_len_std_prediction=predictions.get("Pkt Len Std", 0.0),
            flow_iat_mean_prediction=predictions.get("Flow IAT Mean", 0.0),
            flow_iat_std_prediction=predictions.get("Flow IAT Std", 0.0),
            fwd_pkt_len_mean_prediction=predictions.get("Fwd Pkt Len Mean", 0.0),
            bwd_pkt_len_mean_prediction=predictions.get("Bwd Pkt Len Mean", 0.0),
            fwd_pkts_s_prediction=predictions.get("Fwd Pkts/s", 0.0),
            bwd_pkts_s_prediction=predictions.get("Bwd Pkts/s", 0.0),
            active_mean_prediction=predictions.get("Active Mean", 0.0),
            idle_mean_prediction=predictions.get("Idle Mean", 0.0),

            pkt_len_std_error=errors.get("Pkt Len Std", 0.0),
            flow_iat_mean_error=errors.get("Flow IAT Mean", 0.0),
            flow_iat_std_error=errors.get("Flow IAT Std", 0.0),
            fwd_pkt_len_mean_error=errors.get("Fwd Pkt Len Mean", 0.0),
            bwd_pkt_len_mean_error=errors.get("Bwd Pkt Len Mean", 0.0),
            fwd_pkts_s_error=errors.get("Fwd Pkts/s", 0.0),
            bwd_pkts_s_error=errors.get("Bwd Pkts/s", 0.0),
            active_mean_error=errors.get("Active Mean", 0.0),
            idle_mean_error=errors.get("Idle Mean", 0.0),

            anomaly_score=anomaly_score
        )

class KNN(MapFunction):
    """.
      - predicted_label        -> tacan tip (multiklasno)
      - predicted_binary_label -> Attack/Benign
    """

    def open(self, runtime_context: RuntimeContext):
        self.scaler = preprocessing.StandardScaler()
        self.num_classes = 16  # broj labela u CICIDS
        self.model = neighbors.KNNClassifier(
            n_neighbors=5,
            engine=BalancedLazySearch(
                window_size=400,
                max_classes=self.num_classes,
            ),
            weighted=True,  # blizi susedi imaju vecu tezinu u glasanju
            cleanup_every=0,  # 0 = nikad ne izbacuj stare klase iz rezultata
        )

        self.model_save_num = 1000000
        self.counter = 1

        # --- brojanje tacnosti svakih 1000 poruka (procenat pogodjenih) ---
        self.report_every = 1000   # na koliko poruka ispisujemo/logujemo procenat
        self.total_seen = 0        # ukupno validnih predikcija
        self.total_correct = 0     # ukupno pogodjenih
        self.window_seen = 0       # u tekucem prozoru od 1000
        self.window_correct = 0    # pogodjenih u tekucem prozoru
        self.log_path = "D:/logs.txt"  # fajl za procenat (kreira se ako ne postoji)

        # --- pune metrike svakih 100000 (izdvojeno u metrics_eval.py) ---
        self.metrics_every = 100000          # na koliko redova ispisujemo pun izvestaj
        self.metrics_counter = 0
        self.evaluator = PrequentialEvaluator(rolling_window=50000)
        self.metrics_path = "D:/metrics.txt"  # fajl za pune metrike

    def _row_to_features(self, value):
        try:
            rec = value.as_dict()
        except Exception:
            rec = dict(value)

        features = {k: v for k, v in rec.items() if k not in ("Label", "Timestamp")}
        return features, rec.get("Label"), rec.get("Timestamp"), rec

    def _check_dict(self, features):
        for _, v in features.items():
            if not (isinstance(v, (int, float)) and math.isfinite(v)):
                return False
        return True

    def map(self, value):
        if self.counter % self.model_save_num == 0:
            os.makedirs("D:/backup", exist_ok=True)  # napravi folder ako ne postoji
            with open("D:/backup/knn.pkl", "wb") as f:
                pickle.dump(self.model, f)
            with open("D:/backup/knn_scaler.pkl", "wb") as f:
                pickle.dump(self.scaler, f)
            self.counter = 0
        self.counter += 1
        features, label_orig, timestamp, orig_rec = self._row_to_features(value)
        if not self._check_dict(features):
            output = {
                "Timestamp": timestamp,
                "orig_label": label_orig,
                "binary_label": "Attack" if label_orig != "Benign" else "Benign",
                "predicted_label": "-1",
                "predicted_binary_label": "-1",
                "probability": -1,
                "class_probabilities": "-1",
            }
            return Row(**output)

        try:
            self.scaler.learn_one(features)
            x = self.scaler.transform_one(features)
        except Exception:
            x = features

        try:
            proba = self.model.predict_proba_one(x) or {}
            if proba:
                predicted_class = max(proba, key=proba.get)
                probability = proba.get(predicted_class, 0.0)
                predicted_binary_class = (
                    "Attack" if predicted_class != "Benign" else "Benign"
                )
            else:
                predicted_class = "None"
                predicted_binary_class = "None"
                probability = -1
        except Exception as e:
            print(e)
            proba = {}
            predicted_class = "None"
            predicted_binary_class = "None"
            probability = -1

        # broji samo validne predikcije (preskace warmup 'None' i nevalidne)
        if predicted_class not in ("None", "-1") and label_orig is not None:
            hit = 1 if predicted_class == label_orig else 0
            self.total_seen += 1
            self.total_correct += hit
            self.window_seen += 1
            self.window_correct += hit

            # svakih report_every validnih poruka: ispis u konzolu + u fajl
            if self.window_seen >= self.report_every:
                win_acc = 100.0 * self.window_correct / self.window_seen
                tot_acc = 100.0 * self.total_correct / self.total_seen
                line = (
                    f"[KNN] poslednjih {self.window_seen}: "
                    f"{self.window_correct}/{self.window_seen} = {win_acc:.2f}%  |  "
                    f"ukupno: {self.total_correct}/{self.total_seen} = {tot_acc:.2f}%"
                )
                print(line, flush=True)
                # dopisi u log fajl; "a" mod sam kreira fajl ako ne postoji
                try:
                    with open(self.log_path, "a", encoding="utf-8") as logf:
                        logf.write(line + "\n")
                except Exception as e:
                    print("log write error:", e)
                # resetuj prozor
                self.window_seen = 0
                self.window_correct = 0

        # --- pune metrike (test-then-train) ---
        # evaluator sam preskace warmup 'None'/nevalidne '-1'
        self.evaluator.update(label_orig, predicted_class)
        self.metrics_counter += 1

        # svakih metrics_every redova: pun izvestaj u konzolu + u D:/metrics.txt
        if self.metrics_counter % self.metrics_every == 0:
            report = self.evaluator.full_report()
            print(report, flush=True)
            try:
                with open(self.metrics_path, "a", encoding="utf-8") as mf:
                    mf.write(report + "\n")
            except Exception as e:
                print("metrics write error:", e)

        try:
            # ucenje
            self.model.learn_one(x, label_orig)
        except Exception as e:
            print(e)

        output = {
            "Timestamp": timestamp,
            "orig_label": label_orig,
            "binary_label": "Attack" if label_orig != "Benign" else "Benign",
            "predicted_label": predicted_class,
            "predicted_binary_label": predicted_binary_class,
            "probability": probability,
            "class_probabilities": str(proba),
        }
        return Row(**output)
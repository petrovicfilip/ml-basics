import collections
import operator
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import pickle
import math
from typing import Optional, Tuple

from river import time_series
from river import metrics

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
        self.model = anomaly.LocalOutlierFactor()
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
            with open("./backup/lof.pkl", "wb") as f:
                pickle.dump(self.model, f)
            with open("./backup/lof_scaler.pkl", "wb") as f:
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

class BalancedLazySearch(neighbors.LazySearch):
    """LazySearch sa memorijom balansiranom po klasama.

    Problem originala (vidi lazy.py): jedan globalni FIFO prozor
        self.window = collections.deque(maxlen=window_size)
    Kad dugo dolaze samo Benign primeri, taj deque se napuni Benign-om i
    raniji napadi se izbace -> model skoro uvek predvidja Benign.

    Resenje: umesto jednog deque-a, drzimo PO JEDAN deque po klasi (labeli).
    Svaki pod-bufer je FIFO velicine W // N:
        W = window_size (ukupna memorija, ostaje fiksna)
        N = max_classes  (broj labela)
    Pravila (tacno po specifikaciji):
      - novi sample ide samo u deque SVOJE klase (append gleda y iz (x, y));
      - kad je taj deque pun, deque(maxlen=...) sam izbacuje najstariji
        element TE klase (FIFO po klasi), ne globalno najstariji;
      - search() spaja sve pod-bufere i trazi suseda preko svih klasa.

    Menja se SAMO storage. Metode `append` i `search` su jedine pregazene;
    `update`, `dist_func`, format povratka i sve sto KNNClassifier ocekuje
    ostaje identicno originalnom lazy.py (isti sorted + zip pristup).
    """

    def __init__(
        self,
        window_size: int = 50,
        max_classes: int = 2,
        min_distance_keep: float = 0.0,
        dist_func=None,
    ):
        # parent postavlja window_size, min_distance_keep, dist_func, i
        # self.window = deque(); window setter ispod neutralizuje taj deque.
        super().__init__(
            window_size=window_size,
            min_distance_keep=min_distance_keep,
            dist_func=dist_func,
        )
        self.max_classes = max_classes
        # bar 1 mesto po klasi cak i ako je N > W
        self.per_class_size = max(1, window_size // max_classes)
        # label -> deque(maxlen=per_class_size)
        self._buffers: dict = {}

    @property
    def window(self):
        # KNNClassifier.clean_up_classes() prolazi kroz self.window.
        # Izlozimo sve pod-bufere kao jednu spojenu listu (citanje).
        merged = collections.deque()
        for buf in self._buffers.values():
            merged.extend(buf)
        return merged

    @window.setter
    def window(self, value):
        # Nasledjeni __init__ radi self.window = deque(maxlen=W). Mi ne
        # koristimo globalni prozor (drzimo _buffers), pa to progutamo.
        pass

    def append(self, item, extra=None, **kwargs):
        # item je (x, y) -> klasa je y. Za upit (x, None) se ne poziva append.
        label = item[1]
        if label not in self._buffers:
            self._buffers[label] = collections.deque(maxlen=self.per_class_size)
        # FIFO po klasi: deque sa maxlen sam izbaci najstariji kad je pun
        self._buffers[label].append((item, *(extra or [])))

    def search(self, item, n_neighbors, **kwargs):
        # Identicna logika kao original lazy.py, samo nad spojem pod-bufera
        # umesto nad jednim self.window deque-om.
        points = (
            (*p, self.dist_func(item, p[0]))
            for buf in self._buffers.values()
            for p in buf
        )
        return tuple(
            map(list, zip(*sorted(points, key=operator.itemgetter(-1))[:n_neighbors]))
        )


class KNN(MapFunction):
    """.
      - predicted_label        -> tacan tip (multiklasno)
      - predicted_binary_label -> Attack/Benign
    """

    def open(self, runtime_context: RuntimeContext):
        # KNN racuna rastojanja -> skaliranje je OBAVEZNO, inace par kolona
        # sa velikim opsegom (npr. Flow Byts/s) dominira nad svim ostalim.
        self.scaler = preprocessing.StandardScaler()

        # LazySearch = egzaktni engine sa kliznim prozorom (FIFO).
        # window_size = koliko poslednjih instanci se pamti za pretragu suseda.
        #
        # BalancedLazySearch deli prozor na jednake pod-bufere PO KLASI, da
        # dug Benign burst ne bi izgurao napade iz memorije. Uz 16 labela i
        # window_size=1600, svaka klasa drzi 1600 // 16 = 100 primera.
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

        # --- brojanje tacnosti svakih 1000 poruka ---
        self.report_every = 1000   # na koliko poruka ispisujemo
        self.total_seen = 0        # ukupno validnih predikcija (bez warmup/None)
        self.total_correct = 0     # ukupno pogodjenih (kumulativno)
        self.window_seen = 0       # u tekucem prozoru od 1000
        self.window_correct = 0    # pogodjenih u tekucem prozoru

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
            with open("./backup/knn.pkl", "wb") as f:
                pickle.dump(self.model, f)
            with open("./backup/knn_scaler.pkl", "wb") as f:
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
                # prvi redovi: model jos nema nijednu naucenu instancu
                predicted_class = "None"
                predicted_binary_class = "None"
                probability = -1
        except Exception as e:
            print(e)
            proba = {}
            predicted_class = "None"
            predicted_binary_class = "None"
            probability = -1

        # --- prequential tacnost: meri se PRE ucenja (test-then-train) ---
        # broji samo validne predikcije (preskace warmup 'None' i nevalidne)
        if predicted_class not in ("None", "-1") and label_orig is not None:
            hit = 1 if predicted_class == label_orig else 0
            self.total_seen += 1
            self.total_correct += hit
            self.window_seen += 1
            self.window_correct += hit

            # ispis svakih report_every validnih poruka
            if self.window_seen >= self.report_every:
                win_acc = 100.0 * self.window_correct / self.window_seen
                tot_acc = 100.0 * self.total_correct / self.total_seen
                print(
                    f"[KNN] poslednjih {self.window_seen}: "
                    f"{self.window_correct}/{self.window_seen} = {win_acc:.2f}%  |  "
                    f"ukupno: {self.total_correct}/{self.total_seen} = {tot_acc:.2f}%",
                    flush=True,
                )
                # resetuj prozor
                self.window_seen = 0
                self.window_correct = 0

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
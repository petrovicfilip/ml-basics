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
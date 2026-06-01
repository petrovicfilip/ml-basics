"""
metrics_eval.py
================
Izdvojena logika za evaluaciju online (streaming) multiklasnog klasifikatora
za intrusion detection (CICIDS). Koristi River metrike koje se azuriraju
INKREMENTALNO (prequential / test-then-train): za svaki slog prvo predvidimo,
pa metriku azuriramo (y_true, y_pred), pa tek onda model uci.

Glavna klasa: PrequentialEvaluator
  - drzi globalne metrike (Accuracy, BalancedAccuracy, MacroF1, ...),
  - drzi ConfusionMatrix + ClassificationReport (per-class P/R/F1/Support),
  - dodatno drzi "rolling" (prozorske) metrike da se vidi trenutni trend,
"""

from __future__ import annotations

import collections

from river import metrics
from river import utils


class PrequentialEvaluator:
    """Sakuplja i racuna metrike za online multiklasnu klasifikaciju.

    Parameters
    ----------
    rolling_window
        Velicina prozora za "skorasnje" metrike (npr. poslednjih 10000 slogova).
        Globalne metrike su kumulativne (od pocetka stream-a); rolling pokazuje
        trenutni trend, sto je vazno za concept drift i burst-ove napada.
    ignore_labels
        Predikcije/labela koje se preskacu (warmup 'None', nevalidni '-1').
    """

    def __init__(self, rolling_window: int = 10000, ignore_labels=("None", "-1", None)):
        self.ignore_labels = set(ignore_labels)

        # --- GLOBALNE (kumulativne) metrike ---
        # Accuracy: udeo tacnih. Mana kod neuravnotezenih klasa: visok i kad
        # model uvek pogadja vecinsku klasu (Benign). Zato NIJE dovoljna sama.
        self.accuracy = metrics.Accuracy()

        # BalancedAccuracy: prosek recall-a po klasama -> kaznjava ignorisanje
        # retkih klasa napada. Kljucna metrika za imbalanced IDS.
        self.balanced_accuracy = metrics.BalancedAccuracy()

        # Macro F1: prosek F1 po klasama, sve klase jednako vazne (retke kao i
        # ceste). Micro F1: globalno po slogovima (dominira vecinska klasa).
        # Weighted F1: prosek tezinski po support-u. Dajemo sve tri za poredjenje.
        self.macro_f1 = metrics.MacroF1()
        self.micro_f1 = metrics.MicroF1()
        self.weighted_f1 = metrics.WeightedF1()

        # Macro precision/recall odvojeno (da vidimo lazne uzbune vs promaseni napadi)
        self.macro_precision = metrics.MacroPrecision()
        self.macro_recall = metrics.MacroRecall()

        # CohenKappa: slaganje iznad slucajnog pogadjanja. Korisno kad je
        # raspodela klasa jako neravnomerna. NAPOMENA: River-ov MCC u multiklasnom
        # slucaju nije pouzdan (vraca 0.0 i za savrsene predikcije), a istrazivanja
        # pokazuju da se MCC i Kappa u multiklasi gotovo poklapaju -> koristimo Kappa.
        self.cohen_kappa = metrics.CohenKappa()

        # GeometricMean: koren proizvoda recall-a po klasama -> ako ijedna klasa
        # ima recall 0, ceo G-mean je 0. Najstroziji prema "zaboravljenim" klasama.
        self.geometric_mean = metrics.GeometricMean()

        # --- PER-CLASS sve odjednom: Precision/Recall/F1/Support po klasi +
        #     Macro/Micro/Weighted + accuracy. Glavni izvor per-class brojeva. ---
        self.report = metrics.ClassificationReport()

        # --- Confusion matrix: ko se sa kim brka (npr. DDoS <-> Benign) ---
        self.confusion = metrics.ConfusionMatrix()

        # --- ROLLING (prozorske) verzije kljucnih metrika ---
        # Pokazuju trenutni trend umesto kumulativnog proseka. Vazno jer u IDS
        # stream-u dolaze burst-ovi: kumulativ "razvodni" lokalni pad tacnosti.
        self.rolling_window = rolling_window
        self.roll_accuracy = utils.Rolling(metrics.Accuracy(), window_size=rolling_window)
        self.roll_macro_f1 = utils.Rolling(metrics.MacroF1(), window_size=rolling_window)
        self.roll_balanced_acc = utils.Rolling(
            metrics.BalancedAccuracy(), window_size=rolling_window
        )

        # broj validnih (ne-ignorisanih) uzoraka koje smo evaluirali
        self.n_evaluated = 0
        # broj preskocenih (warmup/nevalidnih)
        self.n_skipped = 0

    def update(self, y_true, y_pred):
        """Azuriraj sve metrike jednim (y_true, y_pred) parom labela.

        Vraca True ako je uzorak evaluiran, False ako je preskocen.
        """
        # preskoci warmup 'None' / nevalidne '-1' / prazne labele
        if y_pred in self.ignore_labels or y_true in self.ignore_labels:
            self.n_skipped += 1
            return False

        self.accuracy.update(y_true, y_pred)
        self.balanced_accuracy.update(y_true, y_pred)
        self.macro_f1.update(y_true, y_pred)
        self.micro_f1.update(y_true, y_pred)
        self.weighted_f1.update(y_true, y_pred)
        self.macro_precision.update(y_true, y_pred)
        self.macro_recall.update(y_true, y_pred)
        self.cohen_kappa.update(y_true, y_pred)
        self.geometric_mean.update(y_true, y_pred)
        self.report.update(y_true, y_pred)
        self.confusion.update(y_true, y_pred)

        self.roll_accuracy.update(y_true, y_pred)
        self.roll_macro_f1.update(y_true, y_pred)
        self.roll_balanced_acc.update(y_true, y_pred)

        self.n_evaluated += 1
        return True

    def global_summary(self) -> str:
        """Kratak jednoredni pregled globalnih metrika (za cest ispis)."""
        return (
            f"n={self.n_evaluated} "
            f"acc={self.accuracy.get() * 100:.2f}% "
            f"bal_acc={self.balanced_accuracy.get() * 100:.2f}% "
            f"macroF1={self.macro_f1.get() * 100:.2f}% "
            
            f"kappa={self.cohen_kappa.get():.4f} "
            f"| roll(acc={self.roll_accuracy.get() * 100:.2f}% "
            f"macroF1={self.roll_macro_f1.get() * 100:.2f}%)"
        )

    def full_report(self) -> str:
        """Detaljan izvestaj: globalne metrike + per-class + confusion matrix."""
        lines = []
        lines.append("=" * 70)
        lines.append(f"EVALUACIJA posle {self.n_evaluated} validnih slogova "
                     f"(preskoceno {self.n_skipped})")
        lines.append("=" * 70)
        lines.append("GLOBALNE METRIKE (kumulativno):")
        lines.append(f"  Accuracy          : {self.accuracy.get() * 100:.2f}%")
        lines.append(f"  Balanced Accuracy : {self.balanced_accuracy.get() * 100:.2f}%")
        lines.append(f"  Macro F1          : {self.macro_f1.get() * 100:.2f}%")
        lines.append(f"  Micro F1          : {self.micro_f1.get() * 100:.2f}%")
        lines.append(f"  Weighted F1       : {self.weighted_f1.get() * 100:.2f}%")
        lines.append(f"  Macro Precision   : {self.macro_precision.get() * 100:.2f}%")
        lines.append(f"  Macro Recall      : {self.macro_recall.get() * 100:.2f}%")
        lines.append(f"  Cohen's Kappa     : {self.cohen_kappa.get():.4f}")
        lines.append(f"  Geometric Mean    : {self.geometric_mean.get():.4f}")
        lines.append("")
        lines.append(f"ROLLING (poslednjih ~{self.rolling_window}):")
        lines.append(f"  Accuracy          : {self.roll_accuracy.get() * 100:.2f}%")
        lines.append(f"  Balanced Accuracy : {self.roll_balanced_acc.get() * 100:.2f}%")
        lines.append(f"  Macro F1          : {self.roll_macro_f1.get() * 100:.2f}%")
        lines.append("")
        lines.append("PER-CLASS (Precision / Recall / F1 / Support):")
        lines.append(str(self.report))
        lines.append("")
        lines.append("CONFUSION MATRIX (red = stvarno, kolona = predvidjeno):")
        lines.append(str(self.confusion))
        lines.append("=" * 70)
        return "\n".join(lines)
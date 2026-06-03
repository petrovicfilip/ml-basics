import collections
import math

from river import anomaly


class WindowedLOF:
    def __init__(self, n_neighbors=10, window_size=2000, rebuild_every=1000,
                 min_dist=1e-9):
        self.n_neighbors = n_neighbors
        self.window_size = window_size
        self.rebuild_every = rebuild_every
        self.min_dist = min_dist

        self.window = collections.deque(maxlen=window_size)  # FIFO prozor
        self.model = anomaly.LocalOutlierFactor(n_neighbors=n_neighbors)
        self._since_rebuild = 0
        self._primed = False  # da li je model imao bar jedan rebuild

    def _euclidean(self, a, b):
        s = 0.0
        for k in a:
            d = a[k] - b.get(k, 0.0)
            s += d * d
        return math.sqrt(s)

    def _is_near_duplicate(self, x):
        # near-duplikat ako je blizi od min_dist nekoj od poslednjih tacaka
        # gledam samo poslednjih 50 da ne placamo O(W) po tacki
        for old in list(self.window)[-50:]:
            if self._euclidean(x, old) <= self.min_dist:
                return True
        return False

    def _rebuild(self):
        new_model = anomaly.LocalOutlierFactor(n_neighbors=self.n_neighbors)
        for pt in self.window:
            new_model.learn_one(pt)
        self.model = new_model
        self._since_rebuild = 0
        self._primed = True

    def score_one(self, x):
        if not self._primed and len(self.window) < self.n_neighbors:
            return 0.0
        try:
            return self.model.score_one(x)
        except Exception:
            return 0.0

    def learn_one(self, x):
        if self._is_near_duplicate(x):   # preskoci redundantne tacke
            return self
        self.window.append(x)            # FIFO
        self._since_rebuild += 1
        if self._since_rebuild >= self.rebuild_every or (
            not self._primed and len(self.window) >= self.n_neighbors
        ):
            self._rebuild()
        else:
            try:
                self.model.learn_one(x)
            except Exception:
                pass
        return self
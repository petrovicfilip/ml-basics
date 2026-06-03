import collections
import operator

from river import neighbors


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
        # self.window = deque()
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
        pass

    def append(self, item, extra=None, **kwargs):
        # item je (x, y) -> klasa je y. Za upit (x, None) se ne poziva append.
        label = item[1]
        if label not in self._buffers:
            self._buffers[label] = collections.deque(maxlen=self.per_class_size)
        # FIFO deque
        self._buffers[label].append((item, *(extra or [])))

    def search(self, item, n_neighbors, **kwargs):
        # Identicna logika kao original lazy.py, samo nad spojem pod-bufera
        points = (
            (*p, self.dist_func(item, p[0]))
            for buf in self._buffers.values()
            for p in buf
        )
        return tuple(
            map(list, zip(*sorted(points, key=operator.itemgetter(-1))[:n_neighbors]))
        )
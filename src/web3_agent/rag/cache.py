"""Bounded process-local TTL cache. Keys include the live corpus revision."""
from collections import OrderedDict
from copy import deepcopy
from time import monotonic


class RetrievalCache:
    """有界 TTL 缓存。默认深拷贝，调用方改到结果不会污染缓存。

    常驻索引是只读对象，且体积大头在切片正文与倒排上，每次命中都递归复制它
    没有任何意义 —— 这类缓存用 clone=False 直接共用同一个对象。
    """

    def __init__(self, max_entries=128, clone=True):
        self.entries = OrderedDict()
        self.max_entries = max_entries
        self.clone = clone

    def get(self, key):
        entry = self.entries.get(key)
        if entry is None:
            return None
        expiry, value = entry
        if expiry <= monotonic():
            del self.entries[key]
            return None
        self.entries.move_to_end(key)
        return deepcopy(value) if self.clone else value

    def put(self, key, value, ttl):
        if ttl <= 0:
            return
        self.entries[key] = (monotonic()+ttl, deepcopy(value) if self.clone else value)
        self.entries.move_to_end(key)
        while len(self.entries) > self.max_entries:
            self.entries.popitem(last=False)

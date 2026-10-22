"""Bounded process-local TTL cache. Keys include the live corpus revision."""
from collections import OrderedDict
from copy import deepcopy
from time import monotonic

class RetrievalCache:
    def __init__(self, max_entries=128):
        self.entries = OrderedDict()
        self.max_entries = max_entries

    def get(self, key):
        entry = self.entries.get(key)
        if entry is None:
            return None
        expiry, value = entry
        if expiry <= monotonic():
            del self.entries[key]
            return None
        self.entries.move_to_end(key)
        return deepcopy(value)

    def put(self, key, value, ttl):
        if ttl <= 0:
            return
        self.entries[key] = (monotonic()+ttl, deepcopy(value))
        self.entries.move_to_end(key)
        while len(self.entries) > self.max_entries:
            self.entries.popitem(last=False)

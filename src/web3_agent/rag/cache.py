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
        # entries：按最近访问顺序维护的缓存项，值为 (过期单调时间, 缓存值)。
        """初始化有界 TTL 缓存及拷贝策略。

        Args:
            max_entries: 缓存可保留的最大键数量。
            clone: 读取和写入缓存时是否深拷贝值。
        """
        self.entries = OrderedDict()
        self.max_entries = max_entries
        self.clone = clone

    def get(self, key):
        # 缓存项为 (expiry, value)：expiry 来自 monotonic，只用于进程内过期判断；value 是被缓存对象。
        """读取未过期缓存并更新 LRU 顺序；按配置返回深拷贝或共享只读对象。

        Args:
            key: 缓存键。

        Returns:
            缓存命中时返回值，未命中或已过期时返回 None；按 clone 设置决定是否深拷贝。
        """
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
        """缓存值并设置单调时钟到期时间，超过容量时淘汰最久未使用条目。

        Args:
            key: 缓存键。
            value: 待校验、转换或缓存的值。
            ttl: 缓存有效秒数；非正值表示不写入缓存。
        """
        if ttl <= 0:
            return
        self.entries[key] = (monotonic()+ttl, deepcopy(value) if self.clone else value)
        self.entries.move_to_end(key)
        while len(self.entries) > self.max_entries:
            self.entries.popitem(last=False)

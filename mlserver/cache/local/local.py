from collections import OrderedDict
from ..cache import ResponseCache


class LocalCache(ResponseCache):
    """In-memory FIFO response cache backed by an ``OrderedDict``."""

    def __init__(self, size=100):
        """Create a cache with a maximum of *size* entries (default 100)."""
        self.cache = OrderedDict()
        self.size_limit = size

    async def insert(self, key: str, value: str):
        """Insert a value, evicting the oldest entry if the cache is full."""
        self.cache[key] = value
        cache_size = await self.size()
        if cache_size > self.size_limit:
            # The cache removes the first entry if it overflows (i.e. in FIFO order)
            self.cache.popitem(last=False)
        return None

    async def lookup(self, key: str) -> str:
        """Return the cached value for *key*, or an empty string if absent."""
        if key in self.cache:
            return self.cache[key]
        else:
            return ""

    async def size(self) -> int:
        """Return the current number of cached entries."""
        return len(self.cache)

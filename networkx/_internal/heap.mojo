from builtin.value import ImplicitlyCopyable
from collections import List
from collections.dict import KeyElement


struct _HeapItem[N: KeyElement & ImplicitlyCopyable](ImplicitlyCopyable):
    var prio: Float64
    var count: Int
    var node: Self.N

    fn __init__(out self, prio: Float64, count: Int, node: Self.N):
        self.prio = prio
        self.count = count
        self.node = node


struct _MinHeap[N: KeyElement & ImplicitlyCopyable]:
    var _data: List[_HeapItem[Self.N]]

    fn __init__(out self):
        self._data = List[_HeapItem[Self.N]]()

    fn is_empty(self) -> Bool:
        return len(self._data) == 0

    fn _less(self, a: _HeapItem[Self.N], b: _HeapItem[Self.N]) -> Bool:
        if a.prio < b.prio:
            return True
        if a.prio > b.prio:
            return False
        return a.count < b.count

    fn push(mut self, item: _HeapItem[Self.N]):
        self._data.append(item)
        var i = len(self._data) - 1
        while i > 0:
            var parent = (i - 1) // 2
            if not self._less(self._data[i], self._data[parent]):
                break
            var tmp = self._data[parent]
            self._data[parent] = self._data[i]
            self._data[i] = tmp
            i = parent

    fn pop_min(mut self) raises -> _HeapItem[Self.N]:
        if len(self._data) == 0:
            raise Error("empty heap")
        var result = self._data[0]
        var last = self._data.pop()
        if len(self._data) == 0:
            return result
        self._data[0] = last
        var i = 0
        while True:
            var left = 2 * i + 1
            var right = 2 * i + 2
            if left >= len(self._data):
                break
            var smallest = left
            if right < len(self._data) and self._less(self._data[right], self._data[left]):
                smallest = right
            if not self._less(self._data[smallest], self._data[i]):
                break
            var tmp = self._data[i]
            self._data[i] = self._data[smallest]
            self._data[smallest] = tmp
            i = smallest
        return result

"""Exact expansion and legacy-equivalence checks for optional fast construction."""
from collections import Counter
import numpy as np
import compressgnn_offline as lib


def expand(v, e, node, n):
    stack = list(map(int, e[v[node]:v[node + 1]])); out = []
    while stack:
        child = stack.pop()
        if child < n:
            out.append(child)
        else:
            children = e[v[child]:v[child + 1]]
            assert np.all(children < child), 'Rule graph must be acyclic'
            stack.extend(map(int, children))
    return Counter(out)


def check(rows, packed):
    v, e, n, nr = packed
    assert n == len(rows) and len(v) == n + nr + 1
    assert v[0] == 0 and v[-1] == len(e) and np.all(np.diff(v) >= 0)
    for i, row in enumerate(rows):
        assert expand(v, e, i, n) == Counter(row)


checks = 0
rng = np.random.default_rng(73)
for n in [1, 2, 17, 128, 512]:
    fixtures = [
        [[] for _ in range(n)],
        [[i] for i in range(n)],
        [list(range(min(n, 16))) for _ in range(n)],
        [sorted(rng.choice(n, size=min(n, 20), replace=False).tolist()) for _ in range(n)],
        [sorted(rng.integers(n, size=8).tolist()) for _ in range(n)],
    ]
    for rows in fixtures:
        v = np.array([0] + list(np.cumsum([len(x) for x in rows])), dtype=np.int32)
        e = np.array([j for row in rows for j in row], dtype=np.int32)
        original = lib.compress_csr(v, e, n)
        for freq in [2, 4, 8, 16, 32]:
            packed = lib.compress_csr_fast(v, e, n, freq)
            check(rows, packed)
            if freq == 2:
                assert np.array_equal(original[0], packed[0])
                assert np.array_equal(original[1], packed[1])
            filtered = lib.filter_csr(*packed, 16)
            check(rows, filtered)
            depth = lib.depth_filter_csr(*filtered, 3, 100000)
            check(rows, depth)
            checks += 1
for freq in [-1, 0, 1]:
    try:
        lib.compress_csr_fast(np.array([0, 0],np.int32),np.array([],np.int32),1,freq)
    except ValueError:
        checks += 1
    else:
        raise AssertionError('Invalid frequency accepted')
print('PASSED', checks, 'cases; raw, filtered and depth-filtered expansion checked')

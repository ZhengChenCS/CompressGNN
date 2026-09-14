"""Select exact batch graph construction before the existing post-filters."""
import importlib
import operator


def resolve_backend(backend):
    if backend not in ('auto', 'cpu', 'cuda'):
        raise ValueError("compression_backend must be 'auto', 'cpu', or 'cuda'")
    if backend in ('auto', 'cuda'):
        try:
            cuda = importlib.import_module('compressgnn_batch_cuda')
        except ImportError as exc:
            if backend == 'cuda':
                raise RuntimeError('CUDA compression extension is not installed; rebuild offline with COMPRESSGNN_BUILD_CUDA=1') from exc
        else:
            if cuda.is_available():
                return 'cuda'
            if backend == 'cuda':
                raise RuntimeError('CUDA compression requested but no usable CUDA device is available')
        return 'cpu'
    return backend


def compress_graph_csr(vlist, elist, min_pair_frequency=16,
                       backend='auto', threads=4, rounds=12):
    frequency = operator.index(min_pair_frequency)
    threads = operator.index(threads)
    rounds = operator.index(rounds)
    if threads < 1 or rounds < 1:
        raise ValueError('compression_threads and compression_rounds must be positive')
    selected = resolve_backend(backend)
    if frequency < 3:
        raise ValueError('Batch compression needs min_pair_frequency >= 3')
    if selected == 'cuda':
        # Runtime failures (including OOM) propagate, rather than silently retrying.
        cuda = importlib.import_module('compressgnn_batch_cuda')
        return cuda.compress_csr_batch(vlist, elist, rounds=rounds, min_pair_frequency=frequency)
    offline = importlib.import_module('compressgnn_offline')
    return offline.compress_csr_batch(vlist, elist, threads=threads, rounds=rounds,
                                      min_pair_frequency=frequency)

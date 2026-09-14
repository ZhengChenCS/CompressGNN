"""Batch dispatch, exact post-filter expansion and loader/runtime compatibility."""
from collections import Counter
from pathlib import Path
from unittest import mock
import importlib
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'loader'), str(ROOT/'layer')]
import compressgnn_offline as offline
import compression_backend as dispatch


def expand(v, e, node, n):
    stack = list(reversed(e[v[node]:v[node+1]].tolist()))
    output = []
    while stack:
        child = stack.pop()
        if child < n:
            output.append(child)
        else:
            children = e[v[child]:v[child+1]]
            assert np.all(children < child)
            stack.extend(reversed(children.tolist()))
    return output


def check(rows, packed, ordered=True):
    v,e,n,nr = packed
    assert len(v)==n+nr+1 and n==len(rows)
    if ordered:
        assert v.dtype==np.int32 and e.dtype==np.int32
    for node,row in enumerate(rows):
        actual=expand(v,e,node,n)
        assert actual==row if ordered else Counter(actual)==Counter(row)


cuda = dispatch.resolve_backend('auto') == 'cuda'
backends = ['cpu'] + (['cuda'] if cuda else [])
rng=np.random.default_rng(17)
fixtures=[[],[[]],[[0]*50], [list(range(32)) for _ in range(64)]]
fixtures += [[rng.integers(32,size=20).tolist() for _ in range(32)] for _ in range(4)]
for rows in fixtures:
    v=np.array([0]+list(np.cumsum([len(row) for row in rows])),np.int64)
    e=np.array([x for row in rows for x in row],np.int64)
    for frequency in [3,16,256]:
        reference=dispatch.compress_graph_csr(v,e,frequency,backend='cpu')
        for backend in backends:
            result=dispatch.compress_graph_csr(v,e,frequency,backend=backend)
            check(rows,result)
            assert np.array_equal(result[0],reference[0]) and np.array_equal(result[1],reference[1])
            # Existing post-filters expect at least one original vertex.
            if rows:
                result=offline.filter_csr(*result,16);check(rows,result,False)
                result=offline.depth_filter_csr(*result,3,100000);check(rows,result,False)
for v,e in [(np.array([0,1],np.int64),np.array([2**32],np.uint64)),
            (np.array([0,1]),np.array([0.5]))]:
    for backend in backends:
        try:dispatch.compress_graph_csr(v,e,backend=backend)
        except ValueError:pass
        else:raise AssertionError('Invalid integer CSR accepted')

real_import=importlib.import_module
with mock.patch.object(dispatch.importlib,'import_module',side_effect=lambda name: (_ for _ in ()).throw(ImportError('not built')) if name=='compressgnn_batch_cuda' else real_import(name)):
    assert dispatch.resolve_backend('auto')=='cpu'
    try:dispatch.resolve_backend('cuda')
    except RuntimeError:pass
    else:raise AssertionError('Explicit missing CUDA backend accepted')
with mock.patch.object(dispatch.importlib,'import_module',return_value=mock.Mock(is_available=lambda:False)):
    assert dispatch.resolve_backend('auto')=='cpu'
for bad in ['wrong',None]:
    try:dispatch.resolve_backend(bad)
    except ValueError:pass
    else:raise AssertionError('Invalid backend accepted')

# Exercise the existing COO/CSR filtering, normalization, graph partitioning and
# CUDA propagation forward/backward. Heavy training dependencies are optional.
try:
    import torch
    from CompressgnnData import CompressgnnData
    from KPropagate import KPropagate
except ImportError as exc:
    print('Loader/runtime smoke test skipped: optional training dependency unavailable:',exc)
else:
    n=64;rows=[list(range(32)) for _ in range(n)]
    v=np.arange(n+1,dtype=np.int32)*32;e=np.tile(np.arange(32,dtype=np.int32),n)
    coo=np.vstack([np.repeat(np.arange(n,dtype=np.int32),32),e])
    x=rng.standard_normal((n,8)).astype(np.float32);mask=np.ones(n,dtype=bool)
    for graph_type,edge in [('coo',coo),('csr',(v,e))]:
        for normalize in [False,True]:
            reference=None
            reference_graph=None
            for backend in backends:
                data=CompressgnnData(x,edge,np.zeros(n,np.int64),mask,mask,mask,
                                     add_self_loop=False,normalize=normalize,graph_type=graph_type,
                                     compression_backend=backend)
                assert data.compression_backend==backend
                if graph_type=='coo':
                    pieces=data.edge_index+data.edge_weight
                else:
                    graphs=[data.edge_index.v2v_graph,data.edge_index.v2r_graph,data.edge_index.r2v_graph]+data.edge_index.r2r_graph
                    pieces=[part for graph in graphs for part in graph.csr()]
                if reference_graph is not None:
                    assert len(pieces)==len(reference_graph)
                    assert all(torch.equal(a,b) for a,b in zip(pieces,reference_graph))
                reference_graph=[part.clone() for part in pieces]
                if torch.cuda.is_available():
                    data.to('cuda');features=data.x.detach().clone().requires_grad_()
                    output=KPropagate()(features,data.edge_index,getattr(data,'edge_weight',None),data.vertex_cnt,data.rule_cnt)
                    grad=torch.autograd.grad(output.square().sum(),features)[0]
                    assert torch.isfinite(output).all() and torch.isfinite(grad).all()
                    if reference is not None:
                        # Existing GPU sparse reductions use floating-point atomic sums.
                        torch.testing.assert_close(output,reference[0],rtol=1e-4,atol=1e-4)
                        torch.testing.assert_close(grad,reference[1],rtol=1e-4,atol=1e-4)
                    reference=(output.detach(),grad.detach())
    print('COO/CSR loader and propagation smoke checks passed')
print('Batch compression tests passed; tested backends:',backends)

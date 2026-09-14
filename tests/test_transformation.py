"""Run with python tests/test_transformation.py; requires CUDA runtime build."""
import sys
from pathlib import Path
root = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(root / 'layer'), str(root / 'loader')]
import torch
from torch_scatter import scatter_mean
import compressgnn_runtime as native
from CompressgnnCluster import Compressgnn_Cluster,Compressgnn_Cluster_Function,_Mean
from CompressgnnReconstruct import Compressgnn_Reconstruct

torch.set_num_threads(2);torch.manual_seed(26);torch.backends.cuda.matmul.allow_tf32=False
checks=[]
def close(a,b,name):
 torch.testing.assert_close(a,b,atol=1e-5,rtol=1e-4);checks.append(name)
def reference_ids(x,w):
 h=x@w;bits=w.size(1);code=torch.zeros(x.size(0),device=x.device,dtype=torch.long)
 for i in range(bits):code=(code<<1)+(h[:,i]>0).long()
 unique,inv=torch.unique(code,sorted=True,return_inverse=True)
 idx=inv+1;return idx,torch.bincount(idx,minlength=unique.numel()+1).int(),unique.numel()
for dtype in [torch.float32,torch.float64]:
 for width in [0,1,31,33,128,602]:
  for n,bits in [(0,8),(1,0),(67,8),(257,16)]:
   x=torch.randn(n,width,device='cuda',dtype=dtype);w=torch.randn(width,bits,device='cuda',dtype=dtype)
   work=native.cluster_workspace(x,bits);idx,count,active=native.cluster_assign(x,w,bits,work)
   ri,rc,ra=reference_ids(x,w);assert active==ra;close(idx,ri,'hash_ids');close(count,rc,'hash_counts')
   y=native.cluster_mean(x,idx,count);ref=scatter_mean(x,idx,dim=0,dim_size=active+1);close(y,ref,'mean')
   grad=torch.randn_like(y);dx=native.cluster_mean_backward(grad,idx,count)
   close(dx,grad.index_select(0,idx)/count.index_select(0,idx).clamp_min(1).to(dtype)[:,None],'mean_backward')
   if bits==8 and n==67:
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
     wi=native.cluster_workspace(x,bits);ii,cc,aa=native.cluster_assign(x,w,bits,wi);yy=native.cluster_mean(x,ii,cc)
    stream.synchronize();close(ii,ri,'stream_hash');close(yy,ref,'stream_mean')
# All vertices collide, and a large-bit workspace remains valid.
for bits in [0,8,24]:
 x=torch.ones(513,33,device='cuda');w=torch.ones(33,bits,device='cuda')
 idx,count,active=native.cluster_assign(x,w,bits,native.cluster_workspace(x,bits))
 assert active==1 and count.tolist()==[0,513];checks.append('collision_count_'+str(bits))
# Fixed assignments, including unused clusters, double-precision finite differences.
idx=torch.tensor([1,1,3,1,3,3],device='cuda');count=torch.bincount(idx,minlength=5).int()
x=torch.randn(6,5,device='cuda',dtype=torch.double,requires_grad=True)
assert torch.autograd.gradcheck(lambda a:_Mean.apply(a,idx,count),(x,),eps=1e-6,atol=1e-5,rtol=1e-3);checks.append('mean_gradcheck')
reconstruct=Compressgnn_Reconstruct()
for dtype in [torch.float32,torch.float64]:
 for width in [1,31,33,128,602]:
  for relu in [None,'relu']:
   reps=torch.randn(5,width,device='cuda',dtype=dtype,requires_grad=True);bias=torch.randn(width,device='cuda',dtype=dtype,requires_grad=True)
   out=reconstruct(reps,idx,bias=bias,activation=relu);ref=reps.index_select(0,idx)+bias
   if relu:ref=ref.relu()
   close(out,ref,'reconstruct')
   go=torch.randn_like(out)
   a=torch.autograd.grad(out,(reps,bias),go);b=torch.autograd.grad(ref,(reps,bias),go)
   close(a[0],b[0],'reconstruct_grad');close(a[1],b[1],'reconstruct_bias_grad')
reps=torch.randn(5,3,device='cuda',dtype=torch.double,requires_grad=True);bias=torch.full((3,),2.,device='cuda',dtype=torch.double,requires_grad=True)
assert torch.autograd.gradcheck(lambda a,b:reconstruct(a,idx,b,'relu'),(reps,bias),eps=1e-6,atol=1e-5,rtol=1e-3);checks.append('reconstruct_gradcheck')
# Epoch reuse must rebuild means and upstream gradients from current features.
for device in ['cpu','cuda']:
 c=Compressgnn_Cluster(7,4,device=device,refresh_interval=5)
 ptr=None
 for epoch in range(12):
  x=torch.randn(19,7,device=device,requires_grad=True);out,index=c(x,epoch=epoch,cache_key='graph-A')
  assert c.last_refreshed==(epoch%5==0)
  if epoch%5:assert index.data_ptr()==ptr
  ptr=index.data_ptr();ref=scatter_mean(x,index,dim=0,dim_size=out.size(0));close(out,ref,'reuse_current_mean')
  close(torch.autograd.grad(out.square().sum(),x)[0],torch.autograd.grad(ref.square().sum(),x)[0],'reuse_current_gradient')
 assert c.refresh_count==3;checks.append('q5_refresh_schedule')
 c.reset_cache();c.set_epoch(0);x=torch.randn(19,7,device=device,requires_grad=True)
 train_out,train_index=c(x);train_ptr=train_index.data_ptr();c.eval();eval_out,eval_index=c(-x)
 assert eval_index.data_ptr()!=train_ptr;c.train();out,index=c(x,epoch=1);assert not c.last_refreshed and index.data_ptr()==train_ptr;checks.append('separate_train_eval_cache')
 c(x,epoch=1,cache_key='different-order');assert c.last_refreshed;checks.append('cache_key_invalidation')
 c(torch.randn(20,7,device=device),epoch=1);assert c.last_refreshed;checks.append('shape_invalidation')
 c.random_vectors.add_(.01);c(x,epoch=1);assert c.last_refreshed;checks.append('plane_invalidation')
 c.load_state_dict(c.state_dict());c(x,epoch=1);assert c.last_refreshed;checks.append('load_invalidation')
 c.double();c(x.double(),epoch=1);assert c.last_refreshed;checks.append('dtype_invalidation')
 # Refresh workspaces after an earlier forward must not corrupt its saved indices/counts.
 c=c.float();a=torch.randn(19,7,device=device,requires_grad=True);b=torch.randn(19,7,device=device,requires_grad=True)
 ya,ia=c(a,epoch=5);yb,ib=c(b,epoch=10)
 ra=scatter_mean(a,ia,dim=0,dim_size=ya.size(0));rb=scatter_mean(b,ib,dim=0,dim_size=yb.size(0))
 ga=torch.autograd.grad(ya.square().sum()+yb.square().sum(),(a,b));gb=torch.autograd.grad(ra.square().sum()+rb.square().sum(),(a,b))
 close(ga[0],gb[0],'saved_assignment_lifetime');close(ga[1],gb[1],'new_assignment_gradient')
 # Level-1 reuse: two shared transforms, one reconstruction, all parameter grads.
 layers=torch.nn.Sequential(torch.nn.Linear(7,9),torch.nn.ReLU(),torch.nn.Linear(9,3)).to(device)
 x=torch.randn(19,7,device=device,requires_grad=True);rep,ii=c(x,epoch=15)
 out=reconstruct(layers(rep),ii);ref=scatter_mean(x,ii,dim=0,dim_size=rep.size(0));ref=layers(ref).index_select(0,ii)
 close(out,ref,'consecutive_transform_reuse')
 params=[x]+list(layers.parameters());aa=torch.autograd.grad(out.square().sum(),params);bb=torch.autograd.grad(ref.square().sum(),params)
 for a,b in zip(aa,bb):close(a,b,'consecutive_transform_gradients')
# Explicit epoch units, loss policy, and immutable-state restoration.
c=Compressgnn_Cluster(7,4,device='cuda',refresh_interval=5)
try:c(torch.zeros(10,7,device='cuda'))
except ValueError:checks.append('epoch_required')
else:raise AssertionError('Implicit call count used as epoch')
a=Compressgnn_Cluster(7,4,device='cuda',refresh_policy='loss',q_min=1,q_max=10,tau_loss=.1)
a.set_epoch(0,loss=1.);a.set_epoch(1,loss=1.);assert a.current_interval==10
a.set_epoch(2,loss=2.);assert a.current_interval==1;checks.append('paper_loss_interval')
x=torch.randn(10,7,device='cuda',requires_grad=True)
y,ii,active=Compressgnn_Cluster_Function.apply(x,a.random_vectors,4,True)
ref=scatter_mean(x,ii,dim=0,dim_size=active+1)
close(y,ref,'legacy_entry_mean');close(torch.autograd.grad(y.sum(),x)[0],torch.autograd.grad(ref.sum(),x)[0],'legacy_entry_grad')
for invalid in [-1,31]:
 try:Compressgnn_Cluster(7,invalid)
 except ValueError:checks.append('invalid_bits')
 else:raise AssertionError('Bad bits accepted')
try:native.cluster_assign(x,torch.randn(7,3,device='cuda'),4,native.cluster_workspace(x,4))
except RuntimeError:checks.append('invalid_planes')
else:raise AssertionError('Bad plane shape accepted')
torch.cuda.synchronize()
print('PASSED', len(checks), flush=True)

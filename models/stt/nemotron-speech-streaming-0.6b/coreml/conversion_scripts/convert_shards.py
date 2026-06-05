import os
os.environ["PATH"]="/usr/bin:/bin:/usr/sbin:/sbin:"+os.environ.get("PATH","")
import sys; sys.path.insert(0,".")
import torch, numpy as np, coremltools as ct
import nemo.collections.asr as nemo_asr
from shard_encoder import HeadShard, BodyShard
OUT="/Users/hanweng/Documents/parakeet-tdt-opt/nemotron_en/shards_2240ms"
os.makedirs(OUT, exist_ok=True)
m=nemo_asr.models.EncDecRNNTBPEModel.from_pretrained("nvidia/nemotron-speech-streaming-en-0.6b",map_location="cpu").eval()
enc=m.encoder; enc.setup_streaming_params()
cc,ctt,cl=enc.get_initial_cache_state(batch_size=1,device="cpu"); cl=cl.to(torch.int32)
cc_b,ct_b=cc.transpose(0,1),ctt.transpose(0,1)
tmf=233; mel=torch.randn(1,128,tmf); mlen=torch.tensor([tmf],dtype=torch.int32)
splits=[0,6,12,18,24]
def conv(mod, name, example, names_in, names_out):
    mod=mod.eval()
    tr=torch.jit.trace(mod, example, strict=False)
    inputs=[ct.TensorType(name=n, shape=tuple(e.shape), dtype=np.float32 if e.dtype==torch.float32 else np.int32) for n,e in zip(names_in,example)]
    outs=[ct.TensorType(name=n) for n in names_out]
    mm=ct.convert(tr, inputs=inputs, outputs=outs, minimum_deployment_target=ct.target.iOS17, compute_precision=ct.precision.FLOAT16, compute_units=ct.ComputeUnit.CPU_AND_NE)
    p=f"{OUT}/{name}.mlpackage"; mm.save(p); print("saved",name)
    return mm
# head
head=HeadShard(enc,6)
hex=(mel,mlen,cc_b[:,0:6],ct_b[:,0:6],cl)
with torch.no_grad(): hout=head(*hex)
a,pos,att,pad,h_cc,h_ct,clo=hout
conv(head,"shard0_head",hex,
     ["features","length","cache_ch","cache_t","cache_len"],
     ["hidden","pos_emb","att_mask","pad_mask","cache_ch_out","cache_t_out","cache_len_out"])
print("intermediates: hidden",tuple(a.shape),"pos",tuple(pos.shape),"att",tuple(att.shape),"pad",tuple(pad.shape))
# bodies
cur=a
for bi in range(1,4):
    s,e=splits[bi],splits[bi+1]
    body=BodyShard(enc,s,e,is_tail=(bi==3))
    bex=(cur,pos,att,pad,cc_b[:,s:e],ct_b[:,s:e])
    with torch.no_grad(): bo=body(*bex)
    cur=bo[0]
    nm="shard%d_%s"%(bi,"tail" if bi==3 else "body")
    conv(body,nm,bex,
         ["hidden","pos_emb","att_mask","pad_mask","cache_ch","cache_t"],
         ["encoded" if bi==3 else "hidden_out","cache_ch_out","cache_t_out"])
print("DONE all shards")

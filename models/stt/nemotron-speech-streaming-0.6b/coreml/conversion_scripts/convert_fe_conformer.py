import os
os.environ["PATH"]="/usr/bin:/bin:/usr/sbin:/sbin:"+os.environ.get("PATH","")
import sys; sys.path.insert(0,".")
import torch, numpy as np, coremltools as ct
import nemo.collections.asr as nemo_asr
from shard_encoder import FrontEndShard, BodyShard
OUT="/Users/hanweng/Documents/parakeet-tdt-opt/nemotron_en/fe_conformer_2240ms"
os.makedirs(OUT,exist_ok=True)
m=nemo_asr.models.EncDecRNNTBPEModel.from_pretrained("nvidia/nemotron-speech-streaming-en-0.6b",map_location="cpu").eval()
enc=m.encoder; enc.setup_streaming_params()
cc,ctt,cl=enc.get_initial_cache_state(batch_size=1,device="cpu"); cl=cl.to(torch.int32)
cc_b,ct_b=cc.transpose(0,1),ctt.transpose(0,1)
tmf=233; mel=torch.randn(1,128,tmf); mlen=torch.tensor([tmf],dtype=torch.int32)
fe=FrontEndShard(enc).eval()
with torch.no_grad(): a,pos,att,pad,clo=fe(mel,mlen,cl)
conf=BodyShard(enc,0,24,is_tail=True).eval()
with torch.no_grad(): enc_out,cco,cto=conf(a,pos,att,pad,cc_b,ct_b)
def cv(mod,name,ex,nin,nout,unit):
    tr=torch.jit.trace(mod.eval(),ex,strict=False)
    inputs=[ct.TensorType(name=n,shape=tuple(e.shape),dtype=np.float32 if e.dtype==torch.float32 else np.int32) for n,e in zip(nin,ex)]
    mm=ct.convert(tr,inputs=inputs,outputs=[ct.TensorType(name=n) for n in nout],
        minimum_deployment_target=ct.target.iOS17,compute_precision=ct.precision.FLOAT16,compute_units=unit)
    mm.save(f"{OUT}/{name}.mlpackage"); print("saved",name)
cv(fe,"frontend",(mel,mlen,cl),["features","length","cache_len"],
   ["hidden","pos_emb","att_mask","pad_mask","cache_len_out"], ct.ComputeUnit.CPU_ONLY)
cv(conf,"conformer",(a,pos,att,pad,cc_b,ct_b),["hidden","pos_emb","att_mask","pad_mask","cache_ch","cache_t"],
   ["encoded","cache_ch_out","cache_t_out"], ct.ComputeUnit.CPU_AND_NE)
print("DONE")

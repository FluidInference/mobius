import os
os.environ["PATH"]="/usr/bin:/bin:/usr/sbin:/sbin:"+os.environ.get("PATH","")
import argparse, glob, time
from pathlib import Path
import coremltools as ct, numpy as np, soundfile as sf
from test_coreml_streaming import NemotronCoreMLStreaming, load_ground_truth, compute_wer

class ShardedStreaming(NemotronCoreMLStreaming):
    def load_all(self, model_dir, shard_dir, suffix):
        md=Path(model_dir); sd=Path(shard_dir); cu=ct.ComputeUnit.CPU_AND_NE
        self.preprocessor=ct.models.MLModel(str(md/"preprocessor.mlpackage"),compute_units=cu)
        self.fused=ct.models.MLModel(str(md/"decoder_joint.mlpackage"),compute_units=cu)
        self.head=ct.models.MLModel(str(sd/f"shard0_head{suffix}.mlpackage"),compute_units=cu)
        self.bodies=[ct.models.MLModel(str(sd/f"shard{i}_{'tail' if i==3 else 'body'}{suffix}.mlpackage"),compute_units=cu) for i in (1,2,3)]
    def encode(self, input_mel, cache_channel, cache_time, cache_len):
        mlen=np.array([self.total_mel_frames],dtype=np.int32)
        o=self.head.predict({"features":input_mel.astype(np.float32),"length":mlen,
            "cache_ch":cache_channel[:,0:6],"cache_t":cache_time[:,0:6],"cache_len":cache_len})
        h=o["hidden"];pos=o["pos_emb"];att=o["att_mask"];pad=o["pad_mask"]
        ccp=[o["cache_ch_out"]];ctp=[o["cache_t_out"]];clo=o["cache_len_out"]
        for bi in range(1,4):
            s,e=bi*6,(bi+1)*6
            ob=self.bodies[bi-1].predict({"hidden":h,"pos_emb":pos,"att_mask":att,"pad_mask":pad,
                "cache_ch":cache_channel[:,s:e],"cache_t":cache_time[:,s:e]})
            h=ob.get("encoded",ob.get("hidden_out")); ccp.append(ob["cache_ch_out"]); ctp.append(ob["cache_t_out"])
        return h, np.concatenate(ccp,axis=1), np.concatenate(ctp,axis=1), clo
    def transcribe_streaming(self, audio):
        audio=audio.astype(np.float32); total=len(audio)
        cc,ct,cl=self._get_initial_cache(); h,c=self._get_initial_decoder_state()
        last=self.blank_idx; toks=[]; mel_cache=None; off=0
        while off<total:
            chunk=audio[off:min(off+self.chunk_samples,total)]
            if len(chunk)<self.chunk_samples: chunk=np.pad(chunk,(0,self.chunk_samples-len(chunk)))
            pre=self.preprocessor.predict({"audio":chunk.reshape(1,-1),"audio_length":np.array([len(chunk)],dtype=np.int32)})
            cmel=pre["mel"]
            imel=np.concatenate([mel_cache,cmel],axis=2) if mel_cache is not None else np.pad(cmel,((0,0),(0,0),(self.pre_encode_cache,0)),mode="constant")
            cf=imel.shape[2]
            if cf<self.total_mel_frames: imel=np.pad(imel,((0,0),(0,0),(0,self.total_mel_frames-cf)),mode="constant")
            elif cf>self.total_mel_frames: imel=imel[:,:,:self.total_mel_frames]
            mel_cache=cmel[:,:,-self.pre_encode_cache:] if cmel.shape[2]>=self.pre_encode_cache else cmel
            encoded,cc,ct,cl=self.encode(imel,cc,ct,cl)
            for t in range(encoded.shape[2]):
                es=encoded[:,:,t:t+1].astype(np.float32)
                for _ in range(10):
                    out=self.fused.predict({"token":np.array([[last]],dtype=np.int32),"token_length":np.array([1],dtype=np.int32),"h_in":h,"c_in":c,"encoder":es})
                    p=int(np.argmax(out["logits"][0,0,0,:]))
                    if p==self.blank_idx: break
                    toks.append(p); last=p; h,c=out["h_out"],out["c_out"]
            off+=self.chunk_samples
        return self._decode_tokens(toks)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model-dir",required=True); ap.add_argument("--shard-dir",required=True)
    ap.add_argument("--suffix",default="_pal6"); ap.add_argument("--dataset",required=True)
    ap.add_argument("--num-files",type=int,default=100); ap.add_argument("--warmup",type=int,default=3)
    a=ap.parse_args()
    inf=ShardedStreaming(a.model_dir); inf.load_all(a.model_dir,a.shard_dir,a.suffix)
    gt=load_ground_truth(a.dataset); files=sorted(glob.glob(f"{a.dataset}/**/*.flac",recursive=True))[:a.num_files+a.warmup]
    te=tw=0; aud=comp=0.0; n=0
    for i,p in enumerate(files):
        au,sr=sf.read(p,dtype="float32"); t0=time.perf_counter(); hyp=inf.transcribe_streaming(au); dt=time.perf_counter()-t0
        if i<a.warmup: continue
        n+=1; aud+=len(au)/sr; comp+=dt
        if Path(p).stem in gt: e,w=compute_wer(gt[Path(p).stem],hyp); te+=e; tw+=w
    print(f"SHARDED+B1  chunk={inf.chunk_mel_frames}  WER={100*te/max(tw,1):.2f}%  RTFx={aud/comp:.1f}")
main()

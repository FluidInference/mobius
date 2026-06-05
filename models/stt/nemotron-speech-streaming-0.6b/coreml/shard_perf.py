import os
os.environ["PATH"]="/usr/bin:/bin:/usr/sbin:/sbin:"+os.environ.get("PATH","")
import time, json, resource, numpy as np, coremltools as ct
SH="/Users/hanweng/Documents/parakeet-tdt-opt/nemotron_en/shards_2240ms"
SINGLE="/Users/hanweng/Documents/parakeet-tdt-opt/nemotron_en/coreml_2240ms/encoder.mlpackage"
md=json.load(open("/Users/hanweng/Documents/parakeet-tdt-opt/nemotron_en/coreml_2240ms/metadata.json"))
tmf=md["total_mel_frames"]; cc=md["cache_channel_shape"]; ctt=md["cache_time_shape"]  # [1,24,70,1024],[1,24,1024,8]
U=ct.ComputeUnit.CPU_AND_NE
mel=np.random.randn(1,128,tmf).astype(np.float32); mlen=np.array([tmf],dtype=np.int32)
cache_ch=np.random.randn(*cc).astype(np.float32); cache_t=np.random.randn(*ctt).astype(np.float32); clen=np.array([0],dtype=np.int32)

# --- single encoder ---
t0=time.perf_counter(); single=ct.models.MLModel(SINGLE,compute_units=U); single_load=time.perf_counter()-t0
sfeed={"mel":mel,"mel_length":mlen,"cache_channel":cache_ch,"cache_time":cache_t,"cache_len":clen}
for _ in range(3): single.predict(sfeed)
t=[time.perf_counter() for _ in [0]]; 
import statistics
ts=[]
for _ in range(15):
    s=time.perf_counter(); so=single.predict(sfeed); ts.append((time.perf_counter()-s)*1000)
single_inf=sorted(ts)[len(ts)//2]
enc_single=np.array(so["encoded"])

# --- shards ---
names=["shard0_head","shard1_body","shard2_body","shard3_tail"]
loads=[]; shards=[]
for n in names:
    t0=time.perf_counter(); mm=ct.models.MLModel(f"{SH}/{n}.mlpackage",compute_units=U); loads.append(time.perf_counter()-t0); shards.append(mm)
def run_shards():
    o=shards[0].predict({"features":mel,"length":mlen,"cache_ch":cache_ch[:,0:6],"cache_t":cache_t[:,0:6],"cache_len":clen})
    h=o["hidden"]; pos=o["pos_emb"]; att=o["att_mask"]; pad=o["pad_mask"]
    for bi in range(1,4):
        s,e=bi*6,(bi+1)*6
        fd={"hidden":h,"pos_emb":pos,"att_mask":att,"pad_mask":pad,"cache_ch":cache_ch[:,s:e],"cache_t":cache_t[:,s:e]}
        o=shards[bi].predict(fd)
        h=o.get("encoded", o.get("hidden_out"))
    return h
for _ in range(3): run_shards()
ts=[]
for _ in range(15):
    s=time.perf_counter(); enc_shard=run_shards(); ts.append((time.perf_counter()-s)*1000)
shard_inf=sorted(ts)[len(ts)//2]
enc_shard=np.array(enc_shard)
peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024*1024)

print(f"SINGLE  load={single_load:.2f}s  infer={single_inf:.2f}ms")
print(f"SHARDS  load_total={sum(loads):.2f}s  per_shard={[round(x,2) for x in loads]}  infer_total={shard_inf:.2f}ms")
print(f"PARITY  encoded max|Δ|={np.abs(enc_single-enc_shard).max():.6e}  shape {enc_single.shape}")
print(f"peak_rss (both loaded) = {peak:.0f}MB")

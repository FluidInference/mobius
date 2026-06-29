import sys, resource, numpy as np, soundfile as sf
from transcribe_ov import NemotronOV
d=sys.argv[1]
m=NemotronOV(d, device="CPU")
a,sr=sf.read("sample.wav",dtype="float32")
if a.ndim>1: a=a.mean(1)
_=m.transcribe_streaming(np.asarray(a,dtype=np.float32), target_lang="auto")
peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
print(f"{d}: peak RSS {peak:.0f} MB")

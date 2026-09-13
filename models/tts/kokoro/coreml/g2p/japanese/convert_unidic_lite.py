"""Trim a MeCab dictionary (unidic-lite) to the fields the Kokoro Japanese frontend needs.

Keeps the double array and token table byte-for-byte; rewrites the feature blob so each
entry's feature string is `pos1,pron,kana` (UniDic indices 0, 9, 17). Applies the same
trim to unk.dic. matrix.bin and char.bin are copied unchanged. Output layout is the
standard MeCab sys.dic layout, so the same reader handles both.
"""
import struct, os, sys, shutil
def trim(src, dst, fields=(0, 9, 17)):
    b = open(src, "rb").read()
    head = list(struct.unpack("<10I", b[:40])); charset = b[40:72]
    lexsize, lsize, rsize, dsize, tsize, fsize = head[3:9]
    off = 72
    darts = b[off:off+dsize]; off += dsize
    tokens = bytearray(b[off:off+tsize]); off += tsize
    feats = b[off:off+fsize]
    new_feats = bytearray(); cache = {}
    for i in range(tsize // 16):
        lc, rc, posid, wcost, foff, comp = struct.unpack_from("<HHHhII", tokens, i*16)
        end = feats.index(b"\0", foff); f = feats[foff:end].decode("utf-8")
        parts = f.split(",")
        # UniDic quotes fields containing commas ("4,0"); pron/kana never do, pos1 never does.
        trimmed = ",".join(parts[k] if k < len(parts) else "*" for k in fields)
        noff = cache.get(trimmed)
        if noff is None:
            noff = len(new_feats); new_feats += trimmed.encode("utf-8") + b"\0"; cache[trimmed] = noff
        struct.pack_into("<HHHhII", tokens, i*16, lc, rc, posid, wcost, noff, comp)
    head[8] = len(new_feats)
    with open(dst, "wb") as o:
        o.write(struct.pack("<10I", *head)); o.write(charset); o.write(darts); o.write(bytes(tokens)); o.write(bytes(new_feats))
    return os.path.getsize(src), os.path.getsize(dst), len(cache)
if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]; os.makedirs(dst, exist_ok=True)
    for name in ("sys.dic", "unk.dic"):
        a, b_, n = trim(f"{src}/{name}", f"{dst}/{name}"); print(f"{name}: {a/1e6:.1f} MB -> {b_/1e6:.1f} MB ({n} unique features)")
    for name in ("matrix.bin", "char.bin"):
        shutil.copy(f"{src}/{name}", f"{dst}/{name}"); print(f"{name}: {os.path.getsize(f'{dst}/{name}')/1e6:.1f} MB (copied)")

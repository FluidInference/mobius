"""Reference MeCab-format reader + Viterbi, to validate the dictionary format before the Swift port.
Reads unidic-lite's sys.dic / matrix.bin / char.bin / unk.dic directly."""
import struct, os, sys, unicodedata
class Dic:
    def __init__(self, path):
        b = open(path, "rb").read()
        (self.magic, self.version, self.type, self.lexsize, self.lsize, self.rsize,
         self.dsize, self.tsize, self.fsize, _) = struct.unpack("<10I", b[:40])
        off = 72
        self.darts = b[off:off+self.dsize]; off += self.dsize
        self.tokens = b[off:off+self.tsize]; off += self.tsize
        self.features = b[off:off+self.fsize]
    def unit(self, i):
        base, check = struct.unpack_from("<iI", self.darts, i*8); return base, check
    def common_prefix_search(self, key: bytes):
        """MeCab Darts: returns list of (value, length)."""
        out = []
        b, _ = self.unit(0); node = 0
        n = len(key)
        for i in range(n):
            p = b  # check terminal at current node (b ^ 0)
            nb, nc = self.unit(p)
            if b == nc and nb < 0:
                out.append((-nb-1, i))
            p = b + key[i] + 1
            nb, nc = self.unit(p)
            if b == nc:
                b = nb
            else:
                return out
        p = b; nb, nc = self.unit(p)
        if b == nc and nb < 0:
            out.append((-nb-1, n))
        return out
    def token(self, idx):
        lc, rc, posid, wcost, foff, comp = struct.unpack_from("<HHHhII", self.tokens, idx*16)
        return lc, rc, posid, wcost, foff
    def feature(self, foff):
        end = self.features.index(b"\0", foff); return self.features[foff:end].decode("utf-8")
    def lookup(self, key: bytes):
        """yield (length_bytes, lid, rid, cost, feature) for all dictionary entries prefixing key"""
        for value, length in self.common_prefix_search(key):
            idx, count = value >> 8, value & 0xff
            for k in range(count):
                lc, rc, posid, wcost, foff = self.token(idx + k)
                yield length, lc, rc, wcost, foff
class CharInfo:
    def __init__(self, path):
        b = open(path, "rb").read()
        n = struct.unpack_from("<I", b, 0)[0]
        self.names = [b[4+i*32:4+(i+1)*32].split(b"\0")[0].decode() for i in range(n)]
        self.map = b[4+n*32:]
    def info(self, cp):
        if cp > 0xFFFF: cp = 0  # MeCab maps > BMP to DEFAULT? (it uses the last code point info); keep simple
        v = struct.unpack_from("<I", self.map, cp*4)[0]
        return dict(type=v & 0x3FFFF, default_type=(v >> 18) & 0xFF, length=(v >> 26) & 0xF, group=(v >> 30) & 1, invoke=(v >> 31) & 1)
class Matrix:
    def __init__(self, path):
        b = open(path, "rb").read(); self.l, self.r = struct.unpack_from("<HH", b, 0); self.b = b
    def cost(self, rid_prev, lid_next):
        return struct.unpack_from("<h", self.b, 4 + (lid_next * self.l + rid_prev) * 2)[0]  # MeCab: matrix_[rcAttr + lsize * lcAttr]
def tokenize(text, dic, unk, chars, matrix, fields=None):
    raw = text.encode("utf-8"); n = len(raw)
    cps = []  # (byte_offset, byte_len, codepoint)
    i = 0
    while i < n:
        c = raw[i]; l = 1 if c < 0x80 else 2 if c < 0xE0 else 3 if c < 0xF0 else 4
        cps.append((i, l, raw[i:i+l].decode("utf-8"))); i += l
    starts = {off: k for k, (off, _, _) in enumerate(cps)}
    INF = 1 << 60
    best = [(INF, None)] * (n + 1); best[0] = (0, None)  # per byte position: (cost, node)
    nodes_at = {0: [(0, 0, 0, None, None)]}  # end_pos -> list of (cost, rid, lid?, feature, backptr)
    # node: (end, cost_so_far, rid, start, feature_off, is_unk, char_type, prev_node)
    ends = {0: [(0, 0, None, None, None, None, None, None)]}
    for k, (off, l, ch) in enumerate(cps):
        if off not in ends: continue
        prev_nodes = ends[off]
        key = raw[off:]
        cands = []
        for length, lc, rc, wcost, foff in dic.lookup(key):
            cands.append((length, lc, rc, wcost, dic.feature(foff), False))
        ci = chars.info(ord(ch))
        cat = ci["default_type"]; group = ci["group"]; invoke = ci["invoke"]; maxlen = ci["length"]
        # unknown word candidates (MeCab: invoke if no dictionary hit or invoke flag)
        if not cands or invoke:
            cat_id = ci["default_type"]
            name = chars.names[cat_id]
            uentries = list(unk.lookup(name.encode()))
            lens = set()
            if group:
                # extend while the same category bit is set
                j = k; total = 0
                while j < len(cps) and (chars.info(ord(cps[j][2]))["type"] >> cat_id) & 1:
                    total += cps[j][1]; j += 1
                    if maxlen and (j - k) > maxlen: break
                lens.add(total)
            for m in range(1, (maxlen or 0) + 1):
                if k + m <= len(cps) and all((chars.info(ord(cps[k+t][2]))["type"] >> cat_id) & 1 for t in range(m)):
                    lens.add(sum(cps[k+t][1] for t in range(m)))
            if not lens: lens.add(l)
            for length in lens:
                for _, lc, rc, wcost, foff in uentries:
                    cands.append((length, lc, rc, wcost, unk.feature(foff), True))
        for length, lc, rc, wcost, feat, is_unk in cands:
            end = off + length
            bestc, bestp = INF, None
            for pn in prev_nodes:
                pcost, prid = pn[1], pn[2]
                c = pcost + wcost + (matrix.cost(prid, lc) if prid is not None else 0)
                if c < bestc: bestc, bestp = c, pn
            node = (end, bestc, rc, off, feat, is_unk, cat, bestp)
            ends.setdefault(end, []).append(node)
    # EOS
    final = min(ends.get(n, []), key=lambda nd: nd[1] + matrix.cost(nd[2], 0), default=None)
    out = []
    nd = final
    while nd is not None and nd[3] is not None:
        out.append(nd); nd = nd[7]
    out.reverse()
    return [(raw[nd[3]:nd[0]].decode(), nd[4], nd[5], nd[6]) for nd in out]
if __name__ == "__main__":
    base = sys.argv[1]
    dic = Dic(f"{base}/sys.dic"); unk = Dic(f"{base}/unk.dic"); chars = CharInfo(f"{base}/char.bin"); matrix = Matrix(f"{base}/matrix.bin")
    from fugashi import Tagger
    tagger = Tagger()
    mism = 0
    for text in sys.argv[2:]:
        toks = tokenize(text, dic, unk, chars, matrix)
        ref = [(w.surface, w.feature.pron or "", w.feature.kana or "", w.is_unk, w.char_type) for w in tagger(text)]
        mine = []
        for s, f, u, ct in toks:
            fs = f.split(","); pron, kana = (fs[9], fs[17]) if len(fs) > 17 else (fs[1], fs[2]) if len(fs) == 3 else ("", "")
            mine.append((s, pron, kana, u, ct))
        ok = mine == ref
        print(("SAME " if ok else "DIFF ") + text)
        if not ok:
            mism += 1; print("   mine:", mine); print("   ref :", ref)
    print("mismatching sentences:", mism, "of", len(sys.argv) - 2)

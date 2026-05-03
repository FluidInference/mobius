"""Benchmark CoreML pipeline (v1.1-zh): per-stage latency + speed over varying lengths.

Usage:
    uv run python benchmark.py --models-dir build/kokoro-v1.1-zh [--n-runs 5]

For each Mandarin test sentence, runs the 7-model chain end-to-end, reports
per-stage latency (median over n-runs after warmup), end-to-end chain time,
audio duration, and speed multiplier (audio_duration / chain_time, in
× real-time).
"""
import argparse
import pathlib
import statistics
import time

import coremltools as ct
import numpy as np
import torch

from kokoro import KModel
from kokoro.pipeline import KPipeline

SR = 24000

# Real Mandarin prose passages — varied prosody, punctuation, tones, and content.
# Run through misaki[zh] G2P (Bopomofo + tone digits) to get realistic phoneme
# distributions (vs synthetic repetitions which underestimate duration variance).
TEXTS = [
    # ~1 short sentence
    "你好。",
    # ~1 medium sentence
    "今天的天气真不错，阳光明媚，微风拂面。",
    # ~3 sentences, conversational
    "她已经等了将近一个小时。公交车一如既往地迟到了。"
    "一阵冷风吹来，吹动了她脚边的落叶。",
    # ~1 short paragraph, narrative
    "在安静的小镇米尔布鲁克，消息传播得比风还快。"
    "等警长到达现场时，半数居民已经聚集在那里，"
    "低声议论着各种猜测，交换着模糊的事实。"
    "没人真正知道发生了什么，但每个人都有自己的看法。",
    # ~Mid-length expository with numbers and punctuation
    "探险队于一九二三年三月十四日出发，共有十二名队员，三辆雪橇，"
    "以及足够六十天的补给。两周内，一场突如其来的暴风雪使先头部队与"
    "补给队失去联系；本应是常规的穿越，变成了求生的殊死搏斗。"
    "当救援到达时，原队中只剩下四人，他们的日记后来被誉为"
    "极地探险的非凡记录。",
    # ~Long passage, mixed register, sized to land near ALBERT's 510-phoneme cap
    "人类飞行的历史，在许多方面，是一部固执拒绝的历史。"
    "几千年来，观察者们看着鸟儿轻盈地翱翔在头顶，断定人类生来"
    "并不是为了与它们一同飞翔。然而，发明家们坚持不懈。"
    "他们用羽毛和蜡制作翅膀，用丝绸和竹子制作滑翔机，"
    "用氢气、氦气，甚至烟火加热的热空气填充气球。"
    "每一次失败都让他们对升力、阻力，以及流体在这些弯曲表面上的"
    "奇特行为有了新的认识。",
]


def phonemize_for_benchmark(pipe, text):
    """Run misaki[zh] G2P on Mandarin text via the Kokoro pipeline; return the
    full phoneme string (Bopomofo + tone digits). Truncates at 510 phonemes
    (ALBERT's hard limit) so the longest passage naturally lands at the
    model ceiling.

    misaki[zh] doesn't expose `tokens` via `pipe.g2p` (returns (text, None)),
    so we drive the pipeline iterator and concatenate the per-chunk `ps`.
    """
    parts = []
    for _gs, ps, _tks in pipe(text, voice='zf_001'):
        if ps:
            parts.append(ps)
    ps = ''.join(parts).strip()
    if len(ps) > 510:
        ps = ps[:510]
    return ps


def build_sentences(pipe):
    return [phonemize_for_benchmark(pipe, t) for t in TEXTS]

# RangeDim cap baked into the converted models (must match convert.py --max-frames).
MAX_FRAMES = 2000


def encode_phonemes(model, phonemes):
    ids = list(filter(lambda i: i is not None, map(lambda p: model.vocab.get(p), phonemes)))
    return torch.LongTensor([[0, *ids, 0]])


def time_call(fn, n_runs):
    """Run fn n_runs times after 2 warmups, return (median_ms, all_runs_ms, last_result)."""
    fn(); fn()  # warmup
    samples = []
    last = None
    for _ in range(n_runs):
        t0 = time.perf_counter()
        last = fn()
        samples.append((time.perf_counter() - t0) * 1000)
    return statistics.median(samples), samples, last


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models-dir', type=pathlib.Path, required=True,
                        help='Directory containing the 7 .mlpackage files (output of convert-coreml.py)')
    parser.add_argument('--n-runs', type=int, default=5)
    parser.add_argument('--voice', default='zf_001', help='Voice id (default: zf_001)')
    parser.add_argument('--lang', default='z', help='Kokoro lang code (default: z = Mandarin)')
    parser.add_argument('--repo-id', default='hexgrad/Kokoro-82M-v1.1-zh',
                        help='HF repo id (default: hexgrad/Kokoro-82M-v1.1-zh)')
    args = parser.parse_args()
    models_dir = args.models_dir

    print(f'[setup] Loading PyTorch model + voice pack ({args.repo_id}, voice={args.voice})...')
    model = KModel(repo_id=args.repo_id); model.eval()
    pipe = KPipeline(lang_code=args.lang, repo_id=args.repo_id, model=model)
    voice_pack = pipe.load_voice(args.voice)

    print('[setup] Loading CoreML models...')
    CU_NE = ct.ComputeUnit.CPU_AND_NE
    CU_ALL = ct.ComputeUnit.ALL
    m_albert = ct.models.MLModel(str(models_dir / 'KokoroAlbert.mlpackage'),    compute_units=CU_NE)
    m_post   = ct.models.MLModel(str(models_dir / 'KokoroPostAlbert.mlpackage'), compute_units=CU_NE)
    m_align  = ct.models.MLModel(str(models_dir / 'KokoroAlignment.mlpackage'),  compute_units=CU_NE)
    m_pros   = ct.models.MLModel(str(models_dir / 'KokoroProsody.mlpackage'),    compute_units=CU_NE)
    m_noise  = ct.models.MLModel(str(models_dir / 'KokoroNoise.mlpackage'),      compute_units=CU_ALL)
    m_voc    = ct.models.MLModel(str(models_dir / 'KokoroVocoder.mlpackage'),    compute_units=CU_NE)
    m_tail   = ct.models.MLModel(str(models_dir / 'KokoroTail.mlpackage'),       compute_units=CU_ALL)

    sentences = build_sentences(pipe)
    rows = []
    stage_rows = []

    for phonemes in sentences:
        input_ids = encode_phonemes(model, phonemes)
        T_enc = input_ids.shape[1]
        ref_s = voice_pack[max(min(len(phonemes) - 1, voice_pack.shape[0] - 1), 0)]
        s = ref_s[:, 128:]
        style_timbre = ref_s[:, :128]
        attention_mask = torch.ones(1, T_enc, dtype=torch.int32)

        albert_feed = {
            "input_ids": input_ids.numpy().astype(np.int32),
            "attention_mask": attention_mask.numpy().astype(np.int32),
        }
        # Per-stage feeds populated by chaining once
        o1 = m_albert.predict(albert_feed)
        # Probe T_a from PostAlbert duration to skip sentences exceeding RangeDim cap
        probe_post = m_post.predict({
            "bert_dur": np.array(o1["bert_dur"]).astype(np.float16),
            "input_ids": input_ids.numpy().astype(np.int32),
            "style_s": s.numpy().astype(np.float16),
            "speed": np.array([1.0], dtype=np.float16),
            "attention_mask": attention_mask.numpy().astype(np.int32),
        })
        T_a_probe = int(np.round(np.array(probe_post["duration"]).flatten()).clip(min=1).sum())
        if T_a_probe > MAX_FRAMES:
            print(f'\n  T_enc={T_enc:3d}  T_a={T_a_probe:4d}  SKIP (exceeds MAX_FRAMES={MAX_FRAMES})')
            continue
        post_feed = {
            "bert_dur": np.array(o1["bert_dur"]).astype(np.float16),
            "input_ids": input_ids.numpy().astype(np.int32),
            "style_s": s.numpy().astype(np.float16),
            "speed": np.array([1.0], dtype=np.float16),
            "attention_mask": attention_mask.numpy().astype(np.int32),
        }
        o2 = m_post.predict(post_feed)
        dur = np.array(o2["duration"]).flatten()
        pd = np.round(dur).clip(min=1).astype(np.int32).reshape(1, -1)
        align_feed = {
            "pred_dur": pd,
            "d": np.array(o2["d"]).astype(np.float16),
            "t_en": np.array(o2["t_en"]).astype(np.float16),
        }
        o3 = m_align.predict(align_feed)
        T_a = int(pd.sum())
        pros_feed = {
            "en": np.array(o3["en"]).astype(np.float16),
            "style_s": s.numpy().astype(np.float16),
        }
        o4 = m_pros.predict(pros_feed)
        noise_feed = {
            "F0_curve": np.array(o4["F0"]).astype(np.float32),
            "style_timbre": style_timbre.numpy().astype(np.float32),
        }
        o5 = m_noise.predict(noise_feed)
        voc_feed = {
            "asr": np.array(o3["asr"]).astype(np.float16),
            "F0_curve": np.array(o4["F0"]).astype(np.float16),
            "N_pred": np.array(o4["N"]).astype(np.float16),
            "x_source_0": np.array(o5["x_source_0"]).astype(np.float16),
            "x_source_1": np.array(o5["x_source_1"]).astype(np.float16),
            "style_timbre": style_timbre.numpy().astype(np.float16),
        }
        o6 = m_voc.predict(voc_feed)
        tail_feed = {"x_pre": np.array(o6["x_pre"]).astype(np.float32)}
        o7 = m_tail.predict(tail_feed)

        audio_samples = int(np.array(o7["audio"]).size)
        audio_dur_s = audio_samples / SR

        # Per-stage timings
        stages = [
            ('Albert',     lambda: m_albert.predict(albert_feed)),
            ('PostAlbert', lambda: m_post.predict(post_feed)),
            ('Alignment',  lambda: m_align.predict(align_feed)),
            ('Prosody',    lambda: m_pros.predict(pros_feed)),
            ('Noise',      lambda: m_noise.predict(noise_feed)),
            ('Vocoder',    lambda: m_voc.predict(voc_feed)),
            ('Tail',       lambda: m_tail.predict(tail_feed)),
        ]
        per_stage = {}
        for name, fn in stages:
            med, _, _ = time_call(fn, args.n_runs)
            per_stage[name] = med

        # Full chain
        def chain():
            a = m_albert.predict(albert_feed)
            b = m_post.predict({**post_feed, "bert_dur": np.array(a["bert_dur"]).astype(np.float16)})
            d_ = np.array(b["duration"]).flatten()
            pdur = np.round(d_).clip(min=1).astype(np.int32).reshape(1, -1)
            c = m_align.predict({"pred_dur": pdur,
                                  "d": np.array(b["d"]).astype(np.float16),
                                  "t_en": np.array(b["t_en"]).astype(np.float16)})
            d2 = m_pros.predict({"en": np.array(c["en"]).astype(np.float16),
                                  "style_s": s.numpy().astype(np.float16)})
            e = m_noise.predict({"F0_curve": np.array(d2["F0"]).astype(np.float32),
                                  "style_timbre": style_timbre.numpy().astype(np.float32)})
            f = m_voc.predict({"asr": np.array(c["asr"]).astype(np.float16),
                                "F0_curve": np.array(d2["F0"]).astype(np.float16),
                                "N_pred": np.array(d2["N"]).astype(np.float16),
                                "x_source_0": np.array(e["x_source_0"]).astype(np.float16),
                                "x_source_1": np.array(e["x_source_1"]).astype(np.float16),
                                "style_timbre": style_timbre.numpy().astype(np.float16)})
            return m_tail.predict({"x_pre": np.array(f["x_pre"]).astype(np.float32)})

        chain_med, _, _ = time_call(chain, args.n_runs)
        speedup = audio_dur_s / (chain_med / 1000)

        rows.append({
            'T_enc': T_enc,
            'T_a': T_a,
            'audio_s': audio_dur_s,
            'chain_ms': chain_med,
            'speedup': speedup,
        })
        stage_rows.append(per_stage)

        print(f'\n  T_enc={T_enc:3d}  T_a={T_a:4d}  audio={audio_dur_s:5.2f}s  '
              f'chain={chain_med:6.1f}ms  speed={speedup:5.1f}x')
        for name in ['Albert', 'PostAlbert', 'Alignment', 'Prosody', 'Noise', 'Vocoder', 'Tail']:
            print(f'    {name:11s} {per_stage[name]:6.2f}ms')

    # Summary table
    print(f'\n{"="*72}')
    print(f'{"T_enc":>6} {"T_a":>5} {"audio_s":>8} {"chain_ms":>10} {"speed":>7}  | per-stage ms')
    print(f'{"-"*72}')
    for r, ps in zip(rows, stage_rows):
        per = '  '.join(f'{n[:3]}={ps[n]:.0f}' for n in ['Albert','PostAlbert','Alignment','Prosody','Noise','Vocoder','Tail'])
        print(f'{r["T_enc"]:>6d} {r["T_a"]:>5d} {r["audio_s"]:>8.2f} {r["chain_ms"]:>10.1f} {r["speedup"]:>6.1f}x  | {per}')

    avg_speedup = statistics.mean(r['speedup'] for r in rows)
    print(f'\nMean speed: {avg_speedup:.1f}x real-time  (higher is faster; >1.0x = real-time)')


if __name__ == '__main__':
    main()

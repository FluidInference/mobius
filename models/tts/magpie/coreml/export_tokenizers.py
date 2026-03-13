"""Export all language tokenizer data from MagpieTTS for Swift inference.

Extracts from each language tokenizer in the AggregatedTTSTokenizer:
- token2id mapping (token string -> local ID)
- phoneme_dict (word -> phoneme list) for IPA G2P tokenizers
- heteronyms list for English
- tokenizer metadata (type, offset, locale, etc.)

Usage:
    python export_tokenizers.py [--nemo-path /path/to/model.nemo] [--output-dir constants]
"""
import argparse
import json
import os
import sys


def export_tokenizers(nemo_path=None, output_dir="constants"):
    print("Loading MagpieTTS model...")
    from nemo.collections.tts.models import MagpieTTSModel
    if nemo_path:
        model = MagpieTTSModel.restore_from(nemo_path, map_location="cpu")
    else:
        model = MagpieTTSModel.from_pretrained("nvidia/magpie_tts_multilingual_357m")
    model.eval()

    tok = model.tokenizer
    os.makedirs(output_dir, exist_ok=True)

    # Build master metadata
    metadata = {
        "tokenizer_names": tok.tokenizer_names,
        "offsets": {},
        "num_tokens": {},
        "types": {},
        "eos_token_id": model.eos_id,
    }

    for name in tok.tokenizer_names:
        sub_tok = tok.tokenizers[name]
        offset = tok.tokenizer_offsets[name]
        num_tokens = tok.num_tokens_per_tokenizer[name]
        tok_type = type(sub_tok).__name__

        metadata["offsets"][name] = offset
        metadata["num_tokens"][name] = num_tokens
        metadata["types"][name] = tok_type

        print(f"\n{'='*60}")
        print(f"Tokenizer: {name}")
        print(f"  Type: {tok_type}")
        print(f"  Offset: {offset}")
        print(f"  Num tokens: {num_tokens}")

        # Export token2id mapping
        if hasattr(sub_tok, '_token2id'):
            t2i = {k: v for k, v in sub_tok._token2id.items()}
            t2i_path = os.path.join(output_dir, f"{name}_token2id.json")
            with open(t2i_path, "w", encoding="utf-8") as f:
                json.dump(t2i, f, ensure_ascii=False, indent=None, separators=(',', ':'))
            size_kb = os.path.getsize(t2i_path) / 1024
            print(f"  token2id: {len(t2i)} entries ({size_kb:.1f} KB)")

        # Export phoneme dict for IPA G2P tokenizers
        if hasattr(sub_tok, 'g2p') and hasattr(sub_tok.g2p, 'phoneme_dict') and sub_tok.g2p.phoneme_dict is not None:
            g2p = sub_tok.g2p
            pd = {}
            for word, prons in g2p.phoneme_dict.items():
                # prons is List[List[str]] — take first pronunciation
                if isinstance(prons, list) and len(prons) > 0:
                    if isinstance(prons[0], list):
                        pd[word] = prons[0]
                    else:
                        pd[word] = prons
            pd_path = os.path.join(output_dir, f"{name}_phoneme_dict.json")
            with open(pd_path, "w", encoding="utf-8") as f:
                json.dump(pd, f, ensure_ascii=False, indent=None, separators=(',', ':'))
            size_mb = os.path.getsize(pd_path) / 1024 / 1024
            print(f"  phoneme_dict: {len(pd)} entries ({size_mb:.1f} MB)")

            # Export G2P metadata
            g2p_meta = {}
            if hasattr(g2p, 'locale'):
                g2p_meta['locale'] = g2p.locale
            if hasattr(g2p, 'grapheme_case'):
                g2p_meta['grapheme_case'] = g2p.grapheme_case
            if hasattr(g2p, 'grapheme_prefix'):
                g2p_meta['grapheme_prefix'] = g2p.grapheme_prefix
            if hasattr(g2p, 'use_stresses'):
                g2p_meta['use_stresses'] = g2p.use_stresses
            if hasattr(g2p, 'ignore_ambiguous_words'):
                g2p_meta['ignore_ambiguous_words'] = g2p.ignore_ambiguous_words
            if hasattr(g2p, 'phoneme_probability'):
                g2p_meta['phoneme_probability'] = g2p.phoneme_probability
            if g2p_meta:
                metadata.setdefault("g2p_config", {})[name] = g2p_meta
                print(f"  G2P config: {g2p_meta}")

            # Export heteronyms
            if hasattr(g2p, 'heteronyms') and g2p.heteronyms:
                het_path = os.path.join(output_dir, f"{name}_heteronyms.json")
                with open(het_path, "w", encoding="utf-8") as f:
                    json.dump(sorted(list(g2p.heteronyms)), f, ensure_ascii=False)
                print(f"  heteronyms: {len(g2p.heteronyms)} entries")

        # For Chinese G2P: export pinyin phoneme dict and tone/letter mappings
        if hasattr(sub_tok, 'g2p') and hasattr(sub_tok.g2p, 'tone_dict'):
            g2p = sub_tok.g2p
            # Pinyin phoneme dict
            if hasattr(g2p, 'phoneme_dict'):
                pd = {k: v if isinstance(v, list) else [v] for k, v in g2p.phoneme_dict.items()}
                pd_path = os.path.join(output_dir, f"{name}_pinyin_dict.json")
                with open(pd_path, "w", encoding="utf-8") as f:
                    json.dump(pd, f, ensure_ascii=False, indent=None, separators=(',', ':'))
                print(f"  pinyin_dict: {len(pd)} entries")

            # Tone mapping
            tone_path = os.path.join(output_dir, f"{name}_tone_dict.json")
            with open(tone_path, "w", encoding="utf-8") as f:
                json.dump(g2p.tone_dict, f, ensure_ascii=False)
            print(f"  tone_dict: {len(g2p.tone_dict)} entries")

            # ASCII letter mapping
            letter_path = os.path.join(output_dir, f"{name}_ascii_letter_dict.json")
            with open(letter_path, "w", encoding="utf-8") as f:
                json.dump(g2p.ascii_letter_dict, f, ensure_ascii=False)
            print(f"  ascii_letter_dict: {len(g2p.ascii_letter_dict)} entries")

        # For Japanese G2P: export phoneme dict
        if hasattr(sub_tok, 'g2p') and hasattr(sub_tok.g2p, 'ascii_letter_dict') and 'japanese' in name:
            g2p = sub_tok.g2p
            if hasattr(g2p, 'phoneme_dict') and g2p.phoneme_dict is not None:
                pd = {k: v if isinstance(v, list) else [v] for k, v in g2p.phoneme_dict.items()}
                pd_path = os.path.join(output_dir, f"{name}_word_dict.json")
                with open(pd_path, "w", encoding="utf-8") as f:
                    json.dump(pd, f, ensure_ascii=False, indent=None, separators=(',', ':'))
                print(f"  word_dict: {len(pd)} entries")

            letter_path = os.path.join(output_dir, f"{name}_ascii_letter_dict.json")
            with open(letter_path, "w", encoding="utf-8") as f:
                json.dump(g2p.ascii_letter_dict, f, ensure_ascii=False)
            print(f"  ascii_letter_dict: {len(g2p.ascii_letter_dict)} entries")

            if hasattr(g2p, 'punctuation'):
                punct_path = os.path.join(output_dir, f"{name}_punctuation.json")
                with open(punct_path, "w", encoding="utf-8") as f:
                    json.dump(list(g2p.punctuation), f, ensure_ascii=False)

        # For char-based tokenizers: export the char set info
        if hasattr(sub_tok, 'text_preprocessing_func'):
            # Store info about preprocessing
            func_name = getattr(sub_tok.text_preprocessing_func, '__name__', str(sub_tok.text_preprocessing_func))
            metadata.setdefault("preprocessing", {})[name] = func_name

        # Export pad_with_space
        if hasattr(sub_tok, 'pad_with_space'):
            metadata.setdefault("pad_with_space", {})[name] = sub_tok.pad_with_space

        # Export punct list
        if hasattr(sub_tok, 'PUNCT_LIST'):
            metadata.setdefault("punct_lists", {})[name] = list(sub_tok.PUNCT_LIST)
        elif hasattr(sub_tok, 'punct_list'):
            metadata.setdefault("punct_lists", {})[name] = list(sub_tok.punct_list)

    # Run reference tokenizations for verification
    print(f"\n{'='*60}")
    print("Reference tokenizations (for Swift verification):")
    test_texts = {
        "english_phoneme": "Hello, this is a test.",
        "spanish_phoneme": "Hola, esto es una prueba.",
        "german_phoneme": "Hallo, das ist ein Test.",
        "mandarin_phoneme": "你好，这是一个测试。",
        "japanese_phoneme": "こんにちは、これはテストです。",
        "french_chartokenizer": "Bonjour, ceci est un test.",
        "hindi_chartokenizer": "नमस्ते, यह एक परीक्षा है।",
        "italian_phoneme": "Ciao, questo è un test.",
        "vietnamese_phoneme": "Xin chào, đây là một bài kiểm tra.",
    }

    references = {}
    for name in tok.tokenizer_names:
        if name in test_texts:
            try:
                # Force phoneme_probability=1.0 for deterministic output
                sub = tok.tokenizers[name]
                if hasattr(sub, 'g2p') and hasattr(sub.g2p, 'phoneme_probability'):
                    old_prob = sub.g2p.phoneme_probability
                    sub.g2p.phoneme_probability = 1.0
                ids = tok.encode(test_texts[name], tokenizer_name=name)
                if hasattr(sub, 'g2p') and hasattr(sub.g2p, 'phoneme_probability'):
                    sub.g2p.phoneme_probability = old_prob
                references[name] = {
                    "text": test_texts[name],
                    "token_ids": ids,
                    "num_tokens": len(ids),
                }
                print(f"  {name}: \"{test_texts[name]}\" -> {len(ids)} tokens")
                print(f"    IDs (first 20): {ids[:20]}")
            except Exception as e:
                print(f"  {name}: ERROR - {e}")
                references[name] = {"text": test_texts[name], "error": str(e)}

    # Save metadata
    metadata_path = os.path.join(output_dir, "tokenizer_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    # Save references
    ref_path = os.path.join(output_dir, "tokenizer_references.json")
    with open(ref_path, "w", encoding="utf-8") as f:
        json.dump(references, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"All tokenizer data saved to {output_dir}/")
    for f_name in sorted(os.listdir(output_dir)):
        if 'tokenizer' in f_name or 'phoneme' in f_name or 'heteronym' in f_name or 'pinyin' in f_name or 'tone' in f_name or 'ascii' in f_name or 'punct' in f_name or 'word_dict' in f_name:
            size = os.path.getsize(os.path.join(output_dir, f_name))
            unit = "KB" if size < 1024 * 1024 else "MB"
            size_val = size / 1024 if unit == "KB" else size / 1024 / 1024
            print(f"  {f_name}: {size_val:.1f} {unit}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nemo-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="constants")
    args = parser.parse_args()
    export_tokenizers(args.nemo_path, args.output_dir)

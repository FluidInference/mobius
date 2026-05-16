"""
FLEURS ↔ Nemotron language-code mapping.

FLEURS uses `xx_yy` codes (e.g. "zh_cn", "es_419"); Nemotron's
`prompt_dictionary` uses `xx-XX` (e.g. "zh-CN", "es-ES"). This module
exposes a single dictionary and two small helpers.

Only the 38 languages covered by both datasets are listed. FLEURS has
more (~102); the extras have no Nemotron prompt and would just be
returned with the `auto` prompt (id 101) if you try them.

The list of Nemotron-supported codes comes from `metadata.json`'s
`prompt_dictionary` (the 38 entries other than `auto`).

Languages where the model emits no space-separated words (CJK, Thai,
Lao, Burmese, Khmer, etc.) score via CER instead of WER. Set CER_LANGS
accordingly.
"""
from typing import Optional

# FLEURS-side code → Nemotron-side lang code.
# When FLEURS has multiple variants of one Nemotron language (e.g. pt_br
# and Nemotron only ships "pt-BR"), the FLEURS code maps to the one
# Nemotron actually supports.
FLEURS_TO_NEMOTRON = {
    "en_us": "en-US",
    "es_419": "es-ES",   # FLEURS Latin-American Spanish; closest prompt
    "de_de": "de-DE",
    "fr_fr": "fr-FR",
    "it_it": "it-IT",
    "ar_eg": "ar-EG",
    "ja_jp": "ja-JP",
    "ko_kr": "ko-KR",
    "pt_br": "pt-BR",
    "ru_ru": "ru-RU",
    "hi_in": "hi-IN",
    "cmn_hans_cn": "zh-CN",
    "yue_hant_hk": "zh-TW",  # closest available; encoder still helps
    "vi_vn": "vi-VN",
    "he_il": "he-IL",
    "nl_nl": "nl-NL",
    "cs_cz": "cs-CZ",
    "da_dk": "da-DK",
    "pl_pl": "pl-PL",
    "nb_no": "no-NO",
    "sv_se": "sv-SE",
    "th_th": "th-TH",
    "tr_tr": "tr-TR",
    "bg_bg": "bg-BG",
    "el_gr": "el-GR",
    "et_ee": "et-EE",
    "fi_fi": "fi-FI",
    "hr_hr": "hr-HR",
    "hu_hu": "hu-HU",
    "lt_lt": "lt-LT",
    "lv_lv": "lv-LV",
    "ro_ro": "ro-RO",
    "sk_sk": "sk-SK",
    "uk_ua": "uk-UA",
    "mt_mt": "mt-MT",
    "sl_si": "sl-SI",
}

# Languages where word-segmentation is unreliable or absent →
# score via character error rate instead of word error rate.
CER_LANGS = {
    "zh-CN", "zh-TW",  # Mandarin (both scripts)
    "ja-JP",
    "th-TH",
    "ko-KR",  # Korean: words exist but FLEURS reference spacing differs
}


def fleurs_to_nemotron(fleurs_code: str) -> Optional[str]:
    return FLEURS_TO_NEMOTRON.get(fleurs_code.lower())


def uses_cer(nemotron_lang: str) -> bool:
    return nemotron_lang in CER_LANGS

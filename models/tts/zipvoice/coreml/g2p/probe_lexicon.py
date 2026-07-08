"""Harvest an espeak-parity English lexicon for the LuxTTS Swift G2P.

espeak-ng (via piper_phonemize, voice en-us) is probed per word in four
controlled carrier contexts; the resulting variants reproduce espeak's
clause-level behavior at runtime without espeak:

  mid    "say W trees"      mid-clause, before a stressed word (base form)
  final  "say W"            clause-final ($strend/$atend/$u+ forms)
  unstr  "say W him"        before only-unstressed words ($strend2 forms)
  vowel  "say W apple"      before a vowel-initial word (the/to/an forms,
                            linking-r, final-t flapping)
  pause  "say W but trees"  before a $pause/$brk word (and/or/but/if/which):
                            no liaison/flap; "to" -> tU
  start  "W trees"          clause-initial ($atstart forms: what -> wˌʌt)
  r      "say W red"        before an r-initial word (word-final schwa is
                            rendered ɚ: vanilla -> vɐnˈɪlɚ; "the" keeps ðə)

A second pass probes each word Capitalized; case-sensitive rows are stored
when they differ (espeak: "I" = unstressed pronoun aɪ, "i" = letter ˈaɪ;
$capital words like Polish/polish).

Sparse storage: variants equal to `mid` are omitted.

Also harvested (aux JSON):
  - multi-word phrase entries (espeak en_list `(a b)` entries, e.g. "in the"
    -> merged pron with no space) with mid/final/vowel variants
  - homograph verb/noun/past variants ($verbf/$nounf/$pastf machinery)
  - flag word sets (verbf/nounf/pastf/verbsf/verbextend) from en_list
  - letter names (for all-caps spell-out)
  - $capital words (Polish vs polish)

Inputs (fetch once):
  en_list      https://raw.githubusercontent.com/rhasspy/espeak-ng/master/dictsource/en_list
               (the rhasspy fork matches piper_phonemize's bundled dictionary)
  freq         https://raw.githubusercontent.com/hermitdave/FrequencyWords/master/content/2018/en/en_full.txt
  words-alpha  https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt

Usage:
    .venv/bin/python -m coreml.g2p.probe_lexicon \
        --en-list en_list --freq en_full.txt \
        --words-alpha words_alpha.txt --top 200000 \
        --extra-vocab coreml/g2p/corpus_en_1000.txt coreml/g2p/dev_corpus.txt \
        --out-lexicon coreml/g2p/lexicon_en_us.tsv \
        --out-aux coreml/g2p/lexicon_aux.json

The FluidAudio bundle is the TSV as raw DEFLATE plus the aux JSON:
    python -c "import zlib; c=zlib.compressobj(9, zlib.DEFLATED, -15); \
        open('luxtts_en_us_lexicon.tsv.zz','wb').write( \
        c.compress(open('coreml/g2p/lexicon_en_us.tsv','rb').read())+c.flush())"
"""

import argparse
import json
import re
import sys
from pathlib import Path

from piper_phonemize import phonemize_espeak

SAY = "sˈeɪ "
TREES = " tɹˈiːz"
HIM = " hˌɪm"
APPLE = " ˈæpəl"
BUT_TREES = " bˌʌt tɹˈiːz"
RED = " ɹˈɛd"

CONTRACTIONS = [
    "don't", "won't", "can't", "isn't", "aren't", "wasn't", "weren't",
    "hasn't", "haven't", "hadn't", "doesn't", "didn't", "couldn't",
    "wouldn't", "shouldn't", "mustn't", "needn't", "ain't", "shan't",
    "i'm", "you're", "we're", "they're", "i've", "you've", "we've",
    "they've", "could've", "would've", "should've", "i'll", "you'll",
    "he'll", "she'll", "we'll", "they'll", "it'll", "that'll", "i'd",
    "you'd", "he'd", "she'd", "we'd", "they'd", "it's", "that's",
    "there's", "here's", "what's", "who's", "she's", "he's", "let's",
    "o'clock", "y'all", "ma'am", "'em", "everyone's", "everybody's",
    "somebody's", "someone's", "nobody's", "nothing's", "one's",
    "world's", "life's",
]


def p(text: str) -> str:
    return "".join("".join(c) for c in phonemize_espeak(text, "en-us"))


def probe_word(word: str):
    """Return (mid, final, unstr, vowel, pause, start) or None."""
    mid_out = p(f"say {word} trees")
    if not (mid_out.startswith(SAY) and mid_out.endswith(TREES)):
        return None
    mid = mid_out[len(SAY):-len(TREES)]
    if not mid:
        return None

    final_out = p(f"say {word}")
    final = final_out[len(SAY):] if final_out.startswith(SAY) else None

    unstr_out = p(f"say {word} him")
    unstr = (
        unstr_out[len(SAY):-len(HIM)]
        if unstr_out.startswith(SAY) and unstr_out.endswith(HIM)
        else None
    )

    vowel_out = p(f"say {word} apple")
    vowel = (
        vowel_out[len(SAY):-len(APPLE)]
        if vowel_out.startswith(SAY) and vowel_out.endswith(APPLE)
        else None
    )

    pause_out = p(f"say {word} but trees")
    pause = (
        pause_out[len(SAY):-len(BUT_TREES)]
        if pause_out.startswith(SAY) and pause_out.endswith(BUT_TREES)
        else None
    )

    start_out = p(f"{word} trees")
    start = start_out[:-len(TREES)] if start_out.endswith(TREES) else None

    r_out = p(f"say {word} red")
    r = (
        r_out[len(SAY):-len(RED)]
        if r_out.startswith(SAY) and r_out.endswith(RED)
        else None
    )

    return mid, final, unstr, vowel, pause, start, r


def parse_en_list(path: Path):
    """Extract phrases, homographs, flag sets, capital words from en_list."""
    phrases = {}
    flags_by_word = {}
    homograph_words = set()
    capital_words = set()

    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.split("//")[0].strip()
        if not line:
            continue
        flags = set(re.findall(r"\$[a-z0-9+]+", line))
        if line.startswith("("):
            m = re.match(r"\(([^)]*)\)", line)
            if not m:
                continue
            phrase = m.group(1).strip().lower()
            if not re.fullmatch(r"[a-z']+( [a-z']+)+", phrase):
                continue
            phrases.setdefault(phrase, set()).update(flags)
            continue
        parts = line.split()
        word = parts[0].lstrip("?0123456789").lower()
        if not re.fullmatch(r"[a-z']+", word):
            continue
        if flags & {"$verb", "$noun", "$past"}:
            homograph_words.add(word)
        if "$capital" in flags:
            capital_words.add(word)
        flags_by_word.setdefault(word, set()).update(flags)

    flag_sets = {
        name: sorted(
            w for w, f in flags_by_word.items() if f"${name}" in f
        )
        for name in [
            "verbf", "verbsf", "nounf", "pastf", "verbextend", "pause",
            "brk", "allcaps", "abbrev",
        ]
    }
    return phrases, sorted(homograph_words), flag_sets, sorted(capital_words)


def compose(words_prons):
    return " ".join(words_prons)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--en-list", required=True)
    parser.add_argument("--freq", required=True, help="'word count' per line")
    parser.add_argument("--words-alpha", required=True)
    parser.add_argument("--top", type=int, default=200000)
    parser.add_argument("--extra-vocab", nargs="*", default=[])
    parser.add_argument("--out-lexicon", required=True)
    parser.add_argument("--out-aux", required=True)
    args = parser.parse_args()

    alpha = set(
        w.strip().lower()
        for w in Path(args.words_alpha).read_text().splitlines()
        if w.strip()
    )
    dictwords = set(
        w.strip().lower()
        for w in Path("/usr/share/dict/words").read_text().splitlines()
        if w.strip()
    )
    propernames = [
        w.strip()
        for w in Path("/usr/share/dict/propernames").read_text().splitlines()
        if w.strip()
    ]

    words = []
    seen = set()
    for line in Path(args.freq).read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        w = parts[0].lower()
        if w in seen or not re.fullmatch(r"[a-z]+", w):
            continue
        if w not in alpha and w not in dictwords:
            continue
        seen.add(w)
        words.append(w)
        if len(words) >= args.top:
            break

    for extra in args.extra_vocab:
        for line in Path(extra).read_text(encoding="utf-8").splitlines():
            for w in re.findall(r"[A-Za-z][A-Za-z']*", line):
                lw = w.lower()
                if lw not in seen:
                    seen.add(lw)
                    words.append(lw)

    for w in CONTRACTIONS:
        if w not in seen:
            seen.add(w)
            words.append(w)

    # hyphenated number compounds from the normalizer (inflect emits
    # "twenty-six"; espeak's flapping inside them is irregular, e.g.
    # thirty-five -> θˈɜːɾifˈaɪv but thirty-nine -> θˈɜːtinˈaɪn, so they
    # are probed whole) plus hyphenated tokens seen in the eval corpora
    tens = ["twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
    ones = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    ordinals = [
        "first", "second", "third", "fourth", "fifth", "sixth", "seventh",
        "eighth", "ninth",
    ]
    hyphen_words = [f"{t}-{o}" for t in tens for o in ones]
    hyphen_words += [f"{t}-{o}" for t in tens for o in ordinals]
    for extra in args.extra_vocab:
        for line in Path(extra).read_text(encoding="utf-8").splitlines():
            hyphen_words += re.findall(
                r"[A-Za-z][A-Za-z']*(?:-[A-Za-z][A-Za-z']*)+", line
            )
    for w in hyphen_words:
        lw = w.lower()
        if lw not in seen:
            seen.add(lw)
            words.append(lw)

    proper_only = [n for n in propernames if n.lower() not in seen]
    for n in proper_only:
        seen.add(n.lower())

    print(f"probing {len(words)} words + {len(proper_only)} proper names")

    lexicon = {}
    failed = 0
    for i, w in enumerate(words):
        result = probe_word(w)
        if result is None:
            failed += 1
            continue
        lexicon[w] = result
        if (i + 1) % 20000 == 0:
            print(f"  {i + 1}/{len(words)}")
    for n in proper_only:
        result = probe_word(n)
        if result is None:
            failed += 1
            continue
        lexicon[n.lower()] = result
    print(f"lexicon: {len(lexicon)} entries ({failed} extraction failures)")

    phrases_raw, homograph_words, flag_sets, capital_words = parse_en_list(
        Path(args.en_list)
    )

    # ---- capitalization pass (before phrases: their baselines need the
    # capital rows): any word whose Capitalized mid form differs gets a
    # full case-sensitive row (espeak: I vs i, Polish vs polish, ...)
    capitals = {}
    for w in list(lexicon):
        cap = w.capitalize()
        if cap == w:
            continue
        out = p(f"say {cap} trees")
        if out.startswith(SAY) and out.endswith(TREES):
            form = out[len(SAY):-len(TREES)]
            if form != lexicon[w][0]:
                row = probe_word(cap)
                if row:
                    capitals[cap] = row
    print(f"capital-sensitive words: {len(capitals)}")

    # ---- phrases: keep only variants whose espeak output differs from
    # word-by-word composition (those are the merge/special entries).
    # Baselines are case-aware so "I had" doesn't produce spurious
    # variants that would shadow "had to" at runtime.
    phrase_table = {}
    for phrase, flags in phrases_raw.items():
        pw = phrase.split()
        if not all(w in lexicon for w in pw):
            # probe missing constituents on demand
            ok = True
            for w in pw:
                if w not in lexicon:
                    r = probe_word(w)
                    if r is None:
                        ok = False
                        break
                    lexicon[w] = r
            if not ok:
                continue
        mid_out = p(f"say {phrase} trees")
        final_out = p(f"say {phrase}")
        vowel_out = p(f"say {phrase} apple")
        cap = phrase[0].upper() + phrase[1:]
        start_out = p(f"{phrase} trees")
        startcap_out = p(f"{cap} trees")
        midcap_out = p(f"say {cap} trees")
        entry = {}
        mids = [lexicon[w][0] for w in pw]
        cap_row = capitals.get(pw[0].capitalize(), lexicon[pw[0]])
        composed = compose(mids)
        composed_midcap = compose([cap_row[0]] + mids[1:])
        composed_start = compose(
            [lexicon[pw[0]][5] or lexicon[pw[0]][0]] + mids[1:]
        )
        composed_startcap = compose([cap_row[5] or cap_row[0]] + mids[1:])
        if mid_out.startswith(SAY) and mid_out.endswith(TREES):
            mid = mid_out[len(SAY):-len(TREES)]
            if mid != composed:
                entry["mid"] = mid
        if midcap_out.startswith(SAY) and midcap_out.endswith(TREES):
            midcap = midcap_out[len(SAY):-len(TREES)]
            if midcap != composed_midcap and midcap != entry.get("mid"):
                entry["midcap"] = midcap
        if final_out.startswith(SAY):
            final = final_out[len(SAY):]
            composed_final = compose(
                mids[:-1] + [lexicon[pw[-1]][1] or lexicon[pw[-1]][0]]
            )
            if final != composed_final:
                entry["final"] = final
        if vowel_out.startswith(SAY) and vowel_out.endswith(APPLE):
            vowel = vowel_out[len(SAY):-len(APPLE)]
            composed_vowel = compose(
                mids[:-1] + [lexicon[pw[-1]][3] or lexicon[pw[-1]][0]]
            )
            if vowel != composed_vowel:
                entry["vowel"] = vowel
        if start_out.endswith(TREES):
            start = start_out[:-len(TREES)]
            if start != entry.get("mid", composed_start):
                entry["start"] = start
        if startcap_out.endswith(TREES):
            startcap = startcap_out[:-len(TREES)]
            if startcap != entry.get(
                "start", entry.get("mid", composed_startcap)
            ) and startcap != composed_startcap:
                entry["startcap"] = startcap
        r_out = p(f"say {phrase} red")
        if r_out.startswith(SAY) and r_out.endswith(RED):
            r = r_out[len(SAY):-len(RED)]
            if r != entry.get("mid", composed):
                entry["r"] = r
        if entry:
            entry["flags"] = sorted(flags)
            phrase_table[phrase] = entry
    print(f"phrase table: {len(phrase_table)} entries")

    # ---- homographs: verb/noun/past context variants, probed for EVERY
    # word (espeak also selects them via en_rules suffix logic — e.g.
    # "estimate" — so the en_list $verb/$noun word list is incomplete).
    # The leads vary ("to" -> tə/tʊ before vowels), so both are accepted.
    homographs = {}
    contexts = [
        ("verb", "say to {} trees", [SAY + "tə ", SAY + "tʊ "]),
        ("noun", "say the {} trees", [SAY + "ðə ", SAY + "ðɪ "]),
        ("past", "say had {} trees", [SAY + "hæd ", SAY + "hˌæd "]),
    ]
    for i, w in enumerate(sorted(lexicon)):
        entry = {}
        for key, template, leads in contexts:
            out = p(template.format(w))
            if not out.endswith(TREES):
                continue
            for lead in leads:
                if out.startswith(lead):
                    form = out[len(lead):-len(TREES)]
                    if form and form != lexicon[w][0]:
                        entry[key] = form
                    break
        if entry:
            homographs[w] = entry
        if (i + 1) % 40000 == 0:
            print(f"  homographs {i + 1}/{len(lexicon)}")
    print(f"homographs with context variants: {len(homographs)}")

    # ---- letter names (for all-caps spell-out)
    letters = {}
    for c in "abcdefghijklmnopqrstuvwxyz":
        out = p(f"say {c.upper()}")
        if out.startswith(SAY):
            letters[c] = out[len(SAY):]

    # ---- write lexicon TSV (sparse variants); capital rows keyed
    # case-sensitively and resolved before the lowercase row at runtime
    def row_line(word, row):
        mid = row[0]
        return "\t".join(
            [word, mid]
            + [(v if v and v != mid else "") for v in row[1:]]
        ).rstrip("\t")

    lines = [row_line(w, lexicon[w]) for w in sorted(lexicon)]
    lines += [row_line(w, capitals[w]) for w in sorted(capitals)]
    Path(args.out_lexicon).write_text("\n".join(lines) + "\n", encoding="utf-8")

    aux = {
        "phrases": phrase_table,
        "homographs": homographs,
        "flag_sets": flag_sets,
        "letters": letters,
    }
    Path(args.out_aux).write_text(
        json.dumps(aux, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"wrote {args.out_lexicon} and {args.out_aux}")


if __name__ == "__main__":
    main()

"""Python reference of the LuxTTS Swift G2P (LuxTtsG2p.swift mirrors this).

Pipeline (matches EmiliaTokenizer's English path):
  1. map_punctuations + EnglishTextNormalizer (upstream, called directly here;
     the Swift side ports it)
  2. tokenize into words / punctuation, split clauses at ,.!?;: (em-dash -> ;
     and "…" is a silent clause break)
  3. per clause: phrase-table longest match (espeak multi-word entries),
     then right-to-left per-word variant selection:
       clause-final           -> final
       followed only by
         unstressed words     -> unstr  ($strend2, resolved right-to-left)
       next word $pause/$brk  -> pause  (blocks liaison/flap; to -> tU)
       next word vowel-onset  -> vowel  (the/to/an, linking-r, flapping)
       clause-initial         -> start  ($atstart: what -> wˌʌt)
       otherwise              -> mid
     Lookup is case-sensitive first (espeak: "I" pronoun aɪ vs "i" letter
     ˈaɪ, Polish vs polish), then lower-case.
  4. homograph verb/noun/past variants driven by the preceding word's
     $verbf/$verbsf/$nounf/$pastf flags ($verbextend keeps the state)
  5. assembly: ' ' between words; ",;:" attach + space; ".!?" attach, no
     space (espeak clause concat); "…" nothing, no space

Usage:
    .venv/bin/python -m coreml.g2p.reference_g2p \
        --lexicon coreml/g2p/lexicon_en_us.tsv --aux coreml/g2p/lexicon_aux.json \
        --corpus coreml/g2p/dev_corpus.txt --out /tmp/ref_dump.jsonl
"""

import argparse
import json
import re
from pathlib import Path

STAGING_TOKENS = Path(__file__).resolve().parents[2] / "build/hf-staging/tokens.txt"

CLAUSE_PUNCT = set(",.!?;:")
VOWEL_SCALARS = set("aeiouæɑɐɔəɛɜɪʊʌʉɒɚᵻ")
STRESS_MARKS = set("ˈˌ")
VOICELESS = set("ptkfθsʃ")
SIBILANT = set("szʃʒ")

WORD_RE = re.compile(r"[A-Za-z][A-Za-z']*|[0-9]+|[^\sA-Za-z0-9]")

MID, FINAL, UNSTR, VOWEL, PAUSE, START, RVAR = range(7)


class Lexicon:
    def __init__(self, lexicon_path, aux_path):
        self.entries = {}
        for line in Path(lexicon_path).read_text(encoding="utf-8").splitlines():
            parts = line.split("\t")
            row = [parts[1]] + [
                parts[i] if len(parts) > i and parts[i] else None
                for i in range(2, 8)
            ]
            self.entries[parts[0]] = tuple(row)
        aux = json.loads(Path(aux_path).read_text(encoding="utf-8"))
        self.phrases = aux["phrases"]
        self.homographs = aux["homographs"]
        self.flag_sets = {k: set(v) for k, v in aux["flag_sets"].items()}
        self.letters = aux["letters"]
        self.pause_words = self.flag_sets["pause"] | self.flag_sets["brk"]
        # all-caps tokens spoken as words (NO, ALL); everything else spells
        self.allcaps_words = (
            self.flag_sets["allcaps"] - self.flag_sets["abbrev"]
        )
        self.max_phrase_len = max(
            (len(k.split()) for k in self.phrases), default=1
        )

    def lookup(self, word):
        """Case-sensitive row first, then lower-case row."""
        entry = self.entries.get(word)
        if entry is None:
            entry = self.entries.get(word.lower())
        return entry


def first_phone(pron):
    for ch in pron:
        if ch not in STRESS_MARKS:
            return ch
    return ""


def letter_name_with_stress(name, mark):
    stripped = "".join(c for c in name if c not in STRESS_MARKS)
    for i, ch in enumerate(stripped):
        if ch in VOWEL_SCALARS:
            return stripped[:i] + mark + stripped[i:]
    return mark + stripped


def add_suffix_s(mid):
    last = mid[-1]
    if last in SIBILANT:
        return mid + "ɪz"
    if last in VOICELESS:
        return mid + "s"
    return mid + "z"


class ReferenceG2p:
    """Units: ('word', token) or ('phrase', [tokens], key). Selection is
    right-to-left so $strend2 chains resolve on final stress states."""

    def __init__(self, lexicon: Lexicon):
        self.lex = lexicon
        self.oov = set()

    # ---- fallbacks -------------------------------------------------------

    def spell_letters(self, word):
        out = []
        letters = [c for c in word.lower() if c.isalpha()]
        for i, c in enumerate(letters):
            name = self.lex.letters.get(c, "")
            mark = "ˈ" if i == len(letters) - 1 else "ˌ"
            out.append(letter_name_with_stress(name, mark))
        return "".join(out)

    def oov_pron(self, word):
        lower = word.lower()
        for suffix in ("'s", "s'"):
            if lower.endswith(suffix):
                base = self.lex.lookup(lower[: -len(suffix)])
                if base:
                    return add_suffix_s(base[MID])
        if lower.endswith("s") and self.lex.lookup(lower[:-1]):
            return add_suffix_s(self.lex.lookup(lower[:-1])[MID])
        if len(word) >= 2 and word.isupper() and "'" not in word:
            return self.spell_letters(word)
        # camelCase: split, parts space-joined; single letters -> names
        if any(c.isupper() for c in word[1:]):
            parts = re.findall(r"[A-Z]?[a-z']+|[A-Z]+(?![a-z])", word)
            if len(parts) > 1:
                out = []
                for part in parts:
                    if len(part) == 1:
                        out.append(
                            letter_name_with_stress(
                                self.lex.letters.get(part.lower(), ""), "ˈ"
                            )
                        )
                    else:
                        sub = self.lex.lookup(part)
                        out.append(sub[MID] if sub else self.oov_pron(part))
                return " ".join(out)
        self.oov.add(word)
        return self.spell_letters(word)

    # ---- flags -----------------------------------------------------------

    CONTRACTION_BASE = {"won't": "will", "can't": "can", "shan't": "shall"}

    def flag_word(self, token):
        """The word (lower-cased base form) whose espeak flags drive the
        expect_verb/noun/past counters. n't/'ll/'ve/'s contractions inherit
        the auxiliary's flags (doesn't -> does)."""
        lower = token.lower().replace("’", "'")
        if lower == "i" and token != "I":
            return None  # lowercase i is the letter, not the pronoun
        if lower in self.CONTRACTION_BASE:
            return self.CONTRACTION_BASE[lower]
        if lower.endswith("n't"):
            return lower[:-3]
        if lower.endswith("'ll"):
            return "will"
        if lower.endswith("'ve"):
            return "have"
        if lower.endswith("'s"):
            return "is"
        return lower

    def update_counters(self, counters, token):
        """espeak translateword.c: $pastf -> expect_past=3 ; $verbf ->
        expect_verb=2 ; $verbsf -> expect_verb_s=2 ; $nounf ->
        expect_noun=2 ; then decrement all (unless $verbextend)."""
        word = self.flag_word(token)
        fs = self.lex.flag_sets
        if word is not None:
            if word in fs["pastf"]:
                counters.update(past=3, verb=0, noun=0)
            elif word in fs["verbf"]:
                counters.update(verb=2, verb_s=0, noun=0)
            elif word in fs["verbsf"]:
                counters.update(verb=0, verb_s=2, past=0, noun=0)
            elif word in fs["nounf"]:
                counters.update(noun=2, verb=0, verb_s=0, past=0)
        if word is None or word not in fs["verbextend"]:
            for key in counters:
                if counters[key] > 0:
                    counters[key] -= 1

    def homograph_key(self, counters, token):
        if counters["verb"] > 0 or (
            counters["verb_s"] > 0 and token.lower().endswith("s")
        ):
            return "verb"
        if counters["past"] > 0:
            return "past"
        if counters["noun"] > 0:
            return "noun"
        return None

    # ---- clause ------------------------------------------------------------

    def phonemize_clause(self, items, clause_initial, break_before=None):
        """items: ('word', token) | ('hyph', [parts]) | ('lit', pron) for
        one clause; break_before[i] marks a quote/paren pause boundary
        before item i. Returns one pronunciation chunk per unit."""
        n = len(items)
        if n == 0:
            return []
        if break_before is None:
            break_before = [False] * n

        # unit segmentation: phrase spans over consecutive 'word' items
        # (a phrase span is only consumed if usable at its position;
        # a span may not straddle a quote/paren boundary)
        units = []  # (kind, start, length, entry)
        i = 0
        while i < n:
            kind, value = items[i]
            if kind != "word":
                units.append((kind, i, 1, None))
                i += 1
                continue
            matched = None
            for length in range(min(self.lex.max_phrase_len, n - i), 1, -1):
                if any(items[k][0] != "word" for k in range(i, i + length)):
                    continue
                if any(break_before[k] for k in range(i + 1, i + length)):
                    continue
                key = " ".join(items[k][1].lower() for k in range(i, i + length))
                entry = self.lex.phrases.get(key)
                if not entry:
                    continue
                at_end = i + length == n
                usable = (
                    "mid" in entry
                    or "midcap" in entry
                    or "vowel" in entry
                    or "r" in entry
                    or (at_end and "final" in entry)
                    or (clause_initial and i == 0 and ("start" in entry or "startcap" in entry))
                )
                if usable:
                    matched = (i, length, entry)
                    break
            if matched:
                units.append(("phrase",) + matched)
                i += matched[1]
            else:
                units.append(("word", i, 1, None))
                i += 1

        # pass 1 (left to right): homograph selection key per unit
        # (espeak expect_verb/noun/past counters), and $pause application:
        # a pause word only pauses from the 3rd word after the last break,
        # and an applied pause starts a new "break" (probed: "gas or water
        # and he" -> "or" pauses, resetting, so "and" is 2nd -> liaison).
        flags = [None] * len(units)
        pause_applied = [False] * len(units)  # unit is an APPLIED pause word
        counters = {"verb": 0, "verb_s": 0, "noun": 0, "past": 0}
        word_pos = 0  # words since last break
        for u, (kind, start, length, entry) in enumerate(units):
            if break_before[start]:
                word_pos = 0
            if kind == "word":
                token = items[start][1]
                flags[u] = self.homograph_key(counters, token)
                if (
                    token.lower() in self.lex.pause_words
                    and word_pos >= 2
                    and start != n - 1
                ):
                    pause_applied[u] = True
                    word_pos = 0  # pause word restarts the count as word 0
            for k in range(start, start + length):
                if items[k][0] == "word":
                    self.update_counters(counters, items[k][1])
                if not pause_applied[u]:
                    word_pos += 1

        # pass 2 (right to left): variant selection
        chunks = [None] * len(units)
        stressed_after = False
        next_first_phone = None
        next_is_break = False  # next unit is an applied pause word

        for u in range(len(units) - 1, -1, -1):
            kind, start, length, entry = units[u]
            at_end = start + length == n
            at_start = clause_initial and start == 0

            if kind == "phrase":
                tokens = [items[k][1] for k in range(start, start + length)]
                chunk = self.select_phrase(
                    entry, tokens, at_end, at_start,
                    next_first_phone, next_is_break, stressed_after,
                )
            elif kind == "hyph":
                chunk = self.select_hyph(
                    items[start][1], flags[u], at_end, at_start,
                    next_first_phone, next_is_break, stressed_after,
                )
            elif kind == "lit":
                chunk = items[start][1]
            else:
                chunk = self.select_word(
                    items[start][1], flags[u], at_end, at_start,
                    next_first_phone, next_is_break, stressed_after,
                )
            chunks[u] = chunk
            if "ˈ" in chunk:
                stressed_after = True
            next_first_phone = first_phone(chunk)
            next_is_break = pause_applied[u] or break_before[start]

        return chunks

    def select_hyph(
        self, parts, flag, at_end, at_start, nxt_phone, nxt_is_break,
        stressed_after,
    ):
        """Hyphen chain: whole-token lexicon row if probed (twenty-six),
        else parts pronounced separately, concatenated without space."""
        key = "-".join(parts)
        if self.lex.lookup(key) is not None:
            return self.select_word(
                key, flag, at_end, at_start, nxt_phone, nxt_is_break,
                stressed_after,
            )
        out = []
        for j, part in enumerate(parts):
            if j == len(parts) - 1:
                out.append(
                    self.select_word(
                        part, None, at_end, False, nxt_phone, nxt_is_break,
                        stressed_after,
                    )
                )
            else:
                entry = self.lex.lookup(part)
                out.append(entry[MID] if entry else self.oov_pron(part))
        return "".join(out)

    def compose_phrase(self, tokens, variant_last):
        chunks = []
        for i, w in enumerate(tokens):
            entry = self.lex.lookup(w)
            if entry is None:
                chunks.append(self.oov_pron(w))
            elif i == len(tokens) - 1:
                chunks.append(variant_last(entry))
            else:
                chunks.append(entry[MID])
        return " ".join(chunks)

    def select_phrase(
        self, entry, tokens, at_end, at_start, nxt_phone, nxt_is_break,
        stressed_after,
    ):
        capitalized = tokens[0][0].isupper()
        if at_start and capitalized and "startcap" in entry:
            return entry["startcap"]
        if at_start and "start" in entry:
            return entry["start"]
        if at_end and "final" in entry:
            return entry["final"]
        if not at_end and not nxt_is_break:
            if nxt_phone == "ɹ" and "r" in entry:
                return entry["r"]
            if nxt_phone in VOWEL_SCALARS and "vowel" in entry:
                return entry["vowel"]
        if capitalized and "midcap" in entry and not at_end:
            return entry["midcap"]
        if "mid" in entry and not at_end:
            return entry["mid"]
        # fall back to word-by-word composition
        if at_end:
            return self.compose_phrase(
                tokens, lambda e: e[FINAL] or e[MID]
            )
        return self.compose_phrase(tokens, lambda e: e[MID])

    def select_word(
        self, token, flag, at_end, at_start, nxt_phone, nxt_is_break,
        stressed_after,
    ):
        # all-caps tokens spell out unless espeak marks them $allcaps words
        if (
            len(token) >= 2
            and token.isupper()
            and token.isalpha()
            and token.lower() not in self.lex.allcaps_words
        ):
            return self.spell_letters(token)

        if flag:
            hom = self.lex.homographs.get(token.lower())
            if hom and flag in hom:
                return hom[flag]

        entry = self.lex.lookup(token)
        if entry is None:
            return self.oov_pron(token)
        mid, final, unstr, vowel, pause, start, rvar = entry

        if at_end:
            return final or mid
        if unstr and not stressed_after and not nxt_is_break:
            selected = unstr
            if vowel and nxt_phone in VOWEL_SCALARS:
                if vowel == mid + "ɹ" and selected.endswith(mid[-1]):
                    selected += "ɹ"
                elif vowel == mid[:-1] + "ɾ" and selected.endswith(mid[-1]):
                    selected = selected[:-1] + "ɾ"
            return selected
        if nxt_is_break:
            return pause or mid
        if nxt_phone == "ɹ" and rvar:
            return rvar
        if vowel and nxt_phone in VOWEL_SCALARS:
            return vowel
        if at_start and start:
            return start
        return mid

    # ---- sentence ----------------------------------------------------------

    def phonemize(self, normalized_text):
        positions = [
            (m.group(0), m.start(), m.end())
            for m in WORD_RE.finditer(normalized_text)
        ]

        # hyphen chains: word(-word)+ with no spaces -> single unit, parts
        # pronounced separately and concatenated without space
        tokens = []  # ("hyph", parts, start, end) | ("tok", tok, start, end)
        i = 0
        while i < len(positions):
            tok, start, end = positions[i]
            if (
                tok[0].isalpha()
                and i + 2 < len(positions)
                and positions[i + 1][0] == "-"
                and positions[i + 1][1] == end
                and positions[i + 2][1] == positions[i + 1][2]
                and positions[i + 2][0][0].isalpha()
            ):
                parts = [tok]
                j = i + 1
                last_end = end
                while (
                    j + 1 < len(positions)
                    and positions[j][0] == "-"
                    and positions[j][1] == last_end
                    and positions[j + 1][0][0].isalpha()
                    and positions[j + 1][1] == positions[j][2]
                ):
                    parts.append(positions[j + 1][0])
                    last_end = positions[j + 1][2]
                    j += 2
                tokens.append(("hyph", parts, start, last_end))
                i = j
                continue
            tokens.append(("tok", tok, start, end))
            i += 1

        pieces = []
        clause = []  # ('word', tok) | ('hyph', parts) | ('lit', pron)
        breaks = []  # per clause item: break (quote/paren) before it?
        clause_initial = True
        pending_break = False
        glue_next = False  # "…"/mid-token "?": join to previous, no space

        def flush(clause_initial_flag):
            nonlocal pending_break
            pending_break = False
            if not clause:
                return None
            chunks = self.phonemize_clause(
                list(clause), clause_initial_flag, list(breaks)
            )
            clause.clear()
            breaks.clear()
            return " ".join(c for c in chunks if c)

        def append_flush(clause_initial_flag, glue):
            phon = flush(clause_initial_flag)
            if phon:
                if glue and pieces and pieces[-1] not in CLAUSE_PUNCT:
                    pieces.append(phon)  # no separator (espeak "…" join)
                else:
                    pieces.append(phon)
            return phon

        for idx, entry in enumerate(tokens):
            kind, value, start, end = entry
            if kind == "hyph":
                clause.append((kind, value))
                breaks.append(pending_break)
                pending_break = False
                continue
            tok = value
            if tok[0].isalpha() or tok.isdigit():
                clause.append(("word", tok))
                breaks.append(pending_break)
                pending_break = False
                continue

            # mid-token punctuation (no space on either side): espeak
            # reads "." as "dot", "!" as "exclamation", drops "," and "?"
            prev_glued = idx > 0 and tokens[idx - 1][3] == start
            next_glued = idx + 1 < len(tokens) and tokens[idx + 1][2] == end
            mid_token = prev_glued and next_glued

            ch = ";" if tok in "—–" else tok
            if ch in CLAUSE_PUNCT or ch == "…":
                if mid_token and ch == ".":
                    clause.append(("lit", "dˈɑːt"))
                    breaks.append(False)
                    continue
                if mid_token and ch == "!":
                    clause.append(("lit", "ˈɛkskləmˌeɪʃən"))
                    breaks.append(False)
                    continue
                if mid_token and ch == ",":
                    continue  # dropped, words stay separated by a space
                if mid_token and ch == "?":
                    append_flush(clause_initial, glue_next)
                    glue_next = True
                    clause_initial = True
                    continue
                append_flush(clause_initial, glue_next)
                glue_next = False
                if ch != "…":
                    pieces.append(ch)
                    pieces.append(" " if ch in ",;:" else "")
                else:
                    glue_next = True
                clause_initial = True
            elif ch in "\"'()[]«»":
                # quotes/parens: transparent but insert a pause boundary
                if clause:
                    pending_break = True
            # dashes/other symbols: transparent
        append_flush(clause_initial, glue_next)

        text = "".join(pieces)
        text = re.sub(r" +", " ", text).strip()
        return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lexicon", required=True)
    parser.add_argument("--aux", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from zipvoice.tokenizer.tokenizer import EmiliaTokenizer

    tokenizer = EmiliaTokenizer(token_file=str(STAGING_TOKENS))
    g2p = ReferenceG2p(Lexicon(args.lexicon, args.aux))

    token2id = tokenizer.token2id
    sentences = [
        s.strip()
        for s in Path(args.corpus).read_text(encoding="utf-8").splitlines()
        if s.strip()
    ]
    with open(args.out, "w", encoding="utf-8") as f:
        for text in sentences:
            normalized = tokenizer.english_normalizer.normalize(
                tokenizer.preprocess_text(text)
            )
            phon = g2p.phonemize(normalized)
            ids = [token2id[c] for c in phon if c in token2id]
            f.write(
                json.dumps(
                    {"text": text, "phonemes": phon, "ids": ids},
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"OOV words: {len(g2p.oov)}")
    if g2p.oov:
        print(sorted(g2p.oov)[:40])


if __name__ == "__main__":
    main()

"""Build the 1,000-sentence English G2P validation corpus for LuxTTS phase 2.

Mix (fixed seed, deterministic):
  - 100 conversational phrases (MiniMax TTS-Multilingual-Test-Set, english)
  - 550 LibriSpeech test-clean transcripts (lowercased, sentence-cased,
    length-filtered, final period added)
  - 250 template sentences exercising numbers, currency, times, dates,
    ordinals, decimals, fractions and percentages
  - 100 template sentences exercising proper names (/usr/share/dict/propernames)

The resulting corpus file is committed (corpus_en_1000.txt) so scoring is
reproducible without the source datasets. Re-run only to regenerate it.

Usage:
    .venv/bin/python -m coreml.g2p.build_corpus \
        --minimax <FluidAudio>/Benchmarks/tts/corpus/minimax/english.txt \
        --librispeech "~/Library/Application Support/FluidAudio/Datasets/LibriSpeech/test-clean" \
        --out coreml/g2p/corpus_en_1000.txt
"""

import argparse
import random
from pathlib import Path

SEED = 20260707

NUMERIC_TEMPLATES = [
    "The meeting starts at {time} on {month} {ordinal}.",
    "It costs {dollars}, which is {percent} more than last year.",
    "She ran {decimal} miles in under {cardinal} minutes.",
    "The invoice total came to {dollars} after the discount.",
    "Chapter {cardinal} begins on page {cardinal2}.",
    "The train leaves at {time}, so be there by {time2}.",
    "About {percent} of the {cardinal} respondents agreed.",
    "The recipe needs {fraction} of a cup of sugar and {cardinal} eggs.",
    "He was born in {year} and moved here in {year2}.",
    "The temperature dropped to {cardinal} degrees overnight.",
    "Their new apartment is {decimal} miles from the office.",
    "The {ordinal} floor has {cardinal} rooms and {cardinal2} windows.",
    "We waited {cardinal} minutes for the {time} bus.",
    "The stock fell {percent} to {dollars} a share.",
    "Version {cardinal} ships on {month} {ordinal}, {year}.",
    "Only {cardinal} of the {cardinal2} tickets are left.",
    "The marathon record is {cardinal} hours and {cardinal2} minutes.",
    "My grandfather turned {cardinal} on the {ordinal} of {month}.",
    "The bill was {dollars} plus a {percent} tip.",
    "Flight {cardinal2} departs at {time} from gate {cardinal}.",
]

NAME_TEMPLATES = [
    "{name} and {name2} drove to the coast on Saturday.",
    "Have you met {name}, the new engineer from the {city} office?",
    "{name} said the project would be done by Friday.",
    "According to {name2}, the results were never published.",
    "{name} introduced {name2} to everyone at the party.",
    "The award went to {name} for her work on the harbor bridge.",
    "{name} couldn't believe what {name2} had written.",
    "Later that evening, {name} called {name2} about the missing report.",
    "{name2} grew up near {city} before moving abroad.",
    "Everyone agreed that {name} deserved the promotion.",
]

CITIES = [
    "Boston", "Denver", "Chicago", "Seattle", "Atlanta", "Portland",
    "Austin", "Phoenix", "Dallas", "Miami",
]
MONTHS = [
    "January", "February", "March", "April", "May", "June", "July",
    "August", "September", "October", "November", "December",
]
FRACTIONS = ["1/2", "1/4", "3/4", "2/3"]


def load_minimax(path: Path, limit: int) -> list[str]:
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    return lines[:limit]


def load_librispeech(root: Path, limit: int, rng: random.Random) -> list[str]:
    sentences = []
    for trans in sorted(root.rglob("*.trans.txt")):
        for line in trans.read_text(encoding="utf-8").splitlines():
            parts = line.split(" ", 1)
            if len(parts) != 2:
                continue
            text = parts[1].strip().lower()
            if not (40 <= len(text) <= 120):
                continue
            sentences.append(text[0].upper() + text[1:] + ".")
    rng.shuffle(sentences)
    return sentences[:limit]


def numeric_sentences(count: int, rng: random.Random) -> list[str]:
    out = []
    while len(out) < count:
        template = NUMERIC_TEMPLATES[len(out) % len(NUMERIC_TEMPLATES)]
        hour, minute = rng.randint(1, 12), rng.randint(0, 59)
        hour2, minute2 = rng.randint(1, 12), rng.randint(0, 59)
        n = rng.randint(2, 99)
        out.append(
            template.format(
                time=f"{hour}:{minute:02d} {rng.choice(['AM', 'PM'])}",
                time2=f"{hour2}:{minute2:02d} {rng.choice(['AM', 'PM'])}",
                month=rng.choice(MONTHS),
                ordinal=f"{n}{'th' if 11 <= n % 100 <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}",
                dollars=f"${rng.randint(1, 900)}.{rng.randint(0, 99):02d}",
                percent=f"{rng.randint(1, 99)}%",
                decimal=f"{rng.randint(1, 40)}.{rng.randint(1, 9)}",
                fraction=rng.choice(FRACTIONS),
                cardinal=str(rng.randint(2, 120)),
                cardinal2=str(rng.randint(2, 900)),
                year=str(rng.randint(1900, 2029)),
                year2=str(rng.randint(1900, 2029)),
            )
        )
    return out


def name_sentences(count: int, rng: random.Random) -> list[str]:
    names = [
        n.strip()
        for n in Path("/usr/share/dict/propernames").read_text().splitlines()
        if n.strip()
    ]
    out = []
    for i in range(count):
        template = NAME_TEMPLATES[i % len(NAME_TEMPLATES)]
        out.append(
            template.format(
                name=rng.choice(names),
                name2=rng.choice(names),
                city=rng.choice(CITIES),
            )
        )
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--minimax", required=True)
    parser.add_argument("--librispeech", required=True)
    parser.add_argument("--out", default="coreml/g2p/corpus_en_1000.txt")
    args = parser.parse_args()

    rng = random.Random(SEED)
    corpus = (
        load_minimax(Path(args.minimax).expanduser(), 100)
        + load_librispeech(Path(args.librispeech).expanduser(), 550, rng)
        + numeric_sentences(250, rng)
        + name_sentences(100, rng)
    )
    assert len(corpus) == 1000, len(corpus)
    Path(args.out).write_text("\n".join(corpus) + "\n", encoding="utf-8")
    print(f"wrote {len(corpus)} sentences to {args.out}")


if __name__ == "__main__":
    main()

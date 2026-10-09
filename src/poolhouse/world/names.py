"""Names for people, organisations and products that belong to nobody.

Every name here is assembled from syllables at the moment it is asked for, never read out
of a list. That is what makes the module safe to ship: the thing a repository must never
carry is a *known* person -- somebody a reader could point at -- and a syllable table knows
no one. A name it produces may well coincide with a person somewhere, the way any string of
letters may; what is refused is a name that arrived here *because* of a person, and none
did. So there is no denylist of plausible real names to keep current, because nothing here
could recognise one.

Six syllable families with different sound inventories, so a company of five thousand does
not read as one culture; a given name and a family name are drawn from families
independently, the way people are.
"""

from __future__ import annotations

import random
import re

__all__ = ["company_name", "person_name", "product_name", "slug"]

# Each family: (onsets, nuclei, codas, given-name endings, family-name endings). The
# inventories are sound shapes, not languages: "clustered and consonant-final", "open and
# vowel-final", and so on. Nothing in them is a word.
_FAMILIES: tuple[tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...],
                       tuple[str, ...], tuple[str, ...]], ...] = (
    # open syllables, vowel-final
    (("m", "l", "r", "n", "s", "v", "t", "d", "k", "z", "f"),
     ("a", "e", "i", "o", "u"),
     ("", "", "", "n", "l", "r"),
     ("a", "o", "ia", "io", "ela", "ino", "en", "el"),
     ("ari", "eno", "ova", "ini", "ale", "oro", "esu", "ande")),
    # clustered onsets, consonant-final
    (("br", "gr", "kr", "st", "th", "dr", "tr", "fl", "sk", "b", "g", "h"),
     ("a", "e", "i", "o", "u", "y"),
     ("n", "r", "k", "s", "th", "d", "l", "m"),
     ("", "", "a", "e", "is", "ar"),
     ("son", "wick", "by", "thorpe", "ard", "sten", "holm", "eth")),
    # long vowels and breathy onsets
    (("kh", "th", "ph", "sh", "ch", "r", "n", "m", "s", "j"),
     ("aa", "ee", "oo", "ai", "a", "i", "u"),
     ("n", "l", "m", "", "sh", "r"),
     ("a", "i", "an", "ir", "eh", "oon"),
     ("ani", "ari", "esh", "oor", "avi", "ini")),
    # doubled consonants, three short vowels
    (("b", "d", "g", "j", "n", "p", "t", "y", "w"),
     ("a", "i", "u", "o"),
     ("kk", "ll", "nn", "tt", "", "", "mb", "ng"),
     ("a", "i", "u", "o", "ai", "ei"),
     ("ata", "oro", "uki", "ande", "obi", "ulu", "ari")),
    # glides and open vowels
    (("y", "w", "h", "l", "m", "k", "t", "p", "n"),
     ("a", "e", "i", "o", "ai", "au", "ei"),
     ("", "", "", "h", "l", "n"),
     ("a", "i", "o", "ani", "eli", "oa"),
     ("hale", "ono", "lani", "eke", "ama", "ura")),
    # hard, short, sibilant
    (("p", "t", "k", "b", "z", "v", "x", "s", "c", "q"),
     ("a", "o", "u", "e", "i"),
     ("x", "z", "q", "k", "t", "s", "", "r"),
     ("o", "a", "ek", "ir", "ul", "an"),
     ("ek", "ov", "ur", "ash", "ic", "zer", "ast")),
)


def _syllable(rng: random.Random, family: int) -> str:
    onsets, nuclei, codas, _given, _family = _FAMILIES[family]
    return rng.choice(onsets) + rng.choice(nuclei) + rng.choice(codas)


def _word(rng: random.Random, family: int, syllables: int, endings: tuple[str, ...]) -> str:
    body = "".join(_syllable(rng, family) for _ in range(syllables))
    word = body + rng.choice(endings)
    # three consonants in a row is a seam, not a sound; so is a letter three times over
    word = re.sub(r"([^aeiouy])([^aeiouy])[^aeiouy]+", r"\1\2", word)
    word = re.sub(r"(.)\1\1+", r"\1\1", word)
    return word[:1].upper() + word[1:]


def person_name(rng: random.Random) -> tuple[str, str]:
    """A given name and a family name, from syllables and nothing else."""
    given_family = rng.randrange(len(_FAMILIES))
    # most people carry a family name from the same sound family; some do not
    family_family = given_family if rng.random() < 0.7 else rng.randrange(len(_FAMILIES))
    given = _word(rng, given_family, rng.choice((1, 1, 2, 2, 2)), _FAMILIES[given_family][3])
    family = _word(rng, family_family, rng.choice((1, 2, 2, 2, 3)), _FAMILIES[family_family][4])
    while not 3 <= len(given) <= 8:
        given = _word(rng, given_family, rng.choice((1, 2)), _FAMILIES[given_family][3])
    while not 4 <= len(family) <= 10:
        family = _word(rng, family_family, rng.choice((1, 2)), _FAMILIES[family_family][4])
    return given, family


# Organisation names are two invented stems glued together and a word that says what sort of
# thing it is. The stems are sound fragments, not words, so a compound is new every time.
_STEM_A = ("Harr", "Vell", "Corr", "Bran", "Stel", "Quen", "Marl", "Tarn", "Oss", "Wend",
           "Kest", "Pell", "Dunm", "Fenn", "Garr", "Hol", "Ash", "Thorn", "Cald", "Sax",
           "Verr", "Norr", "Lind", "Brim", "Tull", "Rav", "Elm", "Ost", "Wick", "Sel")
_STEM_B = ("owen", "ane", "ick", "ard", "more", "well", "brook", "field", "wick", "ford",
           "holt", "ridge", "dale", "gate", "mere", "worth", "stone", "lock", "vane", "cott")
_ORG_WORDS = ("Systems", "Labs", "Works", "Holdings", "Logistics", "Partners", "Group",
              "Technologies", "Industries", "Analytics", "Foundry", "Robotics", "Health",
              "Energy", "Software", "Studio", "Networks", "Capital", "Freight", "Instruments")
_PRODUCT_A = ("Kest", "Lum", "Tal", "Ori", "Vex", "Sol", "Har", "Nim", "Bel", "Cor", "Zin",
              "Ard", "Pol", "Mer", "Ely", "Ost", "Rav", "Til", "Wren", "Ilk")
_PRODUCT_B = ("rel", "en", "low", "on", "is", "ace", "bit", "vane", "ix", "ora", "ent",
              "ope", "ux", "ine", "ar", "ium", "et", "ade", "iq", "o")
_PRODUCT_WORDS = ("", "", "", " Board", " Cloud", " Core", " Edge", " Hub", " Mesh", " One",
                  " Pro", " Sense", " Sync", " Vault", " Studio")


def company_name(rng: random.Random, *, kind: str | None = None) -> str:
    """An organisation that does not exist: two stems and a word for its sort.

    ``kind`` chooses the sort word when the caller knows what sort it is; ``""`` leaves the
    stem on its own, for a caller with a word of its own to add.
    """
    stem = re.sub(r"(.)\1\1+", r"\1\1", rng.choice(_STEM_A) + rng.choice(_STEM_B))
    word = rng.choice(_ORG_WORDS) if kind is None else kind
    return f"{stem} {word}".strip()


def product_name(rng: random.Random) -> str:
    """A product name: one invented word, sometimes with a plain word after it."""
    return rng.choice(_PRODUCT_A) + rng.choice(_PRODUCT_B) + rng.choice(_PRODUCT_WORDS)


def slug(text: str) -> str:
    """Lower-case letters and digits joined by single hyphens; what an id is made of."""
    return re.sub(r"[^a-z0-9]+", "-", text.casefold()).strip("-")

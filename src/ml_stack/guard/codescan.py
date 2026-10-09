"""A coarse reading of a program passed on the command line (``python -c``, ``node -e``)."""

from __future__ import annotations

import re

from ml_stack.guard.harm import Finding

__all__ = ["check"]

MAX_CODE = 4000
DANGEROUS = re.compile(
    r"\b(?:remove|unlink|rmtree|rmdir|rename|replace|truncate|chmod|chown|kill|killpg|system|popen|"
    r"spawn|exec[a-z]*|fork|subprocess|child_process|shutil|rimraf|writeFile\w*|appendFile\w*|"
    r"unlinkSync|rmSync|fs|os|eval|compile|__import__|importlib|socket|urlopen|requests|fetch|"
    r"curl|wget|sendmail|smtplib|ftplib)\b|open\s*\([^)]*['\"][wax+]|rm\s+-|\bsh\b|\bbash\b")


def check(code: str, interpreter: str) -> Finding:
    """``destructive`` when the program names a file, process or network operation, else
    ``unsure`` (a program passed this way is never read in full)."""
    if len(code) > MAX_CODE:
        return Finding("unsure", f"a {interpreter} program too long to read")
    hit = DANGEROUS.search(code)
    if hit:
        return Finding("destructive", f"a {interpreter} one-liner that uses {hit.group(0).strip()} "
                       "(files, processes or the network)")
    return Finding("unsure", f"a {interpreter} one-liner the classifier does not read")

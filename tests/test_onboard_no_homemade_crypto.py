"""No module under poolhouse/fleet/onboard writes its own elliptic-curve or big-integer group
arithmetic. Pairing is the `spake2` package, signing and TLS are `cryptography`, HMAC and
hashes are the standard library. The scan looks for what such code is made of: a three-argument
`pow` (modular exponentiation or inverse), integer literals or shifts that name a curve or
group size, and functions named for group operations."""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "src" / "poolhouse" / "fleet" / "onboard"
BIG = 1 << 128
NAMES = ("scalarmult", "scalar_mult", "point_add", "pointadd", "modinv", "mod_inverse",
         "hash_to_curve", "ec_add", "ec_mul", "multiply_point", "double_point")


def findings(source: str) -> list[str]:
    out = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "pow" \
                and len(node.args) == 3:
            out.append(f"{node.lineno}: three-argument pow")
        if isinstance(node, ast.Constant) and isinstance(node.value, int) \
                and not isinstance(node.value, bool) and abs(node.value) >= BIG:
            out.append(f"{node.lineno}: integer literal of {node.value.bit_length()} bits")
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.LShift) \
                and isinstance(node.right, ast.Constant) and isinstance(node.right.value, int) \
                and node.right.value >= 128:
            out.append(f"{node.lineno}: shift by {node.right.value}")
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) \
                and any(n in node.name.lower() for n in NAMES):
            out.append(f"{node.lineno}: {node.name} is named for group arithmetic")
    return out


def test_the_scan_finds_the_arithmetic_it_is_looking_for():
    homemade = "P = 0xFFFFFFFF00000001000000000000000000000000FFFFFFFFFFFFFFFFFFFFFFFF\n" \
               "def point_add(a, b):\n    return pow(2 * a, -1, P)\nQ = 1 << 255\n"
    assert len(findings(homemade)) == 4


def test_no_onboarding_module_does_its_own_group_arithmetic():
    bad = {path.name: found for path in sorted(ROOT.glob("*.py"))
           if (found := findings(path.read_text()))}
    assert bad == {}, bad


def test_credentials_are_never_read_or_copied_by_onboarding():
    """A Hub token or an API key is never a file in a manifest or a field in a grant."""
    for path in sorted(ROOT.glob("*.py")):
        text = path.read_text()
        for word in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "ANTHROPIC_API_KEY", "poolhouse.credentials",
                     "credentials import"):
            assert word not in text, f"{path.name} mentions {word}"

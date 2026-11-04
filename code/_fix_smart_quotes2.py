"""
Fix stage5_destination_choice.py: reverse CP1252/Latin-1 mojibake then clean smart quotes.

The file was corrupted by reading UTF-8 bytes as Latin-1/CP1252 and re-saving as
Unicode.  For example:
  em-dash U+2014 (UTF-8: E2 80 94) was stored as â (U+00E2) + € (U+20AC) + " (U+201D)
  minus   U+2212 (UTF-8: E2 88 92) was stored as â (U+00E2) + ˆ (U+02C6) + ' (U+2019)
  box ─   U+2500 (UTF-8: E2 94 80) was stored as â (U+00E2) + " (U+201D) + € (U+20AC)
  box ━   U+2501 (UTF-8: E2 94 81) was stored as â (U+00E2) + " (U+201D) + \x81 (U+0081)

We must reverse ALL such sequences FIRST, before replacing any smart quotes, otherwise
the smart-quote sweep splits string literals that contain a mojibake triple internally.
"""
import ast
import sys

# ── Build CP1252 char-to-byte reverse mapping ────────────────────────────────
_cp1252_to_byte: dict[str, int] = {}
for _b in range(256):
    try:
        _c = bytes([_b]).decode("cp1252")
        _cp1252_to_byte[_c] = _b
    except (UnicodeDecodeError, ValueError):
        pass


def _get_byte(ch: str) -> int | None:
    """
    Return the raw byte value for a character in the mojibake-encoded file.

    - Code points 0-255: use Latin-1 (byte == code point).  This handles the
      C1 control characters (U+0081, etc.) that appear when an undefined CP1252
      byte was silently passed through.
    - Code points > 255: look up the CP1252 reverse table (handles U+201D, U+20AC,
      U+2020, U+2019, U+02C6, etc. which are the CP1252-extended characters that
      appear as continuation-byte stand-ins in the corrupted file).
    """
    cp = ord(ch)
    if cp < 256:
        return cp
    return _cp1252_to_byte.get(ch)


def reverse_mojibake(text: str) -> str:
    """
    Walk the string and, wherever a sequence of 2-4 characters whose byte values
    (via _get_byte) form a valid UTF-8 multi-byte sequence, replace them with the
    decoded Unicode code point.
    """
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        b0 = _get_byte(ch)

        # Check whether b0 is a valid UTF-8 leading byte.
        if b0 is None or b0 < 0xC2 or b0 > 0xF4:
            out.append(ch)
            i += 1
            continue

        # Determine expected continuation count.
        if 0xC2 <= b0 <= 0xDF:
            seq_len = 2
        elif 0xE0 <= b0 <= 0xEF:
            seq_len = 3
        else:
            seq_len = 4

        if i + seq_len > n:
            out.append(ch)
            i += 1
            continue

        # Collect continuation bytes (must be in range 0x80-0xBF).
        seq_bytes = [b0]
        valid = True
        for j in range(1, seq_len):
            cb = _get_byte(text[i + j])
            if cb is None or not (0x80 <= cb <= 0xBF):
                valid = False
                break
            seq_bytes.append(cb)

        if not valid:
            out.append(ch)
            i += 1
            continue

        # Attempt UTF-8 decode.
        try:
            decoded = bytes(seq_bytes).decode("utf-8")
        except UnicodeDecodeError:
            out.append(ch)
            i += 1
            continue

        # Accept only single-codepoint non-ASCII results.
        if len(decoded) == 1 and ord(decoded) > 0x7F:
            out.append(decoded)
            i += seq_len
        else:
            out.append(ch)
            i += 1

    return "".join(out)


# ── Load original file ────────────────────────────────────────────────────────
path = "stage5_destination_choice.py"
with open(path, encoding="utf-8") as f:
    src = f.read()

# Strip BOM if present.
if src.startswith("\ufeff"):
    src = src[1:]
    print("  Stripped BOM")

# ── Step 1: reverse mojibake ──────────────────────────────────────────────────
fixed = reverse_mojibake(src)
n_e2 = src.count("\u00e2")
print(f"  Mojibake pass: {n_e2} U+00E2 (â) leads processed")

# ── Step 2: replace remaining smart quotes / dashes ──────────────────────────
# After mojibake reversal the only remaining U+201x / U+2013 / U+2014 chars will
# be those that were used directly as string delimiters or in comments —
# safe to map to ASCII equivalents.
cleanups = [
    ("\u201c", '"'),  # left double quote  → straight double quote
    ("\u201d", '"'),  # right double quote → straight double quote
    ("\u2018", "'"),  # left single quote  → straight single quote
    ("\u2019", "'"),  # right single quote → straight single quote
    ("\u2013", "-"),  # en dash            → hyphen
    ("\u2014", "-"),  # em dash            → hyphen
]
for bad, good in cleanups:
    n = fixed.count(bad)
    if n:
        fixed = fixed.replace(bad, good)
        print(f"  U+{ord(bad):04X}: replaced {n} occurrences")

# ── Step 3: verify syntax ─────────────────────────────────────────────────────
try:
    ast.parse(fixed)
except SyntaxError as e:
    print(f"\nSTILL has SyntaxError: {e}")
    lines = fixed.splitlines()
    if e.lineno and 1 <= e.lineno <= len(lines):
        line = lines[e.lineno - 1]
        print(f"  Line {e.lineno}: {repr(line[:120])}")
        for j, c in enumerate(line):
            if ord(c) > 127:
                print(f"    col {j}: U+{ord(c):04X} {c!r}")
    sys.exit(1)

# ── Step 4: write fixed file ──────────────────────────────────────────────────
with open(path, "w", encoding="utf-8") as f:
    f.write(fixed)
print("OK -- file written and parses cleanly")

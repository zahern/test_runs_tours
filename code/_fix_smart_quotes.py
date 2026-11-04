"""
Fix stage5_destination_choice.py: reverse CP1252 mojibake and clean up smart quotes.

The file was corrupted by reading UTF-8 bytes as CP1252 and re-saving as Unicode.
For example, em-dash (UTF-8: E2 80 94) was stored as the three CP1252 characters
â (U+00E2) + € (U+20AC) + " (U+201D).  We must reverse ALL such sequences FIRST,
before replacing any smart quotes, otherwise the smart-quote sweep will split string
literals that contain a mojibake triple internally.
"""
import ast
import sys

# ── Build CP1252 char → byte mapping ────────────────────────────────────────
cp1252_to_byte: dict[str, int] = {}
for _b in range(256):
    try:
        _ch = bytes([_b]).decode("cp1252")
        cp1252_to_byte[_ch] = _b
    except (UnicodeDecodeError, ValueError):
        pass


def _get_byte(ch: str) -> int | None:
    """
    Return the byte value to use for mojibake reversal.

    For code points 0–255 we use Latin-1 (byte = code point), which handles
    both pure ASCII and C1 control characters that appear in the file when an
    undefined CP1252 byte (e.g. 0x81) was passed through.

    For code points above 255 (the CP1252 "extended" chars like U+201D, U+20AC,
    U+2020, etc.) we look up the CP1252 reverse table to get the original byte.
    """
    cp = ord(ch)
    if cp < 256:
        return cp          # Latin-1: byte equals code point
    return cp1252_to_byte.get(ch)  # CP1252 extended range
    """
    Replace every contiguous run of CP1252 characters whose byte values form a
    valid multi-byte UTF-8 sequence with the decoded Unicode character.

    Only sequences of length 2–4 are considered (covers U+0080 through U+10FFFF).
    """
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        b0 = cp1252_to_byte.get(ch)

        if b0 is None or b0 < 0xC2 or b0 > 0xF4:
            # Not a UTF-8 leading byte — pass through unchanged.
            out.append(ch)
            i += 1
            continue

        # Determine expected sequence length from the leading byte.
        if 0xC2 <= b0 <= 0xDF:
            seq_len = 2
        elif 0xE0 <= b0 <= 0xEF:
            seq_len = 3
        else:  # 0xF0-0xF4
            seq_len = 4

        if i + seq_len > n:
            out.append(ch)
            i += 1
            continue

        # Gather continuation bytes and verify they are in range 0x80-0xBF.
        seq_bytes = [b0]
        valid = True
        for j in range(1, seq_len):
            cb = cp1252_to_byte.get(text[i + j])
            if cb is None or not (0x80 <= cb <= 0xBF):
                valid = False
                break
            seq_bytes.append(cb)

        if not valid:
            out.append(ch)
            i += 1
            continue

        # Try to decode as UTF-8.
        try:
            decoded = bytes(seq_bytes).decode("utf-8")
        except UnicodeDecodeError:
            out.append(ch)
            i += 1
            continue

        # Only accept if it decoded to a single non-ASCII codepoint
        # (guard against spurious multi-char decodes).
        if len(decoded) == 1 and ord(decoded) > 0x7F:
            out.append(decoded)
            i += seq_len
        else:
            out.append(ch)
            i += 1

    return "".join(out)


# ── Load file ────────────────────────────────────────────────────────────────
path = "stage5_destination_choice.py"
with open(path, encoding="utf-8") as f:
    src = f.read()

# Strip BOM if present.
if src.startswith("\ufeff"):
    src = src[1:]
    print("  Stripped BOM")

# ── Step 1: reverse all mojibake ─────────────────────────────────────────────
fixed = reverse_mojibake(src)
n_mojibake = sum(1 for a, b in zip(src, fixed + " ") if a != b)
print(f"  Mojibake pass: ~{src.count(chr(0x00e2))} â-led sequences processed")

# ── Step 2: replace remaining smart quotes / dashes ─────────────────────────
# After mojibake reversal these will only appear as actual string delimiters
# or in comments/docstrings — safe to swap for ASCII equivalents.
cleanups = [
    ("\u201c", '"'),   # left double quote  → straight double quote
    ("\u201d", '"'),   # right double quote → straight double quote
    ("\u2018", "'"),   # left single quote  → straight single quote
    ("\u2019", "'"),   # right single quote → straight single quote
    ("\u2013", "-"),   # en dash            → hyphen
    ("\u2014", "-"),   # em dash            → hyphen
]
for bad, good in cleanups:
    n = fixed.count(bad)
    if n:
        fixed = fixed.replace(bad, good)
        print(f"  U+{ord(bad):04X}: replaced {n} occurrences")

# ── Step 3: verify ───────────────────────────────────────────────────────────
try:
    ast.parse(fixed)
except SyntaxError as e:
    print(f"\nSTILL has SyntaxError: {e}")
    # Show the offending line for diagnosis.
    lines = fixed.splitlines()
    if e.lineno and 1 <= e.lineno <= len(lines):
        line = lines[e.lineno - 1]
        print(f"  Line {e.lineno}: {repr(line[:120])}")
        bads = [(j, ord(c), c) for j, c in enumerate(line) if ord(c) > 127]
        for j, cp, c in bads[:5]:
            print(f"    col {j}: U+{cp:04X} {c!r}")
    sys.exit(1)

# ── Step 4: write ────────────────────────────────────────────────────────────
with open(path, "w", encoding="utf-8") as f:
    f.write(fixed)
print("OK -- file written and parses cleanly")

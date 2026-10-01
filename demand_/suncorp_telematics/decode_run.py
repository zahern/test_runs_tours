import base64, sys
encoded = sys.stdin.read().strip()
decoded = base64.b64decode(encoded).decode()
exec(decoded)

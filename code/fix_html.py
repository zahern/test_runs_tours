#!/usr/bin/env python
import sys
sys.path.insert(0, r'z:\test_runs_tours\code')

# Fix duplicate h2 style in HTML
with open('pipeline_logger.py', 'r') as f:
    content = f.read()

old = '''                "td{border:1px solid #ddd;padding:6px}</style></head><body>"
                "h2{color:#2c3e50;border-bottom:2px solid #2c3e50;padding-bottom:5px}"
                "h2{color:#2c3e50;border-bottom:2px solid #2c3e50;padding-bottom:5px}</style></head><body>"'''

new = '''                "td{border:1px solid #ddd;padding:6px}h2{color:#2c3e50;border-bottom:2px solid #2c3e50;padding-bottom:5px}</style></head><body>"'''

if old in content:
    content = content.replace(old, new)
    print("Fixed HTML style duplication")
    
    with open('pipeline_logger.py', 'w') as f:
        f.write(content)
    print("Saved!")
else:
    print("Pattern not found - checking actual content...")
    # Search for the pattern to understand the actual text
    if 'h2{color:#2c3e50;border-bottom:2px solid #2c3e50;padding-bottom:5px}' in content:
        count = content.count('h2{color:#2c3e50;border-bottom:2px solid #2c3e50;padding-bottom:5px}')
        print(f"Found {count} instances of h2 style - may not be duplicated as string literal")

import ast
with open('Z:/test_runs_tours/code/larch_jax_nested_logit_patch.py', 'r', encoding='utf-8') as f:
    content = f.read()
ast.parse(content)
print('syntax ok')
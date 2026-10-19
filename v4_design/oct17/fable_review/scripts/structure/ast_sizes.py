"""Top-level classes/functions with line counts for each starsim module."""
import ast, sys, pathlib
root = pathlib.Path('/home/cliffk/idm/ssv4/starsim')
files = sys.argv[1:] or sorted(root.glob('*.py'))
for f in files:
    f = pathlib.Path(f) if not isinstance(f, pathlib.Path) else f
    src = f.read_text()
    tree = ast.parse(src)
    total = src.count('\n')
    print(f'\n=== {f.name} ({total} lines) ===')
    rows = []
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            n = node.end_lineno - node.lineno + 1
            kind = 'class' if isinstance(node, ast.ClassDef) else 'def'
            bases = ''
            if kind == 'class':
                bases = '(' + ', '.join(ast.unparse(b) for b in node.bases) + ')'
                nmeth = sum(isinstance(x, ast.FunctionDef) for x in node.body)
                bases += f' [{nmeth} methods]'
            rows.append((n, f'{kind} {node.name}{bases}', node.lineno))
    for n, name, ln in rows:
        print(f'{n:5d}  L{ln:<5d} {name}')
    print(f'  sum of defs: {sum(r[0] for r in rows)}  / {total}')

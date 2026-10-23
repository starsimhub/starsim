import ast, sys, io, tokenize
for fn in sys.argv[1:]:
    src = open(fn).read(); lines = src.splitlines()
    doc = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module, ast.AsyncFunctionDef)) and node.body and isinstance(node.body[0], ast.Expr) and isinstance(getattr(node.body[0], 'value', None), ast.Constant) and isinstance(node.body[0].value.value, str):
            d = node.body[0]; doc |= set(range(d.lineno, d.end_lineno+1))
    blank = sum(1 for l in lines if not l.strip()); com = sum(1 for i,l in enumerate(lines,1) if l.strip().startswith('#') and i not in doc)
    print(f'{fn}: total {len(lines)} code {len(lines)-blank-com-len(doc)} doc {len(doc)} comment {com} blank {blank}')

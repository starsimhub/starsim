"""Find top-level functions/classes and class methods in starsim/ with no reference anywhere else in the repo (incl. docs, tests, library, v4_design examples)."""
import ast, re, pathlib, collections
R = pathlib.Path('/home/cliffk/idm/ssv4')
V = pathlib.Path('/home/cliffk/idm/lingua_franca_design/vignettes')
core = sorted((R/'starsim').glob('*.py')) + sorted((R/'starsim/interop').glob('*.py'))
allfiles = (list((R/'starsim').rglob('*.py')) + list((R/'tests').glob('*.py')) + list((R/'docs').rglob('*.qmd'))
            + list((R/'docs').rglob('*.py')) + list((R/'v4_design').rglob('*.py')) + list((R/'v4_design').rglob('*.md'))
            + list(V.rglob('*.py')))
texts = {p: p.read_text(errors='ignore') for p in allfiles}
big = '\n'.join(texts.values())

def count(name, exclude_text):
    pat = re.compile(r'(?<![A-Za-z0-9_])' + re.escape(name) + r'(?![A-Za-z0-9_])')
    return len(pat.findall(big)) - len(pat.findall(exclude_text))

dunder = re.compile(r'^__.*__$')
hooks = {'init_pre','init_post','init_results','init_vals','init_pars','init_mixing','step','step_state','step_die','start_step','finish_step','finalize','finalize_results','update_results','validate','update','make_contacts','add_pairs','set_prognoses','infect','disp','brief','plot','plot_hist','plot_all','start_run','finish_run','to_json','to_df','to_dict','finish','set_initial_states','init_people','shrink','summarize','describe','net_beta','step_contacts','end_step','begin_step'}
out_top, out_meth = [], []
for f in core:
    src = texts[f]
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            n = node.end_lineno - node.lineno + 1
            if node.name.startswith('_') and not node.name.startswith('__'):
                pass
            defs = re.compile(r'^(class|def)\s+' + re.escape(node.name) + r'\b', re.M)
            # references elsewhere: total occurrences minus definition lines
            pat = re.compile(r'(?<![A-Za-z0-9_])' + re.escape(node.name) + r'(?![A-Za-z0-9_])')
            total = len(pat.findall(big))
            ndefs = len(defs.findall(big))
            # occurrences within own body (self-recursion / docstring)
            own = len(pat.findall(ast.get_source_segment(src, node) or ''))
            ext = total - own
            if ext <= 0:
                out_top.append((n, f.name, node.lineno, ('class ' if isinstance(node, ast.ClassDef) else 'def ') + node.name, total-ndefs))
            if isinstance(node, ast.ClassDef):
                for m in node.body:
                    if isinstance(m, ast.FunctionDef) and not dunder.match(m.name) and m.name not in hooks:
                        mn = m.end_lineno - m.lineno + 1
                        mpat = re.compile(r'(?<![A-Za-z0-9_])' + re.escape(m.name) + r'(?![A-Za-z0-9_])')
                        mtotal = len(mpat.findall(big))
                        mown = len(mpat.findall(ast.get_source_segment(src, m) or ''))
                        if mtotal - mown <= 0 and mn >= 3:
                            out_meth.append((mn, f.name, m.lineno, f'{node.name}.{m.name}'))
print('=== Top-level defs with no reference outside their own body anywhere in repo+docs+vignettes ===')
for r in sorted(out_top, reverse=True):
    print(f'{r[0]:5d}  {r[1]}:{r[2]}  {r[3]}')
print(f'total unreferenced top-level lines: {sum(r[0] for r in out_top)} in {len(out_top)} defs')
print('\n=== Methods (non-dunder, non-hook, >=3 lines) whose name appears nowhere else ===')
for r in sorted(out_meth, reverse=True):
    print(f'{r[0]:5d}  {r[1]}:{r[2]}  {r[3]}')
print(f'total unreferenced method lines: {sum(r[0] for r in out_meth)} in {len(out_meth)} methods')

"""
Count what a piece of code does on the CPU, with Linux's hardware counters (perf), rather than time it

Counts of instructions, floating-point operations, loads and stores depend on the code and the CPU, but hardly on what
else the machine is doing, so two versions of a kernel can be compared on a busy machine; the top-down breakdown says
where the cycles went (useful work, waiting on memory, the front end, mispredicted branches); --record says which
functions they went in; and --lines, which source lines, with the time in a libc function (memset, memcpy) given to
the line that called it.

Mark the part to count in the script (otherwise the whole process is counted, startup and imports included):

    import perfcount # This file, which is on the path when perfcount runs the script
    with perfcount.region():
        sim.run()

Then:

    python perfcount.py script.py [args]                  # Counts, for the marked region
    python perfcount.py --groups ops,topdown script.py    # Only some groups of counters
    python perfcount.py --record script.py                # And the functions the cycles were spent in
    python perfcount.py --lines script.py                 # And the source lines
    python perfcount.py --compare a.py b.py               # Two scripts side by side, with the ratio

Each group of counters is counted in a run of its own, pinned to one performance core, so that no counter is shared
(multiplexed) and the counts are exact; the script should do the same thing each time (e.g. a fixed seed), and is
best a few seconds long (repeat the run inside the region) so the samples of --record and --lines are many.

For lines in the Rust kernels, build them with line tables, which leaves the optimized code as it is:

    cd starsim/solvers/rust && CARGO_PROFILE_RELEASE_DEBUG=line-tables-only ./build

The Numba kernels are loaded from Numba's cache, so perf cannot name them: they show as [JIT], though their counts
are counted. Needs Linux, perf (linux-tools), and binutils (addr2line, c++filt, nm); --lines needs a CPU that records
the last branches taken (Intel's LBR). The event names are those of recent Intel CPUs.
"""
import os
import re
import sys
import shutil
import argparse
import tempfile
import subprocess
import collections
from contextlib import contextmanager

here = os.path.dirname(os.path.abspath(__file__))

groups = dict( # Few enough counters in each to be counted at once on one core
    ops = ['instructions', 'cycles', 'branches', 'mem_inst_retired.all_loads', 'mem_inst_retired.all_stores',
           'fp_arith_inst_retired.scalar_double', 'fp_arith_inst_retired.128b_packed_double', 'fp_arith_inst_retired.256b_packed_double'],
    topdown = ['slots', 'topdown-retiring', 'topdown-bad-spec', 'topdown-fe-bound', 'topdown-be-bound', 'topdown-mem-bound', 'topdown-heavy-ops'],
    cache = ['instructions', 'cycles', 'branch-misses', 'L1-dcache-load-misses', 'LLC-load-misses', 'dTLB-load-misses'],
)
libc = ('libc.so', 'libm.so', 'ld-linux') # Libraries whose time is given to the line that called them (see lines())


#%% In the script being measured

@contextmanager
def region():
    """ Count only what happens inside this block (when the script is run by perfcount; otherwise it does nothing) """
    ctl, ack = os.environ.get('PERFCOUNT_CTL'), os.environ.get('PERFCOUNT_ACK')
    if not ctl:
        yield
        return
    def send(cmd):
        with open(ctl, 'w') as f:
            f.write(f'{cmd}\n')
        with open(ack) as f:
            f.read(5)
    send('enable')
    try:
        yield
    finally:
        send('disable')


#%% Running perf

def perf_cpu():
    """ A performance core to pin to: the third (not core 0, which takes more of the system's interrupts), on a CPU with two kinds of core """
    path = '/sys/devices/cpu_core/cpus'
    first = int(open(path).read().strip().split(',')[0].split('-')[0]) if os.path.exists(path) else 0
    return first + 2


def qualify(event):
    """ An event on the performance cores, on a CPU with two kinds of core """
    return f'cpu_core/{event}/' if os.path.exists('/sys/devices/cpu_core') else event


def run_perf(mode, events, cmd, cpu, marked):
    """ Run a script under perf stat or perf record, counting only its marked region if it has one; returns perf's output file, in a folder to remove """
    tmp = tempfile.mkdtemp(prefix='perfcount_')
    ctl, ack = os.path.join(tmp, 'ctl'), os.path.join(tmp, 'ack')
    os.mkfifo(ctl)
    os.mkfifo(ack)
    fifos = dict(PERFCOUNT_CTL=ctl, PERFCOUNT_ACK=ack) if marked else {} # Only when perf reads them: otherwise a region would wait on them forever
    env = dict(os.environ, **fifos, PYTHONPATH=here + os.pathsep + os.environ.get('PYTHONPATH', ''),
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMBA_NUM_THREADS='1', MPLBACKEND='agg')
    out = os.path.join(tmp, 'out')
    if mode == 'stat':
        perf = ['perf', 'stat', '-x', ',', '-o', out, '-e', ','.join(qualify(e) for e in events)]
    else:
        perf = ['perf', 'record', '-F', '4999', '-o', out, '-e', qualify('cycles')] + (['--call-graph', 'lbr'] if mode == 'lines' else [])
    control = ['--control', f'fifo:{ctl},{ack}', '-D', '-1'] if marked else []
    full = ['taskset', '-c', str(cpu)] + perf + control + ['--', sys.executable, '-X', 'perf'] + cmd # -X perf: Python's own functions are named
    done = subprocess.run(full, env=env, capture_output=True, text=True)
    if done.returncode:
        shutil.rmtree(tmp, ignore_errors=True)
        raise RuntimeError(f'{" ".join(full)} failed:\n{done.stdout[-3000:]}\n{done.stderr[-3000:]}')
    return out


def stat(events, cmd, cpu, marked):
    """ The counts of some events, by name """
    out = run_perf('stat', events, cmd, cpu, marked)
    counts = {}
    for line in open(out):
        parts = line.strip().split(',')
        if len(parts) < 3 or line.startswith('#'):
            continue
        value, name = parts[0], parts[2].replace('cpu_core/', '').rstrip('/')
        counts[name] = float(value) if value.replace('.', '').isdigit() else float('nan')
    shutil.rmtree(os.path.dirname(out), ignore_errors=True)
    return counts


def demangle(syms):
    """ Rust and C++ symbols as their paths (e.g. _rs::kernels::step), without Rust's crate hash """
    syms = list(syms)
    if not syms:
        return {}
    done = subprocess.run(['c++filt'], input='\n'.join(syms), capture_output=True, text=True).stdout.splitlines()
    return {s:re.sub(r'\[[0-9a-f]{16}\]', '', d) for s,d in zip(syms, done)}


def short(lib):
    """ A library's name without its folder or Python's suffix """
    return os.path.basename(lib).split('.cpython')[0]


def record(cmd, cpu, marked, top=25):
    """ The functions the cycles were spent in, by their own time (not that of what they call) """
    out = run_perf('record', [], cmd, cpu, marked)
    rep = subprocess.run(['perf', 'report', '-i', out, '--no-children', '--sort', 'dso,symbol', '--stdio', '--percent-limit', '0.5'], capture_output=True, text=True).stdout
    shutil.rmtree(os.path.dirname(out), ignore_errors=True)
    rows = []
    for line in rep.splitlines():
        parts = line.split()
        if parts and not line.startswith('#') and parts[0].endswith('%') and '] ' in line:
            rows.append((parts[0], short(parts[1]), line.split('] ', 1)[1].rsplit(' -', 2)[0].strip()))
    names = demangle(r[2] for r in rows)
    return [f'{pct:>7s}  {lib[:26]:26s}  {names[sym]}' for pct,lib,sym in rows[:top]]


def lines(cmd, cpu, marked, top=25):
    """
    The source lines the cycles were spent on, from samples of the last branches taken (LBR): each sample is given to
    the line it was on, or, inside libc, to the line that called it (so a memset is given to the loop it replaced)
    """
    def samples(mode):
        """ The frames of each sample, innermost first, as (symbol, offset, library) """
        out = run_perf(mode, [], cmd, cpu, marked)
        script = subprocess.run(['perf', 'script', '-i', out, '-F', 'ip,sym,symoff,dso', '--no-inline'], capture_output=True, text=True).stdout # perf report can crash on call sites by address in a library with line tables
        shutil.rmtree(os.path.dirname(out), ignore_errors=True)
        for block in script.split('\n\n' if mode == 'lines' else '\n'): # A sample is a block of frames, or a line without call chains
            frames = []
            for line in block.strip().splitlines():
                m = re.match(r'\s*([0-9a-f]+)\s+(\S+?)(?:\+0x([0-9a-f]+))?\s+\((.*)\)', line)
                if m:
                    frames.append((m.group(2), int(m.group(3) or '0', 16), m.group(4)))
            if frames:
                yield frames

    # Two recordings: without call chains, where each sample's address is exact; and with the last branches, whose innermost frame is
    # only its function's start, but which says where each call into libc came from
    counts = collections.Counter()
    plain = list(samples('record'))
    if not plain:
        return ['(no samples)']
    total = len(plain)
    in_libc = 0
    for frames in plain:
        sym, off, lib = frames[0]
        if any(k in lib for k in libc):
            in_libc += 1
        else:
            counts[(lib, sym, off, '')] += 1
    if in_libc:
        calls = collections.Counter()
        for frames in samples('lines'):
            if any(k in frames[0][2] for k in libc) and len(frames) > 1:
                sym, off, lib = frames[1]
                calls[(lib, sym, max(off - 1, 0), frames[0][0])] += 1 # The return address is just past the call
        n = sum(calls.values())
        for key,c in calls.items(): # The plain recording's share in libc, split as the calls into it are
            counts[key] += in_libc*c/n
    bases = {} # The address of each symbol in its library, for addr2line
    for lib in {k[0] for k in counts}:
        if os.path.exists(lib):
            nm = subprocess.run(['nm', '--defined-only', lib], capture_output=True, text=True).stdout
            bases[lib] = {p[2]:int(p[0], 16) for p in (l.split() for l in nm.splitlines()) if len(p) == 3}
    where = collections.Counter()
    for (lib,sym,off,via),n in counts.items():
        base = bases.get(lib, {}).get(sym)
        src = ''
        if base is not None:
            chain = subprocess.run(['addr2line', '-e', lib, '-i', hex(base + off)], capture_output=True, text=True).stdout.split()
            own = [c for c in chain if not c.startswith('??') and '/rustc/' not in c and '/.cargo/' not in c] # Inlined frames, innermost first, without Rust's library (iterators, comparisons)
            if own and all(c.endswith(':?') for c in own[:1]):
                src = '(no line: code the compiler added)'
            elif own:
                file = lambda c: c.rsplit(':', 1)[0] # noqa: E731
                outer = next((c for c in own[1:] if file(c) != file(own[0]) and not c.endswith(':?')), None) # The line in another file that used it (a helper's caller)
                src = ' <- '.join(re.sub(r'.*/(src|starsim)/', '', c) for c in [own[0]] + ([outer] if outer else []))
        where[(short(lib), sym, src or '(no line tables)', via)] += n
    names = demangle({k[1] for k in where} | {k[3] for k in where if k[3]})
    rows = []
    for (lib,sym,src,via),n in where.most_common(top):
        called = f' (in {names[via]})' if via else ''
        rows.append(f'{100*n/total:6.1f}%  {lib[:20]:20s}  {names[sym][:40]:40s}  {src}{called}')
    return rows


#%% Reporting

def derived(c):
    """ The numbers worth reading, from the counts """
    out = {}
    if 'instructions' in c and 'cycles' in c:
        out['instructions per cycle'] = c['instructions']/c['cycles']
    if 'fp_arith_inst_retired.scalar_double' in c:
        vec = 2*c.get('fp_arith_inst_retired.128b_packed_double', 0) + 4*c.get('fp_arith_inst_retired.256b_packed_double', 0)
        flops = c['fp_arith_inst_retired.scalar_double'] + vec
        out['double flops'] = flops
        out['share of flops vectorized'] = vec/flops if flops else 0.0
        out['instructions per flop'] = c['instructions']/flops if flops else float('nan')
    if 'mem_inst_retired.all_loads' in c:
        out['loads per instruction'] = c['mem_inst_retired.all_loads']/c['instructions']
        out['stores per instruction'] = c['mem_inst_retired.all_stores']/c['instructions']
    if 'slots' in c:
        for k in ['retiring', 'bad-spec', 'fe-bound', 'be-bound', 'mem-bound', 'heavy-ops']:
            if f'topdown-{k}' in c:
                out[f'{k} (share of slots)'] = c[f'topdown-{k}']/c['slots']
    if 'branch-misses' in c:
        out['branch misses per 1000 instructions'] = 1000*c['branch-misses']/c['instructions']
        out['L1 load misses per 1000 instructions'] = 1000*c['L1-dcache-load-misses']/c['instructions']
        out['LLC load misses per 1000 instructions'] = 1000*c['LLC-load-misses']/c['instructions']
    return out


def fmt(v):
    """ A count with commas, a ratio to three figures """
    if v != v:
        return 'n/a'
    return f'{v:,.0f}' if abs(v) >= 1e4 else f'{v:.3g}'


def show(results, labels):
    """ A table of the counts and the numbers derived from them, a column for each script, and their ratio for two """
    keys = list(dict.fromkeys(k for r in results for k in r[0]))
    dkeys = list(dict.fromkeys(k for r in results for k in r[1]))
    w = max(len(k) for k in keys + dkeys) + 2
    print(f'{"":{w}s}' + ''.join(f'{l[-22:]:>24s}' for l in labels) + ('       ratio' if len(results) == 2 else ''))
    for section,ks,i in [('Counts', keys, 0), ('Derived', dkeys, 1)]:
        print(section)
        for k in ks:
            vals = [r[i].get(k, float('nan')) for r in results]
            ratio = f'{vals[1]/vals[0]:12.3f}' if len(vals) == 2 and vals[0] else ''
            print(f'  {k:{w-2}s}' + ''.join(f'{fmt(v):>24s}' for v in vals) + ratio)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0], epilog=__doc__.split('\n\n', 1)[1], formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--groups', default='ops,topdown,cache', help=f'the groups of counters, from {", ".join(groups)} (default: all)')
    parser.add_argument('--cpu', type=int, default=None, help='the core to pin to (default: a performance core)')
    parser.add_argument('--record', action='store_true', help='also the functions the cycles were spent in')
    parser.add_argument('--lines', action='store_true', help='also the source lines they were spent on')
    parser.add_argument('--whole', action='store_true', help='count the whole process, even if the script marks a region')
    parser.add_argument('--compare', action='store_true', help='the arguments are two or more scripts (with no arguments of their own), shown side by side')
    parser.add_argument('cmd', nargs=argparse.REMAINDER, help='the script and its arguments')
    args = parser.parse_args()
    if not args.cmd:
        parser.error('give a script to run')
    if not shutil.which('perf'):
        parser.error('perf is not installed (e.g. apt install linux-tools-generic)')
    cpu = perf_cpu() if args.cpu is None else args.cpu
    names = [g.strip() for g in args.groups.split(',') if g.strip()]
    cmds = [[c] for c in args.cmd] if args.compare else [args.cmd]
    marks = [not args.whole and 'perfcount.region' in open(cmd[0]).read() for cmd in cmds]
    results = []
    for cmd,marked in zip(cmds, marks):
        if not marked:
            print(f'Note: {cmd[0]} marks no region (with perfcount.region()), so the whole process is counted, startup and imports included')
        counts = {}
        for g in names:
            counts.update(stat(groups[g], cmd, cpu, marked))
        results.append((counts, derived(counts)))
    if names:
        show(results, [' '.join(c) for c in cmds])
    for cmd,marked in zip(cmds, marks):
        if args.record:
            print(f'\nThe functions the cycles were spent in, in {" ".join(cmd)} (their own time):')
            print('\n'.join(record(cmd, cpu, marked)))
        if args.lines:
            print(f'\nThe lines they were spent on, in {" ".join(cmd)} (inside libc, the line that called it):')
            print('\n'.join(lines(cmd, cpu, marked)))


if __name__ == '__main__':
    main()

"""
The ODE in NumPy: the RK4 and Euler step of `ss.kernels.step()`, vectorized over groups (and over the flows whose
rates have the same operations), so that a deterministic run needs no compiling

It takes the same program, and gives the same results to rounding. Anything else (the stochastic solvers, the
entry points other than `run()` and `one_step()`, and a step with an infinite rate) is passed to the kernels as
plain Python (`ss.kernels.uncompiled()`).
"""
import os
import numpy as np
from . import kernels as K

mode = os.getenv('STARSIM_ODE', 'auto') # 'auto': used for the ODE when the kernels are not compiled; 'always'; or 'never'
_layouts = {} # The layout of each program, by its id, with the program kept so the id is not reused


def __getattr__(name):
    """ The other entry points (apply_events, apply_shifts, run_many, ...), from the kernels as plain Python """
    return getattr(K.uncompiled(), name)


def wanted(solver, p=None, y=None):
    """
    Whether a run of this solver is taken here: by default (mode='auto'), for the ODE when the kernels are not compiled
    and the program is large enough that NumPy is faster than the kernels as plain Python. Each time the rates are worked
    out, NumPy takes about 40 µs whatever the size, and plain Python about 10 µs plus 0.6 µs for each operation of
    `ss.kernels.work()` (an SIR in one group: 0.55 s against 0.22 s; in 20 places: 0.60 s against 2.8 s)
    """
    import starsim as ss
    if solver != 'ode' or mode == 'never':
        return False
    if mode == 'always':
        return True
    return ss.options.kernels != 'numba' and not K.chosen and not K.is_compiled() and (p is None or K.work(p, y) > 50)


def matrix(rows, cols, shape):
    """ A matrix with a 1 at each (row, col), added if repeated """
    out = np.zeros(shape)
    np.add.at(out, (np.asarray(rows, dtype=int), np.asarray(cols, dtype=int)), 1.0)
    return out


class Layout:
    """ What a program's structure gives, once: who goes where, as index arrays and matrices of ones """
    def __init__(self, p, nc, ng):
        nf = len(p.src)
        nflows = p.n_trans + p.n_movers
        self.nf = nf
        out = np.flatnonzero(p.src >= 0)
        self.out, self.out_src = out, p.src[out]
        self.ins = np.flatnonzero(p.src < 0)
        self.leave = matrix(p.src[out], out, (nc, nf)) # Compartments by flows
        named = p.named.astype(bool)
        plain = np.flatnonzero(named & (p.record < 0))
        self.named = matrix(p.index[plain], plain, (nflows, nf))
        self.recorded = [(f, p.index[f], p.record[f]) for f in np.flatnonzero(named & (p.record >= 0))]
        tf = np.repeat(np.arange(nf), np.diff(p.tgt_starts)) # The flow of each target
        self.tf = tf
        unmapped = np.flatnonzero(p.tgt_map < 0)
        self.arrive = matrix(p.tgt_comp[unmapped], unmapped, (nc, len(tf)))
        self.mapped = [(m, np.flatnonzero(p.tgt_map == m)) for m in np.unique(p.tgt_map[p.tgt_map >= 0])]
        self.mapped = [(m, ks, matrix(p.tgt_comp[ks], np.arange(len(ks)), (nc, len(ks)))) for m,ks in self.mapped]
        rec = np.flatnonzero(p.tgt_rec >= 0)
        self.tgt_rec = matrix(p.tgt_rec[rec], rec, (nflows, len(tf)))
        self.cf = np.repeat(np.arange(nf), np.diff(p.crd_starts))
        self.credit = matrix(p.crd_index, np.arange(len(self.cf)), (nflows, len(self.cf)))
        self.tape(p)
        sets = len(p.set_starts) - 1
        self.sets = matrix(np.repeat(np.arange(sets), np.diff(p.set_starts)), p.set_comps, (sets, nc)) # Sets by compartments
        self.place = matrix(np.arange(ng), p.places, (ng, max(p.n_places, 1))) # Groups by places
        # Movers: each compartment of each mover, by each of its entries
        cc, ee, mm = [], [], []
        for m in range(p.n_movers):
            comps = p.mov_comps[p.mov_starts[0, m]:p.mov_starts[0, m+1]]
            ents = np.arange(p.mov_starts[1, m], p.mov_starts[1, m+1])
            cc.append(np.repeat(comps, len(ents)))
            ee.append(np.tile(ents, len(comps)))
            mm.append(np.full(len(comps)*len(ents), p.n_trans + m))
        cat = lambda x: np.concatenate(x).astype(int) if x else np.zeros(0, dtype=int)
        self.mov_c, self.mov_e, self.mov_m = cat(cc), cat(ee), cat(mm)
        # Conveyors: the flow out of each, its slots, and the flows that are exits of its last slot (see ss.kernels.release_rates())
        self.conveyors = []
        for i in range(len(p.starts) - nf - 1):
            a, b = p.starts[nf+i], p.starts[nf+i+1]
            if p.ops[a] == K.CONVEY:
                self.conveyors.append((p.args[a], p.args[a+1:b]))
        is_last = np.zeros(nc, dtype=bool)
        for _,slots in self.conveyors:
            is_last[slots[-1]] = True
        self.exits = []
        for f_out,slots in self.conveyors:
            ex = [f for f in range(nf) if p.src[f] == slots[-1] and (p.tgt_starts[f+1] == p.tgt_starts[f] or not is_last[p.tgt_comp[p.tgt_starts[f]:p.tgt_starts[f+1]]].all())]
            self.exits.append(np.array(ex, dtype=int))
        # The time share between places, as a matrix
        self.share = np.zeros((max(p.n_places, 1),)*2)
        np.add.at(self.share, (p.sh_pair[0], p.sh_pair[1]), p.sh_val)
        return


    def tape(self, p):
        """
        The operations of every rate as one graph, each part once (a part used by many rates, or twice in one, is worked out once),
        in levels: each part comes after those it uses, and the parts of a level with the same operation are worked out in one call
        """
        nodes, level = {}, []
        def node(key, lev):
            if key not in nodes:
                nodes[key] = len(nodes)
                level.append(lev)
            return nodes[key]
        self.rate_fs, self.rate_nodes = [], []
        shared = {} # The parts several rates share, which come first (see ss.kernels.LOAD)
        for f in range(-1, self.nf):
            stack, regs = [], dict(shared)
            for k in range(0 if f < 0 else p.starts[f], p.starts[f+1]):
                op, arg = int(p.ops[k]), int(p.args[k])
                if op in (K.CONST, K.EXPOSURE, K.COUNT, K.FRAC, K.TOTAL, K.GROUP):
                    stack.append(node((op, arg), 0))
                elif op == K.LOAD:
                    stack.append(regs[arg])
                elif op == K.STORE:
                    regs[arg] = stack.pop()
                elif op >= K.EXP:
                    c = stack.pop()
                    stack.append(node((op, c), level[c] + 1))
                else:
                    b, a = stack.pop(), stack.pop()
                    stack.append(node((op, a, b), max(level[a], level[b]) + 1))
            if f < 0:
                shared = regs
            elif stack:
                self.rate_fs.append(f)
                self.rate_nodes.append(stack[0])
        self.rate_fs, self.rate_nodes = np.array(self.rate_fs, dtype=int), np.array(self.rate_nodes, dtype=int)
        self.n_nodes = len(nodes)
        groups = {}
        for key,i in nodes.items():
            groups.setdefault((level[i], key[0]), []).append((i,) + key[1:])
        self.steps = []
        for (lev,op),items in sorted(groups.items()):
            cols = np.array(items, dtype=int).T # The node, then what it acts on
            self.steps.append((op, *cols))
        self.counts = any(op in (K.COUNT, K.FRAC, K.TOTAL, K.GROUP) for op,*_ in self.steps)
        return


def layout(p, nc, ng):
    key = id(p)
    if key not in _layouts or _layouts[key][0] is not p:
        if len(_layouts) > 100:
            _layouts.clear()
        _layouts[key] = (p, Layout(p, nc, ng))
    return _layouts[key][1]


def unary(op, x):
    return {K.EXP:np.exp, K.LOG:np.log, K.SQRT:np.sqrt, K.ABS:np.abs, K.SIN:np.sin, K.COS:np.cos}[op](x)


def binary(op, a, b):
    if   op == K.ADD: return a + b
    elif op == K.SUB: return a - b
    elif op == K.MUL: return a * b
    elif op == K.DIV: return np.where(a != 0, a/np.where(a != 0, b, 1.0), 0.0)
    elif op == K.POW: return a ** b
    elif op == K.MIN: return np.minimum(a, b)
    elif op == K.MAX: return np.maximum(a, b)
    elif op == K.LT:  return (a < b).astype(float)
    elif op == K.LE:  return (a <= b).astype(float)
    elif op == K.GT:  return (a > b).astype(float)
    elif op == K.GE:  return (a >= b).astype(float)
    raise ValueError(op)


def rates_of(ys, p, L):
    """ The rate of every flow, by group, as `ss.kernels.step()` works them out (not by place: that is only for the exact solver) """
    rates = p.fixed.copy()
    if not len(L.rate_fs):
        return rates
    ng = ys.shape[1]
    n = ys[p.people].sum(axis=0)
    local_n = (n @ L.place)[p.places]
    expo = np.zeros((len(p.exp_set), ng))
    for e,s in enumerate(p.exp_set):
        if s < 0:
            expo[e] = expo[e+s:e].sum(axis=0)
            continue
        infectious = (ys[p.set_comps[p.set_starts[s]:p.set_starts[s+1]]]*p.rel_trans[e]).sum(axis=0)
        for t in range(len(p.sizes)):
            size, block, lev = p.sizes[t], p.block[t], p.levels[t]
            num = np.bincount(lev, infectious, minlength=size)[:size]
            den = np.bincount(lev, n, minlength=size)[:size]
            if p.use[0, t]: # Move people to where they spend their time
                num = (L.share.T @ num.reshape(-1, block)).ravel()
                den = (L.share.T @ den.reshape(-1, block)).ravel()
            prev = np.divide(num, den, out=np.zeros(size), where=den > 0)
            if p.use[1, t]: # Mix between the levels in each place
                prev = p.mat[t][:size, :size] @ prev if block == size else (prev.reshape(-1, block) @ p.mat[t][:block, :block].T).ravel()
            if p.use[2, t]: # Mix between places
                prev = (L.share @ prev.reshape(-1, block)).ravel()
            c = p.contacts[e, t] if p.contacts.shape[0] > 1 else p.contacts[0, t]
            expo[e] += c*prev[lev]
    if L.counts:
        setsum = L.sets @ ys # Sets by groups
        local = (setsum @ L.place)[:, p.places]
        total = setsum.sum(axis=1, keepdims=True)
    V = np.empty((L.n_nodes, ng))
    for op,out,*x in L.steps:
        if op == K.CONST:
            V[out] = p.consts[x[0]]
        elif op == K.EXPOSURE:
            V[out] = expo[x[0]]
        elif op == K.TOTAL:
            V[out] = total[x[0]]
        elif op == K.COUNT:
            V[out] = local[x[0]]
        elif op == K.GROUP:
            V[out] = setsum[x[0]]
        elif op == K.FRAC:
            V[out] = np.divide(local[x[0]], local_n, out=np.zeros((len(out), ng)), where=local_n > 0)
        elif op >= K.EXP:
            V[out] = unary(op, V[x[0]])
        else:
            V[out] = binary(op, V[x[0]], V[x[1]])
    rates[L.rate_fs] = V[L.rate_nodes]*p.mult[L.rate_fs]
    return rates


def derivs(ys, p, L, rates, h, euler):
    """ The rate of change of each compartment, and the rate of each transition and mover, and the people moved in by the movers """
    nc, ng = ys.shape
    nflows = p.n_trans + p.n_movers
    dy = np.zeros((nc, ng))
    df = np.zeros((nflows, ng))
    moved_in = None
    if len(L.mov_c):
        r = p.mov_rate[L.mov_e]
        g, j = p.mov_pair[0, L.mov_e], p.mov_pair[1, L.mov_e]
        amt = ys[L.mov_c, g]*r
        np.add.at(dy, (L.mov_c, g), -amt)
        np.add.at(dy, (L.mov_c, j), amt)
        np.add.at(df, (L.mov_m, g), amt)
        moved_in = np.zeros((nc, ng))
        np.add.at(moved_in, (L.mov_c, j), amt)
    amount = np.empty_like(rates)
    amount[L.out] = rates[L.out]*ys[L.out_src]
    if euler: # A compartment whose outflows would take more than everyone in it in a substep has them scaled down together
        tot = L.leave @ (rates*h)
        scale = np.where(tot > 1, 1.0/np.where(tot > 1, tot, 1.0), 1.0)
        amount[L.out] *= scale[L.out_src]
    amount[L.ins] = rates[L.ins]*p.inflow[L.ins]
    dy -= L.leave @ amount
    df += L.named @ amount
    for f,i,m in L.recorded:
        df[i] += amount[f] @ p.maps[m]
    each = amount[L.tf]*p.tgt_w # Each target's share
    dy += L.arrive @ each
    for m,ks,M in L.mapped:
        dy += M @ (each[ks] @ p.maps[m])
    df += L.tgt_rec @ each
    if len(L.cf):
        df += L.credit @ (amount[L.cf]*p.crd_w)
    return dy, df, moved_in


def interp(out, table, row, th):
    a, b, c = (2*th - 1)*(th - 1), 4*th*(1 - th), th*(2*th - 1)
    out[:] = a*table[row] + b*table[row+1] + c*table[row+2]


def set_movers(p):
    on = p.mov_par >= 0
    if on.any():
        p.mov_rate[on] = p.mov_base[on]*p.consts[p.mov_par[on], p.mov_pair[0, on]]


def step(rng, y, p, dt, n_sub, solver, flows, tables, row):
    """ One step of the ODE, as `ss.kernels.step()` takes it (y and flows changed in place), with NumPy's warnings of division by zero off, as compiled code has them """
    with np.errstate(divide='ignore', invalid='ignore'):
        return _step(rng, y, p, dt, n_sub, solver, flows, tables, row)


def _step(rng, y, p, dt, n_sub, solver, flows, tables, row):
    if solver >= 2:
        return K.uncompiled().step(rng, y, p, K.uncompiled().workspace(y, p), dt, n_sub, solver, flows, tables, row)
    nc, ng = y.shape
    L = layout(p, nc, ng)
    euler = solver == 1
    stages = 1 if euler else 4
    auto = n_sub <= 0
    parts = (p.fixed, p.mult, p.consts, p.rel_trans)

    def at(sub, stage):
        if row >= 0:
            th = (sub + (0.0 if stage == 0 else (1.0 if stage == 3 else 0.5)))/max(n_sub, 1)
            for k in range(4):
                if len(tables[k]) > 1:
                    interp(parts[k], tables[k], row, th)
            set_movers(p)

    at(0, 0)
    rates = rates_of(y, p, L)
    if np.isinf(rates).any(): # Everyone in a compartment moves at once: as the kernels do it (see ss.kernels.certain())
        return K.uncompiled().step(rng, y, p, K.uncompiled().workspace(y, p), dt, n_sub, solver, flows, tables, row)
    for f_out,slots in L.conveyors: # step_start(): those due to leave each conveyor leave at a constant rate over the step
        p.inflow[f_out] = y[slots[-1]]/dt
    if auto:
        fastest = max(rates[L.out].max(initial=0.0), p.mov_rate.max(initial=0.0))
        n_sub = int(np.ceil(fastest*dt/0.25))
    n_sub = max(n_sub, 1)
    h = dt/n_sub
    for (f_out,slots),ex in zip(L.conveyors, L.exits): # release_rates()
        x = np.maximum(rates[ex], 0.0).sum(axis=0)*dt if len(ex) else np.zeros(ng)
        big = x > 1e-12
        p.inflow[f_out, big] *= x[big]/np.expm1(x[big])
    flows[:] = 0
    sub = 0
    while True:
        dys, dfs = [], []
        ys = y
        for stage in range(stages):
            if not (sub == 0 and stage == 0):
                at(sub, stage)
                rates = rates_of(ys, p, L)
            dy, df, moved_in = derivs(ys, p, L, rates, h, euler)
            dys.append(dy)
            dfs.append(df)
            if auto and sub == 0 and stage == 0: # Nor grows by more than a quarter in a substep
                least = 1e-9*y.sum()
                gain = dy - (moved_in if moved_in is not None else 0.0)
                ok = (gain > 0) & (y > least)
                growing = (gain[ok]/y[ok]).max(initial=0.0)
                if len(p.set_starts) > 1:
                    tot, grow = L.sets @ y, L.sets @ gain
                    ok = (grow > 0) & (tot > least)
                    growing = max(growing, (grow[ok]/tot[ok]).max(initial=0.0))
                more = int(np.ceil(growing*dt/0.25))
                if more > n_sub:
                    n_sub = more
                    h = dt/n_sub
            if stage < stages - 1:
                ys = y + (h if stage == 2 else h/2)*dy
        if euler:
            y += h*dys[0]
            flows += h*dfs[0]
        else:
            y += h/6*(dys[0] + 2*dys[1] + 2*dys[2] + dys[3])
            flows += h/6*(dfs[0] + 2*dfs[1] + 2*dfs[2] + dfs[3])
        sub += 1
        if sub >= n_sub:
            break
    for f_out,slots in L.conveyors: # step_end(): everyone in a conveyor moves along one slot
        y[slots[1:]] = y[slots[:-1]]
        y[slots[0]] = 0
    np.maximum(y, 0, out=y)
    return


def one_step(rng, y, p, dt, solver, n_sub, tables, row):
    """ As `ss.kernels.one_step()` """
    y = y.copy()
    flows = np.empty((p.n_trans + p.n_movers, y.shape[1]))
    step(rng, y, p, dt, n_sub, solver, flows, tables, row)
    return y, flows


def run(rng, y, p, dt, npts, solver, n_sub, fixed, mult, consts, rel_trans, reg, sh, tally, per_step, start):
    """ As `ss.kernels.run()` """
    if solver >= 2:
        return K.uncompiled().run(rng, y, p, dt, npts, solver, n_sub, fixed, mult, consts, rel_trans, reg, sh, tally, per_step, start)
    plain = K.uncompiled()
    y = y.copy()
    stocks = np.empty((npts - start, *y.shape))
    flows = np.zeros((npts - start, p.n_trans + p.n_movers, y.shape[1]))
    tables = (fixed, mult, consts, rel_trans)
    parts = (p.fixed, p.mult, p.consts, p.rel_trans)
    short = 0.0
    stocks[0] = y
    for ti in range(start + 1, npts):
        k = ti - start
        r = (k-1)*per_step
        if len(sh.to):
            plain.apply_shifts(rng, y, sh, ti-1, False, tally[k])
        for j in range(4):
            if r < len(tables[j]):
                parts[j][:] = tables[j][r]
        set_movers(p)
        step(rng, y, p, dt, n_sub, solver, flows[k], tables, r if per_step == 3 else -1)
        if len(reg.src):
            short += plain.apply_events(rng, y, reg, ti, False)
        stocks[k] = y
    return y, stocks, flows, short

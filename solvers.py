"""Fixed-k coloring kernels. Reuse DSATUR, GPX and State from the submitted de06.py."""
from dataclasses import dataclass
from math import ceil, exp, log
from time import perf_counter

from de06 import State, gpx


DEFAULTS = {
    "tabu": {"lam": 0.6, "r_max": 9},
    "sa": {"p0": 0.8, "alpha": 0.95, "moves_per_vertex": 10, "t_min": 0.01},
    "ga": {"population": 20, "mutation_scale": 1, "tournament": 3, "elite": 1},
    "aco": {"ants": 20, "alpha": 1, "beta": 2, "rho": 0.3, "q": 1,
            "tau0": 1, "tau_min": 0.01, "tau_max": 5, "epsilon": 1e-6},
    "vns": {"s_max": 4, "vnd_scale": 1},
    "hea": {"population": 20, "mutation_scale": 1, "tournament": 3, "elite": 1,
            "local_scale": 3, "lam": 0.6, "r_max": 9},
}


@dataclass
class Meter:
    deadline: float
    full: int = 0
    delta: int = 0
    steps: int = 0

    def open(self):
        return perf_counter() < self.deadline

    def state(self, adj, k, col):
        self.full += 1
        return State(adj, k, col)


def other_color(a, k, rng):
    b = rng.randrange(k - 1)
    return b + (b >= a)


def reduce_one(adj, col, k, rng):
    """Drop the smallest class; reassign its vertices greedily, breaking ties by seed."""
    n_colors = len(set(col))
    if n_colors != k + 1 or k < 1:
        raise ValueError("Expected a normalized coloring with k+1 colors.")
    count = [0] * n_colors
    for a in col:
        count[a] += 1
    drop = min(range(n_colors), key=lambda a: (count[a], a))
    out = [-1 if a == drop else a - (a > drop) for a in col]
    for v, a in enumerate(out):
        if a != -1:
            continue
        scores = [0] * k
        for u in adj[v]:
            if out[u] >= 0:
                scores[out[u]] += 1
        least = min(scores)
        out[v] = rng.choice([b for b in range(k) if scores[b] == least])
    return out


def competitors(st, kind, rng=None):
    """Move tuple: (kind, vertex, new color or second vertex)."""
    conflict = sorted(st.conf)
    colors = list(range(st.k))
    vertices = list(range(st.n))
    if rng is not None:
        rng.shuffle(conflict)
        rng.shuffle(colors)
        rng.shuffle(vertices)
    if kind in (0, 1):
        for u in conflict:
            for c in colors:
                if c != st.col[u]:
                    yield (1, u, c)
    if kind in (0, 2):
        for u in conflict:
            for v in vertices:
                if st.col[u] != st.col[v] and not (v in st.conf and v < u):
                    yield (2, u, v)


def pick_move(st, meter, *, kind=0, first=False, rng=None):
    moves = competitors(st, kind, rng if first else None)
    best_d, best = 0, None
    for i, move in enumerate(moves):
        if i % 128 == 0 and not meter.open():
            return None
        t, u, v = move
        d = st.delta(u, v) if t == 1 else st.swap_delta(u, v)
        meter.delta += 1
        if d < best_d:
            if first:
                return move
            best_d, best = d, move
    return best if meter.open() else None


def apply_move(st, move, meter):
    t, u, v = move
    if t == 1:
        st.move(u, v)
    else:
        a, b = st.col[u], st.col[v]
        st.move(u, b)
        st.move(v, a)
    meter.steps += 1


def hill_climb(adj, k, init, rng, meter, first):
    st = meter.state(adj, k, init)
    while st.f and meter.open():
        move = pick_move(st, meter, first=first, rng=rng)
        if move is None:
            break
        apply_move(st, move, meter)
    return st.col, st.f


def tabu(adj, k, init, rng, meter, *, lam, r_max, max_iter=None):
    st = meter.state(adj, k, init)
    best = st.f, st.col[:]
    memory = [[0] * k for _ in adj]
    it = 0
    while best[0] and k > 1 and meter.open() and (max_iter is None or it < max_iter):
        it += 1
        meter.steps += 1
        best_d, cand = float("inf"), []
        for u in sorted(st.conf):
            if not meter.open():
                return best[1], best[0]
            a = st.col[u]
            for b in range(k):
                if b == a:
                    continue
                d = st.delta(u, b)
                meter.delta += 1
                if it <= memory[u][b] and st.f + d >= best[0]:
                    continue
                if d < best_d:
                    best_d, cand = d, [(u, b)]
                elif d == best_d:
                    cand.append((u, b))
        if not cand:
            continue
        u, b = rng.choice(cand)
        memory[u][st.col[u]] = it + int(lam * len(st.conf)) + rng.randint(0, r_max)
        st.move(u, b)
        if st.f < best[0]:
            best = st.f, st.col[:]
    return best[1], best[0]


def anneal(adj, k, init, rng, meter, *, p0, alpha, moves_per_vertex, t_min):
    st = meter.state(adj, k, init)
    best = st.f, st.col[:]
    if not st.f or k <= 1:
        return best[1], best[0]
    positive = []
    for _ in range(min(256, max(32, st.n))):
        if not meter.open():
            break
        u = rng.randrange(st.n)
        d = st.delta(u, other_color(st.col[u], k, rng))
        meter.delta += 1
        if d > 0:
            positive.append(d)
    t0 = -(sum(positive) / len(positive)) / log(p0) if positive else 1.0
    temperature = t0
    while best[0] and meter.open():
        for _ in range(max(1, moves_per_vertex * st.n)):
            if not meter.open():
                break
            u = rng.randrange(st.n)
            b = other_color(st.col[u], k, rng)
            d = st.delta(u, b)
            meter.delta += 1
            meter.steps += 1
            if d <= 0 or rng.random() < exp(-d / temperature):
                st.move(u, b)
                if st.f < best[0]:
                    best = st.f, st.col[:]
                    if not st.f:
                        return best[1], best[0]
        temperature *= alpha
        if temperature < t_min and meter.open():
            st = meter.state(adj, k, best[1])
            temperature = t0
    return best[1], best[0]


def evolve(adj, k, init, rng, meter, *, population, mutation_scale, tournament, elite,
           local_scale=0, lam=0.6, r_max=9):
    """GA and HEA share all GA operators. Only HEA applies a bounded TabuCol."""
    n = len(adj)
    edges = [(u, v) for u, nbr in enumerate(adj) for v in sorted(nbr) if u < v]
    mutation = min(1.0, mutation_scale / n)

    def assess(col):
        if local_scale and meter.open():
            col, f = tabu(adj, k, col, rng, meter, lam=lam, r_max=r_max,
                          max_iter=max(1, round(local_scale * n)))
        else:
            f = sum(col[u] == col[v] for u, v in edges)
            meter.full += 1
        return f, col

    best = assess(init[:])
    pop = [best]
    while best[0] and len(pop) < population and meter.open():
        ind = assess([rng.randrange(k) for _ in adj])
        pop.append(ind)
        if ind[0] < best[0]:
            best = ind
    while best[0] and k > 1 and len(pop) == population and meter.open():
        nxt = sorted(pop, key=lambda x: x[0])[:elite]
        while best[0] and len(nxt) < population and meter.open():
            p1, p2 = (min(rng.sample(pop, min(tournament, len(pop))), key=lambda x: x[0])[1]
                      for _ in range(2))
            child = gpx(p1, p2, k, rng)
            for v, a in enumerate(child):
                if rng.random() < mutation:
                    child[v] = other_color(a, k, rng)
            ind = assess(child)
            meter.steps += 1
            nxt.append(ind)
            if ind[0] < best[0]:
                best = ind
        if len(nxt) == population:
            pop = nxt
    return best[1], best[0]


def ant_colony(adj, k, init, rng, meter, *, ants, alpha, beta, rho, q,
               tau0, tau_min, tau_max, epsilon):
    n = len(adj)
    best = meter.state(adj, k, init).f, init[:]
    if not best[0] or k <= 1 or not meter.open():
        return best[1], best[0]
    tau = [[tau0] * n for _ in adj]
    pairs = [(u, v) for u in range(n) for v in range(u + 1, n) if v not in adj[u]]

    def build_ant():
        uncolored = set(range(n))
        count = [[0] * k for _ in adj]
        sat = [0] * n
        classes = [[] for _ in range(k)]
        col = [-1] * n
        f = 0
        while uncolored:
            if not meter.open():
                return None
            v = max(uncolored, key=lambda u: (sat[u], len(adj[u]), -u))
            uncolored.remove(v)
            weights = []
            for a, cls in enumerate(classes):
                phi = (epsilon + sum(tau[v][u] for u in cls if u not in adj[v]) / len(cls)
                       if cls else tau0)
                weights.append(phi ** alpha / (1 + count[v][a]) ** beta)
            a = rng.choices(range(k), weights=weights)[0]
            f += count[v][a]
            col[v] = a
            classes[a].append(v)
            for u in adj[v]:
                if u in uncolored:
                    sat[u] += count[u][a] == 0
                    count[u][a] += 1
        meter.full += 1
        meter.steps += 1
        return f, col

    while best[0] and meter.open():
        round_best = None
        for _ in range(ants):
            ant = build_ant()
            if ant is None:
                break
            if round_best is None or ant[0] < round_best[0]:
                round_best = ant
            if ant[0] < best[0]:
                best = ant
                if not ant[0]:
                    return best[1], best[0]
        if round_best is None or not meter.open():
            break
        f, col = round_best
        deposit = q / (1 + f)
        for i, (u, v) in enumerate(pairs):
            if i % 128 == 0 and not meter.open():
                break
            value = (1 - rho) * tau[u][v] + (deposit if col[u] == col[v] else 0)
            tau[u][v] = tau[v][u] = min(tau_max, max(tau_min, value))
    return best[1], best[0]


def vns(adj, k, init, rng, meter, *, s_max, vnd_scale):
    n = len(adj)
    best = meter.state(adj, k, init).f, init[:]
    s, limit = 1, min(s_max, n)
    local_limit = max(1, ceil(vnd_scale * n))
    while best[0] and k > 1 and meter.open():
        st = meter.state(adj, k, best[1])
        for u in rng.sample(range(n), s):
            if not meter.open():
                break
            st.move(u, other_color(st.col[u], k, rng))
            meter.steps += 1
            if not st.f:
                return st.col, 0
        kind, accepted = 1, 0
        while kind <= 2 and accepted < local_limit and meter.open():
            move = pick_move(st, meter, kind=kind)
            if move is None:
                kind += 1
                continue
            apply_move(st, move, meter)
            if not st.f:
                return st.col, 0
            kind, accepted = 1, accepted + 1
        if st.f < best[0]:
            best = st.f, st.col[:]
            s = 1
        else:
            s = 1 + s % limit
    return best[1], best[0]


KERNELS = {"tabu": tabu, "sa": anneal, "ga": evolve, "aco": ant_colony, "vns": vns, "hea": evolve}


def solve_fixed(name, adj, k, init, rng, meter, overrides=None):
    options = DEFAULTS[name] | (overrides or {})
    return KERNELS[name](adj, k, init, rng, meter, **options)

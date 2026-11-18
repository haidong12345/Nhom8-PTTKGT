#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import time
import random
import argparse
import statistics

sys.setrecursionlimit(20000)
def read_dimacs(path):
    """Trả về (n, adj, info). adj[u] là tập láng giềng (đỉnh đánh số từ 0)."""
    n = None
    edges = set()
    loops = dup = declared_m = 0
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            t = line.split()
            if not t or t[0] == "c":
                continue
            if t[0] == "p":
                n, declared_m = int(t[2]), int(t[3])
            elif t[0] == "e":
                if n is None:
                    raise ValueError("dòng %d: 'e' xuất hiện trước dòng 'p'" % ln)
                u, v = int(t[1]) - 1, int(t[2]) - 1
                if not (0 <= u < n and 0 <= v < n):
                    raise ValueError("dòng %d: chỉ số đỉnh ngoài [1,%d]" % (ln, n))
                if u == v:
                    loops += 1
                    continue
                key = (min(u, v), max(u, v))
                if key in edges:
                    dup += 1
                    continue
                edges.add(key)
    if n is None:
        raise ValueError("thiếu dòng 'p edge n m'")
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    info = dict(n=n, m=len(edges), declared_m=declared_m, loops=loops, dup=dup)
    return n, adj, info


def write_dimacs(path, n, edges, comment="generated"):
    with open(path, "w") as f:
        f.write("c %s\np edge %d %d\n" % (comment, n, len(edges)))
        for u, v in edges:
            f.write("e %d %d\n" % (u + 1, v + 1))


def write_solution(path, col):
    k = len(set(col))
    with open(path, "w") as f:
        f.write("%d\n%s\n" % (k, " ".join(str(c + 1) for c in col)))


def check(adj, col):
    """(số cạnh xung đột, số màu dùng) - dùng để kiểm chứng mọi lời giải."""
    conflicts = sum(1 for u in range(len(adj)) for v in adj[u] if u < v and col[u] == col[v])
    return conflicts, len(set(col))


def normalize(col):
    """Đánh số lại màu 0..k-1 liên tục."""
    mp = {}
    return [mp.setdefault(c, len(mp)) for c in col]


def clique_lower_bound(adj, tries=300, seed=0):
    rng = random.Random(seed)
    n = len(adj)
    best = []
    by_deg = sorted(range(n), key=lambda v: -len(adj[v]))
    for t in range(tries):
        if t < n:
            start = by_deg[t % n]
        else:
            start = rng.randrange(n)
        clique = [start]
        cand = set(adj[start])
        while cand:
            # chọn đỉnh trong cand có nhiều láng giềng nhất trong cand (có nhiễu)
            if t < n:
                v = max(cand, key=lambda x: (len(adj[x] & cand), -x))
            else:
                v = max(cand, key=lambda x: (len(adj[x] & cand), rng.random()))
            clique.append(v)
            cand &= adj[v]
        if len(clique) > len(best):
            best = clique
    return best

def welsh_powell(adj):
    n = len(adj)
    order = sorted(range(n), key=lambda v: -len(adj[v]))
    col = [-1] * n
    c = 0
    left = n
    while left:
        cls = []
        for v in order:
            if col[v] < 0 and not any(w in adj[v] for w in cls):
                col[v] = c
                cls.append(v)
                left -= 1
        c += 1
    return col


def dsatur(adj):
    n = len(adj)
    col = [-1] * n
    sat = [set() for _ in range(n)]
    deg = [len(a) for a in adj]
    for _ in range(n):
        v = max((u for u in range(n) if col[u] < 0), key=lambda u: (len(sat[u]), deg[u]))
        c = 0
        while c in sat[v]:
            c += 1
        col[v] = c
        for w in adj[v]:
            sat[w].add(c)
    return col

def exact_bb(adj, tlimit=60.0):
    """Trả về (col, k, tối_ưu_đã_chứng_minh, cận_dưới, số_nút)."""
    n = len(adj)
    best_col = dsatur(adj)
    best = max(best_col) + 1
    clique = clique_lower_bound(adj)
    lb = len(clique)
    if best <= lb:
        return best_col, best, True, lb, 0
    deg = [len(a) for a in adj]
    col = [-1] * n
    cnt = [dict() for _ in range(n)]  # cnt[v][c] = số láng giềng của v có màu c

    def assign(v, c):
        col[v] = c
        for w in adj[v]:
            d = cnt[w]
            d[c] = d.get(c, 0) + 1

    def unassign(v, c):
        col[v] = -1
        for w in adj[v]:
            d = cnt[w]
            d[c] -= 1
            if d[c] == 0:
                del d[c]

    for i, v in enumerate(clique):  # cố định màu cho clique => phá đối xứng
        assign(v, i)
    t0 = time.time()
    st = {"best": best, "col": best_col[:], "nodes": 0, "timeout": False}

    def rec(colored, used):
        st["nodes"] += 1
        if st["nodes"] & 1023 == 0 and time.time() - t0 > tlimit:
            st["timeout"] = True
        if st["timeout"]:
            return
        if colored == n:
            st["best"], st["col"] = used, col[:]
            return
        bv, bk = -1, (-1, -1)
        for v in range(n):
            if col[v] < 0:
                k = (len(cnt[v]), deg[v])
                if k > bk:
                    bk, bv = k, v
        v = bv
        for c in range(min(used + 1, st["best"] - 1)):  # c <= best-2; c == used là màu mới
            if c in cnt[v]:
                continue
            assign(v, c)
            rec(colored + 1, used + 1 if c == used else used)
            unassign(v, c)
            if st["timeout"] or st["best"] <= lb:
                return

    rec(len(clique), len(clique))
    proven = (not st["timeout"]) or st["best"] <= lb
    return st["col"], st["best"], proven, lb, st["nodes"]

class State:
    """col, bảng g[v][c] = #láng giềng của v màu c, f = #cạnh xung đột, conf = đỉnh xung đột."""

    def __init__(self, adj, k, col):
        self.adj, self.k, self.n = adj, k, len(adj)
        self.col = list(col)
        self.g = [[0] * k for _ in range(self.n)]
        for u in range(self.n):
            for v in adj[u]:
                self.g[u][self.col[v]] += 1
        self.f = sum(self.g[u][self.col[u]] for u in range(self.n)) // 2
        self.conf = {u for u in range(self.n) if self.g[u][self.col[u]] > 0}

    def delta(self, u, c):
        return self.g[u][c] - self.g[u][self.col[u]]

    def swap_delta(self, u, v):
        """Độ thay đổi #xung đột nếu hoán đổi màu của u và v."""
        a, b = self.col[u], self.col[v]
        d = self.delta(u, b) + self.delta(v, a)
        if v in self.adj[u]:
            d -= 2  # cạnh uv: không xung đột trước và sau, nhưng bị đếm 2 lần
        return d

    def move(self, u, c):
        """Đổi màu u -> c, cập nhật bảng g trong O(bậc(u))."""
        a = self.col[u]
        g, col, conf = self.g, self.col, self.conf
        self.f += g[u][c] - g[u][a]
        col[u] = c
        for w in self.adj[u]:
            gw = g[w]
            gw[a] -= 1
            gw[c] += 1
            if gw[col[w]] > 0:
                conf.add(w)
            else:
                conf.discard(w)
        if g[u][c] > 0:
            conf.add(u)
        else:
            conf.discard(u)


def shrink_coloring(col, k, rng):
    """Chuyển lời giải có >k màu thành khởi tạo k màu: đỉnh màu >=k nhận màu ngẫu nhiên."""
    col = normalize(col)
    return [c if c < k else rng.randrange(k) for c in col]

def tabucol(adj, k, init=None, tlimit=10.0, max_iter=10 ** 9, rng=None):
    rng = rng or random.Random()
    n = len(adj)
    start = init if init is not None else [rng.randrange(k) for _ in range(n)]
    st = State(adj, k, start)
    tabu = [[0] * k for _ in range(n)]
    best_f = st.f
    it = 0
    t0 = time.time()
    while st.f > 0 and it < max_iter:
        it += 1
        if (it & 127) == 0 and time.time() - t0 > tlimit:
            break
        bd, cand = 10 ** 9, []
        g, col = st.g, st.col
        for u in st.conf:
            gu, a = g[u], col[u]
            ga, tu = gu[a], tabu[u]
            for c in range(k):
                if c == a:
                    continue
                d = gu[c] - ga
                if tu[c] > it and not (st.f + d < best_f):  # tabu, trừ khi thoả aspiration
                    continue
                if d < bd:
                    bd, cand = d, [(u, c)]
                elif d == bd:
                    cand.append((u, c))
        if not cand:
            continue
        u, c = rng.choice(cand)
        a = st.col[u]
        tabu[u][a] = it + int(0.6 * len(st.conf)) + rng.randrange(10)
        st.move(u, c)
        if st.f < best_f:
            best_f = st.f
    return st.col, st.f, it

def local_search(adj, k, init=None, tlimit=10.0, noise=0.03, swaps=12, rng=None):
    rng = rng or random.Random()
    n = len(adj)
    st = State(adj, k, init if init is not None else [rng.randrange(k) for _ in range(n)])
    t0 = time.time()
    it = 0
    while st.f > 0:
        it += 1
        if (it & 255) == 0 and time.time() - t0 > tlimit:
            break
        u = rng.choice(tuple(st.conf))
        best_d, best_mv = 10 ** 9, None
        for c in range(k):                      # láng giềng 1: đổi màu u
            if c != st.col[u]:
                d = st.delta(u, c)
                if d < best_d or (d == best_d and rng.random() < 0.5):
                    best_d, best_mv = d, ("m", u, c)
        for _ in range(swaps):                  # láng giềng 2: hoán đổi màu (u, v)
            v = rng.randrange(n)
            if st.col[v] != st.col[u]:
                d = st.swap_delta(u, v)
                if d < best_d:
                    best_d, best_mv = d, ("s", u, v)
        if best_d > 0 and rng.random() < noise:  # random walk để thoát cực trị địa phương
            best_mv = ("m", u, rng.choice([c for c in range(k) if c != st.col[u]]))
            best_d = 0
        if best_d <= 0 and best_mv:
            if best_mv[0] == "m":
                st.move(best_mv[1], best_mv[2])
            else:
                _, a, b = best_mv
                ca, cb = st.col[a], st.col[b]
                st.move(a, cb)
                st.move(b, ca)
    return st.col, st.f, it

def minimize_colors(adj, tlimit, seed=0, method="tabu", lb=0):
    """DSATUR -> thử k = best-1, best-2, ... bằng TabuCol/LS đến hết thời gian."""
    rng = random.Random(seed)
    t0 = time.time()
    col = normalize(dsatur(adj))
    best = max(col) + 1
    t_best = time.time() - t0
    while best > max(lb, 1):
        left = tlimit - (time.time() - t0)
        if left <= 0:
            break
        k = best - 1
        init = shrink_coloring(col, k, rng)
        if method == "tabu":
            new, f, _ = tabucol(adj, k, init, tlimit=left, rng=rng)
        else:
            new, f, _ = local_search(adj, k, init, tlimit=left, rng=rng)
        if f == 0:
            col, best = normalize(new), k
            t_best = time.time() - t0
        else:
            break
    return col, best, t_best

def gen_myciel(k):
    """myciel_k theo DIMACS: áp dụng Mycielski (k-1) lần lên K2. myciel3: 11 đỉnh, chi=4."""
    n, edges = 2, [(0, 1)]
    for _ in range(k - 1):
        new = list(edges)
        for i, j in edges:
            new += [(i, n + j), (j, n + i)]
        new += [(n + i, 2 * n) for i in range(n)]
        n, edges = 2 * n + 1, new
    return n, edges


def gen_queen(r, c):
    idx = lambda i, j: i * c + j
    edges = []
    cells = [(i, j) for i in range(r) for j in range(c)]
    for a in range(len(cells)):
        for b in range(a + 1, len(cells)):
            (i, j), (x, y) = cells[a], cells[b]
            if i == x or j == y or abs(i - x) == abs(j - y):
                edges.append((idx(i, j), idx(x, y)))
    return r * c, edges


def gen_gnp(n, p, seed):
    rng = random.Random(seed)
    return n, [(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < p]

def run_file(path, algs, tlimit, runs, out=None, quiet=False):
    n, adj, info = read_dimacs(path)
    print("== %s : n=%d, m=%d (khai báo %d; bỏ %d cạnh lặp, %d khuyên)"
          % (path, n, info["m"], info["declared_m"], info["dup"], info["loops"]))
    clique = clique_lower_bound(adj)
    lb = len(clique)
    print("   cận dưới clique omega >= %d" % lb)
    best_overall = None

    def report(name, col, t, extra=""):
        nonlocal best_overall
        conf, k = check(adj, col)
        assert conf == 0, "lời giải %s còn %d cạnh xung đột!" % (name, conf)
        print("   %-8s k=%-3d xung đột=%d  %.2fs %s" % (name, k, conf, t, extra))
        if best_overall is None or k < len(set(best_overall)):
            best_overall = normalize(col)

    if "wp" in algs or "all" in algs:
        t = time.time(); col = welsh_powell(adj); report("W-Powell", col, time.time() - t)
    if "dsatur" in algs or "all" in algs:
        t = time.time(); col = dsatur(adj); report("DSATUR", col, time.time() - t)
    if "exact" in algs or "all" in algs:
        t = time.time()
        col, k, proven, lb2, nodes = exact_bb(adj, tlimit)
        report("Exact", col, time.time() - t,
               "(%s, %d nút)" % ("TỐI ƯU chứng minh được" if proven else "hết giờ - chưa chứng minh", nodes))
        if proven:
            print("   => chi(G) = %d" % k)
    for name, method in (("tabu", "tabu"), ("ls", "ls")):
        if name in algs or "all" in algs:
            ks, ts, t0 = [], [], time.time()
            for r in range(runs):
                col, k, tb = minimize_colors(adj, tlimit, seed=r, method=method, lb=lb)
                assert check(adj, col)[0] == 0
                ks.append(k); ts.append(tb)
                if best_overall is None or k < len(set(best_overall)):
                    best_overall = normalize(col)
            label = "TabuCol" if name == "tabu" else "LS(1+swap)"
            print("   %-10s best=%d  mean=%.2f  std=%.2f  #đạt best=%d/%d  t_best(tb)=%.2fs"
                  % (label, min(ks), statistics.mean(ks),
                     statistics.pstdev(ks), ks.count(min(ks)), runs, statistics.mean(ts)))
    if out and best_overall:
        write_solution(out, best_overall)
        print("   đã ghi lời giải tốt nhất (k=%d) vào %s" % (len(set(best_overall)), out))
    return best_overall


def selftest(tlimit, runs, directory="."):
    import os
    os.makedirs(directory, exist_ok=True)
    # ví dụ nhỏ trong đề
    ex = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 5), (3, 6), (4, 5), (4, 7), (5, 6), (5, 7), (6, 7)]
    write_dimacs(os.path.join(directory, "vidu.col"), 7, [(a - 1, b - 1) for a, b in ex], "vi du de 6")
    sets = [("vidu", None)]
    for k in (3, 4, 5):
        n, e = gen_myciel(k); write_dimacs(os.path.join(directory, "myciel%d.col" % k), n, e)
        sets.append(("myciel%d" % k, None))
    for r in (5, 6, 8):
        n, e = gen_queen(r, r); write_dimacs(os.path.join(directory, "queen%d_%d.col" % (r, r)), n, e)
        sets.append(("queen%d_%d" % (r, r), None))
    n, e = gen_gnp(125, 0.1, 1); write_dimacs(os.path.join(directory, "gnp125_0.1.col"), n, e)
    n, e = gen_gnp(125, 0.5, 1); write_dimacs(os.path.join(directory, "gnp125_0.5.col"), n, e)
    for name in ["vidu", "myciel3", "myciel4", "myciel5", "queen5_5", "queen6_6", "queen8_8",
                 "gnp125_0.1", "gnp125_0.5"]:
        run_file(os.path.join(directory, name + ".col"), ["all"], tlimit, runs)
        print()


def main():
    ap = argparse.ArgumentParser(description="Đề 6 - tô màu đồ thị / xếp lịch thi")
    ap.add_argument("file", nargs="?", help="file DIMACS .col")
    ap.add_argument("--alg", default="all", help="wp,dsatur,exact,tabu,ls,all (phân tách bằng dấu phẩy)")
    ap.add_argument("--time", type=float, default=30.0, help="ngân sách thời gian (giây) mỗi lần chạy")
    ap.add_argument("--runs", type=int, default=5, help="số lần chạy với seed khác nhau")
    ap.add_argument("--out", help="ghi lời giải tốt nhất ra file")
    ap.add_argument("--selftest", action="store_true", help="sinh bộ test (myciel, queen, gnp) và chạy")
    ap.add_argument("--dir", default="data", help="thư mục chứa file sinh ra khi --selftest")
    a = ap.parse_args()
    if a.selftest:
        selftest(a.time, a.runs, a.dir)
    elif a.file:
        run_file(a.file, set(a.alg.split(",")), a.time, a.runs, a.out)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()

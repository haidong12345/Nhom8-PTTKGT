#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
de06.py - Đề 6: Xếp lịch thi = bài toán tô màu đồ thị (min số màu, sắc số chi(G)).
Xem README.md để biết thuật toán và cách chạy.
  python de06.py --web      # mở trang web trực quan hoá
"""
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
    best_col = list(st.col)
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
            best_col = list(st.col)
    return best_col, best_f, it

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

def gpx(p1, p2, k, rng):
    """Greedy Partition Crossover: lần lượt lấy lớp màu LỚN NHẤT của cha/mẹ (xen kẽ),
    gán thành một lớp của con, rồi xoá các đỉnh đó khỏi cả hai bố mẹ.
    Đỉnh còn sót được gán màu ngẫu nhiên."""
    n = len(p1)
    A = [set() for _ in range(k)]
    B = [set() for _ in range(k)]
    for v in range(n):
        A[p1[v]].add(v)
        B[p2[v]].add(v)
    parents = (A, B)
    child = [-1] * n
    for i in range(k):
        P = parents[i % 2]
        j = max(range(k), key=lambda c: len(P[c]))
        cls = list(P[j])
        for v in cls:
            child[v] = i
            A[p1[v]].discard(v)
            B[p2[v]].discard(v)
    for v in range(n):
        if child[v] < 0:
            child[v] = rng.randrange(k)
    return child


def hea(adj, k, init=None, tlimit=10.0, pop_size=10, ls_iters=None, rng=None):
    """Hybrid Evolutionary Algorithm cho k màu cố định.
    Mỗi thế hệ: chọn 2 cha mẹ ngẫu nhiên -> GPX -> TabuCol ngắn (ls_iters bước) ->
    con thay cha/mẹ tệ hơn. Nếu quần thể đình trệ thì khởi tạo lại nửa tệ nhất.
    Trả về (col, f, số thế hệ)."""
    rng = rng or random.Random()
    n = len(adj)
    t0 = time.time()
    ls_iters = ls_iters or max(300, 3 * n)
    left = lambda: tlimit - (time.time() - t0)

    def make(start):
        return tabucol(adj, k, start, tlimit=max(left(), 0.01), max_iter=ls_iters, rng=rng)[:2]

    pop = []
    for i in range(pop_size):
        if left() <= 0 and pop:
            break
        start = shrink_coloring(init, k, rng) if (i == 0 and init is not None) else None
        col, f = make(start)
        if f == 0:
            return col, 0, 0
        pop.append((f, col))
    if len(pop) < 2:
        c, f = pop[0][1], pop[0][0]
        return c, f, 0
    best_f = min(f for f, _ in pop)
    gen = last_imp = 0
    while left() > 0:
        gen += 1
        i, j = rng.sample(range(len(pop)), 2)
        child = gpx(pop[i][1], pop[j][1], k, rng)
        col, f = make(child)
        if f == 0:
            return col, 0, gen
        worse = i if pop[i][0] >= pop[j][0] else j
        pop[worse] = (f, col)
        if f < best_f:
            best_f, last_imp = f, gen
        if gen - last_imp > 40 * pop_size:       # đình trệ: làm mới nửa tệ nhất
            pop.sort(key=lambda x: x[0])
            for t in range(len(pop) // 2, len(pop)):
                if left() <= 0:
                    break
                col, f = make(None)
                if f == 0:
                    return col, 0, gen
                pop[t] = (f, col)
            last_imp = gen
    f, col = min(pop, key=lambda x: x[0])
    return col, f, gen

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
        elif method == "hea":
            new, f, _ = hea(adj, k, init, tlimit=left, rng=rng)
        else:
            new, f, _ = local_search(adj, k, init, tlimit=left, rng=rng)
        if f == 0:
            col, best = normalize(new), k
            t_best = time.time() - t0
        else:
            break
    return col, best, t_best

def _open_any(prefix, ext):
    import os
    for e in (ext, ext.upper()):
        p = prefix + e
        if os.path.exists(p):
            return open(p, encoding="utf-8", errors="ignore")
    raise FileNotFoundError(prefix + ext)


def read_toronto(prefix):
    """.crs: mỗi dòng "mã_môn  số_sinh_viên"; .stu: mỗi dòng = danh sách môn của 1 sinh viên.
    Hai môn xuất hiện trên cùng một dòng .stu được nối cạnh.
    Trả về (n, adj, nbw, S, info); nbw[i][j] = số sinh viên chung của môn i và j."""
    for e in (".crs", ".stu", ".CRS", ".STU"):
        if prefix.endswith(e):
            prefix = prefix[:-len(e)]
    courses, idx = [], {}
    with _open_any(prefix, ".crs") as f:
        for line in f:
            t = line.split()
            if t and t[0] not in idx:
                idx[t[0]] = len(courses)
                courses.append(t[0])
    students = []
    with _open_any(prefix, ".stu") as f:
        for line in f:
            t = line.split()
            if not t:
                continue
            ids = []
            for c in dict.fromkeys(t):          # bỏ môn lặp, giữ thứ tự
                if c not in idx:
                    idx[c] = len(courses)
                    courses.append(c)
                ids.append(idx[c])
            students.append(ids)
    n = len(courses)
    nbw = [dict() for _ in range(n)]
    for ids in students:
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                i, j = ids[a], ids[b]
                nbw[i][j] = nbw[i].get(j, 0) + 1
                nbw[j][i] = nbw[j].get(i, 0) + 1
    adj = [set(d.keys()) for d in nbw]
    m = sum(len(a) for a in adj) // 2
    info = dict(n=n, m=m, S=len(students), courses=courses,
                density=(2.0 * m / (n * (n - 1)) if n > 1 else 0.0))
    return n, adj, nbw, len(students), info


def pen(d):
    """Phạt Carter: hai kỳ thi của cùng sinh viên cách d ca (1<=d<=5) bị phạt 2^(5-d)."""
    return (1 << (5 - d)) if 1 <= d <= 5 else 0


def proximity_cost(nbw, slot, S):
    """Chi phí mềm trung bình trên mỗi sinh viên (hàm mục tiêu chuẩn của bộ Toronto)."""
    tot = 0
    for i, d in enumerate(nbw):
        for j, w in d.items():
            if i < j:
                tot += w * pen(abs(slot[i] - slot[j]))
    return tot / float(S)


def order_slots(nbw, col, T, rng, restarts=8):
    """Sắp thứ tự các lớp màu vào T ca thi để giảm chi phí mềm (leo đồi hoán đổi vị trí)."""
    k = max(col) + 1
    A = {}
    for i, d in enumerate(nbw):
        for j, w in d.items():
            if i < j and col[i] != col[j]:
                key = (min(col[i], col[j]), max(col[i], col[j]))
                A[key] = A.get(key, 0) + w
    pairs = list(A.items())

    def cost(perm):
        return sum(w * pen(abs(perm[a] - perm[b])) for (a, b), w in pairs)

    best_perm, best_c = None, None
    for r in range(restarts):
        perm = list(range(T))
        if r:
            rng.shuffle(perm)
        c = cost(perm)
        improved = True
        while improved:
            improved = False
            for x in range(T):
                for y in range(x + 1, T):
                    perm[x], perm[y] = perm[y], perm[x]
                    c2 = cost(perm)
                    if c2 < c:
                        c, improved = c2, True
                    else:
                        perm[x], perm[y] = perm[y], perm[x]
        if best_c is None or c < best_c:
            best_perm, best_c = perm[:], c
    return [best_perm[c] for c in col]


def optimize_proximity(adj, nbw, col, T, S, tlimit, rng):
    """Cho lịch khả thi (<=T ca): ủ mô phỏng (SA) dời từng môn sang ca khác, vẫn giữ khả thi,
    để giảm chi phí khoảng cách (Carter). Trả về (slot, cost_sau_sắp_ca, cost_cuối)."""
    n = len(adj)
    col = normalize(col)
    slot = order_slots(nbw, col, T, rng)
    c_ordered = proximity_cost(nbw, slot, S)
    g = [[0] * T for _ in range(n)]
    for v in range(n):
        for w in adj[v]:
            g[v][slot[w]] += 1
    cost = c_ordered * S
    best, best_slot = cost, slot[:]
    t0 = time.time()
    T0, T1 = 8.0, 0.15
    import math
    temp = T0
    while True:
        el = time.time() - t0
        if el > tlimit:
            break
        temp = T0 * (T1 / T0) ** (el / tlimit)
        for _ in range(500):
            v = rng.randrange(n)
            b = rng.randrange(T)
            a = slot[v]
            if b == a or g[v][b] > 0:
                continue
            d = 0
            for u, w in nbw[v].items():
                su = slot[u]
                d += w * (pen(abs(b - su)) - pen(abs(a - su)))
            if d <= 0 or rng.random() < math.exp(-d / temp):
                slot[v] = b
                for u in adj[v]:
                    g[u][a] -= 1
                    g[u][b] += 1
                cost += d
        if cost < best:
            best, best_slot = cost, slot[:]
    return best_slot, c_ordered, best / float(S)


def gen_toronto(prefix, n_courses=120, n_students=1200, seed=1):
    """Sinh bộ dữ liệu dạng Toronto (để thử khi chưa có file thật)."""
    rng = random.Random(seed)
    w = [1.0 / (i + 1) ** 0.6 for i in range(n_courses)]
    stu, size = [], [0] * n_courses
    for _ in range(n_students):
        k = rng.randint(3, 7)
        cs = set()
        while len(cs) < k:
            cs.add(rng.choices(range(n_courses), weights=w)[0])
        stu.append(sorted(cs))
        for c in cs:
            size[c] += 1
    with open(prefix + ".crs", "w") as f:
        for c in range(n_courses):
            f.write("%04d %d\n" % (c + 1, size[c]))
    with open(prefix + ".stu", "w") as f:
        for cs in stu:
            f.write(" ".join("%04d" % (c + 1) for c in cs) + "\n")


def run_toronto(prefix, tlimit, slots, method="hea", seed=0, out=None):
    n, adj, nbw, S, info = read_toronto(prefix)
    print("== Toronto %s : %d môn, %d sinh viên, %d cặp môn xung đột (mật độ %.3f)"
          % (prefix, n, S, info["m"], info["density"]))
    lb = len(clique_lower_bound(adj))
    t0 = time.time()
    col, k, _ = minimize_colors(adj, tlimit, seed=seed, method=method, lb=lb)
    conf, k = check(adj, col)
    assert conf == 0
    print("   cận dưới clique >= %d ; số ca thi tối thiểu tìm được k = %d (%s, %.1fs)"
          % (lb, k, method, time.time() - t0))
    T = slots if slots and slots >= k else int(-(-3 * k // 2))
    print("   số ca thi cho phép T = %d" % T)
    col = normalize(col)
    rng = random.Random(seed)
    slot_naive = list(col)                       # nhãn màu tuỳ ý làm thứ tự ca
    c0 = proximity_cost(nbw, slot_naive, S)
    slot, c1, c2 = optimize_proximity(adj, nbw, col, T, S, tlimit, rng)
    assert check(adj, slot)[0] == 0 and max(slot) < T
    assert abs(proximity_cost(nbw, slot, S) - c2) < 1e-6
    print("   chi phí khoảng cách / sinh viên: ban đầu (thứ tự ca tuỳ ý) = %.3f"
          " ; sau sắp ca = %.3f ; sau SA = %.3f" % (c0, c1, c2))
    if out:
        with open(out, "w") as f:
            for i, c in enumerate(info["courses"]):
                f.write("%s %d\n" % (c, slot[i] + 1))
        print("   đã ghi lịch thi (mã môn, ca) vào %s" % out)
    return slot

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
    for name, method in (("tabu", "tabu"), ("ls", "ls"), ("hea", "hea")):
        if name in algs or "all" in algs:
            ks, ts, t0 = [], [], time.time()
            for r in range(runs):
                col, k, tb = minimize_colors(adj, tlimit, seed=r, method=method, lb=lb)
                assert check(adj, col)[0] == 0
                ks.append(k); ts.append(tb)
                if best_overall is None or k < len(set(best_overall)):
                    best_overall = normalize(col)
            label = {"tabu": "TabuCol", "ls": "LS(1+swap)", "hea": "HEA(GPX)"}[name]
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
    gen_toronto(os.path.join(directory, "tor_demo"))
    run_toronto(os.path.join(directory, "tor_demo"), tlimit, 0)


def serve_web(port=5500):
    """Chạy máy chủ cục bộ và mở trang trực quan hoá (thư mục visualize/)."""
    import os
    import http.server
    import socketserver
    import threading
    import webbrowser
    root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(root)

    class Handler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass

    socketserver.TCPServer.allow_reuse_address = True
    url = "http://127.0.0.1:%d/visualize/index.html" % port
    with socketserver.TCPServer(("127.0.0.1", port), Handler) as httpd:
        print("Trang trực quan hoá: %s  (nhấn Ctrl+C để dừng)" % url)
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nĐã dừng máy chủ.")


def main():
    ap = argparse.ArgumentParser(description="Đề 6 - tô màu đồ thị / xếp lịch thi")
    ap.add_argument("file", nargs="?", help="file DIMACS .col")
    ap.add_argument("--alg", default="all", help="wp,dsatur,exact,tabu,ls,all (phân tách bằng dấu phẩy)")
    ap.add_argument("--time", type=float, default=30.0, help="ngân sách thời gian (giây) mỗi lần chạy")
    ap.add_argument("--runs", type=int, default=5, help="số lần chạy với seed khác nhau")
    ap.add_argument("--out", help="ghi lời giải tốt nhất ra file")
    ap.add_argument("--toronto", help="tiền tố bộ Toronto (đọc PREFIX.crs và PREFIX.stu)")
    ap.add_argument("--slots", type=int, default=0, help="số ca thi T (Toronto); 0 = 1.5 x k_min")
    ap.add_argument("--gen-toronto", metavar="PREFIX", help="sinh bộ dữ liệu giả lập dạng Toronto")
    ap.add_argument("--web", action="store_true", help="mở trang web trực quan hoá (visualize/index.html)")
    ap.add_argument("--port", type=int, default=5500, help="cổng cho --web")
    ap.add_argument("--selftest", action="store_true", help="sinh bộ test (myciel, queen, gnp) và chạy")
    ap.add_argument("--dir", default="data", help="thư mục chứa file sinh ra khi --selftest")
    a = ap.parse_args()
    if a.web:
        serve_web(a.port)
    elif a.gen_toronto:
        gen_toronto(a.gen_toronto)
        print("đã sinh %s.crs và %s.stu" % (a.gen_toronto, a.gen_toronto))
    elif a.toronto:
        m = "hea" if "hea" in a.alg or a.alg == "all" else ("tabu" if "tabu" in a.alg else "hea")
        run_toronto(a.toronto, a.time, a.slots, method=m, out=(a.out or a.toronto + ".sol"))
    elif a.selftest:
        selftest(a.time, a.runs, a.dir)
    elif a.file:
        run_file(a.file, set(a.alg.split(",")), a.time, a.runs, a.out)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()

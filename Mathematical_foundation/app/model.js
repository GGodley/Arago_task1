// The performance model and optimizer, ported line for line from
// Mathematical_foundation/foundation/{counts,parallelism,timing,profile,optimizer}.py
// so the page can size any matmul without a server.
const MODEL = (() => {
  const TILE = 64, LD = 4, LM = 1;
  const cdiv = (a, b) => Math.floor((a + b - 1) / b);
  const pyRound = x => { const f = Math.floor(x), d = x - f; return d > 0.5 ? f + 1 : d < 0.5 ? f : (f % 2 === 0 ? f : f + 1) };

  function counts(M, K, N) {
    const m = cdiv(M, TILE), k = cdiv(K, TILE), n = cdiv(N, TILE);
    return { M, K, N, m, k, n, COMPUTE: m * n * k, ACC: m * n * (k - 1), LOAD: k * (m + n), SAVE: m * n };
  }

  // parallelism.tilings: name, loads, cap, resident (Little's-law R)
  function tilings(c) {
    const { m, k, n } = c, acc = k > 1 ? 1 : 0, floor = k * (m + n), out = [];
    for (let bm = 1; bm <= m; bm++) for (let bn = 1; bn <= n; bn++)
      out.push({ name: `block${bm}x${bn}`, loads: Math.max(floor, k * (m * cdiv(n, bn) + n * cdiv(m, bm))), cap: (1 + acc) * bm * bn, resident: acc * bm * bn });
    for (let p = 1; p <= m; p++)
      out.push({ name: `Apanel${p}`, loads: Math.max(floor, m * k + n * k * cdiv(m, p)), cap: (1 + acc) * p, resident: p * (k + acc) });
    for (let p = 1; p <= n; p++)
      out.push({ name: `Bpanel${p}`, loads: Math.max(floor, n * k + m * k * cdiv(n, p)), cap: (1 + acc) * p, resident: p * (k + acc) });
    return out;
  }
  const inflightWork = (t, c) => t.loads * (LD + LM * (1 + (c.ACC ? 1 : 0))) + c.COMPUTE * 2 * LM + c.SAVE * LD;
  const littleMemory = (t, c, win) => t.resident + inflightWork(t, c) / win;

  // timing.predict_time
  function predictTime(c, nd, nm, S) {
    const { m, k, n } = c, acc = k > 1 ? 1 : 0;
    const W = m * n * (2 * k - 1), Sv = m * n;
    const tMat = 2 * LD + LM * cdiv(W, nm);
    const tPath = 2 * LD + LM * (1 + (k > 1 ? Math.ceil(Math.log2(k)) : 0));
    const tau = LD + LM * (1 + acc);
    let best = null;
    const consider = (name, L, R) => {
      const F = S - R - 1;
      if (F < 2) return;
      const tDma = (L + Sv) * Math.max(LD / nd, tau / F);
      const T = Math.max(tMat, tDma, tPath);
      if (!best || T < best.T - 1e-9 || (Math.abs(T - best.T) < 1e-9 && L < best.loads))
        best = { T, tMat, tDma, tPath, loads: L, strategy: name, R, F };
    };
    for (let bm = 1; bm <= m; bm++) for (let bn = 1; bn <= n; bn++) consider(`block${bm}x${bn}`, k * (m * cdiv(n, bn) + n * cdiv(m, bm)), acc * bm * bn);
    for (let p = 1; p <= m; p++) consider(`Apanel${p}`, m * k + n * k * cdiv(m, p), p * k + acc * p);
    for (let p = 1; p <= n; p++) consider(`Bpanel${p}`, n * k + m * k * cdiv(n, p), p * k + acc * p);
    if (best) best.limiter = best.tPath >= best.T - 1e-9 ? "path" : best.tDma >= best.tMat ? "dma" : "matrix";
    return best;
  }

  function tilingShape(name, m, k, n, acc) {
    const nums = name.match(/\d+/g).map(Number);
    if (name.startsWith("block")) return { resident: acc * nums[0] * nums[1], fill: nums[0] + nums[1] };
    const p = nums[0];
    return { resident: p * k + acc * p, fill: p * k + k };
  }

  // profile.profile (the "needed" memory curve and the usage curves)
  function profile(c, nd, nm, S, points = 90) {
    const t = predictTime(c, nd, nm, S);
    if (!t) return null;
    const { m, k, n } = c, acc = k > 1 ? 1 : 0;
    const W = m * n * (2 * k - 1), COMPUTE = m * n * k, SAVE = m * n, L = t.loads, T = t.T;
    const tau = LD + LM * (1 + acc);
    const ndEff = Math.max(1, Math.min(nd, t.F * LD / tau));
    const sh = tilingShape(t.strategy, m, k, n, acc);
    const fill = Math.min(sh.fill, L);
    const tFill = Math.min(Math.max(LD, LD * Math.ceil(fill / ndEff)), Math.max(LD, T - LD));
    const Ts = Math.max(1, T - LD - tFill);
    const lamLoad = Math.max(0, L - fill) / Ts, lamComp = COMPUTE / Math.max(1, T - 2 * LD), lamSave = SAVE / Ts;
    const inflight = lamLoad * tau + Math.min(lamComp * 2 * LM, 2 * nm) + lamSave * LD;
    const cap = Math.min(S, m * k + k * n + m * n);
    const base = Math.min(cap, sh.resident + inflight);
    const ramp0 = Math.min(nd, fill);
    const matRate = W * LM / Math.max(1, T - 2 * LD);
    const rest = (L + SAVE) * LD - ndEff * tFill - Math.min(nd, SAVE) * LD;
    const ts = [], mem = [], mat = [], dma = [];
    for (let i = 0; i < points; i++) {
      const x = T * i / (points - 1);
      let mm = x <= tFill ? ramp0 + (base - ramp0) * x / tFill : x < T - LD ? base : base * (T - x) / LD;
      mm = Math.min(mm, cap);
      const mt = x >= LD && x <= T - LD ? matRate : 0;
      const dm = x <= tFill ? ndEff : x >= T - LD ? Math.min(nd, SAVE) : Math.max(0, rest) / Ts;
      ts.push(x); mem.push(100 * mm / S); mat.push(100 * Math.min(mt, nm) / nm); dma.push(100 * Math.min(dm, nd) / nd);
    }
    const memMean = mem.reduce((a, b) => a + b, 0) / points;
    return { T, limiter: t.limiter, tiling: t.strategy, util: W * LM / (nm * T), dma_util: (L + SAVE) * LD / (nd * T),
      mem_peak: Math.max(...mem), mem_mean: memMean, t: ts, mat, dma, mem };
  }

  // optimizer.plan / s_min / s_sat / speed_steps / grid
  function plan(c, ts, target, S) {
    const W = c.COMPUTE + c.ACC, win = target - 2 * LD;
    if (win <= 0) return null;
    const nm = Math.ceil(W / win / LM - 1e-9);
    let best = null;
    for (const t of ts) {
      if (t.cap < nm) continue;
      const nd = Math.ceil((t.loads + c.SAVE) * LD / win - 1e-9);
      const mem = littleMemory(t, c, win);
      if (mem <= S && (!best || nd < best.nd || (nd === best.nd && mem < best.mem))) best = { nd, t, mem };
    }
    if (!best) return null;
    const T = 2 * LD + Math.max(Math.ceil(W / nm) * LM, (best.t.loads + c.SAVE) * LD / best.nd);
    return { Nd: best.nd, Nm: nm, tiling: best.t.name, mem_needed: best.mem, T_plan: T };
  }
  function bounds(c, ts, target) {
    const W = c.COMPUTE + c.ACC, win = target - 2 * LD, nm = Math.ceil(W / win / LM - 1e-9);
    let lo = null, sat = null;
    for (const t of ts) {
      if (t.cap < nm) continue;
      const nd = Math.ceil((t.loads + c.SAVE) * LD / win - 1e-9), mem = littleMemory(t, c, win);
      if (lo === null || mem < lo) lo = mem;
      if (!sat || nd < sat.nd || (nd === sat.nd && mem < sat.mem)) sat = { nd, mem };
    }
    return [lo, sat ? sat.mem : null];
  }
  function speedSteps(c, count = 8) {
    const W = c.COMPUTE + c.ACC, acc = c.k > 1 ? 1 : 0;
    const pmax = W / Math.ceil(W / ((1 + acc) * c.m * c.n));
    const steps = [...new Set(Array.from({ length: count }, (_, i) => Math.max(1, pyRound(Math.pow(pmax, i / (count - 1))))))].sort((a, b) => a - b);
    const out = [], seen = new Set();
    for (const p of steps) {
      const T = Math.ceil(W / p);
      if (p <= Math.max(1, pyRound(pmax)) && !seen.has(T)) { seen.add(T); out.push(p) }
    }
    return out;
  }
  function grid(M, K, N, name) {
    const c = counts(M, K, N), ts = tilings(c), W = c.COMPUTE + c.ACC;
    const speeds = speedSteps(c), targets = speeds.map(p => 2 * LD + Math.ceil(W / p) * LM);
    const bs = targets.map(T => bounds(c, ts, T));
    const top = Math.max(...bs.map(b => b[1] || 0));
    let mems = [];
    for (let e = 2; e < 15; e++) if (2 ** e < top * 2) mems.push(2 ** e);
    if (!mems.length) mems = [4];
    if (mems[mems.length - 1] < top) mems.push(mems[mems.length - 1] * 2);
    const r1 = x => Math.round(x * 10) / 10;
    const cells = targets.map(T => mems.map(S => {
      const p = plan(c, ts, T, S);
      if (!p) return null;
      const pr = profile(c, p.Nd, p.Nm, S);
      return { Nd: p.Nd, Nm: p.Nm, tiling: p.tiling, T: r1(pr.T), T_plan: r1(p.T_plan), limiter: pr.limiter,
        util: r1(100 * pr.util), dma_util: r1(100 * Math.min(1, pr.dma_util)), mem_needed_pct: r1(100 * p.mem_needed / S),
        mem_peak: r1(pr.mem_peak), mem_mean: r1(pr.mem_mean), t: pr.t, mat: pr.mat, dma: pr.dma, mem: pr.mem };
    }));
    return { name: name || `${M}×${K}×${N}`, M, K, N, m: c.m, k: c.k, n: c.n, W, COMPUTE: c.COMPUTE, ACC: c.ACC, SAVE: c.SAVE,
      LOAD_min: c.LOAD, speeds, targets, mems, s_min: bs.map(b => b[0] && r1(b[0])), s_sat: bs.map(b => b[1] && r1(b[1])), cells };
  }
  return { grid, counts, predictTime, profile, plan };
})();
if (typeof module !== "undefined") module.exports = MODEL;

/**
 * Desk One decision layer.
 *
 * Usage-router policy follows grok-bot-jev (Bodila51): same action order,
 * thresholds, and bypass markers as src/router.py + config.example.yaml.
 * Answers from `financeDecision` are the offline stand-in. The desk replaces
 * them with the Jev fast revaluator when that call returns a live card.
 *
 * The finance loop follows the public patterns in yibie/awesome-jev
 * (finance & trading): 30-day Noul, long/flat/short Choice, conviction Score,
 * a decision card that does not send, and a hard-rule veto beside the model.
 */

export const STARTING_CASH = 100_000;

export const BYPASS_MARKERS = ["bypass jev", "bypass jev:", "no jev"] as const;

export const ACTIONS = [
  "reuse_cache",
  "stop_retry",
  "run_deterministic",
  "chat_only",
  "research_capped",
  "allow_subagent",
  "ask_human",
  "proceed_full",
] as const;

export type Action = (typeof ACTIONS)[number];

export const KINDS = [
  "chat",
  "lookup",
  "research",
  "browser",
  "coding",
  "write",
  "account",
  "unknown",
] as const;

export type Kind = (typeof KINDS)[number];

export type Mode = "shadow" | "active";

export type RouterConfig = {
  enabled: boolean;
  mode: Mode;
  model: string;
  thresholds: {
    min_choice_confidence: number;
    reuse_min: number;
    subagent_min: number;
  };
  limits: {
    max_browser_sources: number;
    max_retries_same_error: number;
  };
};

export const DEFAULT_CONFIG: RouterConfig = {
  enabled: true,
  mode: "shadow",
  model: "jev-latest",
  thresholds: {
    min_choice_confidence: 0.55,
    reuse_min: 0.65,
    subagent_min: 0.75,
  },
  limits: {
    max_browser_sources: 5,
    max_retries_same_error: 1,
  },
};

export type TaskState = {
  goal: string;
  kind?: Kind;
  cached_artifact?: boolean;
  cached_note?: string;
  cache_age_min?: number;
  prior_error?: string;
  same_error_count?: number;
  sources_found?: number;
  constraints?: string;
  symbol?: string;
  scan?: boolean;
};

export type RouteDetails = {
  intent: Kind;
  intent_confidence: number;
  intent_probs: Record<string, number>;
  reuse_cache: number;
  needs_subagent: number;
  stop_retry: number;
  complexity_0_1: number;
  max_browser_sources: number;
};

export type RouteResult = {
  action: Action;
  reason: string;
  mode: Mode;
  jev_used: boolean;
  details: Partial<RouteDetails>;
  policy: {
    honor_in_active_mode: boolean;
    shadow_mode_is_advisory: boolean;
  };
};

export type WorkKind = "scan" | "symbol" | "cache" | "book" | "explain" | "stop" | "approve";

export type WorkPlan = {
  kind: WorkKind;
  cap: number;
  enforced: boolean;
};

export type Side = "long" | "flat" | "short";
export type RiskLevel = "low" | "watch" | "high" | "freeze";
export type Verdict = "ALLOW" | "VETO" | "ESCALATE" | "STAND ASIDE";

export type MarketSpec = {
  id: string;
  name: string;
  venue: "perp" | "equity" | "hk";
  base: number;
  vol: number;
  drift: number;
  seed: number;
  spread: number;
  bias: number;
};

export const MARKETS: MarketSpec[] = [
  { id: "BTC", name: "Bitcoin", venue: "perp", base: 67420, vol: 0.016, drift: 0.0011, seed: 11, spread: 4, bias: 0.04 },
  { id: "ETH", name: "Ether", venue: "perp", base: 3520, vol: 0.018, drift: -0.0007, seed: 23, spread: 5, bias: -0.1 },
  { id: "SOL", name: "Solana", venue: "perp", base: 178, vol: 0.026, drift: 0.0016, seed: 41, spread: 7, bias: 0.12 },
  { id: "NVDA", name: "NVIDIA", venue: "equity", base: 131.4, vol: 0.012, drift: 0.0009, seed: 7, spread: 3, bias: 0.08 },
  { id: "0700", name: "Tencent", venue: "hk", base: 418, vol: 0.011, drift: -0.00055, seed: 19, spread: 6, bias: -0.14 },
  { id: "PEPE", name: "Pepe", venue: "perp", base: 0.0118, vol: 0.034, drift: 0.0022, seed: 53, spread: 21, bias: 0.18 },
];

export type Snapshot = {
  price: number;
  ret1: number;
  ret5: number;
  spreadBps: number;
  raw: number;
  deduped: number;
};

export type TypedAnswer = {
  question: string;
  type: "Choice" | "Noul" | "Score";
  answer: string;
  confidence: number | null;
};

export type DecisionCard = {
  symbol: string;
  at: number;
  side: Side;
  higher: number;
  conviction: number;
  sideConfidence: number;
  risk: RiskLevel;
  riskConfidence: number;
  freezeProb: number;
  verdict: Verdict;
  vetoes: string[];
  entry: number | null;
  stop: number | null;
  target: number | null;
  evidence: string[];
  evidenceKept: number;
  evidenceDeduped: number;
  questions: TypedAnswer[];
  pattern: string;
  source?: "jev" | "local";
  status?: string;
};

export type Position = { qty: number; avg: number };

export type FillInput = { symbol: string; qty: number; price: number };

const EPOCH = Date.UTC(2026, 0, 2);

export function marketById(id: string): MarketSpec {
  const found = MARKETS.find((m) => m.id === id);
  if (!found) throw new Error(`Unknown market ${id}`);
  return found;
}

function clamp01(n: number): number {
  return Math.min(0.98, Math.max(0.02, n));
}

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

function wave(seed: number, tDays: number): number {
  let x = 0;
  let norm = 0;
  for (let k = 1; k <= 5; k++) {
    const amp = 1 / k;
    const freq = 0.42 * k + (seed % 5) * 0.03;
    const phase = (((seed * (k + 3)) % 360) * Math.PI) / 180;
    x += amp * Math.sin(tDays * freq + phase);
    norm += amp;
  }
  return x / norm;
}

export function priceAt(spec: MarketSpec, now: number): number {
  const t = (now - EPOCH) / 86_400_000;
  const raw = spec.base * Math.exp(spec.drift * t + spec.vol * 1.35 * wave(spec.seed, t));
  return raw;
}

export function snapshot(spec: MarketSpec, now: number): Snapshot {
  const price = priceAt(spec, now);
  const prev1 = priceAt(spec, now - 86_400_000);
  const prev5 = priceAt(spec, now - 5 * 86_400_000);
  const ret1 = price / prev1 - 1;
  const ret5 = price / prev5 - 1;
  const spreadBps = spec.spread + Math.min(8, Math.abs(ret1) * 220);
  const raw = 80 + (spec.seed % 420);
  const deduped = Math.max(12, Math.round(raw * 0.42));
  return { price, ret1, ret5, spreadBps, raw, deduped };
}

export function series(spec: MarketSpec, now: number, points = 48): { t: number; p: number }[] {
  const span = 6 * 86_400_000;
  const out: { t: number; p: number }[] = [];
  for (let i = 0; i < points; i++) {
    const t = now - span + (span * i) / (points - 1);
    out.push({ t, p: priceAt(spec, t) });
  }
  return out;
}

function distribution(choice: string, confidence: number, labels: readonly string[]): Record<string, number> {
  const rest = (1 - confidence) / (labels.length - 1);
  const out: Record<string, number> = {};
  for (const label of labels) out[label] = label === choice ? confidence : rest;
  return out;
}

export function bypassed(state: TaskState): boolean {
  const raw = [state.goal, state.constraints ?? ""].join(" ").toLowerCase();
  return BYPASS_MARKERS.some((marker) => raw.includes(marker));
}

const INTENT_LABELS = ["chat", "lookup", "research", "browser", "coding", "write", "account"] as const;

function inferIntent(state: TaskState): { choice: Kind; confidence: number } {
  const goal = `${state.goal} ${state.constraints ?? ""}`.toLowerCase();
  const account =
    /\b(place|submit|execute|pay|send|publish|delete)\b/.test(goal) || state.kind === "account";
  if (account) {
    return { choice: "account", confidence: state.kind === "account" ? 0.86 : 0.71 };
  }
  if (state.kind && state.kind !== "unknown") {
    return { choice: state.kind, confidence: 0.8 };
  }
  if (/\b(explain|why|what does|plain language)\b/.test(goal)) {
    return { choice: "chat", confidence: 0.7 };
  }
  if (/\b(position|pnl|risk|status|book|largest)\b/.test(goal)) {
    return { choice: "lookup", confidence: 0.72 };
  }
  if (/\b(browser|click through)\b/.test(goal)) {
    return { choice: "browser", confidence: 0.66 };
  }
  return { choice: "research", confidence: 0.63 };
}

function reuseNoul(state: TaskState): number {
  if (!state.cached_artifact) return 0.08;
  const age = state.cache_age_min ?? 4;
  if (age <= 15) return 0.91;
  if (age <= 45) return 0.73;
  if (age <= 180) return 0.47;
  return 0.22;
}

function stopNoul(state: TaskState): number {
  const n = state.same_error_count ?? 0;
  let stop = 0.12 + Math.min(3, n) * 0.38;
  if (state.prior_error) stop += 0.16;
  return Math.min(0.96, stop);
}

function subagentNoul(state: TaskState, intent: Kind): number {
  const goal = state.goal.toLowerCase();
  if (state.scan || /\bevery|all markets|universe\b/.test(goal)) return 0.86;
  if (intent === "browser") return 0.71;
  if (intent === "coding") return 0.62;
  if (intent === "research") return 0.36;
  return 0.14;
}

function complexityScore(intent: Kind): number {
  const table: Record<Kind, number> = {
    chat: 0.3,
    lookup: 0.45,
    research: 1.22,
    browser: 1.8,
    coding: 1.55,
    write: 1.05,
    account: 0.95,
    unknown: 1,
  };
  return table[intent];
}

export function routeTask(state: TaskState, cfg: RouterConfig = DEFAULT_CONFIG): RouteResult {
  const mode = cfg.mode;
  const policy = {
    honor_in_active_mode: true,
    shadow_mode_is_advisory: mode === "shadow",
  };

  if (!cfg.enabled || bypassed(state)) {
    return {
      action: "proceed_full",
      reason: !cfg.enabled ? "router disabled" : "bypass jev",
      mode,
      jev_used: false,
      details: {},
      policy,
    };
  }

  const thr = cfg.thresholds;
  const limits = cfg.limits;
  const intent = inferIntent(state);
  const reuse = reuseNoul(state);
  const stop = stopNoul(state);
  const sub = subagentNoul(state, intent.choice);
  const complexity = complexityScore(intent.choice);
  const same = state.same_error_count ?? 0;

  let action: Action = "proceed_full";
  let reason = "default full work";

  if (state.cached_artifact && reuse >= thr.reuse_min) {
    action = "reuse_cache";
    reason = `reuse_cache noul=${reuse.toFixed(2)}`;
  } else if (same >= limits.max_retries_same_error && stop >= 0.55) {
    action = "stop_retry";
    reason = `stop_retry noul=${stop.toFixed(2)} same_error_count=${same}`;
  } else if (intent.choice === "lookup" && intent.confidence >= thr.min_choice_confidence) {
    action = "run_deterministic";
    reason = "intent=lookup";
  } else if (intent.choice === "chat" && intent.confidence >= thr.min_choice_confidence) {
    action = "chat_only";
    reason = "intent=chat";
  } else if (intent.choice === "account") {
    action = "ask_human";
    reason = "account/irreversible class — require approval";
  } else if (sub >= thr.subagent_min) {
    action = "allow_subagent";
    reason = `needs_subagent noul=${sub.toFixed(2)}`;
  } else if (intent.choice === "research" || intent.choice === "browser") {
    action = "research_capped";
    reason = `cap sources at ${limits.max_browser_sources}`;
  }

  const details: RouteDetails = {
    intent: intent.choice,
    intent_confidence: round4(intent.confidence),
    intent_probs: Object.fromEntries(
      Object.entries(distribution(intent.choice, intent.confidence, INTENT_LABELS)).map(([k, v]) => [
        k,
        round4(v),
      ]),
    ),
    reuse_cache: round4(reuse),
    needs_subagent: round4(sub),
    stop_retry: round4(stop),
    complexity_0_1: round4(complexity / 2),
    max_browser_sources: limits.max_browser_sources,
  };

  return { action, reason, mode, jev_used: true, details, policy };
}

function round4(n: number): number {
  return Math.round(n * 10_000) / 10_000;
}

export function planWork(route: RouteResult, state: TaskState, cfg: RouterConfig = DEFAULT_CONFIG): WorkPlan {
  const enforced = route.jev_used && route.mode === "active";
  const capFull = 8;
  const capLimited = cfg.limits.max_browser_sources;

  if (!enforced) {
    if (state.scan) return { kind: "scan", cap: capFull, enforced: false };
    if (state.kind === "chat") return { kind: "explain", cap: 0, enforced: false };
    if (state.kind === "lookup") return { kind: "book", cap: 0, enforced: false };
    if (state.kind === "account") return { kind: "approve", cap: 0, enforced: false };
    return { kind: "symbol", cap: capFull, enforced: false };
  }

  switch (route.action) {
    case "reuse_cache":
      return { kind: "cache", cap: 0, enforced: true };
    case "stop_retry":
      return { kind: "stop", cap: 0, enforced: true };
    case "chat_only":
      return { kind: "explain", cap: 0, enforced: true };
    case "run_deterministic":
      return { kind: "book", cap: 1, enforced: true };
    case "ask_human":
      return { kind: "approve", cap: 0, enforced: true };
    case "research_capped":
      return state.scan
        ? { kind: "scan", cap: capLimited, enforced: true }
        : { kind: "symbol", cap: capLimited, enforced: true };
    case "allow_subagent":
      return state.scan
        ? { kind: "scan", cap: capLimited, enforced: true }
        : { kind: "symbol", cap: capLimited, enforced: true };
    default:
      return state.scan
        ? { kind: "scan", cap: capFull, enforced: true }
        : { kind: "symbol", cap: capFull, enforced: true };
  }
}

function evidenceLines(spec: MarketSpec, snap: Snapshot): string[] {
  const day = (snap.ret1 * 100).toFixed(2);
  const week = (snap.ret5 * 100).toFixed(2);
  return [
    `5-session return ${week}%. 1-session return ${day}%.`,
    `Spread ${snap.spreadBps.toFixed(1)} bps. Vol scale ${(spec.vol * 100).toFixed(1)}%.`,
    `${snap.raw} raw items, ${snap.deduped} left after dedupe.`,
    `Slow drift ${spec.drift >= 0 ? "positive" : "negative"} on the ${spec.venue} book.`,
    `Mid ${formatPx(snap.price)}. Seeded path, not a vendor feed.`,
    `Momentum and drift ${Math.sign(snap.ret5) === Math.sign(spec.drift) || spec.drift === 0 ? "agree" : "disagree"}.`,
    `Evidence window is the last six sessions on this tape.`,
    `No account credentials are attached to this state.`,
  ];
}

export function financeDecision(
  spec: MarketSpec,
  now: number,
  cap: number,
  book: { lossFrac: number; grossFrac: number },
): DecisionCard {
  const snap = snapshot(spec, now);
  const z = 78 * snap.ret5 + 5.2 * spec.bias + 28 * snap.ret1 - (spec.vol - 0.014) * 8;
  const higher = clamp01(sigmoid(z));
  let side: Side = "flat";
  if (higher >= 0.58 && spec.vol < 0.04) side = "long";
  else if (higher <= 0.42) side = "short";
  const sideConfidence = clamp01(0.5 + Math.abs(higher - 0.5) * 1.35);
  const conviction = clamp01(
    0.35 + Math.abs(higher - 0.5) * 0.9 + (snap.spreadBps < 10 ? 0.08 : 0) - spec.vol * 2,
  );

  const freezeProb = clamp01(spec.spread / 48 + spec.vol * 6 + (snap.spreadBps > 18 ? 0.12 : 0) - 0.12);
  let risk: RiskLevel = "low";
  if (snap.spreadBps >= 22 || freezeProb >= 0.72) risk = "freeze";
  else if (snap.spreadBps >= 14 || spec.vol >= 0.03) risk = "high";
  else if (spec.vol >= 0.02 || Math.abs(snap.ret5) > 0.03) risk = "watch";
  const riskConfidence = clamp01(0.55 + Math.abs(freezeProb - 0.5) * 0.5);

  const vetoes: string[] = [];
  if (snap.spreadBps >= 18) vetoes.push(`Spread ${snap.spreadBps.toFixed(1)} bps is over the 18 bp hard cap.`);
  if (book.lossFrac <= -0.025) vetoes.push("Session loss is past 2.5% of starting equity.");
  if (book.grossFrac >= 0.8) vetoes.push("Gross exposure is past 80% of equity.");
  if (risk === "freeze") vetoes.push("Risk Choice is freeze. The rule vetoes the ticket.");

  let verdict: Verdict = "ALLOW";
  if (vetoes.length > 0) verdict = "VETO";
  else if (side === "flat") verdict = "STAND ASIDE";
  else if (sideConfidence < DEFAULT_CONFIG.thresholds.min_choice_confidence) verdict = "ESCALATE";

  const atr = snap.price * Math.max(0.008, spec.vol * 1.15);
  const entry = side === "flat" ? null : snap.price;
  const stop = entry == null ? null : side === "long" ? entry - atr : entry + atr;
  const target = entry == null ? null : side === "long" ? entry + atr * 1.8 : entry - atr * 1.8;

  const kept = evidenceLines(spec, snap).slice(0, Math.max(0, cap));

  const questions: TypedAnswer[] = [
    {
      question: "Likely higher over the next 30 days?",
      type: "Noul",
      answer: higher.toFixed(2),
      confidence: null,
    },
    {
      question: "This round: long, flat, or short?",
      type: "Choice",
      answer: side,
      confidence: sideConfidence,
    },
    {
      question: "How much conviction is justified?",
      type: "Score",
      answer: conviction.toFixed(2),
      confidence: null,
    },
    {
      question: "Deposit-style risk level?",
      type: "Choice",
      answer: risk,
      confidence: riskConfidence,
    },
  ];

  return {
    symbol: spec.id,
    at: now,
    side,
    higher,
    conviction,
    sideConfidence,
    risk,
    riskConfidence,
    freezeProb,
    verdict,
    vetoes,
    entry,
    stop: stop != null && stop > 0 ? stop : stop != null ? snap.price * 0.5 : null,
    target: target != null && target > 0 ? target : null,
    evidence: kept,
    evidenceKept: kept.length,
    evidenceDeduped: snap.deduped,
    questions,
    pattern: "Jevinik Noul · jev-trade Choice · jev_stock state · sentiment card · jev-guard veto",
  };
}

export function proposeQty(spec: MarketSpec, price: number, stop: number, equity: number, side: Side): number {
  if (side === "flat" || equity <= 0) return 0;
  const dist = Math.abs(price - stop);
  if (dist <= 0) return 0;
  const riskBudget = equity * 0.004;
  let qty = riskBudget / dist;
  const maxNotional = equity * 0.12;
  if (qty * price > maxNotional) qty = maxNotional / price;
  if (spec.venue === "perp" && price < 5) qty = Math.max(1, Math.round(qty));
  else if (spec.venue === "perp") qty = Math.round(qty * 100) / 100;
  else qty = Math.max(1, Math.floor(qty));
  if (qty * price > equity * 0.2 && spec.venue !== "perp") qty = Math.max(1, Math.floor((equity * 0.12) / price));
  return side === "short" ? -qty : qty;
}

export function applyFill(
  positions: Record<string, Position>,
  fill: FillInput,
): { positions: Record<string, Position>; realized: number } {
  const prev = positions[fill.symbol] ?? { qty: 0, avg: 0 };
  let realized = 0;
  let next: Position | null;

  if (prev.qty === 0 || Math.sign(prev.qty) === Math.sign(fill.qty)) {
    const qty = prev.qty + fill.qty;
    const avg = qty === 0 ? 0 : (prev.avg * prev.qty + fill.price * fill.qty) / qty;
    next = { qty, avg };
  } else if (Math.abs(fill.qty) < Math.abs(prev.qty)) {
    const closed = Math.abs(fill.qty);
    realized = (fill.price - prev.avg) * Math.sign(prev.qty) * closed;
    next = { qty: prev.qty + fill.qty, avg: prev.avg };
  } else {
    const closed = Math.abs(prev.qty);
    realized = (fill.price - prev.avg) * Math.sign(prev.qty) * closed;
    const remainder = prev.qty + fill.qty;
    next = remainder === 0 ? null : { qty: remainder, avg: fill.price };
  }

  const out = { ...positions };
  if (!next || next.qty === 0) delete out[fill.symbol];
  else out[fill.symbol] = next;
  return { positions: out, realized };
}

export function unrealized(positions: Record<string, Position>, marks: Record<string, number>): number {
  let u = 0;
  for (const [symbol, pos] of Object.entries(positions)) {
    const mark = marks[symbol];
    if (mark == null) continue;
    u += (mark - pos.avg) * pos.qty;
  }
  return u;
}

export function grossNotional(positions: Record<string, Position>, marks: Record<string, number>): number {
  let g = 0;
  for (const [symbol, pos] of Object.entries(positions)) {
    const mark = marks[symbol];
    if (mark == null) continue;
    g += Math.abs(pos.qty * mark);
  }
  return g;
}

export function formatPx(n: number): string {
  const abs = Math.abs(n);
  const digits = abs >= 1000 ? 1 : abs >= 100 ? 2 : abs >= 1 ? 3 : 5;
  return n.toFixed(digits);
}

export function formatMoney(n: number): string {
  const sign = n < 0 ? "-" : "";
  const [whole, frac] = Math.abs(n).toFixed(2).split(".");
  const withCommas = whole.replace(/\B(?=(\d{3})+(?!\d))/g, ",");
  return `${sign}$${withCommas}.${frac}`;
}

export function formatQty(n: number): string {
  const abs = Math.abs(n);
  const body = abs >= 100 ? abs.toFixed(0) : abs >= 1 ? abs.toFixed(2) : abs.toFixed(4);
  return n < 0 ? `-${body}` : body;
}

export function ageMinutes(ts: number, now: number): number {
  return Math.max(0, (now - ts) / 60_000);
}

import { useEffect, useMemo, useRef, useState } from "react";
import { Line, LineChart, ResponsiveContainer, YAxis } from "recharts";
import {
  ageMinutes,
  DEFAULT_CONFIG,
  financeDecision,
  formatMoney,
  formatPx,
  formatQty,
  grossNotional,
  holdSide,
  marketById,
  MARKETS,
  planWork,
  priceAt,
  proposeQty,
  routeTask,
  series,
  STARTING_CASH,
  unrealized,
  type DecisionCard,
  type MarketSpec,
  type RouteResult,
  type TaskState,
  type WorkPlan,
} from "../../lib/desk/engine";
import { anchorToTape, headlinesFor, revalueTape } from "../../lib/desk/revalue";
import { useDesk } from "../../lib/desk/store";
import { clock, cn, kicker, mono } from "./ui";

type Pane = "tape" | "decision" | "router";

type Pending = {
  symbol: string;
  qty: number;
  price: number;
  stop: number | null;
  target: number | null;
  label: string;
  blocked: string | null;
};

const PRESET_IDS = ["bias", "side", "card", "cache", "retry", "book", "order", "scan", "why"] as const;
type PresetId = (typeof PRESET_IDS)[number];

const PRESET_LABEL: Record<PresetId, string> = {
  bias: "30-day bias",
  side: "Long or short",
  card: "Decision card",
  cache: "Reuse last card",
  retry: "Retry entry",
  book: "Book risk",
  order: "Paper order",
  scan: "Scan the tape",
  why: "Explain veto",
};

export function Desk() {
  const mode = useDesk((s) => s.mode);
  const enabled = useDesk((s) => s.enabled);
  const realized = useDesk((s) => s.realized);
  const positions = useDesk((s) => s.positions);
  const fills = useDesk((s) => s.fills);
  const logs = useDesk((s) => s.logs);
  const cache = useDesk((s) => s.cache);
  const errors = useDesk((s) => s.errors);
  const selected = useDesk((s) => s.selected);
  const setMode = useDesk((s) => s.setMode);
  const setEnabled = useDesk((s) => s.setEnabled);
  const select = useDesk((s) => s.select);
  const remember = useDesk((s) => s.remember);
  const log = useDesk((s) => s.log);
  const bumpError = useDesk((s) => s.bumpError);
  const clearError = useDesk((s) => s.clearError);
  const commitFill = useDesk((s) => s.commitFill);
  const resetBook = useDesk((s) => s.resetBook);

  const [now, setNow] = useState<number | null>(null);
  const [pane, setPane] = useState<Pane>("decision");
  const [goal, setGoal] = useState("Is BTC likely to trade higher over the next 30 days?");
  const [route, setRoute] = useState<RouteResult | null>(null);
  const [plan, setPlan] = useState<WorkPlan | null>(null);
  const [card, setCard] = useState<DecisionCard | null>(null);
  const [scan, setScan] = useState<DecisionCard[] | null>(null);
  const [note, setNote] = useState<string | null>(null);
  const [pending, setPending] = useState<Pending | null>(null);
  const [confirmReset, setConfirmReset] = useState(false);
  const [payloadOpen, setPayloadOpen] = useState(false);
  const revalueTicket = useRef(0);

  useEffect(() => {
    void useDesk.persist.rehydrate();
    setNow(Date.now());
    const id = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(id);
  }, []);

  const clockNow = now ?? Date.UTC(2026, 8, 23, 4, 0, 0);

  const marks = useMemo(() => {
    const out: Record<string, number> = {};
    for (const spec of MARKETS) out[spec.id] = priceAt(spec, clockNow);
    return out;
  }, [clockNow]);

  const openPnl = unrealized(positions, marks);
  const equity = STARTING_CASH + realized + openPnl;
  const gross = grossNotional(positions, marks);
  const lossFrac = (realized + openPnl) / STARTING_CASH;
  const grossFrac = equity > 0 ? gross / equity : 0;
  const book = { lossFrac, grossFrac };

  function cfg() {
    return { ...DEFAULT_CONFIG, mode, enabled };
  }

  function bookSummary(): string {
    const lines = Object.entries(positions);
    if (lines.length === 0) {
      return `Book is flat. Equity ${formatMoney(equity)}. Realized ${formatMoney(realized)}. Nothing to look up beyond the cash figure.`;
    }
    let top = lines[0][0];
    let topN = 0;
    for (const [symbol, pos] of lines) {
      const n = Math.abs(pos.qty * (marks[symbol] ?? pos.avg));
      if (n >= topN) {
        top = symbol;
        topN = n;
      }
    }
    return `Equity ${formatMoney(equity)}. Realized ${formatMoney(realized)}. Unrealized ${formatMoney(openPnl)}. ${lines.length} open line${lines.length === 1 ? "" : "s"}. Largest is ${top} at ${formatMoney(topN)} notional.`;
  }

  function explain(current: DecisionCard | null): string {
    if (!current) {
      return "No card yet. Run a 30-day bias, or open PEPE — its spread sits over the hard cap so the rule vetoes even when the Choice is long.";
    }
    if (current.vetoes.length > 0) {
      return `${current.symbol}: ${current.vetoes.join(" ")} Jev risk Choice is ${current.risk} (freeze probability ${current.freezeProb.toFixed(2)}). Composed verdict ${current.verdict}. The model does not outvote the hard rule.`;
    }
    return `${current.symbol} verdict is ${current.verdict}. Choice ${current.side} at confidence ${current.sideConfidence.toFixed(2)}. 30-day Noul ${current.higher.toFixed(2)}. Conviction score ${current.conviction.toFixed(2)}. No hard-rule veto.`;
  }

  function stageFromCard(current: DecisionCard | null, label: string) {
    if (!current || current.entry == null || current.stop == null || current.target == null || current.side === "flat") {
      setPending({
        symbol: current?.symbol ?? selected,
        qty: 0,
        price: marks[current?.symbol ?? selected] ?? 0,
        stop: null,
        target: null,
        label,
        blocked: "No actionable card on this symbol. Run a bias first. Nothing was sent.",
      });
      return;
    }
    if (current.verdict !== "ALLOW") {
      setPending({
        symbol: current.symbol,
        qty: 0,
        price: current.entry,
        stop: current.stop,
        target: current.target,
        label,
        blocked: current.vetoes[0] ?? `Verdict is ${current.verdict}. Ticket stays closed.`,
      });
      return;
    }
    const spec = marketById(current.symbol);
    const px = marks[current.symbol] ?? current.entry;
    const qty = proposeQty(spec, px, current.stop, equity, current.side);
    setPending({
      symbol: current.symbol,
      qty,
      price: px,
      stop: current.stop,
      target: current.target,
      label,
      blocked: qty === 0 ? "Size rounded to zero." : null,
    });
  }

  function run(state: TaskState) {
    const config = cfg();
    const next = routeTask(state, config);
    const nextPlan = planWork(next, state, config);
    setGoal(state.goal);
    setRoute(next);
    setPlan(nextPlan);
    setPending(null);
    log({
      ts: Date.now(),
      goal: state.goal,
      action: next.action,
      reason: next.reason,
      mode: next.mode,
      jev_used: next.jev_used,
      enforced: nextPlan.enforced,
    });

    const symbol = state.symbol ?? selected;
    if (state.symbol) select(state.symbol);

    if (nextPlan.kind === "scan" || nextPlan.kind === "symbol") {
      const ids = nextPlan.kind === "scan" ? MARKETS.map((item) => item.id) : [symbol];
      const scanning = nextPlan.kind === "scan";
      const cap = nextPlan.cap;
      const show = (cards: DecisionCard[], note: string) => {
        const cacheNow = useDesk.getState().cache;
        const held = cards.map((card) => holdSide(cacheNow[card.symbol] ?? null, card));
        const ranked = scanning ? [...held].sort((a, b) => rank(b) - rank(a)) : held;
        for (const item of ranked) remember(item);
        if (scanning) {
          setScan(ranked);
          setCard(ranked.find((item) => item.symbol === symbol) ?? ranked[0] ?? null);
        } else {
          setScan(null);
          setCard(ranked[0] ?? null);
        }
        setNote(note);
      };
      const localNow = Date.now();
      show(
        ids.map((id) => financeDecision(marketById(id), localNow, cap, book)),
        "Local tape is up. Jev is revaluing…",
      );
      const ticket = revalueTicket.current + 1;
      revalueTicket.current = ticket;
      void headlinesFor(ids)
        .then((headlines) => revalueTape(ids, book, headlines))
        .then((result) => {
          if (ticket !== revalueTicket.current) return;
          const stamped = Date.now();
          const merged = ids.map((id) => {
            const remote = result.cards.find((item) => item.symbol === id);
            const specForId = marketById(id);
            if (!remote || remote.source !== "jev" || remote.status !== "ok") {
              return financeDecision(specForId, stamped, cap, book);
            }
            return anchorToTape(remote, specForId, stamped, book, cap);
          });
          const live = merged.filter((item) => item.source === "jev");
          const missed = ids.filter((id) => !live.some((item) => item.symbol === id));
          const timing = Number.isFinite(result.elapsed_ms) ? ` in ${Math.round(result.elapsed_ms)} ms` : "";
          const note =
            live.length === 0
              ? "Jev did not return a live card. The tape stayed on the local stand-in."
              : missed.length === 0
                ? `Jev revalued ${live.length} name${live.length === 1 ? "" : "s"}${timing}. Hard rules still veto.`
                : `Jev revalued ${live.length} of ${ids.length}${timing}. ${missed.join(", ")} stayed on the local tape.`;
          show(merged, note);
        })
        .catch(() => {
          if (ticket !== revalueTicket.current) return;
          const stamped = Date.now();
          show(
            ids.map((id) => financeDecision(marketById(id), stamped, cap, book)),
            "Jev revalue failed. The tape stayed on the local stand-in.",
          );
        });
      return;
    }

    if (nextPlan.kind === "cache") {
      const cached = useDesk.getState().cache[symbol] ?? null;
      setCard(cached);
      setScan(null);
      setNote(
        cached
          ? `Reused the card from ${Math.round(ageMinutes(cached.at, Date.now()))} min ago. No new evidence pass.`
          : "Router asked for cache, but this symbol has none. Active mode will not invent a fresh pass.",
      );
      return;
    }

    setScan(null);
    if (nextPlan.kind === "stop") {
      setNote(
        `Stopped. ${next.reason}. Clear the failure count or change the goal. The same entry was not run again.`,
      );
      return;
    }
    if (nextPlan.kind === "book") {
      setNote(bookSummary());
      return;
    }
    if (nextPlan.kind === "explain") {
      setNote(explain(useDesk.getState().cache[symbol] ?? card));
      return;
    }
    const current = useDesk.getState().cache[symbol] ?? (card?.symbol === symbol ? card : null);
    setCard(current);
    setNote("Account class. The router will not treat this as permission to send. Confirm below, or do nothing.");
    stageFromCard(current, "Paper order");
  }

  function presetState(id: PresetId): TaskState {
    const symbol = selected;
    const cached = cache[symbol];
    const age = cached ? ageMinutes(cached.at, Date.now()) : undefined;
    if (id === "bias") {
      return { goal: `Is ${symbol} likely to trade higher over the next 30 days?`, kind: "research", symbol };
    }
    if (id === "side") {
      return { goal: `Choose long, flat, or short on ${symbol} this round`, kind: "research", symbol };
    }
    if (id === "card") {
      return {
        goal: `Build an entry, stop, and target card for ${symbol}. Do not send a live order.`,
        kind: "research",
        symbol,
      };
    }
    if (id === "cache") {
      return {
        goal: `Re-check ${symbol} using the cached decision if it is still fresh`,
        kind: "research",
        symbol,
        cached_artifact: Boolean(cached),
        cache_age_min: age,
        cached_note: cached ? `age=${Math.round(age ?? 0)}m` : "none",
      };
    }
    if (id === "retry") {
      const count = errors[symbol] ?? 0;
      return {
        goal: `Retry the same ${symbol} entry`,
        kind: "research",
        symbol,
        prior_error: count > 0 ? "entry rejected: spread gate" : "",
        same_error_count: count,
      };
    }
    if (id === "book") return { goal: "What is open risk and the largest line?", kind: "lookup" };
    if (id === "order") {
      return { goal: `Place a paper order on ${symbol} at the decision card`, kind: "account", symbol };
    }
    if (id === "scan") {
      return { goal: "Scan every listed market and rank the typed decisions", kind: "research", scan: true };
    }
    return { goal: "Explain the last risk veto in plain language", kind: "chat", symbol };
  }

  function confirmPending() {
    if (!pending || pending.blocked || pending.qty === 0 || now == null) return;
    commitFill({
      ts: Date.now(),
      symbol: pending.symbol,
      qty: pending.qty,
      price: pending.price,
      stop: pending.stop,
      target: pending.target,
    });
    setPending(null);
    setNote(`Paper fill on ${pending.symbol} at ${formatPx(pending.price)}. No live order left this desk.`);
  }

  function requestFlatten(symbol: string) {
    const pos = positions[symbol];
    const price = marks[symbol];
    if (!pos || price == null) return;
    log({
      ts: Date.now(),
      goal: `Flatten ${symbol}`,
      action: "ask_human",
      reason: "flatten is account-class — require approval",
      mode,
      jev_used: enabled,
      enforced: mode === "active" && enabled,
    });
    setPending({
      symbol,
      qty: -pos.qty,
      price,
      stop: null,
      target: null,
      label: "Flatten",
      blocked: null,
    });
    setPane("decision");
  }

  const spec = marketById(selected);
  const ready = now != null;
  const visibleCard = !route ? null : card?.symbol === selected ? card : (cache[selected] ?? null);

  return (
    <div className="desk-one h-full min-h-0 flex-1 overflow-auto bg-bg text-fg">
      <header className="border-b border-line bg-bg">
        <div className="mx-auto flex max-w-6xl flex-wrap items-center gap-3 px-4 py-3">
          <div className="mr-auto flex items-center gap-3">
            <span className="grid size-9 place-items-center rounded-md bg-signal text-sm font-semibold text-bg">
              01
            </span>
            <div>
              <p className="text-base font-semibold leading-none">Desk One</p>
              <p className="mt-1 text-xs text-muted">Paper tape · Jev revalue</p>
            </div>
          </div>
          <div className={cn("text-right", mono)}>
            <p className="text-xs text-muted">Equity</p>
            <p className="text-sm">{formatMoney(equity)}</p>
          </div>
          <div className={cn("text-right", mono)}>
            <p className="text-xs text-muted">Session</p>
            <p className={cn("text-sm", openPnl + realized >= 0 ? "text-signal" : "text-risk")}>
              {formatMoney(openPnl + realized)}
            </p>
          </div>
          <div className="flex rounded-md border border-line p-1">
            <button
              type="button"
              aria-pressed={mode === "shadow"}
              className={cn(
                "h-11 rounded px-3 text-sm",
                mode === "shadow" ? "bg-signal text-bg" : "text-muted",
              )}
              onClick={() => setMode("shadow")}
            >
              Shadow
            </button>
            <button
              type="button"
              aria-pressed={mode === "active"}
              className={cn(
                "h-11 rounded px-3 text-sm",
                mode === "active" ? "bg-signal text-bg" : "text-muted",
              )}
              onClick={() => setMode("active")}
            >
              Active
            </button>
          </div>
          <button
            type="button"
            aria-pressed={enabled}
            className={cn(
              "h-11 rounded-md border px-3 text-sm",
              enabled ? "border-line bg-raised text-fg" : "border-risk bg-raised text-risk",
            )}
            onClick={() => setEnabled(!enabled)}
          >
            {enabled ? "Router on" : "Router off"}
          </button>
        </div>
      </header>

      <div className="mx-auto max-w-6xl px-4 py-4">
        <p className="mb-4 max-w-3xl text-sm text-muted">
          When the action allows a card, Jev revalues the name and keeps matching headlines on it.
          A new side has to clear 0.62 confidence to replace the last one. Hard rules can still veto.
          Fills are paper, and only after you confirm.
        </p>

        <div className="mb-4 grid grid-cols-3 gap-2 lg:hidden">
          {(
            [
              ["tape", "Tape"],
              ["decision", "Decision"],
              ["router", "Router"],
            ] as const
          ).map(([id, label]) => (
            <button
              key={id}
              type="button"
              aria-pressed={pane === id}
              className={cn(
                "h-11 rounded-md border text-sm",
                pane === id ? "border-signal bg-signal text-bg" : "border-line bg-surface text-fg",
              )}
              onClick={() => setPane(id)}
            >
              {label}
            </button>
          ))}
        </div>

        <div className="grid items-start gap-4 lg:grid-cols-[16rem_minmax(0,1fr)_20rem]">
          <section className={cn(pane === "tape" ? "block" : "hidden", "lg:block")}>
            <Tape
              selected={selected}
              marks={marks}
              now={clockNow}
              onSelect={(id) => {
                select(id);
                const cached = useDesk.getState().cache[id];
                if (cached) setCard(cached);
                setPane("decision");
              }}
            />
          </section>

          <section className={cn(pane === "decision" ? "block" : "hidden", "min-w-0 lg:block")}>
            <Decision
              spec={spec}
              now={clockNow}
              route={route}
              plan={plan}
              card={visibleCard}
              scan={scan}
              note={note}
              pending={pending}
              ready={ready}
              onFocus={(id) => {
                select(id);
                const found = scan?.find((item) => item.symbol === id) ?? cache[id] ?? null;
                if (found) setCard(found);
              }}
              onPropose={() => {
                run(presetState("order"));
                setPane("decision");
              }}
              onConfirm={confirmPending}
              onDismiss={() => setPending(null)}
            />
          </section>

          <section className={cn(pane === "router" ? "block" : "hidden", "lg:block")}>
            <RouterPane
              goal={goal}
              setGoal={setGoal}
              route={route}
              plan={plan}
              ready={ready}
              enabled={enabled}
              mode={mode}
              errors={errors[selected] ?? 0}
              symbol={selected}
              payloadOpen={payloadOpen}
              setPayloadOpen={setPayloadOpen}
              onPreset={(id) => {
                run(presetState(id));
                setPane("decision");
              }}
              onRoute={() => {
                run({ goal, kind: "research", symbol: selected });
                setPane("decision");
              }}
              onBump={() => bumpError(selected)}
              onClear={() => clearError(selected)}
            />
          </section>
        </div>

        <Blotter
          positions={positions}
          marks={marks}
          fills={fills}
          logs={logs}
          confirmReset={confirmReset}
          onFlatten={requestFlatten}
          onResetAsk={() => setConfirmReset(true)}
          onReset={() => {
            resetBook();
            setConfirmReset(false);
            setPending(null);
          }}
          onResetCancel={() => setConfirmReset(false)}
        />
        <Wiring />
      </div>
    </div>
  );
}

function rank(card: DecisionCard): number {
  const gate =
    card.verdict === "ALLOW" ? 1 : card.verdict === "ESCALATE" ? 0.55 : card.verdict === "STAND ASIDE" ? 0.3 : 0.12;
  return Math.abs(card.higher - 0.5) * gate;
}

function Tape({
  selected,
  marks,
  now,
  onSelect,
}: {
  selected: string;
  marks: Record<string, number>;
  now: number;
  onSelect: (id: string) => void;
}) {
  return (
    <div className="rounded-lg border border-line bg-surface">
      <div className="flex items-baseline justify-between border-b border-line px-3 py-3">
        <h2 className={kicker}>Tape</h2>
        <span className="text-xs text-muted">6 names</span>
      </div>
      <ul>
        {MARKETS.map((spec) => {
          const px = marks[spec.id] ?? spec.base;
          const prev = priceAt(spec, now - 86_400_000);
          const change = px / prev - 1;
          const active = spec.id === selected;
          return (
            <li key={spec.id} className="border-b border-line last:border-b-0">
              <button
                type="button"
                onClick={() => onSelect(spec.id)}
                className={cn(
                  "flex min-h-11 w-full items-center gap-3 px-3 py-2 text-left",
                  active ? "bg-raised" : "bg-surface",
                )}
              >
                <span className={cn("h-6 w-1 rounded", active ? "bg-signal" : "bg-line")} />
                <span className="min-w-0 flex-1">
                  <span className="block text-sm font-medium">{spec.id === "0700" ? "0700.HK" : spec.id}</span>
                  <span className="block truncate text-xs text-muted">{spec.name}</span>
                </span>
                <span className="text-right">
                  <span className={cn("block text-sm", mono)}>{formatPx(px)}</span>
                  <span className={cn("block text-xs", mono, change >= 0 ? "text-signal" : "text-risk")}>
                    {change >= 0 ? "+" : ""}
                    {(change * 100).toFixed(2)}%
                  </span>
                </span>
              </button>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

function Decision({
  spec,
  now,
  route,
  plan,
  card,
  scan,
  note,
  pending,
  ready,
  onFocus,
  onPropose,
  onConfirm,
  onDismiss,
}: {
  spec: MarketSpec;
  now: number;
  route: RouteResult | null;
  plan: WorkPlan | null;
  card: DecisionCard | null;
  scan: DecisionCard[] | null;
  note: string | null;
  pending: Pending | null;
  ready: boolean;
  onFocus: (id: string) => void;
  onPropose: () => void;
  onConfirm: () => void;
  onDismiss: () => void;
}) {
  const path = series(spec, now);
  const up = path[path.length - 1].p >= path[0].p;

  return (
    <div className="rounded-lg border border-line bg-surface">
      <div className="border-b border-line px-4 py-3">
        <div className="flex flex-wrap items-end justify-between gap-2">
          <div>
            <p className={kicker}>{spec.venue === "hk" ? "Hong Kong" : spec.venue === "equity" ? "Equity" : "Perp"}</p>
            <h2 className="mt-1 text-xl font-medium">
              {spec.id === "0700" ? "0700.HK" : spec.id}
              <span className="ml-2 text-sm font-normal text-muted">{spec.name}</span>
            </h2>
          </div>
          <p className={cn("text-lg", mono)}>{formatPx(path[path.length - 1].p)}</p>
        </div>
        <div className="mt-3 h-16 w-full">
          {ready ? (
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={path}>
                <YAxis hide domain={["dataMin", "dataMax"]} />
                <Line
                  type="monotone"
                  dataKey="p"
                  stroke={up ? "var(--color-signal)" : "var(--color-risk)"}
                  strokeWidth={1.75}
                  dot={false}
                  isAnimationActive={false}
                />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-16 rounded bg-raised" />
          )}
        </div>
      </div>

      <div className="px-4 py-4">
        {!route ? (
          <Empty />
        ) : (
          <>
            <p role="status" className="rounded-md border border-line bg-bg px-3 py-3 text-sm">
              <Banner route={route} plan={plan} />
            </p>
            {note ? <p className="mt-3 text-sm text-muted">{note}</p> : null}
            {scan && scan.length > 0 ? (
              <ol className="mt-4 divide-y divide-line rounded-md border border-line">
                {scan.map((item, index) => (
                  <li key={item.symbol}>
                    <button
                      type="button"
                      onClick={() => onFocus(item.symbol)}
                      className={cn(
                        "flex min-h-11 w-full items-center gap-3 px-3 py-2 text-left text-sm",
                        item.symbol === spec.id ? "bg-raised" : "bg-surface",
                      )}
                    >
                      <span className={cn("w-4 text-xs text-muted", mono)}>{index + 1}</span>
                      <span className="w-14 font-medium">{item.symbol === "0700" ? "0700" : item.symbol}</span>
                      <span className={cn("w-14", mono, tone(item.side))}>{item.side}</span>
                      <span className={cn("w-12", mono)}>{item.higher.toFixed(2)}</span>
                      <span className={cn("ml-auto text-xs", verdictTone(item.verdict))}>{item.verdict}</span>
                    </button>
                  </li>
                ))}
              </ol>
            ) : null}
            {card ? <CardView card={card} /> : null}
            <div className="mt-4 flex flex-wrap gap-2">
              <button
                type="button"
                className="h-11 rounded-md bg-signal px-4 text-sm font-medium text-bg disabled:opacity-40"
                disabled={!ready || !card || card.verdict !== "ALLOW"}
                onClick={onPropose}
              >
                Stage paper order
              </button>
              <span className="self-center text-xs text-muted">Stages an account-class route. It does not fill.</span>
            </div>
            {pending ? (
              <div className="mt-4 rounded-md border border-line bg-bg p-3">
                <p className={kicker}>{pending.label}</p>
                {pending.blocked ? (
                  <p className="mt-2 text-sm text-risk">{pending.blocked}</p>
                ) : (
                  <p className={cn("mt-2 text-sm", mono)}>
                    {pending.symbol} {formatQty(pending.qty)} @ {formatPx(pending.price)}
                    {pending.stop != null ? ` · stop ${formatPx(pending.stop)}` : ""}
                    {pending.target != null ? ` · target ${formatPx(pending.target)}` : ""}
                  </p>
                )}
                <div className="mt-3 flex flex-wrap gap-2">
                  <button
                    type="button"
                    className="h-11 rounded-md bg-fg px-4 text-sm font-medium text-bg disabled:opacity-40"
                    disabled={Boolean(pending.blocked)}
                    onClick={onConfirm}
                  >
                    Confirm paper fill
                  </button>
                  <button
                    type="button"
                    className="h-11 rounded-md border border-line px-4 text-sm"
                    onClick={onDismiss}
                  >
                    Dismiss
                  </button>
                </div>
              </div>
            ) : null}
          </>
        )}
      </div>
    </div>
  );
}

function Empty() {
  const steps = [
    ["1", "Pick a name", "The tape is a seeded paper book, not a brokerage."],
    ["2", "Route the task", "grok-bot-jev classifies it before any research spend."],
    ["3", "Read the typed card", "Noul, Choice, Score. Then jev-guard can veto."],
    ["4", "Confirm or walk", "Active mode honors the action. A fill still needs you."],
  ];
  return (
    <div>
      <h3 className="text-lg font-medium">How the two repos sit on this desk</h3>
      <ol className="mt-4 grid gap-3 sm:grid-cols-2">
        {steps.map(([n, title, body]) => (
          <li key={n} className="rounded-md border border-line bg-bg p-3">
            <p className={cn("text-xs text-signal", mono)}>{n}</p>
            <p className="mt-1 text-sm font-medium">{title}</p>
            <p className="mt-1 text-sm text-muted">{body}</p>
          </li>
        ))}
      </ol>
      <p className="mt-4 text-sm text-muted">
        Start with 30-day bias, then switch the header to Active and try PEPE — the Choice can be long while
        the spread rule still vetoes.
      </p>
    </div>
  );
}

function Banner({ route, plan }: { route: RouteResult; plan: WorkPlan | null }) {
  if (!route.jev_used) {
    return (
      <>
        Router skipped ({route.reason}). Account-class actions still wait for a confirm.{" "}
        {plan ? `Path: ${plan.kind}.` : null}
      </>
    );
  }
  if (route.mode === "shadow") {
    return (
      <>
        Shadow advised <span className={cn(mono)}>{route.action}</span>. {route.reason}. The desk did not
        enforce it
        {plan ? ` and took the ${plan.kind} path instead.` : "."}
      </>
    );
  }
  return (
    <>
      Active honored <span className={cn(mono)}>{route.action}</span>. {route.reason}.
    </>
  );
}

function CardView({ card }: { card: DecisionCard }) {
  const headline =
    card.verdict === "VETO"
      ? "Veto"
      : card.verdict === "ESCALATE"
        ? "Escalate"
        : card.verdict === "STAND ASIDE"
          ? "Flat"
          : card.side === "long"
            ? "Long"
            : "Short";
  const headlineClass =
    card.verdict === "VETO" || card.verdict === "ESCALATE"
      ? "text-risk"
      : card.verdict === "STAND ASIDE"
        ? "text-muted"
        : card.side === "long"
          ? "text-signal"
          : "text-risk";

  return (
    <div className="mt-4">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <p className={kicker}>Composed verdict</p>
          <p className={cn("text-4xl font-medium", headlineClass)}>{headline}</p>
          <p className="mt-1 text-sm text-muted">
            {card.verdict === "ALLOW"
              ? `Choice ${card.side}. Rules did not veto.`
              : card.verdict === "VETO"
                ? `Jev Choice was ${card.side}. A hard rule blocked the ticket.`
                : card.verdict === "ESCALATE"
                  ? "Choice confidence is under 0.55. Do not size it."
                  : "Choice is flat. No ticket."}
          </p>
        </div>
        <p className={cn("text-xs", verdictTone(card.verdict))}>{card.verdict}</p>
      </div>

      <ul className="mt-4 divide-y divide-line border-y border-line">
        {card.questions.map((q) => {
          const width =
            q.type === "Choice" ? Math.round((q.confidence ?? 0) * 100) : Math.round(Number(q.answer) * 100);
          return (
            <li key={q.question} className="grid gap-2 py-3 sm:grid-cols-[minmax(0,1fr)_7rem] sm:items-center">
              <div className="min-w-0">
                <p className="text-sm">{q.question}</p>
                <p className="text-xs text-muted">{q.type}</p>
              </div>
              <div>
                <p className={cn("text-sm", mono)}>
                  {q.answer}
                  {q.confidence != null ? <span className="text-muted"> · {q.confidence.toFixed(2)}</span> : null}
                </p>
                <div className="mt-1 h-1.5 rounded-full bg-bg">
                  <div className="h-1.5 rounded-full bg-signal" style={{ width: `${Math.max(2, width)}%` }} />
                </div>
              </div>
            </li>
          );
        })}
      </ul>

      <dl className="mt-4 grid grid-cols-3 gap-2">
        <Level label="Entry" value={card.entry} />
        <Level label="Stop" value={card.stop} />
        <Level label="Target" value={card.target} />
      </dl>

      <div className="mt-4 grid gap-3 sm:grid-cols-2">
        <div className="rounded-md border border-line bg-bg p-3">
          <p className={kicker}>Jev risk</p>
          <p className="mt-1 text-sm font-medium capitalize">{card.risk}</p>
          <p className={cn("mt-1 text-xs text-muted", mono)}>freeze {card.freezeProb.toFixed(2)}</p>
        </div>
        <div className="rounded-md border border-line bg-bg p-3">
          <p className={kicker}>Hard rules</p>
          {card.vetoes.length === 0 ? (
            <p className="mt-1 text-sm">No veto.</p>
          ) : (
            <ul className="mt-1 space-y-1">
              {card.vetoes.map((veto) => (
                <li key={veto} className="text-sm text-risk">
                  {veto}
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>

      {card.evidence.length > 0 ? (
        <div className="mt-4">
          <p className={kicker}>
            Evidence {card.evidenceKept} kept · {card.evidenceDeduped} after dedupe
          </p>
          <ul className="mt-2 space-y-1">
            {card.evidence.map((line) => (
              <li key={line} className="text-sm text-muted">
                {line}
              </li>
            ))}
          </ul>
        </div>
      ) : null}
      <p className="mt-3 text-xs text-muted">{card.pattern}</p>
    </div>
  );
}

function Level({ label, value }: { label: string; value: number | null }) {
  return (
    <div className="rounded-md bg-bg px-3 py-2">
      <dt className="text-xs text-muted">{label}</dt>
      <dd className={cn("text-sm", mono)}>{value == null ? "—" : formatPx(value)}</dd>
    </div>
  );
}

function RouterPane({
  goal,
  setGoal,
  route,
  plan,
  ready,
  enabled,
  mode,
  errors,
  symbol,
  payloadOpen,
  setPayloadOpen,
  onPreset,
  onRoute,
  onBump,
  onClear,
}: {
  goal: string;
  setGoal: (goal: string) => void;
  route: RouteResult | null;
  plan: WorkPlan | null;
  ready: boolean;
  enabled: boolean;
  mode: "shadow" | "active";
  errors: number;
  symbol: string;
  payloadOpen: boolean;
  setPayloadOpen: (open: boolean) => void;
  onPreset: (id: PresetId) => void;
  onRoute: () => void;
  onBump: () => void;
  onClear: () => void;
}) {
  const details = route?.details;
  return (
    <div className="rounded-lg border border-line bg-surface p-3">
      <h2 className={kicker}>Usage router</h2>
      <p className="mt-2 text-xs text-muted">
        {enabled ? `Mode ${mode}.` : "Kill switch is off."} Thresholds match config.example.yaml: reuse ≥ 0.65,
        choice ≥ 0.55, subagent ≥ 0.75, sources ≤ 5, same-error limit 1.
      </p>
      <div className="mt-3 flex flex-wrap gap-2">
        {PRESET_IDS.map((id) => (
          <button
            key={id}
            type="button"
            disabled={!ready}
            className="h-11 rounded-md border border-line bg-raised px-3 text-sm disabled:opacity-40"
            onClick={() => onPreset(id)}
          >
            {PRESET_LABEL[id]}
          </button>
        ))}
      </div>
      <label className="mt-3 block text-xs text-muted" htmlFor="goal">
        Goal
      </label>
      <textarea
        id="goal"
        value={goal}
        onChange={(event) => setGoal(event.target.value)}
        rows={3}
        className="mt-1 w-full resize-none rounded-md border border-line bg-bg px-3 py-2 text-sm text-fg"
      />
      <button
        type="button"
        disabled={!ready || goal.trim().length === 0}
        className="mt-2 h-11 w-full rounded-md bg-signal text-sm font-medium text-bg disabled:opacity-40"
        onClick={onRoute}
      >
        Route goal
      </button>
      <p className="mt-2 text-xs text-muted">Type "bypass jev" or "no jev" in the goal to trip the bypass marker.</p>

      <div className="mt-4 rounded-md border border-line bg-bg p-3">
        <p className={kicker}>Failed attempts · {symbol}</p>
        <p className={cn("mt-1 text-lg", mono)}>{errors}</p>
        <div className="mt-2 flex gap-2">
          <button type="button" className="h-11 flex-1 rounded-md border border-line text-sm" onClick={onBump}>
            Log failure
          </button>
          <button type="button" className="h-11 flex-1 rounded-md border border-line text-sm" onClick={onClear}>
            Clear
          </button>
        </div>
        <p className="mt-2 text-xs text-muted">Then press Retry entry. Count ≥ 1 stops the same approach in active mode.</p>
      </div>

      {route ? (
        <div className="mt-4">
          <p className={kicker}>Last action</p>
          <p className="mt-1 text-lg font-medium">{route.action}</p>
          <p className="text-sm text-muted">{route.reason}</p>
          {plan ? (
            <p className="mt-1 text-xs text-muted">
              {plan.enforced ? "Enforced" : "Advisory"} · work {plan.kind}
              {plan.cap > 0 ? ` · cap ${plan.cap}` : ""}
            </p>
          ) : null}
          {details?.intent ? (
            <dl className={cn("mt-3 grid grid-cols-2 gap-x-3 gap-y-1 text-xs", mono)}>
              <dt className="text-muted">intent</dt>
              <dd>
                {details.intent} {details.intent_confidence?.toFixed(2)}
              </dd>
              <dt className="text-muted">reuse</dt>
              <dd>{details.reuse_cache?.toFixed(2)}</dd>
              <dt className="text-muted">stop</dt>
              <dd>{details.stop_retry?.toFixed(2)}</dd>
              <dt className="text-muted">subagent</dt>
              <dd>{details.needs_subagent?.toFixed(2)}</dd>
              <dt className="text-muted">complexity</dt>
              <dd>{details.complexity_0_1?.toFixed(2)}</dd>
            </dl>
          ) : (
            <p className="mt-2 text-xs text-muted">No typed answers — the call was skipped.</p>
          )}
          <button
            type="button"
            className="mt-3 h-11 text-sm text-muted"
            onClick={() => setPayloadOpen(!payloadOpen)}
          >
            {payloadOpen ? "Hide payload" : "Show payload"}
          </button>
          {payloadOpen ? (
            <pre className="mt-2 overflow-x-auto rounded-md bg-bg p-3 text-xs text-muted">
              {JSON.stringify(route, null, 2)}
            </pre>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}

function Blotter({
  positions,
  marks,
  fills,
  logs,
  confirmReset,
  onFlatten,
  onResetAsk,
  onReset,
  onResetCancel,
}: {
  positions: Record<string, { qty: number; avg: number }>;
  marks: Record<string, number>;
  fills: { id: string; ts: number; symbol: string; qty: number; price: number }[];
  logs: { id: string; ts: number; goal: string; action: string; mode: string; enforced: boolean }[];
  confirmReset: boolean;
  onFlatten: (symbol: string) => void;
  onResetAsk: () => void;
  onReset: () => void;
  onResetCancel: () => void;
}) {
  const rows = Object.entries(positions);
  return (
    <section className="mt-4 rounded-lg border border-line bg-surface">
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-line px-4 py-3">
        <h2 className={kicker}>Blotter</h2>
        {confirmReset ? (
          <div className="flex gap-2">
            <button type="button" className="h-11 rounded-md bg-risk px-3 text-sm text-bg" onClick={onReset}>
              Confirm reset
            </button>
            <button type="button" className="h-11 rounded-md border border-line px-3 text-sm" onClick={onResetCancel}>
              Cancel
            </button>
          </div>
        ) : (
          <button type="button" className="h-11 rounded-md border border-line px-3 text-sm" onClick={onResetAsk}>
            Reset paper book
          </button>
        )}
      </div>
      <div className="grid gap-4 p-4 lg:grid-cols-2">
        <div className="min-w-0">
          <p className="text-sm font-medium">Positions</p>
          {rows.length === 0 ? (
            <p className="mt-2 text-sm text-muted">No open lines.</p>
          ) : (
            <div className="mt-2 overflow-x-auto">
              <table className="w-full text-left text-sm">
                <thead className="text-xs text-muted">
                  <tr>
                    <th className="py-1 pr-3 font-medium">Name</th>
                    <th className="py-1 pr-3 font-medium">Qty</th>
                    <th className="py-1 pr-3 font-medium">Avg</th>
                    <th className="py-1 pr-3 font-medium">uPnL</th>
                    <th className="py-1 font-medium" />
                  </tr>
                </thead>
                <tbody>
                  {rows.map(([symbol, pos]) => {
                    const mark = marks[symbol] ?? pos.avg;
                    const upnl = (mark - pos.avg) * pos.qty;
                    return (
                      <tr key={symbol} className="border-t border-line">
                        <td className="py-2 pr-3">{symbol}</td>
                        <td className={cn("py-2 pr-3", mono)}>{formatQty(pos.qty)}</td>
                        <td className={cn("py-2 pr-3", mono)}>{formatPx(pos.avg)}</td>
                        <td className={cn("py-2 pr-3", mono, upnl >= 0 ? "text-signal" : "text-risk")}>
                          {formatMoney(upnl)}
                        </td>
                        <td className="py-2">
                          <button type="button" className="h-11 text-sm text-muted" onClick={() => onFlatten(symbol)}>
                            Flatten
                          </button>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
          {fills.length > 0 ? (
            <ul className="mt-3 space-y-1">
              {fills.slice(0, 5).map((fill) => (
                <li key={fill.id} className={cn("text-xs text-muted", mono)}>
                  {clock(fill.ts)} {fill.symbol} {formatQty(fill.qty)} @ {formatPx(fill.price)}
                </li>
              ))}
            </ul>
          ) : null}
        </div>
        <div className="min-w-0">
          <p className="text-sm font-medium">Router log</p>
          {logs.length === 0 ? (
            <p className="mt-2 text-sm text-muted">Routes land here. Shadow rows are advisory.</p>
          ) : (
            <ul className="mt-2 max-h-48 space-y-2 overflow-y-auto">
              {logs.map((line) => (
                <li key={line.id} className="border-t border-line pt-2 text-sm">
                  <span className={cn("text-xs text-muted", mono)}>{clock(line.ts)}</span>{" "}
                  <span className="font-medium">{line.action}</span>
                  <span className="text-muted"> · {line.mode}{line.enforced ? " · enforced" : ""}</span>
                  <p className="truncate text-xs text-muted">{line.goal}</p>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </section>
  );
}

function Wiring() {
  const steps = [
    {
      title: "Skill, not a hidden hook",
      body: "grok-bot-jev is not middleware. The desk wakes, builds a short task state, and calls the router. There is no pre-wake interceptor.",
    },
    {
      title: "Kill switch first",
      body: "Router off, or the words “bypass jev” / “no jev”, skips the model and returns proceed_full. Secrets never go in the state.",
    },
    {
      title: "system_one questions",
      body: "Choice of intent, Noul for cache reuse, stop-retry, and subagent, Score for complexity. Thresholds are the published defaults.",
    },
    {
      title: "Honor only in active",
      body: "Shadow logs the action and keeps going. Active must honor reuse_cache, stop_retry, research_capped, ask_human, and the rest.",
    },
    {
      title: "Finance loop second",
      body: "Only if the action allows work. That is the awesome-jev tape: 30-day Noul, round Choice, conviction Score, entry/stop/target card.",
    },
    {
      title: "Hold the last side",
      body: "A different side replaces the card only above 0.62 confidence, or at 0.42 and below. The band in between escalates and keeps the prior side.",
    },
    {
      title: "Evidence budget",
      body: "Duplicate lines drop, weak lines drop under 0.50, and the router cap still limits how many lines the card may show.",
    },
    {
      title: "Headlines, not orders",
      body: "Matching news titles ride along on the revalue. They are evidence on the card. They do not place a trade.",
    },
    {
      title: "Rules, then a human",
      body: "jev-guard shape: the model proposes a risk level, hard rules can veto, and a paper fill still waits for confirm. Nothing here is live.",
    },
  ];
  return (
    <section className="mt-4 pb-8">
      <h2 className="text-base font-medium">Wiring</h2>
      <div className="mt-3 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {steps.map((step, index) => (
          <article key={step.title} className="rounded-lg border border-line bg-surface p-3">
            <p className={cn("text-xs text-signal", mono)}>0{index + 1}</p>
            <h3 className="mt-1 text-sm font-medium">{step.title}</h3>
            <p className="mt-1 text-sm text-muted">{step.body}</p>
          </article>
        ))}
      </div>
      <p className="mt-3 text-sm text-muted">
        The card calls the fast revaluator when that route is up, and keeps the local tape when it is not.
        Side changes inside the confidence band stay on the previous card.
      </p>
      <ul className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-sm">
        <li>
          <a className="text-signal" href="https://github.com/Bodila51/grok-bot-jev" target="_blank" rel="noreferrer">
            grok-bot-jev
          </a>
        </li>
        <li>
          <a className="text-signal" href="https://github.com/yibie/awesome-jev" target="_blank" rel="noreferrer">
            awesome-jev
          </a>
        </li>
        <li>
          <a
            className="text-signal"
            href="https://github.com/yibie/awesome-jev/blob/main/categories/finance-trading.md"
            target="_blank"
            rel="noreferrer"
          >
            Finance & trading list
          </a>
        </li>
      </ul>
    </section>
  );
}

function tone(side: string): string {
  if (side === "long") return "text-signal";
  if (side === "short") return "text-risk";
  return "text-muted";
}

function verdictTone(verdict: string): string {
  if (verdict === "ALLOW") return "text-signal";
  if (verdict === "VETO" || verdict === "ESCALATE") return "text-risk";
  return "text-muted";
}

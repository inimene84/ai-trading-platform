import { create } from "zustand";
import { persist } from "zustand/middleware";
import { applyFill, type DecisionCard, type Mode, type Position } from "./engine";

export type Fill = {
  id: string;
  ts: number;
  symbol: string;
  qty: number;
  price: number;
  stop: number | null;
  target: number | null;
};

export type LogLine = {
  id: string;
  ts: number;
  goal: string;
  action: string;
  reason: string;
  mode: string;
  jev_used: boolean;
  enforced: boolean;
};

type DeskState = {
  mode: Mode;
  enabled: boolean;
  realized: number;
  positions: Record<string, Position>;
  fills: Fill[];
  logs: LogLine[];
  cache: Record<string, DecisionCard>;
  errors: Record<string, number>;
  selected: string;
  setMode: (mode: Mode) => void;
  setEnabled: (enabled: boolean) => void;
  select: (id: string) => void;
  remember: (card: DecisionCard) => void;
  log: (line: Omit<LogLine, "id">) => void;
  bumpError: (symbol: string) => void;
  clearError: (symbol: string) => void;
  commitFill: (fill: Omit<Fill, "id">) => void;
  resetBook: () => void;
};

let seq = 1;
function nid(): string {
  seq += 1;
  return `d${seq.toString(36)}`;
}

export const useDesk = create<DeskState>()(
  persist(
    (set) => ({
      mode: "shadow",
      enabled: true,
      realized: 0,
      positions: {},
      fills: [],
      logs: [],
      cache: {},
      errors: {},
      selected: "BTC",
      setMode: (mode) => set({ mode }),
      setEnabled: (enabled) => set({ enabled }),
      select: (selected) => set({ selected }),
      remember: (card) => set((s) => ({ cache: { ...s.cache, [card.symbol]: card } })),
      log: (line) => set((s) => ({ logs: [{ ...line, id: nid() }, ...s.logs].slice(0, 40) })),
      bumpError: (symbol) =>
        set((s) => ({ errors: { ...s.errors, [symbol]: (s.errors[symbol] ?? 0) + 1 } })),
      clearError: (symbol) =>
        set((s) => {
          const errors = { ...s.errors };
          delete errors[symbol];
          return { errors };
        }),
      commitFill: (fill) =>
        set((s) => {
          const applied = applyFill(s.positions, { symbol: fill.symbol, qty: fill.qty, price: fill.price });
          return {
            positions: applied.positions,
            realized: s.realized + applied.realized,
            fills: [{ ...fill, id: nid() }, ...s.fills].slice(0, 40),
          };
        }),
      resetBook: () => set({ realized: 0, positions: {}, fills: [], errors: {} }),
    }),
    {
      name: "desk-one",
      skipHydration: true,
    },
  ),
);

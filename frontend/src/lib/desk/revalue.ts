import { apiService } from "../../services/apiService";
import {
  financeDecision,
  priceAt,
  type DecisionCard,
  type MarketSpec,
  type Verdict,
} from "./engine";

export type RevalueCard = DecisionCard & {
  status: string;
  source: "jev" | "mock" | "unavailable" | "local";
  reason?: string;
};

export type RevalueResponse = {
  advisory: boolean;
  sizing_allowed: boolean;
  influence_book: boolean;
  called_jev: boolean;
  cards: RevalueCard[];
  elapsed_ms: number;
};

type Book = { lossFrac: number; grossFrac: number };

export async function revalueTape(symbols: string[], book: Book): Promise<RevalueResponse> {
  return apiService.post<RevalueResponse>("/jev/revalue", {
    symbols,
    loss_frac: book.lossFrac,
    gross_frac: book.grossFrac,
  });
}

export function anchorToTape(
  remote: RevalueCard,
  spec: MarketSpec,
  now: number,
  book: Book,
  cap: number,
): DecisionCard {
  const local = financeDecision(spec, now, cap, book);
  const mark = priceAt(spec, now);
  const basis = remote.entry && remote.entry > 0 ? remote.entry : mark;
  const scale = basis > 0 ? mark / basis : 1;
  const hard = local.vetoes.filter(
    (line) =>
      line.includes("hard cap") ||
      line.includes("Session loss") ||
      line.includes("Gross exposure") ||
      line.includes("freeze"),
  );
  const vetoes = [...remote.vetoes];
  for (const line of hard) {
    if (!vetoes.includes(line)) vetoes.push(line);
  }
  let verdict: Verdict = remote.verdict;
  if (vetoes.length > 0) verdict = "VETO";
  else if (remote.side === "flat") verdict = "STAND ASIDE";
  const evidence = remote.evidence.slice(0, Math.max(0, cap));
  return {
    ...remote,
    at: now,
    entry: remote.side === "flat" ? null : mark,
    stop: remote.stop == null ? null : remote.stop * scale,
    target: remote.target == null ? null : remote.target * scale,
    vetoes,
    verdict,
    evidence,
    evidenceKept: evidence.length,
    source: "jev",
    status: "ok",
  };
}

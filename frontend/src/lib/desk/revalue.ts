import { apiService } from "../../services/apiService";
import { newsDataService } from "../../services/newsDataService";
import {
  financeDecision,
  priceAt,
  selectEvidence,
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

export type HeadlineHint = { symbol: string; title: string };

const SYMBOL_WORDS: Record<string, string[]> = {
  BTC: ["bitcoin", "btc"],
  ETH: ["ethereum", "ether", "eth"],
  SOL: ["solana", "sol"],
  NVDA: ["nvidia", "nvda"],
  "0700": ["tencent", "0700"],
  PEPE: ["pepe"],
};

function mentions(text: string, word: string): boolean {
  const escaped = word.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  return new RegExp(`\\b${escaped}\\b`, "i").test(text);
}

export async function headlinesFor(symbols: string[]): Promise<HeadlineHint[]> {
  try {
    const feed = await newsDataService.getNewsFeed();
    const out: HeadlineHint[] = [];
    for (const symbol of symbols) {
      const words = SYMBOL_WORDS[symbol] ?? [symbol.toLowerCase()];
      let kept = 0;
      for (const item of feed.items) {
        const hay = `${item.title} ${item.summary}`;
        if (!words.some((word) => mentions(hay, word))) continue;
        out.push({ symbol, title: item.title.slice(0, 180) });
        kept += 1;
        if (kept >= 3) break;
      }
    }
    return out.slice(0, 24);
  } catch {
    return [];
  }
}

export async function revalueTape(
  symbols: string[],
  book: Book,
  headlines: HeadlineHint[] = [],
): Promise<RevalueResponse> {
  return apiService.post<RevalueResponse>("/jev/revalue", {
    symbols,
    loss_frac: book.lossFrac,
    gross_frac: book.grossFrac,
    headlines,
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
  const evidence = selectEvidence(remote.evidence, cap);
  return {
    ...remote,
    at: now,
    entry: remote.side === "flat" ? null : mark,
    stop: remote.stop == null ? null : remote.stop * scale,
    target: remote.target == null ? null : remote.target * scale,
    vetoes,
    verdict,
    evidence: evidence.kept,
    evidenceKept: evidence.kept.length,
    evidenceDeduped: remote.evidence.length,
    source: "jev",
    status: "ok",
  };
}

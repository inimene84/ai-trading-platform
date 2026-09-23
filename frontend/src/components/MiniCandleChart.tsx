import React, { useEffect, useRef, useState, useCallback } from 'react';
import { fetchBinance } from '../services/binanceProxy';

interface Candle {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface MiniCandleChartProps {
  key?: React.Key;
  symbol: string;
  displayName: string;
  signal?: 'BUY' | 'SELL' | 'HOLD' | null;
  isSelected?: boolean;
  onSelect?: (symbol: string) => void;
  onQuickTrade?: (symbol: string, side: 'buy' | 'sell') => void;
}

const CHART_H = 96;
const PRICE_H = 74;
const CANDLE_COUNT = 48;
const SLOT = 10;

function cn(...classes: (string | boolean | undefined | null)[]): string {
  return classes.filter(Boolean).join(' ');
}

export function MiniCandleChart({
  symbol,
  displayName,
  signal,
  isSelected,
  onSelect,
  onQuickTrade,
}: MiniCandleChartProps) {
  const [candles, setCandles] = useState<Candle[]>([]);
  const [currentPrice, setCurrentPrice] = useState<number | null>(null);
  const [change24h, setChange24h] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const isForex = symbol.endsWith('=X');
  const binanceSymbol = symbol.replace('/', '');

  const fetchData = useCallback(async () => {
    try {
      if (isForex) {
        // Fetch forex from backend
        const res = await fetch(`/api/backend/trading/price?symbol=${encodeURIComponent(symbol)}`);
        if (!res.ok) throw new Error('Forex fetch failed');
        const data = await res.json();
        const p = data.price ?? data.current_price ?? data.last ?? null;
        if (p !== null) setCurrentPrice(Number(p));
        setChange24h(data.change_pct ?? null);
        // Generate synthetic candles from price if no OHLCV available
        if (data.candles && Array.isArray(data.candles)) {
          setCandles(data.candles.slice(-CANDLE_COUNT).map((c: any) => ({
            time: c.time ?? c[0],
            open: Number(c.open ?? c[1]),
            high: Number(c.high ?? c[2]),
            low: Number(c.low ?? c[3]),
            close: Number(c.close ?? c[4]),
            volume: Number(c.volume ?? c[5] ?? 0),
          })));
        } else {
          // A quote is not OHLC history. Leave the chart empty rather than
          // drawing random candles that look like real market movement.
          setCandles([]);
        }
        setError(false);
      } else {
        // Fetch crypto candles from Binance
        const [klinesRes, tickerRes] = await Promise.all([
          fetchBinance(
            `https://api.binance.com/api/v3/klines?symbol=${binanceSymbol}&interval=1m&limit=${CANDLE_COUNT}`
          ),
          fetchBinance(
            `https://api.binance.com/api/v3/ticker/24hr?symbol=${binanceSymbol}`
          ),
        ]);

        if (!klinesRes.ok || !tickerRes.ok) throw new Error('Binance fetch failed');

        const klines = await klinesRes.json();
        const ticker = await tickerRes.json();

        const parsedCandles: Candle[] = klines.map((k: any[]) => ({
          time: k[0],
          open: parseFloat(k[1]),
          high: parseFloat(k[2]),
          low: parseFloat(k[3]),
          close: parseFloat(k[4]),
          volume: parseFloat(k[5]),
        }));

        setCandles(parsedCandles);
        setCurrentPrice(parseFloat(ticker.lastPrice));
        setChange24h(parseFloat(ticker.priceChangePercent));
        setError(false);
      }
    } catch (e) {
      console.warn(`[MiniCandleChart] ${symbol} fetch error:`, e);
      setError(true);
    } finally {
      setLoading(false);
    }
  }, [symbol, isForex, binanceSymbol]);

  useEffect(() => {
    fetchData();
    timerRef.current = setInterval(fetchData, 30000);
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [fetchData]);

  // ── SVG Candle Rendering ──────────────────────────────────────────────────
  function renderCandles() {
    if (candles.length === 0) return null;
    const prices = candles.flatMap((c) => [c.high, c.low]);
    const minP = Math.min(...prices);
    const maxP = Math.max(...prices);
    const range = maxP - minP || 1;
    const pad = 4;
    const toY = (p: number) => pad + (PRICE_H - pad * 2) * (1 - (p - minP) / range);
    const maxVolume = Math.max(...candles.map((c) => c.volume), 1);
    const volumeTop = PRICE_H + 4;
    const volumeHeight = CHART_H - volumeTop - 2;
    const last = candles[candles.length - 1];
    const lastY = last ? toY(last.close) : null;

    return (
      <g>
        {lastY !== null && (
          <line
            x1={0}
            y1={lastY}
            x2={candles.length * SLOT}
            y2={lastY}
            stroke="#334155"
            strokeWidth={1}
            strokeDasharray="2 2"
            vectorEffect="non-scaling-stroke"
          />
        )}
        {candles.map((c, i) => {
          const x = i * SLOT;
          const bodyW = 6;
          const cx = x + SLOT / 2;
          const side = c.close > c.open ? 'up' : c.close < c.open ? 'down' : 'flat';
          const color = side === 'up' ? '#089981' : side === 'down' ? '#f23645' : '#94a3b8';
          const bodyTop = toY(Math.max(c.open, c.close));
          const bodyBot = toY(Math.min(c.open, c.close));
          const bodyH = Math.max(1.2, bodyBot - bodyTop);
          const volH = Math.max(1, (c.volume / maxVolume) * volumeHeight);
          return (
            <g key={`${c.time}-${i}`}>
              <line
                x1={cx}
                y1={toY(c.high)}
                x2={cx}
                y2={toY(c.low)}
                stroke={color}
                strokeWidth={1}
                vectorEffect="non-scaling-stroke"
              />
              <rect
                x={x + (SLOT - bodyW) / 2}
                y={bodyTop}
                width={bodyW}
                height={bodyH}
                fill={color}
                stroke={color}
                strokeWidth={1}
                vectorEffect="non-scaling-stroke"
              />
              <rect
                x={x + 2}
                y={volumeTop + (volumeHeight - volH)}
                width={6}
                height={volH}
                fill={color}
                opacity={0.45}
              />
            </g>
          );
        })}
      </g>
    );
  }

  // ── Signal badge colors ───────────────────────────────────────────────────
  const signalColors = {
    BUY: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/40',
    SELL: 'bg-rose-500/20 text-rose-400 border-rose-500/40',
    HOLD: 'bg-amber-500/20 text-amber-400 border-amber-500/40',
  };

  const isPositive = (change24h ?? 0) >= 0;

  // ── Format price ─────────────────────────────────────────────────────────
  const formatPrice = (p: number) => {
    if (p >= 1000) return p.toLocaleString('en-US', { maximumFractionDigits: 2 });
    if (p >= 1) return p.toFixed(4);
    return p.toFixed(6);
  };

  return (
    <div
      onClick={() => onSelect?.(symbol)}
      className={cn(
        'relative flex flex-col rounded-xl border cursor-pointer transition-all duration-200 overflow-hidden group',
        'bg-[#141416] hover:bg-[#1a1a1e]',
        isSelected
          ? 'border-emerald-500/60 shadow-emerald-500/10 shadow-md'
          : 'border-zinc-800 hover:border-zinc-700'
      )}
    >
      {/* Header */}
      <div className="flex items-center justify-between px-3 pt-2.5 pb-1">
        <div className="flex items-center gap-1.5">
          <span className="text-[11px] font-bold text-white tracking-wide">
            {displayName}
          </span>
          {signal && (
            <span
              className={cn(
                'text-[9px] font-bold uppercase tracking-wider px-1.5 py-0.5 rounded-full border',
                signalColors[signal]
              )}
            >
              {signal}
            </span>
          )}
        </div>
        {change24h !== null && (
          <span
            className={cn(
              'text-[10px] font-bold font-mono',
              isPositive ? 'text-emerald-400' : 'text-rose-400'
            )}
          >
            {isPositive ? '+' : ''}
            {change24h.toFixed(2)}%
          </span>
        )}
      </div>

      {/* SVG Chart */}
      <div className="px-2">
        {loading ? (
          <div
            className="flex items-center justify-center"
            style={{ height: CHART_H }}
          >
            <div className="w-4 h-4 border-2 border-zinc-600 border-t-emerald-500 rounded-full animate-spin" />
          </div>
        ) : error ? (
          <div
            className="flex items-center justify-center text-zinc-600 text-[10px]"
            style={{ height: CHART_H }}
          >
            No data
          </div>
        ) : (
          <svg
            className="w-full"
            height={CHART_H}
            viewBox={`0 0 ${Math.max(candles.length, 1) * SLOT} ${CHART_H}`}
            preserveAspectRatio="none"
            role="img"
            aria-label={`${displayName} candlestick chart`}
          >
            {renderCandles()}
          </svg>
        )}
      </div>

      {/* Price */}
      <div className="px-3 pb-2 flex items-center justify-between">
        <span className="text-[12px] font-mono font-bold text-white">
          {currentPrice !== null ? formatPrice(currentPrice) : '—'}
        </span>
        {/* Quick Trade Buttons */}
        <div className="flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
          <button
            onClick={(e) => {
              e.stopPropagation();
              onQuickTrade?.(symbol, 'buy');
            }}
            className="px-2 py-0.5 text-[9px] font-bold uppercase bg-emerald-500/20 text-emerald-400 hover:bg-emerald-500/40 rounded border border-emerald-500/40 transition-colors"
          >
            B
          </button>
          <button
            onClick={(e) => {
              e.stopPropagation();
              onQuickTrade?.(symbol, 'sell');
            }}
            className="px-2 py-0.5 text-[9px] font-bold uppercase bg-rose-500/20 text-rose-400 hover:bg-rose-500/40 rounded border border-rose-500/40 transition-colors"
          >
            S
          </button>
        </div>
      </div>

      {/* Selected indicator */}
      {isSelected && (
        <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-gradient-to-r from-emerald-500 to-teal-500" />
      )}
    </div>
  );
}

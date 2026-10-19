import React, { useState, useEffect, useCallback, useRef } from 'react';
import { motion } from 'motion/react';
import {
  BrainCircuit,
  RefreshCw,
  Play,
  TrendingUp,
  TrendingDown,
  Minus,
  AlertTriangle,
  Clock,
  Timer,
  Activity,
} from 'lucide-react';
import { createChart, ColorType, CandlestickSeries, LineSeries, LineStyle } from 'lightweight-charts';
import { cn } from '../lib/utils';
import { apiService } from '../services/apiService';
import type {
  ForecastResult,
  BatchForecastsResponse,
  FeedSchedulerStatus,
  FeedBar,
} from '../services/apiService';

// ── Constants ────────────────────────────────────────────────────────────────

// Every symbol here was checked on 2026-10-19 to return bars from
// /api/feed/bars: crypto = Binance USDC-M perpetuals (TRADING_SYMBOLS plus the
// most liquid USDC perps), FX = cTrader symbols the loop already pulls
// trendbars for. Forecasts run on demand (one Kronos-mini CPU call each).
const UNIVERSE_GROUPS: { label: string; symbols: string[] }[] = [
  {
    label: 'Crypto (Binance USDC perps)',
    symbols: [
      'BTCUSDC', 'ETHUSDC', 'SOLUSDC', 'BNBUSDC', 'XRPUSDC',
      'AVAXUSDC', 'LINKUSDC', 'ADAUSDC', 'DOGEUSDC', 'LTCUSDC',
      'SUIUSDC', 'BCHUSDC', 'NEARUSDC', 'AAVEUSDC', 'ARBUSDC',
    ],
  },
  {
    label: 'Forex (cTrader)',
    symbols: [
      'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD',
      'USDCHF', 'NZDUSD', 'EURGBP', 'EURJPY', 'GBPJPY',
    ],
  },
  { label: 'Metals (cTrader)', symbols: ['XAUUSD', 'XAGUSD', 'XPTUSD', 'XPDUSD'] },
  { label: 'Equities / index', symbols: ['AAPL', 'MSFT', 'NVDA', 'SPX'] },
];

const INTERVALS = ['15m', '1h', '4h'] as const;
const AUTO_REFRESH_MS = 60_000;
const HISTORY_LIMIT = 200;

// ── Helpers ──────────────────────────────────────────────────────────────────

/** Convert backend bar/forecast timestamps to lightweight-charts unix seconds. */
function toChartTime(t: string | number | null | undefined): number {
  if (t === null || t === undefined || t === '') return 0;
  if (typeof t === 'number') return t > 1e12 ? Math.floor(t / 1000) : Math.floor(t);
  if (/^\d+$/.test(t)) return toChartTime(Number(t));
  const ms = Date.parse(t);
  return Number.isNaN(ms) ? 0 : Math.floor(ms / 1000);
}

/** Bars from /api/feed/bars: cTrader rows carry `time`, Binance rows `date`. */
function barTime(b: FeedBar): number {
  return toChartTime(b.time ?? b.date ?? b.timestamp);
}

function fmtPct(v: number | null | undefined): string {
  if (v === null || v === undefined) return '—';
  return `${v >= 0 ? '+' : ''}${v.toFixed(2)}%`;
}

function fmtPrice(v: number | null | undefined): string {
  if (v === null || v === undefined) return '—';
  return v.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 4 });
}

function fmtTs(ts: string | null | undefined): string {
  if (!ts) return '—';
  const d = new Date(ts);
  return Number.isNaN(d.getTime()) ? ts : d.toLocaleString();
}

function signalPalette(signal: string | undefined) {
  const s = (signal || '').toUpperCase();
  if (s === 'BUY') return { bg: 'bg-emerald-500/15', text: 'text-emerald-400', border: 'border-emerald-500/30', Icon: TrendingUp };
  if (s === 'SELL') return { bg: 'bg-rose-500/15', text: 'text-rose-400', border: 'border-rose-500/30', Icon: TrendingDown };
  return { bg: 'bg-amber-500/15', text: 'text-amber-400', border: 'border-amber-500/30', Icon: Minus };
}

// ── Forecast Chart ───────────────────────────────────────────────────────────

function ForecastChart({
  bars,
  forecastPath,
}: {
  bars: FeedBar[];
  forecastPath: { date: string; close: number }[] | null;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<any>(null);
  const histSeriesRef = useRef<any>(null);
  const fcSeriesRef = useRef<any>(null);

  useEffect(() => {
    if (!containerRef.current) return;
    const chart = createChart(containerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: 'transparent' },
        textColor: '#71717a',
      },
      grid: {
        vertLines: { color: '#1f1f22' },
        horzLines: { color: '#1f1f22' },
      },
      width: containerRef.current.clientWidth,
      height: 320,
      timeScale: {
        borderColor: '#27272a',
        timeVisible: true,
        secondsVisible: false,
      },
      rightPriceScale: { borderColor: '#27272a' },
      crosshair: {
        vertLine: { color: '#3f3f46', labelBackgroundColor: '#18181b' },
        horzLine: { color: '#3f3f46', labelBackgroundColor: '#18181b' },
      },
    }) as any;

    histSeriesRef.current = chart.addSeries(CandlestickSeries, {
      upColor: '#10b981',
      downColor: '#f43f5e',
      borderUpColor: '#10b981',
      borderDownColor: '#f43f5e',
      wickUpColor: '#10b981',
      wickDownColor: '#f43f5e',
      title: 'Historical',
    });
    fcSeriesRef.current = chart.addSeries(LineSeries, {
      color: '#f59e0b',
      lineWidth: 2,
      lineStyle: LineStyle.Dashed,
      title: 'Forecast',
    });
    chartRef.current = chart;

    const handleResize = () => {
      if (containerRef.current && chartRef.current) {
        chartRef.current.applyOptions({ width: containerRef.current.clientWidth });
      }
    };
    window.addEventListener('resize', handleResize);
    return () => {
      window.removeEventListener('resize', handleResize);
      chart.remove();
      chartRef.current = null;
      histSeriesRef.current = null;
      fcSeriesRef.current = null;
    };
  }, []);

  useEffect(() => {
    if (!histSeriesRef.current || !fcSeriesRef.current) return;

    const histByTime = new Map<number, FeedBar>();
    for (const b of bars) {
      const t = barTime(b);
      if (t > 0 && Number(b.close) > 0) histByTime.set(t, b);
    }
    const candles = Array.from(histByTime.entries())
      .sort((a, b) => a[0] - b[0])
      .map(([time, b]) => ({
        time: time as any,
        open: Number(b.open),
        high: Number(b.high),
        low: Number(b.low),
        close: Number(b.close),
      }));
    histSeriesRef.current.setData(candles);
    const histData = candles.map(c => ({ time: c.time, value: c.close }));

    const path = forecastPath || [];
    const fcByTime = new Map<number, number>();
    for (const p of path) {
      const t = toChartTime(p.date);
      if (t > 0) fcByTime.set(t, p.close);
    }
    const fcData = Array.from(fcByTime.entries())
      .sort((a, b) => a[0] - b[0])
      .map(([time, value]) => ({ time: time as any, value }));
    // Bridge the gap: start the dashed forecast line at the last historical close
    if (fcData.length > 0 && histData.length > 0) {
      const lastHist = histData[histData.length - 1];
      if (fcData[0].time > lastHist.time) {
        fcData.unshift({ time: lastHist.time, value: lastHist.value });
      }
    }
    fcSeriesRef.current.setData(fcData);

    chartRef.current?.timeScale().fitContent();
  }, [bars, forecastPath]);

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center gap-4 text-[10px] font-bold uppercase tracking-wider">
        <span className="flex items-center gap-1.5 text-emerald-400">
          <span className="inline-block w-4 h-0.5 bg-emerald-500 rounded" /> Historical candles
        </span>
        <span className="flex items-center gap-1.5 text-amber-400">
          <span className="inline-block w-4 border-t-2 border-dashed border-amber-500" /> Kronos forecast path
        </span>
      </div>
      <div ref={containerRef} className="w-full rounded-xl overflow-hidden" />
    </div>
  );
}

// ── Main Component ───────────────────────────────────────────────────────────

export const ForecastPanel: React.FC = () => {
  const [symbol, setSymbol] = useState('BTCUSDC');
  const [interval, setInterval_] = useState<string>('1h');
  const [autoRefresh, setAutoRefresh] = useState(false);
  const [running, setRunning] = useState(false);
  const [forecast, setForecast] = useState<ForecastResult | null>(null);
  const [forecastError, setForecastError] = useState<string | null>(null);
  const [bars, setBars] = useState<FeedBar[]>([]);
  const [batch, setBatch] = useState<BatchForecastsResponse | null>(null);
  const [scheduler, setScheduler] = useState<FeedSchedulerStatus | null>(null);

  const runForecast = useCallback(async () => {
    setRunning(true);
    setForecastError(null);
    try {
      const [fc, barsRes] = await Promise.all([
        apiService.getForecast(symbol, { interval, pred_len: 10, include_path: true }),
        apiService.getFeedBars(symbol, interval, HISTORY_LIMIT).catch(() => null),
      ]);
      setForecast(fc);
      if (fc.error) setForecastError(fc.error);
      if (barsRes?.data) setBars(barsRes.data);
    } catch (err: any) {
      setForecastError(err?.message || 'Forecast request failed');
      setForecast(null);
    } finally {
      setRunning(false);
    }
  }, [symbol, interval]);

  const refreshAux = useCallback(async () => {
    const [b, s] = await Promise.all([
      apiService.getBatchForecasts().catch(() => null),
      apiService.getFeedSchedulerStatus().catch(() => null),
    ]);
    if (b) setBatch(b);
    if (s) setScheduler(s);
  }, []);

  // Initial load + re-run on symbol/interval change
  useEffect(() => {
    void runForecast();
  }, [runForecast]);

  useEffect(() => {
    void refreshAux();
    const iv = window.setInterval(() => void refreshAux(), AUTO_REFRESH_MS);
    return () => window.clearInterval(iv);
  }, [refreshAux]);

  // Optional 60s auto-refresh of the forecast itself
  useEffect(() => {
    if (!autoRefresh) return;
    const iv = window.setInterval(() => void runForecast(), AUTO_REFRESH_MS);
    return () => window.clearInterval(iv);
  }, [autoRefresh, runForecast]);

  const pal = signalPalette(forecast?.signal);
  const SignalIcon = pal.Icon;

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.99 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 1.01 }}
      className="flex-1 overflow-y-auto p-6 flex flex-col gap-6"
    >
      {/* Header */}
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <h2 className="text-2xl font-black tracking-tight text-white">Kronos Forecasts</h2>
            <span className="px-2 py-0.5 rounded-full text-[10px] font-extrabold uppercase bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
              Foundation Model
            </span>
          </div>
          <p className="text-xs text-zinc-400 mt-0.5">
            Time-series forecasts on the unified feed · signal, path overlay & scheduled batch runs
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          {/* Symbol selector */}
          <select
            value={symbol}
            onChange={(e) => setSymbol(e.target.value)}
            className="bg-zinc-900/90 border border-zinc-800 rounded-xl py-2 px-3 text-xs font-mono font-bold text-white focus:outline-none focus:border-emerald-500/50"
          >
            {UNIVERSE_GROUPS.map(g => (
              <optgroup key={g.label} label={g.label}>
                {g.symbols.map(s => (
                  <option key={s} value={s}>{s}</option>
                ))}
              </optgroup>
            ))}
          </select>

          {/* Interval selector */}
          <div className="flex items-center bg-zinc-900/80 p-1 rounded-xl border border-zinc-800">
            {INTERVALS.map(iv => (
              <button
                key={iv}
                onClick={() => setInterval_(iv)}
                className={cn(
                  'px-3 py-1.5 rounded-lg text-xs font-bold transition-all',
                  interval === iv ? 'bg-emerald-500 text-black shadow-md shadow-emerald-500/20' : 'text-zinc-400 hover:text-white',
                )}
              >
                {iv}
              </button>
            ))}
          </div>

          {/* Auto-refresh toggle */}
          <button
            onClick={() => setAutoRefresh(v => !v)}
            className={cn(
              'flex items-center gap-1.5 px-3 py-2 rounded-xl text-xs font-bold transition-all border',
              autoRefresh
                ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400'
                : 'bg-zinc-900/40 border-zinc-800 text-zinc-400 hover:text-zinc-200',
            )}
            title="Re-run forecast every 60s"
          >
            <Timer size={13} className={cn(autoRefresh && 'animate-pulse')} />
            Auto 60s
          </button>

          {/* Run button */}
          <button
            onClick={() => void runForecast()}
            disabled={running}
            className="flex items-center gap-2 px-4 py-2 bg-emerald-500 hover:bg-emerald-400 disabled:opacity-50 text-black rounded-xl text-xs font-black transition-all shadow-lg shadow-emerald-500/20"
          >
            {running ? <RefreshCw size={14} className="animate-spin" /> : <Play size={14} />}
            Run forecast
          </button>
        </div>
      </div>

      {/* Signal card + chart */}
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        {/* Kronos signal card */}
        <div className="bg-[#141416]/90 backdrop-blur-md border border-zinc-800 rounded-2xl p-5 flex flex-col gap-4 shadow-xl">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <BrainCircuit size={16} className="text-emerald-400" />
              <h3 className="text-xs font-bold text-zinc-400 uppercase tracking-wider">Kronos Signal</h3>
            </div>
            <span className="text-[10px] font-mono text-zinc-500">{symbol} · {interval}</span>
          </div>

          {forecastError && (
            <div className="flex items-start gap-2 px-3 py-2 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-400 text-xs">
              <AlertTriangle size={14} className="mt-0.5 shrink-0" />
              <span>{forecastError}</span>
            </div>
          )}

          <div className="flex items-center justify-between">
            <span className={cn(
              'inline-flex items-center gap-2 px-4 py-2 rounded-xl text-lg font-black uppercase border',
              pal.bg, pal.text, pal.border,
            )}>
              <SignalIcon size={18} />
              {forecast?.signal?.toUpperCase() || (running ? '…' : '—')}
            </span>
            <div className="text-right">
              <p className="text-[10px] text-zinc-500 uppercase font-bold">Confidence</p>
              <p className="text-xl font-mono font-black text-white">
                {forecast?.confidence != null ? `${Math.round(forecast.confidence * 100)}%` : '—'}
              </p>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-3 text-xs">
            {[
              { label: 'Predicted close', value: fmtPrice(forecast?.predicted_close), mono: true },
              { label: 'Predicted Δ', value: fmtPct(forecast?.predicted_change_pct), mono: true },
              { label: 'Cum Δ 5-step', value: fmtPct(forecast?.cum_change_5_pct), mono: true },
              { label: 'Cum Δ 10-step', value: fmtPct(forecast?.cum_change_10_pct), mono: true },
              {
                label: 'Reversal risk',
                value: forecast?.reversal_risk != null ? (forecast.reversal_risk ? 'Yes' : 'No') : '—',
                mono: true,
              },
              { label: 'Model backend', value: forecast?.model_backend || '—', mono: false },
            ].map(row => (
              <div key={row.label} className="bg-zinc-900/60 border border-zinc-800/60 rounded-xl px-3 py-2">
                <p className="text-[9px] text-zinc-500 uppercase font-bold tracking-wider">{row.label}</p>
                <p className={cn('text-sm font-bold text-zinc-100 mt-0.5', row.mono && 'font-mono')}>{row.value}</p>
              </div>
            ))}
          </div>

          <p className="text-[10px] text-zinc-600 font-mono mt-auto">
            as of {fmtTs(forecast?.as_of)}
          </p>
        </div>

        {/* Chart */}
        <div className="xl:col-span-2 bg-[#141416]/90 backdrop-blur-md border border-zinc-800 rounded-2xl p-5 shadow-xl flex flex-col gap-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Activity size={16} className="text-emerald-400" />
              <h3 className="text-xs font-bold text-zinc-400 uppercase tracking-wider">
                Price & Forecast Path
              </h3>
            </div>
            <span className="text-[10px] font-mono text-zinc-500">
              {bars.length > 0 ? `${bars.length} bars` : 'no bars'} · as of {fmtTs(forecast?.as_of)}
            </span>
          </div>
          {bars.length === 0 && !running ? (
            <div className="h-[320px] flex items-center justify-center text-zinc-600 text-xs font-mono">
              No historical bars available for {symbol} ({interval})
            </div>
          ) : (
            <ForecastChart bars={bars} forecastPath={forecast?.forecast_path ?? null} />
          )}
        </div>
      </div>

      {/* Batch forecasts table */}
      <div className="bg-[#141416]/90 backdrop-blur-md border border-zinc-800 rounded-2xl overflow-hidden shadow-xl">
        <div className="flex items-center justify-between px-5 py-3.5 border-b border-zinc-800/80 bg-zinc-900/40">
          <h3 className="text-xs font-bold text-zinc-400 uppercase tracking-wider">
            Latest Batch Forecasts
          </h3>
          <span className="text-[10px] font-mono text-zinc-500">as of {fmtTs(batch?.as_of)}</span>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-left">
            <thead>
              <tr className="text-[10px] uppercase tracking-wider text-zinc-500 border-b border-zinc-800/60">
                <th className="px-5 py-2.5 font-bold">Symbol</th>
                <th className="px-5 py-2.5 font-bold">Signal</th>
                <th className="px-5 py-2.5 font-bold">Confidence</th>
                <th className="px-5 py-2.5 font-bold">Cum Δ 5</th>
                <th className="px-5 py-2.5 font-bold">Cum Δ 10</th>
                <th className="px-5 py-2.5 font-bold">Backend</th>
                <th className="px-5 py-2.5 font-bold">As of</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-zinc-800/40 text-xs">
              {(batch?.results || []).map(r => {
                const p = signalPalette(r.signal);
                const I = p.Icon;
                return (
                  <tr key={r.symbol} className="hover:bg-white/[0.03] transition-colors">
                    <td className="px-5 py-2.5 font-mono font-bold text-zinc-100">{r.symbol}</td>
                    <td className="px-5 py-2.5">
                      <span className={cn(
                        'inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-black uppercase border',
                        p.bg, p.text, p.border,
                      )}>
                        <I size={10} />
                        {r.signal || '—'}
                      </span>
                    </td>
                    <td className="px-5 py-2.5 font-mono text-zinc-300">
                      {r.confidence != null ? `${Math.round(r.confidence * 100)}%` : '—'}
                    </td>
                    <td className={cn('px-5 py-2.5 font-mono font-bold',
                      (r.cum_change_5_pct ?? 0) >= 0 ? 'text-emerald-400' : 'text-rose-400')}>
                      {fmtPct(r.cum_change_5_pct)}
                    </td>
                    <td className={cn('px-5 py-2.5 font-mono font-bold',
                      (r.cum_change_10_pct ?? 0) >= 0 ? 'text-emerald-400' : 'text-rose-400')}>
                      {fmtPct(r.cum_change_10_pct)}
                    </td>
                    <td className="px-5 py-2.5 text-zinc-400">{r.model_backend || '—'}</td>
                    <td className="px-5 py-2.5 font-mono text-zinc-500">{fmtTs(r.as_of)}</td>
                  </tr>
                );
              })}
              {(!batch || batch.results.length === 0) && (
                <tr>
                  <td colSpan={7} className="text-center py-10 text-zinc-500 font-mono text-xs">
                    No batch forecasts yet — the scheduler's kronos_batch job populates this table.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Scheduler status strip */}
      <div className="bg-[#141416]/90 backdrop-blur-md border border-zinc-800 rounded-2xl px-5 py-4 shadow-xl">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <Clock size={14} className="text-emerald-400" />
            <h3 className="text-xs font-bold text-zinc-400 uppercase tracking-wider">Feed Scheduler</h3>
          </div>
          <span className={cn(
            'px-2 py-0.5 rounded-full text-[10px] font-extrabold uppercase border',
            scheduler?.enabled
              ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20'
              : 'bg-zinc-800 text-zinc-500 border-zinc-700',
          )}>
            {scheduler ? (scheduler.enabled ? 'Enabled' : 'Disabled') : 'Unavailable'}
          </span>
        </div>
        <div className="flex gap-3 overflow-x-auto pb-1">
          {(scheduler?.jobs || []).map(job => (
            <div
              key={job.name}
              className={cn(
                'flex-shrink-0 min-w-[210px] rounded-xl border px-3.5 py-2.5 bg-zinc-900/60',
                job.last_error
                  ? 'border-rose-500/40'
                  : job.enabled
                    ? 'border-zinc-800'
                    : 'border-zinc-800/50 opacity-60',
              )}
            >
              <div className="flex items-center justify-between gap-2">
                <span className="text-xs font-bold text-zinc-100 font-mono">{job.name}</span>
                <span className={cn(
                  'w-1.5 h-1.5 rounded-full',
                  job.last_error ? 'bg-rose-500' : job.enabled ? 'bg-emerald-400' : 'bg-zinc-600',
                )} />
              </div>
              <p className="text-[10px] font-mono text-zinc-500 mt-1">cron: {job.cron_expr}</p>
              <p className="text-[10px] font-mono text-zinc-500">
                next: {job.next_run ? fmtTs(job.next_run) : '—'}
              </p>
              {job.last_error && (
                <p className="text-[10px] text-rose-400 mt-1 truncate" title={job.last_error}>
                  {job.last_error}
                </p>
              )}
            </div>
          ))}
          {(!scheduler || scheduler.jobs.length === 0) && (
            <p className="text-xs text-zinc-600 font-mono py-2">
              Scheduler status unavailable — is the backend running with FEED_SCHEDULER_ENABLED?
            </p>
          )}
        </div>
      </div>
    </motion.div>
  );
};

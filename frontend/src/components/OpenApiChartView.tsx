import React, { useEffect, useRef, useState, useCallback } from 'react';
import {
  RefreshCw,
  Zap,
  ArrowUpRight,
  ArrowDownRight,
  BarChart2,
  Sparkles,
} from 'lucide-react';
import { apiService } from '../services/apiService';
import { TradingChart } from './TradingChart';
import { cn } from '../lib/utils';
import {
  CANDLE_REQUEST_EVENTS,
  armSpotwareLoader,
  chartCandleCount,
  dismissSpotwareLoader,
  formatSpotwareCandles,
  pushCandlesToChart,
  toLightweightBars,
  toUnixSeconds,
} from '../lib/spotwareCandles';

interface OpenApiChartViewProps {
  initialSymbol?: string;
  initialTimeframe?: string;
  onQuickTrade?: (symbol: string, direction: 'buy' | 'sell') => void;
}

const SUPPORTED_SYMBOLS = [
  { symbol: 'EURUSD', display: 'EUR/USD', category: 'Forex Major' },
  { symbol: 'GBPUSD', display: 'GBP/USD', category: 'Forex Major' },
  { symbol: 'USDJPY', display: 'USD/JPY', category: 'Forex Major' },
  { symbol: 'AUDUSD', display: 'AUD/USD', category: 'Forex Major' },
  { symbol: 'USDCAD', display: 'USD/CAD', category: 'Forex Major' },
  { symbol: 'USDCHF', display: 'USD/CHF', category: 'Forex Major' },
  { symbol: 'NZDUSD', display: 'NZD/USD', category: 'Forex Major' },
  { symbol: 'EURGBP', display: 'EUR/GBP', category: 'Forex Cross' },
  { symbol: 'EURJPY', display: 'EUR/JPY', category: 'Forex Cross' },
  { symbol: 'GBPJPY', display: 'GBP/JPY', category: 'Forex Cross' },
  { symbol: 'XAUUSD', display: 'Gold / USD', category: 'Commodities' },
  { symbol: 'BTCUSD', display: 'Bitcoin / USD', category: 'Crypto' },
  { symbol: 'ETHUSD', display: 'Ethereum / USD', category: 'Crypto' },
  { symbol: 'SOLUSD', display: 'Solana / USD', category: 'Crypto' },
];

const TIMEFRAMES = [
  { id: '1M', label: '1m' },
  { id: '5M', label: '5m' },
  { id: '15M', label: '15m' },
  { id: '30M', label: '30m' },
  { id: '1H', label: '1h' },
  { id: '4H', label: '4h' },
  { id: '1D', label: '1D' },
  { id: '1W', label: '1W' },
];

// Poll cadence for the live candle. The backend serves closed bars from its
// trendbar cache (refetched from Spotware at most every 45s) plus a forming
// bar built from the spot stream, so a short poll does not add Spotware load.
const LIVE_POLL_MS = 5000;

const LAYOUTS = [
  { id: 0, label: 'Single', icon: '1' },
  { id: 1, label: 'Split H', icon: '2H' },
  { id: 2, label: 'Split V', icon: '2V' },
  { id: 5, label: 'Quarters', icon: '4Q' },
];

const OpenApiChartView: React.FC<OpenApiChartViewProps> = ({
  initialSymbol = 'EURUSD',
  initialTimeframe = '5M',
  onQuickTrade,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartInstanceRef = useRef<any>(null);
  const liveTimerRef = useRef<number | null>(null);
  const [activeSymbol, setActiveSymbol] = useState(initialSymbol);
  const [activeTimeframe, setActiveTimeframe] = useState(initialTimeframe);
  const symbolRef = useRef(activeSymbol);
  const timeframeRef = useRef(activeTimeframe);
  symbolRef.current = activeSymbol;
  timeframeRef.current = activeTimeframe;
  const [chartEngine, setChartEngine] = useState<'spotware' | 'lightweight'>('lightweight');
  const [isLoading, setIsLoading] = useState(true);
  const [activeLayout, setActiveLayout] = useState(0);
  const [candleData, setCandleData] = useState<any[]>([]);
  const [spec, setSpec] = useState<any>(null);
  const [quickLots, setQuickLots] = useState(0.1);
  const [canvasFallback, setCanvasFallback] = useState(false);
  const [tradeStatus, setTradeStatus] = useState<string | null>(null);

  useEffect(() => {
    apiService
      .getCTraderSymbolSpec(activeSymbol)
      .then((data) => setSpec(data))
      .catch(() => {});
  }, [activeSymbol]);

  const fetchCandles = useCallback(async (silent = false) => {
    if (!silent) setIsLoading(true);
    try {
      const res: any = await apiService.getCTraderTrendbars(activeSymbol, activeTimeframe, 200);
      if (res && Array.isArray(res.bars) && res.bars.length) {
        setCandleData(toLightweightBars(res.bars));
      }
    } catch (err) {
      if (!silent) console.warn('Failed to fetch trendbars', err);
    } finally {
      if (!silent) setIsLoading(false);
    }
  }, [activeSymbol, activeTimeframe]);

  const fetchBars = useCallback(async (sym: string, tf: string, count = 1000) => {
    const res: any = await apiService.getCTraderTrendbars(sym, tf, count);
    return (res && res.bars) || [];
  }, []);

  const ingest = useCallback(async (chart: any, sym: string, tf: string) => {
    try {
      const bars = await fetchBars(sym, tf);
      if (!bars.length) return false;
      const ok = pushCandlesToChart(chart, bars, sym, tf);
      if (ok) {
        armSpotwareLoader(chart);
        dismissSpotwareLoader(chart);
      }
      if (ok && chartCandleCount(chart, sym, tf) > 0) {
        setCanvasFallback(false);
      }
      return ok;
    } catch (err) {
      console.warn('Spotware candle ingest failed', err);
      return false;
    }
  }, [fetchBars]);

  useEffect(() => {
    fetchCandles();
    // The lightweight engine (the default) used to load candles once per
    // symbol/timeframe change and never again, so the chart froze at page
    // load. Keep it current while the tab is visible.
    const iv = window.setInterval(() => {
      if (document.visibilityState === 'visible') void fetchCandles(true);
    }, LIVE_POLL_MS);
    return () => window.clearInterval(iv);
  }, [fetchCandles]);

  useEffect(() => {
    if (chartEngine !== 'spotware') return;

    let isMounted = true;
    const retryTimers: number[] = [];
    setCanvasFallback(false);

    const attachCandleHandlers = (chart: any) => {
      const handler = async (
        sym: string,
        tf: string,
        _fromTs: number,
        _toTs: number,
        count: number,
        callback: (candles: any[]) => void
      ) => {
        try {
          const bars = await fetchBars(sym || symbolRef.current, tf || timeframeRef.current, count || 200);
          const formatted = formatSpotwareCandles(
            bars,
            sym || symbolRef.current,
            tf || timeframeRef.current
          );
          if (typeof callback === 'function') callback(formatted);
          if (bars.length) {
            pushCandlesToChart(
              chart,
              bars,
              sym || symbolRef.current,
              tf || timeframeRef.current
            );
          }
        } catch (e) {
          console.warn('onCandlesRequest error', e);
        }
      };
      for (const name of CANDLE_REQUEST_EVENTS) {
        try {
          chart.addEventHandler(name, handler);
        } catch {
          /* event name may not exist on this build */
        }
      }
    };

    const startLiveRates = (chart: any, sym: string) => {
      if (liveTimerRef.current) {
        window.clearInterval(liveTimerRef.current);
        liveTimerRef.current = null;
      }
      liveTimerRef.current = window.setInterval(async () => {
        if (!isMounted || !chartInstanceRef.current) return;
        try {
          // Ask for the same count as the initial load so the backend serves
          // it from cache; count=2 forced a separate tiny Spotware request.
          const bars = await fetchBars(symbolRef.current, timeframeRef.current, 200);
          const last = bars[bars.length - 1];
          if (!last) return;
          const ts = toUnixSeconds(last.time ?? last.timestamp);
          const price = Number(last.close);
          if (ts && price && typeof chart.data?.addRate === 'function') {
            chart.data.addRate(symbolRef.current || sym, ts, price, Number(last.volume || 0));
          }
        } catch {
          /* ignore live tick failures */
        }
      }, LIVE_POLL_MS);
    };

    const initChart = () => {
      if (!containerRef.current || !(window as any).T4PChart) {
        return false;
      }

      try {
        containerRef.current.innerHTML = '';

        const options = {
          general: {
            defaultSymbol: activeSymbol,
            defaultTitle: activeSymbol,
            defaultTimeframe: activeTimeframe,
            defaultChartType: 'candles',
            defaultScale: 1,
            defaultLayout: activeLayout,
            displayChange: true,
            displayIndicatorNames: true,
            displayVolume: true,
            saveLayout: false,
            saveIndicators: true,
            saveDrawings: true,
          },
          colors: {
            background: '#090a0f',
            frame: '#1e293b',
            frameActive: '#06b6d4',
            frameFullscreen: '#f59e0b',
            frameControlled: '#ef4444',
            grid: '#141a29',
            axisLine: '#1e293b',
            axisText: '#64748b',
            crossLine: '#06b6d4',
            candleRise: '#10b981',
            candleRiseBorder: '#059669',
            candleFall: '#f43f5e',
            candleFallBorder: '#e11d48',
            candleShadow: '#64748b',
            barRise: '#10b981',
            barFall: '#f43f5e',
            line: '#06b6d4',
            areaLine: '#06b6d4',
            areaBackground: 'rgba(6, 182, 212, 0.12)',
            title: '#f1f5f9',
          },
          toolbar: {
            disable: false,
            timeframes: {
              '1M': '1 Minute',
              '5M': '5 Minutes',
              '15M': '15 Minutes',
              '30M': '30 Minutes',
              '1H': '1 Hour',
              '4H': '4 Hours',
              '1D': '1 Day',
              '1W': '1 Week',
            },
            elements: [
              'symbol',
              'timeframe',
              'type',
              'indicators',
              'drawings',
              'multiview',
              'clear',
              'screenshot',
            ],
          },
        };

        const chart = new (window as any).T4PChart(containerRef.current, options);
        chartInstanceRef.current = chart;
        if (typeof chart.setTimeframe === 'function') {
          chart.setTimeframe(activeTimeframe);
        }
        if (typeof chart.setSymbol === 'function') {
          chart.setSymbol(activeSymbol);
        }
        armSpotwareLoader(chart);
        dismissSpotwareLoader(chart);

        if (chart.data && typeof chart.data.setSymbols === 'function') {
          chart.data.setSymbols(SUPPORTED_SYMBOLS.map((s) => s.symbol));
        }

        attachCandleHandlers(chart);

        const pushNow = () => {
          if (!isMounted || chartInstanceRef.current !== chart) return;
          void ingest(chart, symbolRef.current, timeframeRef.current);
        };

        if (typeof chart.addEventHandler === 'function') {
          chart.addEventHandler('onChartReady', pushNow);
        }

        pushNow();
        for (const delay of [100, 500, 1500, 3000]) {
          retryTimers.push(window.setTimeout(pushNow, delay));
        }
        for (const delay of [50, 150, 400, 1000, 2000, 3500]) {
          retryTimers.push(
            window.setTimeout(() => {
              if (!isMounted || chartInstanceRef.current !== chart) return;
              dismissSpotwareLoader(chart);
            }, delay)
          );
        }

        retryTimers.push(
          window.setTimeout(() => {
            if (!isMounted || chartInstanceRef.current !== chart) return;
            if (chartCandleCount(chart, symbolRef.current, timeframeRef.current) === 0) {
              setCanvasFallback(true);
            }
          }, 10000)
        );

        startLiveRates(chart, symbolRef.current);
        return true;
      } catch (err) {
        console.error('Error initializing Spotware Charting Library:', err);
        if (isMounted) {
          setCanvasFallback(true);
          setChartEngine('lightweight');
        }
        return false;
      }
    };

    const start = () => {
      requestAnimationFrame(() => {
        if (!isMounted) return;
        initChart();
      });
    };

    if (!(window as any).T4PChart) {
      const script = document.createElement('script');
      script.src = '/libs/chart-api.min.js';
      script.async = true;
      script.onload = () => {
        if (!(window as any).T4PChart) {
          console.warn('T4PChart missing from chart-api.min.js — falling back to lightweight charts');
          if (isMounted) setChartEngine('lightweight');
          return;
        }
        if (isMounted) start();
      };
      script.onerror = () => {
        console.warn('Failed to load /libs/chart-api.min.js — falling back to lightweight charts');
        if (isMounted) setChartEngine('lightweight');
      };
      document.body.appendChild(script);
    } else {
      start();
    }

    return () => {
      isMounted = false;
      for (const id of retryTimers) window.clearTimeout(id);
      if (liveTimerRef.current) {
        window.clearInterval(liveTimerRef.current);
        liveTimerRef.current = null;
      }
      if (chartInstanceRef.current && typeof chartInstanceRef.current.drop === 'function') {
        try {
          chartInstanceRef.current.drop();
        } catch {
          /* already torn down */
        }
      }
      if (chartInstanceRef.current && typeof chartInstanceRef.current.destroy === 'function') {
        try {
          chartInstanceRef.current.destroy();
        } catch {
          /* already torn down */
        }
      }
      chartInstanceRef.current = null;
    };
  }, [chartEngine, activeLayout, ingest, fetchBars]);

  useEffect(() => {
    if (chartEngine !== 'spotware') return;
    const chart = chartInstanceRef.current;
    if (!chart) return;
    if (typeof chart.setSymbol === 'function') {
      chart.setSymbol(activeSymbol);
    }
    if (typeof chart.setTitle === 'function') {
      chart.setTitle(activeSymbol);
    } else if (typeof chart.setDisplayName === 'function') {
      chart.setDisplayName(activeSymbol);
    }
    if (typeof chart.setTimeframe === 'function') {
      chart.setTimeframe(activeTimeframe);
    }
    void ingest(chart, activeSymbol, activeTimeframe);
    dismissSpotwareLoader(chart);
  }, [activeSymbol, activeTimeframe, chartEngine, ingest]);

  const handleSymbolChange = (sym: string) => {
    setActiveSymbol(sym);
    if (chartInstanceRef.current && typeof chartInstanceRef.current.setSymbol === 'function') {
      chartInstanceRef.current.setSymbol(sym);
      chartInstanceRef.current.setTitle(sym);
    }
  };

  const handleTimeframeChange = (tf: string) => {
    setActiveTimeframe(tf);
    if (chartInstanceRef.current && typeof chartInstanceRef.current.setTimeframe === 'function') {
      chartInstanceRef.current.setTimeframe(tf);
    }
  };

  const handleLayoutChange = (layoutId: number) => {
    setActiveLayout(layoutId);
    if (chartInstanceRef.current && typeof chartInstanceRef.current.setLayout === 'function') {
      chartInstanceRef.current.setLayout(layoutId);
    }
  };

  const executeOrder = async (direction: 'BUY' | 'SELL') => {
    try {
      setTradeStatus(`Routing ${direction} ${quickLots} lots on ${activeSymbol}...`);
      const res: any = await apiService.placeSmartOrder({
        symbol: activeSymbol,
        direction: direction,
        quantity: quickLots,
        order_type: 'MARKET',
        broker_override: 'ctrader',
      });
      if (res && res.success) {
        setTradeStatus(`Order executed! ID: ${res.order_id || 'Filled'}`);
        setTimeout(() => setTradeStatus(null), 4000);
      } else {
        setTradeStatus(`Notice: ${res.message || res.status || 'Dispatched'}`);
        setTimeout(() => setTradeStatus(null), 4000);
      }
      if (onQuickTrade) {
        onQuickTrade(activeSymbol, direction === 'BUY' ? 'buy' : 'sell');
      }
    } catch (err: any) {
      setTradeStatus(`Error: ${err.message || 'Execution failed'}`);
      setTimeout(() => setTradeStatus(null), 4000);
    }
  };

  return (
    <div className="flex-1 flex flex-col h-full bg-[#08090d] text-slate-100 overflow-hidden select-none">
      <div className="flex items-center justify-between px-4 py-2.5 bg-[#0d0f17] border-b border-slate-800/80 shrink-0 gap-3 flex-wrap">
        <div className="flex items-center gap-3">
          <div className="relative">
            <select
              value={activeSymbol}
              onChange={(e) => handleSymbolChange(e.target.value)}
              className="bg-slate-900/90 text-cyan-400 font-mono font-bold text-sm px-3 py-1.5 rounded-lg border border-slate-700/80 focus:outline-none focus:border-cyan-500 hover:bg-slate-800 transition-colors cursor-pointer"
            >
              {SUPPORTED_SYMBOLS.map((s) => (
                <option key={s.symbol} value={s.symbol} className="bg-slate-900 text-slate-200">
                  {s.display} ({s.category})
                </option>
              ))}
            </select>
          </div>

          {spec && (
            <div className="hidden md:flex items-center gap-2 text-[11px] font-mono text-slate-400 bg-slate-900/60 px-2.5 py-1 rounded-md border border-slate-800">
              <span>Pip: <strong className="text-cyan-400">{spec.pip_size}</strong></span>
              <span className="text-slate-600">|</span>
              <span>Digits: <strong className="text-slate-200">{spec.digits}</strong></span>
              <span className="text-slate-600">|</span>
              <span>1 Lot = <strong className="text-emerald-400">100k</strong></span>
            </div>
          )}
        </div>

        <div className="flex items-center bg-slate-900/80 p-0.5 rounded-lg border border-slate-800">
          {TIMEFRAMES.map((tf) => (
            <button
              key={tf.id}
              onClick={() => handleTimeframeChange(tf.id)}
              className={cn(
                'px-2.5 py-1 text-xs font-mono font-medium rounded-md transition-all',
                activeTimeframe === tf.id
                  ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-sm shadow-cyan-500/10'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
              )}
            >
              {tf.label}
            </button>
          ))}
        </div>

        <div className="flex items-center gap-2">
          {chartEngine === 'spotware' && (
            <div className="hidden lg:flex items-center gap-1 bg-slate-900/80 p-0.5 rounded-lg border border-slate-800">
              {LAYOUTS.map((l) => (
                <button
                  key={l.id}
                  onClick={() => handleLayoutChange(l.id)}
                  title={`Layout: ${l.label}`}
                  className={cn(
                    'px-2 py-0.5 text-[10px] font-mono font-bold rounded transition-all',
                    activeLayout === l.id
                      ? 'bg-amber-500/20 text-amber-300 border border-amber-500/40'
                      : 'text-slate-500 hover:text-slate-300'
                  )}
                >
                  {l.icon}
                </button>
              ))}
            </div>
          )}

          <div className="flex items-center bg-slate-900/90 p-0.5 rounded-lg border border-slate-800 text-[11px] font-medium">
            <button
              onClick={() => {
                if (chartEngine === 'spotware' && canvasFallback) {
                  setCanvasFallback(false);
                  const chart = chartInstanceRef.current;
                  if (chart) void ingest(chart, activeSymbol, activeTimeframe);
                }
                setChartEngine('spotware');
              }}
              className={cn(
                'px-2.5 py-1 rounded transition-all flex items-center gap-1.5',
                chartEngine === 'spotware'
                  ? 'bg-cyan-500 text-slate-950 font-bold shadow'
                  : 'text-slate-400 hover:text-white'
              )}
            >
              <Sparkles size={12} />
              OpenAPI Canvas
            </button>
            <button
              onClick={() => setChartEngine('lightweight')}
              className={cn(
                'px-2.5 py-1 rounded transition-all flex items-center gap-1.5',
                chartEngine === 'lightweight'
                  ? 'bg-cyan-500 text-slate-950 font-bold shadow'
                  : 'text-slate-400 hover:text-white'
              )}
            >
              <BarChart2 size={12} />
              Lightweight
            </button>
          </div>

          <button
            onClick={() => void fetchCandles()}
            className="p-1.5 bg-slate-900 text-slate-400 hover:text-cyan-400 hover:bg-slate-800 rounded-lg border border-slate-800 transition-colors"
            title="Refresh Candles"
          >
            <RefreshCw size={14} className={isLoading ? 'animate-spin text-cyan-400' : ''} />
          </button>
        </div>
      </div>

      <div className="flex-1 relative overflow-hidden bg-[#07080c]">
        {chartEngine === 'spotware' ? (
          <>
            <div
              ref={containerRef}
              id="spotware-chart-root"
              className="w-full h-full absolute inset-0"
              style={{ width: '100%', height: '100%', minHeight: '400px' }}
            />
            {canvasFallback && (
              <div className="absolute inset-0 z-10 bg-[#07080c] p-4 flex flex-col">
                <div className="mb-2 text-[11px] font-mono text-amber-300 bg-amber-950/40 border border-amber-800/50 rounded-lg px-3 py-2">
                  OpenAPI canvas stayed blank — showing live cTrader candles in the lightweight engine. Switch back to OpenAPI Canvas to retry.
                </div>
                <div className="flex-1 bg-slate-950/60 rounded-xl border border-slate-800/80 p-2 overflow-hidden">
                  <TradingChart data={candleData} />
                </div>
              </div>
            )}
          </>
        ) : (
          <div className="w-full h-full p-4 flex flex-col">
            <div className="flex-1 bg-slate-950/60 rounded-xl border border-slate-800/80 p-2 overflow-hidden">
              <TradingChart data={candleData} />
            </div>
          </div>
        )}

        <div className="absolute bottom-4 right-4 z-20 bg-slate-900/90 backdrop-blur-md p-3 rounded-2xl border border-slate-800 shadow-2xl flex flex-col gap-2.5 w-64">
          <div className="flex items-center justify-between">
            <span className="text-[11px] font-mono font-bold text-slate-400 flex items-center gap-1">
              <Zap size={13} className="text-amber-400" />
              QUICK ORDER: <span className="text-white">{activeSymbol}</span>
            </span>
            <span className="text-[10px] font-mono text-cyan-400 bg-cyan-950/60 px-1.5 py-0.5 rounded border border-cyan-800/50">
              cTrader
            </span>
          </div>

          <div className="flex items-center gap-2">
            <label className="text-[10px] text-slate-400 font-mono">Lots:</label>
            <input
              type="number"
              step="0.01"
              min="0.01"
              max="50"
              value={quickLots}
              onChange={(e) => setQuickLots(Math.max(0.01, parseFloat(e.target.value) || 0.01))}
              className="flex-1 bg-slate-950 text-white font-mono text-xs px-2 py-1 rounded border border-slate-800 focus:outline-none focus:border-cyan-500 text-right"
            />
          </div>

          <div className="grid grid-cols-2 gap-2">
            <button
              onClick={() => executeOrder('BUY')}
              className="flex items-center justify-center gap-1 py-2 px-3 rounded-xl bg-emerald-600/20 hover:bg-emerald-600/30 text-emerald-400 border border-emerald-500/40 text-xs font-bold font-mono transition-all active:scale-95 shadow-sm shadow-emerald-500/10"
            >
              <ArrowUpRight size={14} /> BUY
            </button>
            <button
              onClick={() => executeOrder('SELL')}
              className="flex items-center justify-center gap-1 py-2 px-3 rounded-xl bg-rose-600/20 hover:bg-rose-600/30 text-rose-400 border border-rose-500/40 text-xs font-bold font-mono transition-all active:scale-95 shadow-sm shadow-rose-500/10"
            >
              <ArrowDownRight size={14} /> SELL
            </button>
          </div>

          {tradeStatus && (
            <div className="text-[10px] font-mono text-center text-cyan-300 bg-slate-950/80 py-1 px-2 rounded border border-slate-800 animate-pulse">
              {tradeStatus}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default OpenApiChartView;

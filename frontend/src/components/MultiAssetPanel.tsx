import React, { useState, useEffect, useCallback, useRef } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import {
  TrendingUp,
  TrendingDown,
  Minus,
  Edit3,
  X,
  Plus,
  ChevronLeft,
  ChevronRight,
} from 'lucide-react';
import { cn } from '../lib/utils';
import { fetchBinance } from '../services/binanceProxy';

// ── Types ────────────────────────────────────────────────────────────────────

export interface MultiAssetPanelProps {
  currentSymbol: string;
  onSelectSymbol: (symbol: string, provider: 'binance' | 'alphavantage') => void;
  onQuickTrade: (symbol: string, side: 'buy' | 'sell', price: number) => void;
}

interface AssetData {
  symbol: string;
  displayName: string;
  price: number;
  change24h: number;
  changeAbs: number;
  volume: number;
  signal?: 'BUY' | 'SELL' | 'HOLD';
  confidence?: number;
  strategy?: string;
  provider: 'binance' | 'alphavantage';
  isForex: boolean;
  priceHistory: number[];
  loading: boolean;
  error?: string;
}

// ── Constants ────────────────────────────────────────────────────────────────

const CRYPTO_PAIRS = [
  'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT',
  'ADAUSDT', 'DOGEUSDT', 'AVAXUSDT', 'DOTUSDT', 'MATICUSDT',
  'LINKUSDT', 'ATOMUSDT', 'LTCUSDT', 'UNIUSDT', 'NEARUSDT',
];

const FOREX_PAIRS = [
  'EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'EURJPY=X', 'AUDUSD=X',
  'USDCAD=X', 'USDCHF=X', 'NZDUSD=X',
];

const ALL_AVAILABLE = [...CRYPTO_PAIRS, ...FOREX_PAIRS];
const DEFAULT_WATCHLIST = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT', 'DOTUSDT'];
const STORAGE_KEY = 'multiAssetWatchlist';

// ── Helpers ──────────────────────────────────────────────────────────────────

function binanceToYahoo(sym: string): string {
  if (sym.endsWith('=X')) return sym;
  if (sym.endsWith('USDT')) return sym.replace('USDT', '-USD');
  if (sym.endsWith('BTC')) return sym.replace('BTC', '-BTC');
  return sym;
}

function getDisplayName(sym: string): string {
  if (sym.endsWith('=X')) {
    const b = sym.replace('=X', '');
    return `${b.slice(0, 3)}/${b.slice(3)}`;
  }
  if (sym.endsWith('USDT')) return `${sym.replace('USDT', '')}/USDT`;
  if (sym.endsWith('BTC')) return `${sym.replace('BTC', '')}/BTC`;
  return sym;
}

function isForexPair(sym: string): boolean {
  return sym.endsWith('=X');
}

function fmtPrice(price: number, forex: boolean): string {
  if (price === 0) return '\u2014';
  if (forex) return price.toFixed(4);
  if (price >= 1000) return price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  if (price >= 1) return price.toFixed(2);
  return price.toFixed(4);
}

function fmtVol(vol: number): string {
  if (vol >= 1e9) return `${(vol / 1e9).toFixed(1)}B`;
  if (vol >= 1e6) return `${(vol / 1e6).toFixed(1)}M`;
  if (vol >= 1e3) return `${(vol / 1e3).toFixed(1)}K`;
  return vol.toFixed(0);
}

// ── Mini Sparkline ───────────────────────────────────────────────────────────

function MiniSparkline({ data, positive }: { data: number[]; positive: boolean }) {
  if (data.length < 2) return null;
  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;
  const w = 56;
  const h = 22;
  const pad = 2;

  const pts = data.map((v, i) => {
    const x = pad + (i / (data.length - 1)) * (w - 2 * pad);
    const y = h - pad - ((v - min) / range) * (h - 2 * pad);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  });

  const lineStr = pts.join(' ');
  const fillStr = `${pad},${h - pad} ${lineStr} ${w - pad},${h - pad}`;
  const gradId = `spk${positive ? 'g' : 'r'}${Math.random().toString(36).slice(2, 6)}`;

  return (
    <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`} className="flex-shrink-0 opacity-80">
      <defs>
        <linearGradient id={gradId} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={positive ? '#10b981' : '#ef4444'} stopOpacity={0.3} />
          <stop offset="100%" stopColor={positive ? '#10b981' : '#ef4444'} stopOpacity={0} />
        </linearGradient>
      </defs>
      <polygon points={fillStr} fill={`url(#${gradId})`} />
      <polyline points={lineStr} fill="none" stroke={positive ? '#10b981' : '#ef4444'}
        strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

// ── Signal Badge ─────────────────────────────────────────────────────────────

function SignalBadge({ signal, confidence }: { signal?: string; confidence?: number }) {
  if (!signal) return null;
  const palette: Record<string, { bg: string; text: string; border: string; bar: string }> = {
    BUY:  { bg: 'bg-emerald-500/15', text: 'text-emerald-400', border: 'border-emerald-500/30', bar: 'bg-emerald-500' },
    SELL: { bg: 'bg-rose-500/15',    text: 'text-rose-400',    border: 'border-rose-500/30',    bar: 'bg-rose-500' },
    HOLD: { bg: 'bg-amber-500/15',   text: 'text-amber-400',   border: 'border-amber-500/30',   bar: 'bg-amber-500' },
  };
  const c = palette[signal] || palette.HOLD;
  const pct = confidence ? Math.round(confidence * 100) : 0;

  return (
    <div className="flex flex-col gap-1 items-end">
      <span className={cn(
        'inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[9px] font-black uppercase tracking-wider border',
        c.bg, c.text, c.border
      )}>
        {signal === 'BUY' && <TrendingUp size={8} />}
        {signal === 'SELL' && <TrendingDown size={8} />}
        {signal === 'HOLD' && <Minus size={8} />}
        {signal}
      </span>
      {confidence !== undefined && confidence > 0 && (
        <div className="w-full h-1 bg-zinc-800 rounded-full overflow-hidden">
          <div className={cn('h-full rounded-full transition-all duration-500', c.bar)}
            style={{ width: `${pct}%` }} />
        </div>
      )}
    </div>
  );
}

// ── Asset Card ───────────────────────────────────────────────────────────────

function AssetCard({
  asset, isSelected, isEditing, onSelect, onQuickTrade, onRemove,
}: {
  key?: React.Key;
  asset: AssetData;
  isSelected: boolean;
  isEditing: boolean;
  onSelect: () => void;
  onQuickTrade: (side: 'buy' | 'sell') => void;
  onRemove: () => void;
}) {
  const pos = asset.change24h >= 0;

  return (
    <motion.div
      layout
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.9 }}
      whileHover={{ y: -2 }}
      transition={{ duration: 0.2 }}
      onClick={isEditing ? undefined : onSelect}
      className={cn(
        'relative flex-shrink-0 w-[195px] rounded-2xl border p-3 cursor-pointer transition-all duration-200',
        'bg-zinc-900/60 backdrop-blur-sm hover:bg-zinc-900/80',
        isSelected
          ? 'border-emerald-500/50 shadow-[0_0_20px_rgba(16,185,129,0.15)]'
          : 'border-zinc-800/60 hover:border-zinc-700/80',
        asset.loading && 'animate-pulse',
      )}
    >
      {isEditing && (
        <button
          onClick={(e) => { e.stopPropagation(); onRemove(); }}
          className="absolute -top-2 -right-2 z-10 w-5 h-5 bg-rose-500 rounded-full flex items-center justify-center hover:bg-rose-400 transition-colors shadow-lg"
        >
          <X size={10} className="text-white" />
        </button>
      )}

      <div className="flex items-start justify-between mb-1.5">
        <div className="min-w-0 flex-1">
          <p className="text-[9px] text-zinc-500 font-bold uppercase tracking-wider truncate">
            {asset.isForex ? '\uD83C\uDF10 Forex' : '\u20BF Crypto'}
          </p>
          <p className="text-sm font-bold text-white truncate">{asset.displayName}</p>
        </div>
        <SignalBadge signal={asset.signal} confidence={asset.confidence} />
      </div>

      <div className="mb-1.5">
        <span className="text-base font-mono font-bold text-white leading-none">
          {asset.loading ? '...' : fmtPrice(asset.price, asset.isForex)}
        </span>
      </div>

      <div className="flex items-center justify-between mb-1.5">
        <div className={cn('flex items-center gap-1 text-[11px] font-bold', pos ? 'text-emerald-400' : 'text-rose-400')}>
          {pos ? <TrendingUp size={11} /> : <TrendingDown size={11} />}
          <span>{pos ? '+' : ''}{asset.change24h.toFixed(2)}%</span>
        </div>
        <MiniSparkline data={asset.priceHistory} positive={pos} />
      </div>

      {asset.volume > 0 && (
        <p className="text-[9px] text-zinc-600 font-mono mb-1.5">Vol: {fmtVol(asset.volume)}</p>
      )}

      {!isEditing && (
        <div className="flex gap-1 mt-1">
          <button
            onClick={(e) => { e.stopPropagation(); onQuickTrade('buy'); }}
            className="flex-1 flex items-center justify-center gap-1 py-1 rounded-lg bg-emerald-500/10 hover:bg-emerald-500/25 text-emerald-400 text-[10px] font-bold transition-all border border-emerald-500/20 hover:border-emerald-500/40"
          >
            <TrendingUp size={10} /> Buy
          </button>
          <button
            onClick={(e) => { e.stopPropagation(); onQuickTrade('sell'); }}
            className="flex-1 flex items-center justify-center gap-1 py-1 rounded-lg bg-rose-500/10 hover:bg-rose-500/25 text-rose-400 text-[10px] font-bold transition-all border border-rose-500/20 hover:border-rose-500/40"
          >
            <TrendingDown size={10} /> Sell
          </button>
        </div>
      )}
    </motion.div>
  );
}

// ── Main Component ───────────────────────────────────────────────────────────

export function MultiAssetPanel({ currentSymbol, onSelectSymbol, onQuickTrade }: MultiAssetPanelProps) {
  const [watchlist, setWatchlist] = useState<string[]>(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      if (saved) {
        const parsed = JSON.parse(saved);
        if (Array.isArray(parsed) && parsed.length > 0) return parsed;
      }
    } catch {}
    return DEFAULT_WATCHLIST;
  });

  const [assets, setAssets] = useState<Record<string, AssetData>>({});
  const [isEditing, setIsEditing] = useState(false);
  const [showAddDropdown, setShowAddDropdown] = useState(false);
  const [addSearch, setAddSearch] = useState('');
  const scrollRef = useRef<HTMLDivElement>(null);
  const priceHistoryRef = useRef<Record<string, number[]>>({});

  // Persist watchlist
  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(watchlist));
  }, [watchlist]);

  // ── Fetch crypto prices from Binance ────────────────────────────────────
  const fetchCryptoPrices = useCallback(async (symbols: string[]) => {
    const cryptoSymbols = symbols.filter(s => !isForexPair(s));
    if (cryptoSymbols.length === 0) return {};

    const results: Record<string, Partial<AssetData>> = {};
    try {
      const queryStr = cryptoSymbols.map(s => `"${s}"`).join(',');
      const resp = await fetchBinance(
        `https://api.binance.com/api/v3/ticker/24hr?symbols=[${queryStr}]`
      );
      if (resp.ok) {
        const data = await resp.json();
        for (const ticker of data) {
          const sym = ticker.symbol;
          const price = parseFloat(ticker.lastPrice) || 0;
          const hist = priceHistoryRef.current[sym] || [];
          hist.push(price);
          if (hist.length > 20) hist.shift();
          priceHistoryRef.current[sym] = hist;

          results[sym] = {
            price,
            change24h: parseFloat(ticker.priceChangePercent) || 0,
            changeAbs: parseFloat(ticker.priceChange) || 0,
            volume: parseFloat(ticker.quoteVolume) || 0,
            priceHistory: [...hist],
            loading: false,
          };
        }
      }
    } catch (err) {
      console.warn('Binance fetch error:', err);
      for (const sym of cryptoSymbols) {
        results[sym] = { loading: false, error: 'fetch failed' };
      }
    }
    return results;
  }, []);

  // ── Fetch forex prices from backend ────────────────────────────────────
  const fetchForexPrices = useCallback(async (symbols: string[]) => {
    const forexSymbols = symbols.filter(s => isForexPair(s));
    if (forexSymbols.length === 0) return {};

    const results: Record<string, Partial<AssetData>> = {};
    try {
      const resp = await fetch('/api/backend/trading/markets/forex');
      if (resp.ok) {
        const data = await resp.json();
        // Unified-feed shape: { as_of, quotes: [{symbol: "EURUSD", ...}] }
        // Legacy shape: { data: [{symbol: "EUR/USD", price, change24h, up}] } or bare array
        const arr: any[] = Array.isArray(data) ? data : (data.quotes || data.data || []);
        const byDisplay: Record<string, any> = {};
        for (const item of arr) {
          if (!item || typeof item.symbol !== 'string') continue;
          byDisplay[item.symbol] = item;
          if (item.symbol.length === 6 && !item.symbol.includes('/')) {
            byDisplay[`${item.symbol.slice(0, 3)}/${item.symbol.slice(3)}`] = item;
            byDisplay[`${item.symbol}=X`] = item;
          }
        }

        for (const sym of forexSymbols) {
          const base = sym.replace('=X', '');
          const displayKey = base.length === 6 ? `${base.slice(0, 3)}/${base.slice(3)}` : base;
          const rateData = byDisplay[sym] || byDisplay[displayKey] || byDisplay[base];
          if (rateData) {
            const price = rateData.price ?? 0;
            const change = rateData.change_pct ?? rateData.change24h ?? 0;
            const hist = priceHistoryRef.current[sym] || [];
            if (price > 0) {
              hist.push(price);
              if (hist.length > 20) hist.shift();
            }
            priceHistoryRef.current[sym] = hist;

            results[sym] = {
              price,
              change24h: change,
              changeAbs: 0,
              volume: 0,
              priceHistory: [...hist],
              loading: false,
            };
          } else {
            results[sym] = { loading: false, price: 0 };
          }
        }
      }
    } catch (err) {
      console.warn('Forex fetch error:', err);
      for (const sym of forexSymbols) {
        results[sym] = { loading: false, error: 'fetch failed' };
      }
    }
    return results;
  }, []);

  // ── Fetch signals from backend ─────────────────────────────────────────
  const fetchSignals = useCallback(async () => {
    const signalMap: Record<string, { signal: 'BUY' | 'SELL' | 'HOLD'; confidence: number; strategy: string }> = {};
    try {
      const resp = await fetch('/api/backend/trading/signals');
      if (resp.ok) {
        const data = await resp.json();
        const signals = data.signals || [];
        for (const s of signals) {
          const yahooSym = s.symbol;
          // Map to our format: try both Binance and Yahoo format
          const binanceSym = yahooSym.endsWith('=X') ? yahooSym :
            yahooSym.replace('-USD', 'USDT').replace('-BTC', 'BTC');
          const dir = (s.direction || '').toUpperCase();
          if (['BUY', 'SELL', 'HOLD'].includes(dir)) {
            signalMap[binanceSym] = {
              signal: dir as 'BUY' | 'SELL' | 'HOLD',
              confidence: s.confidence || 0,
              strategy: s.strategy || '',
            };
            // Also store under yahoo format
            signalMap[yahooSym] = signalMap[binanceSym];
          }
        }
      }
    } catch (err) {
      console.warn('Signals fetch error:', err);
    }
    return signalMap;
  }, []);

  // ── Master refresh ─────────────────────────────────────────────────────
  const refreshAll = useCallback(async () => {
    const [cryptoData, forexData, signalData] = await Promise.all([
      fetchCryptoPrices(watchlist),
      fetchForexPrices(watchlist),
      fetchSignals(),
    ]);

    setAssets(prev => {
      const next = { ...prev };
      for (const sym of watchlist) {
        const merged = { ...(cryptoData[sym] || {}), ...(forexData[sym] || {}) };
        const sig = signalData[sym] || signalData[binanceToYahoo(sym)];
        next[sym] = {
          // Spreads first so explicit defaults below only fill missing keys
          ...prev[sym],
          ...merged,
          // Ensure required fields always present
          symbol: sym,
          displayName: (merged as { displayName?: string }).displayName ?? prev[sym]?.displayName ?? getDisplayName(sym),
          price: (merged as { price?: number }).price ?? prev[sym]?.price ?? 0,
          change24h: (merged as { change24h?: number }).change24h ?? prev[sym]?.change24h ?? 0,
          changeAbs: (merged as { changeAbs?: number }).changeAbs ?? prev[sym]?.changeAbs ?? 0,
          volume: (merged as { volume?: number }).volume ?? prev[sym]?.volume ?? 0,
          provider: ((merged as { provider?: 'binance' | 'alphavantage' }).provider ?? prev[sym]?.provider ?? (isForexPair(sym) ? 'alphavantage' : 'binance')) as 'binance' | 'alphavantage',
          isForex: (merged as { isForex?: boolean }).isForex ?? prev[sym]?.isForex ?? isForexPair(sym),
          priceHistory: (merged as { priceHistory?: number[] }).priceHistory ?? prev[sym]?.priceHistory ?? [],
          loading: (merged as { loading?: boolean }).loading ?? prev[sym]?.loading ?? false,
          ...(sig ? { signal: sig.signal, confidence: sig.confidence, strategy: sig.strategy } : {}),
        };
      }
      // Remove assets no longer in watchlist
      for (const key of Object.keys(next)) {
        if (!watchlist.includes(key)) delete next[key];
      }
      return next;
    });
  }, [watchlist, fetchCryptoPrices, fetchForexPrices, fetchSignals]);

  // Initial load + auto-refresh every 5 seconds
  useEffect(() => {
    // Initialize assets as loading
    setAssets(prev => {
      const next = { ...prev };
      for (const sym of watchlist) {
        if (!next[sym]) {
          next[sym] = {
            symbol: sym,
            displayName: getDisplayName(sym),
            price: 0,
            change24h: 0,
            changeAbs: 0,
            volume: 0,
            provider: isForexPair(sym) ? 'alphavantage' : 'binance',
            isForex: isForexPair(sym),
            priceHistory: [],
            loading: true,
          };
        }
      }
      return next;
    });

    refreshAll();
    const interval = setInterval(refreshAll, 5000);
    return () => clearInterval(interval);
  }, [refreshAll]);

  // ── Watchlist management ───────────────────────────────────────────────
  const addSymbol = useCallback((sym: string) => {
    if (!watchlist.includes(sym)) {
      setWatchlist(prev => [...prev, sym]);
    }
    setShowAddDropdown(false);
    setAddSearch('');
  }, [watchlist]);

  const removeSymbol = useCallback((sym: string) => {
    setWatchlist(prev => prev.filter(s => s !== sym));
  }, []);

  // ── Scroll helpers ─────────────────────────────────────────────────────
  const scroll = useCallback((dir: 'left' | 'right') => {
    if (scrollRef.current) {
      const amount = 210;
      scrollRef.current.scrollBy({ left: dir === 'left' ? -amount : amount, behavior: 'smooth' });
    }
  }, []);

  // Available symbols for add dropdown (filtered out already-watched)
  const availableToAdd = ALL_AVAILABLE.filter(s => !watchlist.includes(s))
    .filter(s => addSearch === '' || s.toLowerCase().includes(addSearch.toLowerCase()) || getDisplayName(s).toLowerCase().includes(addSearch.toLowerCase()));

  // ── Render ─────────────────────────────────────────────────────────────
  return (
    <div className="bg-[#141416] border border-zinc-800 rounded-2xl p-4">
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <h3 className="text-xs font-bold text-zinc-400 uppercase tracking-wider">Watchlist</h3>
          <span className="text-[10px] text-zinc-600 font-mono">{watchlist.length} assets</span>
        </div>
        <div className="flex items-center gap-2">
          {/* Scroll buttons */}
          <button onClick={() => scroll('left')}
            className="p-1 rounded-lg bg-zinc-800/50 hover:bg-zinc-800 text-zinc-400 hover:text-white transition-colors">
            <ChevronLeft size={14} />
          </button>
          <button onClick={() => scroll('right')}
            className="p-1 rounded-lg bg-zinc-800/50 hover:bg-zinc-800 text-zinc-400 hover:text-white transition-colors">
            <ChevronRight size={14} />
          </button>
          <div className="w-px h-4 bg-zinc-800" />
          {/* Edit toggle */}
          <button
            onClick={() => { setIsEditing(!isEditing); setShowAddDropdown(false); }}
            className={cn(
              'flex items-center gap-1 px-2.5 py-1 rounded-lg text-[10px] font-bold transition-all border',
              isEditing
                ? 'bg-amber-500/15 text-amber-400 border-amber-500/30'
                : 'bg-zinc-800/50 text-zinc-400 border-zinc-800 hover:text-white hover:border-zinc-700'
            )}
          >
            <Edit3 size={11} /> {isEditing ? 'Done' : 'Edit'}
          </button>
        </div>
      </div>

      {/* Scrollable card row */}
      <div className="relative">
        <div ref={scrollRef}
          className="flex gap-3 overflow-x-auto pb-2 scrollbar-none"
          style={{ scrollbarWidth: 'none', msOverflowStyle: 'none' }}
        >
          <AnimatePresence mode="popLayout">
            {watchlist.map(sym => {
              const asset = assets[sym];
              if (!asset) return null;
              return (
                <AssetCard
                  key={sym}
                  asset={asset}
                  isSelected={currentSymbol === sym || currentSymbol === binanceToYahoo(sym)}
                  isEditing={isEditing}
                  onSelect={() => {
                    const prov = isForexPair(sym) ? 'alphavantage' : 'binance';
                    onSelectSymbol(sym, prov);
                  }}
                  onQuickTrade={(side) => onQuickTrade(sym, side, asset.price)}
                  onRemove={() => removeSymbol(sym)}
                />
              );
            })}
          </AnimatePresence>

          {/* Add card button (in edit mode) */}
          {isEditing && (
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              className="relative flex-shrink-0"
            >
              <button
                onClick={() => setShowAddDropdown(!showAddDropdown)}
                className="w-[195px] h-full min-h-[140px] rounded-2xl border-2 border-dashed border-zinc-700 hover:border-emerald-500/50 flex flex-col items-center justify-center gap-2 text-zinc-500 hover:text-emerald-400 transition-all"
              >
                <Plus size={20} />
                <span className="text-[10px] font-bold uppercase tracking-wider">Add Pair</span>
              </button>

              {/* Add dropdown */}
              {showAddDropdown && (
                <div className="absolute top-full left-0 mt-2 w-[220px] bg-zinc-900 border border-zinc-700 rounded-xl shadow-2xl z-50 overflow-hidden">
                  <div className="p-2 border-b border-zinc-800">
                    <input
                      type="text"
                      value={addSearch}
                      onChange={(e) => setAddSearch(e.target.value)}
                      placeholder="Search pairs..."
                      className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-1.5 text-xs text-white placeholder-zinc-500 focus:outline-none focus:border-emerald-500/50"
                      autoFocus
                    />
                  </div>
                  <div className="max-h-[200px] overflow-y-auto">
                    {availableToAdd.length === 0 ? (
                      <p className="p-3 text-xs text-zinc-500 text-center italic">No matches</p>
                    ) : (
                      availableToAdd.slice(0, 15).map(sym => (
                        <button
                          key={sym}
                          onClick={() => addSymbol(sym)}
                          className="w-full flex items-center justify-between px-3 py-2 text-xs hover:bg-zinc-800 transition-colors text-left"
                        >
                          <div>
                            <span className="font-bold text-white">{getDisplayName(sym)}</span>
                            <span className="ml-2 text-[9px] text-zinc-500">
                              {isForexPair(sym) ? 'Forex' : 'Crypto'}
                            </span>
                          </div>
                          <Plus size={12} className="text-emerald-400" />
                        </button>
                      ))
                    )}
                  </div>
                </div>
              )}
            </motion.div>
          )}
        </div>
      </div>
    </div>
  );
}

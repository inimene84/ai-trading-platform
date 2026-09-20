import React, { useState, useEffect, useCallback } from 'react';
import { motion } from 'motion/react';
import { PieChart, TrendingUp, BarChart3, ArrowUpRight, ArrowDownRight, RefreshCw } from 'lucide-react';
import { cn } from '../lib/utils';
import { apiService, Portfolio, PortfolioSnapshot, Position, Trade } from '../services/apiService';
import { useToast } from './Toast';
import { ResponsiveContainer, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip } from 'recharts';

export const PortfolioView = () => {
  const { showToast } = useToast();
  const [portfolio, setPortfolio] = useState<Portfolio | null>(null);
  const [positions, setPositions] = useState<Position[]>([]);
  const [snapshots, setSnapshots] = useState<PortfolioSnapshot[]>([]);
  const [trades, setTrades] = useState<Trade[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);

  const fetchAll = useCallback(async () => {
    try {
      const [pf, pos, hist, tr] = await Promise.all([
        apiService.getPortfolio(),
        apiService.getPositions(),
        apiService.getPortfolioHistory(100),
        apiService.getTrades({ limit: 100, status: 'closed' }),
      ]);
      setPortfolio(pf);
      setPositions(pos.positions || []);
      setSnapshots(hist.snapshots || []);
      setTrades(tr.trades || []);
      setError(false);
    } catch (err) {
      console.warn('Portfolio fetch failed:', err);
      setError(true);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchAll();
    const iv = setInterval(fetchAll, 15_000);
    return () => clearInterval(iv);
  }, [fetchAll]);

  // Computed stats — exchange_reconciliation is not strategy PnL (Phase C).
  const strategyTrades = trades.filter(t => t.strategy !== 'exchange_reconciliation');
  const winningTrades = strategyTrades.filter(t => t.pnl && t.pnl > 0);
  const losingTrades = strategyTrades.filter(t => t.pnl && t.pnl < 0);
  const winRate = strategyTrades.length > 0 ? (winningTrades.length / strategyTrades.length) * 100 : 0;

  // Asset allocation from positions
  const assetAllocationMap: { [key: string]: number } = {};
  positions.forEach(p => {
    const val = p.quantity * p.current_price;
    assetAllocationMap[p.symbol] = (assetAllocationMap[p.symbol] || 0) + val;
  });
  const allocationValues = Object.values(assetAllocationMap);
  const totalPosVal: number = allocationValues.length > 0 ? allocationValues.reduce((a, b) => a + b, 0) : 0;
  const allocationEntries = Object.entries(assetAllocationMap)
    .map(([sym, val]) => ({ symbol: sym, value: val, percent: totalPosVal > 0 ? (val / totalPosVal) * 100 : 0 }))
    .sort((a, b) => b.value - a.value);

  const colors = ['text-amber-500', 'text-blue-500', 'text-emerald-500', 'text-purple-500', 'text-rose-500', 'text-cyan-500'];
  const barColors = ['bg-amber-500', 'bg-blue-500', 'bg-emerald-500', 'bg-purple-500', 'bg-rose-500', 'bg-cyan-500'];

  // Equity curve data
  const equityCurve = snapshots.map(s => ({
    time: s.timestamp ? new Date(s.timestamp).toLocaleDateString() : '',
    value: s.total_value,
  }));

  if (loading) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <div className="text-center space-y-3">
          <div className="w-8 h-8 border-2 border-emerald-500 border-t-transparent rounded-full animate-spin mx-auto" />
          <p className="text-sm text-zinc-500">Loading portfolio…</p>
        </div>
      </div>
    );
  }

  if (error && !portfolio) {
    return (
      <div className="flex-1 flex items-center justify-center p-6">
        <div className="max-w-md rounded-2xl border border-red-500/30 bg-red-500/10 p-6 text-center">
          <h2 className="text-lg font-bold text-red-300">Portfolio unavailable</h2>
          <p className="mt-2 text-sm text-zinc-400">
            The backend could not be reached. No substitute balance is being displayed.
          </p>
          <button
            onClick={() => { setLoading(true); fetchAll(); }}
            className="mt-4 rounded-xl bg-zinc-800 px-4 py-2 text-xs font-bold hover:bg-zinc-700"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  const balance = portfolio?.balance ?? 0;
  const equity = portfolio?.equity ?? 0;
  const realizedToday = portfolio?.realized_today ?? 0;
  const realized7d = portfolio?.realized_7d ?? 0;
  const openUpnl = portfolio?.open_unrealized_pnl ?? portfolio?.unrealized_pnl ?? 0;
  const unionEquity = portfolio?.informational_union_equity ?? portfolio?.display_union_equity;
  const unionLabel = portfolio?.union_label ?? 'union = cTrader + Binance';
  const positionsValue = portfolio?.positions_value ?? 0;
  const available = portfolio?.available ?? balance;

  return (
    <motion.div 
      initial={{ opacity: 0, scale: 0.98 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 1.02 }}
      className="flex-1 overflow-y-auto p-6 space-y-6"
    >
      <div className="flex items-center justify-between mb-2">
        <div>
          <h2 className="text-2xl font-bold tracking-tight">Portfolio Performance</h2>
          <p className="text-sm text-zinc-400">
            {error ? 'Showing cached data (backend unreachable)' : 'Live data from trading engine'}
          </p>
        </div>
        <button
          onClick={() => { setLoading(true); fetchAll(); showToast('Portfolio refreshed', 'info'); }}
          className="flex items-center gap-2 px-4 py-2 bg-zinc-800 hover:bg-zinc-700 rounded-xl text-xs font-bold transition-colors"
        >
          <RefreshCw size={14} /> Refresh
        </button>
      </div>

      {/* Top Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-[#141416] p-6 rounded-2xl border border-zinc-800">
          <p className="text-zinc-500 text-[10px] uppercase font-bold tracking-widest mb-1">Risk Equity</p>
          <p className="text-3xl font-bold font-mono tracking-tight">${equity.toLocaleString(undefined, { minimumFractionDigits: 2 })}</p>
          <div className="mt-4 flex flex-col gap-2">
            <div className="flex justify-between items-center text-sm">
              <span className="text-zinc-500">Available</span>
              <span className="font-mono text-white">${available.toLocaleString(undefined, { minimumFractionDigits: 2 })}</span>
            </div>
            <div className="flex justify-between items-center text-sm">
              <span className="text-zinc-500">In Positions</span>
              <span className="font-mono text-white">${positionsValue.toLocaleString(undefined, { minimumFractionDigits: 2 })}</span>
            </div>
            {typeof unionEquity === 'number' && (
              <div className="flex justify-between items-center text-sm">
                <span className="text-zinc-500">{unionLabel}</span>
                <span className="font-mono text-white">${unionEquity.toLocaleString(undefined, { minimumFractionDigits: 2 })}</span>
              </div>
            )}
          </div>
        </div>
        
        <div className="bg-[#141416] p-6 rounded-2xl border border-zinc-800">
          <p className="text-zinc-500 text-[10px] uppercase font-bold tracking-widest mb-1">Realized (strategy)</p>
          <p className={cn("text-3xl font-bold font-mono tracking-tight", realizedToday >= 0 ? "text-emerald-400" : "text-rose-400")}>
            {realizedToday >= 0 ? '+' : ''}${realizedToday.toLocaleString(undefined, { minimumFractionDigits: 2 })}
          </p>
          <p className="text-[10px] text-zinc-500 mt-1">Today — not lifetime DB PnL</p>
          <div className="mt-4 flex flex-col gap-2">
            <div className="flex justify-between items-center text-sm">
              <span className="text-zinc-500">7d realized</span>
              <span className={cn("font-mono", realized7d >= 0 ? "text-emerald-400" : "text-rose-400")}>
                {realized7d >= 0 ? '+' : ''}${realized7d.toLocaleString(undefined, { minimumFractionDigits: 2 })}
              </span>
            </div>
            <div className="flex justify-between items-center text-sm">
              <span className="text-zinc-500">Open uPnL</span>
              <span className={cn("font-mono", openUpnl >= 0 ? "text-emerald-400" : "text-rose-400")}>
                {openUpnl >= 0 ? '+' : ''}${openUpnl.toLocaleString(undefined, { minimumFractionDigits: 2 })}
              </span>
            </div>
          </div>
        </div>

        <div className="bg-[#141416] p-6 rounded-2xl border border-zinc-800 flex flex-col justify-center">
          <p className="text-zinc-500 text-[10px] uppercase font-bold tracking-widest mb-4">Win/Loss Ratio</p>
          <div className="w-full flex h-4 rounded-full overflow-hidden bg-zinc-800">
            <div className="h-full bg-emerald-500 transition-all duration-700" style={{ width: `${winRate}%` }} />
            <div className="h-full bg-rose-500 transition-all duration-700" style={{ width: `${100 - winRate}%` }} />
          </div>
          <div className="flex justify-between text-xs mt-3">
            <span className="text-emerald-400 font-bold tracking-widest">{winRate.toFixed(0)}% WINS ({winningTrades.length})</span>
            <span className="text-rose-400 font-bold tracking-widest">{(100 - winRate).toFixed(0)}% LOSSES ({losingTrades.length})</span>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Equity Curve */}
        <div className="bg-[#141416] p-6 rounded-2xl border border-zinc-800 h-80 flex flex-col">
          <h3 className="font-semibold mb-4 flex items-center gap-2"><BarChart3 size={16} className="text-emerald-400" /> Equity Curve</h3>
          {equityCurve.length > 1 ? (
            <div className="flex-1">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={equityCurve}>
                  <defs>
                    <linearGradient id="eqGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#10b981" stopOpacity={0.3} />
                      <stop offset="100%" stopColor="#10b981" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" vertical={false} />
                  <XAxis dataKey="time" hide />
                  <YAxis domain={['auto', 'auto']} stroke="#4b5563" fontSize={10} tickFormatter={(v) => `$${v}`} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#111827', border: '1px solid #374151', borderRadius: '8px', fontSize: '10px' }}
                    itemStyle={{ color: '#10b981' }}
                  />
                  <Area type="monotone" dataKey="value" stroke="#10b981" strokeWidth={2} fill="url(#eqGrad)" animationDuration={1500} />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <div className="flex-1 flex items-center justify-center">
              <p className="text-sm text-zinc-500 italic">Equity data will appear after trading cycles run</p>
            </div>
          )}
        </div>

        {/* Asset Allocation */}
        <div className="bg-[#141416] p-6 rounded-2xl border border-zinc-800 h-80 flex flex-col">
          <h3 className="font-semibold mb-4 flex items-center gap-2"><PieChart size={16} className="text-emerald-400" /> Asset Allocation</h3>
          {allocationEntries.length > 0 ? (
            <div className="flex-1 flex items-center justify-center gap-8">
              <div className="relative w-40 h-40 rounded-full border-8 border-zinc-800 flex items-center justify-center">
                <div className="text-center">
                  <p className="text-2xl font-bold">{allocationEntries.length}</p>
                  <p className="text-[10px] text-zinc-500 uppercase tracking-widest">Assets</p>
                </div>
              </div>
              <div className="space-y-4 flex-1">
                {allocationEntries.slice(0, 6).map((asset, i) => (
                  <div key={asset.symbol} className="flex items-center gap-3">
                    <span className={cn("text-xs font-bold w-14 truncate", colors[i % colors.length])}>{asset.symbol}</span>
                    <div className="flex-1 h-1.5 bg-zinc-800 rounded-full overflow-hidden">
                      <div className={cn("h-full transition-all duration-700", barColors[i % barColors.length])} style={{ width: `${asset.percent}%` }} />
                    </div>
                    <span className="text-[10px] text-white font-mono w-10 text-right">{asset.percent.toFixed(0)}%</span>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <div className="flex-1 flex items-center justify-center">
              <p className="text-sm text-zinc-500 italic">No open positions to show allocation</p>
            </div>
          )}
        </div>
      </div>
    </motion.div>
  );
};

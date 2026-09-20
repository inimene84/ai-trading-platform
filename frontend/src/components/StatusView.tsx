import React, { useState, useEffect, useCallback } from 'react';
import { motion } from 'motion/react';
import {
  Cpu, Globe, Shield, Activity, Zap, Play, Pause, RefreshCw,
  CheckCircle2, XCircle, AlertCircle, Server, Wifi, WifiOff,
  BarChart3, MessageSquare, Database, Cloud, Bot, Link2
} from 'lucide-react';
import { cn } from '../lib/utils';
import { apiService, SystemStatus, LoopStatus } from '../services/apiService';
import { marketDataService } from '../services/marketDataService';
import { useToast } from './Toast';

const StatusDot = ({ status }: { status: string }) => {
  const s = status.toLowerCase();
  if (s === 'configured' || s === 'public' || s === 'online' || s === 'running') {
    return <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse shadow-[0_0_6px_rgba(16,185,129,0.6)]" />;
  }
  if (s === 'error') {
    return <div className="w-2 h-2 rounded-full bg-rose-500" />;
  }
  return <div className="w-2 h-2 rounded-full bg-zinc-600" />;
};

const ProviderCard = ({ name, detail, status, type, icon: Icon }: {
  name: string; detail: string; status: string; type?: string; icon: any;
}) => (
  <div className={cn(
    "p-4 rounded-xl border transition-all",
    status === 'configured' || status === 'public'
      ? "bg-emerald-500/5 border-emerald-500/20 hover:border-emerald-500/40"
      : "bg-zinc-900/50 border-zinc-800 hover:border-zinc-700"
  )}>
    <div className="flex items-start justify-between mb-3">
      <div className={cn(
        "p-2 rounded-lg",
        status === 'configured' || status === 'public' ? "bg-emerald-500/10" : "bg-zinc-800"
      )}>
        <Icon size={16} className={cn(
          status === 'configured' || status === 'public' ? "text-emerald-400" : "text-zinc-500"
        )} />
      </div>
      <StatusDot status={status} />
    </div>
    <p className="text-sm font-bold truncate">{name}</p>
    <p className="text-[10px] text-zinc-500 font-mono truncate mt-0.5">{detail}</p>
    {type && (
      <span className={cn(
        "inline-block mt-2 px-2 py-0.5 rounded text-[8px] font-bold uppercase tracking-wider border",
        type === 'local' ? "bg-indigo-500/10 border-indigo-500/20 text-indigo-400" :
        type === 'cloud' ? "bg-blue-500/10 border-blue-500/20 text-blue-400" : "bg-zinc-800 border-zinc-700 text-zinc-500"
      )}>
        {type}
      </span>
    )}
  </div>
);

const MarketDataStatus = () => {
  const [marketStatus, setMarketStatus] = useState(marketDataService.getStatus());

  useEffect(() => {
    return marketDataService.onStatusChange((status) => setMarketStatus(status as 'connected' | 'disconnected' | 'connecting' | 'error'));
  }, []);

  const isConnected = marketStatus === 'connected';
  const isError = marketStatus === 'error';

  return (
    <div className={cn(
      "flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-bold border transition-colors",
      isConnected ? "bg-emerald-500/10 border-emerald-500/20 text-emerald-400" :
      isError ? "bg-rose-500/10 border-rose-500/20 text-rose-400" :
      "bg-zinc-500/10 border-zinc-500/20 text-zinc-400"
    )}>
      <Database size={14} />
      <span>DATA FEED: {marketStatus.toUpperCase()}</span>
    </div>
  );
};

export const StatusView = () => {
  const { showToast } = useToast();
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [loopStatus, setLoopStatus] = useState<LoopStatus | null>(null);
  const [brokersData, setBrokersData] = useState<any>(null);
  const [ctraderTokens, setCtraderTokens] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [connected, setConnected] = useState<boolean | null>(null);
  const [loopAction, setLoopAction] = useState(false);
  const [brokerAction, setBrokerAction] = useState(false);
  const [grokOverview, setGrokOverview] = useState<any>(null);
  const [grokSummary, setGrokSummary] = useState<string | null>(null);
  const [grokLoading, setGrokLoading] = useState(false);
  const [agentManifest, setAgentManifest] = useState<any>(null);

  const fetchAll = useCallback(async () => {
    try {
      const [st, ls, br, tk, manifest] = await Promise.all([
        apiService.getStatus(),
        apiService.getLoopStatus(),
        apiService.getBrokers().catch(() => null),
        apiService.getCTraderTokens().catch(() => null),
        apiService.getAgentConnectManifest().catch(() => null),
      ]);
      setStatus(st);
      setLoopStatus(ls);
      if (br) setBrokersData(br.brokers);
      if (tk) setCtraderTokens(tk);
      if (manifest) setAgentManifest(manifest);
      setConnected(true);
    } catch (err) {
      console.warn('Backend unreachable:', err);
      setConnected(false);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchAll();
    const iv = setInterval(fetchAll, 10_000);
    return () => clearInterval(iv);
  }, [fetchAll]);

  const handleToggleCTrader = async (enable: boolean) => {
    setBrokerAction(true);
    try {
      if (enable) {
        const res = await apiService.enableCTraderLive();
        showToast(res.message || 'cTrader live connection attempted', res.connected ? 'success' : 'info');
      } else {
        const res = await apiService.disableCTraderLive();
        showToast(res.message || 'cTrader paper mode active', 'info');
      }
      fetchAll();
    } catch (err: any) {
      showToast(`cTrader toggle failed: ${err.message}`, 'error');
    } finally {
      setBrokerAction(false);
    }
  };

  const handleStartLoop = async () => {
    setLoopAction(true);
    try {
      await apiService.startLoop({
        interval_minutes: 5,
        symbols: ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT', 'DOTUSDT'],
        strategy: 'combined',
      });
      showToast('Trading loop started', 'success');
      fetchAll();
    } catch (err) {
      showToast('Failed to start loop', 'error');
    } finally {
      setLoopAction(false);
    }
  };

  const handleStopLoop = async () => {
    setLoopAction(true);
    try {
      await apiService.stopLoop();
      showToast('Trading loop stopped', 'info');
      fetchAll();
    } catch {
      showToast('Failed to stop loop', 'error');
    } finally {
      setLoopAction(false);
    }
  };

  const handleGrokOverview = async () => {
    setGrokLoading(true);
    try {
      const overview = await apiService.getGrokOverseerOverview();
      setGrokOverview(overview);
      showToast('Grok overseer snapshot loaded', 'success');
    } catch (err: any) {
      showToast(`Grok overview failed: ${err.message}`, 'error');
    } finally {
      setGrokLoading(false);
    }
  };

  const handleGrokAnalyze = async () => {
    setGrokLoading(true);
    try {
      const res = await apiService.analyzeGrokOverseer('risk and operational health');
      setGrokSummary(res.summary || null);
      setGrokOverview(res.overview || grokOverview);
      showToast('Grok overseer analysis complete', 'success');
    } catch (err: any) {
      showToast(`Grok analyze failed: ${err.message}`, 'error');
    } finally {
      setGrokLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <div className="text-center space-y-3">
          <div className="w-8 h-8 border-2 border-emerald-500 border-t-transparent rounded-full animate-spin mx-auto" />
          <p className="text-sm text-zinc-500">Connecting to backend…</p>
        </div>
      </div>
    );
  }

  if (!connected || !status) {
    return (
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="flex-1 flex flex-col items-center justify-center p-12"
      >
        <div className="bg-[#141416] border border-rose-500/20 rounded-3xl p-12 text-center max-w-md space-y-4">
          <div className="w-16 h-16 bg-rose-500/10 rounded-2xl flex items-center justify-center mx-auto">
            <WifiOff size={32} className="text-rose-400" />
          </div>
          <h3 className="text-xl font-bold">Backend Offline</h3>
          <p className="text-sm text-zinc-400 leading-relaxed">
            Cannot reach the FastAPI backend. Make sure the server is running at the configured URL, then refresh.
          </p>
          <button
            onClick={() => { setLoading(true); fetchAll(); }}
            className="px-6 py-2.5 bg-zinc-800 hover:bg-zinc-700 rounded-xl text-sm font-bold transition-colors"
          >
            <RefreshCw size={14} className="inline mr-2" /> Retry Connection
          </button>
        </div>
      </motion.div>
    );
  }

  const isLoopRunning = loopStatus?.running;

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -10 }}
      className="flex-1 overflow-y-auto p-6 space-y-6"
    >
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold tracking-tight">System Status</h2>
          <p className="text-sm text-zinc-400">Infrastructure health and trading loop control</p>
        </div>
        <div className="flex items-center gap-3">
          <div className={cn(
            "flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-bold border",
            status.dry_run
              ? "bg-amber-500/10 border-amber-500/20 text-amber-400"
              : "bg-emerald-500/10 border-emerald-500/20 text-emerald-400"
          )}>
            <Shield size={14} />
            {status.dry_run ? 'PAPER MODE' : 'LIVE MODE'}
          </div>
          <div className={cn(
            "flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-bold border",
            connected
              ? "bg-emerald-500/10 border-emerald-500/20 text-emerald-400"
              : "bg-rose-500/10 border-rose-500/20 text-rose-400"
          )}>
            {connected ? <Wifi size={14} /> : <WifiOff size={14} />}
            {connected ? 'BACKEND OK' : 'BACKEND OFFLINE'}
          </div>

          <MarketDataStatus />
        </div>
      </div>

      {/* Trading Loop Control */}
      <div className={cn(
        "rounded-2xl border p-6 transition-all",
        isLoopRunning
          ? "bg-emerald-500/5 border-emerald-500/20"
          : "bg-[#141416] border-zinc-800"
      )}>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className={cn(
              "w-12 h-12 rounded-xl flex items-center justify-center",
              isLoopRunning ? "bg-emerald-500/20" : "bg-zinc-800"
            )}>
              <Activity size={24} className={cn(isLoopRunning ? "text-emerald-400" : "text-zinc-500")} />
            </div>
            <div>
              <h3 className="font-bold flex items-center gap-2">
                Automated Trading Loop
                {isLoopRunning && (
                  <span className="flex items-center gap-1 px-2 py-0.5 bg-emerald-500/20 text-emerald-400 text-[9px] font-bold rounded-full">
                    <div className="w-1.5 h-1.5 bg-emerald-400 rounded-full animate-pulse" /> ACTIVE
                  </span>
                )}
              </h3>
              <p className="text-xs text-zinc-500 mt-0.5">
                {isLoopRunning
                  ? `Running every ${loopStatus?.interval_minutes || '?'}min · ${loopStatus?.total_cycles || 0} cycles · Strategy: ${loopStatus?.strategy || 'combined'}`
                  : 'Loop is stopped. Click Start to begin automated trading.'
                }
              </p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            {isLoopRunning ? (
              <button
                onClick={handleStopLoop}
                disabled={loopAction}
                className="flex items-center gap-2 px-5 py-2.5 bg-rose-500 hover:bg-rose-400 text-white rounded-xl text-sm font-bold transition-all shadow-lg shadow-rose-500/20 disabled:opacity-50"
              >
                <Pause size={16} /> Stop Loop
              </button>
            ) : (
              <button
                onClick={handleStartLoop}
                disabled={loopAction}
                className="flex items-center gap-2 px-5 py-2.5 bg-emerald-500 hover:bg-emerald-400 text-black rounded-xl text-sm font-bold transition-all shadow-lg shadow-emerald-500/20 disabled:opacity-50"
              >
                <Play size={16} fill="currentColor" /> Start Loop
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Overview Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-[#141416] border border-zinc-800 rounded-xl p-4">
          <p className="text-[10px] uppercase text-zinc-500 font-bold tracking-widest">Strategies</p>
          <p className="text-2xl font-bold font-mono mt-1">{status.strategies_loaded}</p>
        </div>
        <div className="bg-[#141416] border border-zinc-800 rounded-xl p-4">
          <p className="text-[10px] uppercase text-zinc-500 font-bold tracking-widest">LLM Engines</p>
          <p className="text-2xl font-bold font-mono mt-1">{status.llm_providers.filter(p => p.status === 'configured').length}</p>
        </div>
        <div className="bg-[#141416] border border-zinc-800 rounded-xl p-4">
          <p className="text-[10px] uppercase text-zinc-500 font-bold tracking-widest">Active Brokers</p>
          <p className="text-2xl font-bold font-mono mt-1">{status.brokers.filter(b => b.status === 'configured').length}</p>
        </div>
        <div className="bg-[#141416] border border-zinc-800 rounded-xl p-4">
          <p className="text-[10px] uppercase text-zinc-500 font-bold tracking-widest">Data Feeds</p>
          <p className="text-2xl font-bold font-mono mt-1">{status.data_providers.filter(d => d.status === 'configured' || d.status === 'public').length}</p>
        </div>
      </div>

      {status.equity_books && (
        <div className="bg-[#141416] border border-zinc-800 rounded-xl p-4">
          <p className="text-[10px] uppercase text-zinc-500 font-bold tracking-widest">
            {status.equity_books.union_label || 'union = cTrader + Binance'}
          </p>
          <p className="text-sm text-zinc-300 mt-2">
            Active broker: <span className="font-mono font-bold">{status.active_broker || status.equity_books.active_broker || 'ctrader'}</span>
            {' · '}
            Risk book: <span className="font-mono font-bold">{String(status.equity_books.risk_broker || '—')}</span>
            {status.equity_books.split_book ? ' · split book (union is display-only, not kill equity)' : ''}
          </p>
        </div>
      )}

      {/* LLM Providers */}
      <div>
        <h3 className="text-sm font-bold mb-4 flex items-center gap-2">
          <Cpu size={16} className="text-emerald-400" /> AI / LLM Engines
        </h3>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-3">
          {status.llm_providers.length === 0 ? (
            <div className="col-span-full text-center py-8 text-zinc-500 text-sm">No LLM providers configured</div>
          ) : (
            status.llm_providers.map((p, i) => (
              <div key={i}>
                <ProviderCard name={p.name} detail={p.model} status={p.status} type={p.type} icon={Cpu} />
              </div>
            ))
          )}
        </div>
      </div>

      {/* Multi-Broker Architecture */}
      <div>
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-sm font-bold flex items-center gap-2">
            <Zap size={16} className="text-emerald-400" /> Multi-Broker Hub & Adapters
          </h3>
          <span className="text-[11px] text-zinc-500 font-mono">
            {brokersData ? 'Unified Router Active' : 'Connecting to Router...'}
          </span>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {/* Binance Futures Card */}
          <div className="bg-[#141416] border border-zinc-800 rounded-xl p-5 flex flex-col justify-between hover:border-zinc-700 transition-all">
            <div>
              <div className="flex items-start justify-between mb-3">
                <div className="flex items-center gap-3">
                  <div className="p-2.5 rounded-xl bg-amber-500/10 border border-amber-500/20">
                    <Zap size={20} className="text-amber-400" />
                  </div>
                  <div>
                    <h4 className="font-bold text-sm text-white">Binance Futures</h4>
                    <p className="text-xs text-zinc-400">Crypto Perpetuals (BTC, ETH, SOL, etc.)</p>
                  </div>
                </div>
                <span className={cn(
                  "px-2.5 py-1 rounded-full text-[10px] font-bold uppercase tracking-wider border",
                  brokersData?.binance_futures?.status?.connected
                    ? "bg-emerald-500/10 border-emerald-500/20 text-emerald-400"
                    : "bg-zinc-800 border-zinc-700 text-zinc-400"
                )}>
                  {brokersData?.binance_futures?.status?.connected ? 'Connected' : 'Offline / Standby'}
                </span>
              </div>

              <div className="grid grid-cols-3 gap-2 mt-4 pt-3 border-t border-zinc-800/80 text-xs">
                <div>
                  <span className="text-[10px] text-zinc-500 font-bold block uppercase">Balance</span>
                  <span className="font-mono font-bold text-zinc-200">
                    ${(brokersData?.binance_futures?.status?.balance ?? 0).toLocaleString(undefined, { minimumFractionDigits: 2 })}
                  </span>
                </div>
                <div>
                  <span className="text-[10px] text-zinc-500 font-bold block uppercase">Open Pos</span>
                  <span className="font-mono font-bold text-zinc-200">
                    {brokersData?.binance_futures?.status?.open_positions ?? 0}
                  </span>
                </div>
                <div>
                  <span className="text-[10px] text-zinc-500 font-bold block uppercase">Circuit Breaker</span>
                  <span className={cn(
                    "font-bold text-[11px]",
                    brokersData?.binance_futures?.circuit_breaker?.available ? "text-emerald-400" : "text-rose-400"
                  )}>
                    {brokersData?.binance_futures?.circuit_breaker?.available ? 'HEALTHY' : 'TRIPPED'}
                  </span>
                </div>
              </div>
            </div>

            <div className="mt-4 pt-3 border-t border-zinc-800/80 flex items-center justify-between text-xs text-zinc-500">
              <span>Mode: {brokersData?.binance_futures?.status?.dry_run ? 'Dry-Run / Paper' : 'Live Gateway'}</span>
              <span className="text-zinc-600 font-mono text-[10px]">Direct REST + WS</span>
            </div>
          </div>

          {/* cTrader Open API Card */}
          <div className="bg-[#141416] border border-zinc-800 rounded-xl p-5 flex flex-col justify-between hover:border-zinc-700 transition-all">
            <div>
              <div className="flex items-start justify-between mb-3">
                <div className="flex items-center gap-3">
                  <div className="p-2.5 rounded-xl bg-sky-500/10 border border-sky-500/20">
                    <Globe size={20} className="text-sky-400" />
                  </div>
                  <div>
                    <h4 className="font-bold text-sm text-white">cTrader Open API</h4>
                    <p className="text-xs text-zinc-400">Forex, Metals & Multi-Asset (IC Trading)</p>
                  </div>
                </div>
                <span className={cn(
                  "px-2.5 py-1 rounded-full text-[10px] font-bold uppercase tracking-wider border",
                  brokersData?.ctrader?.status?.connected
                    ? "bg-emerald-500/10 border-emerald-500/20 text-emerald-400"
                    : "bg-sky-500/10 border-sky-500/20 text-sky-400"
                )}>
                  {brokersData?.ctrader?.status?.dry_run ? 'Paper Mode' : 'Live Gateway'}
                </span>
              </div>

              <div className="grid grid-cols-3 gap-2 mt-4 pt-3 border-t border-zinc-800/80 text-xs">
                <div>
                  <span className="text-[10px] text-zinc-500 font-bold block uppercase">Balance</span>
                  <span className="font-mono font-bold text-zinc-200">
                    ${(brokersData?.ctrader?.status?.balance ?? 50000).toLocaleString(undefined, { minimumFractionDigits: 2 })}
                  </span>
                </div>
                <div>
                  <span className="text-[10px] text-zinc-500 font-bold block uppercase">Token Expiry</span>
                  <span className="font-mono font-bold text-zinc-200">
                    {ctraderTokens?.days_until_expiration ? `${ctraderTokens.days_until_expiration}d` : '30d (Valid)'}
                  </span>
                </div>
                <div>
                  <span className="text-[10px] text-zinc-500 font-bold block uppercase">Circuit Breaker</span>
                  <span className={cn(
                    "font-bold text-[11px]",
                    brokersData?.ctrader?.circuit_breaker?.available ? "text-emerald-400" : "text-rose-400"
                  )}>
                    {brokersData?.ctrader?.circuit_breaker?.available ? 'HEALTHY' : 'TRIPPED'}
                  </span>
                </div>
              </div>
            </div>

            <div className="mt-4 pt-3 border-t border-zinc-800/80 flex items-center justify-between">
              <span className="text-xs text-zinc-500">
                Host: {brokersData?.ctrader?.status?.host || 'demo.ctraderapi.com:5035'}
              </span>
              <div className="flex items-center gap-2">
                {brokersData?.ctrader?.status?.dry_run ? (
                  <button
                    onClick={() => handleToggleCTrader(true)}
                    disabled={brokerAction}
                    className="px-3 py-1 bg-sky-500 hover:bg-sky-400 text-black rounded-lg text-xs font-bold transition-all disabled:opacity-50"
                  >
                    Connect Live
                  </button>
                ) : (
                  <button
                    onClick={() => handleToggleCTrader(false)}
                    disabled={brokerAction}
                    className="px-3 py-1 bg-zinc-800 hover:bg-zinc-700 text-zinc-300 rounded-lg text-xs font-bold transition-all border border-zinc-700 disabled:opacity-50"
                  >
                    Switch to Paper
                  </button>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Data Providers */}
      <div>
        <h3 className="text-sm font-bold mb-4 flex items-center gap-2">
          <Database size={16} className="text-emerald-400" /> Market Data Providers
        </h3>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
          {status.data_providers.map((d, i) => (
            <div key={i}>
              <ProviderCard
                name={d.name}
                detail={d.note || (d.status === 'configured' ? 'API key configured' : 'Not configured')}
                status={d.status}
                icon={Activity}
              />
            </div>
          ))}
        </div>
      </div>

      {/* Risk & Monitoring */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Risk Configuration */}
        <div className="bg-[#141416] border border-zinc-800 rounded-2xl p-6">
          <h3 className="text-sm font-bold mb-4 flex items-center gap-2">
            <Shield size={16} className="text-emerald-400" /> Risk Configuration
          </h3>
          <div className="space-y-3">
            {[
              { label: 'Risk Per Trade', value: `${(status.risk_config.risk_per_trade * 100).toFixed(1)}%` },
              { label: 'Max Open Positions', value: status.risk_config.max_positions },
              { label: 'Min Signal Strength', value: `${(status.risk_config.min_signal_strength * 100).toFixed(0)}%` },
              { label: 'Min Risk/Reward', value: `1:${status.risk_config.min_risk_reward}` },
              { label: 'Kelly Criterion', value: status.risk_config.use_kelly ? 'Active' : 'Off' },
              { label: 'VIX Threshold', value: status.risk_config.vix_threshold },
            ].map(item => (
              <div key={item.label} className="flex items-center justify-between py-2 border-b border-zinc-800/50 last:border-none">
                <span className="text-xs text-zinc-500">{item.label}</span>
                <span className="text-xs font-mono font-bold">{item.value}</span>
              </div>
            ))}
            <div className="pt-2">
              <p className="text-[10px] uppercase text-zinc-500 font-bold tracking-wider mb-2">Signal Weights</p>
              <div className="flex gap-2">
                {Object.entries(status.risk_config.weights).map(([k, v]) => (
                  <div key={k} className="flex-1 bg-zinc-900/50 px-3 py-2 rounded-lg text-center">
                    <p className="text-[9px] text-zinc-500 uppercase">{k}</p>
                    <p className="text-xs font-mono font-bold">{(Number(v) * 100).toFixed(0)}%</p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Monitoring */}
        <div className="bg-[#141416] border border-zinc-800 rounded-2xl p-6">
          <h3 className="text-sm font-bold mb-4 flex items-center gap-2">
            <BarChart3 size={16} className="text-emerald-400" /> Monitoring & Automation
          </h3>
          <div className="space-y-4">
            {[
              { name: 'Telegram Alerts', active: status.telegram, icon: MessageSquare },
              { name: 'InfluxDB Metrics', active: status.influxdb, icon: Database },
              { name: 'n8n Automation', active: status.n8n, icon: Cloud },
            ].map(item => (
              <div key={item.name} className={cn(
                "flex items-center justify-between p-4 rounded-xl border transition-all",
                item.active
                  ? "bg-emerald-500/5 border-emerald-500/20"
                  : "bg-zinc-900/30 border-zinc-800"
              )}>
                <div className="flex items-center gap-3">
                  <div className={cn(
                    "p-2 rounded-lg",
                    item.active ? "bg-emerald-500/10" : "bg-zinc-800"
                  )}>
                    <item.icon size={16} className={cn(item.active ? "text-emerald-400" : "text-zinc-600")} />
                  </div>
                  <span className="text-sm font-medium">{item.name}</span>
                </div>
                {item.active ? (
                  <span className="flex items-center gap-1.5 text-emerald-400 text-xs font-bold">
                    <CheckCircle2 size={14} /> Active
                  </span>
                ) : (
                  <span className="flex items-center gap-1.5 text-zinc-500 text-xs font-bold">
                    <XCircle size={14} /> Not Configured
                  </span>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* GrokBOT Overseer & Agent API */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-[#141416] border border-zinc-800 rounded-2xl p-6">
          <h3 className="text-sm font-bold mb-4 flex items-center gap-2">
            <Bot size={16} className="text-violet-400" /> GrokBOT Overseer
          </h3>
          <p className="text-xs text-zinc-500 mb-4">
            Supervisor snapshot for overseer jobs. Endpoints: <code className="text-zinc-400">GET /api/agents/grok-overseer/overview</code>
          </p>
          <div className="flex flex-wrap gap-2 mb-4">
            <button
              onClick={handleGrokOverview}
              disabled={grokLoading}
              className="px-3 py-1.5 bg-violet-500/20 hover:bg-violet-500/30 text-violet-300 rounded-lg text-xs font-bold border border-violet-500/30 disabled:opacity-50"
            >
              {grokLoading ? 'Loading…' : 'Load Snapshot'}
            </button>
            <button
              onClick={handleGrokAnalyze}
              disabled={grokLoading}
              className="px-3 py-1.5 bg-zinc-800 hover:bg-zinc-700 text-zinc-200 rounded-lg text-xs font-bold border border-zinc-700 disabled:opacity-50"
            >
              Run Grok Analysis
            </button>
          </div>
          {grokSummary && (
            <pre className="text-[10px] text-zinc-300 bg-zinc-900/60 p-3 rounded-lg border border-zinc-800 whitespace-pre-wrap max-h-48 overflow-y-auto">
              {grokSummary}
            </pre>
          )}
          {grokOverview?.sections && (
            <div className="mt-3 grid grid-cols-2 gap-2 text-[10px]">
              {Object.keys(grokOverview.sections).slice(0, 6).map((k) => (
                <div key={k} className="px-2 py-1 bg-zinc-900/50 rounded border border-zinc-800 text-zinc-400 capitalize">
                  {k.replace(/_/g, ' ')}
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="bg-[#141416] border border-zinc-800 rounded-2xl p-6">
          <h3 className="text-sm font-bold mb-4 flex items-center gap-2">
            <Link2 size={16} className="text-sky-400" /> Agent Connect API
          </h3>
          {agentManifest ? (
            <div className="space-y-2 text-xs">
              <p className="text-zinc-400 font-mono truncate">{agentManifest.name} v{agentManifest.version}</p>
              <p className="text-zinc-500">OAuth: <span className="text-zinc-300 font-mono">{agentManifest.auth_methods?.[2]?.token_url || '/api/agents/oauth/token'}</span></p>
              <div className="flex flex-wrap gap-1 mt-2">
                {(agentManifest.scopes || []).map((s: string) => (
                  <span key={s} className="px-2 py-0.5 bg-sky-500/10 border border-sky-500/20 text-sky-300 rounded text-[9px] font-bold uppercase">
                    {s}
                  </span>
                ))}
              </div>
              <p className="text-[10px] text-zinc-600 mt-3">Manifest: GET /api/agents/connect</p>
            </div>
          ) : (
            <p className="text-xs text-zinc-500">Agent manifest unavailable — check backend /api/agents/connect</p>
          )}
        </div>
      </div>
    </motion.div>
  );
};

/**
 * API Service — Central bridge between the React frontend and FastAPI backend.
 * All backend communication flows through this module.
 *
 * Backend routing is deployment-controlled, never user/localStorage-controlled.
 * This prevents different services in one browser session talking to different
 * trading backends. Vite/nginx both expose the same `/api/backend` path.
 */

const LOCAL_STORAGE_KEY = 'quantum_trade_settings';

function getAdminApiKey(): string {
  try {
    const envKey = (import.meta as any).env?.VITE_ADMIN_API_KEY;
    if (envKey) return envKey;
    const sessionSecrets = sessionStorage.getItem('quantum_trade_session_secrets');
    if (sessionSecrets) {
      const secrets = JSON.parse(sessionSecrets);
      if (secrets.ADMIN_API_KEY) return secrets.ADMIN_API_KEY;
    }
    const stored = localStorage.getItem(LOCAL_STORAGE_KEY);
    if (stored) {
      const settings = JSON.parse(stored);
      if (settings.ADMIN_API_KEY) return settings.ADMIN_API_KEY;
    }
  } catch { /* ignore */ }
  return '';
}

function getBackendUrl(): string {
  return (import.meta.env.VITE_BACKEND_URL || '/api/backend').replace(/\/+$/, '');
}

async function request<T = any>(path: string, options?: RequestInit): Promise<T> {
  const base = getBackendUrl();
  const url = `${base}${path}`;
  const adminKey = getAdminApiKey();
  const res = await fetch(url, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...(adminKey ? { 'X-API-Key': adminKey } : {}),
      ...(options?.headers || {}),
    },
  });
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(`API ${res.status}: ${text}`);
  }
  return res.json();
}

/**
 * Direct GET against backend routes mounted at the app root under `/api/*`
 * (e.g. `/api/market-data`, `/api/feed`, `/api/forecast`). These routers carry
 * their own `/api/...` prefix, so they are NOT reachable via the `/api/backend`
 * rewrite — this mirrors how `getMultiAssetBars` calls `/api/market-data/bars`.
 */
async function directGet<T = any>(path: string): Promise<T> {
  const adminKey = getAdminApiKey();
  const res = await fetch(path, {
    headers: {
      'Content-Type': 'application/json',
      ...(adminKey ? { 'X-API-Key': adminKey } : {}),
    },
  });
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(`API ${res.status}: ${text}`);
  }
  return res.json();
}

// ── Types ────────────────────────────────────────────────────────────────────

export interface EquityBooks {
  active_broker?: string;
  risk_broker?: string;
  split_book?: boolean;
  union_label?: string;
  informational_union_equity?: number | null;
  [key: string]: unknown;
}

export interface SystemStatus {
  backend: string;
  strategies_loaded: number;
  dry_run: boolean;
  mode: string;
  active_broker?: string;
  llm_providers: { name: string; model: string; status: string; type: string; role?: string }[];
  brokers: { name: string; env: string; status: string }[];
  data_providers: { name: string; status: string; note?: string }[];
  equity_books?: EquityBooks;
  risk_config: {
    risk_per_trade: number;
    max_positions: number;
    min_signal_strength: number;
    min_risk_reward: number;
    use_kelly: boolean;
    vix_threshold: number;
    weights: { technical: number; sentiment: number; macro: number };
  };
  telegram: boolean;
  influxdb: boolean;
  n8n: boolean;
  uptime: string;
  last_cycle: string | null;
  trading_loop: Record<string, any>;
}

/** Quote shape returned by the unified feed (`/api/feed/*`). */
export interface FeedQuote {
  symbol: string;
  asset_class: string;
  price: number | null;
  change_abs: number | null;
  change_pct: number | null;
  high: number | null;
  low: number | null;
  volume: number | null;
  source: string;
  as_of: string;
  stale: boolean;
  error: string | null;
}

export interface FeedOverview {
  as_of: string;
  crypto: FeedQuote[];
  equities: FeedQuote[];
  metals: FeedQuote[];
}

export interface FeedQuotesResponse {
  as_of: string;
  quotes: FeedQuote[];
}

export interface FeedBar {
  time: string | number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface FeedBarsResponse {
  symbol: string;
  asset_class: string;
  source: string;
  data: FeedBar[];
}

export interface ForecastPoint {
  date: string;
  close: number;
}

export interface ForecastResult {
  symbol: string;
  interval: string;
  as_of: string;
  signal: string;
  confidence: number | null;
  predicted_close: number | null;
  predicted_change_pct: number | null;
  cum_change_5_pct: number | null;
  cum_change_10_pct: number | null;
  reversal_risk: boolean | null;
  model_backend: string | null;
  forecast_path: ForecastPoint[] | null;
  error: string | null;
}

export interface BatchForecastsResponse {
  as_of: string | null;
  results: ForecastResult[];
}

export interface SchedulerJob {
  name: string;
  cron_expr: string;
  enabled: boolean;
  last_run: string | null;
  next_run: string | null;
  last_error: string | null;
}

export interface FeedSchedulerStatus {
  enabled: boolean;
  jobs: SchedulerJob[];
}

export interface Position {
  id: number;
  symbol: string;
  direction: string;
  quantity: number;
  entry_price: number;
  current_price: number;
  stop_loss: number | null;
  take_profit: number | null;
  unrealized_pnl: number;
  unrealized_pnl_pct: number;
  strategy: string;
  broker?: string | null;
  opened_at: string | null;
}

export interface Trade {
  id: number;
  timestamp: string | null;
  closed_at: string | null;
  symbol: string;
  direction: string;
  quantity: number;
  entry_price: number;
  exit_price: number | null;
  stop_loss: number | null;
  take_profit: number | null;
  status: string;
  pnl: number | null;
  strategy: string;
  notes: string | null;
}

export interface Signal {
  id: number;
  timestamp: string | null;
  symbol: string;
  strategy: string;
  direction: string;
  confidence: number;
  entry_price: number | null;
  stop_loss: number | null;
  take_profit: number | null;
  status: string;
  reasoning: string | null;
  ai_analysis: Record<string, any> | null;
}

export interface Portfolio {
  balance: number;
  available: number;
  equity: number;
  positions: any[];
  total_pnl: number;
  total_pnl_pct: number;
  total_pnl_is_not_cash?: boolean;
  total_pnl_source?: string;
  lifetime_db_pnl?: number;
  lifetime_db_pnl_is_not_cash?: boolean;
  realized_today?: number;
  realized_7d?: number;
  open_unrealized_pnl?: number;
  unrealized_pnl?: number;
  informational_union_equity?: number | null;
  display_union_equity?: number | null;
  union_label?: string;
  active_broker?: string;
  positions_value: number;
  open_positions_count: number;
  last_updated: string;
}

export interface PortfolioSnapshot {
  id: number;
  timestamp: string | null;
  total_value: number;
  cash: number;
  positions_value: number;
  total_pnl: number;
  open_positions: number;
  cycle_number: number | null;
}

export interface Strategy {
  name: string;
  description: string;
  params: Record<string, any>;
}

export interface LoopStatus {
  running: boolean;
  interval_minutes: number;
  symbols: string[];
  strategy: string;
  last_cycle: string | null;
  total_cycles: number;
  [key: string]: any;
}

// ── API Methods ──────────────────────────────────────────────────────────────

export const apiService = {
  // Generic request helpers
  async get<T = any>(path: string): Promise<T> {
    return request(path, { method: 'GET' });
  },

  async post<T = any>(path: string, body?: any): Promise<T> {
    return request(path, {
      method: 'POST',
      body: JSON.stringify(body),
    });
  },

  // ── System ──────────────────────────────────────────────────────────────
  async getStatus(): Promise<SystemStatus> {
    return request('/trading/status');
  },

  async getConfig() {
    return request('/trading/config');
  },

  async updateConfig(payload: { use_risk_reviewer_llm?: boolean; enable_personas?: boolean }) {
    return request('/trading/config/update', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  },

  async getModels() {
    return request('/trading/models');
  },

  async healthCheck(): Promise<boolean> {
    try {
      await request('/health');
      return true;
    } catch {
      return false;
    }
  },

  // ── Portfolio ───────────────────────────────────────────────────────────
  async getPortfolio(): Promise<Portfolio> {
    return request('/trading/portfolio');
  },

  async getPortfolioHistory(limit = 100): Promise<{ snapshots: PortfolioSnapshot[]; count: number }> {
    return request(`/trading/portfolio/history?limit=${limit}`);
  },

  // ── Positions ───────────────────────────────────────────────────────────
  async getPositions(): Promise<{ positions: Position[]; count: number }> {
    return request('/trading/positions');
  },

  // ── Trades ──────────────────────────────────────────────────────────────
  async getTrades(opts?: { symbol?: string; strategy?: string; status?: string; limit?: number; offset?: number }): Promise<{ trades: Trade[]; total: number }> {
    const params = new URLSearchParams();
    if (opts?.symbol) params.set('symbol', opts.symbol);
    if (opts?.strategy) params.set('strategy', opts.strategy);
    if (opts?.status) params.set('status', opts.status);
    if (opts?.limit) params.set('limit', String(opts.limit));
    if (opts?.offset) params.set('offset', String(opts.offset));
    const qs = params.toString();
    return request(`/trading/trades${qs ? '?' + qs : ''}`);
  },

  // ── Signals ─────────────────────────────────────────────────────────────
  async getSignals(): Promise<{ signals: Signal[] }> {
    return request('/trading/signals');
  },

  async getSignalsHistory(opts?: { symbol?: string; strategy?: string; direction?: string; limit?: number; offset?: number }): Promise<{ signals: Signal[]; total: number }> {
    const params = new URLSearchParams();
    if (opts?.symbol) params.set('symbol', opts.symbol);
    if (opts?.strategy) params.set('strategy', opts.strategy);
    if (opts?.direction) params.set('direction', opts.direction);
    if (opts?.limit) params.set('limit', String(opts.limit));
    if (opts?.offset) params.set('offset', String(opts.offset));
    const qs = params.toString();
    return request(`/trading/signals/history${qs ? '?' + qs : ''}`);
  },

  async getSignalAnalysis(signalId: number) {
    return request(`/trading/analysis/${signalId}`);
  },

  // ── Strategies ──────────────────────────────────────────────────────────
  async getStrategies(): Promise<{ strategies: Strategy[] }> {
    return request('/trading/strategies');
  },

  // ── Trading Loop ────────────────────────────────────────────────────────
  async getLoopStatus(): Promise<LoopStatus> {
    return request('/trading/loop/status');
  },

  async startLoop(config?: { interval_minutes?: number; symbols?: string[]; strategy?: string }) {
    return request('/trading/loop/start', {
      method: 'POST',
      body: JSON.stringify(config || {}),
    });
  },

  async stopLoop() {
    return request('/trading/loop/stop', { method: 'POST' });
  },

  // ── AI Analysis ─────────────────────────────────────────────────────────
  async analyzeSymbol(symbol: string) {
    return request('/trading/analyze', {
      method: 'POST',
      body: JSON.stringify({ symbol }),
    });
  },

  // ── Backtesting ─────────────────────────────────────────────────────────
  async runBacktest(config: { symbol: string; strategy: string; days: number }) {
    return request('/trading/run-backtest', {
      method: 'POST',
      body: JSON.stringify(config),
    });
  },

  // ── Broker Controls ─────────────────────────────────────────────────────
  async getCTraderStatus() {
    return request('/trading/ctrader/status');
  },

  async enableCTrader() {
    return request('/trading/ctrader/enable', { method: 'POST' });
  },

  async disableCTrader() {
    return request('/trading/ctrader/disable', { method: 'POST' });
  },

  // ── Hedge Fund ──────────────────────────────────────────────────────────
  async runHedgeFund(config: any) {
    return request('/hedge-fund/run', {
      method: 'POST',
      body: JSON.stringify(config),
    });
  },

  // ── Market Feeds ────────────────────────────────────────────────────────
  async getStocks(): Promise<{ data: any[] }> {
    return request('/trading/markets/stocks');
  },

  async getForex(): Promise<{ data: any[] }> {
    return request('/trading/markets/forex');
  },

  // ── Account & Balance ────────────────────────────────────────────────────
  async getAccountData(): Promise<{ equity: number; availableBalance: number; dailyPnL: number }> {
    const data = await request<{
      equity: number;
      available_balance: number;
      daily_pnl: number;
    }>('/trading/account/summary');
    return {
      equity: data.equity,
      availableBalance: data.available_balance,
      dailyPnL: data.daily_pnl,
    };
  },

  // ── Flow Runs ──────────────────────────────────────────────────────────────
  async createFlowRun(flowId: string, runData: {
    status: string;
    logs: any[];
    trades: any[];
    executionTime: number;
    halted: boolean;
    haltReason?: string;
  }) {
    return request(`/flows/${flowId}/runs`, {
      method: 'POST',
      body: JSON.stringify(runData),
    });
  },

  // ── Position Actions ────────────────────────────────────────────────────────
  async closePosition(positionId: number | string): Promise<{ success: boolean; exit_price: number; pnl: number; message: string }> {
    return request(`/trading/positions/${positionId}/close`, { method: 'POST' });
  },

  async modifyPosition(positionId: number | string, stopLoss: number | null, takeProfit: number | null): Promise<{ success: boolean }> {
    return request(`/trading/positions/${positionId}/modify`, {
      method: 'PUT',
      body: JSON.stringify({ stop_loss: stopLoss, take_profit: takeProfit }),
    });
  },

  // ── Paper Trading (Fincept Port) ─────────────────────────────────────────────────────
  async initSession(broker: string, mode: 'paper' | 'live', balance = 100000, leverage = 1) {
    return request('/trading/session/init', {
      method: 'POST',
      body: JSON.stringify({ broker, mode, paper_balance: balance, leverage }),
    });
  },

  async getSessionStatus() {
    return request('/trading/session/status');
  },

  async listSessions() {
    return request('/trading/sessions');
  },

  async switchSession(sessionId: string) {
    return request('/trading/session/switch', {
      method: 'POST',
      body: JSON.stringify({ session_id: sessionId }),
    });
  },

  async paperPlaceOrder(symbol: string, side: 'buy' | 'sell', quantity: number, price = 0) {
    return request('/trading/paper/order', {
      method: 'POST',
      body: JSON.stringify({ symbol, side, order_type: price > 0 ? 'limit' : 'market', quantity, price }),
    });
  },

  async placeOrder(
    symbol: string,
    side: 'buy' | 'sell',
    quantity: number,
    orderType: 'market' | 'limit' = 'market',
    price = 0,
    stopLoss?: number,
    takeProfit?: number,
  ) {
    return request('/trading/order', {
      method: 'POST',
      body: JSON.stringify({
        symbol,
        side,
        order_type: orderType,
        quantity,
        price: orderType === 'limit' ? price : 0,
        stop_loss: stopLoss,
        take_profit: takeProfit,
      }),
    });
  },

  async paperCancelOrder(orderId: string) {
    return request('/trading/paper/cancel', {
      method: 'POST',
      body: JSON.stringify({ order_id: orderId }),
    });
  },

  async paperPortfolio() {
    return request('/trading/paper/portfolio');
  },

  async paperPositions() {
    return request('/trading/paper/positions');
  },

  async paperOrders(status = '') {
    return request(`/trading/paper/orders?status=${status}`);
  },

  async paperStats() {
    return request('/trading/paper/stats');
  },

  // ── Opinion Layer ────────────────────────────────────────────────────────
  async analyzeOpinion(symbol: string, bars: any[], includeKronos = true, includeSocial = true, includeAlerts = true, includePersonas = true) {
    return request('/trading/opinion/analyze', {
      method: 'POST',
      body: JSON.stringify({
        symbol,
        bars,
        include_kronos: includeKronos,
        include_social: includeSocial,
        include_alerts: includeAlerts,
        include_personas: includePersonas,
      }),
    });
  },

  async getOpinionWeights() {
    return request('/trading/opinion/weights');
  },

  async updateOpinionWeights(weights: Record<string, number>) {
    return request('/trading/opinion/weights', {
      method: 'POST',
      body: JSON.stringify({ weights }),
    });
  },

  // ── Multi-Broker & Open API ────────────────────────────────────────────────
  async getBrokers() {
    return request('/trading/brokers');
  },

  async getMarkets() {
    return request('/trading/markets');
  },

  async placeSmartOrder(order: {
    symbol: string;
    direction: 'BUY' | 'SELL';
    quantity: number;
    order_type?: 'MARKET' | 'LIMIT';
    price?: number;
    stop_loss?: number;
    take_profit?: number;
    broker_override?: 'binance_futures' | 'ctrader';
  }) {
    return request('/trading/order/smart', {
      method: 'POST',
      body: JSON.stringify(order),
    });
  },

  async getCTraderTokens() {
    return request('/trading/ctrader/tokens');
  },

  async saveCTraderTokens(tokens: {
    account_id: number;
    client_id: string;
    client_secret?: string;
    access_token: string;
    refresh_token: string;
    expires_in?: number;
  }) {
    return request('/trading/ctrader/tokens', {
      method: 'POST',
      body: JSON.stringify(tokens),
    });
  },

  async enableCTraderLive() {
    return request('/trading/ctrader/enable', { method: 'POST' });
  },

  async disableCTraderLive() {
    return request('/trading/ctrader/disable', { method: 'POST' });
  },

  async getCTraderTrendbars(symbol = 'EURUSD', period = 'M5', count = 120, from_ts?: number, to_ts?: number) {
    const params = new URLSearchParams({ symbol, period, count: count.toString() });
    if (from_ts) params.append('from_ts', from_ts.toString());
    if (to_ts) params.append('to_ts', to_ts.toString());
    return request(`/trading/ctrader/trendbars?${params.toString()}`);
  },

  async getCTraderTicks(symbol = 'EURUSD', type = 'BID', hours = 4) {
    const params = new URLSearchParams({ symbol, type, hours: hours.toString() });
    return request(`/trading/ctrader/ticks?${params.toString()}`);
  },

  async getCTraderSymbolSpec(symbol = 'EURUSD') {
    const params = new URLSearchParams({ symbol });
    return request(`/trading/ctrader/symbol-spec?${params.toString()}`);
  },

  async calculatePipMargin(params: {
    symbol: string;
    lots: number;
    price?: number;
    leverage?: number;
    deposit_asset?: string;
  }) {
    return request('/trading/ctrader/calc/pip-margin', {
      method: 'POST',
      body: JSON.stringify(params),
    });
  },

  // ── Signals & Timing Control ───────────────────────────────────────────────
  async scanMarkets(universe?: string[], timeframe = 'M5') {
    return request('/signals/scan-markets', {
      method: 'POST',
      body: JSON.stringify({ universe, timeframe }),
    });
  },

  async scanNewsSignals(lookahead_minutes = 60) {
    return request('/signals/scan-news', {
      method: 'POST',
      body: JSON.stringify({ lookahead_minutes }),
    });
  },

  async getSignalsCandidates(status?: string, broker?: string) {
    const params = new URLSearchParams();
    if (status) params.append('status', status);
    if (broker) params.append('broker', broker);
    return request(`/signals/candidates?${params.toString()}`);
  },

  async getReadySignals() {
    return request('/signals/ready-for-execution');
  },

  async executeSignalCandidate(candidate_id: string, force = false) {
    return request('/signals/execute-candidate', {
      method: 'POST',
      body: JSON.stringify({ candidate_id, force }),
    });
  },

  async getTimingConfig() {
    return request('/signals/timing-config');
  },

  async updateTimingConfig(config: any) {
    return request('/signals/timing-config', {
      method: 'POST',
      body: JSON.stringify(config),
    });
  },

  // ── AI Agent ──────────────────────────────────────────────────────────────────────────────
  async aiAgentTrade(prompt: string, provider = 'xai', model = 'grok-beta') {
    return request('/trading/ai/agent-trade', {
      method: 'POST',
      body: JSON.stringify({ prompt, provider, model }),
    });
  },

  // ── Assistant (OmniRoute) ───────────────────────────────────────────────────────────────
  async assistantChat(message: string, history: { role: string; text: string }[] = []) {
    return request('/assistant/chat', {
      method: 'POST',
      body: JSON.stringify({ message, history }),
    });
  },

  // ── External agents / Grok overseer ───────────────────────────────────────────────────────
  async getAgentConnectManifest() {
    return request('/agents/connect');
  },

  async getGrokOverseerOverview() {
    return request('/agents/grok-overseer/overview');
  },

  async analyzeGrokOverseer(focus = 'risk and operational health', include_positions = true) {
    return request('/agents/grok-overseer/analyze', {
      method: 'POST',
      body: JSON.stringify({ focus, include_positions }),
    });
  },

  async getMultiAssetBars(symbol: string, timeframe = '1h', limit = 100) {
    const params = new URLSearchParams({ symbol, timeframe, limit: String(limit) });
    const res = await fetch(`/api/market-data/bars?${params.toString()}`);
    if (!res.ok) {
      const text = await res.text().catch(() => res.statusText);
      throw new Error(`Market data error: ${text}`);
    }
    return res.json();
  },

  async getFeedOverview(): Promise<FeedOverview> {
    return directGet('/api/feed/overview');
  },

  async getFeedQuotes(symbols?: string[], assetClass?: string): Promise<FeedQuotesResponse> {
    const params = new URLSearchParams();
    if (symbols && symbols.length > 0) params.set('symbols', symbols.join(','));
    if (assetClass) params.set('asset_class', assetClass);
    const qs = params.toString();
    return directGet(`/api/feed/quotes${qs ? '?' + qs : ''}`);
  },

  async getFeedBars(symbol: string, timeframe = '1h', limit = 100): Promise<FeedBarsResponse> {
    const params = new URLSearchParams({ symbol, timeframe, limit: String(limit) });
    return directGet(`/api/feed/bars?${params.toString()}`);
  },

  async getFeedSchedulerStatus(): Promise<FeedSchedulerStatus> {
    return directGet('/api/feed/scheduler/status');
  },

  async getForecast(
    symbol: string,
    opts?: { interval?: string; pred_len?: number; include_path?: boolean },
  ): Promise<ForecastResult> {
    const params = new URLSearchParams();
    if (opts?.interval) params.set('interval', opts.interval);
    if (opts?.pred_len !== undefined) params.set('pred_len', String(opts.pred_len));
    if (opts?.include_path !== undefined) params.set('include_path', String(opts.include_path));
    const qs = params.toString();
    return directGet(`/api/forecast/${encodeURIComponent(symbol)}${qs ? '?' + qs : ''}`);
  },

  async getBatchForecasts(): Promise<BatchForecastsResponse> {
    return directGet('/api/forecast/batch/latest');
  },
  // ── SSE ────────────────────────────────────────────────────────────────────────────────────
  connectEventStream(topics: string[], onMessage: (msg: any) => void) {
    const base = getBackendUrl();
    const url = `${base}/trading/stream?topics=${encodeURIComponent(topics.join(','))}`;
    const es = new EventSource(url);
    es.onmessage = (e) => {
      try {
        onMessage(JSON.parse(e.data));
      } catch (err) {
        console.warn('SSE parse error', err);
      }
    };
    es.onerror = (e) => console.error('SSE error', e);
    return es;
  },
};

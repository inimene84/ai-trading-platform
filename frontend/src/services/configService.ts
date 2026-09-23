/**
 * Config Service
 * Handles secure retrieval of API keys and secrets.
 * Sources (browser only, never build-time env):
 * 1. Session storage (secrets entered via the UI, cleared on browser close)
 * 2. Local storage (non-secret settings)
 */

const LOCAL_STORAGE_KEY = 'quantum_trade_settings';
const SESSION_SECRETS_KEY = 'quantum_trade_session_secrets';

export const configService = {
  getSecret(key: string): string | undefined {
    // Secrets are never read from build-time env (process.env / import.meta.env).
    // Vite inlines the whole import.meta.env object for dynamic lookups, so any
    // VITE_* secret present at build time would ship in the public JS bundle.
    // Keys live server-side; the browser only holds what the operator enters.

    // 2. Session-only secrets (never persist credentials across browser restarts)
    try {
      const stored = sessionStorage.getItem(SESSION_SECRETS_KEY);
      if (stored) {
        const secrets = JSON.parse(stored);
        if (secrets[key]) return secrets[key];
      }
    } catch (e) {
      console.error('Error reading session secrets:', e);
    }

    // 3. Non-secret local settings
    try {
      const stored = localStorage.getItem(LOCAL_STORAGE_KEY);
      if (stored) {
        const settings = JSON.parse(stored);
        return settings[key];
      }
    } catch (e) {
      console.error('Error reading from local storage:', e);
    }

    return undefined;
  },

  /**
   * Check if a secret is managed by the system (environment variable)
   */
  isSystemManaged(_key: string): boolean {
    // Build-time env secrets are no longer supported in the browser.
    return false;
  },

  /**
   * List of all supported secret keys
   */
  getKeys() {
    return [
      'GEMINI_API_KEY',
      'XAI_API_KEY',
      'OPENAI_API_KEY',
      'ANTHROPIC_API_KEY',
      'BINANCE_API_KEY',
      'BINANCE_API_SECRET',
      'CTRADER_CLIENT_ID',
      'CTRADER_CLIENT_SECRET',
      'CTRADER_ACCESS_TOKEN',
      'CTRADER_REFRESH_TOKEN',
      'CTRADER_ACCOUNT_ID',
      'CTRADER_ENV',
      'AGENT_CLIENT_ID',
      'AGENT_CLIENT_SECRET',
      'AGENT_API_KEY',
      'COINGECKO_API_KEY',
      'COINMARKETCAP_API_KEY',
      'ALPHAVANTAGE_API_KEY',
      'POLYGON_API_KEY',
      'FRED_API_KEY',
      'NEWSAPI_KEY',
      'TWELVEDATA_API_KEY',
      'POSTGRES_URL',
      'POSTGRES_HOST',
      'POSTGRES_PORT',
      'POSTGRES_USER',
      'POSTGRES_PASSWORD',
      'POSTGRES_DB',
      'SUPABASE_URL',
      'SUPABASE_ANON_KEY',
      'SUPABASE_SERVICE_ROLE_KEY',
      'MYSQL_URL',
      'MYSQL_HOST',
      'MYSQL_PORT',
      'MYSQL_USER',
      'MYSQL_PASSWORD',
      'MYSQL_DB',
      'INFLUXDB_URL',
      'INFLUXDB_TOKEN',
      'INFLUXDB_ORG',
      'INFLUXDB_BUCKET',
      'INFLUXDB_PRECISION',
      'GRAFANA_URL',
      'GRAFANA_API_KEY',
      'TELEGRAM_BOT_TOKEN',
      'TELEGRAM_CHAT_ID',
      'N8N_WEBHOOK_URL',
      'RISK_PER_TRADE',
      'MAX_POSITIONS',
      'DEFAULT_STOP_LOSS',
      'DEFAULT_TAKE_PROFIT',
      'DAILY_LOSS_LIMIT'
    ];
  }
};

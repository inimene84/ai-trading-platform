import React, { useState, useEffect } from 'react';
import { 
  Shield, 
  Key, 
  Cpu, 
  Globe, 
  Activity, 
  Lock, 
  Eye, 
  EyeOff, 
  Save,
  AlertCircle,
  CheckCircle2,
  Server,
  User,
  Database,
  BarChart3,
  Cloud
} from 'lucide-react';
import { cn } from '../lib/utils';
import { motion } from 'motion/react';
import { configService } from '../services/configService';
import { useToast } from './Toast';
import { apiService } from '../services/apiService';

interface SettingFieldProps {
  label: string;
  secretKey: string;
  value: string;
  onChange: (val: string) => void;
  type?: 'text' | 'password';
  placeholder?: string;
  description?: string;
  error?: string;
}

const SettingField = ({ label, secretKey, value, onChange, type = 'password', placeholder, description, error }: SettingFieldProps) => {
  const [show, setShow] = useState(false);
  const isSystemManaged = configService.isSystemManaged(secretKey);
  const displayValue = isSystemManaged ? '••••••••••••••••' : value;

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <label className="text-[10px] uppercase tracking-wider text-zinc-500 font-bold">{label}</label>
          {isSystemManaged ? (
            <span className="flex items-center gap-1 px-1.5 py-0.5 bg-indigo-500/10 text-indigo-400 text-[8px] font-bold rounded border border-indigo-500/20 uppercase tracking-tighter">
              <Server size={8} /> System Managed
            </span>
          ) : (
            <span className="flex items-center gap-1 px-1.5 py-0.5 bg-zinc-800 text-zinc-500 text-[8px] font-bold rounded border border-zinc-700 uppercase tracking-tighter">
              <User size={8} /> User Provided
            </span>
          )}
        </div>
        {!isSystemManaged && type === 'password' && (
          <button 
            onClick={() => setShow(!show)}
            className="text-zinc-500 hover:text-zinc-300 transition-colors"
          >
            {show ? <EyeOff size={14} /> : <Eye size={14} />}
          </button>
        )}
      </div>
      <div className="relative">
        <input
          type={isSystemManaged ? 'password' : (type === 'password' ? (show ? 'text' : 'password') : 'text')}
          value={displayValue}
          onChange={(e) => !isSystemManaged && onChange(e.target.value)}
          placeholder={isSystemManaged ? 'Managed by Environment' : placeholder}
          disabled={isSystemManaged}
          className={cn(
            "w-full bg-zinc-900/50 border rounded-xl py-2.5 px-4 text-xs font-mono transition-all",
            isSystemManaged 
              ? "border-indigo-500/20 text-indigo-300/50 cursor-not-allowed" 
              : error 
                ? "border-rose-500/50 focus:border-rose-500 focus:outline-none" 
                : "border-zinc-800 focus:outline-none focus:border-emerald-500/50"
          )}
        />
      </div>
      {error && <p className="text-[10px] text-rose-400 font-medium">{error}</p>}
      {description && !error && <p className="text-[10px] text-zinc-600 italic">{description}</p>}
    </div>
  );
};

const SettingsSection = ({ title, icon: Icon, children }: { title: string, icon: any, children: React.ReactNode }) => (
  <div className="bg-[#141416] border border-zinc-800 rounded-2xl p-6 space-y-6">
    <div className="flex items-center gap-3 border-b border-zinc-800 pb-4">
      <div className="p-2 bg-emerald-500/10 rounded-lg">
        <Icon size={18} className="text-emerald-400" />
      </div>
      <h3 className="font-bold text-sm tracking-tight">{title}</h3>
    </div>
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
      {children}
    </div>
  </div>
);

export const SettingsView = () => {
  const { showToast } = useToast();
  const [settings, setSettings] = useState<Record<string, string>>({});
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [isSaved, setIsSaved] = useState(false);
  const [useRiskReviewerLlm, setUseRiskReviewerLlm] = useState(true);
  const [enablePersonas, setEnablePersonas] = useState(false);

  useEffect(() => {
    const saved = localStorage.getItem('quantum_trade_settings');
    const sessionSecrets = sessionStorage.getItem('quantum_trade_session_secrets') || localStorage.getItem('quantum_trade_session_secrets');
    if (saved || sessionSecrets) {
      const parsed = {
        ...(saved ? JSON.parse(saved) : {}),
        ...(sessionSecrets ? JSON.parse(sessionSecrets) : {}),
      };
      // Migrate legacy persisted secrets out of localStorage.
      const isSecret = (key: string) => (
        key.includes('KEY') || key.includes('TOKEN')
        || key.includes('SECRET') || key.includes('PASSWORD')
      );
      const persistent: Record<string, string> = {};
      const secrets: Record<string, string> = {};
      Object.entries(parsed).forEach(([key, value]) => {
        (isSecret(key) ? secrets : persistent)[key] = value as string;
      });
      localStorage.setItem('quantum_trade_settings', JSON.stringify(persistent));
      sessionStorage.setItem('quantum_trade_session_secrets', JSON.stringify(secrets));
      setSettings(parsed);
      
      // Initial validation
      const initialErrors: Record<string, string> = {};
      Object.entries(parsed).forEach(([key, val]) => {
        const error = validateSetting(key, val as string);
        if (error) initialErrors[key] = error;
      });
      setErrors(initialErrors);
    }

    async function loadBackendConfig() {
      try {
        const cfg = await apiService.getConfig();
        const limits = cfg.risk_limits || {};
        if (limits.use_risk_reviewer_llm !== undefined) {
          setUseRiskReviewerLlm(limits.use_risk_reviewer_llm);
        }
        if (limits.enable_personas !== undefined) {
          setEnablePersonas(limits.enable_personas);
        }
      } catch (err) {
        console.error('Failed to load backend config in settings:', err);
      }
    }
    loadBackendConfig();
  }, []);

  const handleToggleRiskReviewer = async (val: boolean) => {
    try {
      setUseRiskReviewerLlm(val);
      await apiService.updateConfig({ use_risk_reviewer_llm: val });
      showToast(`Single LLM Risk Reviewer ${val ? 'enabled' : 'disabled'}`, 'success');
    } catch (err) {
      showToast('Failed to update LLM Risk Reviewer setting', 'error');
      setUseRiskReviewerLlm(!val);
    }
  };

  const handleTogglePersonas = async (val: boolean) => {
    try {
      setEnablePersonas(val);
      await apiService.updateConfig({ enable_personas: val });
      showToast(`Multi-Persona Committee ${val ? 'enabled' : 'disabled'}`, 'success');
    } catch (err) {
      showToast('Failed to update Multi-Persona Committee setting', 'error');
      setEnablePersonas(!val);
    }
  };

  const validateSetting = (key: string, val: string) => {
    if (!val) return null; // Allow empty
    
    // Check for trailing/leading whitespaces globally for all inputs
    if (val.trim() !== val) {
      return 'Cannot contain leading or trailing spaces.';
    }

    if (key.includes('KEY') || key.includes('TOKEN') || key.includes('SECRET')) {
      if (val.includes(' ')) return 'API keys cannot contain spaces.';
      if (val.length < 8) return 'Key is too short (min 8 chars).';
    } else if (key.includes('URL') && !key.includes('POSTGRES') && !key.includes('MYSQL') && val.length > 0) {
      if (val.includes(' ')) return 'URLs cannot contain spaces.';
      if (!val.startsWith('http://') && !val.startsWith('https://')) {
        return 'URL must start with http:// or https://';
      }
    } else if (key === 'RISK_PER_TRADE' || key.includes('LOSS') || key.includes('PROFIT')) {
      if (isNaN(Number(val))) return 'Must be a valid number.';
      if (Number(val) < 0) return 'Cannot be negative.';
      if (Number(val) > 100) return 'Cannot exceed 100%.';
    } else if (key === 'MAX_POSITIONS') {
      if (isNaN(Number(val)) || !Number.isInteger(Number(val))) return 'Must be an integer.';
      if (Number(val) <= 0) return 'Must be greater than 0.';
    }
    return null;
  };

  const handleSave = () => {
    if (Object.keys(errors).length > 0) {
      showToast('Please fix errors before saving.', 'error');
      return;
    }
    const isSecret = (key: string) => (
      key.includes('KEY')
      || key.includes('TOKEN')
      || key.includes('SECRET')
      || key.includes('PASSWORD')
    );
    const persistent: Record<string, string> = {};
    const secrets: Record<string, string> = {};
    Object.entries(settings).forEach(([key, value]) => {
      (isSecret(key) ? secrets : persistent)[key] = String(value ?? '');
    });
    localStorage.setItem('quantum_trade_settings', JSON.stringify(persistent));
    sessionStorage.setItem('quantum_trade_session_secrets', JSON.stringify(secrets));
    localStorage.setItem('quantum_trade_session_secrets', JSON.stringify(secrets));
    setIsSaved(true);
    showToast('Configuration saved.', 'success');
    setTimeout(() => setIsSaved(false), 3000);
  };

  const updateSetting = (key: string, val: string) => {
    setSettings(prev => ({ ...prev, [key]: val }));
    const error = validateSetting(key, val);
    setErrors(prev => {
      const next = { ...prev };
      if (error) next[key] = error;
      else delete next[key];
      return next;
    });
  };

  const hasErrors = Object.keys(errors).length > 0;

  return (
    <div className="flex-1 overflow-y-auto p-8 space-y-8 max-w-5xl mx-auto w-full">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Configuration & Secrets</h1>
          <p className="text-zinc-500 text-sm">Manage your API keys and broker credentials securely.</p>
        </div>
        <button 
          onClick={handleSave}
          disabled={hasErrors}
          className={cn(
            "flex items-center gap-2 px-6 py-2.5 rounded-xl text-sm font-bold transition-all shadow-lg",
            isSaved ? "bg-emerald-500 text-black shadow-emerald-500/20" : 
            hasErrors ? "bg-zinc-800 text-zinc-500 cursor-not-allowed border border-zinc-700" :
            "bg-white text-black hover:bg-zinc-200"
          )}
        >
          {isSaved ? (
            <><CheckCircle2 size={18} /> Saved</>
          ) : (
            <><Save size={18} /> Save Changes</>
          )}
        </button>
      </div>

      <div className="bg-amber-500/5 border border-amber-500/20 rounded-2xl p-4 flex gap-4 items-start">
        <AlertCircle className="text-amber-500 flex-shrink-0" size={20} />
        <div className="space-y-1">
          <p className="text-xs font-bold text-amber-500 uppercase tracking-wider">Security Notice</p>
          <p className="text-xs text-amber-200/70 leading-relaxed">
            These keys are stored in your browser's local storage. For production environments, 
            always use the platform's <strong>Secrets Panel</strong> to inject environment variables securely.
          </p>
        </div>
      </div>

      <SettingsSection title="Platform Connection" icon={Server}>
        <SettingField 
          label="Backend URL" 
          secretKey="BACKEND_URL"
          value={settings.BACKEND_URL || ''} 
          error={errors.BACKEND_URL}
          onChange={(v) => updateSetting('BACKEND_URL', v)}
          type="text"
          placeholder="http://localhost:8000"
          description="FastAPI backend URL. Leave blank to use the default proxy (/api/backend)."
        />
        <SettingField 
          label="Admin API Key" 
          secretKey="ADMIN_API_KEY"
          value={settings.ADMIN_API_KEY || ''} 
          error={errors.ADMIN_API_KEY}
          onChange={(v) => updateSetting('ADMIN_API_KEY', v)}
          type="password"
          placeholder="Required when backend ADMIN_API_KEY is set"
          description="Sent as X-API-Key for loop start/stop and config updates."
        />
      </SettingsSection>

      <SettingsSection title="Agent API & OAuth" icon={Shield}>
        <SettingField 
          label="Agent Client ID" 
          secretKey="AGENT_CLIENT_ID"
          value={settings.AGENT_CLIENT_ID || ''} 
          error={errors.AGENT_CLIENT_ID}
          onChange={(v) => updateSetting('AGENT_CLIENT_ID', v)}
          type="text"
          description="OAuth client_id for external agents (MCP, GrokBOT overseer). Falls back to ADMIN_API_KEY."
        />
        <SettingField 
          label="Agent Client Secret" 
          secretKey="AGENT_CLIENT_SECRET"
          value={settings.AGENT_CLIENT_SECRET || ''} 
          error={errors.AGENT_CLIENT_SECRET}
          onChange={(v) => updateSetting('AGENT_CLIENT_SECRET', v)}
          type="password"
        />
        <SettingField 
          label="Agent API Key (Bearer token)" 
          secretKey="AGENT_API_KEY"
          value={settings.AGENT_API_KEY || ''} 
          error={errors.AGENT_API_KEY}
          onChange={(v) => updateSetting('AGENT_API_KEY', v)}
          type="password"
          description="Issued by POST /api/agents/oauth/token. Connect manifest: GET /api/agents/connect"
        />
      </SettingsSection>

      <SettingsSection title="AI Brains (LLM Engines)" icon={Cpu}>
        <SettingField 
          label="xAI (Grok) API Key" 
          secretKey="XAI_API_KEY"
          value={settings.XAI_API_KEY || ''} 
          error={errors.XAI_API_KEY}
          onChange={(v) => updateSetting('XAI_API_KEY', v)}
        />
        <SettingField 
          label="OpenAI API Key" 
          secretKey="OPENAI_API_KEY"
          value={settings.OPENAI_API_KEY || ''} 
          error={errors.OPENAI_API_KEY}
          onChange={(v) => updateSetting('OPENAI_API_KEY', v)}
        />
        <SettingField 
          label="Anthropic API Key" 
          secretKey="ANTHROPIC_API_KEY"
          value={settings.ANTHROPIC_API_KEY || ''} 
          error={errors.ANTHROPIC_API_KEY}
          onChange={(v) => updateSetting('ANTHROPIC_API_KEY', v)}
        />
      </SettingsSection>

      <SettingsSection title="Market Data Providers" icon={Globe}>
        <SettingField 
          label="CoinGecko API Key" 
          secretKey="COINGECKO_API_KEY"
          value={settings.COINGECKO_API_KEY || ''} 
          error={errors.COINGECKO_API_KEY}
          onChange={(v) => updateSetting('COINGECKO_API_KEY', v)}
        />
        <SettingField 
          label="CoinMarketCap API Key" 
          secretKey="COINMARKETCAP_API_KEY"
          value={settings.COINMARKETCAP_API_KEY || ''} 
          error={errors.COINMARKETCAP_API_KEY}
          onChange={(v) => updateSetting('COINMARKETCAP_API_KEY', v)}
        />
        <SettingField 
          label="Alpha Vantage API Key" 
          secretKey="ALPHAVANTAGE_API_KEY"
          value={settings.ALPHAVANTAGE_API_KEY || ''} 
          error={errors.ALPHAVANTAGE_API_KEY}
          onChange={(v) => updateSetting('ALPHAVANTAGE_API_KEY', v)}
        />
        <SettingField 
          label="Polygon.io API Key" 
          secretKey="POLYGON_API_KEY"
          value={settings.POLYGON_API_KEY || ''} 
          error={errors.POLYGON_API_KEY}
          onChange={(v) => updateSetting('POLYGON_API_KEY', v)}
          description="Free tier limited to 5 requests per minute."
        />
        <SettingField 
          label="FRED API Key" 
          secretKey="FRED_API_KEY"
          value={settings.FRED_API_KEY || ''} 
          error={errors.FRED_API_KEY}
          onChange={(v) => updateSetting('FRED_API_KEY', v)}
          description="Access to Federal Reserve Economic Data."
        />
        <SettingField 
          label="NewsAPI Key" 
          secretKey="NEWSAPI_KEY"
          value={settings.NEWSAPI_KEY || ''} 
          error={errors.NEWSAPI_KEY}
          onChange={(v) => updateSetting('NEWSAPI_KEY', v)}
        />
        <SettingField 
          label="Twelve Data API Key" 
          secretKey="TWELVEDATA_API_KEY"
          value={settings.TWELVEDATA_API_KEY || ''} 
          error={errors.TWELVEDATA_API_KEY}
          onChange={(v) => updateSetting('TWELVEDATA_API_KEY', v)}
          placeholder="Your Twelve Data API Key"
          description="Free tier limited to 8 requests per minute."
        />
      </SettingsSection>

      <SettingsSection title="Execution Brokers" icon={Lock}>
        <div className="col-span-full grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="space-y-4 p-4 bg-zinc-900/30 rounded-xl border border-zinc-800">
            <h4 className="text-xs font-bold text-zinc-400 uppercase tracking-widest">cTrader OpenAPI</h4>
            <SettingField 
              label="Client ID" 
              secretKey="CTRADER_CLIENT_ID"
              value={settings.CTRADER_CLIENT_ID || ''} 
              error={errors.CTRADER_CLIENT_ID}
              onChange={(v) => updateSetting('CTRADER_CLIENT_ID', v)}
              type="text"
            />
            <SettingField 
              label="Client Secret" 
              secretKey="CTRADER_CLIENT_SECRET"
              value={settings.CTRADER_CLIENT_SECRET || ''} 
              error={errors.CTRADER_CLIENT_SECRET}
              onChange={(v) => updateSetting('CTRADER_CLIENT_SECRET', v)}
            />
            <SettingField 
              label="Access Token" 
              secretKey="CTRADER_ACCESS_TOKEN"
              value={settings.CTRADER_ACCESS_TOKEN || ''} 
              error={errors.CTRADER_ACCESS_TOKEN}
              onChange={(v) => updateSetting('CTRADER_ACCESS_TOKEN', v)}
            />
            <SettingField 
              label="Refresh Token" 
              secretKey="CTRADER_REFRESH_TOKEN"
              value={settings.CTRADER_REFRESH_TOKEN || ''} 
              error={errors.CTRADER_REFRESH_TOKEN}
              onChange={(v) => updateSetting('CTRADER_REFRESH_TOKEN', v)}
            />
            <SettingField 
              label="Account ID" 
              secretKey="CTRADER_ACCOUNT_ID"
              value={settings.CTRADER_ACCOUNT_ID || ''} 
              error={errors.CTRADER_ACCOUNT_ID}
              onChange={(v) => updateSetting('CTRADER_ACCOUNT_ID', v)}
              type="text"
            />
            <SettingField 
              label="Environment (demo/live)" 
              secretKey="CTRADER_ENV"
              value={settings.CTRADER_ENV || 'demo'} 
              error={errors.CTRADER_ENV}
              onChange={(v) => updateSetting('CTRADER_ENV', v)}
              type="text"
              description="Use demo for paper; live requires CTRADER_PAPER_MODE=false in .env"
            />
          </div>
          <div className="space-y-4 p-4 bg-zinc-900/30 rounded-xl border border-zinc-800">
            <h4 className="text-xs font-bold text-zinc-400 uppercase tracking-widest">Binance Spot</h4>
            <SettingField 
              label="API Key" 
              secretKey="BINANCE_API_KEY"
              value={settings.BINANCE_API_KEY || ''} 
              error={errors.BINANCE_API_KEY}
              onChange={(v) => updateSetting('BINANCE_API_KEY', v)}
              type="text"
            />
            <SettingField 
              label="API Secret" 
              secretKey="BINANCE_API_SECRET"
              value={settings.BINANCE_API_SECRET || ''} 
              error={errors.BINANCE_API_SECRET}
              onChange={(v) => updateSetting('BINANCE_API_SECRET', v)}
            />
          </div>
        </div>
      </SettingsSection>

      <SettingsSection title="Databases & Storage" icon={Database}>
        <div className="col-span-full grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="space-y-4 p-4 bg-zinc-900/30 rounded-xl border border-zinc-800">
            <h4 className="text-xs font-bold text-zinc-400 uppercase tracking-widest">PostgreSQL</h4>
            <SettingField 
              label="Postgres URL (Direct)" 
              secretKey="POSTGRES_URL"
              value={settings.POSTGRES_URL || ''} 
              error={errors.POSTGRES_URL}
              onChange={(v) => updateSetting('POSTGRES_URL', v)}
              placeholder="postgresql://user:pass@host:port/db"
              description="Full connection string for quick setup."
            />
            <div className="grid grid-cols-2 gap-4">
              <SettingField label="Host" secretKey="POSTGRES_HOST" value={settings.POSTGRES_HOST || ''} error={errors.POSTGRES_HOST} onChange={(v) => updateSetting('POSTGRES_HOST', v)} type="text" />
              <SettingField label="Port" secretKey="POSTGRES_PORT" value={settings.POSTGRES_PORT || '5432'} error={errors.POSTGRES_PORT} onChange={(v) => updateSetting('POSTGRES_PORT', v)} type="text" />
            </div>
            <div className="grid grid-cols-2 gap-4">
              <SettingField label="User" secretKey="POSTGRES_USER" value={settings.POSTGRES_USER || ''} error={errors.POSTGRES_USER} onChange={(v) => updateSetting('POSTGRES_USER', v)} type="text" />
              <SettingField label="Password" secretKey="POSTGRES_PASSWORD" value={settings.POSTGRES_PASSWORD || ''} error={errors.POSTGRES_PASSWORD} onChange={(v) => updateSetting('POSTGRES_PASSWORD', v)} />
            </div>
            <SettingField label="Database Name" secretKey="POSTGRES_DB" value={settings.POSTGRES_DB || ''} error={errors.POSTGRES_DB} onChange={(v) => updateSetting('POSTGRES_DB', v)} type="text" />
          </div>

          <div className="space-y-4 p-4 bg-zinc-900/30 rounded-xl border border-zinc-800">
            <h4 className="text-xs font-bold text-zinc-400 uppercase tracking-widest">Supabase</h4>
            <SettingField 
              label="Project URL" 
              secretKey="SUPABASE_URL"
              value={settings.SUPABASE_URL || ''} 
              error={errors.SUPABASE_URL}
              onChange={(v) => updateSetting('SUPABASE_URL', v)}
              type="text"
              placeholder="https://xyz.supabase.co"
            />
            <SettingField 
              label="Anon Public Key" 
              secretKey="SUPABASE_ANON_KEY"
              value={settings.SUPABASE_ANON_KEY || ''} 
              error={errors.SUPABASE_ANON_KEY}
              onChange={(v) => updateSetting('SUPABASE_ANON_KEY', v)}
            />
            <SettingField 
              label="Service Role Key" 
              secretKey="SUPABASE_SERVICE_ROLE_KEY"
              value={settings.SUPABASE_SERVICE_ROLE_KEY || ''} 
              error={errors.SUPABASE_SERVICE_ROLE_KEY}
              onChange={(v) => updateSetting('SUPABASE_SERVICE_ROLE_KEY', v)}
              description="Required for server-side operations bypassing RLS."
            />
          </div>

          <div className="space-y-4 p-4 bg-zinc-900/30 rounded-xl border border-zinc-800">
            <h4 className="text-xs font-bold text-zinc-400 uppercase tracking-widest">MySQL</h4>
            <SettingField 
              label="MySQL URL (Direct)" 
              secretKey="MYSQL_URL"
              value={settings.MYSQL_URL || ''} 
              error={errors.MYSQL_URL}
              onChange={(v) => updateSetting('MYSQL_URL', v)}
              placeholder="mysql://user:pass@host:port/db"
            />
            <div className="grid grid-cols-2 gap-4">
              <SettingField label="Host" secretKey="MYSQL_HOST" value={settings.MYSQL_HOST || ''} error={errors.MYSQL_HOST} onChange={(v) => updateSetting('MYSQL_HOST', v)} type="text" />
              <SettingField label="Port" secretKey="MYSQL_PORT" value={settings.MYSQL_PORT || '3306'} error={errors.MYSQL_PORT} onChange={(v) => updateSetting('MYSQL_PORT', v)} type="text" />
            </div>
            <div className="grid grid-cols-2 gap-4">
              <SettingField label="User" secretKey="MYSQL_USER" value={settings.MYSQL_USER || ''} error={errors.MYSQL_USER} onChange={(v) => updateSetting('MYSQL_USER', v)} type="text" />
              <SettingField label="Password" secretKey="MYSQL_PASSWORD" value={settings.MYSQL_PASSWORD || ''} error={errors.MYSQL_PASSWORD} onChange={(v) => updateSetting('MYSQL_PASSWORD', v)} />
            </div>
            <SettingField label="Database Name" secretKey="MYSQL_DB" value={settings.MYSQL_DB || ''} error={errors.MYSQL_DB} onChange={(v) => updateSetting('MYSQL_DB', v)} type="text" />
          </div>
        </div>
      </SettingsSection>

      <SettingsSection title="Monitoring & Automation" icon={BarChart3}>
        <div className="col-span-full grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="space-y-4 p-4 bg-zinc-900/30 rounded-xl border border-zinc-800">
            <h4 className="text-xs font-bold text-zinc-400 uppercase tracking-widest">InfluxDB (Time Series)</h4>
            <SettingField 
              label="InfluxDB URL" 
              secretKey="INFLUXDB_URL"
              value={settings.INFLUXDB_URL || ''} 
              error={errors.INFLUXDB_URL}
              onChange={(v) => updateSetting('INFLUXDB_URL', v)}
              type="text"
              placeholder="http://localhost:8086"
            />
            <SettingField 
              label="Access Token" 
              secretKey="INFLUXDB_TOKEN"
              value={settings.INFLUXDB_TOKEN || ''} 
              error={errors.INFLUXDB_TOKEN}
              onChange={(v) => updateSetting('INFLUXDB_TOKEN', v)}
            />
            <div className="grid grid-cols-2 gap-4">
              <SettingField label="Organization" secretKey="INFLUXDB_ORG" value={settings.INFLUXDB_ORG || ''} error={errors.INFLUXDB_ORG} onChange={(v) => updateSetting('INFLUXDB_ORG', v)} type="text" />
              <SettingField label="Bucket" secretKey="INFLUXDB_BUCKET" value={settings.INFLUXDB_BUCKET || ''} error={errors.INFLUXDB_BUCKET} onChange={(v) => updateSetting('INFLUXDB_BUCKET', v)} type="text" />
            </div>
            <SettingField 
              label="Precision" 
              secretKey="INFLUXDB_PRECISION"
              value={settings.INFLUXDB_PRECISION || 's'} 
              error={errors.INFLUXDB_PRECISION}
              onChange={(v) => updateSetting('INFLUXDB_PRECISION', v)}
              type="text"
              placeholder="s, ms, us, ns"
            />
          </div>
          <div className="space-y-4 p-4 bg-zinc-900/30 rounded-xl border border-zinc-800">
            <h4 className="text-xs font-bold text-zinc-400 uppercase tracking-widest">Grafana</h4>
            <SettingField 
              label="Grafana URL" 
              secretKey="GRAFANA_URL"
              value={settings.GRAFANA_URL || ''} 
              error={errors.GRAFANA_URL}
              onChange={(v) => updateSetting('GRAFANA_URL', v)}
              type="text"
              placeholder="https://your-grafana-instance.com"
            />
            <SettingField 
              label="API Key / Service Token" 
              secretKey="GRAFANA_API_KEY"
              value={settings.GRAFANA_API_KEY || ''} 
              error={errors.GRAFANA_API_KEY}
              onChange={(v) => updateSetting('GRAFANA_API_KEY', v)}
              description="Required for automated dashboard creation and data push."
            />
          </div>
          <div className="space-y-4 p-4 bg-zinc-900/30 rounded-xl border border-zinc-800">
            <h4 className="text-xs font-bold text-zinc-400 uppercase tracking-widest">Telegram Alerts</h4>
            <SettingField 
              label="Telegram Bot Token" 
              secretKey="TELEGRAM_BOT_TOKEN"
              value={settings.TELEGRAM_BOT_TOKEN || ''} 
              error={errors.TELEGRAM_BOT_TOKEN}
              onChange={(v) => updateSetting('TELEGRAM_BOT_TOKEN', v)}
              description="Token from @BotFather."
            />
            <SettingField 
              label="Telegram Chat ID" 
              secretKey="TELEGRAM_CHAT_ID"
              value={settings.TELEGRAM_CHAT_ID || ''} 
              error={errors.TELEGRAM_CHAT_ID}
              onChange={(v) => updateSetting('TELEGRAM_CHAT_ID', v)}
              type="text"
              description="Your personal or group chat ID."
            />
          </div>
          <div className="space-y-4 p-4 bg-zinc-900/30 rounded-xl border border-zinc-800">
            <h4 className="text-xs font-bold text-zinc-400 uppercase tracking-widest">n8n Automation</h4>
            <SettingField 
              label="n8n Webhook URL" 
              secretKey="N8N_WEBHOOK_URL"
              value={settings.N8N_WEBHOOK_URL || ''} 
              error={errors.N8N_WEBHOOK_URL}
              onChange={(v) => updateSetting('N8N_WEBHOOK_URL', v)}
              type="text"
              placeholder="https://primary-production.up.railway.app/webhook/..."
              description="Trigger external workflows when trades execute."
            />
          </div>
        </div>
      </SettingsSection>

      <SettingsSection title="AI Opinion & Veto Gate (Dynamic Switch)" icon={Cpu}>
        <div className="col-span-full space-y-6">
          <div className="flex items-center justify-between p-5 bg-zinc-900/30 rounded-2xl border border-zinc-800 hover:border-zinc-700/80 transition-all">
            <div className="space-y-1.5 pr-4">
              <h4 className="text-xs font-bold text-zinc-200 uppercase tracking-wider flex items-center gap-2">
                Single LLM Risk Reviewer (Claude)
                <span className="px-1.5 py-0.5 bg-emerald-500/10 text-emerald-400 text-[8px] font-bold rounded border border-emerald-500/20 uppercase tracking-tighter">Recommended</span>
              </h4>
              <p className="text-[10px] text-zinc-500 leading-relaxed max-w-2xl">
                Invokes a single, highly capable model (Claude via LiteLLM) to inspect the complete trade ticket (symbol, action, size, SL, TP, current funding rate) and relevant Qdrant news headlines. It acts as a strict risk veto gate. Cheap, extremely fast (~2 seconds), and highly intelligent.
              </p>
            </div>
            <label className="relative inline-flex items-center cursor-pointer flex-shrink-0">
              <input
                type="checkbox"
                checked={useRiskReviewerLlm}
                onChange={(e) => handleToggleRiskReviewer(e.target.checked)}
                className="sr-only peer"
              />
              <div className="w-11 h-6 bg-zinc-850 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-zinc-900 after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-zinc-600 after:border-zinc-500 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-emerald-500 peer-checked:after:bg-black peer-checked:after:border-black"></div>
            </label>
          </div>

          <div className="flex items-center justify-between p-5 bg-zinc-900/30 rounded-2xl border border-zinc-800 hover:border-zinc-700/80 transition-all">
            <div className="space-y-1.5 pr-4">
              <h4 className="text-xs font-bold text-zinc-200 uppercase tracking-wider">
                Multi-Persona Committee (18 Agents)
              </h4>
              <p className="text-[10px] text-zinc-500 leading-relaxed max-w-2xl">
                Runs the full 18-agent persona consensus system (Charlie Munger, Nassim Taleb, etc.). Generates detailed multi-agent opinions on the setup. Resource-intensive and slow (~45-80 seconds per symbol), but offers multiple detailed perspectives.
              </p>
            </div>
            <label className="relative inline-flex items-center cursor-pointer flex-shrink-0">
              <input
                type="checkbox"
                checked={enablePersonas}
                onChange={(e) => handleTogglePersonas(e.target.checked)}
                className="sr-only peer"
              />
              <div className="w-11 h-6 bg-zinc-850 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-zinc-900 after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-zinc-600 after:border-zinc-500 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-emerald-500 peer-checked:after:bg-black peer-checked:after:border-black"></div>
            </label>
          </div>
        </div>
      </SettingsSection>

      <SettingsSection title="Risk Management & Guardrails" icon={Shield}>
        <SettingField 
          label="Risk Per Trade (%)" 
          secretKey="RISK_PER_TRADE"
          value={settings.RISK_PER_TRADE || '1.0'} 
          error={errors.RISK_PER_TRADE}
          onChange={(v) => updateSetting('RISK_PER_TRADE', v)}
          type="text"
          description="Percentage of equity to risk on a single trade."
        />
        <SettingField 
          label="Max Open Positions" 
          secretKey="MAX_POSITIONS"
          value={settings.MAX_POSITIONS || '5'} 
          error={errors.MAX_POSITIONS}
          onChange={(v) => updateSetting('MAX_POSITIONS', v)}
          type="text"
          description="Maximum number of active trades allowed simultaneously."
        />
        <SettingField 
          label="Default Stop Loss (%)" 
          secretKey="DEFAULT_STOP_LOSS"
          value={settings.DEFAULT_STOP_LOSS || '2.0'} 
          error={errors.DEFAULT_STOP_LOSS}
          onChange={(v) => updateSetting('DEFAULT_STOP_LOSS', v)}
          type="text"
          description="Automatic stop loss for new orders."
        />
        <SettingField 
          label="Default Take Profit (%)" 
          secretKey="DEFAULT_TAKE_PROFIT"
          value={settings.DEFAULT_TAKE_PROFIT || '4.0'} 
          error={errors.DEFAULT_TAKE_PROFIT}
          onChange={(v) => updateSetting('DEFAULT_TAKE_PROFIT', v)}
          type="text"
          description="Automatic take profit for new orders."
        />
        <SettingField 
          label="Daily Drawdown Limit (%)" 
          secretKey="DAILY_LOSS_LIMIT"
          value={settings.DAILY_LOSS_LIMIT || '5.0'} 
          error={errors.DAILY_LOSS_LIMIT}
          onChange={(v) => updateSetting('DAILY_LOSS_LIMIT', v)}
          type="text"
          description="Pause all workflows if daily realized loss exceeds this value."
        />
      </SettingsSection>
    </div>
  );
};

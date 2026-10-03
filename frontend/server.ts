import express from 'express';
import http from 'http';
import cors from 'cors';
import dotenv from 'dotenv';
import path from 'path';
import { fileURLToPath } from 'url';
import { createServer as createViteServer } from 'vite';
import pg from 'pg';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = Number(process.env.FRONTEND_PORT || process.env.PORT || 5173);
const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8000';

app.use(cors());
app.use(express.json());

// --- PostgreSQL Setup (OPTIONAL) ---
// The platform runs on SQLite in the backend by default; PostgreSQL here is an
// OPTIONAL store for historical backtest candles. If no DB is configured or
// empty, requests fall through to the FastAPI historical-data provider.
const { Pool } = pg;
const PG_CONFIGURED = Boolean(process.env.DATABASE_URL || process.env.POSTGRES_HOST);
let pool: InstanceType<typeof Pool> | null = null;

if (PG_CONFIGURED) {
  pool = new Pool({
    connectionString: process.env.DATABASE_URL || `postgresql://${process.env.POSTGRES_USER || 'postgres'}:${process.env.POSTGRES_PASSWORD || ''}@${process.env.POSTGRES_HOST || 'localhost'}:${process.env.POSTGRES_PORT || '5432'}/${process.env.POSTGRES_DB || 'quantum_trade'}`,
  });

  // Guard: an idle-client error must NOT crash the process.
  pool.on('error', (err) => {
    console.warn('PostgreSQL pool error (backtest store unavailable):', err.message);
  });

  // Test DB connection
  pool.query('SELECT NOW()', (err) => {
    if (err) {
      console.warn('PostgreSQL configured but unreachable; using FastAPI historical provider.', err.message);
    } else {
      console.log('PostgreSQL connected successfully.');
    }
  });
} else {
  console.log('PostgreSQL not configured; using FastAPI historical provider.');
}

// --- API Routes ---

// Get historical data for backtesting
app.get('/api/historical', async (req, res) => {
  const { symbol, interval, limit = 1000 } = req.query;
  
  try {
    // Try to fetch from DB only if a pool was configured.
    if (pool) {
      const result = await pool.query(
        'SELECT * FROM historical_data WHERE symbol = $1 AND interval = $2 ORDER BY time DESC LIMIT $3',
        [symbol, interval, limit]
      );

      if (result.rows.length > 0) {
        return res.json(result.rows.reverse());
      }
    }
  } catch (err) {
    console.error('DB fetch error, falling back to mock:', err);
  }

  // Never present random candles as historical market data. Fall back to the
  // FastAPI market-data implementation; if that is unavailable, fail clearly.
  try {
    const params = new URLSearchParams({
      symbol: String(symbol || ''),
      interval: String(interval || '1h'),
      limit: String(limit),
    });
    const backendRes = await fetch(`${BACKEND_URL}/api/historical?${params}`);
    const body = await backendRes.text();
    res.status(backendRes.status).set(
      'Content-Type', backendRes.headers.get('content-type') || 'application/json',
    );
    res.send(body);
  } catch (error: any) {
    res.status(503).json({
      error: 'Historical market data unavailable',
      detail: error.message,
    });
  }
});

async function proxyTelemetry(req: express.Request, res: express.Response, path: string) {
  try {
    const headers: Record<string, string> = { 'Content-Type': 'application/json' };
    const apiKey = req.headers['x-api-key'];
    const authorization = req.headers.authorization;
    if (typeof apiKey === 'string') headers['X-API-Key'] = apiKey;
    if (authorization) headers.Authorization = authorization;
    const response = await fetch(`${BACKEND_URL}/api/telemetry/${path}`, {
      method: 'POST',
      headers,
      body: JSON.stringify(req.body),
    });
    res.status(response.status).set(
      'Content-Type', response.headers.get('content-type') || 'application/json',
    );
    res.send(await response.text());
  } catch (error: any) {
    console.error(`Telemetry proxy error (${path}):`, error.message);
    res.status(502).json({ error: 'Telemetry backend unavailable' });
  }
}

app.post('/api/telemetry/influx', (req, res) => proxyTelemetry(req, res, 'influx'));
app.post('/api/telemetry/telegram', (req, res) => proxyTelemetry(req, res, 'telegram'));
app.post('/api/telemetry/n8n', (req, res) => proxyTelemetry(req, res, 'n8n'));
// --- Backend API Proxy (works in both dev and production) ---

// --- News API Proxy → FastAPI backend /api/news/* ---
app.use('/api/news', async (req, res) => {
  const targetUrl = `${BACKEND_URL}/api/news${req.url || '/'}`;
  try {
    const fetchOpts: RequestInit = { method: req.method, headers: { 'Content-Type': 'application/json' } };
    if (req.method !== 'GET' && req.method !== 'HEAD' && req.body) {
      fetchOpts.body = JSON.stringify(req.body);
    }
    const backendRes = await fetch(targetUrl, fetchOpts);
    res.status(backendRes.status).set('Content-Type', backendRes.headers.get('content-type') || 'application/json');
    res.send(await backendRes.text());
  } catch (error: any) {
    console.error(`News proxy error: ${targetUrl}`, error.message);
    res.status(502).json({ error: 'News backend unavailable', detail: error.message });
  }
});

// --- Market Data API Proxy → FastAPI backend /api/market-data/* ---
app.use('/api/market-data', async (req, res) => {
  const targetUrl = `${BACKEND_URL}/api/market-data${req.url || '/'}`;
  try {
    const fetchOpts: RequestInit = { method: req.method, headers: { 'Content-Type': 'application/json' } };
    if (req.method !== 'GET' && req.method !== 'HEAD' && req.body) {
      fetchOpts.body = JSON.stringify(req.body);
    }
    const backendRes = await fetch(targetUrl, fetchOpts);
    res.status(backendRes.status).set('Content-Type', backendRes.headers.get('content-type') || 'application/json');
    res.send(await backendRes.text());
  } catch (error: any) {
    console.error(`Market data proxy error: ${targetUrl}`, error.message);
    res.status(502).json({ error: 'Market data backend unavailable', detail: error.message });
  }
});

// --- Unified Feed API Proxy → FastAPI backend /api/feed/* ---
app.use('/api/feed', async (req, res) => {
  const targetUrl = `${BACKEND_URL}/api/feed${req.url || '/'}`;
  try {
    const fetchOpts: RequestInit = { method: req.method, headers: { 'Content-Type': 'application/json' } };
    if (req.method !== 'GET' && req.method !== 'HEAD' && req.body) {
      fetchOpts.body = JSON.stringify(req.body);
    }
    const backendRes = await fetch(targetUrl, fetchOpts);
    res.status(backendRes.status).set('Content-Type', backendRes.headers.get('content-type') || 'application/json');
    res.send(await backendRes.text());
  } catch (error: any) {
    console.error(`Feed proxy error: ${targetUrl}`, error.message);
    res.status(502).json({ error: 'Feed backend unavailable', detail: error.message });
  }
});

// --- Forecast API Proxy → FastAPI backend /api/forecast/* ---
app.use('/api/forecast', async (req, res) => {
  const targetUrl = `${BACKEND_URL}/api/forecast${req.url || '/'}`;
  try {
    const fetchOpts: RequestInit = { method: req.method, headers: { 'Content-Type': 'application/json' } };
    if (req.method !== 'GET' && req.method !== 'HEAD' && req.body) {
      fetchOpts.body = JSON.stringify(req.body);
    }
    const backendRes = await fetch(targetUrl, fetchOpts);
    res.status(backendRes.status).set('Content-Type', backendRes.headers.get('content-type') || 'application/json');
    res.send(await backendRes.text());
  } catch (error: any) {
    console.error(`Forecast proxy error: ${targetUrl}`, error.message);
    res.status(502).json({ error: 'Forecast backend unavailable', detail: error.message });
  }
});


app.use('/api/backend', async (req, res) => {
  const targetPath = req.url || '/';
  const targetUrl = `${BACKEND_URL}${targetPath}`;
  
  try {
    const headers: Record<string, string> = { 'Content-Type': 'application/json' };
    const apiKey = req.headers['x-api-key'];
    const authorization = req.headers.authorization;
    if (typeof apiKey === 'string') headers['X-API-Key'] = apiKey;
    if (authorization) headers.Authorization = authorization;
    const fetchOpts: RequestInit = {
      method: req.method,
      headers,
    };
    if (req.method !== 'GET' && req.method !== 'HEAD' && req.body) {
      fetchOpts.body = JSON.stringify(req.body);
    }
    const backendRes = await fetch(targetUrl, fetchOpts);
    const contentType = backendRes.headers.get('content-type') || 'application/json';
    res.status(backendRes.status).set('Content-Type', contentType);
    const data = await backendRes.text();
    res.send(data);
  } catch (error: any) {
    console.error(`Backend proxy error: ${targetUrl}`, error.message);
    res.status(502).json({ error: 'Backend unavailable', detail: error.message });
  }
});

// --- Vite Middleware ---
async function startServer() {
  const httpServer = http.createServer(app);

  if (process.env.NODE_ENV !== 'production') {
    const vite = await createViteServer({
      server: {
        middlewareMode: true,
        allowedHosts: true,
        hmr: {
          server: httpServer,
        },
      },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), 'dist');
    app.use(express.static(distPath));
    app.get('*', (req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }


  httpServer.on('error', (err: NodeJS.ErrnoException) => {
    if (err.code === 'EADDRINUSE') {
      console.error(`Port ${PORT} is already in use. Another frontend instance is likely running.`);
      console.error(`Set FRONTEND_PORT to a free port, or stop the existing process. Exiting cleanly.`);
      process.exit(1);
    }
    console.error('HTTP server error:', err);
    process.exit(1);
  });

  httpServer.listen(PORT, '0.0.0.0', () => {
    console.log(`Server running on http://localhost:${PORT}`);
  });
}

startServer();

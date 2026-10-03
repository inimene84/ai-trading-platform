import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import path from 'path';
import {defineConfig, loadEnv} from 'vite';

export default defineConfig(({mode}) => {
  const env = loadEnv(mode, '.', '');
  return {
    plugins: [react(), tailwindcss()],
    build: {
      rollupOptions: {
        output: {
          manualChunks(id) {
            if (id.includes('node_modules')) {
              if (id.includes('react') || id.includes('react-dom') || id.includes('scheduler')) {
                return 'vendor-react';
              }
              if (id.includes('@xyflow')) {
                return 'vendor-flow';
              }
              if (id.includes('motion')) {
                return 'vendor-motion';
              }
              if (id.includes('lucide-react')) {
                return 'vendor-icons';
              }
              if (id.includes('@google/genai')) {
                return 'vendor-genai';
              }
            }
          },
        },
      },
    },
    resolve: {
      alias: {
        '@': path.resolve(import.meta.dirname, '.'),
      },
    },
    server: {
      allowedHosts: true,
      // HMR is disabled in AI Studio via DISABLE_HMR env var.
      // Do not modify—file watching is disabled to prevent flickering during agent edits.
      hmr: process.env.DISABLE_HMR !== 'true',
      proxy: {
        // Proxy all /api/backend requests to the FastAPI backend
        '/api/backend': {
          target: env.BACKEND_URL || 'http://localhost:8000',
          changeOrigin: true,
          rewrite: (p) => p.replace(/^\/api\/backend/, ''),
        },
        // Direct /api/news and /api/market-data (same FastAPI backend)
        '/api/news': {
          target: env.BACKEND_URL || 'http://localhost:8000',
          changeOrigin: true,
        },
        '/api/market-data': {
          target: env.BACKEND_URL || 'http://localhost:8000',
          changeOrigin: true,
        },
        '/api/feed': {
          target: env.BACKEND_URL || 'http://localhost:8000',
          changeOrigin: true,
        },
        '/api/forecast': {
          target: env.BACKEND_URL || 'http://localhost:8000',
          changeOrigin: true,
        },
        '/api/historical': {
          target: env.BACKEND_URL || 'http://localhost:8000',
          changeOrigin: true,
        },
        '/api/telemetry': {
          target: env.BACKEND_URL || 'http://localhost:8000',
          changeOrigin: true,
        },
        '/api/signals': {
          target: env.BACKEND_URL || 'http://localhost:8000',
          changeOrigin: true,
        },
      },
    },
  };
});

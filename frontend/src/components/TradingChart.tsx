import React, { useEffect, useRef, useState } from 'react';
import {
  CandlestickSeries,
  ColorType,
  CrosshairMode,
  HistogramSeries,
  LineStyle,
  createChart,
  type CandlestickData,
  type HistogramData,
  type IChartApi,
  type ISeriesApi,
  type MouseEventParams,
  type Time,
  type UTCTimestamp,
} from 'lightweight-charts';

interface CandleBar {
  time: string | number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number;
}

interface TradingChartProps {
  data: CandleBar[];
}

type CandleSide = 'up' | 'down' | 'flat';

const UP = '#089981';
const DOWN = '#f23645';
const UP_VOLUME = 'rgba(8, 153, 129, 0.55)';
const DOWN_VOLUME = 'rgba(242, 54, 69, 0.55)';
const FLAT = '#94a3b8';
const CHART_BG = '#0b0e14';

interface OhlcReadout {
  time: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  side: CandleSide;
}

function candleSide(open: number, close: number): CandleSide {
  if (close > open) return 'up';
  if (close < open) return 'down';
  return 'flat';
}

function sideColor(side: CandleSide): string {
  switch (side) {
    case 'up':
      return UP;
    case 'down':
      return DOWN;
    case 'flat':
      return FLAT;
    default: {
      const exhaustive: never = side;
      return exhaustive;
    }
  }
}

function toUnixSeconds(value: string | number): number | null {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return value > 1e12 ? Math.floor(value / 1000) : Math.floor(value);
  }
  const parsed = Date.parse(String(value));
  if (!Number.isFinite(parsed)) return null;
  return Math.floor(parsed / 1000);
}

function formatPrice(price: number): string {
  const abs = Math.abs(price);
  const digits = abs >= 1000 ? 2 : abs >= 1 ? 4 : abs >= 0.01 ? 5 : 8;
  return price.toLocaleString('en-US', {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

function formatVolume(volume: number): string {
  if (volume >= 1_000_000_000) return `${(volume / 1_000_000_000).toFixed(2)}B`;
  if (volume >= 1_000_000) return `${(volume / 1_000_000).toFixed(2)}M`;
  if (volume >= 1_000) return `${(volume / 1_000).toFixed(1)}K`;
  return volume.toFixed(0);
}

function formatBarTime(time: Time): string {
  if (typeof time === 'number') {
    return new Date(time * 1000).toLocaleString(undefined, {
      month: 'short',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
    });
  }
  if (typeof time === 'string') return time;
  return `${time.year}-${String(time.month).padStart(2, '0')}-${String(time.day).padStart(2, '0')}`;
}

interface PreparedBars {
  candles: CandlestickData<Time>[];
  volume: HistogramData<Time>[];
}

function prepareBars(data: CandleBar[]): PreparedBars {
  const byTime = new Map<number, CandleBar>();
  for (const bar of data) {
    const time = toUnixSeconds(bar.time);
    const open = Number(bar.open);
    const high = Number(bar.high);
    const low = Number(bar.low);
    const close = Number(bar.close);
    if (time === null || time <= 0) continue;
    if (![open, high, low, close].every((value) => Number.isFinite(value))) continue;
    if (high < low) continue;
    byTime.set(time, {
      time,
      open,
      high: Math.max(high, open, close),
      low: Math.min(low, open, close),
      close,
      volume: Number.isFinite(Number(bar.volume)) ? Number(bar.volume) : 0,
    });
  }
  const ordered = [...byTime.entries()].sort((left, right) => left[0] - right[0]);
  const candles: CandlestickData<Time>[] = [];
  const volume: HistogramData<Time>[] = [];
  for (const [time, bar] of ordered) {
    const stamp = time as UTCTimestamp;
    const side = candleSide(bar.open, bar.close);
    candles.push({
      time: stamp,
      open: bar.open,
      high: bar.high,
      low: bar.low,
      close: bar.close,
    });
    volume.push({
      time: stamp,
      value: bar.volume ?? 0,
      color: side === 'down' ? DOWN_VOLUME : side === 'flat' ? 'rgba(148, 163, 184, 0.45)' : UP_VOLUME,
    });
  }
  return { candles, volume };
}

function readoutFrom(time: Time, bar: CandlestickData<Time>, volume: number): OhlcReadout {
  return {
    time: formatBarTime(time),
    open: bar.open,
    high: bar.high,
    low: bar.low,
    close: bar.close,
    volume,
    side: candleSide(bar.open, bar.close),
  };
}

export const TradingChart: React.FC<TradingChartProps> = ({ data }) => {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candleRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const volumeRef = useRef<ISeriesApi<'Histogram'> | null>(null);
  const volumeByTime = useRef<Map<number, number>>(new Map());
  const fittedFrom = useRef<number | null>(null);
  const [readout, setReadout] = useState<OhlcReadout | null>(null);
  const [empty, setEmpty] = useState(data.length === 0);

  useEffect(() => {
    const container = chartContainerRef.current;
    if (!container) return;

    const chart = createChart(container, {
      layout: {
        background: { type: ColorType.Solid, color: CHART_BG },
        textColor: '#94a3b8',
        fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
        fontSize: 11,
        panes: {
          separatorColor: '#1e293b',
          separatorHoverColor: '#334155',
          enableResize: true,
        },
      },
      grid: {
        vertLines: { color: 'rgba(148, 163, 184, 0.06)' },
        horzLines: { color: 'rgba(148, 163, 184, 0.06)' },
      },
      width: Math.max(1, container.clientWidth),
      height: Math.max(360, container.clientHeight),
      autoSize: false,
      timeScale: {
        borderColor: '#1e293b',
        timeVisible: true,
        secondsVisible: false,
        rightOffset: 6,
        barSpacing: 8,
        minBarSpacing: 3,
        fixLeftEdge: false,
        fixRightEdge: false,
      },
      rightPriceScale: {
        borderColor: '#1e293b',
        scaleMargins: { top: 0.08, bottom: 0.04 },
        entireTextOnly: true,
      },
      leftPriceScale: { visible: false },
      crosshair: {
        mode: CrosshairMode.Normal,
        vertLine: {
          color: '#64748b',
          width: 1,
          style: LineStyle.Dashed,
          labelBackgroundColor: '#0f172a',
        },
        horzLine: {
          color: '#64748b',
          width: 1,
          style: LineStyle.Dashed,
          labelBackgroundColor: '#0f172a',
        },
      },
      localization: {
        priceFormatter: formatPrice,
      },
      handleScroll: {
        mouseWheel: true,
        pressedMouseMove: true,
        horzTouchDrag: true,
        vertTouchDrag: false,
      },
      handleScale: {
        axisPressedMouseMove: true,
        mouseWheel: true,
        pinch: true,
      },
      kineticScroll: {
        touch: true,
        mouse: false,
      },
    });

    const candles = chart.addSeries(CandlestickSeries, {
      upColor: UP,
      downColor: DOWN,
      borderVisible: true,
      borderUpColor: UP,
      borderDownColor: DOWN,
      wickUpColor: UP,
      wickDownColor: DOWN,
      priceLineVisible: true,
      priceLineColor: '#e2e8f0',
      priceLineWidth: 1,
      lastValueVisible: true,
    });
    const volume = chart.addSeries(
      HistogramSeries,
      {
        priceFormat: { type: 'volume' },
        priceLineVisible: false,
        lastValueVisible: false,
      },
      1,
    );
    const volumePane = chart.panes()[1];
    if (volumePane) {
      volumePane.setHeight(Math.max(72, Math.round(Math.max(360, container.clientHeight) * 0.22)));
    }

    chartRef.current = chart;
    candleRef.current = candles;
    volumeRef.current = volume;

    const onCrosshair = (param: MouseEventParams<Time>) => {
      if (!param.time || !param.seriesData) {
        setReadout(null);
        return;
      }
      const row = param.seriesData.get(candles);
      if (!row || !('open' in row) || !('close' in row)) {
        setReadout(null);
        return;
      }
      const stamp = typeof param.time === 'number' ? param.time : 0;
      const barVolume = volumeByTime.current.get(stamp) ?? 0;
      setReadout(readoutFrom(param.time, row as CandlestickData<Time>, barVolume));
    };
    chart.subscribeCrosshairMove(onCrosshair);

    const observer = new ResizeObserver((entries) => {
      const rect = entries[0]?.contentRect;
      if (!rect || !chartRef.current) return;
      const height = Math.max(360, Math.floor(rect.height));
      chartRef.current.applyOptions({
        width: Math.max(1, Math.floor(rect.width)),
        height,
      });
      const pane = chartRef.current.panes()[1];
      if (pane) pane.setHeight(Math.max(64, Math.round(height * 0.22)));
    });
    observer.observe(container);

    return () => {
      chart.unsubscribeCrosshairMove(onCrosshair);
      observer.disconnect();
      chart.remove();
      chartRef.current = null;
      candleRef.current = null;
      volumeRef.current = null;
    };
  }, []);

  useEffect(() => {
    const candleSeries = candleRef.current;
    const volumeSeries = volumeRef.current;
    const chart = chartRef.current;
    if (!candleSeries || !volumeSeries || !chart) return;

    const prepared = prepareBars(data);
    setEmpty(prepared.candles.length === 0);
    volumeByTime.current = new Map(
      prepared.volume.map((bar) => [Number(bar.time), Number(bar.value)]),
    );
    try {
      candleSeries.setData(prepared.candles);
      volumeSeries.setData(prepared.volume);
      const firstTime = prepared.candles.length ? Number(prepared.candles[0].time) : null;
      if (firstTime !== null && fittedFrom.current !== firstTime && prepared.candles.length > 1) {
        chart.timeScale().fitContent();
        fittedFrom.current = firstTime;
      }
      const last = prepared.candles[prepared.candles.length - 1];
      const lastVolume = prepared.volume[prepared.volume.length - 1];
      if (last) {
        setReadout(readoutFrom(last.time, last, Number(lastVolume?.value ?? 0)));
      } else {
        setReadout(null);
      }
    } catch (err) {
      console.warn('[TradingChart] failed to apply candles', err);
    }
  }, [data]);

  const color = readout ? sideColor(readout.side) : FLAT;

  return (
    <div className="relative w-full h-full min-h-[420px] bg-[#0b0e14]">
      <div className="pointer-events-none absolute left-3 top-2 z-10 flex flex-wrap items-center gap-x-3 gap-y-1 font-mono text-[11px]">
        {readout ? (
          <>
            <span className="text-slate-500">{readout.time}</span>
            <span className="text-slate-400">
              O <span style={{ color }}>{formatPrice(readout.open)}</span>
            </span>
            <span className="text-slate-400">
              H <span style={{ color }}>{formatPrice(readout.high)}</span>
            </span>
            <span className="text-slate-400">
              L <span style={{ color }}>{formatPrice(readout.low)}</span>
            </span>
            <span className="text-slate-400">
              C <span style={{ color }}>{formatPrice(readout.close)}</span>
            </span>
            <span className="text-slate-500">Vol {formatVolume(readout.volume)}</span>
          </>
        ) : (
          <span className="text-slate-500">OHLC</span>
        )}
      </div>
      {empty && (
        <div className="pointer-events-none absolute inset-0 z-10 flex items-center justify-center text-xs font-mono text-slate-500">
          No candles
        </div>
      )}
      <div ref={chartContainerRef} className="w-full h-full min-h-[420px]" />
    </div>
  );
};

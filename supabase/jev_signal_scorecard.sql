-- Jev Ensemble scorecard. Filled by workflows/12_jev_outcome_filler_workflow.json.
-- One signal per symbol/timeframe/candle (latest run wins) so repeated test runs
-- in the same hour are not double counted.
create or replace view public.jev_signal_scorecard
with (security_invoker = true) as
with dedup as (
  select distinct on (symbol, timeframe, date_trunc('hour', created_at))
         symbol, timeframe, decision, confidence, next_candle_return, outcome, created_at
  from public.jev_signals
  order by symbol, timeframe, date_trunc('hour', created_at), created_at desc
)
select
  decision,
  count(*)                                                        as signals,
  count(*) filter (where outcome is not null and outcome <> 'NO_DATA') as scored,
  count(*) filter (where outcome = 'WIN')                         as wins,
  count(*) filter (where outcome = 'LOSS')                        as losses,
  count(*) filter (where outcome = 'SCRATCH')                     as scratches,
  round(100.0 * count(*) filter (where outcome = 'WIN')
        / nullif(count(*) filter (where outcome in ('WIN','LOSS')), 0), 1) as win_rate_pct,
  round(100 * avg(case when decision = 'LONG'  then next_candle_return
                       when decision = 'SHORT' then -next_candle_return end)::numeric, 3) as avg_signed_return_pct,
  count(*) filter (where outcome = 'SKIP_UP')                     as skipped_up,
  count(*) filter (where outcome = 'SKIP_DOWN')                   as skipped_down,
  count(*) filter (where outcome = 'SKIP_QUIET')                  as skipped_quiet,
  round(100 * avg(abs(next_candle_return))::numeric, 3)           as avg_abs_move_pct,
  round(avg(confidence)::numeric, 3)                              as avg_confidence
from dedup
group by decision
order by signals desc;

comment on view public.jev_signal_scorecard is
  'Per-decision hit rate of Jev Ensemble signals vs the next full candle (see workflow 12).';

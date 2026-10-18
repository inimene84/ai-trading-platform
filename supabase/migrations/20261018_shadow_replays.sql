-- Shadow Model Replay schema (idempotent)
-- Writes are service_role only. RLS stays enabled with no anon/authenticated
-- policies so the PostgREST anon key cannot INSERT/UPDATE/SELECT.

create table if not exists public.shadow_replays (
  id                  bigint generated always as identity primary key,
  signal_id           text not null,
  symbol              text not null,
  preset              text not null,
  shadow_model        text,
  shadow_direction    text,
  shadow_confidence   numeric,
  original_direction  text,
  entry_price         numeric,
  exit_price          numeric,
  forward_return_pct  numeric,
  correct             boolean,
  original_correct    boolean,
  settled             boolean not null default false,
  decision_ts         timestamptz not null,
  settle_ts           timestamptz,
  created_at          timestamptz not null default now(),
  constraint shadow_replays_signal_preset_uniq unique (signal_id, preset)
);

create index if not exists shadow_replays_unsettled_idx
  on public.shadow_replays (settled, decision_ts) where settled = false;

create index if not exists shadow_replays_preset_idx
  on public.shadow_replays (preset, settled);

alter table public.shadow_replays enable row level security;

drop policy if exists "service role only" on public.shadow_replays;
drop policy if exists "shadow_replays_service_or_anon_write" on public.shadow_replays;
-- Intentionally no policies: service_role bypasses RLS. Anon/authenticated have no access.

revoke all on table public.shadow_replays from anon, authenticated, public;

create or replace view public.shadow_replay_leaderboard as
select
  preset,
  shadow_model,
  count(*) as total_rows,
  count(*) filter (where correct is not null) as directional_calls,
  round(
    100.0 * count(*) filter (where shadow_direction not in ('neutral', ''))
    / nullif(count(*), 0),
    1
  ) as coverage_pct,
  round(avg(case when correct then 1.0 else 0.0 end) * 100.0, 1) as shadow_hit_rate_pct,
  round(avg(case when original_correct then 1.0 else 0.0 end) * 100.0, 1) as baseline_hit_rate_pct,
  round(avg(forward_return_pct)::numeric, 3) as avg_forward_return_pct
from public.shadow_replays
where settled = true
group by preset, shadow_model
order by shadow_hit_rate_pct desc nulls last;

revoke all on table public.shadow_replay_leaderboard from anon, authenticated, public;

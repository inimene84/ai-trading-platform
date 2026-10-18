-- Postmortem Failure Logger schema + Bear Verdict log (idempotent)
-- Writes are service_role only. RLS stays enabled with no anon/authenticated
-- policies so the PostgREST anon key cannot INSERT/UPDATE/SELECT.

create table if not exists public.postmortems (
  trade_id          text primary key,
  symbol            text not null,
  direction         text not null,
  broker            text,
  mode              text default 'paper',
  opened_at         timestamptz,
  closed_at         timestamptz,
  realized_pnl      numeric,
  total_fees        numeric default 0,
  exit_reason       text,
  cause             text not null,
  gates_at_entry    jsonb default '{}'::jsonb,
  llm_confidence    numeric,
  reflection        text,
  reflection_model  text,
  created_at        timestamptz not null default now()
);

create index if not exists postmortems_cause_idx  on public.postmortems (cause);
create index if not exists postmortems_symbol_idx on public.postmortems (symbol);
create index if not exists postmortems_mode_idx   on public.postmortems (mode);

alter table public.postmortems enable row level security;

drop policy if exists "service role only" on public.postmortems;
drop policy if exists "postmortems_service_or_anon_write" on public.postmortems;
-- Intentionally no policies: service_role bypasses RLS. Anon/authenticated have no access.

revoke all on table public.postmortems from anon, authenticated, public;

create or replace view public.postmortem_cause_summary as
select
  mode,
  cause,
  count(*) as trades,
  round(sum(coalesce(realized_pnl, 0))::numeric, 2) as total_pnl,
  round(avg(coalesce(realized_pnl, 0))::numeric, 2) as avg_pnl,
  round(sum(coalesce(total_fees, 0))::numeric, 2) as total_fees
from public.postmortems
group by mode, cause
order by total_pnl asc;

revoke all on table public.postmortem_cause_summary from anon, authenticated, public;

-- Bear-veto debate log table so bear hit-rate is measurable in shadow & enforce modes
create table if not exists public.bear_verdicts (
  id                bigint generated always as identity primary key,
  symbol            text not null,
  direction         text not null,
  mode              text not null default 'shadow',
  veto              boolean not null default false,
  veto_effective    boolean not null default false,
  abstain           boolean not null default false,
  bear_strength     numeric,
  reason            text,
  resolved_model    text,
  resolved_provider text,
  latency_ms        numeric,
  entry_price       numeric,
  confidence        numeric,
  regime            text,
  bull_case         text,
  bear_case         text,
  created_at        timestamptz not null default now()
);

create index if not exists bear_verdicts_symbol_idx on public.bear_verdicts (symbol, created_at desc);
create index if not exists bear_verdicts_veto_idx   on public.bear_verdicts (veto, mode);

alter table public.bear_verdicts enable row level security;

drop policy if exists "bear_verdicts_service_or_anon_write" on public.bear_verdicts;
-- Intentionally no policies: service_role bypasses RLS. Anon/authenticated have no access.

revoke all on table public.bear_verdicts from anon, authenticated, public;

-- EconoLens AI persistent app-usage counter
-- Author: Rishabh Shah
-- Advisor: Dr. Qingyang Xiao
--
-- Run this script once in the Supabase SQL Editor.
-- The app calls the RPC function and never needs direct table access.

create table if not exists public.app_usage (
    id integer primary key check (id = 1),
    total_uses bigint not null default 0,
    last_used_at timestamptz not null default now()
);

insert into public.app_usage (id, total_uses, last_used_at)
values (1, 0, now())
on conflict (id) do nothing;

alter table public.app_usage enable row level security;

-- Do not expose the counter table directly to anonymous users.
revoke all on table public.app_usage from anon;
revoke all on table public.app_usage from authenticated;

create or replace function public.increment_econolens_usage()
returns bigint
language plpgsql
security definer
set search_path = public
as $$
declare
    new_total bigint;
begin
    insert into public.app_usage (id, total_uses, last_used_at)
    values (1, 1, now())
    on conflict (id) do update
    set total_uses = public.app_usage.total_uses + 1,
        last_used_at = now()
    returning total_uses into new_total;

    return new_total;
end;
$$;

revoke all on function public.increment_econolens_usage() from public;
grant execute on function public.increment_econolens_usage() to anon;
grant execute on function public.increment_econolens_usage() to authenticated;

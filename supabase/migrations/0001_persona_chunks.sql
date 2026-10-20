-- Persona Lab — stockage vectoriel des chunks de corpus persona (V1)
--
-- Embeddings : Voyage AI voyage-4, 1024 dimensions, calculés à partir de embeddingText.
-- Accès : réservé au rôle service_role (clé secrète Supabase, côté serveur uniquement),
--         limité à SELECT, INSERT et UPDATE : aucune suppression possible via les clés API.
-- Aucune policy RLS : anon et authenticated n'ont aucun accès.
-- Pas d'index HNSW en V1 : recherche exacte, suffisante pour quelques dizaines de chunks.

begin;

create extension if not exists vector with schema extensions;

-- ---------------------------------------------------------------------------
-- Table
-- ---------------------------------------------------------------------------

create table public.persona_chunks (
  persona_id text not null,
  id         text not null,
  content    text not null,
  embedding  extensions.vector(1024) not null,
  metadata   jsonb not null,
  updated_at timestamptz not null default now(),

  constraint persona_chunks_pkey primary key (persona_id, id),
  constraint persona_chunks_persona_id_format check (persona_id ~ '^[a-z0-9_-]+$'),
  constraint persona_chunks_id_format check (id ~ '^[A-Za-z0-9_-]+$'),
  constraint persona_chunks_content_not_empty check (length(btrim(content)) > 0),
  constraint persona_chunks_metadata_is_object check (jsonb_typeof(metadata) = 'object'),
  constraint persona_chunks_metadata_persona_matches check (metadata->>'personaId' = persona_id),
  constraint persona_chunks_metadata_has_source check (metadata ? 'sourceId')
);

comment on table public.persona_chunks is
  'Chunks des corpus persona (Persona Lab). Lecture, insertion et mise à jour réservées à service_role ; aucune suppression via les clés API.';
comment on column public.persona_chunks.id is
  'Identifiant stable du chunk dans le corpus du persona (ex. D04-q4), utilisé pour les citations.';
comment on column public.persona_chunks.content is
  'Texte original du chunk, identique aux lignes source (chunk.text).';
comment on column public.persona_chunks.embedding is
  'Embedding voyage-4 (1024 dimensions, input_type document) de metadata.embeddingText.';
comment on column public.persona_chunks.metadata is
  'Métadonnées du pipeline : provenance, statut épistémique, traçabilité, avertissements, paramètres d''embedding.';

-- updated_at reflète la dernière écriture, y compris lors d'un upsert.
create function public.persona_chunks_set_updated_at()
returns trigger
language plpgsql
set search_path = ''
as $$
begin
  new.updated_at := now();
  return new;
end;
$$;

create trigger persona_chunks_set_updated_at
  before update on public.persona_chunks
  for each row
  execute function public.persona_chunks_set_updated_at();

-- ---------------------------------------------------------------------------
-- Sécurité
-- ---------------------------------------------------------------------------

alter table public.persona_chunks enable row level security;
-- Volontairement aucune policy : seuls les rôles qui contournent la RLS (service_role) y accèdent.

-- Supabase accorde par défaut tous les droits (y compris TRUNCATE) sur les nouveaux objets
-- de public à anon, authenticated et service_role : on repart de zéro, puis on n'accorde
-- à service_role que le strict nécessaire au script d'indexation (upsert) et à la recherche.
-- V1 : aucun droit DELETE ni TRUNCATE. La clé serveur ne peut supprimer aucun chunk ;
-- le retrait de chunks obsolètes reste une opération manuelle d'administration.
revoke all on table public.persona_chunks from public, anon, authenticated, service_role;
grant select, insert, update on table public.persona_chunks to service_role;

-- Fonction de trigger : appelée par Postgres, aucun rôle n'a besoin de l'exécuter directement.
revoke all on function public.persona_chunks_set_updated_at() from public, anon, authenticated, service_role;

-- ---------------------------------------------------------------------------
-- Recherche par similarité cosinus
-- ---------------------------------------------------------------------------

create function public.match_persona_chunks(
  p_persona_id text,
  p_query_embedding extensions.vector(1024),
  p_match_count integer default 5
)
returns table (
  id text,
  content text,
  metadata jsonb,
  similarity double precision
)
language sql
stable
security invoker
set search_path = ''
as $$
  select
    c.id,
    c.content,
    c.metadata,
    1 - (c.embedding operator(extensions.<=>) p_query_embedding) as similarity
  from public.persona_chunks as c
  where c.persona_id = p_persona_id
  order by c.embedding operator(extensions.<=>) p_query_embedding, c.id
  limit least(greatest(coalesce(p_match_count, 5), 1), 50);
$$;

comment on function public.match_persona_chunks(text, extensions.vector, integer) is
  'Top-k des chunks d''un persona par similarité cosinus (1 - distance). k borné entre 1 et 50. Réservé à service_role.';

-- Postgres accorde EXECUTE à PUBLIC par défaut, et Supabase à anon/authenticated.
revoke all on function public.match_persona_chunks(text, extensions.vector, integer) from public, anon, authenticated;
grant execute on function public.match_persona_chunks(text, extensions.vector, integer) to service_role;

commit;

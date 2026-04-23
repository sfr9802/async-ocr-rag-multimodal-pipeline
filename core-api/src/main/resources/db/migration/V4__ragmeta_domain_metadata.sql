-- Phase 9: domain/category/language metadata for filterable retrieval.
--
-- The retriever's ParsedQuery.filters is a free-text dict but the post-
-- filter only honors a small whitelist (domain / category / language).
-- Backing those keys with first-class columns lets us index them and
-- gives the LLM query parser a stable enum to populate.
--
-- Note: ragmeta.documents.category already exists (V2). This migration
-- adds the two missing columns (domain, language) and the per-column
-- indexes we'll need once the enterprise + anime corpora share an index.
-- The existing category index from V2 is left as-is.
--
-- Backfill: every row predating Phase 9 was an anime-corpus English
-- doc. Mark them so post-Phase-9 filters that target {domain:anime,
-- language:en} still return them after a rebuild-free upgrade.

ALTER TABLE ragmeta.documents ADD COLUMN IF NOT EXISTS domain   VARCHAR(32);
ALTER TABLE ragmeta.documents ADD COLUMN IF NOT EXISTS language VARCHAR(8);

CREATE INDEX IF NOT EXISTS ragmeta_documents_domain_idx
    ON ragmeta.documents (domain);
CREATE INDEX IF NOT EXISTS ragmeta_documents_language_idx
    ON ragmeta.documents (language);

UPDATE ragmeta.documents
   SET domain   = COALESCE(domain,   'anime'),
       language = COALESCE(language, 'en')
 WHERE domain IS NULL
    OR language IS NULL;

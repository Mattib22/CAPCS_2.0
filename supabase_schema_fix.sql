-- ============================================================
-- Schema fix: rounds table — one row per product round (bias cycle)
-- Run this in the Supabase SQL editor.
-- ============================================================

DROP TABLE IF EXISTS rounds;

CREATE TABLE rounds (
    id               uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id       uuid REFERENCES sessions(id) ON DELETE CASCADE,
    user_key         text NOT NULL,
    round_number     integer,          -- sequential within session: 1, 2, 3...
    bias             text,             -- bias identified in this cycle
    explanation      text,             -- plain-English explanation of the bias
    perspective      text,             -- alternative option proposed by CASPER
    accepted         boolean DEFAULT false,  -- true = user accepted the counterattack
    confidence_before integer,         -- clarity % at start of this cycle
    confidence_after  integer,         -- clarity % after user responded
    confidence_shift  integer DEFAULT 0,
    initial_leaning   text,            -- user's stated leaning at start of cycle
    listening_qa      jsonb DEFAULT '[]'  -- [{question, answer, leaning}] for replay
);

CREATE INDEX rounds_user_key_idx  ON rounds (user_key);
CREATE INDEX rounds_session_id_idx ON rounds (session_id);

-- ============================================================
-- Schema fix: sessions — add what_shifted column
-- ============================================================

ALTER TABLE sessions ADD COLUMN IF NOT EXISTS what_shifted text;

-- ============================================================
-- Schema fix: profile_history — add saved_at if missing
-- ============================================================

ALTER TABLE profile_history ADD COLUMN IF NOT EXISTS saved_at text;

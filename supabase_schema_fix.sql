-- ============================================================
-- Schema fix: rounds table (recreate with correct columns)
-- Run this in the Supabase SQL editor.
-- ============================================================

-- 1. Drop and recreate rounds with all expected columns
DROP TABLE IF EXISTS rounds;

CREATE TABLE rounds (
    id               uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id       uuid REFERENCES sessions(id) ON DELETE CASCADE,
    user_key         text NOT NULL,
    round_number     integer,
    round_state      text,
    timestamp        text,
    bias             text,
    explanation      text,
    perspective      text,
    question         text,
    conversation_message text,
    followups        jsonb DEFAULT '[]',
    answer           text,
    answer_depth     text,
    answer_emotion   text,
    answer_certainty text,
    answer_key_signal text,
    how_shifted      text,
    shifted          boolean DEFAULT false,
    leaning          text,
    confidence       integer,
    shift            integer DEFAULT 0,
    confidence_shift integer DEFAULT 0
);

-- Index for the load_log query
CREATE INDEX rounds_user_key_idx ON rounds (user_key);
CREATE INDEX rounds_session_id_idx ON rounds (session_id);

-- ============================================================
-- Schema fix: profile_history table (add saved_at if missing)
-- ============================================================

-- Option A: add the missing column to an existing table
ALTER TABLE profile_history
    ADD COLUMN IF NOT EXISTS saved_at text;

-- If profile_history doesn't exist at all, run this instead of Option A:
-- CREATE TABLE profile_history (
--     id               uuid PRIMARY KEY DEFAULT gen_random_uuid(),
--     user_key         text NOT NULL,
--     age_range        text,
--     education_level  text,
--     education_field  text,
--     values           text,
--     passions         text,
--     current_situation text,
--     current_job      text,
--     main_constraint  text,
--     who_is_affected  text,
--     decision_style   text,
--     known_bias       text,
--     success_criteria text,
--     version          text,
--     completed_at     text,
--     saved_at         text
-- );
-- CREATE INDEX profile_history_user_key_idx ON profile_history (user_key);

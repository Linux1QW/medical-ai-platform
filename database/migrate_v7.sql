-- Migration v7: Add max_rounds column to consultations
ALTER TABLE consultations ADD COLUMN max_rounds INT DEFAULT 20;

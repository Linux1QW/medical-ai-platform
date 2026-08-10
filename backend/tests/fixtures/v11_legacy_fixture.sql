-- V1.1 Legacy Migration Fixture
-- Contains edge cases for migration validation:
-- 1. Completed evaluation with valid run_id
-- 2. Evaluation without run_id (needs backfill)
-- 3. Consultation that can be uniquely matched
-- 4. Orphan evaluation (no matching consultation)
-- 5. Duplicate candidate conflicts
--
-- Note: Actual schema must exist before running this fixture.
-- The fixture tests the data safety checks in the migration.
-- Run this AFTER baseline schema (pre-v1.1) is created.

-- ============================================================
-- Scenario: Legacy data that should pass migration prechecks
-- ============================================================

-- Case 1: Valid consultation + evaluation with run_id (no conflict)
INSERT INTO consultations (id, patient_id, doctor_id, status, created_at)
VALUES (1001, 1, 1, 'completed', NOW());

INSERT INTO evaluation_runs (id, consultation_id, graph_version, scoring_policy_version, checkpoint_thread_id, status, created_at, updated_at)
VALUES ('run-uuid-0001', 1001, 'evaluation-graph-v1', 'v1', 'thread-001', 'completed', NOW(), NOW());

INSERT INTO evaluations (
    consultation_id, inquiry_score, inquiry_analysis,
    knowledge_score, knowledge_analysis,
    humanistic_score, humanistic_analysis,
    diagnosis_score, diagnosis_analysis,
    treatment_score, treatment_analysis,
    total_score, overall_summary, improvement_suggestions,
    run_id, created_at
) VALUES (
    1001, 85.0, 'Good history taking',
    80.0, 'Adequate knowledge',
    90.0, 'Excellent communication',
    85.0, 'Accurate diagnosis',
    80.0, 'Appropriate treatment',
    84.0, 'Overall good performance', 'Continue improving differential diagnosis',
    'run-uuid-0001', NOW()
);

-- Case 2: Evaluation without run_id (needs backfill before v1.1 migration)
INSERT INTO consultations (id, patient_id, doctor_id, status, created_at)
VALUES (1002, 2, 1, 'completed', NOW());

INSERT INTO evaluations (
    consultation_id, inquiry_score, inquiry_analysis,
    knowledge_score, knowledge_analysis,
    humanistic_score, humanistic_analysis,
    diagnosis_score, diagnosis_analysis,
    treatment_score, treatment_analysis,
    total_score, overall_summary, improvement_suggestions,
    run_id, created_at
) VALUES (
    1002, 70.0, 'Missing some key questions',
    NULL, 'Needs improvement',
    75.0, 'Could be more empathetic',
    70.0, 'Missed differential',
    65.0, 'Suboptimal treatment plan',
    70.0, 'Below average performance', 'Focus on systematic approach',
    NULL, NOW()
);

-- Case 3: Another valid evaluation with unique run_id
INSERT INTO consultations (id, patient_id, doctor_id, status, created_at)
VALUES (1003, 3, 2, 'completed', NOW());

INSERT INTO evaluation_runs (id, consultation_id, graph_version, scoring_policy_version, checkpoint_thread_id, status, created_at, updated_at)
VALUES ('run-uuid-0003', 1003, 'evaluation-graph-v1', 'v1', 'thread-003', 'completed', NOW(), NOW());

INSERT INTO evaluations (
    consultation_id, inquiry_score, inquiry_analysis,
    knowledge_score, knowledge_analysis,
    humanistic_score, humanistic_analysis,
    diagnosis_score, diagnosis_analysis,
    treatment_score, treatment_analysis,
    total_score, overall_summary, improvement_suggestions,
    run_id, created_at
) VALUES (
    1003, 95.0, 'Excellent history taking',
    90.0, 'Strong knowledge base',
    95.0, 'Very empathetic',
    90.0, 'Thorough differential',
    95.0, 'Evidence-based treatment',
    93.0, 'Excellent performance', 'Maintain current standards',
    'run-uuid-0003', NOW()
);

-- ============================================================
-- NEGATIVE CASE: Duplicate run_id (should BLOCK migration)
-- Uncomment to test that migration correctly rejects duplicates
-- ============================================================
-- INSERT INTO evaluations (
--     consultation_id, inquiry_score, inquiry_analysis,
--     humanistic_score, humanistic_analysis,
--     diagnosis_score, diagnosis_analysis,
--     treatment_score, treatment_analysis,
--     overall_summary, improvement_suggestions,
--     run_id, created_at
-- ) VALUES (
--     1004, 50.0, 'Poor',
--     50.0, 'Poor',
--     50.0, 'Poor',
--     50.0, 'Poor',
--     'Bad', 'Needs work',
--     'run-uuid-0001', NOW()  -- DUPLICATE run_id!
-- );

-- ============================================================
-- NEGATIVE CASE: Orphan run_id (should BLOCK FK creation)
-- Uncomment to test that migration correctly rejects orphans
-- ============================================================
-- INSERT INTO evaluations (
--     consultation_id, inquiry_score, inquiry_analysis,
--     humanistic_score, humanistic_analysis,
--     diagnosis_score, diagnosis_analysis,
--     treatment_score, treatment_analysis,
--     overall_summary, improvement_suggestions,
--     run_id, created_at
-- ) VALUES (
--     1005, 60.0, 'Fair',
--     60.0, 'Fair',
--     60.0, 'Fair',
--     60.0, 'Fair',
--     'Fair', 'Needs improvement',
--     'run-uuid-nonexistent', NOW()  -- ORPHAN run_id!
-- );

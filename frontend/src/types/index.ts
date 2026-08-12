export interface User {
  id: number;
  username: string;
  email: string;
  real_name: string;
  role: 'doctor' | 'admin';
  department: string;
  avatar: string;
  created_at: string;
}

export interface Token {
  access_token: string;
  token_type: string;
  user: User;
}

export interface VirtualPatient {
  id: number;
  name: string;
  age: number;
  gender: 'male' | 'female';
  personality_type: '配合型' | '焦虑型' | '沉默型' | '对抗型';
  chief_complaint: string;
  medical_history: string;
  symptoms: string;
  expected_diagnosis?: string;
  system_prompt?: string;
  difficulty_level: number;
  created_at: string;
}

export interface Consultation {
  id: number;
  doctor_id: number;
  patient_id: number;
  patient_name?: string;
  personality_type?: string;
  doctor_username?: string;
  status: 'in_progress' | 'completed' | 'evaluated';
  started_at: string;
  ended_at: string | null;
  total_score?: number;
  duration_minutes?: number;
  summary: string;
  diagnosis: string;
  treatment_plan: string;
  max_rounds?: number;
  created_at: string;
}

export interface Message {
  id: number;
  consultation_id: number;
  role: 'doctor' | 'patient';
  content: string;
  sequence: number;
  created_at: string;
}

export interface ConsultationDetail extends Consultation {
  messages: Message[];
}

export interface Citation {
  citation_id: string;
  claim: string;
  source: string;
  page?: number | null;
  heading_path: string;
  text_snippet: string;
  rerank_score?: number | null;
}

export interface Evaluation {
  id: number;
  consultation_id: number;
  inquiry_score: number;
  inquiry_analysis: string;
  knowledge_score?: number | null;
  knowledge_analysis: string;
  humanistic_score: number;
  humanistic_analysis: string;
  diagnosis_score: number;
  diagnosis_analysis: string;
  treatment_score: number;
  treatment_analysis: string;
  total_score?: number | null;
  overall_summary: string;
  improvement_suggestions: string;
  created_at: string;
  citation_data?: Citation[] | null;
  rag_trace_data?: Record<string, unknown> | null;
  retrieval_status: 'not_run' | 'sufficient' | 'insufficient' | 'unavailable' | 'error';
  evidence_stance: 'supports' | 'contradicts' | 'mixed' | 'undetermined';
  human_review_needed: boolean;
  review_reason?: string | null;
  evaluation_status: 'completed' | 'needs_review';
}

export interface UserStatItem {
  user_id: number;
  username: string;
  real_name: string;
  department: string;
  total_consultations: number;
  total_evaluations: number;
  avg_inquiry_score: number;
  avg_knowledge_score: number;
  avg_humanistic_score: number;
  avg_diagnosis_score: number;
  avg_treatment_score: number;
  avg_total_score: number;
}

export interface StatsSummary {
  total_consultations: number;
  total_evaluations: number;
  avg_inquiry_score: number;
  avg_knowledge_score: number;
  avg_humanistic_score: number;
  avg_diagnosis_score: number;
  avg_treatment_score: number;
  avg_total_score: number;
  score_distribution: { range: string; count: number }[];
  user_stats?: UserStatItem[];
}

// ── Task 9: 异步评估任务状态机 ─────────────────────────────────────

/** 评估任务运行状态 */
export type EvaluationJobStatus =
  | 'queued'
  | 'running'
  | 'completed'
  | 'reviewed'
  | 'needs_review'
  | 'failed'
  | 'cancelled';

/** POST 创建评估后的响应（仅排队，不等待完成） */
export interface EvaluationSubmit {
  run_id: string;
  websocket_url: string;
  status: EvaluationJobStatus;
  consultation_id: number;
}

/** 评估 run 轮询状态 */
export interface EvaluationRunStatus {
  run_id: string;
  status: EvaluationJobStatus;
  progress: number;
  message: string;
  cancel_requested?: boolean;
  error_code?: string | null;
}

/** 评估锁状态 */
export interface EvaluationLockStatus {
  consultation_id: number;
  status: string | null;
  run_id: string | null;
  is_active: boolean;
  locked_at: string | null;
  expires_at: string | null;
}

/** 评估取消响应 */
export interface EvaluationCancelResponse {
  consultation_id: number;
  status: string;
  cancel_requested?: boolean;
  previous_lock_status: string | null;
}

/** WS 进度事件 */
export interface EvaluationProgressEvent {
  type: 'auth_ok' | 'progress' | 'latest' | 'error' | 'done';
  run_id?: string;
  sequence?: number;
  progress: number;
  message: string;
  error_code?: string;
}

// ── Task 12: 评估报告前端升级 ─ 新增类型 ──────────────────────────────────

/** Rubric 判定结果 */
export type RubricVerdict = 'pass' | 'partial' | 'fail' | 'not_applicable' | 'unassessed';

/** 原子 Rubric 项 */
export interface RubricItem {
  item_id: string;
  dimension: string;
  verdict: RubricVerdict;
  score: number | null;
  severity: 'high' | 'medium' | 'low';
  description: string;
  evidence_spans: string[];
  citation_ids: string[];
}

/** 证据链接类型 */
export type EvidenceLinkType = 'supports' | 'contradicts' | 'insufficient';

/** 临床主张状态 */
export type ClaimStatus = 'supported' | 'partially_supported' | 'unsupported' | 'conflicting';

/** 临床主张 */
export interface ClinicalClaim {
  claim_id: string;
  claim_type: 'diagnosis' | 'treatment' | 'risk' | 'education';
  content: string;
  status: ClaimStatus;
  evidence_links: EvidenceLink[];
  needs_review: boolean;
}

/** 证据链接 */
export interface EvidenceLink {
  citation_id: string;
  link_type: EvidenceLinkType;
  entailment_score: number;
  evidence_span: string;
}

/** 风险类型 */
export type RiskType = 'emergency' | 'medication' | 'population' | 'privacy' | 'evidence_conflict';

/** 风险发现 */
export interface RiskFinding {
  finding_id: string;
  risk_type: RiskType;
  severity: 'high' | 'medium' | 'low';
  description: string;
  evidence_span: string;
  policy_action: string;
  needs_review: boolean;
}

// ── Task 10: 管理员复核工作台类型 ─────────────────────────────────────

/** 待复核列表项 */
export interface PendingReviewItem {
  evaluation_id: number;
  consultation_id: number;
  doctor_username: string;
  patient_name: string;
  review_reason: string;
  human_review_needed: boolean;
  created_at: string;
}

/** 待复核列表响应 */
export interface PendingReviewList {
  items: PendingReviewItem[];
  total: number;
  limit: number;
  offset: number;
}

/** 复核状态 */
export interface ReviewStatus {
  evaluation_id: number;
  status: 'pending' | 'in_review' | 'completed';
  reviewer_id: number | null;
  reviewed_at: string | null;
}

/** 五维分数调整 */
export interface ScoreAdjustments {
  inquiry_score?: number | null;
  knowledge_score?: number | null;
  humanistic_score?: number | null;
  diagnosis_score?: number | null;
  treatment_score?: number | null;
}

/** 复核提交请求体 */
export interface ReviewSubmission {
  feedback: string;
  score_adjustments?: ScoreAdjustments;
}

/** 复核提交响应 */
export interface ReviewSubmitResult {
  success: boolean;
  evaluation_id: number;
}

/** 报告类型 */
export type ReportKind = 'smoke' | 'regression' | 'benchmark' | 'legacy_unknown';

/** 报告清单 */
export interface ReportManifest {
  report_kind: ReportKind;
  report_id: string;
  created_at: string;
  case_count: number;
  dataset_version: string;
  model_version: string;
  prompt_version: string;
  judge_version: string;
  kb_version: string;
  scoring_policy_version: string;
  seed: number;
}

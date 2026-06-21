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
  expected_diagnosis: string;
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
  retrieval_status: string;
  evidence_stance: string;
  human_review_needed: boolean;
  review_reason?: string | null;
  evaluation_status: string;
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

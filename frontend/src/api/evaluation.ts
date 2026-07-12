import request from '../utils/request';
import type { Evaluation, StatsSummary } from '../types';

export const createEvaluation = (consultation_id: number): Promise<Evaluation> =>
  request.post('/evaluations/', { consultation_id });

export const getEvaluation = (consultation_id: number): Promise<Evaluation> =>
  request.get(`/evaluations/${consultation_id}`);

export const getStats = (): Promise<StatsSummary> => request.get('/stats/');

// 查询评估锁状态
export const getEvaluationLockStatus = (consultationId: number) =>
  request.get(`/evaluations/${consultationId}/lock-status`) as Promise<{
    consultation_id: number;
    status: string | null;
    run_id: string | null;
    is_active: boolean;
    locked_at: string | null;
    expires_at: string | null;
  }>;

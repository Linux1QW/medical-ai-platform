import request from '../utils/request';
import type { Evaluation, EvaluationSubmit, EvaluationRunStatus, EvaluationLockStatus, EvaluationCancelResponse, StatsSummary } from '../types';

/** 创建评估（仅排队，60s 超时） */
export const createEvaluation = (consultation_id: number): Promise<EvaluationSubmit> =>
  request.post('/evaluations/', { consultation_id }, { timeout: 60000 });

export const getEvaluation = (consultation_id: number): Promise<Evaluation> =>
  request.get(`/evaluations/${consultation_id}`);

export const getStats = (): Promise<StatsSummary> => request.get('/stats/');

/** 查询评估锁状态 */
export const getEvaluationLockStatus = (consultationId: number): Promise<EvaluationLockStatus> =>
  request.get(`/evaluations/${consultationId}/lock-status`);

/** 查询评估 run 状态（轮询用） */
export const getEvaluationRunStatus = (runId: string): Promise<EvaluationRunStatus> =>
  request.get(`/evaluations/runs/${runId}/status`);

/** 取消评估 run */
export const cancelEvaluationRun = (runId: string): Promise<EvaluationCancelResponse> =>
  request.post(`/evaluations/runs/${runId}/cancel`);

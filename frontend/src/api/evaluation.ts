import request from '../utils/request';
import type { Evaluation, StatsSummary } from '../types';

export const createEvaluation = (consultation_id: number): Promise<Evaluation> =>
  request.post('/evaluations/', { consultation_id });

export const getEvaluation = (consultation_id: number): Promise<Evaluation> =>
  request.get(`/evaluations/${consultation_id}`);

export const getStats = (): Promise<StatsSummary> => request.get('/stats/');

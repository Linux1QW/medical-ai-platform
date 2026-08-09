import request from '../utils/request';
import type { PendingReviewList, ReviewStatus, ReviewSubmission, ReviewSubmitResult } from '../types';

export const listPendingReviews = (limit = 50, offset = 0): Promise<PendingReviewList> =>
  request.get('/reviews/pending', { params: { limit, offset } });

export const getReviewStatus = (evaluationId: number): Promise<ReviewStatus> =>
  request.get(`/reviews/${evaluationId}/status`);

export const submitReview = (
  evaluationId: number,
  payload: ReviewSubmission,
): Promise<ReviewSubmitResult> => request.post(`/reviews/${evaluationId}/submit`, payload);

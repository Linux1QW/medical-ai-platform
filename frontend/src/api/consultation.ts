import request from '../utils/request';
import type { Consultation, ConsultationDetail, Message } from '../types';

export const startConsultation = (patient_id: number): Promise<Consultation> =>
  request.post('/consultations/', { patient_id });

export const getConsultations = (): Promise<Consultation[]> =>
  request.get('/consultations/');

export interface ConsultationQueryParams {
  username?: string;
  personality?: string;
  start_time?: string;
  end_time?: string;
}

export const getAllConsultations = (params?: ConsultationQueryParams): Promise<Consultation[]> =>
  request.get('/consultations/all', { params });

export const getConsultationDetail = (id: number): Promise<ConsultationDetail> =>
  request.get(`/consultations/${id}`);

export const sendMessage = (consultation_id: number, content: string): Promise<Message[]> =>
  request.post(`/consultations/${consultation_id}/messages`, { content });

export const submitDiagnosis = (
  consultation_id: number,
  diagnosis: string,
  treatment_plan: string,
): Promise<Consultation> =>
  request.post(`/consultations/${consultation_id}/submit-diagnosis`, {
    diagnosis,
    treatment_plan,
  });

export const endConsultation = (id: number): Promise<Consultation> =>
  request.post(`/consultations/${id}/end`);

export const extendRounds = (id: number): Promise<Consultation> =>
  request.post(`/consultations/${id}/extend`);

export const deleteConsultation = (id: number): Promise<{ detail: string }> =>
  request.delete(`/consultations/${id}`);

import request from '../utils/request';
import type { VirtualPatient } from '../types';

export const getPatients = (params?: {
  personality_type?: string;
  difficulty_level?: number;
}): Promise<VirtualPatient[]> => request.get('/patients/', { params });

export const getPatient = (id: number): Promise<VirtualPatient> =>
  request.get(`/patients/${id}`);

export const createPatient = (data: Partial<VirtualPatient>): Promise<VirtualPatient> =>
  request.post('/patients/', data);

export const updatePatient = (id: number, data: Partial<VirtualPatient>): Promise<VirtualPatient> =>
  request.put(`/patients/${id}`, data);

export const deletePatient = (id: number): Promise<void> =>
  request.delete(`/patients/${id}`);

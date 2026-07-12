import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { ConfigProvider, Spin } from 'antd';
import zhCN from 'antd/locale/zh_CN';
import React, { Suspense } from 'react';

import MainLayout from './layouts/MainLayout';
import { useAuth } from './store/useAuth';

const LoginPage = React.lazy(() => import('./pages/Login'));
const RegisterPage = React.lazy(() => import('./pages/Register'));
const DashboardPage = React.lazy(() => import('./pages/Dashboard'));
const PatientListPage = React.lazy(() => import('./pages/PatientList'));
const ConsultationListPage = React.lazy(() => import('./pages/ConsultationList'));
const ConsultationPage = React.lazy(() => import('./pages/Consultation'));
const EvaluationPage = React.lazy(() => import('./pages/Evaluation'));
const AdminStatsPage = React.lazy(() => import('./pages/AdminStats'));
const AdminPatientsPage = React.lazy(() => import('./pages/AdminPatients'));
const AdminConsultationsPage = React.lazy(() => import('./pages/AdminConsultations'));
const ProfilePage = React.lazy(() => import('./pages/Profile'));

const PageFallback = (
  <div style={{ display: 'flex', justifyContent: 'center', padding: 100 }}>
    <Spin size="large" />
  </div>
);

function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const { isLoggedIn } = useAuth();
  return isLoggedIn ? <>{children}</> : <Navigate to="/login" />;
}

function AdminRoute({ children }: { children: React.ReactNode }) {
  const { isLoggedIn, isAdmin } = useAuth();
  if (!isLoggedIn) return <Navigate to="/login" />;
  if (!isAdmin) return <Navigate to="/dashboard" />;
  return <>{children}</>;
}

function App() {
  return (
    <ConfigProvider locale={zhCN} theme={{ token: { colorPrimary: '#4f46e5', borderRadius: 8 } }}>
      <BrowserRouter>
        <Suspense fallback={PageFallback}>
          <Routes>
            <Route path="/login" element={<LoginPage />} />
            <Route path="/register" element={<RegisterPage />} />
            <Route
              path="/"
              element={
                <ProtectedRoute>
                  <MainLayout />
                </ProtectedRoute>
              }
            >
              <Route index element={<Navigate to="/dashboard" />} />
              <Route path="dashboard" element={<DashboardPage />} />
              <Route path="patients" element={<PatientListPage />} />
              <Route path="consultations" element={<ConsultationListPage />} />
              <Route path="consultation/:id" element={<ConsultationPage />} />
              <Route path="evaluation/:id" element={<EvaluationPage />} />
              <Route path="stats" element={<AdminStatsPage />} />
              <Route path="admin" element={<Navigate to="/stats" replace />} />
              <Route
                path="admin/consultations"
                element={
                  <AdminRoute>
                    <AdminConsultationsPage />
                  </AdminRoute>
                }
              />
              <Route path="profile" element={<ProfilePage />} />
              <Route
                path="admin/patients"
                element={
                  <AdminRoute>
                    <AdminPatientsPage />
                  </AdminRoute>
                }
              />
            </Route>
            <Route path="*" element={<Navigate to="/dashboard" />} />
          </Routes>
        </Suspense>
      </BrowserRouter>
    </ConfigProvider>
  );
}

export default App;

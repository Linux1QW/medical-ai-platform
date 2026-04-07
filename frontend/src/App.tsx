import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { ConfigProvider } from 'antd';
import zhCN from 'antd/locale/zh_CN';

import MainLayout from './layouts/MainLayout';
import LoginPage from './pages/Login';
import RegisterPage from './pages/Register';
import DashboardPage from './pages/Dashboard';
import PatientListPage from './pages/PatientList';
import ConsultationListPage from './pages/ConsultationList';
import ConsultationPage from './pages/Consultation';
import EvaluationPage from './pages/Evaluation';
import AdminStatsPage from './pages/AdminStats';
import AdminPatientsPage from './pages/AdminPatients';
import AdminConsultationsPage from './pages/AdminConsultations';
import ProfilePage from './pages/Profile';
import { useAuth } from './store/useAuth';

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
            <Route
              path="admin"
              element={
                <AdminRoute>
                  <AdminStatsPage />
                </AdminRoute>
              }
            />
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
      </BrowserRouter>
    </ConfigProvider>
  );
}

export default App;

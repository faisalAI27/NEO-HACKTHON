import { useState } from 'react';
import { ThemeProvider, CssBaseline } from '@mui/material';
import theme from './theme';
import LandingPage from './pages/LandingPage';
import DashboardPage from './pages/DashboardPage';

type Page = 'landing' | 'dashboard';

function App() {
  const [currentPage, setCurrentPage] = useState<Page>('landing');

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      {currentPage === 'landing' ? (
        <LandingPage onGetStarted={() => setCurrentPage('dashboard')} />
      ) : (
        <DashboardPage onBack={() => setCurrentPage('landing')} />
      )}
    </ThemeProvider>
  );
}

export default App;

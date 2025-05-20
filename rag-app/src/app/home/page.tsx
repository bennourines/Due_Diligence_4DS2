'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { 
  Box, 
  Typography, 
  Grid, 
  Container, 
  CircularProgress 
} from '@mui/material';
import ChatIcon from '@mui/icons-material/Chat';
import DescriptionIcon from '@mui/icons-material/Description';
import AnalyticsIcon from '@mui/icons-material/Analytics';
import ShowChartIcon from '@mui/icons-material/ShowChart';

// Import components
import Header from '../components/Header';
import Footer from '../components/Footer';
import FeatureCard from '../components/FeatureCard';
import WelcomeSection from '../components/WelcomeSection';
import GuestWelcome from '../components/GuestWelcome';
    
export default function HomePage() {
  const router = useRouter();
  const [isLoading, setIsLoading] = useState(true);
  const [username, setUsername] = useState('User');
  const [isAuthenticated, setIsAuthenticated] = useState(false);

  useEffect(() => {
    // Check authentication status and get user info
    const checkAuth = () => {
      try {
        const user = localStorage.getItem('user');
        
        if (user) {
          const userData = JSON.parse(user);
          setUsername(userData.username || userData.name || 'User');
          setIsAuthenticated(true);
        } else {
          setIsAuthenticated(false);
        }
      } catch (error) {
        console.error('Error parsing user data:', error);
        setIsAuthenticated(false);
      } finally {
        setIsLoading(false);
      }
    };

    checkAuth();
  }, []);

  const handleLogout = () => {
    // Clear user data from localStorage
    localStorage.removeItem('user');
    
    // Update authentication state
    setIsAuthenticated(false);
    
    // Redirect to login
    router.push('/login');
  };

  const handleLoginClick = () => {
    router.push('/login');
  };

  if (isLoading) {
    return (
      <Box
        sx={{
          height: '100vh',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          bgcolor: 'var(--black-color)',
          color: 'var(--yellow-color)',
        }}
      >
        <CircularProgress sx={{ color: 'var(--yellow-color)', mb: 2 }} />
        <Typography>Loading...</Typography>
      </Box>
    );
  }

  return (
    <Box
      sx={{
        minHeight: '100vh',
        display: 'flex',
        flexDirection: 'column',
        bgcolor: 'var(--black-color)',
      }}
    >
      <Header onLogout={handleLogout} isAuthenticated={isAuthenticated} />
      
      <Container maxWidth="lg" sx={{ mt: 4, mb: 4, flexGrow: 1 }}>
        {isAuthenticated ? (
          <>
            {/* Welcome Section for authenticated users */}
            <WelcomeSection username={username} />
            
            {/* Features Section */}
            <Typography variant="h4" gutterBottom sx={{ color: 'var(--yellow-color)', mb: 3 }}>
              Features
            </Typography>
          <Grid container spacing={4}>
            {/* First row */}
            <Grid item xs={12} sm={6} md={6} lg={6}>
              <FeatureCard 
                title="Chat with Documents" 
                description="Upload whitepapers, project documents, and analyze them instantly. Get specific answers about cryptocurrencies directly from their documentation."
                icon={<ChatIcon sx={{ fontSize: 60, color: 'var(--yellow-color)' }} />}
                action={() => router.push('/chat')}
                buttonText="Try Now"
              />
            </Grid>
            <Grid item xs={12} sm={6} md={6} lg={6}>
              <FeatureCard 
                title="Due Diligence Report" 
                description="Generate comprehensive cryptocurrency due diligence reports including tokenomics, team background, security assessments, and market positioning." 
                icon={<DescriptionIcon sx={{ fontSize: 60, color: 'var(--yellow-color)' }} />}
                action={() => router.push('/duedel')}
                buttonText="Explore"
              />
            </Grid>
            
            {/* Second row */}
            <Grid item xs={12} sm={6} md={6} lg={6}>
              <FeatureCard 
                title="Forecast Report" 
                description="Access detailed price prediction reports with technical analysis, trend identification, and market sentiment indicators for any cryptocurrency."
                icon={<AnalyticsIcon sx={{ fontSize: 60, color: 'var(--yellow-color)' }} />}
                action={() => router.push('/forecast')}
                buttonText="Generate Report"
              />
            </Grid>
            <Grid item xs={12} sm={6} md={6} lg={6}>
              <FeatureCard 
                title="Coin History" 
                description="Visualize historical performance with interactive charts featuring key technical indicators, support/resistance levels, and trading signals."
                icon={<ShowChartIcon sx={{ fontSize: 60, color: 'var(--yellow-color)' }} />}
                action={() => router.push('/analyze')}
                buttonText="View Charts"
              />
            </Grid>
          </Grid>
          </>
        ) : (
          <>
            {/* Welcome Section for guests */}
            <GuestWelcome />
            
            {/* Features Preview for guests */}
            <Typography variant="h4" gutterBottom sx={{ color: 'var(--yellow-color)', mb: 3 }}>
              Features
            </Typography>
            <Grid container spacing={4}>
              {/* First row */}
              <Grid item xs={12} sm={6} md={6} lg={6}>
                <FeatureCard 
                  title="Chat with Documents" 
                  description="Upload whitepapers, project documents, and analyze them instantly. Get specific answers about cryptocurrencies directly from their documentation."
                  icon={<ChatIcon sx={{ fontSize: 60, color: 'var(--yellow-color)' }} />}
                  action={handleLoginClick}
                  buttonText="Sign In to Try"
                />
              </Grid>
              <Grid item xs={12} sm={6} md={6} lg={6}>
                <FeatureCard 
                  title="Due Diligence Report" 
                  description="Generate comprehensive cryptocurrency due diligence reports including tokenomics, team background, security assessments, and market positioning." 
                  icon={<DescriptionIcon sx={{ fontSize: 60, color: 'var(--yellow-color)' }} />}
                  action={handleLoginClick}
                  buttonText="Sign In to Explore"
                />
              </Grid>
              
              {/* Second row */}
              <Grid item xs={12} sm={6} md={6} lg={6}>
                <FeatureCard 
                  title="Forecast Report" 
                  description="Access detailed price prediction reports with technical analysis, trend identification, and market sentiment indicators for any cryptocurrency."
                  icon={<AnalyticsIcon sx={{ fontSize: 60, color: 'var(--yellow-color)' }} />}
                  action={handleLoginClick}
                  buttonText="Sign In to Generate"
                />
              </Grid>
              <Grid item xs={12} sm={6} md={6} lg={6}>
                <FeatureCard 
                  title="Coin History" 
                  description="Visualize historical performance with interactive charts featuring key technical indicators, support/resistance levels, and trading signals."
                  icon={<ShowChartIcon sx={{ fontSize: 60, color: 'var(--yellow-color)' }} />}
                  action={handleLoginClick}
                  buttonText="Sign In to View"
                />
              </Grid>
            </Grid>
          </>
        )}
      </Container>
      
      <Footer />
    </Box> 
  );
}
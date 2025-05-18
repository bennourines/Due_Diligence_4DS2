'use client';

import React from 'react';
import { Box, Container } from '@mui/material';
import UserReportsList from '../DeuDelComponents/UserReportsList';
import Footer from '../components/Footer';
import Head from '../components/Head';

const ReportsPage: React.FC = () => {
  return (
    <Box
      sx={{
        minHeight: '100vh',
        display: 'flex',
        flexDirection: 'column',
        bgcolor: 'var(--black-color)',
      }}
    >
      <Head />
      <Container maxWidth="lg" sx={{ mt: 4, mb: 4, flexGrow: 1 }}>
        <div className="relative">
          {/* Neural network background */}
          <div className="grid-background"></div>
          
          {/* Accent dots */}
          <div 
            className="accent-dot" 
            style={{ 
              top: '10%', 
              left: '5%',
              position: 'absolute',
              width: '150px',
              height: '150px',
              borderRadius: '50%',
              opacity: 0.3,
              zIndex: -1,
              background: 'var(--gradient-glow)'
            }}
          ></div>
          
          <div 
            className="accent-dot" 
            style={{ 
              bottom: '20%', 
              right: '10%',
              position: 'absolute',
              width: '150px',
              height: '150px',
              borderRadius: '50%',
              opacity: 0.3,
              zIndex: -1,
              background: 'var(--gradient-glow)'
            }}
          ></div>
          
          <div className="flex flex-col min-h-screen bg-deep p-6">
            <header className="mb-8">
              <h1 className="text-2xl font-bold">Your Due Diligence Reports</h1>
              <h3 className="text-sm text-secondary mt-2">View and download your cryptocurrency reports</h3>
            </header>
            
            <UserReportsList />
          </div>
        </div>
      </Container>
      <Footer />
    </Box>
  );
};

export default ReportsPage;
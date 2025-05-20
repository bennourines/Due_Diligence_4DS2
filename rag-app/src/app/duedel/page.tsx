// page.tsx
'use client';

import React from 'react';
import { Box, Container } from '@mui/material';
import { DueDiligenceProvider } from '../context/DueDiligenceContext';
import DueDiligenceSteps from '../DeuDelComponents/DueDiligenceSteps';
import Footer from '../components/Footer';
import Head from '../components/Head';


const DueDiligencePage: React.FC = () => {
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
          
          <DueDiligenceProvider>
            <DueDiligenceSteps />
          </DueDiligenceProvider>
        </div>
      </Container>
       <Footer />
    </Box>
  );
};

export default DueDiligencePage;
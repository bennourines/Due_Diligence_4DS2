'use client';

import React from 'react';
import { Box, Typography, Container, Button } from '@mui/material';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import { useRouter } from 'next/navigation';
import DashboardEmbed from '../components/DashboardEmbed';

export default function ForecastPage() {
  const router = useRouter();
  
  // URL to your hosted Dash app
  // You'll need to replace this with the actual URL where your Dash app is running
  const dashboardUrl ='http://localhost:8050';

  return (
    <Box
      sx={{
        minHeight: '100vh',
        display: 'flex',
        flexDirection: 'column',
        bgcolor: 'var(--bg-deep)',
      }}
    >
      <Container maxWidth="xl" sx={{ mt: 4, mb: 4, flexGrow: 1 }}>
        <Box 
          sx={{ 
            display: 'flex', 
            alignItems: 'center', 
            justifyContent: 'space-between',
            mb: 3 
          }}
        >
          <Button 
            startIcon={<ArrowBackIcon />} 
            onClick={() => router.back()}
            sx={{
              color: 'var(--text-primary)',
              '&:hover': {
                backgroundColor: 'rgba(79, 70, 229, 0.1)',
              }
            }}
          >
            Back
          </Button>
          <Typography 
            variant="h4" 
            component="h1" 
            sx={{ 
              color: 'var(--text-primary)',
              fontFamily: 'var(--font-primary)',
              letterSpacing: '2px',
              textAlign: 'center',
              position: 'relative',
              '&:after': {
                content: '""',
                position: 'absolute',
                bottom: '-8px',
                left: '50%',
                transform: 'translateX(-50%)',
                width: '60px',
                height: '3px',
                background: 'var(--gradient-primary)',
                borderRadius: '2px',
              }
            }}
          >
            Cryptocurrency history analysis
          </Typography>
          <Box sx={{ width: 100 }} /> {/* Spacer for alignment */}
        </Box>
        
        <Box sx={{ mt: 4 }}>
          <DashboardEmbed 
            dashboardUrl={dashboardUrl} 
            title="Cryptocurrency Trading Signals Dashboard"
            height="85vh"
          />
        </Box>
      </Container>
    </Box>
  );
}
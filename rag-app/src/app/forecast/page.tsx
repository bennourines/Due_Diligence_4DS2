'use client';

import React, { useState } from 'react';
import { 
  Box, 
  Typography, 
  Container, 
  TextField, 
  Button, 
  CircularProgress,
  Alert,
  Paper,
  Grid,
  Divider,
  IconButton
} from '@mui/material';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import DownloadIcon from '@mui/icons-material/Download';
import SearchIcon from '@mui/icons-material/Search';
import AnalyticsIcon from '@mui/icons-material/Analytics';
import AutoGraphIcon from '@mui/icons-material/AutoGraph';
import { useRouter } from 'next/navigation';
import Footer from '../components/Footer';
import Head from '../components/Head';

// Define types for API responses
interface CryptoAnalysisResponse {
  forecast_presentation?: string;
  analysis_presentation?: string;
  merged_presentation?: string;
  error?: string;
}

export default function ForecastReportPage() {
  const router = useRouter();
  const [coin, setCoin] = useState<string>('');
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<CryptoAnalysisResponse | null>(null);
  const [progressStatus, setProgressStatus] = useState<string>('');
  
  // API endpoint for crypto analysis
  const API_ENDPOINT = 'http://localhost:8001';

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!coin.trim()) {
      setError('Please enter a cryptocurrency name');
      return;
    }
    
    setIsLoading(true);
    setError(null);
    setResult(null);
    setProgressStatus('Initiating analysis...');
    
    try {
      // Status updates to simulate progress
      const statusUpdates = [
        'Fetching market data...',
        'Running technical analysis...',
        'Generating forecast charts...',
        'Creating analysis presentation...',
        'Creating forecast presentation...',
        'Merging presentations...',
        'Finalizing report...'
      ];
      
      // Start the status update cycle
      let statusIndex = 0;
      const statusInterval = setInterval(() => {
        if (statusIndex < statusUpdates.length) {
          setProgressStatus(statusUpdates[statusIndex]);
          statusIndex++;
        } else {
          clearInterval(statusInterval);
        }
      }, 3000);
      
      // Make the API request
      const response = await fetch(`${API_ENDPOINT}/analyze-crypto/`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ coin: coin.trim() }),
      });
      
      // Clear the status interval once we get a response
      clearInterval(statusInterval);
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to analyze cryptocurrency');
      }
      
      const data = await response.json();
      setResult(data);
      setProgressStatus('Analysis complete!');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An unknown error occurred');
      setProgressStatus('Analysis failed');
    } finally {
      setIsLoading(false);
    }
  };
  
  const handleDownload = async () => {
    if (!result?.merged_presentation) return;
    
    try {
      setProgressStatus('Preparing download...');
      
      // Request the file from the API
      const response = await fetch(`${API_ENDPOINT}/download-presentation/?path=${encodeURIComponent(result.merged_presentation)}`);
      
      if (!response.ok) {
        throw new Error('Failed to download presentation');
      }
      
      // Get the filename from the path
      const filename = result.merged_presentation.split('/').pop() || 'combined_report.pptx';
      
      // Convert the response to a blob and download it
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
      
      setProgressStatus('Download complete!');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to download presentation');
    }
  };

  return (
    <Box
      sx={{
        minHeight: '100vh',
        display: 'flex',
        flexDirection: 'column',
        bgcolor: 'var(--bg-deep)',
        position: 'relative',
        overflow: 'hidden',
      }}
    >
      {/* Neural network background */}
      <div className="grid-background"></div>
      
      {/* Accent dots */}
      <div className="accent-dot" style={{ top: '10%', left: '5%' }}></div>
      <div className="accent-dot" style={{ top: '70%', left: '80%' }}></div>
      <div className="accent-dot" style={{ top: '40%', left: '90%' }}></div>
      <Head/>
      {/* Header */}
      <Box
        sx={{
          p: 2,
          backdropFilter: 'blur(10px)',
          borderBottom: 'var(--border-light)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          zIndex: 2,
        }}
      >
        <Button 
          startIcon={<ArrowBackIcon />} 
          onClick={() => router.push('/')}
          sx={{
            color: 'var(--text-primary)',
            '&:hover': {
              backgroundColor: 'rgba(79, 70, 229, 0.1)',
            }
          }}
        >
          Back to Home
        </Button>
        
        <Typography 
          variant="h5" 
          component="h1" 
          sx={{ 
            color: 'var(--text-primary)',
            fontFamily: 'var(--font-primary)',
            letterSpacing: '2px',
            fontWeight: 600,
          }}
        >
          Cryptocurrency Forecast Report
        </Typography>
        
        <Box sx={{ width: 100 }} /> {/* Spacer for alignment */}
      </Box>
      
      {/* Main content */}
      <Container 
        maxWidth="lg" 
        sx={{ 
          mt: 6, 
          mb: 4, 
          flexGrow: 1,
          display: 'flex',
          flexDirection: 'column',
          zIndex: 1,
        }}
      >
        <Paper 
          className="container" 
          elevation={0}
          sx={{ 
            p: 4, 
            mb: 4,
            backgroundColor: 'rgba(30, 41, 59, 0.7)',
          }}
        >
          <Grid container spacing={4}>
            <Grid item xs={12} md={6}>
              <Typography variant="h5" gutterBottom sx={{ color: 'var(--text-primary)' }}>
                Generate Cryptocurrency Analysis
              </Typography>
              <Typography sx={{ color: 'var(--text-secondary)', mb: 3 }}>
                Enter the name of a cryptocurrency to generate a comprehensive analysis and forecast report. Our system will analyze market data, technical indicators, and trends to create a detailed PowerPoint presentation.
              </Typography>
              
              <Box component="form" onSubmit={handleSubmit} sx={{ mt: 2 }}>
                <TextField
                  fullWidth
                  label="Cryptocurrency Name"
                  variant="outlined"
                  value={coin}
                  onChange={(e) => setCoin(e.target.value)}
                  placeholder="Enter cryptocurrency name (e.g., Bitcoin, Ethereum, Solana)"
                  disabled={isLoading}
                  sx={{
                    mb: 3,
                    '& .MuiOutlinedInput-root': {
                      '& fieldset': {
                        borderColor: 'var(--primary-color)',
                      },
                      '&:hover fieldset': {
                        borderColor: 'var(--primary-light)',
                      },
                      '&.Mui-focused fieldset': {
                        borderColor: 'var(--primary-light)',
                      },
                    },
                    '& .MuiInputLabel-root': {
                      color: 'var(--text-secondary)',
                    },
                    '& .MuiInputBase-input': {
                      color: 'var(--text-primary)',
                    },
                  }}
                />
                
                <Button
                  type="submit"
                  variant="contained"
                  startIcon={<AnalyticsIcon />}
                  disabled={isLoading || !coin.trim()}
                  sx={{
                    backgroundColor: 'var(--primary-color)',
                    color: 'var(--text-primary)',
                    py: 1.5,
                    px: 4,
                    '&:hover': {
                      backgroundColor: 'var(--primary-dark)',
                    },
                    '&.Mui-disabled': {
                      backgroundColor: 'rgba(79, 70, 229, 0.3)',
                      color: 'var(--text-secondary)',
                    },
                  }}
                >
                  {isLoading ? 'Analyzing...' : 'Generate Report'}
                </Button>
              </Box>
            </Grid>
            
            <Grid item xs={12} md={6} sx={{ position: 'relative' }}>
              <Box 
                sx={{ 
                  height: '100%',
                  display: 'flex',
                  flexDirection: 'column',
                  justifyContent: 'center',
                  position: 'relative',
                  pl: { md: 4 },
                  borderLeft: { md: 'var(--border-light)' },
                }}
              >
                <AutoGraphIcon 
                  sx={{ 
                    fontSize: 120, 
                    color: 'var(--primary-glow)',
                    position: 'absolute',
                    opacity: 0.4,
                    right: '5%',
                    top: '10%',
                    zIndex: 0,
                  }}
                />
                
                <Typography variant="h6" gutterBottom sx={{ color: 'var(--text-primary)', position: 'relative', zIndex: 1 }}>
                  What You'll Get:
                </Typography>
                
                <Box sx={{ mb: 2, position: 'relative', zIndex: 1 }}>
                  <Typography component="div" sx={{ color: 'var(--text-primary)', mb: 0.5, fontSize: '1rem' }}>
                    • Technical Analysis Indicators
                  </Typography>
                  <Typography component="div" sx={{ color: 'var(--text-secondary)', ml: 2, mb: 1, fontSize: '0.9rem' }}>
                    RSI, MACD, Moving Averages, and more
                  </Typography>
                  
                  <Typography component="div" sx={{ color: 'var(--text-primary)', mb: 0.5, fontSize: '1rem' }}>
                    • Price Trend Analysis
                  </Typography>
                  <Typography component="div" sx={{ color: 'var(--text-secondary)', ml: 2, mb: 1, fontSize: '0.9rem' }}>
                    Support/resistance levels and pattern identification
                  </Typography>
                  
                  <Typography component="div" sx={{ color: 'var(--text-primary)', mb: 0.5, fontSize: '1rem' }}>
                    • Future Price Forecasts
                  </Typography>
                  <Typography component="div" sx={{ color: 'var(--text-secondary)', ml: 2, mb: 1, fontSize: '0.9rem' }}>
                    Short and long-term price projections
                  </Typography>
                  
                  <Typography component="div" sx={{ color: 'var(--text-primary)', mb: 0.5, fontSize: '1rem' }}>
                    • Downloadable PowerPoint Presentation
                  </Typography>
                  <Typography component="div" sx={{ color: 'var(--text-secondary)', ml: 2, fontSize: '0.9rem' }}>
                    Complete report ready for sharing or presenting
                  </Typography>
                </Box>
              </Box>
            </Grid>
          </Grid>
        </Paper>
        
        {/* Loading and Status */}
        {isLoading && (
          <Paper 
            className="container pulse-element" 
            elevation={0}
            sx={{ 
              p: 4, 
              mb: 4,
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              backgroundColor: 'rgba(30, 41, 59, 0.7)',
              minHeight: '200px',
            }}
          >
            <div className="loader" style={{ marginBottom: '24px' }} />
            <Typography variant="h6" sx={{ color: 'var(--text-primary)', mb: 1 }}>
              {progressStatus}
            </Typography>
            <Typography sx={{ color: 'var(--text-secondary)', textAlign: 'center' }}>
              This process may take several minutes. We're analyzing market data and generating a comprehensive report.
            </Typography>
          </Paper>
        )}
        
        {/* Error message */}
        {error && (
          <Alert 
            severity="error" 
            sx={{ 
              mb: 4,
              backgroundColor: 'rgba(211, 47, 47, 0.2)',
              color: '#f48fb1',
              border: '1px solid rgba(211, 47, 47, 0.3)',
            }}
          >
            {error}
          </Alert>
        )}
        
        {/* Results */}
        {result && !isLoading && (
          <Paper 
            className="container" 
            elevation={0}
            sx={{ 
              p: 4, 
              mb: 4,
              backgroundColor: 'rgba(30, 41, 59, 0.7)',
            }}
          >
            <Typography variant="h5" gutterBottom sx={{ color: 'var(--text-primary)' }}>
              Analysis Complete
            </Typography>
            
            <Box sx={{ mb: 3 }}>
              <Typography sx={{ color: 'var(--text-secondary)', mb: 3 }}>
                Your cryptocurrency analysis for <span style={{ color: 'var(--primary-light)', fontWeight: 'bold' }}>{coin}</span> is ready. You can now download the complete presentation.
              </Typography>
              
              <Divider sx={{ borderColor: 'rgba(79, 70, 229, 0.2)', my: 3 }} />
              
              <Grid container spacing={2} alignItems="center">
                <Grid item xs={12} sm={8}>
                  <Typography sx={{ color: 'var(--text-primary)', fontWeight: 'bold' }}>
                    Combined Report:
                  </Typography>
                  <Typography sx={{ color: 'var(--text-secondary)', fontStyle: 'italic', fontSize: '0.9rem' }}>
                    {result.merged_presentation}
                  </Typography>
                </Grid>
                <Grid item xs={12} sm={4} sx={{ textAlign: 'right' }}>
                  <Button
                    variant="contained"
                    startIcon={<DownloadIcon />}
                    onClick={handleDownload}
                    disabled={!result.merged_presentation}
                    sx={{
                      backgroundColor: 'var(--primary-color)',
                      color: 'var(--text-primary)',
                      py: 1.5,
                      '&:hover': {
                        backgroundColor: 'var(--primary-dark)',
                      },
                    }}
                  >
                    Download Presentation
                  </Button>
                </Grid>
              </Grid>
            </Box>
          </Paper>
        )}
        
      </Container>
      <Footer />
    </Box>
    
  );
}
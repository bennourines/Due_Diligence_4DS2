'use client';

import { Typography, Button, Paper } from '@mui/material';
import LoginIcon from '@mui/icons-material/Login';
import PersonAddIcon from '@mui/icons-material/PersonAdd';

const GuestWelcome = () => {
  const handleLoginClick = () => {
    // Force complete page reload with login URL
    window.location.replace('/login');
  };
  
  const handleRegisterClick = () => {
    // Force complete page reload with register URL
    window.location.replace('/register');
  };
  
  return (
    <Paper 
      elevation={3}
      sx={{
        p: { xs: 3, md: 6 },
        mb: 4,
        bgcolor: 'rgba(0, 0, 0, 0.7)',
        color: 'var(--yellow-color)',
        borderRadius: 2,
        border: '1px solid rgba(255, 215, 0, 0.3)',
      }}
    >
      <Typography variant="h3" gutterBottom>
        Welcome to RAG Assistant!
      </Typography>
      <Typography variant="h5" paragraph>
        Your intelligent document assistant powered by AI.
      </Typography>
      <Typography variant="body1" paragraph>
        Sign in to access our Retrieval-Augmented Generation system that provides accurate, context-aware responses based on your documents. Upload files, ask questions, and get intelligent answers.
      </Typography>
      <Button 
        variant="contained"
        size="large"
        startIcon={<LoginIcon />}
        onClick={handleLoginClick}
        sx={{
          mt: 2,
          mr: 2,
          bgcolor: 'var(--yellow-color)',
          color: 'var(--black-color)',
          '&:hover': {
            bgcolor: 'rgba(255, 215, 0, 0.8)',
          }
        }}
      >
        Login
      </Button>
      <Button 
        variant="outlined"
        size="large"
        startIcon={<PersonAddIcon />}
        onClick={handleRegisterClick}
        sx={{
          mt: 2,
          color: 'var(--yellow-color)',
          borderColor: 'var(--yellow-color)',
          '&:hover': {
            bgcolor: 'rgba(255, 215, 0, 0.1)',
            borderColor: 'var(--yellow-color)',
          }
        }}
      >
        Create Account
      </Button>
    </Paper>
  );
};

export default GuestWelcome;
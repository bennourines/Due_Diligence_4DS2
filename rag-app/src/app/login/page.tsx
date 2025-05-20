'use client';

import { useState } from 'react';
import { Box, TextField, Button, Paper, Typography } from '@mui/material';
import { useRouter } from 'next/navigation';

export default function Login() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const router = useRouter();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');

    try {
      const response = await fetch('/api/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          email,
          password,
        }),
      });

      const data = await response.json();

      if (response.ok) {
        // Store user data in localStorage if needed
        localStorage.setItem('user', JSON.stringify(data.user));
        // Change this line from '/chat' to '/home' to redirect to the home page
        router.push('/home');
      } else {
        setError(data.message || 'Login failed');
      }
    } catch (error) {
      setError('An error occurred during login');
      console.error('Login error:', error);
    }
  };

  return (
    <Box
      sx={{
        height: '100vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        bgcolor: 'var(--black-color)',
      }}
    >
      <Paper
        elevation={3}
        component="form"
        onSubmit={handleSubmit}
        sx={{
          p: 4,
          width: '100%',
          maxWidth: 400,
          bgcolor: 'var(--black-color)',
          border: '1px solid var(--yellow-color)',
        }}
      >
        <Typography
          variant="h4"
          sx={{
            mb: 4,
            color: 'var(--yellow-color)',
            textAlign: 'center',
          }}
        >
          Login
        </Typography>

        {error && (
          <Typography
            sx={{
              color: 'error.main',
              mb: 2,
              textAlign: 'center',
            }}
          >
            {error}
          </Typography>
        )}

        <TextField
          fullWidth
          label="Email"
          variant="outlined"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          sx={{
            mb: 2,
            '& .MuiOutlinedInput-root': {
              color: 'var(--yellow-color)',
              '& fieldset': {
                borderColor: 'var(--yellow-color)',
              },
              '&:hover fieldset': {
                borderColor: 'var(--yellow-color)',
              },
              '&.Mui-focused fieldset': {
                borderColor: 'var(--yellow-color)',
              },
            },
            '& .MuiInputLabel-root': {
              color: 'var(--yellow-color)',
            },
          }}
        />

        <TextField
          fullWidth
          label="Password"
          type="password"
          variant="outlined"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          sx={{
            mb: 3,
            '& .MuiOutlinedInput-root': {
              color: 'var(--yellow-color)',
              '& fieldset': {
                borderColor: 'var(--yellow-color)',
              },
              '&:hover fieldset': {
                borderColor: 'var(--yellow-color)',
              },
              '&.Mui-focused fieldset': {
                borderColor: 'var(--yellow-color)',
              },
            },
            '& .MuiInputLabel-root': {
              color: 'var(--yellow-color)',
            },
          }}
        />

        <Button
          type="submit"
          fullWidth
          variant="contained"
          sx={{
            bgcolor: 'var(--yellow-color)',
            color: 'var(--black-color)',
            mb: 2,
            '&:hover': {
              bgcolor: 'var(--black-color)',
              color: 'var(--yellow-color)',
              border: '1px solid var(--yellow-color)',
            },
          }}
        >
          Login
        </Button>
        <Button
          fullWidth
          variant="outlined"
          onClick={() => router.push('/register')}
          sx={{
            color: 'var(--yellow-color)',
            borderColor: 'var(--yellow-color)',
            '&:hover': {
              bgcolor: 'var(--yellow-color)',
              color: 'var(--black-color)',
              borderColor: 'var(--yellow-color)',
            },
          }}
        >
          Create Account
        </Button>
      </Paper>
    </Box>
  );
}
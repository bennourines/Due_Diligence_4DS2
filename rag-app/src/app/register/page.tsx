'use client';

import { useState } from 'react';
import { 
  Box, 
  TextField, 
  Button, 
  Paper, 
  Typography, 
  FormControl, 
  InputLabel, 
  Select, 
  MenuItem,
  Grid,
  FormHelperText,
  SelectChangeEvent
} from '@mui/material';
import { useRouter } from 'next/navigation';
import { AdapterDateFns } from '@mui/x-date-pickers/AdapterDateFns';
import { LocalizationProvider, DatePicker } from '@mui/x-date-pickers';

// Define types for form data
interface FormData {
  firstName: string;
  lastName: string;
  username: string;
  email: string;
  password: string;
  confirmPassword: string;
  dateOfBirth: Date | null;
  role: string;
  address: string;
  phoneNumber: string;
}

export default function Register() {
  const [formData, setFormData] = useState<FormData>({
    firstName: '',
    lastName: '',
    username: '',
    email: '',
    password: '',
    confirmPassword: '',
    dateOfBirth: null,
    role: '',
    address: '',
    phoneNumber: ''
  });
  const [error, setError] = useState<string>('');
  const router = useRouter();

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement> | SelectChangeEvent) => {
    const { name, value } = e.target;
    setFormData({
      ...formData,
      [name]: value
    });
  };

  const handleDateChange = (date: Date | null) => {
    setFormData({
      ...formData,
      dateOfBirth: date
    });
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');

    // Validation
    if (formData.password !== formData.confirmPassword) {
      setError('Passwords do not match');
      return;
    }

    if (!formData.email || !formData.password || !formData.firstName || 
        !formData.lastName || !formData.username || !formData.dateOfBirth ||
        !formData.role) {
      setError('Please fill in all required fields');
      return;
    }

    try {
      const response = await fetch('/api/register', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          email: formData.email,
          password: formData.password,
          firstName: formData.firstName,
          lastName: formData.lastName,
          username: formData.username,
          dateOfBirth: formData.dateOfBirth,
          role: formData.role,
          address: formData.address,
          phoneNumber: formData.phoneNumber
        }),
      });

      const data = await response.json();

      if (response.ok) {
        router.push('/login');
      } else {
        setError(data.message || 'Registration failed');
      }
    } catch (error) {
      setError('An error occurred during registration');
      console.error('Registration error:', error);
    }
  };

  return (
    <Box
      sx={{
        minHeight: '100vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        bgcolor: 'var(--black-color)',
        py: 4
      }}
    >
      <Paper
        elevation={3}
        component="form"
        onSubmit={handleSubmit}
        sx={{
          p: 4,
          width: '100%',
          maxWidth: 800,
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
          Create Account
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

        <Grid container spacing={2}>
          <Grid item xs={12} sm={6}>
            <TextField
              fullWidth
              label="First Name *"
              name="firstName"
              variant="outlined"
              value={formData.firstName}
              onChange={handleChange}
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
          </Grid>
          
          <Grid item xs={12} sm={6}>
            <TextField
              fullWidth
              label="Last Name *"
              name="lastName"
              variant="outlined"
              value={formData.lastName}
              onChange={handleChange}
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
          </Grid>

          <Grid item xs={12} sm={6}>
            <TextField
              fullWidth
              label="Username *"
              name="username"
              variant="outlined"
              value={formData.username}
              onChange={handleChange}
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
          </Grid>

          <Grid item xs={12} sm={6}>
            <TextField
              fullWidth
              label="Email *"
              name="email"
              type="email"
              variant="outlined"
              value={formData.email}
              onChange={handleChange}
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
          </Grid>

          <Grid item xs={12} sm={6}>
            <LocalizationProvider dateAdapter={AdapterDateFns}>
              <DatePicker
                label="Date of Birth *"
                value={formData.dateOfBirth}
                onChange={handleDateChange}
                slotProps={{
                  textField: {
                    fullWidth: true,
                    variant: "outlined",
                    sx: {
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
                    }
                  }
                }}
              />
            </LocalizationProvider>
          </Grid>

          <Grid item xs={12} sm={6}>
            <FormControl 
              fullWidth 
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
            >
              <InputLabel id="role-label">Role *</InputLabel>
              <Select
                labelId="role-label"
                name="role"
                value={formData.role}
                label="Role *"
                onChange={handleChange}
              >
                <MenuItem value="Investor">Investor</MenuItem>
                <MenuItem value="Trader">Trader</MenuItem>
                <MenuItem value="Developer">Developer</MenuItem>
                <MenuItem value="Analyst">Analyst</MenuItem>
                <MenuItem value="Student">Student</MenuItem>
                <MenuItem value="Other">Other</MenuItem>
              </Select>
            </FormControl>
          </Grid>

          <Grid item xs={12}>
            <TextField
              fullWidth
              label="Address"
              name="address"
              variant="outlined"
              value={formData.address}
              onChange={handleChange}
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
          </Grid>

          <Grid item xs={12} sm={6}>
            <TextField
              fullWidth
              label="Phone Number"
              name="phoneNumber"
              variant="outlined"
              value={formData.phoneNumber}
              onChange={handleChange}
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
          </Grid>

          <Grid item xs={12} sm={6}>
            <TextField
              fullWidth
              label="Password *"
              name="password"
              type="password"
              variant="outlined"
              value={formData.password}
              onChange={handleChange}
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
          </Grid>

          <Grid item xs={12} sm={6}>
            <TextField
              fullWidth
              label="Confirm Password *"
              name="confirmPassword"
              type="password"
              variant="outlined"
              value={formData.confirmPassword}
              onChange={handleChange}
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
          </Grid>
        </Grid>

        <FormHelperText sx={{ mb: 2, color: 'var(--yellow-color)' }}>Fields marked with * are required</FormHelperText>

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
          Register
        </Button>

        <Button
          fullWidth
          variant="outlined"
          onClick={() => router.push('/login')}
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
          Back to Login
        </Button>
      </Paper>
    </Box>
  );
}
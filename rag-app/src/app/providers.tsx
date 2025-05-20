'use client';

import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';

const theme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#4F46E5', // Electric Indigo - primary accent
      light: '#818CF8', // Lighter accent
      dark: '#3730A3', // Darker accent
    },
    secondary: {
      main: '#06B6D4', // Cyan for contrast elements
    },
    background: {
      default: '#0F172A', // Deep blue background
      paper: '#1E293B',  // Surface elements background
    },
    text: {
      primary: '#F8FAFC',
      secondary: '#CBD5E1',
    },
    error: {
      main: '#EF4444',
    },
    warning: {
      main: '#F59E0B',
    },
    info: {
      main: '#3B82F6',
    },
    success: {
      main: '#10B981',
    },
  },
  typography: {
    fontFamily: "'Rajdhani', 'Orbitron', sans-serif",
    h1: {
      letterSpacing: '0.2rem',
    },
    h2: {
      letterSpacing: '0.15rem',
    },
    h3: {
      letterSpacing: '0.1rem',
    },
    h4: {
      letterSpacing: '0.05rem',
    },
    h5: {
      letterSpacing: '0.05rem',
    },
    h6: {
      letterSpacing: '0.05rem',
    },
    button: {
      letterSpacing: '0.1rem',
      textTransform: 'uppercase',
    },
  },
  shape: {
    borderRadius: 8,
  },
  components: {
    MuiCssBaseline: {
      styleOverrides: {
        '@import': "url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@300;400;500;600;700&family=Orbitron:wght@400;500;600;700&display=swap')",
        body: {
          scrollbarWidth: 'thin',
          scrollbarColor: '#4F46E5 #0F172A',
          '&::-webkit-scrollbar': {
            width: '6px',
            height: '6px',
          },
          '&::-webkit-scrollbar-track': {
            background: '#0F172A',
          },
          '&::-webkit-scrollbar-thumb': {
            backgroundColor: '#4F46E5',
            borderRadius: '3px',
          },
          '&::-webkit-scrollbar-thumb:hover': {
            background: 'linear-gradient(135deg, #4F46E5, #06B6D4)',
          },
        },
      },
    },
    MuiTextField: {
      styleOverrides: {
        root: {
          '& .MuiOutlinedInput-root': {
            '& fieldset': {
              borderColor: 'rgba(79, 70, 229, 0.3)',
              transition: 'all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1)',
            },
            '&:hover fieldset': {
              borderColor: '#4F46E5',
            },
            '&.Mui-focused fieldset': {
              borderColor: '#4F46E5',
              boxShadow: '0 0 15px rgba(79, 70, 229, 0.25)',
            },
          },
          '& .MuiInputLabel-root': {
            color: '#818CF8',
          },
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: '8px',
          textTransform: 'uppercase',
          letterSpacing: '1px',
          fontWeight: 500,
          transition: 'all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1)',
          '&:hover': {
            boxShadow: '0 0 15px rgba(79, 70, 229, 0.25)',
          },
        },
        contained: {
          background: 'linear-gradient(135deg, #4F46E5, #3730A3)',
          '&:hover': {
            background: 'linear-gradient(135deg, #4F46E5, #06B6D4)',
          },
        },
        outlined: {
          borderColor: 'rgba(79, 70, 229, 0.3)',
          '&:hover': {
            borderColor: '#4F46E5',
            backgroundColor: 'rgba(79, 70, 229, 0.1)',
          },
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundImage: 'none',
          backgroundColor: 'rgba(30, 41, 59, 0.7)',
          backdropFilter: 'blur(10px)',
          transition: 'all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1)',
          '&:hover': {
            boxShadow: '0 0 25px rgba(79, 70, 229, 0.15)',
          },
        },
        elevation1: {
          boxShadow: '0 0 15px rgba(79, 70, 229, 0.1)',
        },
      },
    },
    MuiAppBar: {
      styleOverrides: {
        root: {
          background: 'linear-gradient(90deg, #0F172A, #1E293B)',
          boxShadow: '0 0 15px rgba(79, 70, 229, 0.15)',
        },
      },
    },
    MuiSwitch: {
      styleOverrides: {
        root: {
          width: 42,
          height: 26,
          padding: 0,
          '& .MuiSwitch-switchBase': {
            padding: 0,
            margin: 2,
            transitionDuration: '300ms',
            '&.Mui-checked': {
              transform: 'translateX(16px)',
              color: '#fff',
              '& + .MuiSwitch-track': {
                backgroundColor: '#4F46E5',
                opacity: 0.5,
                border: 0,
              },
            },
            '&.Mui-focusVisible .MuiSwitch-thumb': {
              color: '#4F46E5',
              border: '6px solid #fff',
            },
          },
          '& .MuiSwitch-thumb': {
            boxSizing: 'border-box',
            width: 22,
            height: 22,
          },
          '& .MuiSwitch-track': {
            borderRadius: 26 / 2,
            backgroundColor: 'rgba(79, 70, 229, 0.2)',
            opacity: 1,
          },
        },
      },
    },
    MuiIconButton: {
      styleOverrides: {
        root: {
          transition: 'all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1)',
          '&:hover': {
            backgroundColor: 'rgba(79, 70, 229, 0.1)',
            boxShadow: '0 0 10px rgba(79, 70, 229, 0.25)',
          },
        },
      },
    },
    MuiList: {
      styleOverrides: {
        root: {
          padding: '8px',
          backgroundColor: 'rgba(30, 41, 59, 0.7)',
          backdropFilter: 'blur(10px)',
        },
      },
    },
    MuiListItem: {
      styleOverrides: {
        root: {
          borderRadius: '8px',
          transition: 'all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1)',
          '&:hover': {
            backgroundColor: 'rgba(79, 70, 229, 0.1)',
          },
        },
      },
    },
    MuiDivider: {
      styleOverrides: {
        root: {
          borderColor: 'rgba(79, 70, 229, 0.2)',
        },
      },
    },
    MuiAlert: {
      styleOverrides: {
        root: {
          borderRadius: '8px',
          backdropFilter: 'blur(10px)',
        },
        standardError: {
          borderLeft: '4px solid #EF4444',
          backgroundColor: 'rgba(15, 23, 42, 0.8)',
        },
        standardWarning: {
          borderLeft: '4px solid #F59E0B',
          backgroundColor: 'rgba(15, 23, 42, 0.8)',
        },
        standardInfo: {
          borderLeft: '4px solid #3B82F6',
          backgroundColor: 'rgba(15, 23, 42, 0.8)',
        },
        standardSuccess: {
          borderLeft: '4px solid #10B981',
          backgroundColor: 'rgba(15, 23, 42, 0.8)',
        },
      },
    },
  },
});

export function Providers({ children }: { children: React.ReactNode }) {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      {children}
    </ThemeProvider>
  );
}
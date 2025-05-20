'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { Typography, Button, Paper, CircularProgress } from '@mui/material';
import ChatIcon from '@mui/icons-material/Chat';

interface WelcomeSectionProps {
  username?: string;
}

// Define a type for the user data
interface UserData {
  _id?: string;
  username?: string;
  firstName?: string;
  lastName?: string;
  email?: string;
  name?: string;
}

const WelcomeSection = ({ username: propUsername }: WelcomeSectionProps) => {
  const router = useRouter();
  const [firstName, setFirstName] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  
  useEffect(() => {
    const fetchUserData = async () => {
      setLoading(true);
      setError(null);
      
      try {
        // First try to get userId from localStorage
        const userString = localStorage.getItem('user');
        let userData: UserData | null = null;
        let userId: string | null = null;
        
        if (userString) {
          try {
            userData = JSON.parse(userString);
            // Use optional chaining to safely access _id property
            userId = userData?._id || null;
          } catch (parseError) {
            console.error('Error parsing user data from localStorage:', parseError);
          }
        }
        
        // If we have a userId, fetch from MongoDB
        if (userId) {
          try {
            const response = await fetch(`/api/User?userId=${userId}`);
            
            if (!response.ok) {
              // If API call fails, fall back to localStorage data
              console.error('Failed to fetch user data from API, using localStorage');
              
              if (userData) {
                // Use data from localStorage - with optional chaining for safety
                if (userData?.firstName) {
                  setFirstName(userData.firstName);
                } else if (propUsername) {
                  setFirstName(propUsername);
                } else {
                  setFirstName(userData?.username || userData?.name || 'User');
                }
              } else if (propUsername) {
                setFirstName(propUsername);
              } else {
                setFirstName('User');
              }
            } else {
              // Use data from API
              const mongoData: UserData = await response.json();
              
              if (mongoData?.firstName) {
                setFirstName(mongoData.firstName);
              } else if (mongoData?.name) {
                setFirstName(mongoData.name);
              } else if (mongoData?.username) {
                setFirstName(mongoData.username);
              } else if (propUsername) {
                setFirstName(propUsername);
              } else {
                setFirstName('User');
              }
            }
          } catch (fetchError) {
            console.error('Error fetching from API:', fetchError);
            // Fallback to localStorage data
            if (userData?.firstName) {
              setFirstName(userData.firstName);
            } else if (userData?.username) {
              setFirstName(userData.username);
            } else if (userData?.name) {
              setFirstName(userData.name);
            } else if (propUsername) {
              setFirstName(propUsername);
            } else {
              setFirstName('User');
            }
          }
        } else {
          // If no userId, use prop or default
          if (userData && userData.firstName) {
            setFirstName(userData.firstName);
          } else if (propUsername) {
            setFirstName(propUsername);
          } else {
            setFirstName('User');
          }
        }
      } catch (err) {
        console.error('Error fetching user data:', err);
        setError('Failed to load user data');
        
        // Fallback to the username prop if there's an error
        if (propUsername) {
          setFirstName(propUsername);
        } else {
          setFirstName('User');
        }
      } finally {
        setLoading(false);
      }
    };
    
    fetchUserData();
  }, [propUsername]);
  
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
        {loading ? (
          <CircularProgress size={30} color="inherit" />
        ) : (
          <>Welcome, {firstName}!</>
        )}
      </Typography>
      <Typography variant="h5" paragraph>
        Your cryptocurrency due diligence assistant is ready.
      </Typography>
      <Typography variant="body1" paragraph>
        LST (Legal, Secure, Trust) helps you perform comprehensive cryptocurrency analysis and due diligence. Our system provides accurate, context-aware insights based on market data, project documentation, and technical indicators to help you make informed investment decisions.
      </Typography>
      <Button 
        variant="contained"
        size="large"
        startIcon={<ChatIcon />}
        onClick={() => router.push('/chat')}
        sx={{
          mt: 2,
          bgcolor: 'var(--yellow-color)',
          color: 'var(--black-color)',
          '&:hover': {
            bgcolor: 'rgba(255, 215, 0, 0.8)',
          }
        }}
      >
        Start Chatting
      </Button>
    </Paper>
  );
};

export default WelcomeSection;
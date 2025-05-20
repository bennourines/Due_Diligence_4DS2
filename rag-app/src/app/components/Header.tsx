'use client';

import { useRouter } from 'next/navigation';
import { AppBar, Toolbar, Button, Box } from '@mui/material';
import Image from 'next/image';
import HomeIcon from '@mui/icons-material/Home';
import ChatIcon from '@mui/icons-material/Chat';
import LogoutIcon from '@mui/icons-material/Logout';
import LoginIcon from '@mui/icons-material/Login';

interface HeaderProps {
  onLogout: () => void;
  isAuthenticated: boolean;
}

const Header = ({ onLogout, isAuthenticated }: HeaderProps) => {
  const router = useRouter();
  
  const handleLoginClick = () => {
    // Force complete page reload with login URL
    window.location.replace('/login');
  };
  
  return (
    <AppBar 
      position="static" 
      sx={{ 
        bgcolor: 'var(--black-color)',
        height: { xs: '70px', md: '80px' }, // Increased height
        boxShadow: '0 3px 5px rgba(0, 0, 0, 0.2)'
      }}
    >
      <Toolbar sx={{ 
        height: '100%', 
        padding: { xs: '0 10px', md: '0 24px' },
        display: 'flex',
        justifyContent: 'space-between' 
      }}>
        {/* Logo container - taking more space */}
        <Box 
          sx={{ 
            display: 'flex',
            alignItems: 'center',
            height: '100%',
            flex: '1 0 auto',
            maxWidth: '125px',
            paddingRight: { xs: 1, md: 4 },
            cursor: 'pointer'
          }}
          onClick={() => router.push('/home')}
        >
          <Image
            src="/LST.png"
            alt="LST Logo"
            width={540}  // Significantly increased size
            height={540}  // Increased height
            style={{ 
              objectFit: 'contain',
              width: '100%',
              maxHeight: '250%',
            }}
            priority
          />
        </Box>
        
        {/* Navigation buttons */}
        <Box 
          sx={{ 
            display: 'flex', 
            alignItems: 'center',
            justifyContent: 'flex-end',
            gap: { xs: 1, md: 2 }
          }}
        >
          <Button 
            color="inherit" 
            startIcon={<HomeIcon />}
            sx={{ 
              color: 'var(--yellow-color)',
              fontSize: { xs: '0.875rem', md: '1rem' },
              paddingX: { xs: 1, md: 2 }
            }}
            onClick={() => router.push('/home')}
          >
            HOME
          </Button>
          
          {isAuthenticated ? (
            <>
            <Button 
                color="inherit"
                startIcon={<ChatIcon />}
                sx={{ 
                  color: 'var(--yellow-color)',
                  fontSize: { xs: '0.875rem', md: '1rem' },
                  paddingX: { xs: 1, md: 2 }
                }}
                onClick={() => router.push('/reports')}
              >
                REPORTS
              </Button>
              <Button 
                color="inherit"
                startIcon={<ChatIcon />}
                sx={{ 
                  color: 'var(--yellow-color)',
                  fontSize: { xs: '0.875rem', md: '1rem' },
                  paddingX: { xs: 1, md: 2 }
                }}
                onClick={() => router.push('/chat')}
              >
                CHAT
              </Button>
              <Button 
                color="inherit"
                startIcon={<LogoutIcon />}
                sx={{ 
                  color: 'var(--yellow-color)',
                  fontSize: { xs: '0.875rem', md: '1rem' },
                  paddingX: { xs: 1, md: 2 }
                }}
                onClick={onLogout}
              >
                LOGOUT    
              </Button>
            </>
          ) : (
            <Button 
              color="inherit"
              startIcon={<LoginIcon />}
              sx={{ 
                color: 'var(--yellow-color)',
                fontSize: { xs: '0.875rem', md: '1rem' },
                paddingX: { xs: 1, md: 2 }
              }}
              onClick={handleLoginClick}
            >
              LOGIN
            </Button>
          )}
        </Box>
      </Toolbar>
    </AppBar>
  );
};

export default Header;
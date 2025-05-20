'use client';

import { Card, CardContent, Typography, Box, Button } from '@mui/material';

interface FeatureCardProps {
  title: string;
  description: string;
  icon: React.ReactNode;
  action: () => void;
  buttonText: string;
}

const FeatureCard = ({ title, description, icon, action, buttonText }: FeatureCardProps) => {
  return (
    <Card 
      sx={{ 
        height: '100%', 
        display: 'flex', 
        flexDirection: 'column',
        bgcolor: 'rgba(0, 0, 0, 0.7)',
        color: 'var(--yellow-color)',
        border: '1px solid rgba(255, 215, 0, 0.3)',
        transition: 'transform 0.3s ease-in-out, box-shadow 0.3s ease-in-out',
        '&:hover': {
          transform: 'translateY(-5px)',
          boxShadow: '0 10px 20px rgba(255, 215, 0, 0.2)',
        }
      }}
    >
      <CardContent sx={{ flexGrow: 1 }}>
        <Box sx={{ display: 'flex', justifyContent: 'center', mb: 2 }}>
          {icon}
        </Box>
        <Typography gutterBottom variant="h5" component="h2" sx={{ textAlign: 'center' }}>
          {title}
        </Typography>
        <Typography sx={{ mb: 2 }}>
          {description}
        </Typography>
        <Box sx={{ display: 'flex', justifyContent: 'center', mt: 'auto' }}>
          <Button 
            variant="contained" 
            onClick={action}
            sx={{
              bgcolor: 'var(--yellow-color)',
              color: 'var(--black-color)',
              '&:hover': {
                bgcolor: 'rgba(255, 215, 0, 0.8)',
              }
            }}
          >
            {buttonText}
          </Button>
        </Box>
      </CardContent>
    </Card>
  );
};

export default FeatureCard;
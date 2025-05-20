'use client';

import { Box, Typography } from '@mui/material';

const Footer = () => {
  return (
    <Box 
      component="footer" 
      sx={{ 
        py: 3, 
        mt: 'auto',
        bgcolor: 'var(--black-color)',
        color: 'var(--yellow-color)',
        textAlign: 'center',
        borderTop: '1px solid rgba(255, 215, 0, 0.2)'
      }}
    >
      <Typography variant="body2">
        © {new Date().getFullYear()} RAG Assistant. All rights reserved.
      </Typography>
    </Box>
  );
};

export default Footer;
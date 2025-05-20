'use client';

import { Box, CircularProgress, Typography } from '@mui/material';

export default function Home() {
  return (
    <Box
      sx={{
        height: '100vh',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        bgcolor: 'var(--black-color)',
        color: 'var(--yellow-color)',
      }}
    >
      <CircularProgress sx={{ color: 'var(--yellow-color)', mb: 2 }} />
      <Typography>Loading...</Typography>
    </Box>
  );
}
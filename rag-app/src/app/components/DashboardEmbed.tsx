'use client';

import React, { useState, useEffect } from 'react';
import { Box, CircularProgress, Typography } from '@mui/material';

interface DashboardEmbedProps {
  dashboardUrl: string;
  title?: string;
  height?: string;
}

const DashboardEmbed: React.FC<DashboardEmbedProps> = ({
  dashboardUrl,
  title = 'Cryptocurrency Dashboard',
  height = '100vh',
}) => {
  const [loading, setLoading] = useState<boolean>(true);

  // Handle iframe load event
  const handleIframeLoad = () => {
    setLoading(false);
  };

  return (
    <Box
      sx={{
        width: '100%',
        height: height,
        position: 'relative',
        borderRadius: 'var(--border-radius)',
        border: 'var(--border-light)',
        backgroundColor: 'rgba(30, 41, 59, 0.7)',
        backdropFilter: 'blur(10px)',
        boxShadow: 'var(--shadow-glow)',
        overflow: 'hidden',
      }}
    >
      {loading && (
        <Box
          sx={{
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            backgroundColor: 'var(--bg-deep)',
            zIndex: 1,
          }}
        >
          <div className="loader" />
          <Typography
            variant="body1"
            sx={{
              color: 'var(--text-primary)',
              mt: 2,
              fontFamily: 'var(--font-primary)',
            }}
          >
            Loading Dashboard...
          </Typography>
        </Box>
      )}
      <iframe
        src={dashboardUrl}
        title={title}
        width="100%"
        height="100%"
        style={{
          border: 'none',
          borderRadius: 'var(--border-radius)',
        }}
        onLoad={handleIframeLoad}
      />
    </Box>
  );
};

export default DashboardEmbed;
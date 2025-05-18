// next.config.js
/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  experimental: {
    serverActions: {
      bodySizeLimit: '10mb',
    },
  },
  // Extend HTTP timeouts
  httpAgentOptions: {
    keepAlive: true,
    // Dramatically increase timeouts for long-running FastAPI operations
    headersTimeout: 70 * 60 * 1000, 
    responseTimeout: 100 * 60 * 1000, 
  },
  // Configure Node.js options
  serverRuntimeConfig: {
    // Will only be available on the server side
    apiTimeout: 90 * 60 * 1000, // 90 minutes
  },
};

module.exports = nextConfig;
import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';

export async function middleware(request: NextRequest) {
  const path = request.nextUrl.pathname;
  
  // Only redirect root path to home
  if (path === '/') {
    return NextResponse.redirect(new URL('/home', request.url));
  }

  // Allow all other routes - authentication will be handled client-side
  return NextResponse.next();
}

export const config = {
  matcher: ['/'],
};
import { NextResponse } from 'next/server';
import { authenticateUser, connectToDatabase } from '../auth';

export async function POST(request: Request) {
  try {
    const { email, password } = await request.json();

    if (!email || !password) {
      return NextResponse.json(
        { message: 'Email and password are required' },
        { status: 400 }
      );
    }

    // Authentication includes connection check
    const { user } = await authenticateUser(email, password);

    // Create a default chat for the user
    try {
      const db = await connectToDatabase();
      const defaultChat = await db.collection('Chat').insertOne({
        userId: user._id.toString(),
        title: 'New Chat',
        createdAt: new Date()
      });
      
      // Set authentication cookie
      const response = NextResponse.json({ 
        user,
        defaultChatId: defaultChat.insertedId 
      });
      
      // Set an HTTP-only cookie with the token
      response.cookies.set({
        name: 'token',
        value: user._id.toString(), // Consider using a proper JWT token in production
        httpOnly: true,
        maxAge: 60 * 60 * 24 * 7, // 1 week
        path: '/'
      });
      
      return response;
    } catch (dbError) {
      console.error('Error creating default chat:', dbError);
      
      // Return user data even if chat creation fails
      const response = NextResponse.json({ user });
      response.cookies.set({
        name: 'token',
        value: user._id.toString(),
        httpOnly: true,
        maxAge: 60 * 60 * 24 * 7,
        path: '/'
      });
      
      return response;
    }
  } catch (error: any) {
    console.error('Login error:', error);
    return NextResponse.json(
      { message: error.message || 'Authentication failed' },
      { status: 401 }
    );
  }
}
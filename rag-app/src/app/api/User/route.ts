import { NextRequest, NextResponse } from 'next/server';
import { connectToDatabase } from '../auth'; // Use your existing database connection
import { ObjectId } from 'mongodb';

export async function GET(request: NextRequest) {
  try {
    // Get the user ID from query params
    const userId = request.nextUrl.searchParams.get('userId');
    
    if (!userId) {
      return NextResponse.json(
        { message: 'User ID is required' },
        { status: 400 }
      );
    }

    // Connect to the database
    const db = await connectToDatabase();
    const userCollection = db.collection('User');
    
    // Find the user
    const user = await userCollection.findOne({ 
      _id: new ObjectId(userId)
    });
    
    if (!user) {
      return NextResponse.json(
        { message: 'User not found' },
        { status: 404 }
      );
    }
    
    // Return the user data (excluding sensitive fields like password)
    const userData = {
      _id: user._id,
      firstName: user.firstName || '',
      lastName: user.lastName || '',
      email: user.email || '',
      username: user.username || '',
      name: user.name || ''
    };
    
    return NextResponse.json(userData);
  } catch (error) {
    console.error('Error fetching user data:', error);
    return NextResponse.json(
      { 
        message: 'Failed to fetch user data', 
        error: error instanceof Error ? error.message : 'Unknown error' 
      },
      { status: 500 }
    );
  }
}
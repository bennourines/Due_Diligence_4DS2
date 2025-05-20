import { NextResponse } from 'next/server';
import { connectToDatabase, getClient } from '../auth';

export async function POST(request: Request) {
  try {
    const { 
      email, 
      password, 
      firstName, 
      lastName, 
      username, 
      dateOfBirth, 
      role, 
      address, 
      phoneNumber 
    } = await request.json();

    // Validate required fields
    if (!email || !password || !firstName || !lastName || !username || !dateOfBirth || !role) {
      return NextResponse.json(
        { message: 'Required fields are missing' },
        { status: 400 }
      );
    }

    // Basic email validation
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(email)) {
      return NextResponse.json(
        { message: 'Invalid email format' },
        { status: 400 }
      );
    }

    // Password strength validation
    if (password.length < 6) {
      return NextResponse.json(
        { message: 'Password must be at least 6 characters long' },
        { status: 400 }
      );
    }

    // Username validation (no spaces, special characters limited)
    const usernameRegex = /^[a-zA-Z0-9_.-]+$/;
    if (!usernameRegex.test(username)) {
      return NextResponse.json(
        { message: 'Username can only contain letters, numbers, underscores, periods, and hyphens' },
        { status: 400 }
      );
    }

    let db;
    try {
      db = await connectToDatabase();
      const usersCollection = db.collection('User');

      // Check if email already exists
      const existingEmail = await usersCollection.findOne({ email });
      if (existingEmail) {
        return NextResponse.json(
          { message: 'Email already registered' },
          { status: 400 }
        );
      }

      // Check if username already exists
      const existingUsername = await usersCollection.findOne({ username });
      if (existingUsername) {
        return NextResponse.json(
          { message: 'Username already taken' },
          { status: 400 }
        );
      }

      // Create new user with all fields
      await usersCollection.insertOne({
        email,
        password, // In a real application, password should be hashed
        firstName,
        lastName,
        username,
        dateOfBirth: new Date(dateOfBirth),
        role,
        address: address || '',
        phoneNumber: phoneNumber || '',
        createdAt: new Date()
      });

      return NextResponse.json(
        { message: 'Registration successful' },
        { status: 201 }
      );
    } catch (dbError: any) {
      console.error('Database operation error:', dbError);
      return NextResponse.json(
        { message: 'Failed to register user' },
        { status: 500 }
      );
    } finally {
      const client = await getClient();
      if (db && client) {
        try {
          await client.close();
        } catch (closeError: unknown) {
          console.error('Error closing database connection:', closeError);
        }
      }
    }
  } catch (error: any) {
    console.error('Registration error:', error);
    return NextResponse.json(
      { message: error.message || 'Registration failed' },
      { status: 500 }
    );
  }
}
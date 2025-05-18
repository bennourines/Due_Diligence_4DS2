import { NextResponse } from 'next/server';
import { connectToDatabase } from '../auth';

export async function GET() {
  try {
    const db = await connectToDatabase();
    const collections = await db.collections();
    
    return NextResponse.json({
      status: 'success',
      message: 'Successfully connected to MongoDB',
      collections: collections.map(collection => collection.collectionName)
    });
  } catch (error) {
    console.error('Database connection test failed:', error);
    return NextResponse.json(
      {
        status: 'error',
        message: error instanceof Error ? error.message : 'Failed to connect to database'
      },
      { status: 500 }
    );
  }
}
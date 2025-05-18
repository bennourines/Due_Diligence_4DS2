import { NextRequest, NextResponse } from 'next/server';
import { getUserReports, getReportById } from '../../lib/mongodb';

export async function GET(request: NextRequest) {
  try {
    // Get user ID from query params
    const userId = request.nextUrl.searchParams.get('userId');
    const reportId = request.nextUrl.searchParams.get('reportId');
    
    if (!userId) {
      return NextResponse.json(
        { message: 'User ID is required. Please provide a valid user ID.' },
        { status: 400 }
      );
    }

    let result;
    // If reportId is provided, get a specific report
    if (reportId) {
      result = await getReportById(userId, reportId);
      if (!result) {
        return NextResponse.json(
          { message: 'Report not found' },
          { status: 404 }
        );
      }
    } else {
      // Otherwise, get all reports for the user
      result = await getUserReports(userId);
    }
    
    return NextResponse.json(result);
  } catch (error) {
    console.error('Error getting reports:', error);
    return NextResponse.json(
      { 
        message: 'Failed to get reports', 
        error: error instanceof Error ? error.message : 'Unknown error' 
      },
      { status: 500 }
    );
  }
}
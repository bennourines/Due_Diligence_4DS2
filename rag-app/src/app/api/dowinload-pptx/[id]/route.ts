import { NextRequest, NextResponse } from 'next/server';
import { getUserReports } from '../../../lib/mongodb'; // Adjust import to match your project structure
import path from 'path';
import fs from 'fs';

export async function GET(
  request: NextRequest,
  { params }: { params: { id: string } }
) {
  try {
    const reportId = params.id;
    
    if (!reportId) {
      return NextResponse.json(
        { message: 'Report ID is required' },
        { status: 400 }
      );
    }
    
    console.log(`Attempting to download report with ID: ${reportId}`);
    
    // Get the report from the database
    // Modify this to match your database query function
    const report = await getUserReports(reportId);
    
    if (!report) {
      return NextResponse.json(
        { message: 'Report not found' },
        { status: 404 }
      );
    }
    
    // Extract the file path from the report
    const filePath = report.filePath || report.pptxUrl;
    
    if (!filePath) {
      return NextResponse.json(
        { message: 'Report file path not found' },
        { status: 404 }
      );
    }

    console.log(`Serving file from path: ${filePath}`);
    
    // Check if file exists locally
    const localPath = path.join(process.cwd(), filePath);
    if (!fs.existsSync(localPath)) {
      console.log(`File not found at: ${localPath}`);
      return NextResponse.json(
        { message: 'File not found on server' },
        { status: 404 }
      );
    }
    
    // Read the file
    const fileBuffer = fs.readFileSync(localPath);
    const filename = path.basename(filePath);
    
    // Create response with appropriate headers
    const response = new NextResponse(fileBuffer, {
      status: 200,
      headers: {
        'Content-Disposition': `attachment; filename=${filename}`,
        'Content-Type': 'application/vnd.openxmlformats-officedocument.presentationml.presentation'
      }
    });
    
    return response;
  } catch (error) {
    console.error('Error downloading report:', error);
    return NextResponse.json(
      { 
        message: 'Failed to download report', 
        error: error instanceof Error ? error.message : 'Unknown error',
      },
      { status: 500 }
    );
  }
}
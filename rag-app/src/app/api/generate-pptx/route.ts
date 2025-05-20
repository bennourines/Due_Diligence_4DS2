import { NextRequest, NextResponse } from 'next/server';
import { v4 as uuidv4 } from 'uuid';
import { storePptxReportMetadata } from '../../lib/mongodb';

// Increase the Next.js API timeout to 1.5 hours (90 minutes)
export const maxDuration = 90 * 60; // 90 minutes in seconds

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { selected_questions, doc_id, coin_name, userId } = body;
    
    // Ensure we have a user ID
    if (!userId) {
      return NextResponse.json(
        { message: 'User ID is required. Please provide a valid user ID.' },
        { status: 400 }
      );
    }

    // Validate request data
    if (!selected_questions || !Array.isArray(selected_questions) || selected_questions.length === 0) {
      return NextResponse.json(
        { 
          message: 'At least one question must be selected',
          error: 'Missing or invalid selected_questions'
        }, 
        { status: 400 }
      );
    }

    if (!coin_name) {
      return NextResponse.json(
        { 
          message: 'Coin name is required',
          error: 'Missing coin_name'
        }, 
        { status: 400 }
      );
    }

    // Log the request for debugging
    console.log('Sending request to FastAPI:', {
      selected_questions_count: selected_questions.length,
      doc_id: doc_id || 'all',
      coin_name,
      userId: userId
    });

    // Make request to FastAPI backend
    const backendUrl = 'http://localhost:8001';
    const fullUrl = `${backendUrl}/generate-presentation`;
    console.log(`Sending request to: ${fullUrl}`);
    
    // Create fetch request with a long timeout (85 minutes - slightly less than our max to allow for processing)
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 85 * 60 * 1000);
    
    try {
      const response = await fetch(fullUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          selected_questions,
          doc_id: doc_id || 'all',
          coin_name
        }),
        signal: controller.signal,
        // Add longer timeouts if using node-fetch
        // @ts-ignore
        timeout: 85 * 60 * 1000, // 85 minutes
      });
      
      clearTimeout(timeoutId);
      
      // Log response status and headers for debugging
      console.log('Response status:', response.status);
      console.log('Response status text:', response.statusText);
      
      // Get the response as text first
      const responseText = await response.text();
      console.log('Response body (first 500 chars):', responseText.substring(0, 500));
      
      if (!response.ok) {
        console.error('Error response from FastAPI:', {
          status: response.status,
          statusText: response.statusText,
          body: responseText.substring(0, 1000) // Log first 1000 chars
        });
        
        // Try to parse as JSON, but have a fallback
        let errorDetail = 'Unknown error from backend';
        try {
          if (responseText && responseText.trim()) {
            const errorData = JSON.parse(responseText);
            errorDetail = errorData.detail || errorData.message || errorData.error || 'Error from backend';
          }
        } catch (e) {
          // If parsing fails, use the response text
          errorDetail = responseText || 'Error from backend';
        }
        
        return NextResponse.json(
          { message: 'Failed to generate presentation', error: errorDetail },
          { status: response.status }
        );
      }

      // Parse JSON with error handling
      let data;
      try {
        data = responseText ? JSON.parse(responseText) : {};
      } catch (parseError) {
        console.error('Failed to parse JSON response:', parseError);
        return NextResponse.json(
          {
            message: 'Invalid JSON response from backend',
            error: 'Parse error',
            responseText: responseText.substring(0, 500) // Include partial response for debugging
          },
          { status: 500 }
        );
      }
      
      console.log('Success response from FastAPI:', data);
      
      // If successfully generated, store in MongoDB
      if (data.file_path) {
        try {
          // Generate a unique report ID
          const reportId = uuidv4();
          
          // Get the question IDs from the selected_questions array
          const questionIds = selected_questions.map((q: any) => q.id);
          
          // Store PPTX metadata in MongoDB
          // @ts-ignore - Suppress TypeScript error related to MongoDB types
          await storePptxReportMetadata(userId, {
            reportId,
            coinName: coin_name,
            filePath: data.file_path,
            timestamp: Date.now(),
            questionCount: selected_questions.length,
            questions: questionIds
          });
          
          // Add the report ID to the response
          data.reportId = reportId;
          
          console.log(`PPTX metadata stored in MongoDB for user ${userId}, report ${reportId}`);
        } catch (dbError) {
          console.error('Failed to store PPTX metadata in MongoDB:', dbError);
          // Continue with the response even if MongoDB storage fails
          // But include a warning in the response
          data.dbWarning = 'Presentation was generated but metadata storage failed';
        }
      }
      
      return NextResponse.json(data);
    } catch (fetchError) {
      clearTimeout(timeoutId);
      throw fetchError; // Re-throw to be caught by the outer try/catch
    }
  } catch (error) {
    console.error('Error generating presentation:', error);
    return NextResponse.json(
      { 
        message: 'Failed to generate presentation', 
        error: error instanceof Error ? error.message : 'Unknown error',
        cause: error instanceof Error && error.cause ? JSON.stringify(error.cause) : undefined
      },
      { status: 500 }
    );
  }
}
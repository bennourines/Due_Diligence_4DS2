// Fix for src/app/api/proxy-to-workflow/route.ts
import { NextRequest, NextResponse } from 'next/server';
import { Client } from '@upstash/qstash';
import { v4 as uuidv4 } from 'uuid';
import { storePptxReportMetadata } from '../../lib/mongodb';

// Define your types
interface QuestionItem {
  id: string;
  text: string;
  section_id: number;
  section_title: string;
}

interface ReportMetadata {
  reportId: string;
  userId?: string;
  coinName: string;
  filePath: string;
  timestamp: number;
  questionCount: number;
  questions: string[];
  status?: string;
  error?: string | null;
}

/**
 * This route acts as a proxy between your frontend and Upstash Workflow.
 * Instead of calling the Workflow directly, the frontend calls this endpoint,
 * which then properly sends the request to QStash to trigger the workflow.
 * 
 * This approach also stores initial metadata and returns a response immediately,
 * mimicking the behavior of the workflow without requiring direct access.
 */
export async function POST(request: NextRequest) {
  try {
    // Parse the incoming request body
    const body = await request.json();
    const { selected_questions, doc_id, coin_name, userId } = body;
    
    // Validate inputs
    if (!userId) {
      return NextResponse.json({ message: 'User ID is required' }, { status: 400 });
    }

    if (!selected_questions || !Array.isArray(selected_questions) || selected_questions.length === 0) {
      return NextResponse.json({ message: 'At least one question must be selected' }, { status: 400 });
    }

    if (!coin_name) {
      return NextResponse.json({ message: 'Coin name is required' }, { status: 400 });
    }
    
    // Generate a unique report ID
    const reportId = uuidv4();
    const timestamp = Date.now();
    
    // Create expected file path
    const safeCoinsName = coin_name.replace(/[^a-zA-Z0-9]/g, '_');
    const expectedFileName = `${safeCoinsName}_due_diligence_${reportId.substring(0, 8)}.pptx`;
    const expectedFilePath = `reports/${expectedFileName}`;
    
    // Store initial metadata in MongoDB
    await storePptxReportMetadata(userId, {
      reportId,
      coinName: coin_name,
      filePath: expectedFilePath,
      timestamp,
      questionCount: selected_questions.length,
      questions: selected_questions.map(q => q.id),
      status: 'queued',
      error: null
    } as ReportMetadata);
    
    console.log(`Created initial report entry with ID: ${reportId}`);
    
    // Create a QStash client (if environment variable is set)
    let qstashClient = null;
    if (process.env.QSTASH_TOKEN) {
      qstashClient = new Client({
        token: process.env.QSTASH_TOKEN,
      });
    } else {
      console.log('QSTASH_TOKEN environment variable is not set, skipping QStash integration');
    }
    
    try {
      if (qstashClient) {
        // Get your workflow endpoint URL (adjust as needed)
        const workflowEndpoint = process.env.NEXT_PUBLIC_APP_URL 
          ? `${process.env.NEXT_PUBLIC_APP_URL}/api/generate-pptx`
          : 'http://localhost:3000/api/generate-pptx';
  
        // Publish the message to QStash, which will trigger your workflow
        // Include the report ID in the request
        const publishResponse = await qstashClient.publishJSON({
          url: workflowEndpoint,
          body: {
            ...body,
            reportId  // Include the generated reportId
          },
        });
        
        console.log('Request sent to QStash:', publishResponse);
      } else {
        // Fallback: Make a direct call to the FastAPI backend
        // This is for development when QStash is not available
        console.log('Using fallback: direct call to FastAPI backend');
        const backendUrl = 'http://localhost:8001';
        const fullUrl = `${backendUrl}/generate-presentation`;
        
        console.log(`Sending direct request to FastAPI: ${fullUrl}`);
        
        // Update status to processing
        await storePptxReportMetadata(userId, {
          reportId,
          coinName: coin_name,
          filePath: expectedFilePath,
          timestamp,
          questionCount: selected_questions.length,
          questions: selected_questions.map(q => q.id),
          status: 'processing',
          error: null
        } as ReportMetadata);
        
        // Use fetch with a longer timeout
        // The fetch is done asynchronously without waiting for completion
        fetch(fullUrl, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            selected_questions,
            doc_id: doc_id || 'all',
            coin_name,
            reportId
          }),
        }).then(async (response) => {
          if (response.ok) {
            const data = await response.json();
            
            // Update report status to completed
            await storePptxReportMetadata(userId, {
              reportId, 
              filePath: data.file_path || expectedFilePath,
              coinName: coin_name,
              timestamp,
              questionCount: selected_questions.length,
              questions: selected_questions.map(q => q.id),
              status: 'completed', 
              error: null
            } as ReportMetadata);
            
            console.log(`Updated report ${reportId} status to completed`);
          } else {
            // Handle error response
            const errorText = await response.text();
            
            await storePptxReportMetadata(userId, {
              reportId,
              filePath: expectedFilePath,
              coinName: coin_name,
              timestamp,
              questionCount: selected_questions.length,
              questions: selected_questions.map(q => q.id),
              status: 'error', 
              error: `FastAPI error: ${errorText}`
            } as ReportMetadata);
          }
        }).catch(async (error) => {
          console.error(`Network error for report ${reportId}:`, error);
          
          await storePptxReportMetadata(userId, {
            reportId,
            filePath: expectedFilePath,
            coinName: coin_name,
            timestamp,
            questionCount: selected_questions.length,
            questions: selected_questions.map(q => q.id),
            status: 'error', 
            error: `Network error: ${error instanceof Error ? error.message : String(error)}`
          } as ReportMetadata);
        });
      }
    } catch (qstashError) {
      console.error('Error in QStash or backend processing:', qstashError);
      // Don't fail the request - we've already stored initial metadata
    }
    
    // Return a response to the frontend
    return NextResponse.json({
      reportId,
      status: 'queued',
      message: 'Report generation queued successfully',
      file_path: expectedFilePath 
    });
  } catch (error) {
    console.error('Error in proxy-to-workflow:', error);
    return NextResponse.json(
      { 
        error: error instanceof Error ? error.message : 'Unknown error',
        message: 'Failed to queue report generation'
      }, 
      { status: 500 }
    );
  }
}
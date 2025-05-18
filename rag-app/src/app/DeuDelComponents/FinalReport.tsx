// Updated FinalReport.tsx with server status checking
import React, { useRef, useState, useEffect } from 'react';
import { SECTIONS, QUESTIONS, Question } from '../data/dueDiligenceData';
import { 
  Download, 
  ExternalLink, 
  ArrowLeft, 
  Save, 
  Database, 
  AlertTriangle, 
  Loader,
  ServerCrash,
  RefreshCw
} from 'lucide-react';
import { FinalReportData } from '../context/DueDiligenceContext';
import { useDueDiligence } from '../context/DueDiligenceContext';
import LongLoadingIndicator from './LongLoadingIndicator';

interface FinalReportProps {
  report: FinalReportData;
  selectedQuestions: Record<string, boolean>;
}

// Helper function to find a question by ID
const findQuestionById = (questionId: string): Question | undefined => {
  const [sectionId, questionIndex] = questionId.split('-');
  const sectionIdNumber = parseInt(sectionId, 10);
  const questions = QUESTIONS[sectionIdNumber];
  return questions?.find(q => q.id === questionId);
};

// Helper to extract filename from path
function getFilenameFromPath(filePath: string): string {
  if (!filePath) return '';
  // Split by forward slash and backward slash to handle different path formats
  const parts = filePath.split(/[/\\]/);
  return parts[parts.length - 1];
}

const FinalReport: React.FC<FinalReportProps> = ({ report, selectedQuestions }) => {
  // State for download status and errors
  const [isDownloading, setIsDownloading] = useState(false);
  const [downloadError, setDownloadError] = useState<string | null>(null);
  const [serverStatus, setServerStatus] = useState<'checking' | 'available' | 'unavailable'>('checking');
  const [isCheckingServer, setIsCheckingServer] = useState(false);
  const [serverError, setServerError] = useState<string | null>(null);
  
  // Access the PPTX generation functions from context
  const { 
    isGeneratingPptx, 
    generatePptxReport, 
    pptxReportUrl, 
    pptxError,
    reportId,
    dbStorageStatus,
    dbStorageError
  } = useDueDiligence();
  
  // Add reference for direct download
  const downloadLinkRef = useRef<HTMLAnchorElement>(null);
  
  // Check server status on component mount and periodically
  useEffect(() => {
    checkServerStatus();
    
    // Check server status every 30 seconds if unavailable
    const intervalId = setInterval(() => {
      if (serverStatus === 'unavailable' && !isCheckingServer && !isGeneratingPptx) {
        checkServerStatus();
      }
    }, 30000);
    
    return () => clearInterval(intervalId);
  }, [serverStatus, isCheckingServer, isGeneratingPptx]);
  
  // Function to check if the backend server is running
  const checkServerStatus = async () => {
    setIsCheckingServer(true);
    setServerError(null);
    
    try {
      // Simple check to see if the server is running
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 5000); // 5 second timeout
      
      const response = await fetch('http://localhost:8001/health', {
        signal: controller.signal
      });
      
      clearTimeout(timeoutId);
      
      if (response.ok) {
        setServerStatus('available');
      } else {
        setServerStatus('unavailable');
        setServerError(`Server returned status ${response.status}`);
      }
    } catch (error) {
      console.error('Error checking server status:', error);
      setServerStatus('unavailable');
      
      // Provide a more helpful error message
      if (error instanceof TypeError && error.message.includes('fetch')) {
        setServerError('Unable to connect to server. Please ensure it is running at http://localhost:8001');
      } else if (error instanceof DOMException && error.name === 'AbortError') {
        setServerError('Server connection timed out. The server might be running but is slow to respond.');
      } else {
        setServerError(error instanceof Error ? error.message : 'Unknown error checking server status');
      }
    } finally {
      setIsCheckingServer(false);
    }
  };
  
  // Group selected questions by section
  const questionsBySection = SECTIONS.map(section => {
    const sectionQuestions = Object.keys(selectedQuestions)
      .filter(id => id.startsWith(`${section.id}-`) && selectedQuestions[id])
      .map(id => {
        const question = findQuestionById(id);
        return question || { id, text: 'Question not found' };
      });
    
    return {
      ...section,
      questions: sectionQuestions
    };
  }).filter(section => section.questions.length > 0);

  const handleStartNew = () => {
    // Refresh the page to start over
    window.location.reload();
  };

  const handleGeneratePptx = async () => {
    // Check server status first
    if (serverStatus !== 'available') {
      setIsCheckingServer(true);
      
      try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 5000);
        
        const response = await fetch('http://localhost:8001/health', {
          signal: controller.signal
        });
        
        clearTimeout(timeoutId);
        
        if (response.ok) {
          setServerStatus('available');
        } else {
          setServerStatus('unavailable');
          setServerError(`Server returned status ${response.status}`);
          alert('The backend server is not available. Please ensure it is running before generating a report.');
          setIsCheckingServer(false);
          return;
        }
      } catch (error) {
        console.error('Error checking server status:', error);
        setServerStatus('unavailable');
        
        let errorMessage = 'Unable to connect to the backend server';
        
        if (error instanceof TypeError && error.message.includes('fetch')) {
          errorMessage = 'Unable to connect to the server. Please ensure it is running at http://localhost:8001';
        } else if (error instanceof DOMException && error.name === 'AbortError') {
          errorMessage = 'Server connection timed out. The server might be running but is slow to respond.';
        }
        
        setServerError(errorMessage);
        alert(`${errorMessage} Please start the FastAPI server before generating a report.`);
        setIsCheckingServer(false);
        return;
      }
      
      setIsCheckingServer(false);
    }
    
    // If server is available, proceed with generating the report
    try {
      await generatePptxReport();
    } catch (error) {
      console.error('Error generating report:', error);
      
      // Check if it's a connection error and update server status
      if (error instanceof Error && 
          (error.message.includes('Unable to connect') || error.message.includes('fetch failed'))) {
        setServerStatus('unavailable');
        setServerError('Lost connection to the server during report generation');
      }
    }
  };
  
  // Direct download function
const handleDirectDownload = async () => {
  if (!pptxReportUrl || !reportId) {
    setDownloadError('Report ID not available. Please try downloading from the Reports page.');
    return;
  }
  
  // Show a loading state for download
  setIsDownloading(true);
  
  try {
    // Extract just the filename from the path
    const filename = getFilenameFromPath(pptxReportUrl);
    console.log(`Attempting direct download of: ${filename}`);
    
    // Use the reportId to construct the download URL
    const downloadUrl = `/api/download-report/${encodeURIComponent(reportId)}`;
    
    console.log(`Download URL: ${downloadUrl}`);
    
    // Use download link element for direct downloading
    if (downloadLinkRef.current) {
      downloadLinkRef.current.href = downloadUrl;
      downloadLinkRef.current.download = filename;
      downloadLinkRef.current.click();
      
      // Wait a moment before turning off loading state
      setTimeout(() => {
        setIsDownloading(false);
      }, 2000);
    }
  } catch (error) {
    setIsDownloading(false);
    console.error('Direct download failed:', error);
    
    // Show an error message to the user
    setDownloadError('Failed to download the report. Please try again or visit the Reports page to download.');
    
    // Clear the error after 5 seconds
    setTimeout(() => {
      setDownloadError(null);
    }, 5000);
  }
};
  // Render storage status badge
  const renderStorageStatus = () => {
    if (!pptxReportUrl) return null;
    
    switch (dbStorageStatus) {
      case 'pending':
        return (
          <div className="flex items-center space-x-2 text-amber-400 mt-2">
            <Database size={16} className="mr-2 animate-pulse" />
            <span>Saving report to database...</span>
          </div>
        );
      case 'success':
        return (
          <div className="flex items-center space-x-2 text-emerald-400 mt-2">
            <Database size={16} className="mr-2" />
            <span>Report saved to database (ID: {reportId?.substring(0, 8)}...)</span>
          </div>
        );
      case 'error':
        return (
          <div className="flex items-center space-x-2 text-red-400 mt-2">
            <AlertTriangle size={16} className="mr-2" />
            <span>Database error: {dbStorageError || 'Failed to save report'}</span>
          </div>
        );
      default:
        return null;
    }
  };

  // Render server status indicator
  const renderServerStatus = () => {
    if (serverStatus === 'checking' || isCheckingServer) {
      return (
        <div className="flex items-center space-x-2 text-amber-400 mt-2">
          <Loader size={16} className="animate-spin mr-2" />
          <span>Checking server status...</span>
        </div>
      );
    } else if (serverStatus === 'unavailable') {
      return (
        <div className="flex flex-col mt-2">
          <div className="flex items-center space-x-2 text-red-400">
            <ServerCrash size={16} className="mr-2" />
            <span>Backend server is not available</span>
            <button
              onClick={checkServerStatus}
              className="ml-2 p-1 bg-[rgba(79,70,229,0.2)] hover:bg-[rgba(79,70,229,0.3)] rounded"
              disabled={isCheckingServer}
            >
              <RefreshCw size={14} className={isCheckingServer ? "animate-spin" : ""} />
            </button>
          </div>
          {serverError && (
            <div className="text-xs text-red-300 mt-1 ml-6">
              {serverError}
            </div>
          )}
          <div className="text-xs text-amber-300 mt-1 ml-6">
            Make sure your FastAPI server is running at http://localhost:8001
          </div>
        </div>
      );
    } else {
      return (
        <div className="flex items-center space-x-2 text-green-400 mt-2">
          <div className="w-2 h-2 bg-green-400 rounded-full mr-2"></div>
          <span>Server is online</span>
        </div>
      );
    }
  };

  return (
    <>
      {/* Show loading indicator when generating PowerPoint */}
      <LongLoadingIndicator 
        isLoading={isGeneratingPptx} 
        initialMessage={`Generating presentation for ${report.coinName}...`}
      />
      
      <div className="flex flex-col min-h-screen bg-deep p-6">
        <header className="mb-8">
          <div className="flex justify-between items-start">
            <div>
              <h1 className="text-2xl font-bold">Due Diligence Report</h1>
              <h2 className="text-xl mt-1">{report.coinName}</h2>
              
              {/* Display server status */}
              {renderServerStatus()}
            </div>
            <div className="flex space-x-4">
              <button 
                onClick={handleGeneratePptx} 
                className="flex items-center border-2 border-[rgba(79,70,229,0.3)]"
                disabled={isGeneratingPptx || serverStatus === 'unavailable' || isCheckingServer}
                style={{ 
                  borderWidth: '2px',
                  opacity: isGeneratingPptx || serverStatus === 'unavailable' || isCheckingServer ? 0.5 : 1
                }}
              >
                {isGeneratingPptx ? (
                  <>
                    <Loader size={16} className="mr-2 animate-spin" />
                    <span className="animate-pulse">Processing...</span>
                  </>
                ) : (
                  <>
                    <Download size={16} className="mr-2" />
                    Generate PowerPoint
                  </>
                )}
              </button>
              
              {pptxReportUrl && (
                <>
                  {/* Direct Download Button */}
                  <button 
                    onClick={handleDirectDownload} 
                    className="flex items-center border-2 border-[rgba(79,70,229,0.3)]"
                    disabled={isDownloading}
                    style={{ borderWidth: '2px' }}
                  >
                    {isDownloading ? (
                      <>
                        <Loader size={16} className="mr-2 animate-spin" />
                        Downloading...
                      </>
                    ) : (
                      <>
                        <Save size={16} className="mr-2" />
                        Download Report
                      </>
                    )}
                  </button>
                  
                  {/* Hidden download link for direct download */}
                  <a 
                    ref={downloadLinkRef} 
                    className="hidden"
                    target="_blank"
                    rel="noopener noreferrer"
                  />
                </>
              )}
            </div>
          </div>
          <p className="text-sm text-secondary mt-2">
            Report generated on {new Date(report.timestamp).toLocaleString()}
          </p>
          {pptxError && (
            <div className="text-sm text-red-500 mt-2 p-2 rounded bg-red-500/10">
              <div className="font-semibold flex items-center">
                <AlertTriangle size={14} className="mr-1" />
                Error:
              </div>
              <div className="ml-5">{pptxError}</div>
            </div>
          )}
          {downloadError && (
            <div className="text-sm bg-red-500/10 p-3 rounded mt-2">
              <p className="text-red-500">{downloadError}</p>
              <p className="text-sm text-secondary mt-1">
                If download fails, you can access your report from the 
                <a href="/reports" className="text-primary-light ml-1 underline">
                  Reports page
                </a>.
              </p>
            </div>
          )}
          {renderStorageStatus()}
        </header>
        
        <div className="container flex-grow mb-8">
          <div className="mb-8 p-4 border-2 border-primary-color rounded-lg bg-bg-surface" style={{ borderWidth: '2px' }}>
            <h2 className="text-lg font-semibold mb-2">Report Summary</h2>
            <p>Selected {report.questions.length} questions across {questionsBySection.length} sections for {report.coinName}</p>
            {reportId && (
              <p className="mt-2">Report ID: {reportId}</p>
            )}
          </div>
          
          <div className="space-y-8">
            {questionsBySection.map(section => (
              <div key={section.id} className="card p-4 border-2 border-[rgba(79,70,229,0.3)]" style={{ borderWidth: '2px' }}>
                <h2 className="text-lg font-semibold mb-4">{section.title}</h2>
                <div className="space-y-3">
                  {section.questions.map(question => (
                    <div key={question.id} className="p-3 border-2 border-[rgba(79,70,229,0.3)] rounded-lg" style={{ borderWidth: '2px' }}>
                      <div className="flex items-center">
                        <span className="mr-2 text-primary-light">•</span>
                        <span>{question.text}</span>
                      </div>
                      
                      <div className="mt-2 ml-5 text-sm text-text-secondary">
                        <div className="italic">The answer will be generated in the PowerPoint presentation</div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
        
        <div className="flex justify-between mt-8">
          <button 
            onClick={handleStartNew} 
            className="flex items-center border-2 border-[rgba(79,70,229,0.3)]"
            style={{ borderWidth: '2px' }}
          >
            <ArrowLeft size={16} className="mr-2" />
            Start New Report
          </button>
          
          <a href="/reports" className="flex items-center text-primary-light">
            View Your Reports
            <ExternalLink size={16} className="ml-2" />
          </a>
        </div>
      </div>
    </>
  );
};

export default FinalReport;
// components/UserReportsList.tsx
import React, { useEffect, useState } from 'react';
import { Database, Download, Loader, Calendar, FileText, AlertTriangle } from 'lucide-react';
import { getCurrentUserId } from '../lib/authUtils';

interface Report {
  reportId: string;
  coinName: string;
  filePath: string;
  timestamp: number;
  questionCount: number;
  questions: string[];
}

const UserReportsList: React.FC = () => {
  const [reports, setReports] = useState<Report[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchReports = async () => {
      try {
        setIsLoading(true);
        setError(null);
        
        // Get the current user ID
        const userId = getCurrentUserId();
        if (!userId) {
          setError('User ID not found. Please log in to view your reports.');
          setIsLoading(false);
          return;
        }
        
        // Fetch reports from the API
        const response = await fetch(`/api/Reports?userId=${userId}`);
        
        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.message || 'Failed to fetch reports');
        }
        
        const data = await response.json();
        setReports(data || []);
      } catch (error) {
        console.error('Error fetching reports:', error);
        setError(error instanceof Error ? error.message : 'Unknown error');
      } finally {
        setIsLoading(false);
      }
    };
    
    fetchReports();
  }, []);
  
  const handleDownload = (filePath: string) => {
    try {
      // Extract just the filename from the path
      const parts = filePath.split(/[/\\]/);
      const filename = parts[parts.length - 1];
      
      // Create download URL
      const downloadUrl = `http://localhost:8001/download-pres/${encodeURIComponent(filename)}`;
      
      // Create a temporary link element and trigger the download
      const link = document.createElement('a');
      link.href = downloadUrl;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    } catch (error) {
      console.error('Download failed:', error);
      alert('Failed to download the report. Please try again.');
    }
  };

  if (isLoading) {
    return (
      <div className="flex flex-col items-center justify-center p-6">
        <Loader size={32} className="animate-spin mb-4" />
        <p>Loading your reports...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-6 border-2 border-red-300 rounded-lg">
        <div className="flex items-center mb-2">
          <AlertTriangle size={20} className="text-red-500 mr-2" />
          <h3 className="text-lg font-semibold">Error</h3>
        </div>
        <p>{error}</p>
      </div>
    );
  }

  if (reports.length === 0) {
    return (
      <div className="p-6 border-2 border-primary-color rounded-lg bg-bg-surface">
        <h3 className="text-lg font-semibold mb-2">No Reports</h3>
        <p>You haven't created any due diligence reports yet.</p>
        <button 
          onClick={() => window.location.href = '/duedel'}
          className="mt-4 flex items-center border-2 border-[rgba(79,70,229,0.3)]"
          style={{ borderWidth: '2px' }}
        >
          <FileText size={16} className="mr-2" />
          Create New Report
        </button>
      </div>
    );
  }

  return (
    <div className="container flex-grow mb-8">
      <h2 className="text-xl font-semibold mb-4">Your Due Diligence Reports</h2>
      
      <div className="space-y-4">
        {reports.map((report) => (
          <div key={report.reportId} className="card p-4 border-2 border-[rgba(79,70,229,0.3)] rounded-lg" style={{ borderWidth: '2px' }}>
            <div className="flex justify-between items-start">
              <div>
                <h3 className="text-lg font-medium">{report.coinName}</h3>
                <div className="flex items-center mt-1 text-text-secondary">
                  <Calendar size={14} className="mr-1" />
                  <span className="text-sm">{new Date(report.timestamp).toLocaleString()}</span>
                </div>
                <div className="flex items-center mt-1">
                  <Database size={14} className="mr-1" />
                  <span className="text-sm">ID: {report.reportId.substring(0, 8)}...</span>
                </div>
                <p className="mt-2">
                  {report.questionCount} questions selected
                </p>
              </div>
              
              <button 
                onClick={() => handleDownload(report.filePath)}
                className="flex items-center border-2 border-[rgba(79,70,229,0.3)]"
                style={{ borderWidth: '2px' }}
              >
                <Download size={16} className="mr-2" />
                Download
              </button>
            </div>
          </div>
        ))}
      </div>
      
      <button 
        onClick={() => window.location.href = '/duedel'}
        className="mt-6 flex items-center border-2 border-[rgba(79,70,229,0.3)]"
        style={{ borderWidth: '2px' }}
      >
        <FileText size={16} className="mr-2" />
        Create New Report
      </button>
    </div>
  );
};

export default UserReportsList;
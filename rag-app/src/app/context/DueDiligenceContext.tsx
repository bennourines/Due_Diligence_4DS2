// context/DueDiligenceContext.tsx
import React, { createContext, useContext, useState, useEffect } from 'react';
import { SECTIONS, QUESTIONS } from '../data/dueDiligenceData';
import { getCurrentUserId } from '../lib/authUtils';

// Define the FinalReportData type here
export interface FinalReportData {
  timestamp: number;
  questions: string[];
  coinName: string;
  reportId?: string;
  pptxUrl?: string;
}

interface DueDiligenceContextType {
  currentStep: number;
  selectedQuestions: Record<string, boolean>;
  completedSteps: Record<number, boolean>;
  finalReport: FinalReportData | null;
  coinName: string | null;
  setCoinName: (name: string) => void;
  handleNext: () => void;
  handlePrevious: () => void;
  handleStepClick: (stepId: number) => void;
  toggleQuestion: (id: string) => void;
  selectedCount: number;
  generateReport: () => void;
  isGeneratingPptx: boolean;
  generatePptxReport: () => Promise<string | null>;
  pptxReportUrl: string | null;
  pptxError: string | null;
  reportId: string | null;
  dbStorageStatus: 'idle' | 'pending' | 'success' | 'error';
  dbStorageError: string | null;
}

const DueDiligenceContext = createContext<DueDiligenceContextType | undefined>(undefined);

export const DueDiligenceProvider: React.FC<{children: React.ReactNode}> = ({ children }) => {
  const [currentStep, setCurrentStep] = useState<number>(1);
  const [selectedQuestions, setSelectedQuestions] = useState<Record<string, boolean>>({});
  const [completedSteps, setCompletedSteps] = useState<Record<number, boolean>>({});
  const [finalReport, setFinalReport] = useState<FinalReportData | null>(null);
  const [isGeneratingPptx, setIsGeneratingPptx] = useState<boolean>(false);
  const [pptxReportUrl, setPptxReportUrl] = useState<string | null>(null);
  const [pptxError, setPptxError] = useState<string | null>(null);
  const [coinName, setCoinName] = useState<string | null>(null);
  const [reportId, setReportId] = useState<string | null>(null);
  const [dbStorageStatus, setDbStorageStatus] = useState<'idle' | 'pending' | 'success' | 'error'>('idle');
  const [dbStorageError, setDbStorageError] = useState<string | null>(null);
  const [userId, setUserId] = useState<string | null>(null);

  // Get the user ID on mount
  useEffect(() => {
    if (typeof window !== 'undefined') {
      // Try to get the user ID from the authentication system
      const currentUserId = getCurrentUserId();
      setUserId(currentUserId);
      
      // If no user ID is found, we don't set a default
      // The user will need to log in to generate reports
      if (!currentUserId) {
        console.warn('No user ID found. User will need to log in to save reports.');
      }
    }
  }, []);

  // Calculate selected count for the current step
  const selectedCount = Object.keys(selectedQuestions)
    .filter(id => id.startsWith(`${currentStep}-`) && selectedQuestions[id])
    .length;

  const handleNext = () => {
    // Mark current step as completed
    setCompletedSteps(prev => ({
      ...prev,
      [currentStep]: true
    }));

    // If we're at the last step, generate the report
    if (currentStep === SECTIONS.length) {
      generateReport();
    } else {
      // Otherwise, move to the next step
      setCurrentStep(prev => prev + 1);
    }
  };

  const handlePrevious = () => {
    if (currentStep > 1) {
      setCurrentStep(prev => prev - 1);
    }
  };

  const handleStepClick = (stepId: number) => {
    // Only allow navigation to completed steps or the current step
    if (completedSteps[stepId] || stepId === currentStep || stepId === 1) {
      setCurrentStep(stepId);
    }
  };

  const toggleQuestion = (id: string) => {
    setSelectedQuestions(prev => ({
      ...prev,
      [id]: !prev[id]
    }));
  };

  const generateReport = () => {
    // Mark the last step as completed
    setCompletedSteps(prev => ({
      ...prev,
      [SECTIONS.length]: true
    }));

    // Create the final report object
    const selectedQuestionIds = Object.keys(selectedQuestions)
      .filter(id => selectedQuestions[id]);

    setFinalReport({
      timestamp: Date.now(),
      questions: selectedQuestionIds,
      coinName: coinName || 'Unknown Cryptocurrency'
    });
  };

  // Helper function to find a question by ID
  const findQuestionById = (questionId: string) => {
    // Parse the section ID from the question ID (format is "sectionId-questionIndex")
    const [sectionIdStr] = questionId.split('-');
    const sectionId = parseInt(sectionIdStr, 10);
    
    // Get the questions for this section
    const sectionQuestions = QUESTIONS[sectionId];
    
    // Find the specific question
    return sectionQuestions?.find(q => q.id === questionId);
  };

  const generatePptxReport = async (): Promise<string | null> => {
    try {
      // Check if we have a userId
      if (!userId) {
        throw new Error('User ID not available. Please log in to generate a report.');
      }
      
      setIsGeneratingPptx(true);
      setPptxError(null);
      setDbStorageStatus('pending');
      setDbStorageError(null);
      
      // Create properly formatted question items for the API
      const questionsToProcess = Object.keys(selectedQuestions)
        .filter(id => selectedQuestions[id])
        .map(id => {
          // Get section ID from the question ID
          const [sectionIdStr] = id.split('-');
          const sectionId = parseInt(sectionIdStr, 10);
          
          // Find the section title
          const section = SECTIONS.find(s => s.id === sectionId);
          
          // Find the question using our helper
          const question = findQuestionById(id);
          
          // Return properly formatted QuestionItem
          return {
            id,
            text: question?.text || `Question ${id}`,
            section_id: sectionId,
            section_title: section?.title || `Section ${sectionId}`
          };
        });
      
      console.log('Sending questions to API:', questionsToProcess);
      
      // Call the API to generate the presentation
      const response = await fetch('/api/generate-pptx', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          selected_questions: questionsToProcess,
          doc_id: 'all',
          coin_name: coinName || 'Cryptocurrency',
          userId: userId // Include the user ID
        }),
      });

      // Get response as text first to handle potential JSON parsing issues
      const responseText = await response.text();
      console.log('API Response Text:', responseText.substring(0, 200) + '...'); // Log first 200 chars

      // Try to parse the JSON
      let data;
      try {
        data = responseText ? JSON.parse(responseText) : {};
      } catch (parseError) {
        console.error('Failed to parse API response as JSON:', parseError);
        console.error('Raw response:', responseText.substring(0, 500)); // Log first 500 chars
        throw new Error('Invalid JSON response from server. Please check server logs.');
      }
      
      if (!response.ok) {
        console.error('Error response from API:', {
          status: response.status,
          data
        });
        throw new Error(data.error || data.message || 'Error generating presentation');
      }
      
      console.log('API response:', data);
      
      // Check if the response contains the expected file_path
      if (!data.file_path) {
        console.error('Missing file_path in API response:', data);
        throw new Error('Server response missing file path');
      }
      
      // Set the PPTX URL
      setPptxReportUrl(data.file_path);
      
      // Set the report ID if it was returned
      if (data.reportId) {
        setReportId(data.reportId);
        setDbStorageStatus('success');
        
        // Update the final report with the PPTX URL and report ID
        if (finalReport) {
          setFinalReport({
            ...finalReport,
            pptxUrl: data.file_path,
            reportId: data.reportId
          });
        }
      } else if (data.dbWarning) {
        // If there was a warning about DB storage
        setDbStorageStatus('error');
        setDbStorageError(data.dbWarning);
      }
      
      return data.file_path;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to generate presentation';
      setPptxError(errorMessage);
      setDbStorageStatus('error');
      setDbStorageError(errorMessage);
      console.error('Error generating presentation:', err);
      return null;
    } finally {
      setIsGeneratingPptx(false);
    }
  };

  const value = {
    currentStep,
    selectedQuestions,
    completedSteps,
    finalReport,
    coinName,
    setCoinName,
    handleNext,
    handlePrevious,
    handleStepClick,
    toggleQuestion,
    selectedCount,
    generateReport,
    isGeneratingPptx,
    generatePptxReport,
    pptxReportUrl,
    pptxError,
    reportId,
    dbStorageStatus,
    dbStorageError
  };

  return (
    <DueDiligenceContext.Provider value={value}>
      {children}
    </DueDiligenceContext.Provider>
  );
};

export const useDueDiligence = () => {
  const context = useContext(DueDiligenceContext);
  if (context === undefined) {
    throw new Error('useDueDiligence must be used within a DueDiligenceProvider');
  }
  return context;
};
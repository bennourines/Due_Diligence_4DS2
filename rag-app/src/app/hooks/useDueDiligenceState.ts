// hooks/useDueDiligenceState.ts
import { useState } from 'react';
import { SECTIONS } from '../data/dueDiligenceData';

export interface FinalReportData {
  questions: string[];
  timestamp: string;
  completed: boolean;
}

export interface DueDiligenceState {
  currentStep: number;
  selectedQuestions: Record<string, boolean>;
  completedSteps: Record<number, boolean>;
  finalReport: FinalReportData | null;
  handleNext: () => void;
  handlePrevious: () => void;
  handleStepClick: (stepId: number) => void;
  toggleQuestion: (questionId: string) => void;
  generateFinalReport: () => void;
  getSelectedQuestionsForStep: (stepId: number) => string[];
  selectedCount: number;
}

const useDueDiligenceState = (): DueDiligenceState => {
  const [currentStep, setCurrentStep] = useState<number>(1);
  const [selectedQuestions, setSelectedQuestions] = useState<Record<string, boolean>>({});
  const [completedSteps, setCompletedSteps] = useState<Record<number, boolean>>({});
  const [finalReport, setFinalReport] = useState<FinalReportData | null>(null);

  // Handle moving to the next step
  const handleNext = (): void => {
    if (currentStep < SECTIONS.length) {
      // Mark current step as completed
      setCompletedSteps(prev => ({
        ...prev,
        [currentStep]: true
      }));
      setCurrentStep(prevStep => prevStep + 1);
    } else {
      // Handle completion of the final step
      setCompletedSteps(prev => ({
        ...prev,
        [currentStep]: true
      }));
      
      // Generate the final report
      generateFinalReport();
    }
  };

  // Handle moving to the previous step
  const handlePrevious = (): void => {
    if (currentStep > 1) {
      setCurrentStep(prevStep => prevStep - 1);
    }
  };

  // Handle jumping to a specific step
  const handleStepClick = (stepId: number): void => {
    // Only allow jumping to completed steps or the current step + 1
    if (completedSteps[stepId] || stepId === currentStep || stepId === currentStep + 1) {
      setCurrentStep(stepId);
    }
  };

  // Handle question selection/deselection
  const toggleQuestion = (questionId: string): void => {
    setSelectedQuestions(prev => {
      const updatedSelected = { ...prev };
      
      if (prev[questionId]) {
        delete updatedSelected[questionId];
      } else {
        updatedSelected[questionId] = true;
      }
      
      return updatedSelected;
    });
  };

  // Generate the final report based on selected questions
  const generateFinalReport = (): void => {
    // Here you would format the selected questions into a report structure
    // For now, we'll just set the IDs of selected questions
    setFinalReport({
      questions: Object.keys(selectedQuestions),
      timestamp: new Date().toISOString(),
      completed: true
    });
  };

  // Get all selected questions for a specific step
  const getSelectedQuestionsForStep = (stepId: number): string[] => {
    // Get the section prefix for this step (e.g., "1-" for section 1)
    const sectionPrefix = `${stepId}-`;
    return Object.keys(selectedQuestions).filter(id => id.startsWith(sectionPrefix));
  };

  // Count selected questions for the current step
  const selectedCount = getSelectedQuestionsForStep(currentStep).length;
  
  return {
    currentStep,
    selectedQuestions,
    completedSteps,
    finalReport,
    handleNext,
    handlePrevious,
    handleStepClick,
    toggleQuestion,
    generateFinalReport,
    getSelectedQuestionsForStep,
    selectedCount
  };
};

export default useDueDiligenceState;
// components/NavigationButtons.tsx
import React from 'react';
import { ChevronRight, ChevronLeft, Check, AlertCircle } from 'lucide-react';

interface NavigationButtonsProps {
  currentStep: number;
  totalSteps: number;
  selectedCount: number;
  totalQuestions: number;
  handlePrevious: () => void;
  handleNext: () => void;
}

const NavigationButtons: React.FC<NavigationButtonsProps> = ({ 
  currentStep, 
  totalSteps, 
  selectedCount, 
  totalQuestions, 
  handlePrevious, 
  handleNext 
}) => {
  return (
    <div className="flex flex-col items-center w-full">
      {/* Warning message above buttons if needed */}
      {selectedCount === 0 && (
        <div className="text-yellow-400 mb-4 flex items-center text-sm">
          <AlertCircle size={16} className="mr-1" />
          No questions selected (you can still proceed)
        </div>
      )}
      
      {/* Navigation buttons side by side */}
      <div className="flex space-x-6">
        {/* Previous button */}
        <button 
          onClick={handlePrevious}
          disabled={currentStep === 1}
          className={`
            flex items-center p-4 rounded-lg border-2 border-[rgba(79,70,229,0.3)]
            ${currentStep === 1 ? 'opacity-50 cursor-not-allowed' : 'hover:border-primary-color'}
          `}
        >
          <ChevronLeft size={20} className="mr-2" />
          Previous
        </button>
        
        {/* Next/Complete button - right next to Previous */}
        {currentStep === totalSteps ? (
          <button 
            onClick={handleNext}
            className="flex items-center p-4 rounded-lg border-2 bg-primary-color text-white hover:bg-primary-dark"
          >
            Complete Report
            <Check size={20} className="ml-2" />
          </button>
        ) : (
          <button 
            onClick={handleNext}
            className="flex items-center p-4 rounded-lg border-2 border-[rgba(79,70,229,0.3)] hover:border-primary-color"
          >
            Next
            <ChevronRight size={20} className="ml-2" />
          </button>
        )}
      </div>
    </div>
  );
};

export default NavigationButtons;
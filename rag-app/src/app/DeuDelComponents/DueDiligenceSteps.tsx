// DeuDelComponents/DueDiligenceSteps.tsx
import React from 'react';
import { SECTIONS, QUESTIONS } from '../data/dueDiligenceData';
import StepIndicator from './StepIndicator';
import QuestionList from './QuestionList';
import NavigationButtons from './NavigationButtons';
import SectionStatus from './SectionStatus';
import FinalReport from './FinalReport';
import CoinSelection from './CoinSelection';
import { useDueDiligence } from '../context/DueDiligenceContext';
import { Grid } from 'lucide-react';

const DueDiligenceSteps: React.FC = () => {
  const { 
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
    selectedCount 
  } = useDueDiligence();

  // If no coin is selected yet, show the coin selection screen
  if (!coinName) {
    return <CoinSelection onSelectCoin={setCoinName} />;
  }

  // Use nullish coalescing to safely handle potentially undefined values
  const currentQuestions = QUESTIONS[currentStep] ?? [];
  const totalQuestions = currentQuestions.length;

  // If the final report is complete, show the report view
  if (finalReport && completedSteps[SECTIONS.length]) {
    return <FinalReport report={finalReport} selectedQuestions={selectedQuestions} />;
  }

  // Ensure we have a valid section
  const currentSection = SECTIONS.find(section => section.id === currentStep) || SECTIONS[0];

  const barStyle = {
    background: 'linear-gradient(to right, #4F46E5, #06B6D4)',
    borderRadius: '4px',
    marginTop: '4px',
    marginBottom: '4px'
  };

  return (
    <div className="flex flex-col min-h-screen bg-deep p-6">
      {/* Header */}
      <header className="mb-8">
        <h1 className="text-2xl font-bold">Cryptocurrency Due Diligence Report: {coinName}</h1>
        <h3 className="text-sm text-secondary mt-2">Complete the following sections to generate your report</h3>
      </header>

      {/* Step indicators */}
      <StepIndicator 
        sections={SECTIONS}
        currentStep={currentStep}
        completedSteps={completedSteps}
        onStepClick={handleStepClick}
        selectedCount={selectedCount}
        totalQuestions={totalQuestions}
      />

      {/* Current section content */}
      <div className="container flex-grow mb-8">
        <h2 className="text-xl font-semibold mb-4">{currentSection.title}</h2>
        
        <SectionStatus 
          selectedCount={selectedCount}
          totalQuestions={totalQuestions}
        />
        <div style={barStyle}>
          <QuestionList 
            questions={currentQuestions}
            selectedQuestions={selectedQuestions}
            toggleQuestion={toggleQuestion}
          />
        </div>
      </div>

      {/* Navigation buttons */}
      <NavigationButtons 
        currentStep={currentStep}
        totalSteps={SECTIONS.length}
        selectedCount={selectedCount}
        totalQuestions={totalQuestions}
        handlePrevious={handlePrevious}
        handleNext={handleNext}
      />
    </div>
  );
};

export default DueDiligenceSteps;
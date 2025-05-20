  import React from 'react';
  import { Section } from '../data/dueDiligenceData';

  interface StepIndicatorProps {
    sections: Section[];
    currentStep: number;
    completedSteps: Record<number, boolean>;
    onStepClick: (stepId: number) => void;
    selectedCount: number;
    totalQuestions: number;
  }

  const StepIndicator: React.FC<StepIndicatorProps> = ({ 
    sections, 
    currentStep, 
    completedSteps,
    onStepClick,
  }) => {
    // Find the current section
    const currentSection = sections.find(section => section.id === currentStep) || sections[0];
    
    // Inline styles to force horizontal layout
    const horizontalContainerStyle = {
      display: 'flex',
      flexDirection: 'row' as const,
      width: '100%',
      justifyContent: 'space-between',
      marginBottom: '8px'
    };
    
    const numberStyle = (isActive: boolean) => ({
      flex: '1',
      textAlign: 'center' as const,
      cursor: 'pointer',
      color: isActive ? 'var(--primary-color)' : 'var(--text-secondary)'
    });
    
    const barStyle = {
      width: '100%',
      height: '40px',
      background: 'linear-gradient(to right, #4F46E5, #06B6D4)',
      borderRadius: '4px',
      marginTop: '4px',
      marginBottom: '4px'
    };
    
    return (
      <div style={{ marginBottom: '24px' }}>
        {/* Force horizontal layout with inline styles */}
        <div style={horizontalContainerStyle}>
          {sections.map(section => (
            <div 
              key={section.id}
              style={numberStyle(section.id === currentStep)}
              onClick={() => onStepClick(section.id)}
            >
              {section.id}
            </div>
          ))}
        </div>
        
        {/* Blue bar */}
        <div style={barStyle}></div>
        

        <div style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', marginTop: '4px' }}>
          Step {currentStep} of {sections.length}
        </div>
      </div>
    );
  };

  export default StepIndicator;
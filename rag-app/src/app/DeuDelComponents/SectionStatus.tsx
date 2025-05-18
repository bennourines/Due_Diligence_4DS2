// components/SectionStatus.tsx
import React from 'react';

interface SectionStatusProps {
  selectedCount: number;
  totalQuestions: number;
}

const SectionStatus: React.FC<SectionStatusProps> = ({ selectedCount, totalQuestions }) => {
  return (
    <div className="mb-4 flex items-center">
      <div className="agent-status">
        {selectedCount > 0 ? (
          <span>{selectedCount} of {totalQuestions} questions selected</span>
        ) : (
          <span>No questions selected yet</span>
        )}
      </div>
    </div>
  );
};

export default SectionStatus;
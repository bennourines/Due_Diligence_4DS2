import React from 'react';
import { Check } from 'lucide-react';

interface Question {
  id: string;
  text: string;
}

interface QuestionListProps {
  questions: Question[];
  selectedQuestions: Record<string, boolean>;
  toggleQuestion: (id: string) => void;
}

const QuestionList: React.FC<QuestionListProps> = ({ questions, selectedQuestions, toggleQuestion }) => {
  return (
    <div className="w-full relative">
      {/* Increased spacing between questions */}
      <div className="space-y-6 w-full">
        {questions.map((question, qIndex) => (
          <div 
            key={question.id} 
            className="w-full mb-6" // Added margin-bottom for extra spacing
          >
            <button 
              onClick={() => toggleQuestion(question.id)}
              className={`
                p-4 w-full rounded-lg border-2 flex items-center justify-between
                ${selectedQuestions[question.id]
                  ? 'bg-primary-color hover:bg-primary-dark text-white' 
                  : 'border-[rgba(79,70,229,0.3)] hover:border-primary-color'
                }
              `}
            >
              <div className="flex items-center">
                {/* Question text */}
                <span>{question.text}</span>
              </div>
              
              {/* Check icon for selected questions */}
              {selectedQuestions[question.id] && (
                <Check size={20} className="ml-auto" />
              )}
            </button>
          </div>
        ))}
      </div>
    </div>
  );
};

export default QuestionList;
import React, { useState, useEffect } from 'react';
import { Loader } from 'lucide-react';

interface LongLoadingIndicatorProps {
  initialMessage?: string;
  isLoading: boolean;
}

const LongLoadingIndicator: React.FC<LongLoadingIndicatorProps> = ({ 
  initialMessage = "Generating presentation...", 
  isLoading 
}) => {
  const [elapsedTime, setElapsedTime] = useState(0);
  const [message, setMessage] = useState(initialMessage);
  
  // Update the elapsed time every second
  useEffect(() => {
    if (!isLoading) return;
    
    const interval = setInterval(() => {
      setElapsedTime((prev) => prev + 1);
    }, 1000);
    
    return () => clearInterval(interval);
  }, [isLoading]);
  
  // Update the message based on elapsed time
  useEffect(() => {
    if (elapsedTime > 600 * 1000 * 60) {
      setMessage("This is taking longer than expected. We're still working on it...");
    } else if (elapsedTime > 2 * 60) {
      setMessage("The presentation is being generated. This may take several minutes...");
    }
  }, [elapsedTime]);
  
  // Format the elapsed time as MM:SS
  const formatTime = (seconds: number): string => {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    
    return `${minutes.toString().padStart(2, '0')}:${remainingSeconds.toString().padStart(2, '0')}`;
  };
  
  if (!isLoading) return null;
  
  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white p-6 rounded-lg shadow-lg max-w-md w-full">
        <div className="flex flex-col items-center space-y-4">
          <Loader size={48} className="text-primary-color animate-spin" />
          
          <h3 className="text-lg font-semibold text-center">{message}</h3>
          
          <div className="w-full bg-gray-200 rounded-full h-2.5">
            <div 
              className="bg-primary-color h-2.5 rounded-full pulse-element"
              style={{ width: '100%' }}
            ></div>
          </div>
          
          <p className="text-sm text-gray-600">
            Elapsed time: {formatTime(elapsedTime)}
          </p>
          
          <div className="text-xs text-gray-500 text-center">
            <p>Please be patient. The AI is answering your questions and creating slides.</p>
            <p className="mt-1">Do not close or refresh this page.</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default LongLoadingIndicator;
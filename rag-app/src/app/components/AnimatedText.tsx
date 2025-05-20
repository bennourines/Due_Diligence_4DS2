'use client';

import React, { useState, useEffect, useRef } from 'react';
import { Typography, Box, SxProps } from '@mui/material';

interface AnimatedTextProps {
  text: string;
  className?: string;
  sx?: SxProps;
  role?: 'user' | 'assistant';
  delay?: number;
  onComplete?: () => void;
}

const AnimatedText: React.FC<AnimatedTextProps> = ({
  text,
  className,
  sx,
  role = 'assistant',
  delay = 20,
  onComplete
}) => {
  const [displayedText, setDisplayedText] = useState('');
  const [isComplete, setIsComplete] = useState(false);
  const [formattedHtml, setFormattedHtml] = useState('');
  const timeoutRef = useRef<NodeJS.Timeout | null>(null);

  // TextFormatter logic: Format text with code blocks, links, etc.
  const formatText = (input: string) => {
    if (!input) return input;

    let formattedText = input;

    // Handle code blocks with backticks
    formattedText = formattedText.replace(
      /```([\s\S]*?)```/g, 
      (match, code) => {
        return `<pre><code>${code.trim()}</code></pre>`;
      }
    );

    // Handle inline code
    formattedText = formattedText.replace(
      /`([^`]+)`/g, 
      (match, code) => {
        return `<code>${code}</code>`;
      }
    );

    // Handle line breaks (preserve them)
    formattedText = formattedText.replace(/\n/g, '<br />');

    // Handle URLs
    formattedText = formattedText.replace(
      /(https?:\/\/[^\s]+)/g,
      (match) => {
        return `<a href="${match}" target="_blank" rel="noopener noreferrer">${match}</a>`;
      }
    );

    // Handle bold text
    formattedText = formattedText.replace(
      /\*\*(.*?)\*\*/g,
      (match, text) => {
        return `<strong>${text}</strong>`;
      }
    );

    // Handle italic text
    formattedText = formattedText.replace(
      /\*(.*?)\*/g,
      (match, text) => {
        return `<em>${text}</em>`;
      }
    );

    return formattedText;
  };

  // Format text when animation completes
  useEffect(() => {
    if (isComplete) {
      try {
        const formatted = formatText(displayedText);
        setFormattedHtml(formatted);
      } catch (error) {
        console.error('Error formatting text:', error);
        setFormattedHtml(displayedText);
      }
    }
  }, [displayedText, isComplete]);

  // Animation effect
  useEffect(() => {
    // For user messages, show immediately
    if (role === 'user') {
      setDisplayedText(text);
      setIsComplete(true);
      onComplete?.();
      return;
    }
    
    // Reset state for new messages
    setDisplayedText('');
    setIsComplete(false);
    setFormattedHtml('');
    
    let index = 0;
    
    // Clear any existing timeout
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }
    
    // Handle special content like code blocks more naturally
    const animateText = () => {
      if (index <= text.length) {
        const currentText = text.substring(0, index);
        setDisplayedText(currentText);
        
        // Determine next delay based on character type
        let nextDelay = delay;
        
        // Speed up for code blocks and slow down for important punctuation
        if (text[index] === '`' || text[index] === '\n') {
          nextDelay = delay / 2; // Speed up for code formatting
        } else if ('.!?:'.includes(text[index] || '')) {
          nextDelay = delay * 3; // Slow down for end of sentences
        }
        
        index++;
        timeoutRef.current = setTimeout(animateText, nextDelay);
      } else {
        setIsComplete(true);
        onComplete?.();
      }
    };
    
    // Start animation
    timeoutRef.current = setTimeout(animateText, delay);
    
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, [text, delay, role, onComplete]);

  // Simple rendering for in-progress text
  const renderInProgressText = () => {
    return displayedText.split('\n').map((line, i) => (
      <React.Fragment key={i}>
        {line}
        {i < displayedText.split('\n').length - 1 && <br />}
      </React.Fragment>
    ));
  };

  return (
    <Typography
      component="div"
      className={`${className || ''} animated-text`}
      sx={{
        whiteSpace: 'pre-line',
        ...sx,
        position: 'relative',
        '& code': {
          fontFamily: 'monospace',
          backgroundColor: role === 'assistant' ? 'rgba(0, 0, 0, 0.2)' : 'rgba(255, 255, 255, 0.15)',
          padding: '2px 4px',
          borderRadius: '3px',
          color: role === 'assistant' ? 'var(--yellow-color)' : 'var(--black-color)',
        },
        '& pre': {
          backgroundColor: role === 'assistant' ? 'rgba(0, 0, 0, 0.3)' : 'rgba(255, 255, 255, 0.1)',
          padding: '8px',
          borderRadius: '4px',
          overflowX: 'auto',
          width: '100%',
          fontFamily: 'monospace',
        },
        '& pre code': {
          backgroundColor: 'transparent',
          padding: 0,
          display: 'block',
        },
        '& a': {
          color: 'inherit',
          textDecoration: 'underline',
        },
        '& strong': {
          fontWeight: 'bold',
        },
        '& em': {
          fontStyle: 'italic',
        },
      }}
    >
      {isComplete ? (
        <div
          dangerouslySetInnerHTML={{ __html: formattedHtml }}
          className="formatted-content"
        />
      ) : (
        renderInProgressText()
      )}
      
      {/* Blinking cursor at the end while typing */}
      {!isComplete && role === 'assistant' && (
        <Box 
          component="span" 
          sx={{ 
            display: 'inline-block',
            width: '2px',
            height: '1em',
            backgroundColor: 'var(--yellow-color)',
            marginLeft: '2px',
            verticalAlign: 'middle',
            animation: 'blink 0.8s step-end infinite'
          }}
        />
      )}
      
      {/* Add cursor blinking animation */}
      <style jsx global>{`
        @keyframes blink {
          0%, 100% { opacity: 1; }
          50% { opacity: 0; }
        }
        
        .animated-text pre {
          margin: 8px 0;
        }
      `}</style>
    </Typography>
  );
};

export default AnimatedText;
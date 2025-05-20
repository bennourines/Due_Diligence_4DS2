'use client';

import React, { useState, useEffect, useRef } from 'react';
import { Typography, Box, SxProps } from '@mui/material';
import { marked } from 'marked'; // You'll need to install this package

interface AdvancedAnimatedTextProps {
  text: string;
  className?: string;
  sx?: SxProps;
  role?: 'user' | 'assistant';
  delay?: number;
  onComplete?: () => void;
  enableMarkdown?: boolean;
}

const AdvancedAnimatedText: React.FC<AdvancedAnimatedTextProps> = ({
  text,
  className,
  sx,
  role = 'assistant',
  delay = 20,
  onComplete,
  enableMarkdown = true
}) => {
  const [displayedText, setDisplayedText] = useState('');
  const [isComplete, setIsComplete] = useState(false);
  const [htmlContent, setHtmlContent] = useState('');
  const timeoutRef = useRef<NodeJS.Timeout | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Process markdown when text changes
  useEffect(() => {
    if (enableMarkdown && isComplete) {
      try {
        // Configure marked options if needed
        marked.setOptions({
          breaks: true,
          gfm: true,
          // Remove the headerIds option as it doesn't exist in MarkedOptions
        });
        
        // Using async/await with proper typing handling
        const parseMarkdown = async () => {
          try {
            const parsedHtml = await marked.parse(displayedText);
            // Ensure the result is a string
            if (typeof parsedHtml === 'string') {
              setHtmlContent(parsedHtml);
            } else {
              console.error('Marked parse did not return a string');
              setHtmlContent(displayedText);
            }
          } catch (error) {
            console.error('Error parsing markdown:', error);
            setHtmlContent(displayedText);
          }
        };
        
        parseMarkdown();
      } catch (error) {
        console.error('Error in markdown processing:', error);
        setHtmlContent(displayedText);
      }
    }
  }, [displayedText, isComplete, enableMarkdown]);

  // Animation effect
  useEffect(() => {
    // For user messages or when markdown is disabled, show immediately
    if (role === 'user') {
      setDisplayedText(text);
      setIsComplete(true);
      onComplete?.();
      return;
    }
    
    // Reset state for new messages
    setDisplayedText('');
    setIsComplete(false);
    setHtmlContent('');
    
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

  // Render text differently based on completion and markdown setting
  const renderContent = () => {
    if (enableMarkdown && isComplete) {
      return (
        <div
          dangerouslySetInnerHTML={{ __html: htmlContent }}
          className="markdown-content"
        />
      );
    } else {
      // Simple text rendering with line breaks preserved
      return displayedText.split('\n').map((line, i) => (
        <React.Fragment key={i}>
          {line}
          {i < displayedText.split('\n').length - 1 && <br />}
        </React.Fragment>
      ));
    }
  };

  return (
    <Typography
      component="div"
      className={className}
      ref={containerRef}
      sx={{
        whiteSpace: 'pre-line',
        ...sx,
        position: 'relative',
        '& code': {
          fontFamily: 'monospace',
          backgroundColor: 'rgba(0, 0, 0, 0.1)',
          padding: '2px 4px',
          borderRadius: '3px',
        },
        '& pre': {
          backgroundColor: role === 'assistant' ? 'rgba(0, 0, 0, 0.3)' : 'rgba(255, 255, 255, 0.1)',
          padding: '8px',
          borderRadius: '4px',
          overflowX: 'auto',
          fontFamily: 'monospace',
        },
      }}
    >
      {renderContent()}
      
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
        
        .markdown-content a {
          color: inherit;
          text-decoration: underline;
        }
        
        .markdown-content ul, .markdown-content ol {
          padding-left: 20px;
          margin: 8px 0;
        }
        
        .markdown-content blockquote {
          border-left: 3px solid;
          margin-left: 0;
          padding-left: 10px;
          opacity: 0.8;
        }
      `}</style>
    </Typography>
  );
};

export default AdvancedAnimatedText;
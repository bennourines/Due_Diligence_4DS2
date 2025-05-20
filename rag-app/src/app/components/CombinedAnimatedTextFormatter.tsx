'use client';

import React, { useState, useEffect, useRef, ReactNode } from 'react';
import { Typography, Box, Link, SxProps, Theme } from '@mui/material';
import { marked } from 'marked';

interface CombinedAnimatedTextFormatterProps {
  text: string;
  className?: string;
  sx?: SxProps<Theme>;
  role?: 'user' | 'assistant';
  onComplete?: () => void;
  enableMarkdown?: boolean;
  isNewMessage?: boolean; // New prop to determine if this is a new message
}

const CombinedAnimatedTextFormatter: React.FC<CombinedAnimatedTextFormatterProps> = ({
  text,
  className,
  sx,
  role = 'assistant',
  onComplete,
  enableMarkdown = true,
  isNewMessage = false // Default to false (no animation for existing messages)
}) => {
  // Fixed delay constant
  const delay = 25; // ms

  const [displayedText, setDisplayedText] = useState('');
  const [isComplete, setIsComplete] = useState(!isNewMessage); // Initialize as complete if not a new message
  const [htmlContent, setHtmlContent] = useState<ReactNode[]>([]);
  const timeoutRef = useRef<NodeJS.Timeout | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [skipAnimations, setSkipAnimations] = useState(false);
  const [initialRender, setInitialRender] = useState(true);

  // TextFormatter logic: Format the text with bold text, code blocks and clickable links
  const formatText = (text: string): ReactNode[] => {
    if (!text) return [];

    // Create a version with markdown-processed HTML for the final rendering
    let processedText = text;

    // Handle code blocks with backticks
    processedText = processedText.replace(
      /```([\s\S]*?)```/g, 
      (match, code) => {
        return `<pre><code>${code.trim()}</code></pre>`;
      }
    );

    // Handle inline code
    processedText = processedText.replace(
      /`([^`]+)`/g, 
      (match, code) => {
        return `<code>${code}</code>`;
      }
    );

    // Split the text by sections that need special formatting
    // First, handle text between ** markers for bold
    const boldRegex = /\*\*(.*?)\*\*/g;
    // Then handle URLs
    const urlRegex = /(https?:\/\/[^\s]+)/g;
    
    // Process the text to handle both patterns
    let formattedNodes: ReactNode[] = [];
    let lastIndex = 0;
    let currentText = processedText;

    // Handle specific phrases to make bold (like "Relevant web search results:")
    const specificBoldPhrases: string[] = ["Relevant web search results:"];
    
    for (const phrase of specificBoldPhrases) {
      const phraseIndex = currentText.indexOf(phrase);
      if (phraseIndex !== -1) {
        // Replace the phrase with asterisks to be caught by the bold regex
        currentText = 
          currentText.substring(0, phraseIndex) + 
          '**' + phrase + '**' + 
          currentText.substring(phraseIndex + phrase.length);
      }
    }
    
    // Create temporary structure to hold all split parts and their types
    interface TextPart {
      type: 'text' | 'bold' | 'code' | 'codeblock';
      content: string;
      index: number;
    }
    
    const parts: TextPart[] = [];
    
    // Process bold sections
    let boldMatch: RegExpExecArray | null;
    while ((boldMatch = boldRegex.exec(currentText)) !== null) {
      if (boldMatch.index > lastIndex) {
        parts.push({
          type: 'text',
          content: currentText.substring(lastIndex, boldMatch.index),
          index: lastIndex
        });
      }
      
      parts.push({
        type: 'bold',
        content: boldMatch[1],
        index: boldMatch.index
      });
      
      lastIndex = boldMatch.index + boldMatch[0].length;
    }
    
    // Add the remaining text
    if (lastIndex < currentText.length) {
      parts.push({
        type: 'text',
        content: currentText.substring(lastIndex),
        index: lastIndex
      });
    }
    
    // Reset for URL processing
    lastIndex = 0;
    
    // Process each part to look for URLs and code
    for (let i = 0; i < parts.length; i++) {
      const part = parts[i];
      
      if (part.type === 'bold') {
        // Don't process URLs inside bold text
        formattedNodes.push(
          <Box component="strong" key={`bold-${part.index}`} display="inline">
            {part.content}
          </Box>
        );
      } else if (part.type === 'code') {
        formattedNodes.push(
          <Box 
            component="code" 
            key={`code-${part.index}`} 
            sx={{
              fontFamily: 'var(--font-primary)',
              backgroundColor: role === 'assistant' ? 'rgba(0, 0, 0, 0.2)' : 'rgba(255, 255, 255, 0.15)',
              padding: '2px 4px',
              borderRadius: '3px',
              color: role === 'assistant' ? 'var(--yellow-color)' : 'var(--black-color)',
            }}
          >
            {part.content}
          </Box>
        );
      } else if (part.type === 'codeblock') {
        formattedNodes.push(
          <Box 
            component="pre" 
            key={`codeblock-${part.index}`}
            sx={{
              backgroundColor: role === 'assistant' ? 'rgba(0, 0, 0, 0.3)' : 'rgba(255, 255, 255, 0.1)',
              padding: '8px',
              borderRadius: '4px',
              overflowX: 'auto',
              width: '100%',
             fontFamily: 'var(--font-primary)',
            }}
          >
            <Box component="code">{part.content}</Box>
          </Box>
        );
      } else {
        // Look for URLs in regular text
        const text = part.content;
        lastIndex = 0;
        let urlMatch: RegExpExecArray | null;
        const textParts: ReactNode[] = [];
        
        while ((urlMatch = urlRegex.exec(text)) !== null) {
          if (urlMatch.index > lastIndex) {
            textParts.push(text.substring(lastIndex, urlMatch.index));
          }
          
          textParts.push(
            <Link
              key={`link-${part.index}-${urlMatch.index}`}
              href={urlMatch[0]}
              target="_blank"
              rel="noopener noreferrer"
              sx={{ color: role === 'assistant' ? 'var(--yellow-color)' : 'var(--black-color)', textDecoration: 'underline' }}
            >
              {urlMatch[0]}
            </Link>
          );
          
          lastIndex = urlMatch.index + urlMatch[0].length;
        }
        
        // Add remaining text
        if (lastIndex < text.length) {
          textParts.push(text.substring(lastIndex));
        }
        
        // Add all text parts to processed text
        for (let j = 0; j < textParts.length; j++) {
          if (typeof textParts[j] === 'string') {
            formattedNodes.push(<React.Fragment key={`text-${part.index}-${j}`}>{textParts[j]}</React.Fragment>);
          } else {
            formattedNodes.push(textParts[j]);
          }
        }
      }
    }
    
    return formattedNodes;
  };

  // Process markdown when text changes and animation completes
  useEffect(() => {
    if (isComplete) {
      try {
        // Use TextFormatter logic to format the text
        const formattedNodes = formatText(text); // Use the full text, not displayedText
        setHtmlContent(formattedNodes);
        
        // Apply additional markdown formatting if enabled
        if (enableMarkdown) {
          // Configure marked options
          marked.setOptions({
            breaks: true,
            gfm: true,
          });
          
          // Async markdown parsing for complex elements
          const parseMarkdown = async () => {
            try {
              // For complex markdown elements that TextFormatter doesn't handle
              const parsedHtml = await marked.parse(text); // Use the full text
              
              // We keep the TextFormatter processing for compatibility,
              // but could use the marked output for a more complete markdown rendering
              // if desired in the future
            } catch (error) {
              console.error('Error parsing markdown:', error);
            }
          };
          
          parseMarkdown();
        }
      } catch (error) {
        console.error('Error in formatting text:', error);
        setHtmlContent([text]); // Fall back to full text
      }
    }
  }, [text, isComplete, enableMarkdown, role]);

  // Set initial state based on whether this is a new message
  useEffect(() => {
    if (initialRender) {
      setInitialRender(false);
      
      if (!isNewMessage) {
        // For existing messages, immediately show the full text without animation
        setDisplayedText(text);
        setIsComplete(true);
        onComplete?.();
      }
    }
  }, [initialRender, isNewMessage, text, onComplete]);

  // Animation effect - only runs for new messages
  useEffect(() => {
    // Skip animation for existing messages or user messages
    if (!isNewMessage || role === 'user') {
      setDisplayedText(text);
      setIsComplete(true);
      onComplete?.();
      return;
    }
    
    // Reset state for new messages
    if (isNewMessage && initialRender === false) {
      setDisplayedText('');
      setIsComplete(false);
      setHtmlContent([]);
      setSkipAnimations(false);
      
      let index = 0;
      
      // Clear any existing timeout
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
      
      // Handle special content like code blocks more naturally
      const animateText = () => {
        if (skipAnimations) {
          // Skip rest of animation if user requested
          setDisplayedText(text);
          setIsComplete(true);
          onComplete?.();
          return;
        }
        
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
    }
  }, [text, delay, role, onComplete, skipAnimations, isNewMessage, initialRender]);

  // Simple version for in-progress text
  const renderInProgressText = () => {
    // Simple text rendering with line breaks preserved
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
      className={className}
      ref={containerRef}
      onClick={() => setSkipAnimations(true)} // Skip animation on click
      sx={{
        whiteSpace: 'pre-line',
        ...sx,
        position: 'relative',
        '& code': {
          fontFamily: 'var(--font-primary)',
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
          fontFamily: 'var(--font-primary)',
          margin: '8px 0',
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
      }}
    >
      {isComplete ? (
        <Box className="formatted-content">
          {htmlContent.map((node, i) => (
            <React.Fragment key={i}>{node}</React.Fragment>
          ))}
        </Box>
      ) : (
        renderInProgressText()
      )}
      
      {/* Blinking cursor at the end while typing - only for new messages being animated */}
      {!isComplete && isNewMessage && role === 'assistant' && (
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
        
        .formatted-content ul, .formatted-content ol {
          padding-left: 20px;
          margin: 8px 0;
        }
        
        .formatted-content blockquote {
          border-left: 3px solid;
          margin-left: 0;
          padding-left: 10px;
          opacity: 0.8;
        }
      `}</style>
    </Typography>
  );
};

export default CombinedAnimatedTextFormatter;
import React, { ReactNode } from 'react';
import { Box, Link } from '@mui/material';
import { SxProps, Theme } from '@mui/material/styles';

interface TextFormatterProps {
  text: string;
  className?: string;
  sx?: SxProps<Theme>;
}

const TextFormatter: React.FC<TextFormatterProps> = ({ text, className, sx }) => {
  // Function to format the text with bold text and clickable links
  const formatText = (text: string): ReactNode[] => {
    if (!text) return [];

    // Split the text by sections that need special formatting
    // First, handle text between ** markers
    const boldRegex = /\*\*(.*?)\*\*/g;
    // Then handle URLs
    const urlRegex = /(https?:\/\/[^\s]+)/g;
    
    // Process the text to handle both patterns
    let processedText: ReactNode[] = [];
    let lastIndex = 0;
    let currentText = text;

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
      type: 'text' | 'bold';
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
    
    // Process each part to look for URLs
    for (let i = 0; i < parts.length; i++) {
      const part = parts[i];
      
      if (part.type === 'bold') {
        // Don't process URLs inside bold text
        processedText.push(
          <Box component="strong" key={`bold-${part.index}`} display="inline">
            {part.content}
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
              sx={{ color: 'primary.main' }}
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
            processedText.push(<React.Fragment key={`text-${part.index}-${j}`}>{textParts[j]}</React.Fragment>);
          } else {
            processedText.push(textParts[j]);
          }
        }
      }
    }
    
    return processedText;
  };

  return (
    <Box 
      className={className} 
      sx={{ 
        whiteSpace: 'pre-line',
        ...sx 
      }}
    >
      {formatText(text).map((part, i) => (
        <React.Fragment key={i}>{part}</React.Fragment>
      ))}
    </Box>
  );
};

export default TextFormatter;
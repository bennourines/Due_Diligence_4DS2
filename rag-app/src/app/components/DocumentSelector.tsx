// Fixed DocumentSelector with proper filename extraction
import { useState, useEffect } from 'react';
import { Box, Button, FormControl, Select, MenuItem, InputLabel, Typography, SelectChangeEvent } from '@mui/material';

// Add the Message interface definition
interface Message {
  _id?: string;
  role: 'user' | 'assistant';
  content: string;
  sources?: Array<{
    source: string;
    doc_id: string;
    page: number;
    file_type: string;
  }>;
  createdAt?: string;
  isNew?: boolean;
}

interface DocumentSelectorProps {
  message: Message;
  onGenerateQuestions: (fileId: string) => void;
  isLoading: boolean;
}

// Interface to store both display name and ID
interface DocumentInfo {
  displayName: string;
  id: string;
}

const DocumentSelector = ({ message, onGenerateQuestions, isLoading }: DocumentSelectorProps) => {
  const [selectedFile, setSelectedFile] = useState<string>('all');
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);

  useEffect(() => {
    // Extract filenames from the success message
    const content = message.content;
    console.log("Processing message content:", content);
    
    // FIXED REGEX: Look for PDF filenames more precisely
    const pdfRegex = /\b([a-zA-Z0-9\s_-]+\.pdf)\b/gi;
    const matches = content.match(pdfRegex);
    
    const extractedDocs: DocumentInfo[] = [];
    
    if (matches && matches.length > 0) {
      console.log("Extracted PDF filenames:", matches);
      
      // Process each filename
      matches.forEach(filename => {
        // Clean the filename - remove any trailing punctuation
        const cleanFilename = filename.replace(/[.,;:]$/, '');
        
        // For document ID, remove extension and replace spaces with underscores
        const docId = cleanFilename.split('.')[0].replace(/\s+/g, '_');
        
        extractedDocs.push({
          displayName: cleanFilename,
          id: docId
        });
      });
    } else {
      // Fallback: Try to extract using a different approach
      // Look for a specific pattern like "uploaded and processed: file1.pdf, file2.pdf"
      const uploadPattern = /processed:?\s*([^.]+\.pdf(?:,\s*[^.]+\.pdf)*)/i;
      const uploadMatch = content.match(uploadPattern);
      
      if (uploadMatch && uploadMatch[1]) {
        // Split the file list by commas
        const fileList = uploadMatch[1].split(',');
        fileList.forEach(file => {
          const cleanFile = file.trim();
          // Make sure it looks like a PDF
          if (cleanFile.toLowerCase().endsWith('.pdf')) {
            const docId = cleanFile.split('.')[0].replace(/\s+/g, '_');
            extractedDocs.push({
              displayName: cleanFile,
              id: docId
            });
          }
        });
      }
    }
    
    // If we still found nothing, try one more approach
    if (extractedDocs.length === 0) {
      // Split the content into words and look for .pdf
      const words = content.split(/\s+/);
      for (let i = 0; i < words.length; i++) {
        if (words[i].toLowerCase().endsWith('.pdf')) {
          // Clean up any punctuation
          const cleanFile = words[i].replace(/[.,;:]$/, '');
          const docId = cleanFile.split('.')[0].replace(/\s+/g, '_');
          extractedDocs.push({
            displayName: cleanFile,
            id: docId
          });
        }
      }
    }
    
    // Ultimate fallback - if we still can't find the files
    if (extractedDocs.length === 0) {
      // Try to get file count from the message
      const fileCountMatch = content.match(/(\d+)\s*files?\s*ha(?:ve|s)/i);
      if (fileCountMatch && fileCountMatch[1]) {
        const count = parseInt(fileCountMatch[1], 10);
        for (let i = 1; i <= count; i++) {
          extractedDocs.push({
            displayName: `Document ${i}`,
            id: `document_${i}`
          });
        }
      } else {
        // Last resort fallback
        extractedDocs.push({
          displayName: 'Document',
          id: 'document'
        });
      }
    }
    
    console.log("Final document list:", extractedDocs);
    setDocuments(extractedDocs);
    // Default to all documents
    setSelectedFile('all');
    
  }, [message]);

  const handleFileChange = (event: SelectChangeEvent<string>) => {
    setSelectedFile(event.target.value);
  };

  const handleGenerateClick = () => {
    if (selectedFile === 'all') {
      onGenerateQuestions('all');
    } else {
      // Find the document ID for the selected display name
      const selectedDoc = documents.find(doc => doc.displayName === selectedFile);
      const docId = selectedDoc ? selectedDoc.id : selectedFile.split('.')[0].replace(/\s+/g, '_');
      
      console.log(`Generating questions for: ${selectedFile} (ID: ${docId})`);
      onGenerateQuestions(docId);
    }
  };

  return (
    <Box sx={{ mt: 2 }}>
      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
        <Typography
          variant="subtitle2"
          sx={{
            color: 'var(--text-accent)',
            fontFamily: 'var(--font-primary)',
            letterSpacing: '0.5px',
            fontWeight: 'bold'
          }}
        >
          Generate Questions From Your Documents
        </Typography>
        
        {/* Document selector dropdown */}
        <FormControl 
          variant="outlined" 
          size="small"
          sx={{
            minWidth: 300,
            '& .MuiOutlinedInput-root': {
              color: 'var(--text-primary)',
              borderColor: 'rgba(79, 70, 229, 0.3)',
              backgroundColor: 'rgba(15, 23, 42, 0.3)',
              '&:hover .MuiOutlinedInput-notchedOutline': {
                borderColor: 'var(--primary-color)',
              },
              '&.Mui-focused .MuiOutlinedInput-notchedOutline': {
                borderColor: 'var(--primary-color)',
              }
            },
            '& .MuiInputLabel-root': {
              color: 'var(--text-secondary)',
            }
          }}
        >
          <InputLabel id="document-select-label">Document</InputLabel>
          <Select
            labelId="document-select-label"
            value={selectedFile}
            onChange={handleFileChange}
            label="Document"
          >
            <MenuItem value="all">All Documents</MenuItem>
            {documents.map((doc, index) => (
              <MenuItem key={index} value={doc.displayName}>{doc.displayName}</MenuItem>
            ))}
          </Select>
        </FormControl>
        
        {/* Display document ID for debugging (can be removed in production) */}
        <Typography variant="caption" sx={{ color: 'rgba(255, 255, 255, 0.5)' }}>
          {selectedFile !== 'all' ? 
            `Document ID: ${documents.find(doc => doc.displayName === selectedFile)?.id || 'unknown'}` : 
            'Document ID: all'}
        </Typography>
        
        {/* Generate button */}
        <Button
          variant="outlined"
          size="small"
          onClick={handleGenerateClick}
          disabled={isLoading}
          className="pulse-element"
          sx={{
            borderColor: 'var(--primary-color)',
            color: 'var(--text-primary)',
            borderRadius: 'var(--border-radius)',
            fontFamily: 'var(--font-primary)',
            letterSpacing: '1px',
            textTransform: 'uppercase',
            fontWeight: 500,
            position: 'relative',
            overflow: 'hidden',
            padding: '8px 16px',
            transition: 'var(--transition-smooth)',
            backdropFilter: 'blur(4px)',
            alignSelf: 'flex-start',
            '&:hover': {
              borderColor: 'transparent',
              backgroundColor: 'rgba(79, 70, 229, 0.1)',
              boxShadow: 'var(--shadow-glow)'
            },
            '&.Mui-disabled': {
              borderColor: 'rgba(79, 70, 229, 0.3)',
              color: 'rgba(79, 70, 229, 0.3)',
            }
          }}
        >
          Generate Questions
        </Button>
      </Box>
    </Box>
  );
};

export default DocumentSelector;
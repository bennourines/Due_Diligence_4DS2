// Component to manage uploaded document selection and generate questions
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

const DocumentSelector = ({ message, onGenerateQuestions, isLoading }: DocumentSelectorProps) => {
  const [selectedFile, setSelectedFile] = useState<string>('');
  const [uploadedFiles, setUploadedFiles] = useState<string[]>([]);

  useEffect(() => {
    // Extract file names from the message content
    const content = message.content;
    if (content.includes('have been successfully uploaded')) {
      // Multiple files case
      const fileListMatch = content.match(/files: (.*?) have been/);
      if (fileListMatch && fileListMatch[1]) {
        const fileNames = fileListMatch[1].split(', ').map((name: string) => name.trim());
        setUploadedFiles(fileNames);
        if (fileNames.length > 0) {
          setSelectedFile(fileNames[0]);
        }
      }
    } else if (content.includes('has been successfully uploaded')) {
      // Single file case
      const fileMatch = content.match(/File (.*?) has been/);
      if (fileMatch && fileMatch[1]) {
        const fileName = fileMatch[1].trim();
        setUploadedFiles([fileName]);
        setSelectedFile(fileName);
      }
    }
  }, [message]);

  // Fix the event type for Select's onChange handler
  const handleFileChange = (event: SelectChangeEvent<string>) => {
    setSelectedFile(event.target.value as string);
  };

  const handleGenerateClick = () => {
    const fileId = selectedFile.split('.')[0]; // Remove file extension to get the ID
    onGenerateQuestions(fileId);
  };

  return (
    <Box sx={{ mt: 2 }}>
      {uploadedFiles.length > 1 ? (
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
          <Typography
            variant="caption"
            sx={{
              color: 'var(--text-accent)',
              fontFamily: 'var(--font-primary)',
              letterSpacing: '0.5px'
            }}
          >
            Select a document to generate questions:
          </Typography>
          
          <FormControl 
            variant="outlined" 
            size="small"
            sx={{
              minWidth: 200,
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
              {uploadedFiles.map((file, index) => (
                <MenuItem key={index} value={file}>{file}</MenuItem>
              ))}
            </Select>
          </FormControl>
          
          <Button
            variant="outlined"
            size="small"
            onClick={handleGenerateClick}
            disabled={isLoading || !selectedFile}
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
      ) : (
        <Box sx={{ display: 'flex', justifyContent: 'flex-end' }}>
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
      )}
    </Box>
  );
};

export default DocumentSelector;
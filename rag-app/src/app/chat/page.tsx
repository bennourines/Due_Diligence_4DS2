'use client';

import { useState, useRef, useEffect } from 'react';
import { Box, TextField, IconButton, Paper, Typography, CircularProgress, Snackbar, Alert, Button, Switch, FormControlLabel, Slider } from '@mui/material';
import ChatHistory from '../components/ChatHistory';
import SendIcon from '@mui/icons-material/Send';
import AttachFileIcon from '@mui/icons-material/AttachFile';
import LogoutIcon from '@mui/icons-material/Logout';
import { useRouter } from 'next/navigation';
import ChevronLeftIcon from '@mui/icons-material/ChevronLeft';
import ChevronRightIcon from '@mui/icons-material/ChevronRight';
// Import the AdvancedAnimatedText component instead of TextFormatter
import CombinedAnimatedTextFormatter from '../components/CombinedAnimatedTextFormatter';
import DocumentSelector from '../components/DocumentSelector';

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

export default function Home() {
  const router = useRouter();
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [activeChat, setActiveChat] = useState<string>('');
  const [includeWebSearch, setIncludeWebSearch] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [refreshChats, setRefreshChats] = useState(0); // Counter to trigger chat history refresh
  const messagesEndRef = useRef<null | HTMLDivElement>(null);
  // Add new state variables for animation control
  const [typingSpeed, setTypingSpeed] = useState(25); // Default typing speed in ms
  const [skipAnimations, setSkipAnimations] = useState(false); // Flag to skip animations
  const [activeDocumentId, setActiveDocumentId] = useState<string | null>(null);
  const [showChatHistory, setShowChatHistory] = useState(false);

  useEffect(() => {
    const user = localStorage.getItem('user');
    if (user) {
      // Load user's chat history on initial load
      const userData = JSON.parse(user);
      fetchChats(userData._id);
    }
  }, []);

  useEffect(() => {
    // When active chat changes, load its messages
    if (activeChat) {
      loadMessages(activeChat);
    }
  }, [activeChat]);

  const fetchChats = async (userId: string) => {
    try {
      setIsLoading(true);
      const response = await fetch(`/api/chat?userId=${userId}`);
      const data = await response.json();
      
      // If user has existing chats, select the first one
      if (data.chats && data.chats.length > 0) {
        setActiveChat(data.chats[0]._id);
      } else {
        // If no chats exist, create a new one
        await createNewChat(userId);
      }
    } catch (error) {
      console.error('Error fetching chats:', error);
      setError('Failed to load chat history. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Function to handle animation completion
  const handleAnimationComplete = (index: number) => {
    if (index === messages.length - 1) {
      scrollToBottom();
    }
  };

  const createNewChat = async (userId: string) => {
    try {
      setIsLoading(true);
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          userId,
          action: 'create_chat',
          title: `Chat ${new Date().toLocaleString()}`
        })
      });
      
      if (!response.ok) {
        throw new Error(`Failed to create chat: ${response.status}`);
      }
      
      const data = await response.json();
      
      if (!data.chatId) {
        throw new Error('No chat ID returned from server');
      }
      
      setActiveChat(data.chatId);
      setMessages([]); // Clear messages when creating a new chat
      setRefreshChats(prev => prev + 1); // Trigger chat history refresh
      
      return data.chatId;
    } catch (error) {
      console.error('Error creating chat:', error);
      setError('Failed to create a new chat. Please try again.');
      return null;
    } finally {
      setIsLoading(false);
    }
  };

  const loadMessages = async (chatId: string) => {
    try {
      setIsLoading(true);
      const response = await fetch(`/api/chat?chatId=${chatId}`);
      
      if (!response.ok) {
        throw new Error(`Failed to load messages: ${response.status}`);
      }
      
      const data = await response.json();
      setMessages(data.messages || []);
      // Reset animation skip flag when loading new messages
      setSkipAnimations(false);
    } catch (error) {
      console.error('Error loading messages:', error);
      setError('Failed to load messages. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;
  
    const user = localStorage.getItem('user');
    if (!user) return;
  
    const userData = JSON.parse(user);
    const userMessage = { 
      role: 'user', 
      content: input, 
      isNew: true
    } as Message;
    
    // Check if we have an active chat, if not create one
    let currentChatId = activeChat;
    if (!currentChatId) {
      currentChatId = await createNewChat(userData._id);
      if (!currentChatId) {
        setError('Unable to create a chat. Please try again.');
        return;
      }
    }
    
    // Optimistically update UI
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);
    setSkipAnimations(false);
  
    try {
      // Save user message to MongoDB
      const msgResponse = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          chatId: currentChatId,
          userId: userData._id,
          message: userMessage
        })
      });
      
      if (!msgResponse.ok) {
        throw new Error(`Failed to save message: ${msgResponse.status}`);
      }
  
      // Get web search results if enabled - using the helper function
      const webSearchResults = await performWebSearch(input);
  
      // Get AI response
      const aiResponse = await fetch('http://localhost:8001/qa', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          id: activeDocumentId || 'all',
          question: input,
        })
      });
  
      if (!aiResponse.ok) {
        throw new Error(`Failed to get AI response: ${aiResponse.status}`);
      }
  
      const data = await aiResponse.json();
      let content = data.answer;
      
      // Append web search results if available
      if (webSearchResults && webSearchResults.status === 'ok' && webSearchResults.results.length > 0) {
        content += '\n\nRelevant web search results:\n';
        webSearchResults.results.forEach((result: any, index: number) => {
          content += `\n${index + 1}. ${result.title}\n   ${result.snippet}\n   Link: ${result.link}\n`;
        });
      }
  
      const assistantMessage = {
        role: 'assistant',
        content: content,
        sources: data.sources,
        isNew: true
      } as Message;
  
      // Save assistant message to MongoDB
      await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          chatId: currentChatId,
          userId: userData._id,
          message: assistantMessage
        })
      });
  
      setMessages(prev => [...prev, assistantMessage]);
      setRefreshChats(prev => prev + 1);
    } catch (error) {
      // Error handling...
    } finally {
      setIsLoading(false);
    }
  };

  const performWebSearch = async (searchQuery: string) => {
    if (!includeWebSearch) return null;
    
    try {
      const searchResponse = await fetch('http://localhost:8001/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          keyword: searchQuery,
          num_results: 5
        })
      });
      
      if (!searchResponse.ok) {
        console.error('Search request failed:', searchResponse.status);
        return null;
      }
      
      return await searchResponse.json();
    } catch (error) {
      console.error('Error during web search:', error);
      return null;
    }
  };

const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
  const files = e.target.files;
  if (!files || files.length === 0) return;

  const user = localStorage.getItem('user');
  if (!user) return;

  const userData = JSON.parse(user);
  
  // Check if we have an active chat, if not create one
  let currentChatId = activeChat;
  if (!currentChatId) {
    currentChatId = await createNewChat(userData._id);
    if (!currentChatId) {
      setError('Unable to create a chat. Please try again.');
      return;
    }
  }

  try {
    setIsLoading(true);
    
    // Create a message about the files being uploaded
    const fileNames = Array.from(files).map(file => file.name).join(", ");
    const initialMessage = {
      role: 'assistant',
      content: `Uploading files: ${fileNames}...`,
      isNew: true
    } as Message;
    
    setMessages(prev => [...prev, initialMessage]);

    const formData = new FormData();
    // Append all files to FormData with the same field name
    Array.from(files).forEach(file => {
      formData.append('files', file);
    });

    const response = await fetch('http://localhost:8001/ingest', {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`Failed to upload files: ${response.status}`);
    }

    const data = await response.json();
    
    // Create a detailed system message about the file upload results
    let resultMessage: Message;
    
    if (data.status === 'ok') {
      resultMessage = {
        role: 'assistant',
        content: `${files.length} file${files.length > 1 ? 's have' : ' has'} been successfully uploaded and processed. A total of ${data.chunks_added} chunks were added to the vector store. How can I help you?`,
      } as Message;
    } else if (data.status === 'partial') {
      const successFiles = data.processed_files.map((f: any) => f.filename).join(", ");
      const errorFiles = data.errors.join("\n");
      resultMessage = {
        role: 'assistant',
        content: `Some files were processed successfully (${successFiles}), but others encountered errors:\n${errorFiles}\n\nHow can I help you with the successfully processed documents?`,
      } as Message;
    } else {
      resultMessage = {
        role: 'assistant',
        content: `Error processing files: ${data.message || data.errors.join("\n")}`,
      } as Message;
    }

    // Save this message to the active chat
    await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        chatId: currentChatId,
        userId: userData._id,
        message: resultMessage
      })
    });
    
    // Update the messages state by replacing the initial "uploading" message
    setMessages(prev => [...prev.slice(0, -1), resultMessage]);
    
    // Refresh the chat list to update last messages
    setRefreshChats(prev => prev + 1);
  } catch (error) {
    console.error('Error uploading files:', error);
    const errorMessage = {
      role: 'assistant',
      content: `Sorry, there was an error uploading the files: ${error}`,
    } as Message;
    
    // Replace the "uploading" message with the error message
    setMessages(prev => [...prev.slice(0, -1), errorMessage]);
    setError('Error uploading files. Please try again.');
  } finally {
    setIsLoading(false);
    
    // Reset the file input
    if (e.target) {
      e.target.value = '';
    }
  }
};

  const handleCreateNewChat = () => {
    const user = localStorage.getItem('user');
    if (user) {
      const userData = JSON.parse(user);
      createNewChat(userData._id);
    }
  };

  const handleDeleteChat = (chatId: string) => {
    // If the deleted chat was the active one, clear messages and set active chat to empty
    if (activeChat === chatId) {
      setActiveChat('');
      setMessages([]);
      
      // If there are other chats, select the first one
      const user = localStorage.getItem('user');
      if (user) {
        const userData = JSON.parse(user);
        fetchChats(userData._id);
      }
    }
  };

  const handleCloseError = () => {
    setError(null);
  };

  const handleLogout = () => {
    localStorage.removeItem('user');
    localStorage.clear()
    router.push('/home');
  };


  const extractQuestions = (content: string): string[] => {
    const questions: string[] = [];
    
    // Find the questions section in the message
    const lines = content.split('\n');
    let inQuestionsSection = false;
    
    for (const line of lines) {
      // Check if this is a numbered question line (e.g., "1. What is...")
      if (/^\d+\.\s/.test(line.trim())) {
        // Extract just the question text without the number
        const question = line.trim().replace(/^\d+\.\s/, '');
        questions.push(question);
        inQuestionsSection = true;
      } 
      // If we were in the questions section and hit a blank line, we're done
      else if (inQuestionsSection && line.trim() === '') {
        inQuestionsSection = false;
      }
    }
    
    return questions;
  };
  
  // Handler to generate questions for a document
// Updated handleGenerateQuestions function for page.tsx
const handleGenerateQuestions = async (fileId: string) => {
  try {
    // Store the active document ID for future questions
    setActiveDocumentId(fileId);
    
    setIsLoading(true);
    
    // Call the backend API
    const response = await fetch('http://localhost:8001/generate-questions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        file_id: fileId
      })
    });
    
    if (!response.ok) {
      throw new Error(`Failed to generate questions: ${response.status}`);
    }
    
    const data = await response.json();
    
    if (data.status === 'error') {
      throw new Error(data.message);
    }
    
    // Format the questions
    let questionsText = "Here are some questions I've generated about this document:\n\n";
    data.questions.forEach((q: { question: string }, index: number) => {
      questionsText += `${index + 1}. ${q.question}\n`;
    });
    questionsText += "\nYou can click on any question to get an answer.";
    
    // Create the assistant message
    const assistantMessage = {
      role: 'assistant',
      content: questionsText,
      isNew: true
    } as Message;
    
    // Add to messages
    setMessages(prev => [...prev, assistantMessage]);
    
    // Save to the chat history
    const user = localStorage.getItem('user');
    if (user) {
      const userData = JSON.parse(user);
      await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          chatId: activeChat,
          userId: userData._id,
          message: assistantMessage
        })
      });
    }
  } catch (error) {
    console.error('Error generating questions:', error);
    setError('Failed to generate questions. Please try again.');
  } finally {
    setIsLoading(false);
  }
};
  
  // Handler for when a user selects a question
  const handleSelectQuestion = async (question: string) => {
    try {
      // Create a user message with the selected question
      const userMessage = { 
        role: 'user', 
        content: question, 
        isNew: true 
      } as Message;
      
      // Check if we have an active chat
      if (!activeChat) {
        setError('No active chat. Please create a new chat first.');
        return;
      }
      
      // Add user message to UI
      setMessages(prev => [...prev, userMessage]);
      setIsLoading(true);
      
      // Save user message to chat history
      const user = localStorage.getItem('user');
      if (!user) return;
      
      const userData = JSON.parse(user);
      
      await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          chatId: activeChat,
          userId: userData._id,
          message: userMessage
        })
      });
      
      // Get web search results if enabled
      const webSearchResults = await performWebSearch(question);
      
      // Get AI response - use the stored document ID if available
      const aiResponse = await fetch('http://localhost:8001/qa', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          id: activeDocumentId || 'all', // Use the tracked document ID if available
          question: question,
        })
      });
      
      if (!aiResponse.ok) {
        throw new Error(`Failed to get AI response: ${aiResponse.status}`);
      }
      
      const data = await aiResponse.json();
      let content = data.answer;
      
      // Append web search results if available
      if (webSearchResults && webSearchResults.status === 'ok' && webSearchResults.results.length > 0) {
        content += '\n\nRelevant web search results:\n';
        webSearchResults.results.forEach((result: any, index: number) => {
          content += `\n${index + 1}. ${result.title}\n   ${result.snippet}\n   Link: ${result.link}\n`;
        });
      }
      
      // Create assistant message
      const assistantMessage = {
        role: 'assistant',
        content: content,
        sources: data.sources,
        isNew: true
      } as Message;
      
      // Save assistant message
      await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          chatId: activeChat,
          userId: userData._id,
          message: assistantMessage
        })
      });
      
      // Update UI
      setMessages(prev => [...prev, assistantMessage]);
      
    } catch (error) {
      console.error('Error handling question selection:', error);
      setError('Failed to answer question. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

return (
  <Box sx={{ display: 'flex', height: '100vh', position: 'relative' }}>
    {/* Neural network background */}
    <Box className="grid-background" />
    
    {/* Ambient glow effects */}
    <Box 
      className="accent-dot" 
      sx={{ 
        top: '15%', 
        left: '10%', 
        background: 'var(--gradient-glow)' 
      }} 
    />
    <Box 
      className="accent-dot" 
      sx={{ 
        bottom: '20%', 
        right: '5%', 
        background: 'var(--gradient-glow)',
        width: '200px',
        height: '200px'
      }} 
    />
    
    {/* Logout button with Agentic AI styling */}
    <Box sx={{ 
      position: 'absolute',
      top: 16,
      right: 16,
      zIndex: 1000
    }}>
      <IconButton
        onClick={handleLogout}
        sx={{
          color: 'var(--text-primary)',
          bgcolor: 'rgba(15, 23, 42, 0.7)',
          border: 'var(--border-light)',
          borderRadius: 'var(--border-radius)',
          transition: 'var(--transition-smooth)',
          backdropFilter: 'blur(4px)',
          '&:hover': {
            boxShadow: 'var(--shadow-glow)',
            bgcolor: 'rgba(79, 70, 229, 0.1)',
          },
          '&:hover::before': {
            left: '100%'
          },
          '&::before': {
            content: '""',
            position: 'absolute',
            top: 0,
            left: '-100%',
            width: '100%',
            height: '100%',
            background: 'linear-gradient(90deg, transparent, rgba(79, 70, 229, 0.2), transparent)',
            transition: 'var(--transition-smooth)'
          }
        }}
      >
        <LogoutIcon />
      </IconButton>
    </Box>
    
{/* Chat History Component with Toggle */}
<Box
  sx={{
    display: 'flex',
    position: 'relative',
    zIndex: 10,
    transition: 'var(--transition-smooth)',
    width: showChatHistory ? 'auto' : '0px',
    overflow: 'hidden',
  }}
>
  {showChatHistory && (
    <ChatHistory 
      activeChat={activeChat} 
      onSelectChat={setActiveChat} 
      onCreateNewChat={handleCreateNewChat}
      onDeleteChat={handleDeleteChat}
      refreshTrigger={refreshChats}
      setShowChatHistory={setShowChatHistory} // Pass the function
    />
  )}
</Box>

{/* Button to show chat history when it's hidden */}
{!showChatHistory && (
  <IconButton
    onClick={() => setShowChatHistory(true)}
    sx={{
      position: 'absolute',
      left: 16,
      top: '50%',
      transform: 'translateY(-50%)',
      zIndex: 1000,
      color: 'var(--text-primary)',
      bgcolor: 'rgba(15, 23, 42, 0.7)',
      border: 'var(--border-light)',
      borderRadius: 'var(--border-radius)',
      height: '40px',
      width: '40px',
      backdropFilter: 'blur(4px)',
      boxShadow: 'var(--shadow-glow)',
      '&:hover': {
        boxShadow: '0 0 15px rgba(79, 70, 229, 0.4)',
        bgcolor: 'rgba(79, 70, 229, 0.1)',
      }
    }}
  >
    <ChevronRightIcon />
  </IconButton>
)}

    
    <Box 
      sx={{ 
        flex: 1, 
        p: 2, 
        display: 'flex', 
        flexDirection: 'column',
        transition: 'var(--transition-smooth)',
      }}
    >
      {/* Agent Status Indicator */}
      <Box 
        className="agent-status"
        sx={{ 
          alignSelf: 'flex-end', 
          mb: 2,
          fontFamily: 'var(--font-primary)'
        }}
      >
        AI Agent Active
      </Box>
      
      {/* Main Chat Area */}
      <Paper 
        elevation={0}
        sx={{
          flex: 1,
          mb: 2,
          p: 2,
          overflow: 'auto',
          bgcolor: 'rgba(30, 41, 59, 0.7)',
          border: 'var(--border-light)',
          borderRadius: 'var(--border-radius)',
          boxShadow: 'var(--shadow-glow)',
          backdropFilter: 'blur(10px)',
          transition: 'var(--transition-smooth)',
          display: 'flex',
          flexDirection: 'column',
          backgroundSize: 'cover',
          backgroundPosition: 'center',
          backgroundRepeat: 'no-repeat',
          position: 'relative',
          '&:hover': {
            boxShadow: '0 0 25px rgba(79, 70, 229, 0.3)'
          },
          '&::before': {
            content: '""',
            position: 'absolute',
            top: 0,
            right: 0,
            bottom: 0,
            left: 0,
            background: 'radial-gradient(circle at 50% 50%, rgba(15, 23, 42, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%)',
            pointerEvents: 'none',
            zIndex: 0,
            borderRadius: 'var(--border-radius)'
          },
          '& > *': {
            position: 'relative',
            zIndex: 1
          }
        }}
      >
        {messages.length === 0 && !isLoading ? (
          <Box sx={{ 
            display: 'flex', 
            flexDirection: 'column',
            justifyContent: 'center', 
            alignItems: 'center',
            height: '100%',
            color: 'var(--text-primary)',
            opacity: 0.7
          }}>
            <Typography 
              variant="h6" 
              sx={{ 
                letterSpacing: '2px',
                position: 'relative',
                textTransform: 'uppercase',
                fontFamily: 'var(--font-primary)',
                textShadow: '0 0 10px var(--primary-glow)',
                color: 'var(--text-accent)'
              }}
            >
              Activate AI Agent
            </Typography>
            <Typography 
              variant="body2" 
              sx={{ 
                mt: 1,
                letterSpacing: '1px',
                fontFamily: 'var(--font-primary)'
              }}
            >
              Upload a document or ask a question
            </Typography>
          </Box>
        ) : (
          messages.map((message, index) => (
            <Box
              key={index}
              onClick={() => setSkipAnimations(true)} // Skip animations when clicking
              sx={{
                mb: 2,
                alignSelf: message.role === 'user' ? 'flex-end' : 'flex-start',
                maxWidth: '70%',
              }}
            >
              <Paper
                elevation={0}
                sx={{
                  p: 2,
                  bgcolor: message.role === 'user' 
                    ? 'rgba(79, 70, 229, 0.1)' 
                    : 'rgba(15, 23, 42, 0.7)',
                  color: message.role === 'user' 
                    ? 'var(--text-primary)' 
                    : 'var(--text-primary)',
                  border: 'var(--border-light)',
                  borderRadius: 'var(--border-radius)',
                  backdropFilter: 'blur(10px)',
                  boxShadow: message.role === 'user'
                    ? '0 0 10px rgba(79, 70, 229, 0.1)'
                    : 'var(--shadow-glow)',
                  transition: 'var(--transition-smooth)',
                  borderLeft: message.role === 'assistant' 
                    ? '3px solid var(--primary-color)' 
                    : 'none',
                  borderRight: message.role === 'user' 
                    ? '3px solid var(--primary-color)' 
                    : 'none',
                  '&:hover': {
                    boxShadow: '0 0 15px rgba(79, 70, 229, 0.3)'
                  }
                }}
              >
                {/* AI response indicator */}
                {message.role === 'assistant' && (
                  <Box 
                    sx={{ 
                      display: 'flex', 
                      alignItems: 'center', 
                      mb: 1,
                      opacity: 0.8,
                      color: 'var(--primary-light)'
                    }}
                  >
                    <Box 
                      component="span" 
                      sx={{ 
                        width: 8, 
                        height: 8, 
                        borderRadius: '50%', 
                        bgcolor: 'var(--primary-light)',
                        mr: 1,
                        boxShadow: '0 0 8px var(--primary-glow)'
                      }} 
                    />
                    <Typography 
                      variant="caption"
                      sx={{ 
                        fontFamily: 'var(--font-primary)',
                        letterSpacing: '1px',
                        textTransform: 'uppercase',
                        fontSize: '0.7rem'
                      }}
                    >
                      Agent Response
                    </Typography>
                  </Box>
                )}
                
                <CombinedAnimatedTextFormatter 
                  text={message.content}
                  className="whitespace-pre-line"
                  role={message.role}
                  isNewMessage={message.isNew}
                  onComplete={() => handleAnimationComplete(index)}
                  sx={{ 
                    mb: message.sources && message.sources.length > 0 ? 2 : 0,
                    color: 'var(--text-primary)',
                    fontFamily: 'var(--font-primary)',
                    letterSpacing: '0.5px'
                  }}
                />
                
                {/* Generate Questions button with Agentic AI styling */}
                {message.role === 'assistant' && 
                  message.content.includes('has been successfully uploaded and processed') && (
                  message.content.includes('have been successfully uploaded and processed')) && (
                  <DocumentSelector 
                    message={message}
                    onGenerateQuestions={handleGenerateQuestions}
                    isLoading={isLoading}
                  />
                )}

                {/* Clickable questions with Agentic AI styling */}
                {message.role === 'assistant' && 
                  message.content.includes("Here are some questions I've generated about this document") && (
                  <Box sx={{ mt: 2 }}>
                    <Typography
                      variant="subtitle2"
                      sx={{
                        color: 'var(--text-accent)',
                        opacity: 0.9,
                        mb: 1,
                        fontStyle: 'italic',
                        fontFamily: 'var(--font-primary)',
                        letterSpacing: '0.5px'
                      }}
                    >
                      Click a question to initiate analysis:
                    </Typography>
                    {extractQuestions(message.content).map((question: string, qIndex: number) => (
                      <Button
                        key={qIndex}
                        variant="outlined"
                        size="small"
                        onClick={() => handleSelectQuestion(question)}
                        disabled={isLoading}
                        sx={{
                          color: 'var(--text-primary)',
                          borderColor: 'rgba(79, 70, 229, 0.2)',
                          backgroundColor: 'rgba(15, 23, 42, 0.3)',
                          textAlign: 'left',
                          justifyContent: 'flex-start',
                          textTransform: 'none',
                          display: 'block',
                          width: '100%',
                          my: 1,
                          py: 1.5,
                          px: 2,
                          borderRadius: 'var(--border-radius)',
                          fontFamily: 'var(--font-primary)',
                          letterSpacing: '0.5px',
                          transition: 'var(--transition-smooth)',
                          backdropFilter: 'blur(4px)',
                          position: 'relative',
                          overflow: 'hidden',
                          '&:hover': {
                            backgroundColor: 'rgba(79, 70, 229, 0.1)',
                            borderColor: 'var(--primary-color)',
                            boxShadow: 'var(--shadow-glow)'
                          },
                          '&.Mui-disabled': {
                            opacity: 0.5,
                            color: 'rgba(79, 70, 229, 0.5)',
                          },
                          '&::before': {
                            content: '""',
                            position: 'absolute',
                            top: 0,
                            left: '-100%',
                            width: '100%',
                            height: '100%',
                            background: 'linear-gradient(90deg, transparent, rgba(79, 70, 229, 0.1), transparent)',
                            transition: 'var(--transition-smooth)'
                          },
                          '&:hover::before': {
                            left: '100%'
                          }
                        }}
                      >
                        <Box sx={{ display: 'flex', alignItems: 'center' }}>
                          {/* Question number with AI styling */}
                          <Box sx={{ 
                            minWidth: 28, 
                            height: 28, 
                            borderRadius: '50%', 
                            bgcolor: 'rgba(79, 70, 229, 0.1)', 
                            border: '1px solid rgba(79, 70, 229, 0.3)',
                            display: 'flex', 
                            alignItems: 'center', 
                            justifyContent: 'center',
                            mr: 1.5,
                            fontSize: '0.8rem',
                            fontWeight: 'bold',
                            color: 'var(--primary-light)',
                            fontFamily: 'var(--font-primary)',
                            boxShadow: '0 0 5px rgba(79, 70, 229, 0.2)'
                          }}>
                            {qIndex + 1}
                          </Box>
                          {/* Question text */}
                          <Box sx={{ flexGrow: 1 }}>
                            {question}
                          </Box>
                        </Box>
                      </Button>
                    ))}
                  </Box>
                )}
              </Paper>
            </Box>
          ))
        )}
        <div ref={messagesEndRef} />
        {isLoading && (
          <Box sx={{ display: 'flex', justifyContent: 'center', my: 2 }}>
            {/* Custom agentic AI loader */}
            <Box className="loader" />
          </Box>
        )}
      </Paper>
      
      {/* Message input area with Agentic AI styling */}
      <Paper 
      component="form" 
      onSubmit={handleSubmit}
      elevation={0}
      sx={{
        p: 2,
        display: 'flex',
        alignItems: 'center',
        gap: 1,
        bgcolor: 'rgba(30, 41, 59, 0.7)',
        border: 'var(--border-light)',
        borderRadius: 'var(--border-radius)',
        backdropFilter: 'blur(10px)',
        boxShadow: 'var(--shadow-glow)',
        transition: 'var(--transition-smooth)',
        '&:hover': {
          boxShadow: '0 0 25px rgba(79, 70, 229, 0.3)'
        }
      }}
    >
      <input
        type="file"
        id="file-upload"
        onChange={handleFileUpload}
        multiple  
        style={{ display: 'none' }}
      />
      <IconButton
        component="label"
        htmlFor="file-upload"
        sx={{
          color: 'var(--primary-light)',
          transition: 'var(--transition-smooth)',
          position: 'relative',
          overflow: 'hidden',
          '&:hover': {
            color: 'var(--text-primary)',
            backgroundColor: 'rgba(79, 70, 229, 0.1)',
            boxShadow: '0 0 10px var(--primary-glow)'
          },
          '&::before': {
            content: '""',
            position: 'absolute',
            top: 0,
            left: '-100%',
            width: '100%',
            height: '100%',
            background: 'linear-gradient(90deg, transparent, rgba(79, 70, 229, 0.2), transparent)',
            transition: 'var(--transition-smooth)'
          },
          '&:hover::before': {
            left: '100%'
          },
          '&.Mui-disabled': {
            color: 'rgba(79, 70, 229, 0.3)'
          }
        }}
        disabled={isLoading}
        title="Upload multiple files" 
      >
        <AttachFileIcon />
      </IconButton>
        <Box sx={{ display: 'flex', alignItems: 'center', flex: 1, gap: 1 }}>
          <FormControlLabel
            control={
              <Switch
                checked={includeWebSearch}
                onChange={(e) => setIncludeWebSearch(e.target.checked)}
                sx={{
                  '& .MuiSwitch-switchBase': {
                    color: 'rgba(79, 70, 229, 0.3)',
                    '&:hover': {
                      backgroundColor: 'rgba(79, 70, 229, 0.08)',
                    },
                  },
                  '& .MuiSwitch-switchBase.Mui-checked': {
                    color: 'var(--primary-color)',
                    '&:hover': {
                      backgroundColor: 'rgba(79, 70, 229, 0.08)',
                    },
                  },
                  '& .MuiSwitch-switchBase.Mui-checked + .MuiSwitch-track': {
                    backgroundColor: 'var(--primary-color)',
                    opacity: 0.5
                  },
                  '& .MuiSwitch-track': {
                    backgroundColor: 'rgba(79, 70, 229, 0.2)'
                  }
                }}
              />
            }
            label={
              <Typography 
                sx={{ 
                  color: 'var(--text-primary)', 
                  fontSize: '0.875rem',
                  fontFamily: 'var(--font-primary)',
                  letterSpacing: '0.5px'
                }}
              >
                Include Web Search
              </Typography>
            }
          />
          <TextField
            fullWidth
            variant="outlined"
            placeholder="Type your query for the AI agent..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            disabled={isLoading}
            sx={{
              flex: 1,
              '& .MuiOutlinedInput-root': {
                color: 'var(--text-primary)',
                fontFamily: 'var(--font-primary)',
                letterSpacing: '0.5px',
                transition: 'var(--transition-smooth)',
                borderRadius: 'var(--border-radius)',
                backgroundColor: 'rgba(15, 23, 42, 0.3)',
                backdropFilter: 'blur(4px)',
                '& fieldset': {
                  borderColor: 'rgba(79, 70, 229, 0.3)',
                  borderRadius: 'var(--border-radius)',
                  transition: 'var(--transition-smooth)',
                },
                '&:hover fieldset': {
                  borderColor: 'var(--primary-color)',
                },
                '&.Mui-focused fieldset': {
                  borderColor: 'var(--primary-color)',
                  boxShadow: 'var(--shadow-glow)',
                },
                '&.Mui-focused': {
                  boxShadow: 'var(--shadow-glow)',
                }
              },
              '& .MuiInputBase-input::placeholder': {
                color: 'var(--text-secondary)',
                opacity: 0.7,
                fontFamily: 'var(--font-primary)',
              },
            }}
            InputProps={{
              sx: {
                "&.Mui-focused": {
                  boxShadow: 'var(--shadow-glow)',
                }
              }
            }}
          />
        </Box>
        <IconButton 
          type="submit" 
          sx={{
            color: 'var(--primary-light)',
            transition: 'var(--transition-smooth)',
            position: 'relative',
            overflow: 'hidden',
            '&:hover': {
              color: 'var(--text-primary)',
              backgroundColor: 'rgba(79, 70, 229, 0.1)',
              boxShadow: '0 0 10px var(--primary-glow)'
            },
            '&::before': {
              content: '""',
              position: 'absolute',
              top: 0,
              left: '-100%',
              width: '100%',
              height: '100%',
              background: 'linear-gradient(90deg, transparent, rgba(79, 70, 229, 0.2), transparent)',
              transition: 'var(--transition-smooth)'
            },
            '&:hover::before': {
              left: '100%'
            },
            '&.Mui-disabled': {
              color: 'rgba(79, 70, 229, 0.3)',
            }
          }} 
          disabled={!input.trim() || isLoading}
        >
          <SendIcon />
        </IconButton>
      </Paper>
    </Box>
    
    {/* Snackbar with Agentic AI styling */}
    <Snackbar 
      open={!!error} 
      autoHideDuration={6000} 
      onClose={handleCloseError}
      anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
    >
      <Alert 
        onClose={handleCloseError} 
        severity="error" 
        sx={{ 
          width: '100%',
          bgcolor: 'rgba(15, 23, 42, 0.8)',
          color: 'var(--text-primary)',
          borderLeft: '4px solid #f44336',
          backdropFilter: 'blur(10px)',
          fontFamily: 'var(--font-primary)',
          letterSpacing: '0.5px',
          boxShadow: '0 0 20px rgba(244, 67, 54, 0.3)'
        }}
      >
        {error}
      </Alert>
    </Snackbar>
  </Box>
)};
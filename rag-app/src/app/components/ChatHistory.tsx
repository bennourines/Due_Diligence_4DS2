'use client';

import { useEffect, useState } from 'react';
import { Paper, Typography, CircularProgress, Box, Button, IconButton } from '@mui/material';
import AddIcon from '@mui/icons-material/Add';
import DeleteIcon from '@mui/icons-material/Delete';
import ChevronLeftIcon from '@mui/icons-material/ChevronLeft';
import ChevronRightIcon from '@mui/icons-material/ChevronRight';
import { useRouter } from 'next/navigation';
import HomeIcon from '@mui/icons-material/Home';

interface Chat {
  _id: string;
  title: string;
  createdAt: string;
  lastMessage?: string;
}

interface ChatHistoryProps {
  activeChat?: string;
  onSelectChat?: (chatId: string) => void;
  onCreateNewChat?: () => void;
  refreshTrigger?: number; // New prop to trigger refresh
  onDeleteChat?: (chatId: string) => void;
  setShowChatHistory: (show: boolean) => void; // Prop to control visibility from parent
}

export default function ChatHistory({ 
  activeChat, 
  onSelectChat, 
  onCreateNewChat, 
  onDeleteChat, 
  refreshTrigger = 0,
  setShowChatHistory // Use this prop instead of local state
}: ChatHistoryProps) {
  const [chats, setChats] = useState<Chat[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const router = useRouter();
  
  // Add this handler function
  const handleHomeClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    router.push('/home');
  };
  
  // Remove the local state for showChatHistory - we'll use the prop instead
  // const [showChatHistory, setShowChatHistory] = useState(true);

  // Function to hide chat history with stopPropagation
  const handleHideClick = (e: React.MouseEvent) => {
    e.stopPropagation(); // Prevent event bubbling
    setShowChatHistory(false); // This now updates the parent state
    console.log("Hide button clicked"); // For debugging
  };

  // Function to fetch chats
  const handleDeleteChat = async (chatId: string, e: React.MouseEvent) => {
    e.stopPropagation(); // Prevent triggering the chat selection
    
    try {
      const user = localStorage.getItem('user');
      if (!user) return;
      
      const userData = JSON.parse(user);
      const response = await fetch(`/api/chat?chatId=${chatId}&userId=${userData._id}`, {
        method: 'DELETE',
      });
      
      if (!response.ok) {
        throw new Error(`Failed to delete chat: ${response.status}`);
      }
      
      // Refresh the chat list after deletion
      fetchChats();
      
      // If the deleted chat was the active one, notify parent component
      if (activeChat === chatId && onDeleteChat) {
        onDeleteChat(chatId);
      }
    } catch (err) {
      console.error('Error deleting chat:', err);
      setError('Failed to delete chat. Please try again.');
    }
  };

  const fetchChats = async () => {
    try {
      setLoading(true);
      const user = localStorage.getItem('user');
      if (!user) {
        setError('No user found. Please log in.');
        setLoading(false);
        return;
      }

      const userData = JSON.parse(user);
      const response = await fetch(`/api/chat?userId=${userData._id}`);
      
      if (!response.ok) {
        throw new Error(`Failed to fetch chats: ${response.status}`);
      }
      
      const data = await response.json();
      if (data.chats) {
        // Sort chats by creation date (newest first)
        const sortedChats = data.chats.sort((a: Chat, b: Chat) => 
          new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime()
        );
        setChats(sortedChats);
      }
    } catch (err) {
      console.error('Error fetching chats:', err);
      setError('Failed to load chats. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchChats();
  }, [refreshTrigger]); // Re-fetch when refreshTrigger changes

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString(undefined, {
      month: 'short',
      day: 'numeric',
      year: 'numeric'
    });
  };

return (
  <Paper
    elevation={3}
    sx={{
      width: '250px',
      height: '100vh',
      bgcolor: 'var(--bg-deep)',
      fontFamily: 'var(--font-primary)',
      borderRight: '1px solid var(--primary-color)',
      overflow: 'auto',
      p: 2,
      display: 'flex',
      flexDirection: 'column',
    }}
  >
<Box 
  sx={{ 
    display: 'flex', 
    justifyContent: 'space-between', 
    alignItems: 'center',
    mb: 2,
    borderBottom: '1px solid var(--primary-color)',
    pb: 1,
    bgcolor: 'rgba(15, 23, 42, 0.8)', // Added darker background color
    borderRadius: 'var(--border-radius)',
    p: 1, // Added padding for better appearance
    boxShadow: '0 2px 4px rgba(0, 0, 0, 0.1)' // Added subtle shadow for depth
  }}>
  <Typography
    variant="h6"
    sx={{
      fontFamily: 'var(--font-primary)',
      color: 'var(--text-primary)',
      borderRadius: 'var(--border-radius)',
      pl: 1 // Added left padding for better text alignment
    }}
  >
    Chat History
  </Typography>
  
  {/* Actions container with both buttons */}
  <Box sx={{ display: 'flex', gap: 1 }}>
    {/* Toggle button to hide chat history - Use handleHideClick */}
    <IconButton
      onClick={handleHideClick}
      sx={{
        minWidth: 'auto',
        p: '4px',
        color: 'var(--primary-light)',
        '&:hover': {
          bgcolor: 'rgba(79, 70, 229, 0.1)',
        },
      }}
    >
      <ChevronLeftIcon fontSize="small" />
    </IconButton>
    
    {/* Create new chat button */}
    <IconButton
      onClick={onCreateNewChat}
      sx={{
        minWidth: 'auto',
        p: '4px',
        color: 'var(--primary-light)',
        '&:hover': {
          bgcolor: 'rgba(79, 70, 229, 0.1)',
        },
      }}
    >
      <AddIcon fontSize="small" />
    </IconButton>

    <IconButton
      onClick={handleHomeClick}
      sx={{
        minWidth: 'auto',
        p: '4px',
        color: 'var(--primary-light)',
        '&:hover': {
          bgcolor: 'rgba(79, 70, 229, 0.1)',
        },
      }}
    >
      <HomeIcon fontSize="small" />
    </IconButton>
  </Box>
</Box>
    
    {loading ? (
      <Box sx={{ display: 'flex', justifyContent: 'center', my: 4 }}>
        <CircularProgress sx={{ 
          fontFamily: 'var(--font-primary)',
          color: 'var(--primary-color)',
          borderRadius: 'var(--border-radius)',
        }} />
      </Box>
    ) : error ? (
      <Typography sx={{ 
        fontFamily: 'var(--font-primary)',
        color: 'var(--text-primary)',
        borderRadius: 'var(--border-radius)',
        opacity: 0.7, 
        textAlign: 'center', 
        mt: 2 
      }}>
        {error}
      </Typography>
    ) : chats.length === 0 ? (
      <Typography sx={{ 
        color: 'var(--text-secondary)', 
        opacity: 0.7, 
        textAlign: 'center', 
        mt: 2 
      }}>
        No chats yet. Start a new conversation!
      </Typography>
    ) : (
      chats.map((chat) => (
        <Paper
          variant="outlined"
  
          key={chat._id}
          elevation={1}
          onClick={() => onSelectChat?.(chat._id)}
          sx={{
            mb: 2,
            p: 2,
            bgcolor: activeChat === chat._id 
              ? 'rgba(255, 255, 255, 0.2)' 
              : 'rgba(4, 4, 4, 0.3)',
            color: 'var(--text-primary)',
            border: '1px solid var(--primary-color)',
            cursor: 'pointer',
            borderRadius: 'var(--border-radius)',
            transition: 'var(--transition-smooth)',
            '&:hover': {
              bgcolor: activeChat === chat._id 
                ? 'rgba(255, 255, 255, 0.3)' 
                : 'rgba(0, 0, 0, 0.1)',
              boxShadow: 'var(--shadow-glow)',
            },
            position: 'relative',
          }}
        >
          <Typography 
            variant="subtitle1" 
            sx={{ 
              fontWeight: 'bold',
              fontFamily: 'var(--font-primary)',
              letterSpacing: '0.5px',
            }}>
            {chat.title}
          </Typography>
          <Typography
            variant="caption"
            sx={{
              display: 'block',
              opacity: 0.8,
              mb: 0.5,
              fontFamily: 'var(--font-primary)',
            }}
          >
            {formatDate(chat.createdAt)}
          </Typography>
          {chat.lastMessage && (
            <Typography
              variant="body2"
              sx={{
                opacity: 0.8,
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap',
                paddingRight: '24px', // Make room for delete button
                fontFamily: 'var(--font-primary)',
              }}
            >
              {chat.lastMessage}
            </Typography>
          )}
          <IconButton
            size="small"
            onClick={(e) => handleDeleteChat(chat._id, e)}
            sx={{
              position: 'absolute',
              right: 8,
              bottom: 8,
              color: 'var(--primary-light)',
              opacity: 0.7,
              '&:hover': {
                opacity: 1,
                bgcolor: 'rgba(79, 70, 229, 0.1)',
              },
            }}
          >
            <DeleteIcon fontSize="small" />
          </IconButton>
        </Paper>
      ))
    )}
  </Paper>
)};
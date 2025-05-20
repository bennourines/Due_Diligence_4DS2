import { NextRequest, NextResponse } from 'next/server';
import { connectToDatabase } from '../auth';
import { ObjectId } from 'mongodb';
import { revalidatePath } from 'next/cache';

// Helper function to add last message to chat objects
async function addLastMessagesToChats(db: any, chats: any[]) {
  const enrichedChats = [];
  
  for (const chat of chats) {
    // Convert chat._id to string for comparison
    const chatId = chat._id.toString();
    
    // Find the most recent message for this chat
    const lastMessage = await db.collection('ChatMessage')
      .find({ chatId })
      .sort({ createdAt: -1 })
      .limit(1)
      .toArray();
    
    // Add the last message content if available
    if (lastMessage.length > 0) {
      enrichedChats.push({
        ...chat,
        lastMessage: lastMessage[0].content.substring(0, 50) + (lastMessage[0].content.length > 50 ? '...' : '')
      });
    } else {
      enrichedChats.push(chat);
    }
  }
  
  return enrichedChats;
}

export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams;
    const chatId = searchParams.get('chatId');
    const userId = searchParams.get('userId');

    if (!chatId && (!userId || typeof userId !== 'string')) {
      return NextResponse.json({ error: 'Missing chatId or userId parameter' }, { status: 400 });
    }

    const db = await connectToDatabase();

    if (chatId) {
      // Verify that the chat exists
      const chat = await db.collection('Chat').findOne({
        _id: new ObjectId(chatId)
      });

      if (!chat) {
        return NextResponse.json({ error: 'Chat not found' }, { status: 404 });
      }

      // Get messages for a specific chat
      const messages = await db.collection('ChatMessage')
        .find({ chatId })
        .sort({ createdAt: 1 }) // Sort by creation time ascending
        .toArray();
      return NextResponse.json({ messages });
    } else {
      // Get all chats for a user
      const chats = await db.collection('Chat')
        .find({ userId })
        .sort({ createdAt: -1 }) // Sort by creation time descending
        .toArray();
      
      // Add the last message to each chat
      const enrichedChats = await addLastMessagesToChats(db, chats);
      
      return NextResponse.json({ chats: enrichedChats });
    }
  } catch (error) {
    console.error('Error fetching chat data:', error);
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 });
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { chatId, userId, message, action } = body;

    if (!userId) {
      return NextResponse.json({ error: 'Missing userId' }, { status: 400 });
    }

    const db = await connectToDatabase();

    // Check if the user exists in the database
    const user = await db.collection('User').findOne({ _id: new ObjectId(userId) });
    if (!user && action !== 'create_chat') {
      return NextResponse.json({ error: 'User not found' }, { status: 404 });
    }

    if (action === 'create_chat') {
      if (!body.title) {
        return NextResponse.json({ error: 'Missing title' }, { status: 400 });
      }
      
      const result = await db.collection('Chat').insertOne({
        userId,
        title: body.title,
        createdAt: new Date()
      });
      
      if (!result.insertedId) {
        return NextResponse.json({ error: 'Failed to create chat' }, { status: 500 });
      }
      
      return NextResponse.json({ 
        chatId: result.insertedId.toString(),
        message: 'Chat created successfully' 
      });
    }

    // For message actions, validate chatId and message
    if (!chatId || !message) {
      return NextResponse.json({ error: 'Missing required fields' }, { status: 400 });
    }

    // Verify that the chat exists before adding messages to it
    let chat;
    try {
      chat = await db.collection('Chat').findOne({
        _id: new ObjectId(chatId)
      });
    } catch (e) {
      return NextResponse.json({ error: 'Invalid chat ID format' }, { status: 400 });
    }

    if (!chat) {
      return NextResponse.json({ error: 'Chat not found' }, { status: 404 });
    }

    // Insert the message
    const savedMessage = await db.collection('ChatMessage').insertOne({
      chatId,
      userId,
      role: message.role,
      content: message.content,
      sources: message.sources || [],
      createdAt: new Date()
    });

    return NextResponse.json({ 
      messageId: savedMessage.insertedId.toString(),
      message: 'Message saved successfully'
    });
  } catch (error) {
    console.error('Error processing chat action:', error);
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 });
  }
}

export async function DELETE(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams;
    const chatId = searchParams.get('chatId');
    const userId = searchParams.get('userId');

    if (!chatId || !userId) {
      return NextResponse.json({ error: 'Missing chatId or userId parameter' }, { status: 400 });
    }

    const db = await connectToDatabase();

    // Verify that the chat exists and belongs to the user
    const chat = await db.collection('Chat').findOne({
      _id: new ObjectId(chatId),
      userId: userId
    });

    if (!chat) {
      return NextResponse.json({ error: 'Chat not found or does not belong to user' }, { status: 404 });
    }

    // Delete all messages associated with the chat
    await db.collection('ChatMessage').deleteMany({ chatId });

    // Delete the chat itself
    const result = await db.collection('Chat').deleteOne({ _id: new ObjectId(chatId) });

    if (result.deletedCount === 0) {
      return NextResponse.json({ error: 'Failed to delete chat' }, { status: 500 });
    }

    return NextResponse.json({ message: 'Chat deleted successfully' });
  } catch (error) {
    console.error('Error deleting chat:', error);
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 });
  }
}
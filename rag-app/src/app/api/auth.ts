import { MongoClient } from 'mongodb';

const uri = 'mongodb+srv://mohamedaminebenammar:Kiki98765@cluster0.81oai.mongodb.net/';
const dbName = 'DuedelligenceNext';

let client: MongoClient | null = null;
let isConnecting = false;
let connectionPromise: Promise<MongoClient> | null = null;

async function getMongoClient(
  retryAttempt = 1,
  maxRetries = 3
): Promise<MongoClient> {
  // 1) If we've already connected, just return the existing client
  if (client) {
    try {
      // Modern way to check connection status in newer MongoDB drivers
      await client.db(dbName).command({ ping: 1 });
      return client;
    } catch (err) {
      // Connection dropped or invalid, will create a new one
      client = null;
    }
  }

  // 2) If a connection attempt is already in flight, wait for it
  if (isConnecting && connectionPromise) {
    return connectionPromise;
  }

  // 3) Otherwise, kick off a new connection
  isConnecting = true;
  connectionPromise = (async () => {
    try {
      const tempClient = new MongoClient(uri, {
        maxPoolSize: 10,
        minPoolSize: 5,
        connectTimeoutMS: 30_000,
        serverSelectionTimeoutMS: 30_000,
        socketTimeoutMS: 45_000,
        waitQueueTimeoutMS: 30_000,
      });

      await tempClient.connect();
      // optional ping to be extra-safe
      await tempClient.db(dbName).command({ ping: 1 });
      console.log("MongoDB connection established successfully");

      client = tempClient;
      return client;
    } catch (err) {
      console.error("MongoDB connection error:", err);
      if (retryAttempt < maxRetries) {
        // exponential back‑off
        const backoff = Math.min(1_000 * 2 ** (retryAttempt - 1), 10_000);
        console.log(`Retrying connection in ${backoff}ms (attempt ${retryAttempt} of ${maxRetries})`);
        await new Promise((r) => setTimeout(r, backoff));
        return getMongoClient(retryAttempt + 1, maxRetries);
      }
      throw err;
    } finally {
      isConnecting = false;
      connectionPromise = null;
    }
  })();

  return connectionPromise;
}

export async function connectToDatabase() {
  try {
    const mongoClient = await getMongoClient();
    return mongoClient.db(dbName);
  } catch (error) {
    console.error('MongoDB connection error:', error);
    client = null;     // reset so next call will retry
    if (error instanceof Error) {
      throw new Error(`Failed to connect to database: ${error.message}`);
    }
    throw new Error('Failed to connect to database');
  }
}

export async function getClient() {
  return getMongoClient();
}

export async function authenticateUser(email: string, password: string) {
  // Ensure we're connected before proceeding
  const db = await connectToDatabase();
  const users = db.collection('User');

  try {
    const user = await users.findOne({ email });
    if (!user) {
      throw new Error('No account found with this email');
    }
    if (user.password !== password) {
      throw new Error('Incorrect password');
    }
    return { user: { email: user.email, _id: user._id } };
  } catch (error) {
    console.error('Authentication error:', error);
    // Reset client on connection errors to force reconnect on next attempt
    if (error instanceof Error && 
        (error.name === 'MongoNetworkError' || error.message.includes('not connected'))) {
      client = null;
    }
    throw error;
  }
}
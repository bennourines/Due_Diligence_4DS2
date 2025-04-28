# ... (existing imports)
import logging
import time
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware # Import CORS middleware
# ... (rest of imports)

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) 
# --- Lifespan Events ---
# ... (existing lifespan function)

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Due Diligence API", # Example title
    description="API for the Crypto Due Diligence Assistant", # Example description
    version="0.1.0", # Example version
    # ... (existing app config)
)

# --- Middleware ---

# CORS Middleware (Add this section)
# Define allowed origins (where the frontend is running)
# IMPORTANT: Adjust origins for production deployment
origins = [
    "http://localhost",      # Base localhost
    "http://localhost:8501", # Default Streamlit port
    # Add the URL of your deployed frontend if applicable
    # e.g., "https://your-frontend-domain.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True, # Allow cookies if needed for auth methods
    allow_methods=["*"],    # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],    # Allow all headers
)
logger.info(f"CORS middleware enabled for origins: {origins}")

# Request Timing Middleware (Existing)
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    # Get the actual response object by calling call_next
    actual_response = await call_next(request)
    process_time = time.time() - start_time
    # Add the header to the actual response
    actual_response.headers["X-Process-Time"] = str(process_time)
    logger.info(f"Request {request.method} {request.url.path} processed in {process_time:.4f} seconds")
    # Return the actual response object, not the shadowed module name
    return actual_response

# --- API Routers ---
# ... (existing router includes)

# --- Root Endpoint ---
# ... (existing root endpoint)

# --- Custom Exception Handlers ---
# ... (existing exception handlers)

# --- Run with Uvicorn ---
# ... (existing uvicorn run block)

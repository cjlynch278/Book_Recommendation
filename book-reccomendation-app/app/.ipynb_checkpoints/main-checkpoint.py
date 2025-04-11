import asyncio
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.db.database import create_db_and_tables
from app.api.routes.health import router as health_router

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
print(f"Using database: {settings.DATABASE_URL}")

# Initialize FastAPI app
app = FastAPI(
    title="Book Recommendation API",
    version="1.0",
    description="An AI-powered book recommendation service."
)

# CORS Middleware (Allows frontend apps to make API requests)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Routes
#app.include_router(users.router, prefix="/api/users", tags=["Users"])
app.include_router(health_router, prefix="/api/health", tags=["Health"])

# Startup Event (Run database migrations)
@app.on_event("startup")
async def startup_event():
    logger.info("Starting Book Recommendation API...")
    await create_db_and_tables()

# Shutdown Event (Cleanup tasks)
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down Book Recommendation API...")

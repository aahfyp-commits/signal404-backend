from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def create_app() -> FastAPI:
    """Application factory"""
    app = FastAPI(
        title="Deepfake Audio Detection API",
        description="LSTM-based deepfake audio detection using MFCC features",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # CORS for frontend connection
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Change to your frontend domain in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Root endpoint
    @app.get("/")
    async def root():
        return {
            "message": "Deepfake Audio Detection API",
            "version": "1.0.0",
            "status": "running",
            "docs": "http://localhost:8000/docs",
            "health": "http://localhost:8000/api/health"
        }
    
    # Try to load routes, but don't fail if model is missing
    try:
        from app.api.routes import router
        app.include_router(router, prefix="/api")
    except Exception as e:
        logging.warning(f"Could not load routes: {e}")
        logging.warning("API will serve only root endpoint")
    
    @app.on_event("startup")
    async def startup_event():
        logging.info("🚀 API starting up...")
    
    @app.on_event("shutdown")
    async def shutdown_event():
        logging.info("👋 API shutting down...")
    
    return app

app = create_app()
import os
import sys
import uvicorn

def main():
    """Run the FastAPI server from project root"""
    # Add frontend directory to Python path
    frontend_dir = os.path.join(os.path.dirname(__file__), "frontend")
    sys.path.insert(0, frontend_dir)
    
    print("Starting AI SpillGuard Server...")
    print("Server will be available at: http://localhost:8000")
    print("API Documentation: http://localhost:8000/docs")
    print("Press Ctrl+C to stop the server")
    
    try:
        # Change to frontend directory for proper file paths
        os.chdir(frontend_dir)
        
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Error starting server: {e}")

if __name__ == "__main__":
    main()
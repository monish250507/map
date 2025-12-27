import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.abspath('.'))

# Import the app
from eco_routing_updated.main import app

# Run the server
if __name__ == "__main__":
    import uvicorn
    print("Starting eco-routing backend server on http://127.0.0.1:8000")
    print("Please wait for the server to fully start...")
    uvicorn.run(app, host="127.0.0.1", port=8000)
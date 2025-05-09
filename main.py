import os
import logging
from app import app
from routes import configure_routes

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Configure routes
configure_routes(app)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

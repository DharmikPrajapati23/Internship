#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Print a message to the Render logs indicating the script is running
echo "Starting Streamlit application via entrypoint.sh"

# Run the Streamlit application.
# --server.port $PORT: Tells Streamlit to listen on the port provided by Render.
# --server.enableCORS false: Disables CORS checks for simplicity in deployment.
# --server.enableXsrfProtection false: Disables XSRF protection for simplicity.
# These parameters are crucial for deployment on platforms like Render.
streamlit run app.py --server.port $PORT --server.enableCORS false --server.enableXsrfProtection false

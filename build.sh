#!/bin/bash
# Build script for Render deployment
# This installs both Node.js and Python dependencies

echo "🔧 Installing Node.js dependencies..."
npm install

echo "🐍 Installing Python dependencies..."
pip3 install -r requirements.txt

echo "✅ Build complete!"

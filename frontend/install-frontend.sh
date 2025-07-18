#!/bin/bash

echo "🚀 Installing React Frontend for Indian Stock Trading Bot"
echo "========================================================"
echo

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed or not in PATH"
    echo "Please install Node.js from https://nodejs.org"
    echo "Recommended version: Node.js 16 or higher"
    exit 1
fi

echo "✅ Node.js found"
node --version
echo

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo "❌ npm is not installed or not in PATH"
    exit 1
fi

echo "✅ npm found"
npm --version
echo

# Install dependencies
echo "📦 Installing React dependencies..."
echo "This may take a few minutes..."
echo

npm install

if [ $? -ne 0 ]; then
    echo "❌ Failed to install dependencies"
    echo "Please check your internet connection and try again"
    exit 1
fi

echo
echo "✅ Dependencies installed successfully!"
echo
echo "🎉 React frontend is ready!"
echo
echo "🚀 To start the development server:"
echo "   npm start"
echo
echo "🏗️  To build for production:"
echo "   npm run build"
echo
echo "🧪 To run tests:"
echo "   npm test"
echo
echo "📚 The frontend will be available at:"
echo "   http://localhost:3000"
echo
echo "🔗 Make sure the backend is running at:"
echo "   http://127.0.0.1:5000"
echo

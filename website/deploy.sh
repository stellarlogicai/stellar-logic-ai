#!/bin/bash

# Stellar Logic AI - Netlify Deployment Script
echo "🚀 STELLOR LOGIC AI - NETLIFY DEPLOYMENT"
echo "=================================="

# Check if Netlify CLI is installed
if ! command -v netlify &> /dev/null; then
    echo "❌ Netlify CLI not found. Installing..."
    npm install -g netlify-cli
fi

# Check if logged in to Netlify
if ! netlify whoami &> /dev/null; then
    echo "🔐 Please login to Netlify:"
    netlify login
fi

# Deploy to Netlify
echo "📦 Deploying to Netlify..."
netlify deploy --prod --dir=.

echo "✅ Deployment complete!"
echo "🌐 Your site is live at: https://stellarlogicai.netlify.app"
echo "📊 Check Netlify dashboard for deployment details"

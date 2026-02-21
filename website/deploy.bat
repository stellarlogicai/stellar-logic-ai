@echo off
REM Stellar Logic AI - Netlify Deployment Script (Windows)
echo 🚀 STELLOR LOGIC AI - NETLIFY DEPLOYMENT
echo ==================================

REM Check if Netlify CLI is installed
netlify >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Netlify CLI not found. Installing...
    npm install -g netlify-cli
)

REM Check if logged in to Netlify
netlify whoami >nul 2>&1
if %errorlevel% neq 0 (
    echo 🔐 Please login to Netlify:
    netlify login
)

REM Deploy to Netlify
echo 📦 Deploying to Netlify...
netlify deploy --prod --dir=.

echo ✅ Deployment complete!
echo 🌐 Your site is live at: https://stellarlogicai.netlify.app
echo 📊 Check Netlify dashboard for deployment details
pause

# Environment Setup Guide

This guide explains how to set up the `OPENROUTER_API_KEY` environment variable for the Rayfield Maintenance Automation platform.

## What is OPENROUTER_API_KEY?

The `OPENROUTER_API_KEY` is required to access the Qwen AI model for generating intelligent analysis reports. This key is used to authenticate with the OpenRouter API service.

## How to Get Your API Key

1. **Visit OpenRouter**: Go to https://openrouter.ai/
2. **Sign up/Login**: Create an account or log in
3. **Get Free Credits**: OpenRouter provides free credits for new users
4. **Copy Your API Key**: Find your API key in the dashboard

## Local Development Setup

### Option 1: Using a .env file (Recommended)

1. **Create a .env file** in your project root:
   ```bash
   echo "OPENROUTER_API_KEY=your_actual_api_key_here" > .env
   ```

2. **Install python-dotenv** (if not already installed):
   ```bash
   pip install python-dotenv
   ```

3. **The app will automatically load** the .env file

### Option 2: Set Environment Variable Directly

**Windows (PowerShell):**
```powershell
$env:OPENROUTER_API_KEY="your_actual_api_key_here"
```

**Windows (Command Prompt):**
```cmd
set OPENROUTER_API_KEY=your_actual_api_key_here
```

**Linux/Mac:**
```bash
export OPENROUTER_API_KEY="your_actual_api_key_here"
```

## Render Deployment Setup

When deploying to Render:

1. **Go to your Render dashboard**
2. **Navigate to your service**
3. **Go to Environment tab**
4. **Add environment variable:**
   - **Key:** `OPENROUTER_API_KEY`
   - **Value:** `your_actual_api_key_here`

## Security Notes

- **Never commit your API key** to version control
- **Use environment variables** for all deployments
- **Keep your API key private** and secure
- **Rotate keys regularly** for security 
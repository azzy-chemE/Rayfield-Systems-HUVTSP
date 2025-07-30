# Render Deployment Guide

## Memory and Timeout Issues

If you're experiencing `WORKER TIMEOUT` and `SIGKILL` errors on Render, follow these steps:

### 1. Environment Variables
Set these in your Render dashboard:
- `OPENROUTER_API_KEY`: Your OpenRouter API key
- `RENDER`: Set to `true` (this enables lightweight mode automatically)

### 2. Render Configuration
The `render.yaml` file is already configured with:
- Single worker (`--workers 1`)
- Extended timeout (`--timeout 300`)
- Memory-efficient settings

### 3. Lightweight Mode
The application automatically runs in lightweight mode on Render, which:
- Skips chart generation to save memory
- Still provides AI analysis and data insights
- Reduces processing time

### 4. Troubleshooting Steps

#### If you still get timeout errors:
1. **Check your data size**: Large CSV files (>10MB) may cause issues
2. **Reduce data**: Consider sampling your data for testing
3. **Monitor logs**: Check Render logs for specific error messages

#### If you get memory errors:
1. **Enable lightweight mode**: Set `RENDER=true` in environment variables
2. **Reduce dependencies**: The app already uses minimal memory settings
3. **Upgrade plan**: Consider upgrading to a paid Render plan for more resources

### 5. Local Testing
To test the lightweight mode locally:
```bash
export RENDER=true
python app.py
```

### 6. Alternative Solutions
If issues persist:
1. **Use a smaller dataset** for testing
2. **Disable chart generation** entirely by setting `lightweight: true` in the frontend
3. **Consider alternative hosting** like Heroku or Railway for more resources

## Expected Behavior
- **Local development**: Full functionality with charts
- **Render deployment**: Lightweight mode (no charts) to prevent timeouts
- **AI analysis**: Still works in both modes
- **Data insights**: Available in both modes

## Monitoring
Check your Render logs for:
- Memory usage warnings
- Timeout messages
- API key errors
- Chart generation errors 
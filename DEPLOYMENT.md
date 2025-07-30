# Render Deployment Guide

## Memory and Timeout Issues

If you're experiencing `WORKER TIMEOUT` and `SIGKILL` errors on Render, follow these steps:

### 1. Environment Variables
Set these in your Render dashboard:
- `OPENROUTER_API_KEY`: Your OpenRouter API key
- `RENDER`: Set to `true` (this enables lightweight mode automatically)
- `PYTHON_VERSION`: Set to `3.9.16` (already in render.yaml)

### 2. Render Configuration
The `render.yaml` file is already configured with:
- Single worker (`--workers 1`)
- Extended timeout (`--timeout 300`)
- Memory-efficient settings
- Proper Python version

### 3. Lightweight Mode
The application automatically runs in lightweight mode on Render, which:
- Skips chart generation to save memory
- Still provides AI analysis and data insights
- Reduces processing time
- Prevents timeout errors

### 4. Troubleshooting Steps

#### If you still get timeout errors:
1. **Check your data size**: Large CSV files (>10MB) may cause issues
2. **Reduce data**: Consider sampling your data for testing
3. **Monitor logs**: Check Render logs for specific error messages
4. **Verify environment variables**: Ensure all required variables are set

#### If you get memory errors:
1. **Enable lightweight mode**: Set `RENDER=true` in environment variables
2. **Reduce dependencies**: The app already uses minimal memory settings
3. **Upgrade plan**: Consider upgrading to a paid Render plan for more resources
4. **Check file uploads**: Large files may cause memory issues

#### If you get import errors:
1. **Check requirements.txt**: All dependencies are listed
2. **Verify Python version**: Using 3.9.16 as specified
3. **Check build logs**: Look for missing dependencies

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
- Import errors

## Render-Specific Optimizations

### 1. Automatic Detection
The app automatically detects Render environment using:
- `RENDER` environment variable
- `PORT` environment variable  
- `RENDER_EXTERNAL_URL` environment variable

### 2. Error Handling
- File operations wrapped in try-catch blocks
- Graceful fallbacks for missing directories
- Memory cleanup after operations
- Timeout warnings for long operations

### 3. Memory Management
- Single worker configuration
- Lightweight mode by default on Render
- Garbage collection after operations
- Reduced chart resolution

### 4. File System
- Robust directory creation
- Error handling for file operations
- Fallback mechanisms for missing files
- Temporary file cleanup

## Deployment Checklist

Before deploying to Render:
- [ ] Set `OPENROUTER_API_KEY` environment variable
- [ ] Set `RENDER=true` environment variable
- [ ] Verify `render.yaml` configuration
- [ ] Check `requirements.txt` has all dependencies
- [ ] Test locally with `RENDER=true`

## Common Issues and Solutions

### Issue: "No module named 'seaborn'"
**Solution**: All dependencies are in `requirements.txt`, Render will install them automatically.

### Issue: "Worker timeout"
**Solution**: The app automatically uses lightweight mode on Render to prevent timeouts.

### Issue: "Memory error"
**Solution**: The app includes memory optimization and cleanup mechanisms.

### Issue: "File not found"
**Solution**: The app includes robust error handling for file operations.

The application is optimized for Render deployment and should work reliably! 🚀 
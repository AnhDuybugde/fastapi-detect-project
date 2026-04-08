# Frontend Deployment Script for Vercel

## **Step 1: Install Vercel CLI**
```bash
# Install Vercel CLI
npm i -g vercel

# Login to Vercel
vercel login
```

## **Step 2: Deploy Frontend**
```bash
# Navigate to frontend folder
cd frontend

# Deploy to Vercel
vercel --prod

# Follow prompts:
# - Set up and deploy "~/frontend"? [Y/n] Y
# - Which scope do you want to deploy to? (Your account)
# - Link to existing project? [y/N] N
# - What's your project's name? fastapi-detect-project
# - In which directory is your code located? ./
# - Want to override the settings? [y/N] N
```

## **Step 3: Verify Deployment**
```bash
# Test the deployed frontend
curl https://fastapi-detect-project.vercel.app

# Should return the HTML content
```

## **Step 4: Test Integration**
```bash
# Test API calls from frontend
# Open browser: https://fastapi-detect-project.vercel.app
# Check if API status shows "Da ket noi"
# Try uploading an image
```

## **Troubleshooting**

### **If deployment fails:**
```bash
# Clean cache
vercel --prod --force

# Check vercel.json
cat vercel.json

# Verify file structure
ls -la
```

### **If API calls fail:**
```bash
# Check CORS
curl -H "Origin: https://fastapi-detect-project.vercel.app" \
     -H "Access-Control-Request-Method: POST" \
     -H "Access-Control-Request-Headers: X-Requested-With" \
     -X OPTIONS \
     https://fastapi-detect-project.onrender.com/api/health

# Should return 200 with CORS headers
```

### **If frontend not loading:**
```bash
# Check Vercel logs
vercel logs

# Check deployment URL
vercel ls
```

## **Expected Result**

After successful deployment:
- **Frontend URL**: https://fastapi-detect-project.vercel.app
- **API Status**: Should show "Da ket noi"
- **Image Upload**: Should work with Render backend
- **Detection Results**: Should display bounding boxes

## **Manual Verification**

1. **Open browser**: https://fastapi-detect-project.vercel.app
2. **Check API status**: Should be green "Da ket noi"
3. **Upload image**: Test with any image file
4. **View results**: Should show detection results
5. **Check performance**: Should be fast with CDN

## **Next Steps**

Once frontend is deployed:
1. Test full workflow
2. Monitor performance
3. Check error logs
4. Optimize if needed

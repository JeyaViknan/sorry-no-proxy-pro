# 🚀 Render Deployment Guide

This guide will help you deploy the attendance app with face verification to Render.

## 📋 Prerequisites

1. **Render Account**: Sign up at [render.com](https://render.com)
2. **GitHub Repository**: Push your code to GitHub (or use Render's Git integration)
3. **Google Service Account**: Set up Google Sheets API credentials
4. **Python 3.x**: Render supports Python 3.x (will be installed during build)

## 🔧 Step 1: Prepare Your Repository

### Files Required in Repository:
- ✅ `server.js` - Express server
- ✅ `package.json` - Node.js dependencies
- ✅ `requirements.txt` - Python dependencies
- ✅ `face_verification.py` - Face verification script
- ✅ `face_classifier_model.pkl` - Trained model (must be in repo)
- ✅ `data/labels.xlsx` - Registration number mappings
- ✅ `data/images/` - Student face images directory
- ✅ `index.html` - Frontend
- ✅ `styles.css` - Styles
- ✅ `js/html5-qrcode.min.js` - QR scanner library
- ✅ `render.yaml` - Render configuration (optional)

## 🔑 Step 2: Environment Variables Setup

In Render Dashboard → Your Service → Environment:

### Required Environment Variables:

1. **PORT** (Auto-set by Render, but you can override)
   ```
   PORT=10000
   ```

2. **GOOGLE_SERVICE_ACCOUNT_EMAIL**
   ```
   GOOGLE_SERVICE_ACCOUNT_EMAIL=your-service-account@project-id.iam.gserviceaccount.com
   ```

3. **GOOGLE_PRIVATE_KEY**
   ```
   GOOGLE_PRIVATE_KEY=-----BEGIN PRIVATE KEY-----\nYour\nPrivate\nKey\nHere\n-----END PRIVATE KEY-----
   ```
   ⚠️ **Important**: Include `\n` for newlines in the private key, or paste the entire key with actual newlines.

4. **SHEET_ID**
   ```
   SHEET_ID=your-google-sheet-id-here
   ```
   (Get this from your Google Sheet URL: `https://docs.google.com/spreadsheets/d/{SHEET_ID}/edit`)

## 🏗️ Step 3: Render Service Configuration

### Option A: Using Render Dashboard (Recommended)

1. **Go to Render Dashboard** → New → Web Service

2. **Connect Repository**:
   - Connect your GitHub/GitLab repository
   - Select the repository containing your code

3. **Configure Service**:
   - **Name**: `attendance-app` (or your preferred name)
   - **Environment**: `Node`
   - **Region**: Choose closest to your users
   - **Branch**: `main` (or your default branch)
   - **Root Directory**: Leave empty (if root) or specify if in subdirectory

4. **Build & Deploy Settings**:
   - **Build Command**: 
     ```bash
     npm install && pip3 install -r requirements.txt
     ```
   - **Start Command**: 
     ```bash
     node server.js
     ```

5. **Advanced Settings**:
   - **Plan**: Free (or paid for better performance)
   - **Auto-Deploy**: Yes (deploys on every push)

### Option B: Using render.yaml (Infrastructure as Code)

If you created `render.yaml`, Render will automatically detect it:

1. **Push render.yaml to your repository**
2. **In Render Dashboard**: New → Blueprint → Connect repository
3. Render will automatically create the service from `render.yaml`

## 📝 Step 4: Update Frontend API Endpoints

Update `index.html` to use your Render URL:

```javascript
const apiEndpoint = "https://your-app-name.onrender.com/register";
const verifyEndpoint = "https://your-app-name.onrender.com/verify-face";
```

Replace `your-app-name` with your actual Render service name.

## 🔍 Step 5: Verify Deployment

### Test Endpoints:

1. **Health Check**:
   ```
   GET https://your-app-name.onrender.com/
   ```
   Should return: `✅ Server is running! Use POST /register to register.`

2. **Face Verification**:
   ```bash
   POST https://your-app-name.onrender.com/verify-face
   Body: {
     "registerNumber": "25BAI1704",
     "faceImage": "data:image/jpeg;base64,..."
   }
   ```

3. **Registration**:
   ```bash
   POST https://your-app-name.onrender.com/register
   Body: {
     "registerNumber": "25BAI1704",
     "faceImage": "data:image/jpeg;base64,..."
   }
   ```

## 🐛 Troubleshooting

### Issue: Python dependencies not installing
**Solution**: Ensure `requirements.txt` exists and has correct packages:
```
pandas>=1.5.0
numpy>=1.23.0
opencv-python>=4.6.0
openpyxl>=3.0.0
```

### Issue: Model file not found
**Solution**: 
- Ensure `face_classifier_model.pkl` is committed to Git
- Check file size (Render free tier has limits)
- Verify path in `face_verification.py` uses `__dirname`

### Issue: Google Sheets API errors
**Solution**:
- Verify environment variables are set correctly
- Check private key format (must include `\n` for newlines)
- Ensure service account has access to the Google Sheet
- Share Google Sheet with service account email

### Issue: Face verification timeout
**Solution**:
- Free tier has 15-minute timeout
- Consider upgrading to paid plan for better performance
- Optimize image processing in Python script

### Issue: Static files not loading
**Solution**:
- Verify `app.use(express.static(__dirname))` is in server.js
- Check file paths are correct
- Ensure all static files are in repository

## 📊 Render Service Endpoints Summary

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check / Server status |
| `/verify-face` | POST | Verify face matches registration number |
| `/register` | POST | Register attendance (with face verification) |
| `/index.html` | GET | Frontend application |
| `/styles.css` | GET | CSS styles |
| `/js/html5-qrcode.min.js` | GET | QR scanner library |

## 🔒 Security Notes

1. **CORS**: Currently set to `origin: "*"` - consider restricting in production
2. **Environment Variables**: Never commit `.env` file to Git
3. **Private Key**: Store securely in Render environment variables
4. **HTTPS**: Render provides HTTPS automatically

## 📈 Monitoring

- **Logs**: View in Render Dashboard → Your Service → Logs
- **Metrics**: Available in paid plans
- **Alerts**: Set up in Render Dashboard for service downtime

## 🚀 Post-Deployment Checklist

- [ ] Environment variables configured
- [ ] Frontend API endpoints updated with Render URL
- [ ] Test health check endpoint
- [ ] Test face verification endpoint
- [ ] Test registration endpoint
- [ ] Verify Google Sheets integration
- [ ] Test full workflow (QR scan → Face capture → Registration)
- [ ] Check logs for any errors
- [ ] Update any hardcoded URLs in code

## 💡 Tips

1. **Free Tier Limitations**:
   - Services spin down after 15 minutes of inactivity
   - First request after spin-down may be slow (~30 seconds)
   - Consider using a paid plan for production

2. **Performance**:
   - Model loading happens on first request (may be slow)
   - Consider caching model in memory
   - Optimize image processing

3. **Scaling**:
   - Free tier: 1 instance
   - Paid plans: Auto-scaling available

---

**Need Help?** Check Render documentation: https://render.com/docs

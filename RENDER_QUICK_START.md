# ⚡ Quick Start: Deploy to Render

## 🎯 Quick Deployment Steps

### 1. Push Code to GitHub
```bash
git add .
git commit -m "Prepare for Render deployment"
git push origin main
```

### 2. Create Render Service

1. Go to [render.com](https://render.com) → Dashboard
2. Click **"New +"** → **"Web Service"**
3. Connect your GitHub repository
4. Configure:

   **Basic Settings:**
   - Name: `attendance-app`
   - Environment: `Node`
   - Region: `Oregon` (or closest to you)
   - Branch: `main`

   **Build & Deploy:**
   - Build Command: `npm install && pip3 install -r requirements.txt`
   - Start Command: `node server.js`

### 3. Set Environment Variables

In Render Dashboard → Your Service → Environment → Add Environment Variable:

| Key | Value | Notes |
|-----|-------|-------|
| `GOOGLE_SERVICE_ACCOUNT_EMAIL` | `your-email@project.iam.gserviceaccount.com` | From Google Cloud Console |
| `GOOGLE_PRIVATE_KEY` | `-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----` | Include `\n` for newlines |
| `SHEET_ID` | `your-sheet-id` | From Google Sheet URL |

**Note:** `PORT` is automatically set by Render (usually 10000)

### 4. Update Frontend URLs

Edit `index.html` line 46-47:
```javascript
const apiEndpoint = "https://attendance-app.onrender.com/register";
const verifyEndpoint = "https://attendance-app.onrender.com/verify-face";
```

Replace `attendance-app` with your actual service name.

### 5. Deploy

- Click **"Create Web Service"**
- Wait for build to complete (~2-5 minutes)
- Your app will be live at: `https://your-service-name.onrender.com`

## ✅ Verify Deployment

1. Visit: `https://your-service-name.onrender.com/`
   - Should see: "✅ Server is running!"

2. Visit: `https://your-service-name.onrender.com/index.html`
   - Should see the QR scanner interface

## 🔍 Available Endpoints

- `GET /` - Health check
- `GET /index.html` - Frontend app
- `POST /verify-face` - Face verification
- `POST /register` - Register attendance

## ⚠️ Important Notes

1. **Free Tier**: Service spins down after 15 min inactivity (first request may be slow)
2. **Model File**: Ensure `face_classifier_model.pkl` is in your repository
3. **Data Files**: Ensure `data/labels.xlsx` and `data/images/` are committed
4. **Python**: Render auto-installs Python 3.x during build

## 🐛 Common Issues

**Build fails?**
- Check `requirements.txt` exists
- Verify Python packages are correct

**Face verification fails?**
- Check model file is in repository
- Verify data/images/ directory exists
- Check logs in Render Dashboard

**Google Sheets error?**
- Verify environment variables are set
- Check private key format (include `\n`)
- Ensure service account has sheet access

---

**Full Guide**: See `DEPLOYMENT.md` for detailed instructions.

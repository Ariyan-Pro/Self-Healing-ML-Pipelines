# 🚀 Deploy Self-Healing ML Pipelines to Hugging Face Spaces

## Complete Step-by-Step Guide

This guide will help you deploy your Self-Healing ML Pipelines with the Human Veto Dashboard to Hugging Face Spaces.

---

## 📋 Prerequisites

1. **Hugging Face Account**: You already have one (`Ariyan-Pro`)
2. **Existing Space**: You already have `Ariyan-Pro/Self-Healing-ML-Pipelines`
3. **No Local Cloning Required**: We'll do everything through the HF web interface

---

## 🎯 What You'll Get

After deployment:
- ✅ Landing page at: `https://ariyan-pro-self-healing-ml-pipelines.hf.space/`
- ✅ Dashboard at: `https://ariyan-pro-self-healing-ml-pipelines.hf.space/dashboard`
- ✅ API endpoints available for programmatic access
- ✅ Persistent veto storage (survives restarts)

---

## 📦 Files You Need to Upload

### Option A: Manual Upload (Recommended for You)

Since you prefer manual control, here's exactly what to upload:

#### 1. **Core Application Files** (Upload to root of HF Space)

| File | From Your Computer | Purpose |
|------|-------------------|---------|
| `app.py` | Create this file (see below) | Main Gradio/Flask app entry point |
| `requirements.txt` | Already exists ✓ | Python dependencies |
| `index.html` | Your existing landing page | Main landing page |
| `README.md` | Already exists ✓ | Documentation |

#### 2. **API Directory** (Create `api/` folder in HF Space)

Upload these files into the `api/` folder:

| File | Purpose |
|------|---------|
| `api/api_server.py` | Main API server |
| `api/human_veto_endpoint.py` | Human Veto API endpoint |
| `api/veto_dashboard.html` | Dashboard UI |
| `api/__init__.py` | Python package marker |

#### 3. **Logs Directory** (Create `logs/` folder)

Create an empty `logs/` folder - it will auto-populate with veto data.

---

## 🔧 Step-by-Step Upload Process

### Step 1: Go to Your Hugging Face Space

1. Navigate to: https://huggingface.co/spaces/Ariyan-Pro/Self-Healing-ML-Pipelines
2. Click on the **"Files"** tab
3. You'll see your current file structure

### Step 2: Create the Main App File

Click **"Add file"** → **"Create a new file"**

**Filename**: `app.py`

**Content** (copy-paste this entire block):

```python
#!/usr/bin/env python3
"""
Main Entry Point for Hugging Face Spaces Deployment
Self-Healing ML Pipelines with Human Veto Dashboard
"""

import subprocess
import threading
import time
from pathlib import Path

def start_api_server():
    """Start the API server in a background thread."""
    from api.api_server import MainAPIServer
    
    server = MainAPIServer(host='0.0.0.0', port=8080)
    server.start(blocking=False)
    return server

def main():
    """Main entry point for Hugging Face Spaces."""
    
    print("="*70)
    print("🚀 Self-Healing ML Pipelines - Hugging Face Spaces")
    print("="*70)
    
    # Start API server in background
    print("\n📡 Starting API Server...")
    api_server = start_api_server()
    
    # Keep the main thread alive
    print("\n✅ Application is running!")
    print(f"   Landing Page: http://localhost:8080/")
    print(f"   Dashboard: http://localhost:8080/dashboard")
    print(f"   API: http://localhost:8080/api/v1/human-veto")
    print("="*70)
    
    # Keep running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
        api_server.stop()

if __name__ == "__main__":
    main()
```

Click **"Commit new file to main"**

### Step 3: Upload the API Files

For each file below:
1. Click **"Add file"** → **"Upload files"**
2. Select the file from your computer
3. Make sure it goes in the correct folder
4. Click **"Commit changes to main"**

#### Upload to Root:
- ✅ `index.html` (your existing landing page)
- ✅ `requirements.txt` (already there, verify it has the dependencies below)

#### Create `api/` Folder and Upload:
1. Click **"Add file"** → **"Upload files"**
2. When uploading, in the file path, type: `api/api_server.py`
3. Upload your local `api/api_server.py` file
4. Repeat for:
   - `api/human_veto_endpoint.py`
   - `api/veto_dashboard.html`
   - `api/__init__.py`

### Step 4: Verify requirements.txt

Make sure your `requirements.txt` contains:

```txt
flask>=2.0.0
requests>=2.25.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
joblib>=1.1.0
PyYAML>=6.0
```

If not, edit it by clicking on the file → pencil icon → update → commit.

### Step 5: Create logs Directory

1. Click **"Add file"** → **"Create a new file"**
2. Filename: `logs/.gitkeep`
3. Content: (leave empty or add a comment)
4. Commit

This ensures the `logs/` folder exists for storing veto data.

### Step 6: Configure the Space

1. Go to **"Settings"** tab
2. Under **"Space SDK"**, make sure it's set to **"Gradio"** or **"Docker"**
3. For this deployment, we recommend **"Docker"** for full control

#### If Using Docker:

Create a `Dockerfile` in the root:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["python", "app.py"]
```

### Step 7: Wait for Build

After uploading all files:
1. Go to the **"App"** tab
2. You'll see "Building..." status
3. Wait 2-5 minutes for the build to complete
4. Once ready, you'll see "Running" with a green dot

---

## 🎉 Access Your Deployed Application

Once deployed, you can access:

### Landing Page
```
https://ariyan-pro-self-healing-ml-pipelines.hf.space/
```

### Human Veto Dashboard
Click the "🎛️ Dashboard" button in the navbar, or go directly to:
```
https://ariyan-pro-self-healing-ml-pipelines.hf.space/dashboard
```

### API Endpoints
```bash
# List pending vetoes
curl https://ariyan-pro-self-healing-ml-pipelines.hf.space/api/v1/human-veto

# Create a veto
curl -X POST https://ariyan-pro-self-healing-ml-pipelines.hf.space/api/v1/human-veto \
  -H "Content-Type: application/json" \
  -d '{"action_id": "test-123", "action_type": "model_retrain", "reason": "Testing"}'

# Health check
curl https://ariyan-pro-self-healing-ml-pipelines.hf.space/health
```

---

## 🔍 Troubleshooting

### Issue: Space shows "Error"

**Solution**: 
1. Go to **"Settings"** → **"Logs"**
2. Check the build logs for errors
3. Common issues:
   - Missing dependencies in `requirements.txt`
   - File path errors (check case sensitivity)
   - Port conflicts (make sure we use 8080)

### Issue: Dashboard button doesn't work

**Solution**:
1. Verify `index.html` has the dashboard link (line 786)
2. Check that `api/veto_dashboard.html` exists
3. Try direct URL: `https://YOUR-SPACE.hf.space/dashboard`

### Issue: API returns 404

**Solution**:
1. Make sure `app.py` is running the API server
2. Check logs to confirm server started on port 8080
3. Verify `api/` folder structure is correct

---

## 📊 Testing Your Deployment

### Test 1: Landing Page Loads
✅ Visit your space URL  
✅ See your beautiful landing page  
✅ All sections visible (Features, Architecture, Validation, etc.)

### Test 2: Dashboard Accessible
✅ Click "🎛️ Dashboard" in navbar  
✅ Dashboard loads with statistics  
✅ Shows: Pending, Approved, Rejected counts

### Test 3: Create a Veto
✅ On dashboard, click "New Veto Request"  
✅ Fill in the form  
✅ Submit  
✅ See it appear in "Pending Veto Requests"

### Test 4: Approve/Reject
✅ Click "Approve" or "Reject" on a pending veto  
✅ See status change  
✅ Watch statistics update

### Test 5: Persistence
✅ Refresh the page  
✅ Your vetoes should still be there (stored in `logs/veto_store.json`)

---

## 🔄 Updating Your Deployment

To make changes later:

1. Go to your Space → **"Files"** tab
2. Click on any file to edit it
3. Or click **"Add file"** to upload new files
4. After committing, HF will automatically rebuild
5. Wait 1-2 minutes for updates to go live

---

## 💡 Pro Tips

### Tip 1: Private vs Public
- Your space is currently **public** (anyone can see it)
- To make it private: Settings → Privacy → Set to Private
- Note: Private spaces require Hugging Face Pro for some features

### Tip 2: Custom Domain
- You can add a custom domain in Settings
- Great for professional portfolios

### Tip 3: Environment Variables
- Add secrets in Settings → "Repository secrets"
- Useful for API keys, database URLs, etc.

### Tip 4: Monitor Usage
- Check the "Usage" tab to see traffic
- Helpful for understanding adoption

---

## 📁 Final File Structure

Your HF Space should look like this:

```
/
├── app.py                    ← Main entry point (CREATE THIS)
├── index.html                ← Your landing page (UPLOAD)
├── requirements.txt          ← Dependencies (VERIFY)
├── README.md                 ← Documentation (EXISTS)
├── Dockerfile                ← Container config (OPTIONAL)
│
├── api/                      ← API folder (CREATE & UPLOAD)
│   ├── __init__.py
│   ├── api_server.py
│   ├── human_veto_endpoint.py
│   └── veto_dashboard.html
│
└── logs/                     ← Data storage (CREATE)
    └── veto_store.json       ← Auto-created on first use
```

---

## 🎯 Quick Checklist

Before you start, make sure you have:

- [ ] Local copy of `index.html`
- [ ] Local copy of `api/api_server.py`
- [ ] Local copy of `api/human_veto_endpoint.py`
- [ ] Local copy of `api/veto_dashboard.html`
- [ ] Local copy of `api/__init__.py`
- [ ] Access to HF Space: `Ariyan-Pro/Self-Healing-ML-Pipelines`
- [ ] 10-15 minutes of focused time

---

## 🆘 Need Help?

If you get stuck:

1. **Check the Logs**: Settings → Logs
2. **Verify File Structure**: Files tab should match the structure above
3. **Test Locally First**: Run `python app.py` locally to verify it works
4. **HF Documentation**: https://huggingface.co/docs/hub/spaces

---

## ✅ Success Indicators

You've successfully deployed when:

1. ✅ Landing page loads without errors
2. ✅ Dashboard button is visible in navbar
3. ✅ Clicking dashboard shows the Human Veto UI
4. ✅ You can create, approve, and reject vetoes
5. ✅ Data persists after page refresh
6. ✅ API endpoints respond correctly

---

**Good luck with your deployment! 🚀**

Your Self-Healing ML Pipelines project with the Human Veto Dashboard will be live and accessible to anyone with the link!

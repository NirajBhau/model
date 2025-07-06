# Streamlit Cloud Deployment Guide

## 🚀 Quick Deployment Steps

1. **Push your code to GitHub:**
   ```bash
   git add .
   git commit -m "Fix deployment issues"
   git push origin main
   ```

2. **Connect to Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub account
   - Select your repository: `churn-model`
   - Set main file path: `app.py`
   - Click "Deploy"

## 🔧 Troubleshooting Common Issues

### **1. Dependency Conflicts**
If you see numpy/pandas version conflicts:
- ✅ **Fixed**: Updated `requirements.txt` with compatible versions
- ✅ **Added**: `packages.txt` for system dependencies

### **2. Build Failures**
If the build fails:
- Check that all files are committed to GitHub
- Ensure `requirements.txt` is in the root directory
- Verify Python version compatibility

### **3. Runtime Errors**
If the app crashes after deployment:
- Check the logs in Streamlit Cloud dashboard
- Verify the CSV file path is correct
- Ensure all imports are working

## 📁 Required Files for Deployment

```
churn-model/
├── app.py                    # Main application
├── requirements.txt          # Python dependencies
├── packages.txt             # System dependencies
├── .streamlit/config.toml   # Streamlit configuration
├── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Dataset
└── README.md                # Documentation
```

## 🐛 Common Error Solutions

### **Error: "No module named 'pandas'"**
- Solution: Check `requirements.txt` has correct versions

### **Error: "File not found"**
- Solution: Ensure CSV file is in the repository root

### **Error: "Memory limit exceeded"**
- Solution: Optimize data loading with `@st.cache_data`

### **Error: "Build timeout"**
- Solution: Reduce model complexity or use pre-trained models

## ✅ Verification Checklist

- [ ] All files committed to GitHub
- [ ] `requirements.txt` has compatible versions
- [ ] CSV file is in the repository
- [ ] `app.py` is the main file
- [ ] No syntax errors in the code
- [ ] All imports are available in requirements.txt

## 🆘 Getting Help

If deployment still fails:

1. **Check Streamlit Cloud logs** for specific error messages
2. **Test locally first** with `streamlit run app.py`
3. **Simplify the app** temporarily to isolate the issue
4. **Contact Streamlit support** if the issue persists

## 🔄 Redeployment

After fixing issues:
1. Commit changes to GitHub
2. Streamlit Cloud will automatically redeploy
3. Check the new deployment logs
4. Test the live application

---

**Happy Deploying! 🚀** 
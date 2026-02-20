# 🔧 SERVICE ACCOUNT SETUP INSTRUCTIONS

## 📋 HOW TO GET YOUR ACTUAL SERVICE ACCOUNT KEY:

### **✅ STEP 1: GO TO GOOGLE CLOUD CONSOLE**
1. **Visit:** https://console.cloud.google.com
2. **Select your project** (or create new one)
3. **Go to:** IAM & Admin → Service Accounts

### **✅ STEP 2: FIND YOUR SERVICE ACCOUNT**
1. **Look for:** `stellar-logic-ai-assistant`
2. **Click on the service account name**

### **✅ STEP 3: GO TO KEYS TAB**
1. **Click on:** "KEYS" tab
2. **You should see:** Your existing keys listed

### **✅ STEP 4: CREATE NEW KEY**
1. **Click:** "+ ADD KEY" → "Create new key"
2. **Key settings:**
   - **Key type:** JSON
   - **Key name:** stellar-logic-ai-email-key
   - **Description:** Email service for AI assistant

### **✅ STEP 5: DOWNLOAD AND REPLACE**
1. **Click:** "CREATE" button
2. **Download:** The JSON file
3. **Replace:** The content in `./credentials/service-account-key.json`
   - **Copy the entire "private_key" value** from the downloaded file
   - **Replace:** `PASTE_ACTUAL_PRIVATE_KEY_HERE` with the actual private key

### **✅ STEP 6: TEST THE SETUP**
```bash
node test-service-account-email.js
```

## 🔧 WHAT THE PRIVATE KEY SHOULD LOOK LIKE:
```json
{
  "type": "service_account",
  "project_id": "your-project-id",
  "private_key_id": "your-private-key-id",
  "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKw...\n-----END PRIVATE KEY-----\n",
  "client_email": "stellar-logic-ai-assistant@stella-logic-ai-assistant.iam.gserviceaccount.com",
  "client_id": "116693304809772109774",
  ...
}
```

## 🎯 NEXT STEPS:
1. **Get actual private key** from Google Cloud Console
2. **Replace placeholder** in service-account-key.json
3. **Test email service** with the test script
4. **Verify all templates work** - Investor, customer, document

## 🚀 BENEFITS OF SERVICE ACCOUNT:
- **✅ No OAuth flow** - Direct API access
- **✅ No user limits** - Unlimited access
- **✅ Better for automation** - Designed for applications
- **✅ More reliable** - No token management
- **✅ Production ready** - No testing restrictions

---

**Follow these steps to get your actual service account key and your AI assistant will have reliable email capabilities!** 🔧✨

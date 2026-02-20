const ServiceAccountEmailService = require('./services/service-account-email');
require('dotenv').config();

async function testOAuthEmail() {
  console.log('🔧 Testing Stellar Logic AI OAuth 2.0 Email Service...\n');
  
  const emailService = new ServiceAccountEmailService();
  
  try {
    // Test 1: Generate auth URL
    console.log('1️⃣ Generating OAuth URL...');
    const authUrl = emailService.getAuthUrl();
    console.log('✅ OAuth URL generated:');
    console.log(authUrl);
    console.log('\n📋 Instructions:');
    console.log('1. Copy the URL above');
    console.log('2. Open in browser');
    console.log('3. Authorize the application');
    console.log('4. Copy the authorization code from the redirect');
    console.log('5. Use the code to get access token');
    
    console.log('\n🎉 OAuth 2.0 setup is ready!');
    console.log('\n📋 OAuth 2.0 Benefits:');
    console.log('   - Standard authentication method');
    console.log('   - User consent flow');
    console.log('   - Token refresh capability');
    console.log('   - Well-documented');
    console.log('   - Compatible with all systems');
    console.log('   - No OpenSSL issues');
    
  } catch (error) {
    console.error('❌ OAuth test failed:', error.message);
    console.log('\n🔧 Troubleshooting:');
    console.log('1. Check .env file for GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET');
    console.log('2. Verify redirect URI matches Google Cloud Console');
    console.log('3. Check Gmail API is enabled');
    console.log('4. Verify service account permissions');
  }
}

// Run test
testOAuthEmail().catch(console.error);

const ServiceAccountEmailService = require('./services/service-account-email');
require('dotenv').config();

async function exchangeCodeForTokens(code) {
  console.log('🔧 Exchanging authorization code for tokens...\n');
  
  const emailService = new ServiceAccountEmailService();
  
  try {
    const { tokens } = await emailService.auth.getToken(code);
    
    console.log('✅ Tokens received successfully!');
    console.log('📋 Access Token:', tokens.access_token);
    console.log('🔄 Refresh Token:', tokens.refresh_token);
    console.log('⏰ Expiry:', new Date(tokens.expiry_date).toLocaleString());
    
    // Set the tokens for future use
    emailService.auth.setCredentials(tokens);
    
    console.log('\n🎉 OAuth 2.0 authentication complete!');
    console.log('\n📋 Next Steps:');
    console.log('1. Test email sending functionality');
    console.log('2. Save refresh token for future use');
    console.log('3. Integrate with AI assistant');
    
    return tokens;
    
  } catch (error) {
    console.error('❌ Token exchange failed:', error.message);
    console.log('\n🔧 Troubleshooting:');
    console.log('1. Check authorization code is correct');
    console.log('2. Verify redirect URI matches');
    console.log('3. Check client ID and secret');
    console.log('4. Ensure Gmail API is enabled');
    throw error;
  }
}

// Get authorization code from command line argument
const authCode = process.argv[2];

if (!authCode) {
  console.log('❌ Please provide authorization code as argument:');
  console.log('Usage: node exchange-oauth-tokens.js YOUR_AUTHORIZATION_CODE');
  process.exit(1);
}

// Exchange code for tokens
exchangeCodeForTokens(authCode).catch(console.error);

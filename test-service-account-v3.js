const ServiceAccountEmailService = require('./services/service-account-email');
require('dotenv').config();

async function testServiceAccountEmailV3() {
  console.log('🔧 Testing Stellar Logic AI Service Account Email Service v3...\n');
  
  const emailService = new ServiceAccountEmailService();
  
  try {
    // Test 1: Send a test email
    console.log('1️⃣ Sending test email...');
    const testEmail = process.env.GMAIL_EMAIL;
    const result = await emailService.sendEmail(
      testEmail,
      'Stellar Logic AI - Service Account Test v3',
      `
        <h2>🚀 Service Account Email Test Successful!</h2>
        <p>Your Stellar Logic AI email service is working correctly with Service Account v3.</p>
        <div style="background-color: #f8f9fa; padding: 20px; margin: 20px 0; border-radius: 5px;">
          <h3>Service Account Benefits:</h3>
          <ul>
            <li>✅ No OAuth flow required</li>
            <li>✅ No redirect URI issues</li>
            <li>✅ No user limits</li>
            <li>✅ Direct API access</li>
            <li>✅ Better for automation</li>
            <li>✅ More reliable</li>
          </ul>
        </div>
        <p>Your AI assistant can now send professional emails without OAuth complications!</p>
        <p>Best regards,<br>
        <strong>Stellar Logic AI Team</strong></p>
      `
    );
    
    console.log('✅ Email sent successfully!');
    console.log(`📧 Message ID: ${result.id}`);
    console.log(`📨 Sent to: ${testEmail}`);
    
    // Test 2: Send investor update
    console.log('\n2️⃣ Testing investor update template...');
    await emailService.sendInvestorUpdate(
      testEmail,
      'Quarterly',
      'Test investor update v3: Your Stellar Logic AI investment is performing well with the new Service Account setup.'
    );
    console.log('✅ Investor update template sent!');
    
    // Test 3: Send customer follow-up
    console.log('\n3️⃣ Testing customer follow-up template...');
    await emailService.sendCustomerFollowUp(
      testEmail,
      'Demo Follow-up',
      'Test customer follow-up v3: Thank you for your interest in Stellar Logic AI v3. Here are the next steps for integration.'
    );
    console.log('✅ Customer follow-up template sent!');
    
    // Test 4: Send document share
    console.log('\n4️⃣ Testing document share template...');
    await emailService.sendDocumentShare(
      testEmail,
      'Service Account Setup Guide',
      './SERVICE_ACCOUNT_SETUP.md'
    );
    console.log('✅ Document share template sent!');
    
    console.log('\n🎉 All tests passed! Your AI assistant is ready with Service Account v3!');
    console.log('\n📋 Service Account Benefits:');
    console.log('   - No OAuth flow required');
    console.log('   - No redirect URI issues');
    console.log('   - No user limits');
    console.log('   - Direct API access');
    console.log('   - Better for automation');
    console.log('   - More reliable');
    console.log('   - Production ready');
    
  } catch (error) {
    console.error('❌ Service Account test failed:', error.message);
    console.log('\n🔧 Troubleshooting:');
    console.log('1. Make sure service-account-key-v3.json exists in ./credentials/');
    console.log('2. Check that Gmail API is enabled in Google Cloud Console');
    console.log('3. Verify service account has Gmail API scope');
    console.log('4. Check service account permissions');
    console.log('5. Verify project ID is correct');
  }
}

// Run the test
testServiceAccountEmailV3().catch(console.error);

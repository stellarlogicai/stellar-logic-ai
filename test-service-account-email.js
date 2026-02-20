const ServiceAccountEmailService = require('./services/service-account-email');
require('dotenv').config();

async function testServiceAccountEmail() {
  console.log('🔧 Testing Stellar Logic AI Service Account Email Service...\n');
  
  const emailService = new ServiceAccountEmailService();
  
  try {
    // Test 1: Send a test email
    console.log('1️⃣ Sending test email...');
    const testEmail = process.env.GMAIL_EMAIL;
    const result = await emailService.sendEmail(
      testEmail,
      'Stellar Logic AI - Service Account Test',
      `
        <h2>🚀 Service Account Email Test Successful!</h2>
        <p>Your Stellar Logic AI email service is working correctly with Service Account authentication.</p>
        <div style="background-color: #f8f9fa; padding: 20px; margin: 20px 0; border-radius: 5px;">
          <h3>Service Account Benefits:</h3>
          <ul>
            <li>✅ No OAuth flow required</li>
            <li>✅ No redirect URI issues</li>
            <li>✅ No user limits</li>
            <li>✅ Direct API access</li>
            <li>✅ Better for automation</li>
          </ul>
        </div>
        <p>Your AI assistant can now send professional emails without OAuth complications!</p>
        <p>Best regards,<br>Stellar Logic AI Team</p>
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
      'Test investor update: Your Stellar Logic AI investment is performing well with market expansion across 7 industries.'
    );
    console.log('✅ Investor update template sent!');
    
    // Test 3: Send customer follow-up
    console.log('\n3️⃣ Testing customer follow-up template...');
    await emailService.sendCustomerFollowUp(
      testEmail,
      'Demo Follow-up',
      'Test customer follow-up: Thank you for your interest in Stellar Logic AI. Here are the next steps for integration.'
    );
    console.log('✅ Customer follow-up template sent!');
    
    // Test 4: Send document share
    console.log('\n4️⃣ Testing document share template...');
    await emailService.sendDocumentShare(
      testEmail,
      'Automotive Plugin Overview',
      './products/plugins/automotive/AUTOMOTIVE_PLUGIN_OVERVIEW.md'
    );
    console.log('✅ Document share template sent!');
    
    console.log('\n🎉 All tests passed! Your AI assistant is ready with Service Account authentication!');
    console.log('\n📋 Benefits of Service Account:');
    console.log('   - No OAuth flow complications');
    console.log('   - No redirect URI management');
    console.log('   - No user limits');
    console.log('   - Better for AI automation');
    console.log('   - More reliable for production');
    
  } catch (error) {
    console.error('❌ Service Account test failed:', error.message);
    console.log('\n🔧 Troubleshooting:');
    console.log('1. Make sure service-account-key.json exists in ./credentials/');
    console.log('2. Check that Gmail API is enabled in Google Cloud Console');
    console.log('3. Verify service account has Gmail API scope');
    console.log('4. Check service account permissions');
  }
}

// Run the test
testServiceAccountEmail().catch(console.error);

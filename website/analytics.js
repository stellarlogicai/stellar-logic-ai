// Stellar Logic AI - Analytics Tracking Script
// This script tracks user interactions across all pages

(function() {
    'use strict';
    
    // Configuration
    const config = {
        siteName: 'Stellar Logic AI',
        siteUrl: 'https://stellarlogicai.netlify.app',
        trackingEndpoint: '/api/analytics',
        enableConsoleLogging: true,
        enableLocalStorage: true
    };
    
    // Analytics data structure
    let analyticsData = {
        sessionId: generateSessionId(),
        userId: getUserId(),
        pageViews: [],
        events: [],
        sessionStart: new Date().toISOString(),
        sessionEnd: null,
        userAgent: navigator.userAgent,
        referrer: document.referrer,
        location: window.location.href,
        screenResolution: `${screen.width}x${screen.height}`,
        language: navigator.language
    };
    
    // Generate unique session ID
    function generateSessionId() {
        return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }
    
    // Get or generate user ID
    function getUserId() {
        let userId = localStorage.getItem('stellar_logic_ai_user_id');
        if (!userId) {
            userId = 'user_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
            localStorage.setItem('stellar_logic_ai_user_id', userId);
        }
        return userId;
    }
    
    // Track page view
    function trackPageView() {
        const pageView = {
            url: window.location.href,
            title: document.title,
            timestamp: new Date().toISOString(),
            referrer: document.referrer
        };
        
        analyticsData.pageViews.push(pageView);
        
        if (config.enableConsoleLogging) {
            console.log('📊 Analytics - Page View:', pageView);
        }
        
        // Send to server
        sendToServer('pageview', pageView);
    }
    
    // Track custom events
    function trackEvent(eventName, properties = {}) {
        const event = {
            name: eventName,
            properties: properties,
            url: window.location.href,
            timestamp: new Date().toISOString()
        };
        
        analyticsData.events.push(event);
        
        if (config.enableConsoleLogging) {
            console.log('📊 Analytics - Event:', event);
        }
        
        // Send to server
        sendToServer('event', event);
    }
    
    // Track form submissions
    function trackFormSubmission(formId, formData) {
        trackEvent('form_submission', {
            form_id: formId,
            form_data: formData,
            page: window.location.pathname
        });
    }
    
    // Track button clicks
    function trackButtonClick(buttonId, buttonText, buttonUrl) {
        trackEvent('button_click', {
            button_id: buttonId,
            button_text: buttonText,
            button_url: buttonUrl,
            page: window.location.pathname
        });
    }
    
    // Track scroll depth
    function trackScrollDepth() {
        const scrollDepth = Math.round((window.scrollY + window.innerHeight) / document.body.scrollHeight * 100);
        
        if (scrollDepth >= 25 && !analyticsData.scroll25) {
            analyticsData.scroll25 = true;
            trackEvent('scroll_depth', { depth: 25 });
        }
        
        if (scrollDepth >= 50 && !analyticsData.scroll50) {
            analyticsData.scroll50 = true;
            trackEvent('scroll_depth', { depth: 50 });
        }
        
        if (scrollDepth >= 75 && !analyticsData.scroll75) {
            analyticsData.scroll75 = true;
            trackEvent('scroll_depth', { depth: 75 });
        }
        
        if (scrollDepth >= 90 && !analyticsData.scroll90) {
            analyticsData.scroll90 = true;
            trackEvent('scroll_depth', { depth: 90 });
        }
    }
    
    // Track time on page
    function trackTimeOnPage() {
        const timeOnPage = Math.round((Date.now() - pageStartTime) / 1000);
        trackEvent('time_on_page', { 
            seconds: timeOnPage,
            page: window.location.pathname 
        });
    }
    
    // Send data to server
    function sendToServer(type, data) {
        // For now, just log to console
        // In production, this would send to your analytics endpoint
        if (config.enableLocalStorage) {
            const storedData = JSON.parse(localStorage.getItem('stellar_logic_ai_analytics') || '[]');
            storedData.push({
                type: type,
                data: data,
                timestamp: new Date().toISOString()
            });
            
            // Keep only last 100 entries
            if (storedData.length > 100) {
                storedData.splice(0, storedData.length - 100);
            }
            
            localStorage.setItem('stellar_logic_ai_analytics', JSON.stringify(storedData));
        }
        
        // In production, uncomment to send to server:
        // fetch(config.trackingEndpoint, {
        //     method: 'POST',
        //     headers: {
        //         'Content-Type': 'application/json',
        //     },
        //     body: JSON.stringify({
        //         type: type,
        //         data: data,
        //         sessionId: analyticsData.sessionId,
        //         userId: analyticsData.userId
        //     })
        // }).catch(error => {
        //     console.error('Analytics tracking error:', error);
        // });
    }
    
    // Initialize tracking
    let pageStartTime = Date.now();
    
    // Track initial page view
    trackPageView();
    
    // Track scroll events
    let scrollTimeout;
    window.addEventListener('scroll', function() {
        clearTimeout(scrollTimeout);
        scrollTimeout = setTimeout(trackScrollDepth, 100);
    });
    
    // Track button clicks
    document.addEventListener('click', function(e) {
        const target = e.target;
        if (target.tagName === 'BUTTON' || target.tagName === 'A') {
            const buttonId = target.id || target.getAttribute('data-id') || 'unknown';
            const buttonText = target.textContent || target.innerText || 'unknown';
            const buttonUrl = target.href || 'unknown';
            trackButtonClick(buttonId, buttonText, buttonUrl);
        }
    });
    
    // Track form submissions
    document.addEventListener('submit', function(e) {
        const form = e.target;
        const formId = form.id || form.getAttribute('data-id') || 'unknown';
        const formData = new FormData(form);
        const formDataObj = {};
        for (let [key, value] of formData.entries()) {
            formDataObj[key] = value;
        }
        trackFormSubmission(formId, formDataObj);
    });
    
    // Track page unload
    window.addEventListener('beforeunload', function() {
        trackTimeOnPage();
        analyticsData.sessionEnd = new Date().toISOString();
        
        if (config.enableLocalStorage) {
            localStorage.setItem('stellar_logic_ai_session', JSON.stringify(analyticsData));
        }
    });
    
    // Track visibility changes
    document.addEventListener('visibilitychange', function() {
        if (document.hidden) {
            trackEvent('page_hidden', { page: window.location.pathname });
        } else {
            trackEvent('page_visible', { page: window.location.pathname });
        }
    });
    
    // Expose tracking functions globally
    window.StellarLogicAI = window.StellarLogicAI || {};
    window.StellarLogicAI.analytics = {
        trackEvent: trackEvent,
        trackPageView: trackPageView,
        trackFormSubmission: trackFormSubmission,
        trackButtonClick: trackButtonClick,
        getData: function() { return analyticsData; }
    };
    
    // Log initialization
    if (config.enableConsoleLogging) {
        console.log('📊 Stellar Logic AI Analytics initialized');
        console.log('📊 Session ID:', analyticsData.sessionId);
        console.log('📊 User ID:', analyticsData.userId);
    }
    
})();

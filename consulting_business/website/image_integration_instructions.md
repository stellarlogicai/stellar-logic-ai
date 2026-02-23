# Website Image Integration Instructions

## 🖼️ Images to Create with ChatGPT

### 📋 File Names and Dimensions:

1. **hero-banner.png** - 1200x400px
   - Prompt: "Create a professional AI security consulting website banner image. Modern tech aesthetic with purple and blue gradient colors. Abstract digital security visualization with glowing data streams and protective shield elements. Professional corporate style, clean design, 1200x400 pixels. Include subtle neural network patterns and security lock icons in background. Minimalist yet sophisticated."

2. **service-audit.png** - 400x300px
   - Prompt: "Professional AI security audit service illustration. Show a magnifying glass examining digital data streams with security shields protecting them. Modern tech aesthetic, purple and blue color scheme. Clean vector style, 400x300 pixels. Include subtle binary code patterns and lock icons. Professional corporate design suitable for consulting website."

3. **service-ml-development.png** - 400x300px
   - Prompt: "Custom machine learning development illustration. Show AI neural networks being built with glowing connections and data flowing through algorithms. Modern tech visualization, purple and blue gradient colors. Clean professional style, 400x300 pixels. Include subtle gear and brain icon elements. Corporate consulting aesthetic."

4. **service-strategy.png** - 400x300px
   - Prompt: "AI strategy consulting illustration. Show a strategic roadmap with AI elements, charts, and growth arrows. Professional business visualization, purple and blue color scheme. Clean modern design, 400x300 pixels. Include subtle chess piece and lightbulb icons representing strategy. Corporate consulting aesthetic."

5. **service-optimization.png** - 400x300px
   - Prompt: "AI performance optimization illustration. Show speed gauges, performance meters, and optimized data streams flowing efficiently. Modern tech visualization with speed indicators. Purple and blue gradient colors, clean design, 400x300 pixels. Include subtle lightning and speedometer icons. Professional corporate style."

6. **service-implementation.png** - 400x300px
   - Prompt: "AI implementation support illustration. Show a rocket launching with AI elements, representing successful deployment. Modern tech aesthetic with upward movement and success indicators. Purple and blue colors, clean design, 400x300 pixels. Include subtle cloud and gear icons. Professional consulting style."

7. **about-team.png** - 600x400px
   - Prompt: "Professional AI consulting team collaboration illustration. Show diverse professionals working together on AI security projects. Modern office setting with computers and digital displays. Purple and blue accent colors, clean corporate style, 600x400 pixels. Professional business atmosphere, realistic but stylized."

8. **trust-badges.png** - 600x100px
   - Prompt: "Create a set of 6 professional trust badge icons for AI consulting website. Each icon should be 100x100 pixels with transparent background. Icons needed: Security Shield, 24/7 Support Clock, Proven Results Star, Expert Team People, Cost-Effective Dollar Sign, Rapid Delivery Rocket. Use purple and blue gradient colors, modern flat design style."

## 🔧 HTML Integration Steps:

### Step 1: Add Hero Background Image
Replace the hero section CSS with:
```css
.hero {
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.9), rgba(118, 75, 162, 0.9)),
                url('images/hero-banner.png') center/cover;
    padding: 120px 0 60px;
    text-align: center;
    color: white;
    position: relative;
    overflow: hidden;
}
```

### Step 2: Add Service Images
Update each service card to include images:
```html
<div class="service-card">
    <div class="service-image">
        <img src="images/service-audit.png" alt="AI Security Audit">
    </div>
    <h3><span class="service-icon">🔍</span> AI Security Audit</h3>
    <!-- rest of content -->
</div>
```

### Step 3: Add Service Image CSS
Add to CSS:
```css
.service-image {
    margin-bottom: 1rem;
    text-align: center;
}

.service-image img {
    max-width: 100%;
    height: auto;
    border-radius: 10px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.1);
}
```

### Step 4: Add About Section Image
Update about section:
```html
<div class="about-content">
    <div class="about-image">
        <img src="images/about-team.png" alt="Stellar Logic AI Team">
    </div>
    <div class="about-text">
        <!-- existing text content -->
    </div>
</div>
```

### Step 5: Add About Image CSS
Add to CSS:
```css
.about-image {
    text-align: center;
    margin-bottom: 2rem;
}

.about-image img {
    max-width: 100%;
    height: auto;
    border-radius: 15px;
    box-shadow: 0 5px 15px rgba(0,0,0,0.1);
}
```

## 📁 Save Location:
Save all images to: `c:\Users\merce\Documents\helm-ai\consulting_business\website\images\`

## 🎯 Final Result:
Your website will have professional, custom images that match your brand and impress potential clients!

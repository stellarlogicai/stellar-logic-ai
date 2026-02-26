# Background Image Replacement Plan

## Current Issues
- Background images were not properly cut from master sheet
- Images don't center properly
- Images may not be the right dimensions or quality
- Need professional, consistent hero backgrounds

## Required Background Images

### 1. Homepage Hero Background
**File:** `images/hero-banner.png`
**Dimensions:** 1920x1080 (16:9 ratio)
**Theme:** AI security, technology, abstract digital protection
**Style:** Dark tech background with subtle AI/cybersecurity elements
**Opacity:** Will be set to 0.15, so design should work well at low opacity

### 2. About Us Hero Background
**File:** `images/about-hero-bg.jpg`
**Dimensions:** 1920x1080 (16:9 ratio)
**Theme:** Professional team, company culture, office environment
**Style:** Professional business setting with tech elements
**Opacity:** Will be set to 0.15

### 3. Case Studies Hero Background
**File:** `images/case-studies-hero-bg.jpg`
**Dimensions:** 1920x1080 (16:9 ratio)
**Theme:** Success, results, professional consulting
**Style:** Abstract success patterns, charts, professional environment
**Opacity:** Will be set to 0.15

### 4. Consulting Services Hero Background
**File:** `images/consulting-hero-bg.jpg`
**Dimensions:** 1920x1080 (16:9 ratio)
**Theme:** AI consulting, professional services
**Style:** Business consulting, AI technology, professional environment
**Opacity:** Will be set to 0.15

### 5. Pricing Hero Background
**File:** `images/pricing-hero-bg.jpg`
**Dimensions:** 1920x1080 (16:9 ratio)
**Theme:** Value, investment, professional pricing
**Style:** Professional business/finance theme with tech elements
**Opacity:** Will be set to 0.15

### 6. Client Portal Hero Background
**File:** `images/portal-hero-bg.jpg`
**Dimensions:** 1920x1080 (16:9 ratio)
**Theme:** Security, access, client dashboard
**Style:** Secure login, dashboard interface, security elements
**Opacity:** Will be set to 0.15

## Design Requirements

### Technical Specifications
- **Format:** PNG for homepage, JPG for others (as currently used)
- **Resolution:** 1920x1080 minimum (can be larger for high-DPI displays)
- **Color Profile:** sRGB
- **Compression:** Optimized for web (balance quality and file size)

### Visual Style Guidelines
- **Dark theme:** All images should work well with dark overlay
- **Subtle design:** Since opacity is 0.15, images should be subtle
- **Professional:** No cheesy stock photos or overly dramatic elements
- **Consistent:** All images should have a cohesive visual style
- **Tech-focused:** Should reflect AI/security/technology theme

### Color Palette
- **Primary:** Dark blues, purples, blacks
- **Accent:** Subtle tech blues, purples
- **Avoid:** Bright colors that might clash with the gradient overlay

## Image Generation Prompts

### Homepage Hero Background
```
Professional abstract AI security background, dark technology theme with subtle digital protection elements, neural network patterns, cybersecurity shield motifs, deep blue and purple color scheme, minimalist design, high resolution, suitable for website hero section with 15% opacity overlay
```

### About Us Hero Background
```
Professional modern office environment with technology elements, collaborative workspace, subtle AI/security theme, clean and professional atmosphere, deep blue and purple tones, minimalist corporate style, suitable for website hero section with 15% opacity overlay
```

### Case Studies Hero Background
```
Abstract business success patterns, subtle data visualization elements, professional consulting theme, growth charts and success metrics visualized abstractly, deep blue and purple color scheme, corporate professional style, suitable for website hero section with 15% opacity overlay
```

### Consulting Services Hero Background
```
Professional business consulting environment, subtle AI technology integration, corporate meeting setting with modern tech elements, deep blue and purple tones, professional services theme, clean minimalist design, suitable for website hero section with 15% opacity overlay
```

### Pricing Hero Background
```
Professional finance and technology theme, abstract value and investment patterns, subtle pricing and value visualization, corporate financial environment with tech elements, deep blue and purple color scheme, professional business style, suitable for website hero section with 15% opacity overlay
```

### Client Portal Hero Background
```
Secure dashboard interface theme, abstract login and security elements, subtle data protection patterns, professional cybersecurity interface design, deep blue and purple tones, modern security UI elements, suitable for website hero section with 15% opacity overlay
```

## Implementation Steps

1. **Generate New Images**
   - Use the prompts above to generate new background images
   - Ensure all images are 1920x1080 minimum resolution
   - Test each image at 15% opacity with the gradient overlay

2. **Replace Current Images**
   - Replace each file in the `images/` folder
   - Keep the same filenames to avoid updating HTML
   - Test each page to ensure proper display

3. **Quality Check**
   - Verify images center properly with new CSS settings
   - Check that images look good at 15% opacity
   - Ensure consistent visual style across all pages

4. **Performance Optimization**
   - Compress images for web without losing quality
   - Check file sizes (aim for under 500KB per image)
   - Test loading speeds

## Current CSS Settings (Already Fixed)
```css
background-size: cover;
background-position: center center;
background-repeat: no-repeat;
background-attachment: fixed;
opacity: 0.15;
```

## Notes
- The CSS fixes are already in place
- Only need to replace the actual image files
- New images should be designed specifically for 15% opacity display
- Consistency across all pages is crucial for professional appearance

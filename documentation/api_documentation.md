# STELLOR LOGIC AI - API DOCUMENTATION

## 1. AUTHENTICATION

### API Keys
- Contact support for API key
- Include key in request headers
- Key rotation every 90 days

### Headers
```
Authorization: Bearer YOUR_API_KEY
Content-Type: application/json
```

## 2. ENDPOINTS

### Health Check
```
GET /health
Response: {
  "status": "healthy",
  "timestamp": "2024-01-01T00:00:00Z",
  "uptime": "2 days, 3 hours",
  "version": "1.0.0"
}
```

### Cheat Detection
```
POST /api/detect
Request: {
  "frame_id": "frame_123",
  "user_id": "user_456",
  "timestamp": "2024-01-01T00:00:00Z",
  "frame_data": "base64_encoded_image"
}
Response: {
  "success": true,
  "result": {
    "cheat_detected": false,
    "confidence": 0.15,
    "cheat_types": [],
    "processing_time_ms": 45.2
  }
}
```

### Batch Detection
```
POST /api/batch_detect
Request: {
  "frames": [
    {"frame_id": "frame_1", "frame_data": "..."},
    {"frame_id": "frame_2", "frame_data": "..."}
  ]
}
Response: {
  "success": true,
  "results": [...],
  "total_frames": 2,
  "processing_time_ms": 89.3
}
```

### User Risk Assessment
```
POST /api/user_risk
Request: {
  "user_id": "user_456",
  "timeframe": "7d"
}
Response: {
  "success": true,
  "risk_assessment": {
    "risk_score": 0.25,
    "risk_level": "LOW",
    "factors": [...],
    "recommendation": "Continue monitoring"
  }
}
```

## 3. ERROR RESPONSES

### Standard Error Format
```json
{
  "success": false,
  "error": "Error description",
  "error_code": "ERROR_CODE",
  "timestamp": "2024-01-01T00:00:00Z"
}
```

### Common Error Codes
- `INVALID_API_KEY`: Authentication failed
- `RATE_LIMIT_EXCEEDED`: Too many requests
- `INVALID_FRAME_DATA`: Malformed image data
- `PROCESSING_ERROR`: Internal processing error
- `SERVICE_UNAVAILABLE`: System maintenance

## 4. RATE LIMITING

### Limits
- 100 requests per minute
- 10,000 requests per hour
- 100,000 requests per day

### Headers
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1640995200
```

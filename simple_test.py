#!/usr/bin/env python3
"""
SUPER SIMPLE Gaming Test - No Hanging Guaranteed
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

print("🎮 SUPER SIMPLE GAMING TEST")
print("=" * 40)

# Generate simple data
print("📊 Generating data...")
np.random.seed(42)
n_samples = 1000
X = np.random.randn(n_samples, 5)
y = (X[:, 0] > 0.5).astype(int)

print(f"✅ Generated {n_samples} samples")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
print("🤖 Training model...")
start_time = time.time()

model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X_train, y_train)

training_time = time.time() - start_time
print(f"⏱️ Training completed in {training_time:.2f} seconds")

# Test model
print("🧪 Testing model...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"🎯 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print("🎉 TEST COMPLETED SUCCESSFULLY!")
print("✅ No hanging - working perfectly!")

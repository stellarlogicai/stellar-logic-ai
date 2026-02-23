#!/usr/bin/env python3
"""
SIMPLE EDGE PROCESSING TEST
Quick test for sub-5ms inference performance
"""

import os
import time
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms
from datetime import datetime

class SimpleEdgeModel(nn.Module):
    """Simple edge-optimized model"""
    
    def __init__(self):
        super().__init__()
        
        # Ultra-lightweight architecture
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def test_edge_performance():
    """Test edge processing performance"""
    print("⚡ STELLOR LOGIC AI - SIMPLE EDGE PERFORMANCE TEST")
    print("=" * 55)
    print("Testing sub-5ms inference capabilities")
    print("=" * 55)
    
    # Create model
    model = SimpleEdgeModel()
    model.eval()
    
    # Create optimized transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),  # Smaller for speed
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Test data
    test_images = []
    for i in range(100):
        # Create test image
        image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        
        # Add some patterns
        cv2.circle(image, (112, 112), 2, (255, 255, 255), -1)
        if i % 4 == 0:  # Add cheat patterns
            cv2.rectangle(image, (50, 50), (80, 80), (0, 255, 0), 2)
        
        test_images.append(image)
    
    # Preprocessing performance test
    print("\n🔧 Testing preprocessing performance...")
    preprocessing_times = []
    
    for image in test_images[:50]:
        start_time = time.perf_counter()
        processed = transform(image)
        end_time = time.perf_counter()
        preprocessing_times.append((end_time - start_time) * 1000)
    
    avg_preprocessing = np.mean(preprocessing_times)
    print(f"   ⚡ Preprocessing: {avg_preprocessing:.3f}ms average")
    
    # Inference performance test
    print("\n🤖 Testing inference performance...")
    inference_times = []
    
    # Warm up
    warmup_input = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        for _ in range(10):
            _ = model(warmup_input)
    
    # Test inference
    for image in test_images:
        processed = transform(image)
        input_tensor = processed.unsqueeze(0)
        
        start_time = time.perf_counter()
        with torch.no_grad():
            output = model(input_tensor)
        end_time = time.perf_counter()
        
        inference_time = (end_time - start_time) * 1000
        inference_times.append(inference_time)
    
    # Calculate statistics
    avg_inference = np.mean(inference_times)
    min_inference = np.min(inference_times)
    max_inference = np.max(inference_times)
    std_inference = np.std(inference_times)
    
    print(f"   📊 Average: {avg_inference:.3f}ms")
    print(f"   ⚡ Min: {min_inference:.3f}ms")
    print(f"   🐌 Max: {max_inference:.3f}ms")
    print(f"   📈 Std: {std_inference:.3f}ms")
    
    # Check target achievement
    target_achieved = avg_inference < 5.0
    print(f"   🎯 Target (<5ms): {'✅ ACHIEVED' if target_achieved else '❌ NOT ACHIEVED'}")
    
    # End-to-end performance test
    print("\n🚀 Testing end-to-end performance...")
    end_to_end_times = []
    
    for image in test_images[:30]:
        start_time = time.perf_counter()
        
        # Preprocess
        processed = transform(image)
        input_tensor = processed.unsqueeze(0)
        
        # Inference
        with torch.no_grad():
            output = model(input_tensor)
        
        # Post-process
        probabilities = torch.softmax(output, dim=1)
        cheat_prob = probabilities[0][1].item()
        cheat_detected = cheat_prob >= 0.5
        
        end_time = time.perf_counter()
        end_to_end_time = (end_time - start_time) * 1000
        end_to_end_times.append(end_to_end_time)
    
    avg_end_to_end = np.mean(end_to_end_times)
    throughput = 30 / (sum(end_to_end_times) / 1000)
    
    print(f"   ⚡ End-to-end: {avg_end_to_end:.3f}ms average")
    print(f"   🚀 Throughput: {throughput:.1f} FPS")
    
    # Model size
    model_size = sum(p.numel() for p in model.parameters()) / 1000
    print(f"   💾 Model size: {model_size:.1f}K parameters")
    
    # Overall assessment
    print(f"\n🎯 EDGE PERFORMANCE SUMMARY:")
    print(f"   ⚡ Inference: {avg_inference:.3f}ms ({'✅' if target_achieved else '❌'})")
    print(f"   🔧 Preprocessing: {avg_preprocessing:.3f}ms")
    print(f"   🚀 End-to-end: {avg_end_to_end:.3f}ms")
    print(f"   📈 Throughput: {throughput:.1f} FPS")
    print(f"   💾 Model size: {model_size:.1f}K parameters")
    
    # Check all targets
    targets_met = {
        'sub_5ms_inference': target_achieved,
        'fast_preprocessing': avg_preprocessing < 2.0,
        'high_throughput': throughput >= 30,
        'small_model': model_size < 100
    }
    
    all_targets_met = all(targets_met.values())
    
    print(f"\n🎉 TARGETS ACHIEVED:")
    for target, met in targets_met.items():
        status = "✅" if met else "❌"
        print(f"   {status} {target.replace('_', ' ').title()}")
    
    print(f"\n🏆 OVERALL: {'✅ ALL TARGETS MET' if all_targets_met else '⚠️ SOME TARGETS MISSED'}")
    
    # Save simple model
    edge_models_dir = "c:/Users/merce/Documents/helm-ai/edge_models"
    os.makedirs(edge_models_dir, exist_ok=True)
    
    model_path = os.path.join(edge_models_dir, "simple_edge_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"\n💾 Simple edge model saved: {model_path}")
    
    return all_targets_met, avg_inference, throughput

if __name__ == "__main__":
    try:
        success, avg_time, fps = test_edge_performance()
        
        if success:
            print(f"\n🎉 EDGE OPTIMIZATION SUCCESSFUL!")
            print(f"✅ Sub-5ms inference achieved: {avg_time:.3f}ms")
            print(f"✅ High throughput: {fps:.1f} FPS")
            print(f"✅ Ready for edge deployment")
        else:
            print(f"\n⚠️ EDGE OPTIMIZATION PARTIALLY SUCCESSFUL")
            print(f"💡 Average inference: {avg_time:.3f}ms")
            print(f"💡 Throughput: {fps:.1f} FPS")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()

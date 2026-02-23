#!/usr/bin/env python3
"""
CHECK TRAINING PROGRESS
Monitor improved cheat detection model training
"""

import os
import json
import time
from datetime import datetime

def check_training_progress():
    """Check if improved models have been created"""
    models_dir = "c:/Users/merce/Documents/helm-ai/models"
    
    print("🔍 CHECKING IMPROVED CHEAT DETECTION MODELS")
    print("=" * 50)
    
    # Check for improved models
    improved_models = [
        "improved_general_cheat_detection_model.pth",
        "improved_esp_detection_model.pth", 
        "improved_aimbot_detection_model.pth",
        "improved_wallhack_detection_model.pth"
    ]
    
    models_found = []
    for model_file in improved_models:
        model_path = os.path.join(models_dir, model_file)
        if os.path.exists(model_path):
            # Get file size and modification time
            size = os.path.getsize(model_path)
            mtime = datetime.fromtimestamp(os.path.getmtime(model_path))
            models_found.append({
                'file': model_file,
                'size_mb': size / (1024*1024),
                'modified': mtime
            })
            print(f"✅ {model_file}")
            print(f"   Size: {size/(1024*1024):.2f} MB")
            print(f"   Modified: {mtime}")
        else:
            print(f"❌ {model_file} - NOT FOUND")
    
    # Check for training summary
    summary_path = os.path.join(models_dir, "improved_training_summary.json")
    if os.path.exists(summary_path):
        print(f"\n📊 TRAINING SUMMARY FOUND:")
        try:
            with open(summary_path, 'r') as f:
                summary = json.load(f)
            
            for model_name, result in summary.items():
                accuracy = result.get('final_accuracy', 0)
                f1_score = result.get('final_f1', 0)
                print(f"   🤖 {model_name}: {accuracy:.4f} ({accuracy*100:.2f}%) accuracy, F1: {f1_score:.4f}")
                
                if accuracy >= 0.90:
                    print(f"      🎉 TARGET ACHIEVED!")
                elif accuracy >= 0.80:
                    print(f"      ✅ GOOD PERFORMANCE")
                else:
                    print(f"      ⚠️ NEEDS IMPROVEMENT")
        
        except Exception as e:
            print(f"   ❌ Error reading summary: {str(e)}")
    else:
        print(f"\n📊 TRAINING SUMMARY: NOT FOUND")
        print(f"   Training may still be in progress...")
    
    # Overall status
    print(f"\n📈 OVERALL STATUS:")
    print(f"   Models Found: {len(models_found)}/{len(improved_models)}")
    
    if len(models_found) == len(improved_models):
        print(f"   🎉 ALL MODELS COMPLETED!")
        
        # Check if all models meet 90% target
        if os.path.exists(summary_path):
            try:
                with open(summary_path, 'r') as f:
                    summary = json.load(f)
                
                models_90_plus = sum(1 for result in summary.values() 
                                   if result.get('final_accuracy', 0) >= 0.90)
                
                if models_90_plus == len(improved_models):
                    print(f"   🏆 ALL MODELS ACHIEVED 90%+ TARGET!")
                else:
                    print(f"   📊 {models_90_plus}/{len(improved_models)} models achieved 90%+ target")
            except:
                pass
    else:
        print(f"   ⏳ TRAINING IN PROGRESS...")
        print(f"   🚀 Please wait for completion")
    
    return models_found

if __name__ == "__main__":
    check_training_progress()

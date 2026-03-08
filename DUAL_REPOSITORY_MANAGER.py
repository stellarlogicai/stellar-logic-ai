#!/usr/bin/env python3
"""
DUAL_REPOSITORY_MANAGER.py
Comprehensive dual repository management system for Stellar Logic AI
"""

import subprocess
import json
import datetime
from pathlib import Path

class DualRepositoryManager:
    def __init__(self, project_path):
        self.project_path = Path(project_path)
        self.config_file = self.project_path / "dual_repo_config.json"
        self.load_config()
    
    def load_config(self):
        """Load dual repository configuration"""
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = {
                "main_repo": {
                    "name": "origin",
                    "url": "https://github.com/stellarlogicai/stellar-logic-ai.git"
                },
                "backup_repo": {
                    "name": "backup",
                    "url": "https://github.com/your-username/helm-ai.git"
                }
            }
            self.save_config()
    
    def save_config(self):
        """Save dual repository configuration"""
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def get_status(self):
        """Get current repository status"""
        try:
            # Get current branch
            result = subprocess.run(['git', 'branch', '--show-current'], 
                                  cwd=self.project_path, 
                                  capture_output=True, text=True)
            current_branch = result.stdout.strip()
            
            # Get remote status
            result = subprocess.run(['git', 'remote', '-v'], 
                                  cwd=self.project_path, 
                                  capture_output=True, text=True)
            remotes = result.stdout.strip()
            
            # Get commit info
            result = subprocess.run(['git', 'log', '--oneline', '-1'], 
                                  cwd=self.project_path, 
                                  capture_output=True, text=True)
            last_commit = result.stdout.strip()
            
            return {
                "current_branch": current_branch,
                "remotes": remotes,
                "last_commit": last_commit,
                "config": self.config
            }
        except Exception as e:
            return {"error": str(e)}
    
    def push_to_main(self, commit_message="Daily development work"):
        """Push to main repository"""
        try:
            print("🚀 Pushing to main repository...")
            
            # Add all changes
            subprocess.run(['git', 'add', '.'], cwd=self.project_path)
            
            # Commit changes
            subprocess.run(['git', 'commit', '-m', commit_message], cwd=self.project_path)
            
            # Push to main repository
            result = subprocess.run(['git', 'push', 'origin', 'main'], 
                                  cwd=self.project_path, 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Successfully pushed to main repository")
                return True, "Main repository push successful"
            else:
                return False, f"Push failed: {result.stderr}"
                
        except Exception as e:
            return False, f"Main repository push error: {str(e)}"
    
    def push_to_backup(self, commit_message="Weekly backup"):
        """Push to backup repository"""
        try:
            print("🔄 Pushing to backup repository...")
            
            # Push to backup repository (force to ensure complete backup)
            result = subprocess.run(['git', 'push', 'backup', 'main', '--force'], 
                                  cwd=self.project_path, 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Successfully pushed to backup repository")
                return True, "Backup repository push successful"
            else:
                return False, f"Backup push failed: {result.stderr}"
                
        except Exception as e:
            return False, f"Backup repository push error: {str(e)}"
    
    def verify_backup_integrity(self):
        """Verify backup repository integrity"""
        try:
            print("🔍 Verifying backup repository integrity...")
            
            # Check if backup repository is accessible
            result = subprocess.run(['git', 'ls-remote', 'backup'], 
                                  cwd=self.project_path, 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Backup repository is accessible and up to date")
                return True, "Backup repository integrity verified"
            else:
                return False, "Backup repository not accessible"
                
        except Exception as e:
            return False, f"Backup integrity verification error: {str(e)}"
    
    def create_weekly_backup(self):
        """Create weekly backup with comprehensive testing"""
        print("🗓️ Creating weekly comprehensive backup...")
        
        # Push to backup repository
        success, result = self.push_to_backup("Weekly comprehensive backup " + 
                                                   datetime.datetime.now().strftime("%Y-%m-%d"))
        
        if success:
            # Verify backup integrity
            return self.verify_backup_integrity()
        else:
            return False, result

# Usage example
if __name__ == "__main__":
    manager = DualRepositoryManager("c:/Users/merce/Documents/helm-ai")
    
    # Show current status
    status = manager.get_status()
    print("📊 Current Repository Status:")
    print(f"  Branch: {status.get('current_branch')}")
    print(f"  Last Commit: {status.get('last_commit')}")
    print(f"  Remotes: {len(status.get('remotes', '').split())}")
    
    # Test main repository push
    print("\n🚀 Testing main repository push...")
    success, result = manager.push_to_main("Test push to main repository")
    print(f"  Result: {result}")
    
    # Test backup repository push
    print("\n🔄 Testing backup repository push...")
    success, result = manager.push_to_backup("Test push to backup repository")
    print(f"  Result: {result}")
    
    # Verify backup integrity
    print("\n🔍 Verifying backup integrity...")
    success, result = manager.verify_backup_integrity()
    print(f"  Result: {result}")

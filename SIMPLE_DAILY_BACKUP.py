#!/usr/bin/env python3
"""
SIMPLE_DAILY_BACKUP.py
Simple and reliable daily backup system for Stellar Logic AI
"""

import os
import shutil
import json
import hashlib
import datetime
from pathlib import Path

class SimpleBackupSystem:
    def __init__(self, project_path, backup_base_path):
        self.project_path = Path(project_path)
        self.backup_base_path = Path(backup_base_path)
        self.daily_backup_path = self.backup_base_path / "daily"
        
        # Create backup directory
        self.daily_backup_path.mkdir(parents=True, exist_ok=True)
    
    def create_daily_backup(self):
        """Create daily backup with validation"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"stellar_logic_ai_backup_{timestamp}"
        backup_path = self.daily_backup_path / backup_name
        
        print(f"🚀 Creating daily backup: {backup_name}")
        
        try:
            # Create backup directory
            backup_path.mkdir(exist_ok=True)
            
            # Copy important files
            files_copied = self.copy_important_files(backup_path)
            
            # Create backup report
            self.create_backup_report(backup_path, timestamp, files_copied)
            
            print(f"✅ Daily backup completed successfully: {backup_name}")
            print(f"📁 Copied {files_copied} files")
            return True, backup_name
            
        except Exception as e:
            print(f"❌ Backup failed: {str(e)}")
            # Cleanup failed backup
            if backup_path.exists():
                shutil.rmtree(backup_path)
            return False, f"Backup creation failed: {str(e)}"
    
    def copy_important_files(self, backup_path):
        """Copy important project files"""
        print("📁 Copying important files...")
        
        # Important directories to backup
        important_dirs = [
            "src",
            "website", 
            "consulting_business",
            "branding",
            "production"
        ]
        
        # Important files to backup
        important_files = [
            "README.md",
            "index.html",
            "contact.html",
            "about.html",
            "products.html",
            "pricing.html",
            "careers.html",
            "compliance.html",
            "support.html",
            "terms.html",
            "privacy.html",
            "favicon.ico",
            "favicon_16x16.png",
            "favicon_32x32.png",
            "favicon_64x64.png",
            "Stellar_Logic_AI_Logo.png",
            "Stellar_Logic_AI_Logo.svg",
            "Helm_AI_Logo.png",
            "helm-ai-logo.png"
        ]
        
        # Exclude patterns
        exclude_patterns = {
            '__pycache__', '.git', 'node_modules', '.venv', 
            'env', '.pytest_cache', '.vscode', '.DS_Store',
            'Thumbs.db', '*.log', 'logs/', 'tmp/', 'temp/',
            '.env', '*.key', '*.pem', '*.crt', '*.p12'
        }
        
        files_copied = 0
        
        # Copy directories
        for dir_name in important_dirs:
            src_dir = self.project_path / dir_name
            if src_dir.exists() and src_dir.is_dir():
                dst_dir = backup_path / dir_name
                shutil.copytree(src_dir, dst_dir, ignore=shutil.ignore_patterns(
                    patterns=['*__pycache__', '*.pyc', '*.log', '*.tmp', '.DS_Store', 'Thumbs.db']
                ))
                # Count files in directory
                for file_path in dst_dir.rglob('*'):
                    if file_path.is_file():
                        files_copied += 1
                print(f"✅ Copied directory: {dir_name}")
        
        # Copy individual files
        for file_name in important_files:
            src_file = self.project_path / file_name
            if src_file.exists() and src_file.is_file():
                dst_file = backup_path / file_name
                dst_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_file, dst_file)
                files_copied += 1
                print(f"✅ Copied file: {file_name}")
        
        return files_copied
    
    def create_backup_report(self, backup_path, timestamp, files_copied):
        """Create backup report"""
        report = {
            "backup_name": backup_path.name,
            "timestamp": timestamp,
            "files_copied": files_copied,
            "backup_status": "SUCCESS",
            "backup_type": "SIMPLE_DAILY_BACKUP"
        }
        
        report_file = backup_path / "backup_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📋 Backup report created: {report_file}")
    
    def verify_backup(self, backup_path):
        """Verify backup integrity"""
        print("🔍 Verifying backup integrity...")
        
        if not backup_path.exists():
            return False, "Backup directory does not exist"
        
        report_file = backup_path / "backup_report.json"
        if not report_file.exists():
            return False, "Backup report not found"
        
        try:
            with open(report_file, 'r') as f:
                report = json.load(f)
            
            if report.get("backup_status") != "SUCCESS":
                return False, "Backup status not successful"
            
            print("✅ Backup verification passed")
            return True, "Backup verification successful"
            
        except Exception as e:
            return False, f"Backup verification failed: {str(e)}"
    
    def list_backups(self):
        """List all available backups"""
        backups = []
        for backup_dir in self.daily_backup_path.iterdir():
            if backup_dir.is_dir() and backup_dir.name.startswith("stellar_logic_ai_backup_"):
                backup_info = {
                    "name": backup_dir.name,
                    "path": str(backup_dir),
                    "created": datetime.datetime.fromtimestamp(backup_dir.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                }
                backups.append(backup_info)
        
        return sorted(backups, key=lambda x: x["created"], reverse=True)

# Test the backup system
if __name__ == "__main__":
    project_path = "c:/Users/merce/Documents/helm-ai"
    backup_base_path = "c:/Users/merce/Documents/helm-ai/backups"
    
    backup_system = SimpleBackupSystem(project_path, backup_base_path)
    
    # Create backup
    success, result = backup_system.create_daily_backup()
    
    if success:
        print(f"✅ Daily backup successful: {result}")
        
        # Verify backup
        verify_success, verify_result = backup_system.verify_backup(backup_system.daily_backup_path / result)
        if verify_success:
            print(f"✅ Backup verification successful: {verify_result}")
        else:
            print(f"❌ Backup verification failed: {verify_result}")
    else:
        print(f"❌ Daily backup failed: {result}")
    
    # List available backups
    print("\n📋 Available backups:")
    backups = backup_system.list_backups()
    for backup in backups[:5]:  # Show last 5 backups
        print(f"  📁 {backup['name']} (Created: {backup['created']})")

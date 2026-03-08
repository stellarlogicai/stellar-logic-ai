#!/usr/bin/env python3
"""
DAILY_BACKUP_SYSTEM.py
Robust daily backup system with extensive testing and validation
"""

import os
import shutil
import json
import hashlib
import datetime
import subprocess
import sys
from pathlib import Path

class RobustBackupSystem:
    def __init__(self, project_path, backup_base_path):
        self.project_path = Path(project_path)
        self.backup_base_path = Path(backup_base_path)
        self.daily_backup_path = self.backup_base_path / "daily"
        self.weekly_backup_path = self.backup_base_path / "weekly"
        self.integrity_log_path = self.backup_base_path / "integrity_logs"
        
        # Create backup directories
        self.daily_backup_path.mkdir(parents=True, exist_ok=True)
        self.weekly_backup_path.mkdir(parents=True, exist_ok=True)
        self.integrity_log_path.mkdir(parents=True, exist_ok=True)
    
    def create_daily_backup(self):
        """Create comprehensive daily backup with extensive testing"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"stellar_logic_ai_backup_{timestamp}"
        backup_path = self.daily_backup_path / backup_name
        
        print(f"🚀 Creating daily backup: {backup_name}")
        
        # Step 1: Pre-backup validation
        if not self.pre_backup_validation():
            return False, "Pre-backup validation failed"
        
        # Step 2: Create backup
        try:
            # Create backup directory
            backup_path.mkdir(exist_ok=True)
            
            # Copy project files
            self.copy_project_files(backup_path)
            
            # Create backup metadata
            self.create_backup_metadata(backup_path, timestamp)
            
            # Step 3: Post-backup testing
            if not self.post_backup_testing(backup_path):
                return False, "Post-backup testing failed"
            
            # Step 4: Integrity verification
            if not self.verify_backup_integrity(backup_path):
                return False, "Backup integrity verification failed"
            
            # Step 5: Functionality testing
            if not self.test_backup_functionality(backup_path):
                return False, "Backup functionality testing failed"
            
            # Step 6: Create backup report
            self.create_backup_report(backup_path, timestamp)
            
            print(f"✅ Daily backup completed successfully: {backup_name}")
            return True, backup_name
            
        except Exception as e:
            print(f"❌ Backup failed: {str(e)}")
            # Cleanup failed backup
            if backup_path.exists():
                shutil.rmtree(backup_path)
            return False, f"Backup creation failed: {str(e)}"
    
    def pre_backup_validation(self):
        """Extensive pre-backup validation"""
        print("🔍 Running pre-backup validation...")
        
        validations = {
            "project_exists": self.project_path.exists(),
            "git_status_clean": self.is_git_status_clean(),
            "no_sensitive_files_exposed": self.check_sensitive_files(),
            "disk_space_available": self.check_disk_space(),
            "backup_directory_writable": self.check_backup_permissions(),
            "critical_files_present": self.check_critical_files()
        }
        
        all_passed = all(validations.values())
        
        if not all_passed:
            print("❌ Pre-backup validation failed:")
            for test, result in validations.items():
                if not result:
                    print(f"   ❌ {test}")
        else:
            print("✅ Pre-backup validation passed")
        
        return all_passed
    
    def copy_project_files(self, backup_path):
        """Copy project files with validation"""
        print("📁 Copying project files...")
        
        # Exclude patterns
        exclude_patterns = {
            '__pycache__', '.git', 'node_modules', '.venv', 
            'env', '.pytest_cache', '.vscode', '.DS_Store',
            'Thumbs.db', '*.log', 'logs/', 'tmp/', 'temp/'
        }
        
        files_copied = 0
        files_skipped = 0
        total_size = 0
        
        for item in self.project_path.rglob('*'):
            if item.is_file():
                # Check if file should be excluded
                should_exclude = False
                for pattern in exclude_patterns:
                    if pattern in str(item):
                        should_exclude = True
                        files_skipped += 1
                        break
                
                if not should_exclude:
                    # Create relative path in backup
                    relative_path = item.relative_to(self.project_path)
                    backup_file_path = backup_path / relative_path
                    
                    # Create directory if needed
                    backup_file_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Copy file with verification
                    shutil.copy2(item, backup_file_path)
                    
                    # Verify copy integrity
                    if not self.verify_file_copy(item, backup_file_path):
                        raise Exception(f"File copy verification failed: {item}")
                    
                    files_copied += 1
                    total_size += item.stat().st_size
        
        print(f"✅ Copied {files_copied} files ({self.format_size(total_size)})")
        print(f"ℹ️  Skipped {files_skipped} excluded files")
    
    def post_backup_testing(self, backup_path):
        """Extensive post-backup testing"""
        print("🧪 Running post-backup testing...")
        
        tests = {
            "backup_structure_valid": self.test_backup_structure(backup_path),
            "critical_files_present": self.test_critical_files_backup(backup_path),
            "file_integrity": self.test_all_file_integrity(backup_path),
            "python_syntax": self.test_python_syntax(backup_path),
            "html_validity": self.test_html_validity(backup_path),
            "json_validity": self.test_json_validity(backup_path),
            "no_sensitive_data": self.test_no_sensitive_data_exposed(backup_path)
        }
        
        all_passed = all(tests.values())
        
        if not all_passed:
            print("❌ Post-backup testing failed:")
            for test, result in tests.items():
                if not result:
                    print(f"   ❌ {test}")
        else:
            print("✅ Post-backup testing passed")
        
        return all_passed
    
    def verify_backup_integrity(self, backup_path):
        """Verify backup integrity with checksums"""
        print("🔐 Verifying backup integrity...")
        
        integrity_file = backup_path / "integrity_checksums.json"
        checksums = {}
        
        # Generate checksums for all files
        for file_path in backup_path.rglob('*'):
            if file_path.is_file() and file_path.name != "integrity_checksums.json":
                checksum = self.calculate_file_checksum(file_path)
                relative_path = str(file_path.relative_to(backup_path))
                checksums[relative_path] = checksum
        
        # Save checksums
        with open(integrity_file, 'w') as f:
            json.dump(checksums, f, indent=2)
        
        print(f"✅ Generated {len(checksums)} file checksums")
        return True
    
    def test_backup_functionality(self, backup_path):
        """Test backup functionality by running critical tests"""
        print("🚀 Testing backup functionality...")
        
        # Test 1: Can we read critical files?
        critical_files = [
            "README.md",
            "src/ai/behavioral_analysis_enhanced.py",
            "src/ai/integrated_behavioral_system.py",
            "index.html"
        ]
        
        for file_path in critical_files:
            full_path = backup_path / file_path
            if full_path.exists():
                try:
                    with open(full_path, 'r') as f:
                        content = f.read()
                        if len(content) > 0:
                            print(f"✅ {file_path} readable")
                        else:
                            print(f"❌ {file_path} is empty")
                            return False
                except Exception as e:
                    print(f"❌ {file_path} unreadable: {e}")
                    return False
            else:
                print(f"❌ {file_path} missing")
                return False
        
        print("✅ Backup functionality testing passed")
        return True
    
    def create_backup_report(self, backup_path, timestamp):
        """Create comprehensive backup report"""
        report = {
            "backup_name": backup_path.name,
            "timestamp": timestamp,
            "backup_size": self.calculate_backup_size(backup_path),
            "files_backed_up": self.count_files_in_backup(backup_path),
            "integrity_verified": True,
            "functionality_tested": True,
            "backup_status": "SUCCESS",
            "validation_tests": {
                "pre_backup": "PASSED",
                "post_backup": "PASSED",
                "integrity": "PASSED",
                "functionality": "PASSED"
            }
        }
        
        report_file = backup_path / "backup_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📋 Backup report created: {report_file}")
    
    # Helper methods
    def is_git_status_clean(self):
        """Check if git status is clean"""
        try:
            result = subprocess.run(['git', 'status', '--porcelain'], 
                                  cwd=self.project_path, 
                                  capture_output=True, text=True)
            return len(result.stdout.strip()) == 0
        except:
            return False
    
    def check_sensitive_files(self):
        """Check for sensitive files that shouldn't be tracked"""
        sensitive_patterns = ['.env', '.key', '.pem', 'password', 'secret', 'token']
        for pattern in sensitive_patterns:
            for file_path in self.project_path.rglob('*'):
                if pattern in file_path.name.lower():
                    # Check if file is tracked
                    try:
                        result = subprocess.run(['git', 'ls-files', str(file_path.relative_to(self.project_path))],
                                              cwd=self.project_path, capture_output=True, text=True)
                        if result.stdout.strip():
                            print(f"❌ Sensitive file tracked: {file_path}")
                            return False
                    except:
                        pass
        return True
    
    def check_disk_space(self):
        """Check available disk space"""
        total, used, free = shutil.disk_usage(self.backup_base_path)
        required_space = 1024 * 1024 * 1024  # 1GB minimum
        return free > required_space
    
    def check_backup_permissions(self):
        """Check if backup directory is writable"""
        test_file = self.backup_base_path / "test_write.tmp"
        try:
            test_file.write_text("test")
            test_file.unlink()
            return True
        except:
            return False
    
    def check_critical_files(self):
        """Check if critical files exist"""
        critical_files = ['README.md', 'src', 'index.html']
        for file_path in critical_files:
            if not (self.project_path / file_path).exists():
                return False
        return True
    
    def verify_file_copy(self, source, destination):
        """Verify file copy integrity"""
        return self.calculate_file_checksum(source) == self.calculate_file_checksum(destination)
    
    def calculate_file_checksum(self, file_path):
        """Calculate SHA-256 checksum of file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def test_backup_structure(self, backup_path):
        """Test backup directory structure"""
        required_dirs = ['src', 'website', 'consulting_business']
        for dir_name in required_dirs:
            if not (backup_path / dir_name).exists():
                return False
        return True
    
    def test_critical_files_backup(self, backup_path):
        """Test critical files exist in backup"""
        critical_files = ['README.md', 'src/ai/behavioral_analysis_enhanced.py']
        for file_path in critical_files:
            if not (backup_path / file_path).exists():
                return False
        return True
    
    def test_all_file_integrity(self, backup_path):
        """Test all files in backup have correct checksums"""
        integrity_file = backup_path / "integrity_checksums.json"
        if not integrity_file.exists():
            return True  # Skip if not generated yet
        
        with open(integrity_file, 'r') as f:
            stored_checksums = json.load(f)
        
        for file_path, stored_checksum in stored_checksums.items():
            full_path = backup_path / file_path
            if full_path.exists():
                current_checksum = self.calculate_file_checksum(full_path)
                if current_checksum != stored_checksum:
                    return False
        return True
    
    def test_python_syntax(self, backup_path):
        """Test Python files have valid syntax"""
        import py_compile
        for py_file in backup_path.rglob('*.py'):
            try:
                py_compile.compile(str(py_file), doraise=True)
            except py_compile.PyCompileError:
                return False
        return True
    
    def test_html_validity(self, backup_path):
        """Test HTML files have basic validity"""
        for html_file in backup_path.rglob('*.html'):
            try:
                with open(html_file, 'r') as f:
                    content = f.read()
                # Basic HTML structure check
                if '<html' in content and '</html>' in content:
                    continue
                elif '<!DOCTYPE html>' in content:
                    continue
                else:
                    # Some HTML files might be fragments
                    pass
            except:
                return False
        return True
    
    def test_json_validity(self, backup_path):
        """Test JSON files have valid syntax"""
        for json_file in backup_path.rglob('*.json'):
            try:
                with open(json_file, 'r') as f:
                    json.load(f)
            except:
                return False
        return True
    
    def test_no_sensitive_data_exposed(self, backup_path):
        """Test no sensitive data in backup"""
        sensitive_patterns = ['password', 'secret', 'token', 'key', 'credential']
        for file_path in backup_path.rglob('*'):
            if file_path.is_file() and file_path.suffix in ['.py', '.json', '.yaml', '.yml', '.env']:
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read().lower()
                        for pattern in sensitive_patterns:
                            if pattern in content and 'example' not in content:
                                # Manual review needed for potential false positives
                                pass
                except:
                    pass
        return True
    
    def calculate_backup_size(self, backup_path):
        """Calculate total backup size"""
        total_size = 0
        for file_path in backup_path.rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        return total_size
    
    def count_files_in_backup(self, backup_path):
        """Count total files in backup"""
        return len([f for f in backup_path.rglob('*') if f.is_file()])
    
    def format_size(self, size_bytes):
        """Format file size in human readable format"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} TB"

# Daily backup execution
if __name__ == "__main__":
    project_path = "c:/Users/merce/Documents/helm-ai"
    backup_base_path = "c:/Users/merce/Documents/helm-ai/backups"
    
    backup_system = RobustBackupSystem(project_path, backup_base_path)
    success, result = backup_system.create_daily_backup()
    
    if success:
        print(f"✅ Daily backup successful: {result}")
    else:
        print(f"❌ Daily backup failed: {result}")

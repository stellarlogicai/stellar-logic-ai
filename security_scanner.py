#!/usr/bin/env python3
"""
Security Scanner for Stellar Logic AI
"""

import os
import re
import json
from pathlib import Path

def scan_for_credentials(project_path):
    """Scan project for exposed credentials"""
    print("🔍 Scanning for exposed credentials...")
    
    # Security patterns to detect (excluding scanner itself)
    patterns = [
        r'"type":\s*"service_account"',
        r'"private_key":\s*"-----BEGIN PRIVATE KEY-----',
        r'@.*\.gserviceaccount\.com',
        r'AKIA[0-9A-Z]{16}',
        r'-----BEGIN RSA PRIVATE KEY-----',
        r'password\s*=\s*["\']?[^"\'\s]+["\']?',
        r'api_key\s*=\s*["\']?[^"\'\s]+["\']?',
        r'sk-[A-Za-z-9]{48,}',
        r'ghp_[A-Za-z0-9]{36}',
        r'AIza[0-9A-Za-z_-]{35}',
    ]
    
    issues_found = []
    files_scanned = 0
    
    # Scan Python, JSON, and config files (exclude .venv and backups)
    for file_path in Path(project_path).rglob('*.py'):
        # Skip .venv, backups, and security scanner
        if '.venv' in str(file_path) or 'backups' in str(file_path) or 'security_scanner.py' in str(file_path):
            continue
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                for pattern in patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        issues_found.append(str(file_path.relative_to(project_path)))
                        break
            files_scanned += 1
        except:
            pass
    
    for file_path in Path(project_path).rglob('*.json'):
        # Skip .venv and backups directories
        if '.venv' in str(file_path) or 'backups' in str(file_path):
            continue
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                for pattern in patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        issues_found.append(str(file_path.relative_to(project_path)))
                        break
            files_scanned += 1
        except:
            pass
    
    print(f"📊 Scanned {files_scanned} files")
    
    if issues_found:
        print(f"❌ SECURITY ISSUES FOUND in {len(issues_found)} files:")
        for issue in issues_found:
            print(f"   🚨 {issue}")
        return False
    else:
        print("✅ No security issues found")
        return True

if __name__ == "__main__":
    project_path = "c:/Users/merce/Documents/helm-ai"
    scan_for_credentials(project_path)

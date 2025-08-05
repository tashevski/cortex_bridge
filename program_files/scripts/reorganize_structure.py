#!/usr/bin/env python3
"""
Script to reorganize the project structure for better organization.
This script will:
1. Create new directories
2. Move files to their new locations
3. Update import statements
"""

import os
import shutil
import re
from pathlib import Path
from typing import Dict, List, Tuple

# Define the reorganization mapping
REORGANIZATION_MAP = {
    # Config files
    'utils/config.py': 'config/config.py',
    'utils/runtime_config.py': 'config/runtime_config.py',
    
    # Database files
    'utils/db_helpers.py': 'database/db_helpers.py',
    'utils/enhanced_conversation_db.py': 'database/enhanced_conversation_db.py',
    'data/vector_db': 'database/data/vector_db',
    
    # Scripts
    'check_confidence.py': 'scripts/check_confidence.py',
    'reset_database.py': 'scripts/reset_database.py',
    
    # Notebooks
    'audio_features_inspector.ipynb': 'notebooks/audio_features_inspector.ipynb',
    'dev_area.ipynb': 'notebooks/dev_area.ipynb',
}

# Define import replacements
IMPORT_REPLACEMENTS = [
    # Config imports
    (r'from utils\.config import', 'from config.config import'),
    (r'from utils\.runtime_config import', 'from config.runtime_config import'),
    (r'import utils\.config', 'import config.config'),
    
    # Database imports
    (r'from utils\.db_helpers import', 'from database.db_helpers import'),
    (r'from utils\.enhanced_conversation_db import', 'from database.enhanced_conversation_db import'),
    (r'import utils\.db_helpers', 'import database.db_helpers'),
    (r'import utils\.enhanced_conversation_db', 'import database.enhanced_conversation_db'),
]

def create_directories(base_path: Path):
    """Create new directory structure"""
    new_dirs = [
        'config',
        'database',
        'database/data',
        'scripts',
        'notebooks'
    ]
    
    for dir_name in new_dirs:
        dir_path = base_path / dir_name
        dir_path.mkdir(exist_ok=True)
        print(f"Created directory: {dir_path}")
        
        # Create __init__.py for Python packages
        if dir_name in ['config', 'database']:
            init_file = dir_path / '__init__.py'
            if not init_file.exists():
                init_file.write_text('')
                print(f"Created {init_file}")

def move_files(base_path: Path) -> List[Tuple[str, str]]:
    """Move files according to reorganization map"""
    moved_files = []
    
    for old_path, new_path in REORGANIZATION_MAP.items():
        old_full = base_path / old_path
        new_full = base_path / new_path
        
        if old_full.exists():
            # Create parent directory if needed
            new_full.parent.mkdir(parents=True, exist_ok=True)
            
            # Move file or directory
            if old_full.is_dir():
                if new_full.exists():
                    shutil.rmtree(new_full)
                shutil.move(str(old_full), str(new_full))
            else:
                shutil.move(str(old_full), str(new_full))
            
            moved_files.append((old_path, new_path))
            print(f"Moved: {old_path} -> {new_path}")
        else:
            print(f"Warning: {old_path} not found")
    
    return moved_files

def update_imports_in_file(file_path: Path) -> bool:
    """Update imports in a single Python file"""
    if not file_path.suffix == '.py':
        return False
        
    try:
        content = file_path.read_text()
        original_content = content
        
        for pattern, replacement in IMPORT_REPLACEMENTS:
            content = re.sub(pattern, replacement, content)
        
        if content != original_content:
            file_path.write_text(content)
            return True
    except Exception as e:
        print(f"Error updating {file_path}: {e}")
    
    return False

def update_all_imports(base_path: Path):
    """Update imports in all Python files"""
    python_files = list(base_path.rglob('*.py'))
    updated_count = 0
    
    for py_file in python_files:
        if update_imports_in_file(py_file):
            updated_count += 1
            print(f"Updated imports in: {py_file.relative_to(base_path)}")
    
    print(f"\nUpdated imports in {updated_count} files")

def create_backup(base_path: Path):
    """Create a backup of critical files before reorganization"""
    backup_dir = base_path / 'backup_before_reorg'
    backup_dir.mkdir(exist_ok=True)
    
    # Backup key files
    files_to_backup = [
        'utils/config.py',
        'utils/runtime_config.py',
        'utils/db_helpers.py',
        'utils/enhanced_conversation_db.py',
    ]
    
    for file_path in files_to_backup:
        full_path = base_path / file_path
        if full_path.exists():
            backup_path = backup_dir / file_path.replace('/', '_')
            shutil.copy2(full_path, backup_path)
            print(f"Backed up: {file_path}")

def main(auto_confirm=False):
    """Main reorganization function"""
    base_path = Path(__file__).parent.parent
    
    print("Project Structure Reorganization")
    print("================================")
    print(f"Base path: {base_path}")
    
    # Ask for confirmation unless auto_confirm is True
    if not auto_confirm:
        response = input("\nThis will reorganize your project structure. Continue? (y/n): ")
        if response.lower() != 'y':
            print("Reorganization cancelled")
            return
    else:
        print("\nAuto-confirmed, proceeding with reorganization...")
    
    print("\n1. Creating backup...")
    create_backup(base_path)
    
    print("\n2. Creating new directories...")
    create_directories(base_path)
    
    print("\n3. Moving files...")
    moved_files = move_files(base_path)
    
    print("\n4. Updating imports...")
    update_all_imports(base_path)
    
    # Clean up empty directories
    if (base_path / 'data').exists() and not list((base_path / 'data').iterdir()):
        (base_path / 'data').rmdir()
        print("\nRemoved empty data directory")
    
    print("\n✅ Reorganization complete!")
    print(f"Moved {len(moved_files)} files/directories")
    print("\nNext steps:")
    print("1. Test the application to ensure everything works")
    print("2. Update any hardcoded paths in configuration files")
    print("3. Delete the backup directory once confirmed working")

if __name__ == "__main__":
    import sys
    # Check if --auto flag is passed
    auto_confirm = '--auto' in sys.argv
    main(auto_confirm=auto_confirm)
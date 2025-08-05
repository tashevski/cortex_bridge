# How to save a Python dictionary as JSON
import json
from datetime import datetime

# Example dictionary
sample_dict = {
    "name": "John Doe",
    "age": 30,
    "city": "New York",
    "skills": ["Python", "JavaScript", "SQL"],
    "active": True,
    "metadata": {
        "created": "2024-01-01",
        "version": "1.0"
    }
}

print("=== JSON Dictionary Saving Examples ===\n")

# Method 1: Save to file with json.dump() - Basic
print("Method 1: Basic json.dump()")
with open('data.json', 'w') as f:
    json.dump(sample_dict, f, indent=4)
print("✅ Saved to data.json")

# Method 2: Save to file with custom formatting
print("\nMethod 2: Custom formatting with sort_keys")
with open('data_formatted.json', 'w') as f:
    json.dump(sample_dict, f, indent=2, sort_keys=True)
print("✅ Saved to data_formatted.json")

# Method 3: Convert to JSON string first, then save
print("\nMethod 3: Convert to string first with json.dumps()")
json_string = json.dumps(sample_dict, indent=4)
with open('data_from_string.json', 'w') as f:
    f.write(json_string)
print("✅ Saved to data_from_string.json")

# Method 4: Save with custom encoding for international characters
print("\nMethod 4: UTF-8 encoding with ensure_ascii=False")
with open('data_utf8.json', 'w', encoding='utf-8') as f:
    json.dump(sample_dict, f, indent=4, ensure_ascii=False)
print("✅ Saved to data_utf8.json")

# Method 5: Save with custom serialization for non-serializable objects
print("\nMethod 5: Custom encoder for non-serializable objects")
class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        return str(obj)

# Example with datetime (which isn't JSON serializable by default)
dict_with_date = {
    "timestamp": datetime.now(),
    "data": sample_dict
}

with open('data_with_custom_encoder.json', 'w') as f:
    json.dump(dict_with_date, f, indent=4, cls=CustomEncoder)
print("✅ Saved to data_with_custom_encoder.json")

# Method 6: Using default parameter for simple cases
print("\nMethod 6: Using default parameter")
with open('data_with_default.json', 'w') as f:
    json.dump(dict_with_date, f, indent=4, default=str)
print("✅ Saved to data_with_default.json")

# Method 7: Save with compression (gzip)
print("\nMethod 7: Save with gzip compression")
import gzip

with gzip.open('data.json.gz', 'wt', encoding='utf-8') as f:
    json.dump(sample_dict, f, indent=4)
print("✅ Saved to data.json.gz (compressed)")

# Method 8: Pretty print to console
print("\nMethod 8: Pretty print to console")
print(json.dumps(sample_dict, indent=4))

# Method 9: Save with error handling
print("\nMethod 9: Save with error handling")
try:
    with open('data_safe.json', 'w') as f:
        json.dump(sample_dict, f, indent=4)
    print("✅ Saved to data_safe.json")
except Exception as e:
    print(f"❌ Error saving file: {e}")

# Method 10: Save with backup
print("\nMethod 10: Save with backup")
import os
import shutil

filename = 'data_with_backup.json'
backup_filename = 'data_with_backup.json.backup'

# Create backup if file exists
if os.path.exists(filename):
    shutil.copy2(filename, backup_filename)
    print(f"✅ Created backup: {backup_filename}")

# Save new file
with open(filename, 'w') as f:
    json.dump(sample_dict, f, indent=4)
print(f"✅ Saved to {filename}")

print("\n=== All JSON files created successfully! ===")
print("\nFiles created:")
for file in ['data.json', 'data_formatted.json', 'data_from_string.json', 
             'data_utf8.json', 'data_with_custom_encoder.json', 'data_with_default.json',
             'data.json.gz', 'data_safe.json', 'data_with_backup.json']:
    if os.path.exists(file):
        size = os.path.getsize(file)
        print(f"  {file} ({size} bytes)") 
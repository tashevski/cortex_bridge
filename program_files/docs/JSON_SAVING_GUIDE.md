# How to Save Python Dictionaries as JSON

This guide covers various methods for saving Python dictionaries to JSON files, from basic to advanced techniques.

## Basic Methods

### Method 1: Simple JSON Dump
```python
import json

data = {"name": "John", "age": 30}
with open('data.json', 'w') as f:
    json.dump(data, f, indent=4)
```

### Method 2: Formatted JSON with Sorting
```python
with open('data.json', 'w') as f:
    json.dump(data, f, indent=2, sort_keys=True)
```

### Method 3: Convert to String First
```python
json_string = json.dumps(data, indent=4)
with open('data.json', 'w') as f:
    f.write(json_string)
```

## Advanced Methods

### Method 4: UTF-8 Encoding for International Characters
```python
with open('data.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=4, ensure_ascii=False)
```

### Method 5: Custom Encoder for Non-Serializable Objects
```python
from datetime import datetime

class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        return str(obj)

data_with_date = {"timestamp": datetime.now(), "data": data}
with open('data.json', 'w') as f:
    json.dump(data_with_date, f, indent=4, cls=CustomEncoder)
```

### Method 6: Using Default Parameter
```python
with open('data.json', 'w') as f:
    json.dump(data_with_date, f, indent=4, default=str)
```

## Specialized Methods

### Method 7: Compressed JSON (Gzip)
```python
import gzip

with gzip.open('data.json.gz', 'wt', encoding='utf-8') as f:
    json.dump(data, f, indent=4)
```

### Method 8: Error Handling
```python
try:
    with open('data.json', 'w') as f:
        json.dump(data, f, indent=4)
    print("✅ Saved successfully")
except Exception as e:
    print(f"❌ Error: {e}")
```

### Method 9: With Backup
```python
import os
import shutil

filename = 'data.json'
backup_filename = 'data.json.backup'

# Create backup if file exists
if os.path.exists(filename):
    shutil.copy2(filename, backup_filename)

# Save new file
with open(filename, 'w') as f:
    json.dump(data, f, indent=4)
```

## Key Parameters

- `indent`: Number of spaces for indentation (makes JSON readable)
- `sort_keys`: Sort dictionary keys alphabetically
- `ensure_ascii`: Set to `False` for proper UTF-8 encoding
- `cls`: Custom JSON encoder class
- `default`: Function to handle non-serializable objects

## Common Use Cases

1. **Configuration files**: Save settings and parameters
2. **Data export**: Export processed data for analysis
3. **API responses**: Save API data for caching
4. **Logging**: Save structured log data
5. **Backup**: Create data backups with timestamps

## Best Practices

1. Always use `indent` for human-readable files
2. Use `ensure_ascii=False` for international text
3. Handle non-serializable objects with custom encoders
4. Include error handling for file operations
5. Consider compression for large files
6. Create backups before overwriting existing files

## Example with Your Project Data

For the conversations data in your project:

```python
# Save conversations dictionary
if 'conversations' in globals():
    with open('conversations.json', 'w') as f:
        json.dump(conversations, f, indent=4, default=str)
    print("✅ Conversations saved to conversations.json")
```

The `default=str` parameter handles datetime objects that might be in your conversation data. 
import os
import json 
import threading

path = os.path.join('process.json')

if os.path.exists(path):
    print("✅ File exists. Resetting content...")
else:
    print("📄 File not found. Creating new one...")

with open(path, "w", encoding="utf-8") as f:
    json.dump({}, f, indent=2)
    
print("✅ process.json is now reset to {}")

lock = threading.Lock()

def get_all():
    with lock:
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
        else:
            data = {}

        return data
    
def error_field(filePath):
    fieldName = os.path.basename(filePath)
    
    with lock:
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
        else:
            data = {}

        data[fieldName + "_inversion"]["current_step"] = -1

        with open(path, 'w') as f:
            json.dump(data, f)

def set_inversion_image(fieldName, jsonObject):
    with lock:
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
        else:
            data = {}

        data[fieldName + "_inversion"] = jsonObject

        with open(path, 'w') as f:
            json.dump(data, f)
            
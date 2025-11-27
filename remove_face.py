#!/usr/bin/env python3
"""
Helper script to remove a person's face encodings from the system.
Usage: python remove_face.py <Name>
"""

import sys
import os
import numpy as np
import json

def remove_person(name):
    """Remove all encodings for a specific person."""
    encodings_path = os.path.join("face_dataset", "face_encodings.npy")
    names_path = os.path.join("face_dataset", "face_encodings_names.json")
    
    if not os.path.exists(encodings_path) or not os.path.exists(names_path):
        print(f"Error: Encodings files not found")
        return False
    
    # Load current data
    encodings = np.load(encodings_path).tolist()
    with open(names_path, 'r') as f:
        names = json.load(f)
    
    if name not in names:
        print(f"'{name}' not found in encodings")
        return False
    
    # Count before
    count_before = names.count(name)
    
    # Filter out the person
    filtered_encodings = [e for i, e in enumerate(encodings) if names[i] != name]
    filtered_names = [n for n in names if n != name]
    
    # Save updated data
    np.save(encodings_path, np.array(filtered_encodings))
    with open(names_path, 'w') as f:
        json.dump(filtered_names, f)
    
    print(f"Removed {count_before} encodings for '{name}'")
    print(f"Remaining people: {set(filtered_names)}")
    return True

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python remove_face.py <Name>")
        print("Example: python remove_face.py Aldridge")
        sys.exit(1)
    
    name = sys.argv[1]
    remove_person(name)


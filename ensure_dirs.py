import os
from pathlib import Path

Path("data").mkdir(exist_ok=True)
Path("data/uploads/sources").mkdir(parents=True, exist_ok=True)

metadata_file = Path("data/documents_metadata.json")
if not metadata_file.exists():
    metadata_file.write_text('{}')
    print("✓ Created documents_metadata.json")

print("✓ Directories ready")
#!/usr/bin/env python3
"""Clear outputs and execution counts from .ipynb files under Week-* directories.
Usage: python scripts/clear_notebook_outputs.py
"""
import os
import sys

try:
    import nbformat
except Exception as e:
    print("ERROR: nbformat not installed.")
    print("Run: python -m pip install nbformat")
    sys.exit(2)

changed = []
processed = 0

cwd = os.getcwd()
entries = sorted([d for d in os.listdir(cwd) if d.startswith('Week-') and os.path.isdir(d)])
if not entries:
    print("No Week-* directories found in", cwd)
    sys.exit(0)

for week in entries:
    for root, dirs, files in os.walk(week):
        for name in files:
            if name.endswith('.ipynb'):
                path = os.path.join(root, name)
                processed += 1
                try:
                    nb = nbformat.read(path, as_version=4)
                except Exception as e:
                    print(f"Failed to read {path}: {e}")
                    continue
                modified = False
                for cell in nb.get('cells', []):
                    if cell.get('outputs'):
                        cell['outputs'] = []
                        modified = True
                    if 'execution_count' in cell and cell.get('execution_count') is not None:
                        cell['execution_count'] = None
                        modified = True
                # also clear notebook-level "widgets" or "execution" metadata if present
                metadata = nb.get('metadata', {})
                if metadata.get('execution', None):
                    metadata.pop('execution', None)
                    modified = True
                if modified:
                    try:
                        nbformat.write(nb, path)
                        changed.append(path)
                    except Exception as e:
                        print(f"Failed to write {path}: {e}")

print('\nSummary:')
print(f'  Week-* directories scanned: {len(entries)}')
print(f'  Notebooks processed: {processed}')
print(f'  Notebooks modified (outputs cleared): {len(changed)}')
if changed:
    print('\nModified files:')
    for p in changed:
        print('  ' + p)

if not changed:
    print('\nNo notebooks needed output-clearing.')

# Exit code 0 even if nothing changed
sys.exit(0)

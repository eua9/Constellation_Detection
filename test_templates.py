#!/usr/bin/env python3
"""
Quick test script to load and display templates.
"""

from templates import load_templates

if __name__ == "__main__":
    templates = load_templates('templates_config.json')
    print(f"\nLoaded {len(templates)} templates:")
    for name, points in templates.items():
        print(f"  - {name}: {len(points)} stars")


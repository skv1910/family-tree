"""
Migration script to copy existing JSON data to Supabase.
Run this once after setting up Supabase to migrate your existing data.

Usage:
    python migrate_to_supabase.py

Make sure to set environment variables:
    SUPABASE_URL=your-project-url
    SUPABASE_KEY=your-anon-key
"""

import json
import os
from pathlib import Path

from supabase import create_client

# Load environment variables or use defaults
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

DATA_PATH = Path(__file__).parent / "data" / "family.json"


def migrate():
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("Error: Please set SUPABASE_URL and SUPABASE_KEY environment variables")
        print("\nExample:")
        print('  export SUPABASE_URL="https://your-project.supabase.co"')
        print('  export SUPABASE_KEY="your-anon-key"')
        return

    # Load JSON data
    if not DATA_PATH.exists():
        print(f"Error: Data file not found at {DATA_PATH}")
        return

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Connect to Supabase
    client = create_client(SUPABASE_URL, SUPABASE_KEY)
    print(f"Connected to Supabase: {SUPABASE_URL}")

    # Migrate people
    people = data.get("people", {})
    print(f"\nMigrating {len(people)} people...")
    for pid, person in people.items():
        try:
            client.table("people").upsert({
                "id": pid,
                "name": person.get("name"),
                "gender": person.get("gender"),
                "image_path": person.get("image_path"),
                "birth_year": person.get("birth_year"),
                "death_year": person.get("death_year"),
                "birth_order": person.get("birth_order"),
            }).execute()
            print(f"  ✓ {person.get('name', pid)}")
        except Exception as e:
            print(f"  ✗ {person.get('name', pid)}: {e}")

    # Migrate edges
    edges = data.get("edges", [])
    print(f"\nMigrating {len(edges)} parent-child relationships...")
    for edge in edges:
        if len(edge) == 2:
            try:
                # Check if edge already exists
                result = client.table("edges").select("*").eq("parent_id", edge[0]).eq("child_id", edge[1]).execute()
                if not result.data:
                    client.table("edges").insert({
                        "parent_id": edge[0],
                        "child_id": edge[1],
                    }).execute()
                    print(f"  ✓ {edge[0]} -> {edge[1]}")
                else:
                    print(f"  - {edge[0]} -> {edge[1]} (already exists)")
            except Exception as e:
                print(f"  ✗ {edge[0]} -> {edge[1]}: {e}")

    # Migrate spouses
    spouses = data.get("spouses", [])
    print(f"\nMigrating {len(spouses)} spouse relationships...")
    for spouse in spouses:
        if len(spouse) == 2:
            try:
                # Check if spouse relationship already exists
                result = client.table("spouses").select("*").eq("person1_id", spouse[0]).eq("person2_id", spouse[1]).execute()
                if not result.data:
                    client.table("spouses").insert({
                        "person1_id": spouse[0],
                        "person2_id": spouse[1],
                    }).execute()
                    print(f"  ✓ {spouse[0]} ❤️ {spouse[1]}")
                else:
                    print(f"  - {spouse[0]} ❤️ {spouse[1]} (already exists)")
            except Exception as e:
                print(f"  ✗ {spouse[0]} ❤️ {spouse[1]}: {e}")

    print("\n✅ Migration complete!")


if __name__ == "__main__":
    migrate()

#!/usr/bin/env python3
"""
Backup script for Family Tree application.
Fetches data from Supabase and saves to local JSON files.

Usage:
    ./backup_data.py              # Creates timestamped backup
    ./backup_data.py --restore    # Restores from most recent backup to Supabase
"""

import json
import sys
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
SECRETS_PATH = BASE_DIR / ".streamlit" / "secrets.toml"


def get_supabase_client():
    """Get Supabase client from secrets."""
    try:
        from supabase import create_client
        import toml

        if not SECRETS_PATH.exists():
            print("Error: .streamlit/secrets.toml not found")
            return None

        secrets = toml.load(SECRETS_PATH)
        url = secrets.get('SUPABASE_URL')
        key = secrets.get('SUPABASE_KEY')

        if not url or not key:
            print("Error: SUPABASE_URL or SUPABASE_KEY not found in secrets")
            return None

        return create_client(url, key)
    except ImportError as e:
        print(f"Error: Missing dependency - {e}")
        print("Run: pip install supabase toml")
        return None


def fetch_from_supabase(client) -> dict:
    """Fetch all data from Supabase."""
    data = {'people': {}, 'edges': [], 'spouses': []}

    # Load people
    result = client.table('people').select('*').execute()
    for row in result.data:
        data['people'][row['id']] = {
            'name': row['name'],
            'gender': row['gender'],
            'image_path': row.get('image_path'),
            'birth_year': row.get('birth_year'),
            'death_year': row.get('death_year'),
            'birth_order': row.get('birth_order'),
        }

    # Load edges (parent-child relationships)
    result = client.table('edges').select('*').execute()
    for row in result.data:
        data['edges'].append([row['parent_id'], row['child_id']])

    # Load spouses
    result = client.table('spouses').select('*').execute()
    for row in result.data:
        data['spouses'].append([row['person1_id'], row['person2_id']])

    return data


def backup():
    """Backup data from Supabase to local JSON files."""
    print("Starting backup from Supabase...")

    client = get_supabase_client()
    if not client:
        return False

    try:
        data = fetch_from_supabase(client)

        print(f"Fetched from Supabase:")
        print(f"  People: {len(data['people'])}")
        print(f"  Edges: {len(data['edges'])}")
        print(f"  Spouses: {len(data['spouses'])}")

        # Ensure data directory exists
        DATA_DIR.mkdir(parents=True, exist_ok=True)

        # Save timestamped backup
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = DATA_DIR / f"backup_{timestamp}.json"
        with open(backup_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Saved backup to: {backup_path}")

        # Update main family.json
        main_path = DATA_DIR / "family.json"
        with open(main_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Updated: {main_path}")

        print("\nBackup completed successfully!")
        return True

    except Exception as e:
        print(f"Error during backup: {e}")
        return False


def restore():
    """Restore data from most recent backup to Supabase."""
    print("Starting restore to Supabase...")

    # Find most recent backup
    backups = sorted(DATA_DIR.glob("backup_*.json"), reverse=True)
    if not backups:
        # Fall back to family.json
        main_path = DATA_DIR / "family.json"
        if main_path.exists():
            backup_path = main_path
        else:
            print("Error: No backup files found")
            return False
    else:
        backup_path = backups[0]

    print(f"Restoring from: {backup_path}")

    with open(backup_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Data to restore:")
    print(f"  People: {len(data['people'])}")
    print(f"  Edges: {len(data['edges'])}")
    print(f"  Spouses: {len(data['spouses'])}")

    client = get_supabase_client()
    if not client:
        return False

    try:
        # Clear existing data
        print("Clearing existing data...")
        client.table('edges').delete().neq('parent_id', '').execute()
        client.table('spouses').delete().neq('person1_id', '').execute()
        client.table('people').delete().neq('id', '').execute()

        # Insert people
        print("Inserting people...")
        for pid, person in data['people'].items():
            client.table('people').upsert({
                'id': pid,
                'name': person.get('name'),
                'gender': person.get('gender'),
                'image_path': person.get('image_path'),
                'birth_year': person.get('birth_year'),
                'death_year': person.get('death_year'),
                'birth_order': person.get('birth_order'),
            }).execute()

        # Insert edges
        print("Inserting edges...")
        for edge in data['edges']:
            client.table('edges').insert({
                'parent_id': edge[0],
                'child_id': edge[1],
            }).execute()

        # Insert spouses
        print("Inserting spouses...")
        for spouse in data['spouses']:
            client.table('spouses').insert({
                'person1_id': spouse[0],
                'person2_id': spouse[1],
            }).execute()

        print("\nRestore completed successfully!")
        return True

    except Exception as e:
        print(f"Error during restore: {e}")
        return False


def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--restore':
        success = restore()
    else:
        success = backup()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

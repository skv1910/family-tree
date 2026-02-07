#!/usr/bin/env python3
"""
Generate PDF backup of Family Tree data.

Usage:
    ./generate_pdf.py                    # Creates PDF from local JSON
    ./generate_pdf.py --from-supabase    # Fetches from Supabase first, then creates PDF
"""

import json
import sys
from datetime import datetime
from pathlib import Path

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    from reportlab.lib.enums import TA_CENTER
except ImportError:
    print("Error: reportlab not installed. Run: pip install reportlab")
    sys.exit(1)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"


def load_data(from_supabase=False):
    """Load family data from JSON or Supabase."""
    if from_supabase:
        try:
            from supabase import create_client
            import toml

            secrets_path = BASE_DIR / ".streamlit" / "secrets.toml"
            if secrets_path.exists():
                secrets = toml.load(secrets_path)
                url = secrets.get('SUPABASE_URL')
                key = secrets.get('SUPABASE_KEY')

                if url and key:
                    client = create_client(url, key)
                    data = {'people': {}, 'edges': [], 'spouses': []}

                    result = client.table('people').select('*').execute()
                    for row in result.data:
                        data['people'][row['id']] = {
                            'name': row['name'],
                            'gender': row['gender'],
                            'birth_year': row.get('birth_year'),
                            'death_year': row.get('death_year'),
                        }

                    result = client.table('edges').select('*').execute()
                    for row in result.data:
                        data['edges'].append([row['parent_id'], row['child_id']])

                    result = client.table('spouses').select('*').execute()
                    for row in result.data:
                        data['spouses'].append([row['person1_id'], row['person2_id']])

                    print(f"Loaded from Supabase: {len(data['people'])} people")
                    return data
        except Exception as e:
            print(f"Warning: Could not load from Supabase ({e}), using local JSON")

    # Load from local JSON
    json_path = DATA_DIR / "family.json"
    if json_path.exists():
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Loaded from local JSON: {len(data.get('people', {}))} people")
        return data

    print("Error: No data source available")
    return None


def generate_pdf(data, output_path=None):
    """Generate PDF from family data."""
    if not data:
        return False

    people = data.get('people', {})
    edges = data.get('edges', [])
    spouses = data.get('spouses', [])

    # Build relationship maps
    spouse_map = {}
    for p1, p2 in spouses:
        spouse_map[p1] = p2
        spouse_map[p2] = p1

    # Output path
    if output_path is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = DATA_DIR / f"family_tree_{timestamp}.pdf"
    else:
        output_path = Path(output_path)

    # Ensure directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create PDF document
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=1*cm,
        leftMargin=1*cm,
        topMargin=1*cm,
        bottomMargin=1*cm
    )

    # Styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=20,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#1e293b')
    )
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        spaceBefore=20,
        spaceAfter=10,
        textColor=colors.HexColor('#334155')
    )

    # Build content
    content = []

    # Title
    content.append(Paragraph("🌳 Family Tree Data Backup", title_style))
    content.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    content.append(Paragraph(f"Total Members: {len(people)}", styles['Normal']))
    content.append(Spacer(1, 0.5*cm))

    # Family Members Table
    content.append(Paragraph("Family Members", heading_style))

    table_data = [['Name', 'Gender', 'Birth', 'Death', 'Spouse']]
    for pid in sorted(people.keys(), key=lambda x: people[x].get('name', '')):
        p = people[pid]
        name = p.get('name', 'Unknown')
        gender = 'Male' if p.get('gender') == 'male' else 'Female'
        birth = p.get('birth_year') or '-'
        death = p.get('death_year') or '-'
        spouse_id = spouse_map.get(pid)
        spouse_name = people.get(spouse_id, {}).get('name', '-') if spouse_id else '-'
        table_data.append([name, gender, str(birth), str(death), spouse_name])

    table = Table(table_data, colWidths=[5*cm, 2*cm, 2*cm, 2*cm, 5*cm])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e293b')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f1f5f9')]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    content.append(table)

    # Marriages section
    content.append(PageBreak())
    content.append(Paragraph("Marriages / Couples", heading_style))

    marriage_data = [['Person 1', 'Person 2']]
    seen = set()
    for p1, p2 in spouses:
        key = tuple(sorted([p1, p2]))
        if key in seen:
            continue
        seen.add(key)
        n1 = people.get(p1, {}).get('name', p1)
        n2 = people.get(p2, {}).get('name', p2)
        marriage_data.append([n1, n2])

    marriage_table = Table(marriage_data, colWidths=[8*cm, 8*cm])
    marriage_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f87171')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#fef2f2')]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#fecaca')),
    ]))
    content.append(marriage_table)

    # Parent-Child relationships
    content.append(Spacer(1, 0.3*cm))
    content.append(Paragraph("Parent-Child Relationships", heading_style))

    parent_groups = {}
    for parent, child in edges:
        pname = people.get(parent, {}).get('name', parent)
        cname = people.get(child, {}).get('name', child)
        parent_groups.setdefault(pname, set()).add(cname)

    relationship_data = [['Parent', 'Children']]
    for parent, children in sorted(parent_groups.items()):
        children_str = ', '.join(sorted(children))
        relationship_data.append([parent, children_str])

    rel_table = Table(relationship_data, colWidths=[5*cm, 11*cm])
    rel_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#22c55e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0fdf4')]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#bbf7d0')),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    content.append(rel_table)

    # Footer
    content.append(Spacer(1, 0.5*cm))
    content.append(Paragraph("This backup can be used to restore family tree data if needed.", styles['Italic']))

    # Build PDF
    doc.build(content)
    print(f"\nPDF created: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")
    return True


def main():
    from_supabase = '--from-supabase' in sys.argv

    print("=" * 50)
    print("Family Tree PDF Generator")
    print("=" * 50)

    data = load_data(from_supabase=from_supabase)
    if data:
        # Also save to the standard backup path
        success = generate_pdf(data, DATA_DIR / "family_tree_backup.pdf")
        sys.exit(0 if success else 1)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()

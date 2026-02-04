"""
Family Tree Application - Backend
A Streamlit-based family tree visualization application with Supabase database.
"""

import base64
import html as html_module
import json
import uuid
from pathlib import Path
from collections import defaultdict

import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

# Try to import supabase, fall back to JSON if not available
try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False

# Paths
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "family.json"
IMAGE_DIR = BASE_DIR / "assets" / "images"
PLACEHOLDER_PATH = BASE_DIR / "assets" / "placeholder.png"
MALE_PLACEHOLDER = IMAGE_DIR / "Male.jpeg"
FEMALE_PLACEHOLDER = IMAGE_DIR / "Female.jpeg"
TEMPLATE_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

# Constants
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif"}
COLORS = ["#6366f1", "#22c55e", "#f59e0b", "#ef4444", "#a855f7", "#14b8a6", "#f97316", "#3b82f6"]
NODE_W, NODE_H = 120, 150
H_GAP, V_GAP = 40, 120
COUPLE_GAP = 30


# =============================================================================
# Supabase Connection
# =============================================================================

def get_supabase_client() -> "Client | None":
    """Get Supabase client if configured."""
    if not SUPABASE_AVAILABLE:
        return None

    try:
        url = st.secrets.get("SUPABASE_URL")
        key = st.secrets.get("SUPABASE_KEY")
        if url and key:
            return create_client(url, key)
    except Exception:
        pass
    return None


# =============================================================================
# File Operations
# =============================================================================

def ensure_dirs() -> None:
    """Ensure required directories exist."""
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)


def ensure_placeholder() -> None:
    """Create placeholder image if it doesn't exist."""
    PLACEHOLDER_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not PLACEHOLDER_PATH.exists() or PLACEHOLDER_PATH.stat().st_size < 100:
        img = Image.new("RGB", (120, 120), color=(200, 200, 200))
        img.save(PLACEHOLDER_PATH, format="PNG")


def load_data_from_json() -> dict:
    """Load family data from local JSON file."""
    ensure_dirs()
    if not DATA_PATH.exists():
        return {"people": {}, "edges": [], "spouses": []}
    with DATA_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    data.setdefault("people", {})
    data.setdefault("edges", [])
    data.setdefault("spouses", [])
    return data


def save_data_to_json(data: dict) -> None:
    """Save family data to local JSON file."""
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with DATA_PATH.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_data_from_supabase(client: "Client") -> dict:
    """Load family data from Supabase database."""
    data = {"people": {}, "edges": [], "spouses": []}

    # Load people
    result = client.table("people").select("*").execute()
    for row in result.data:
        data["people"][row["id"]] = {
            "name": row["name"],
            "gender": row["gender"],
            "image_path": row.get("image_path"),
            "birth_year": row.get("birth_year"),
            "death_year": row.get("death_year"),
            "birth_order": row.get("birth_order"),
        }

    # Load edges (parent-child relationships)
    result = client.table("edges").select("*").execute()
    for row in result.data:
        data["edges"].append([row["parent_id"], row["child_id"]])

    # Load spouses
    result = client.table("spouses").select("*").execute()
    for row in result.data:
        data["spouses"].append([row["person1_id"], row["person2_id"]])

    return data


def save_person_to_supabase(client: "Client", pid: str, person: dict) -> None:
    """Save or update a person in Supabase."""
    client.table("people").upsert({
        "id": pid,
        "name": person.get("name"),
        "gender": person.get("gender"),
        "image_path": person.get("image_path"),
        "birth_year": person.get("birth_year"),
        "death_year": person.get("death_year"),
        "birth_order": person.get("birth_order"),
    }).execute()


def delete_person_from_supabase(client: "Client", pid: str) -> None:
    """Delete a person and their relationships from Supabase."""
    # Delete edges involving this person
    client.table("edges").delete().eq("parent_id", pid).execute()
    client.table("edges").delete().eq("child_id", pid).execute()
    # Delete spouse relationships
    client.table("spouses").delete().eq("person1_id", pid).execute()
    client.table("spouses").delete().eq("person2_id", pid).execute()
    # Delete the person
    client.table("people").delete().eq("id", pid).execute()


def add_edge_to_supabase(client: "Client", parent_id: str, child_id: str) -> None:
    """Add a parent-child edge to Supabase."""
    # Check if edge already exists
    result = client.table("edges").select("*").eq("parent_id", parent_id).eq("child_id", child_id).execute()
    if not result.data:
        client.table("edges").insert({
            "parent_id": parent_id,
            "child_id": child_id,
        }).execute()


def add_spouse_to_supabase(client: "Client", person1_id: str, person2_id: str) -> None:
    """Add a spouse relationship to Supabase."""
    client.table("spouses").insert({
        "person1_id": person1_id,
        "person2_id": person2_id,
    }).execute()


def load_data() -> dict:
    """Load family data from Supabase or fall back to JSON."""
    client = get_supabase_client()
    if client:
        try:
            return load_data_from_supabase(client)
        except Exception as e:
            st.warning(f"Failed to load from Supabase: {e}. Using local data.")
    return load_data_from_json()


def save_data(data: dict) -> None:
    """Save family data - for JSON fallback only."""
    save_data_to_json(data)


def load_template(name: str) -> str:
    """Load HTML template from templates directory."""
    template_path = TEMPLATE_DIR / name
    return template_path.read_text(encoding="utf-8")


def load_static_file(path: str) -> str:
    """Load static file (CSS or JS) from static directory."""
    file_path = STATIC_DIR / path
    return file_path.read_text(encoding="utf-8")


# =============================================================================
# Data Operations
# =============================================================================

def add_edge(data: dict, parent_id: str, child_id: str) -> None:
    """Add a parent-child edge."""
    edge = [parent_id, child_id]
    if edge not in data["edges"]:
        data["edges"].append(edge)


def add_spouse(data: dict, person1_id: str, person2_id: str) -> None:
    """Add a spouse relationship."""
    if "spouses" not in data:
        data["spouses"] = []
    data["spouses"].append([person1_id, person2_id])


# =============================================================================
# Image Handling
# =============================================================================

def resolve_image_path(image_path: str | None, person: dict | None = None) -> Path:
    """Resolve the image path, falling back to gender-specific or generic placeholder."""
    if image_path:
        candidate = BASE_DIR / image_path
        if candidate.exists():
            return candidate
    gender = (person or {}).get("gender", "").lower()
    if gender == "female" and FEMALE_PLACEHOLDER.exists():
        return FEMALE_PLACEHOLDER
    if gender == "male" and MALE_PLACEHOLDER.exists():
        return MALE_PLACEHOLDER
    return PLACEHOLDER_PATH


def image_to_data_uri(path: Path) -> str:
    """Convert image file to base64 data URI."""
    raw = path.read_bytes()
    b64 = base64.b64encode(raw).decode("utf-8")
    ext = path.suffix.lower()
    mime = "image/png" if ext in (".png", ".gif") else "image/jpeg"
    return f"data:{mime};base64,{b64}"


# =============================================================================
# Layout Computation
# =============================================================================

class FamilyTreeLayout:
    """Computes the layout for the family tree visualization."""

    def __init__(self, data: dict):
        self.data = data
        self.people = data.get("people", {})
        self.edges = data.get("edges", [])
        self.spouse_pairs = data.get("spouses", [])

        # Build relationship maps
        self.children_of = defaultdict(set)
        self.parents_of = defaultdict(set)
        self._build_relationships()

        # Build spouse map
        self.spouse_map = {}
        self._build_spouse_map()

        # Layout results
        self.positions = {}
        self.connections = []

    def _build_relationships(self):
        """Build parent-child relationship maps."""
        for e in self.edges:
            if len(e) == 2:
                parent, child = e[0], e[1]
                self.children_of[parent].add(child)
                self.parents_of[child].add(parent)

    def _build_spouse_map(self):
        """Build spouse mapping from explicit pairs and shared children."""
        # From explicit spouse pairs
        for p1, p2 in self.spouse_pairs:
            self.spouse_map[p1] = p2
            self.spouse_map[p2] = p1

        # From shared children
        for child, parents in self.parents_of.items():
            parents = list(parents)
            if len(parents) == 2:
                self.spouse_map[parents[0]] = parents[1]
                self.spouse_map[parents[1]] = parents[0]

    def find_spouse(self, pid: str) -> str | None:
        """Get the spouse of a person."""
        return self.spouse_map.get(pid)

    def is_inlaw_spouse(self, pid: str) -> bool:
        """Check if person is an in-law (no parents, but has spouse who has parents)."""
        if pid in self.parents_of:
            return False
        spouse = self.find_spouse(pid)
        if spouse and spouse in self.parents_of:
            return True
        return False

    def get_children(self, parent_ids: list) -> list:
        """Get sorted children of given parents."""
        kids = set()
        for pid in parent_ids:
            if pid:
                kids.update(self.children_of.get(pid, set()))

        def sort_key(k):
            p = self.people.get(k, {})
            order = p.get("birth_order", 999)
            return (order, p.get("name", ""))

        return sorted(kids, key=sort_key)

    def calc_width(self, person_id: str, placed: set) -> int:
        """Calculate the width needed for a person and their descendants."""
        if person_id in placed:
            return 0

        spouse = self.find_spouse(person_id)
        unit = [person_id]
        if spouse and spouse not in placed:
            unit.append(spouse)

        unit_width = len(unit) * NODE_W + (len(unit) - 1) * COUPLE_GAP
        children = [c for c in self.get_children(unit) if c not in placed]

        if not children:
            return unit_width

        child_width = 0
        temp = placed | set(unit)
        for c in children:
            child_width += self.calc_width(c, temp) + H_GAP
            temp.add(c)
            sp = self.find_spouse(c)
            if sp:
                temp.add(sp)

        return max(unit_width, child_width - H_GAP)

    def layout(self, person_id: str, cx: float, y: float, placed: set):
        """Recursively layout a person and their descendants."""
        if person_id in placed:
            return

        spouse = self.find_spouse(person_id)
        unit = [person_id]
        if spouse and spouse not in placed:
            unit.append(spouse)

        unit_w = len(unit) * NODE_W + (len(unit) - 1) * COUPLE_GAP
        x = cx - unit_w / 2

        for i, pid in enumerate(unit):
            self.positions[pid] = (x + i * (NODE_W + COUPLE_GAP) + NODE_W / 2, y)
            placed.add(pid)

        if len(unit) == 2:
            self.connections.append(('spouse', self.positions[unit[0]][0], self.positions[unit[1]][0], y + 40))

        children = [c for c in self.get_children(unit) if c not in placed]
        if children:
            widths = [self.calc_width(c, placed.copy()) for c in children]
            total = sum(widths) + H_GAP * (len(children) - 1)
            parent_cx = sum(self.positions[p][0] for p in unit) / len(unit)

            child_x = parent_cx - total / 2
            child_centers = []
            for c, w in zip(children, widths):
                child_centers.append((c, child_x + w / 2))
                child_x += w + H_GAP

            cy = y + V_GAP + NODE_H
            self.connections.append(('parent', parent_cx, y + NODE_H - 20, cy - 30, [(ccx, cy) for _, ccx in child_centers]))

            for c, ccx in child_centers:
                self.layout(c, ccx, cy, placed)

    def compute(self) -> tuple:
        """Compute the full layout and return positions, connections, dimensions."""
        # Find true roots
        true_roots = [pid for pid in self.people if pid not in self.parents_of and not self.is_inlaw_spouse(pid)]

        # Group roots into couples
        root_units = []
        used = set()
        for r in true_roots:
            if r in used:
                continue
            spouse = self.find_spouse(r)
            if spouse and spouse in true_roots and spouse not in used:
                root_units.append([r, spouse])
                used.add(r)
                used.add(spouse)
            else:
                root_units.append([r])
                used.add(r)

        # Layout all roots
        placed = set()
        x = 100
        for unit in root_units:
            w = self.calc_width(unit[0], placed)
            self.layout(unit[0], x + w / 2, 30, placed)
            x += w + H_GAP * 2

        # Calculate dimensions
        if self.positions:
            max_x = max(p[0] for p in self.positions.values()) + NODE_W
            max_y = max(p[1] for p in self.positions.values()) + NODE_H + 50
        else:
            max_x, max_y = 1000, 600

        width = max(1600, max_x + 200)
        height = max(900, max_y + 100)

        return self.positions, self.connections, width, height


# =============================================================================
# SVG Generation
# =============================================================================

def build_connection_lines(connections: list) -> list:
    """Build SVG elements for connection lines."""
    lines_svg = []
    for conn in connections:
        if conn[0] == 'spouse':
            _, x1, x2, y = conn
            lines_svg.append(f'<line x1="{x1+25}" y1="{y}" x2="{x2-25}" y2="{y}" stroke="#cbd5e1" stroke-width="3"/>')
            lines_svg.append(f'<text x="{(x1+x2)/2}" y="{y+5}" text-anchor="middle" font-size="14">❤️</text>')
        elif conn[0] == 'parent':
            _, px, py, my, kids = conn
            lines_svg.append(f'<line x1="{px}" y1="{py}" x2="{px}" y2="{my}" stroke="#cbd5e1" stroke-width="3"/>')
            if kids:
                xs = [k[0] for k in kids]
                lines_svg.append(f'<line x1="{min(xs)}" y1="{my}" x2="{max(xs)}" y2="{my}" stroke="#cbd5e1" stroke-width="3"/>')
                for kx, ky in kids:
                    lines_svg.append(f'<line x1="{kx}" y1="{my}" x2="{kx}" y2="{ky}" stroke="#cbd5e1" stroke-width="3"/>')
    return lines_svg


def build_node_svg(pid: str, x: float, y: float, person: dict, color: str) -> str:
    """Build SVG element for a single person node."""
    name = html_module.escape(person.get("name", "Unknown"))
    short_name = name if len(name) <= 14 else name[:12] + "..."
    birth = person.get("birth_year") or ""
    death = person.get("death_year") or ""
    years = f"{birth}-{death if death else 'Present'}" if birth else ""
    img = image_to_data_uri(resolve_image_path(person.get("image_path"), person))
    esc_pid = html_module.escape(pid, quote=True)

    nx, ny = x - NODE_W / 2, y
    return f'''
    <g class="node" data-id="{esc_pid}" transform="translate({nx},{ny})">
        <rect width="{NODE_W}" height="{NODE_H-10}" rx="12" fill="white" stroke="{color}" stroke-width="3"/>
        <defs><clipPath id="c{esc_pid}"><circle cx="{NODE_W/2}" cy="45" r="35"/></clipPath></defs>
        <g class="photo-area">
            <image href="{img}" x="{NODE_W/2-35}" y="10" width="70" height="70" clip-path="url(#c{esc_pid})" preserveAspectRatio="xMidYMid slice"/>
            <circle cx="{NODE_W/2}" cy="45" r="35" fill="none" stroke="{color}" stroke-width="2"/>
        </g>
        <text x="{NODE_W/2}" y="100" text-anchor="middle" font-size="11" font-weight="600" fill="#1e293b">{short_name}</text>
        <text x="{NODE_W/2}" y="115" text-anchor="middle" font-size="9" fill="#64748b">{years}</text>
    </g>'''


def build_svg(positions: dict, connections: list, people: dict, width: int, height: int) -> str:
    """Build the complete SVG element."""
    lines_svg = build_connection_lines(connections)
    nodes_svg = []
    for i, (pid, (x, y)) in enumerate(positions.items()):
        person = people.get(pid, {})
        color = COLORS[i % len(COLORS)]
        nodes_svg.append(build_node_svg(pid, x, y, person, color))

    return f'''<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<defs><filter id="shadow"><feDropShadow dx="0" dy="2" stdDeviation="4" flood-color="rgba(0,0,0,0.08)"/></filter></defs>
<g filter="url(#shadow)">{chr(10).join(lines_svg)}</g>
<g>{chr(10).join(nodes_svg)}</g>
</svg>'''


# =============================================================================
# HTML Generation
# =============================================================================

def build_tree_html(data: dict) -> str:
    """Build the complete HTML for the family tree visualization."""
    people = data.get("people", {})

    if not people:
        return load_template("empty.html")

    # Compute layout
    layout = FamilyTreeLayout(data)
    positions, connections, width, height = layout.compute()

    # Build SVG
    svg_content = build_svg(positions, connections, people, width, height)

    # Load template and static files
    template = load_template("tree.html")
    css_content = load_static_file("css/styles.css")
    js_content = load_static_file("js/main.js")

    # Inline CSS and JS (since Streamlit iframe can't load external files)
    html = template.replace(
        '<link rel="stylesheet" href="{{CSS_URL}}">',
        f'<style>\n{css_content}\n</style>'
    )
    html = html.replace(
        '<script src="{{JS_URL}}"></script>',
        f'<script>\n{js_content}\n</script>'
    )
    html = html.replace('{{SVG_CONTENT}}', svg_content)

    return html


# =============================================================================
# Streamlit Application
# =============================================================================

def handle_add_person(data: dict, query_params: dict) -> bool:
    """Handle adding a new person from query parameters. Returns True if data was modified."""
    if query_params.get("action") != "add":
        return False

    name = query_params.get("name", "").strip()
    if not name:
        return False

    people = data.get("people", {})
    pid = uuid.uuid4().hex[:8]

    # Create new person
    data["people"][pid] = {
        "name": name,
        "gender": query_params.get("gender", "male"),
        "image_path": None,
        "birth_year": query_params.get("birth", "").strip() or None,
        "death_year": query_params.get("death", "").strip() or None,
    }

    # Handle relationships
    rel = query_params.get("relation", "root")
    target = query_params.get("target")

    if rel == "child" and target and target in people:
        # Add edge from target parent
        add_edge(data, target, pid)

        # Find and add edge from spouse
        spouse = _find_spouse(data, target, people)
        if spouse and spouse in people:
            add_edge(data, spouse, pid)

    elif rel == "parent" and target and target in people:
        add_edge(data, pid, target)

    elif rel == "spouse" and target and target in people:
        add_spouse(data, target, pid)

    save_data(data)
    return True


def _find_spouse(data: dict, target: str, people: dict) -> str | None:
    """Find the spouse of a target person."""
    # Check explicit spouses list
    for sp in data.get("spouses", []):
        if target in sp:
            return sp[0] if sp[1] == target else sp[1]

    # Check via shared children
    for edge in data.get("edges", []):
        if edge[0] == target:
            child_id = edge[1]
            for other_edge in data.get("edges", []):
                if other_edge[1] == child_id and other_edge[0] != target:
                    return other_edge[0]

    return None


@st.dialog("Add Person")
def add_person_dialog(data: dict, people: dict):
    """Dialog for adding a new person."""
    name = st.text_input("Name *", placeholder="Full name")

    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["male", "female"])
    with col2:
        birth = st.text_input("Birth Year", placeholder="e.g. 1980")

    death = st.text_input("Death Year", placeholder="Leave blank if alive")

    if people:
        people_opts = {"": "-- New root (no relation) --"}
        people_opts.update({pid: p.get("name", pid) for pid, p in people.items()})
        keys = list(people_opts.keys())

        related_to = st.selectbox("Related to", options=keys, format_func=lambda x: people_opts[x])

        if related_to:
            relation = st.selectbox("Relationship type", ["child", "parent", "spouse"],
                                   help="child = new person is child of selected\nparent = new person is parent of selected\nspouse = married to selected")
        else:
            relation = None
    else:
        related_to = None
        relation = None

    col3, col4 = st.columns(2)
    with col3:
        if st.button("➕ Add Person", use_container_width=True, type="primary"):
            if name.strip():
                pid = uuid.uuid4().hex[:8]
                person_data = {
                    "name": name.strip(),
                    "gender": gender,
                    "image_path": None,
                    "birth_year": birth.strip() or None,
                    "death_year": death.strip() or None,
                }

                client = get_supabase_client()
                if client:
                    # Save to Supabase
                    save_person_to_supabase(client, pid, person_data)
                    if relation and related_to and related_to in people:
                        if relation == "child":
                            add_edge_to_supabase(client, related_to, pid)
                            spouse = _find_spouse(data, related_to, people)
                            if spouse and spouse in people:
                                add_edge_to_supabase(client, spouse, pid)
                        elif relation == "parent":
                            add_edge_to_supabase(client, pid, related_to)
                        elif relation == "spouse":
                            add_spouse_to_supabase(client, related_to, pid)
                else:
                    # Fall back to JSON
                    data["people"][pid] = person_data
                    if relation and related_to and related_to in people:
                        if relation == "child":
                            add_edge(data, related_to, pid)
                            spouse = _find_spouse(data, related_to, people)
                            if spouse and spouse in people:
                                add_edge(data, spouse, pid)
                        elif relation == "parent":
                            add_edge(data, pid, related_to)
                        elif relation == "spouse":
                            add_spouse(data, related_to, pid)
                    save_data(data)
                st.rerun()
            else:
                st.error("Name is required")
    with col4:
        if st.button("Cancel", use_container_width=True):
            st.rerun()


@st.dialog("Edit Person")
def edit_person_dialog(data: dict, people: dict, preselected_id: str = None):
    """Dialog for editing an existing person."""
    if not people:
        st.info("No people to edit. Add someone first!")
        if st.button("Close"):
            st.rerun()
        return

    people_opts = {pid: p.get("name", pid) for pid, p in people.items()}
    keys = list(people_opts.keys())

    # Pre-select person if specified
    default_idx = 0
    if preselected_id and preselected_id in keys:
        default_idx = keys.index(preselected_id)

    edit_id = st.selectbox("Select person to edit", options=keys, format_func=lambda x: people_opts[x], index=default_idx)

    if edit_id:
        person = people[edit_id]

        edit_name = st.text_input("Name *", value=person.get("name", ""))

        col1, col2 = st.columns(2)
        with col1:
            g_opts = ["male", "female"]
            g_idx = g_opts.index(person.get("gender", "male")) if person.get("gender") in g_opts else 0
            edit_gender = st.selectbox("Gender", g_opts, index=g_idx)
        with col2:
            edit_birth = st.text_input("Birth Year", value=person.get("birth_year") or "")

        edit_death = st.text_input("Death Year", value=person.get("death_year") or "")

        col3, col4, col5 = st.columns(3)
        with col3:
            if st.button("💾 Save", use_container_width=True, type="primary"):
                if edit_name.strip():
                    person_data = {
                        "name": edit_name.strip(),
                        "gender": edit_gender,
                        "image_path": person.get("image_path"),
                        "birth_year": edit_birth.strip() or None,
                        "death_year": edit_death.strip() or None,
                        "birth_order": person.get("birth_order"),
                    }

                    client = get_supabase_client()
                    if client:
                        save_person_to_supabase(client, edit_id, person_data)
                    else:
                        data["people"][edit_id].update(person_data)
                        save_data(data)
                    st.rerun()
                else:
                    st.error("Name is required")
        with col4:
            if st.button("🗑️ Delete", use_container_width=True):
                client = get_supabase_client()
                if client:
                    delete_person_from_supabase(client, edit_id)
                else:
                    del data["people"][edit_id]
                    data["edges"] = [e for e in data["edges"] if edit_id not in e]
                    data["spouses"] = [s for s in data.get("spouses", []) if edit_id not in s]
                    save_data(data)
                st.rerun()
        with col5:
            if st.button("Cancel", use_container_width=True):
                st.rerun()


def edit_person_dialog_with_target(data: dict, people: dict, target_id: str):
    """Wrapper to call edit dialog with a pre-selected person."""
    edit_person_dialog(data, people, target_id)


def main() -> None:
    """Main application entry point."""
    st.set_page_config(
        page_title="Family Tree",
        page_icon="🌳",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # Custom CSS for clean layout with unified header
    st.markdown("""
        <style>
        #MainMenu,footer,header,[data-testid=stToolbar]{visibility:hidden;height:0}
        .block-container{padding:0!important;max-width:100%!important}
        [data-testid=stAppViewContainer]{padding:0!important}
        iframe{border:none!important}

        /* Hide sidebar completely */
        [data-testid="stSidebar"] {
            display: none !important;
        }

        /* Style the unified header bar */
        .stMainBlockContainer [data-testid="stHorizontalBlock"]:first-of-type {
            background: linear-gradient(135deg, #1e293b 0%, #334155 100%) !important;
            padding: 12px 24px !important;
            margin: 0 !important;
            gap: 10px !important;
            align-items: center !important;
            display: flex !important;
            flex-wrap: nowrap !important;
        }
        .stMainBlockContainer [data-testid="stHorizontalBlock"]:first-of-type [data-testid="column"] {
            padding: 0 !important;
            display: flex !important;
            align-items: center !important;
            justify-content: flex-start !important;
            min-height: auto !important;
        }
        /* Remove extra spacing from column content */
        .stMainBlockContainer [data-testid="stHorizontalBlock"]:first-of-type [data-testid="column"] > div {
            width: auto !important;
        }
        /* Title button styling - looks like a title but clickable */
        .stMainBlockContainer [data-testid="stHorizontalBlock"]:first-of-type [data-testid="column"]:first-child button {
            background: transparent !important;
            border: none !important;
            color: white !important;
            font-size: 20px !important;
            font-weight: 600 !important;
            height: auto !important;
            min-height: auto !important;
            max-height: none !important;
            padding: 0 !important;
            margin: 0 !important;
            cursor: pointer !important;
            white-space: nowrap !important;
        }
        .stMainBlockContainer [data-testid="stHorizontalBlock"]:first-of-type [data-testid="column"]:first-child button:hover {
            opacity: 0.8 !important;
        }
        /* Add/Edit button styling - 1:5 ratio (40px x 200px) */
        .stMainBlockContainer [data-testid="stHorizontalBlock"]:first-of-type [data-testid="column"]:not(:first-child) button {
            box-sizing: border-box !important;
            height: 40px !important;
            min-height: 40px !important;
            max-height: 40px !important;
            width: 200px !important;
            min-width: 200px !important;
            max-width: 200px !important;
            padding: 0 !important;
            font-size: 18px !important;
            border-radius: 10px !important;
            font-weight: 500 !important;
            line-height: 40px !important;
            margin: 0 !important;
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            white-space: nowrap !important;
        }
        .stMainBlockContainer [data-testid="stHorizontalBlock"]:first-of-type button[data-testid="baseButton-primary"] {
            background: #f87171 !important;
            border: none !important;
        }
        .stMainBlockContainer [data-testid="stHorizontalBlock"]:first-of-type [data-testid="column"]:not(:first-child) button[data-testid="baseButton-secondary"] {
            background: rgba(30, 41, 59, 0.8) !important;
            color: white !important;
            border: 1px solid rgba(255,255,255,0.3) !important;
        }
        /* Button columns align right */
        .stMainBlockContainer [data-testid="stHorizontalBlock"]:first-of-type [data-testid="column"]:nth-child(3),
        .stMainBlockContainer [data-testid="stHorizontalBlock"]:first-of-type [data-testid="column"]:nth-child(4) {
            justify-content: flex-end !important;
        }
        /* Remove default Streamlit element spacing */
        .stMainBlockContainer [data-testid="stHorizontalBlock"]:first-of-type .stMarkdown {
            margin: 0 !important;
            padding: 0 !important;
        }
        .stMainBlockContainer [data-testid="stHorizontalBlock"]:first-of-type .stButton {
            margin: 0 !important;
            padding: 0 !important;
        }

        /* Dialog overlay - light background */
        [data-testid="stModal"] {
            background: rgba(241, 245, 249, 0.85) !important;
        }
        [data-testid="stModal"] > div,
        [data-testid="stModal"] > div > div {
            position: static !important;
            display: contents !important;
        }
        [data-testid="stDialog"] {
            position: fixed !important;
            top: 50% !important;
            left: 50% !important;
            transform: translate(-50%, -50%) !important;
            max-width: 420px !important;
            width: 90vw !important;
            max-height: 85vh !important;
            overflow-y: auto !important;
            box-shadow: 0 8px 32px rgba(0,0,0,0.12) !important;
            border-radius: 12px !important;
            background: white !important;
            margin: 0 !important;
        }
        [data-testid="stDialog"] [data-testid="stVerticalBlock"] {
            gap: 1rem !important;
        }
        </style>
    """, unsafe_allow_html=True)

    ensure_dirs()
    ensure_placeholder()
    data = load_data()
    people = data.get("people", {})

    # Check for dialog triggers from URL query params
    query_params = st.query_params
    mode = query_params.get("mode", None)
    target = query_params.get("target", None)
    dialog = query_params.get("dialog", None)

    if dialog == "add" or mode == "add":
        st.query_params.clear()
        add_person_dialog(data, people)
    elif mode == "edit" and target:
        st.query_params.clear()
        edit_person_dialog_with_target(data, people, target)

    # Mobile/Desktop toggle in session state
    if "mobile_view" not in st.session_state:
        st.session_state.mobile_view = False

    # Unified header with title and buttons
    if st.session_state.mobile_view:
        # Mobile layout - stacked
        title_col, add_col, edit_col = st.columns([0.5, 0.25, 0.25])
        with title_col:
            if st.button("🌳 Family Tree 📱", key="title_toggle", help="Switch to Desktop view"):
                st.session_state.mobile_view = False
                st.rerun()
        with add_col:
            if st.button("Add", key="add_btn", type="primary"):
                add_person_dialog(data, people)
        with edit_col:
            if st.button("Edit", key="edit_btn"):
                edit_person_dialog(data, people)
    else:
        # Desktop layout - title left, buttons right
        title_col, spacer, add_col, edit_col = st.columns([0.15, 0.49, 0.18, 0.18])
        with title_col:
            if st.button("🌳 Family Tree 🖥️", key="title_toggle", help="Switch to Mobile view"):
                st.session_state.mobile_view = True
                st.rerun()
        with add_col:
            if st.button("Add", key="add_btn", type="primary"):
                add_person_dialog(data, people)
        with edit_col:
            if st.button("Edit", key="edit_btn"):
                edit_person_dialog(data, people)

    # Render the tree using Streamlit components (iframe header hidden via CSS)
    tree_html = build_tree_html(data)
    components.html(tree_html, height=850, scrolling=True)


if __name__ == "__main__":
    main()

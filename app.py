"""
Family Tree Application - Backend
A Streamlit-based family tree visualization application.
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


def load_data() -> dict:
    """Load family data from JSON file."""
    ensure_dirs()
    if not DATA_PATH.exists():
        return {"people": {}, "edges": [], "spouses": []}
    with DATA_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    data.setdefault("people", {})
    data.setdefault("edges", [])
    data.setdefault("spouses", [])
    return data


def save_data(data: dict) -> None:
    """Save family data to JSON file."""
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with DATA_PATH.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


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
        <image href="{img}" x="{NODE_W/2-35}" y="10" width="70" height="70" clip-path="url(#c{esc_pid})" preserveAspectRatio="xMidYMid slice"/>
        <circle cx="{NODE_W/2}" cy="45" r="35" fill="none" stroke="{color}" stroke-width="2"/>
        <text x="{NODE_W/2}" y="100" text-anchor="middle" font-size="11" font-weight="600" fill="#1e293b">{short_name}</text>
        <text x="{NODE_W/2}" y="115" text-anchor="middle" font-size="9" fill="#64748b">{years}</text>
        <circle class="add-btn" cx="{NODE_W-8}" cy="12" r="10" fill="{color}" onclick="openAddForm('{esc_pid}')"/>
        <text x="{NODE_W-8}" y="16" text-anchor="middle" fill="white" font-size="14" font-weight="bold" style="pointer-events:none">+</text>
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


def main() -> None:
    """Main application entry point."""
    st.set_page_config(
        page_title="Family Tree",
        page_icon="🌳",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # Custom CSS for layout
    st.markdown("""
        <style>
        #MainMenu,footer,header,[data-testid=stToolbar]{visibility:hidden;height:0}
        .block-container{padding:0!important;max-width:100%!important}
        [data-testid=stAppViewContainer]{padding:0!important}
        iframe{border:none!important}
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background: white;
            padding-top: 1rem;
        }
        [data-testid="stSidebar"] .block-container {
            padding: 1rem !important;
        }
        </style>
    """, unsafe_allow_html=True)

    ensure_dirs()
    ensure_placeholder()
    data = load_data()
    people = data.get("people", {})

    # Check if we have a target person from query params (clicked + on a node)
    qp = dict(st.query_params)
    target_id = qp.get("target")
    target_name = people.get(target_id, {}).get("name", "") if target_id else ""

    # Sidebar form for adding people (opens when + is clicked on a node)
    with st.sidebar:
        if target_name:
            st.header(f"➕ Add Relative to {target_name}")
        else:
            st.header("➕ Add Person")
        
        with st.form("add_person_form", clear_on_submit=True):
            name = st.text_input("Name *", placeholder="Full name")
            
            col1, col2 = st.columns(2)
            with col1:
                gender = st.selectbox("Gender", ["male", "female"])
            with col2:
                # If target is set, default to "child" relation
                relation_options = ["child", "parent", "spouse"] if target_id else ["root", "child", "parent", "spouse"]
                relation = st.selectbox("Relation", relation_options)
            
            # Target person selector (for relationships) - pre-select if target_id is set
            target = target_id
            if relation != "root" and people and not target_id:
                people_options = {pid: p.get("name", pid) for pid, p in people.items()}
                target = st.selectbox(
                    "Related to",
                    options=list(people_options.keys()),
                    format_func=lambda x: people_options[x]
                )
            
            col3, col4 = st.columns(2)
            with col3:
                birth = st.text_input("Birth Year", placeholder="e.g. 1980")
            with col4:
                death = st.text_input("Death Year", placeholder="Leave blank if alive")
            
            submitted = st.form_submit_button("Add Person", use_container_width=True, type="primary")
            
            if submitted and name.strip():
                pid = uuid.uuid4().hex[:8]
                data["people"][pid] = {
                    "name": name.strip(),
                    "gender": gender,
                    "image_path": None,
                    "birth_year": birth.strip() or None,
                    "death_year": death.strip() or None,
                }
                
                if relation == "child" and target and target in people:
                    add_edge(data, target, pid)
                    spouse = _find_spouse(data, target, people)
                    if spouse and spouse in people:
                        add_edge(data, spouse, pid)
                elif relation == "parent" and target and target in people:
                    add_edge(data, pid, target)
                elif relation == "spouse" and target and target in people:
                    add_spouse(data, target, pid)
                
                save_data(data)
                st.query_params.clear()
                st.rerun()
            elif submitted and not name.strip():
                st.error("Name is required")

    # Render the tree
    html = build_tree_html(data)
    components.html(html, height=950, scrolling=False)


if __name__ == "__main__":
    main()

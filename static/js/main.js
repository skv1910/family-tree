// Family Tree JavaScript

// DOM Elements
var canvas = document.getElementById('canvas');
var inner = document.getElementById('inner');

// State
var scale = 1;

/**
 * Update the transform scale of the canvas
 */
function updateTransform() {
    inner.style.transform = 'scale(' + scale + ')';
    document.getElementById('zl').textContent = Math.round(scale * 100) + '%';
}

/**
 * Zoom in by 10%
 */
function zoomIn() {
    scale = Math.min(2, scale + 0.1);
    updateTransform();
}

/**
 * Zoom out by 10%
 */
function zoomOut() {
    scale = Math.max(0.3, scale - 0.1);
    updateTransform();
}

/**
 * Reset view to default scale and scroll position
 */
function resetView() {
    scale = 1;
    updateTransform();
    canvas.scrollTo(0, 0);
}

/**
 * Open the add form for a specific person
 * This sets a query parameter and triggers Streamlit to open the sidebar
 * @param {string} personId - The ID of the person to add relative to
 */
function openAddForm(personId) {
    // Set query parameter to pass the target person to Streamlit
    var params = new URLSearchParams();
    params.set('target', personId);
    
    // Navigate to open sidebar with target pre-selected
    // Use postMessage to communicate with Streamlit parent
    try {
        // First, try to open the sidebar
        var sidebarButton = window.parent.document.querySelector('[data-testid="collapsedControl"]');
        if (sidebarButton) {
            sidebarButton.click();
        }
        
        // Then update the URL with target parameter
        var newUrl = window.parent.location.pathname + '?' + params.toString();
        window.parent.history.pushState({}, '', newUrl);
        
        // Trigger Streamlit rerun
        window.parent.postMessage({type: 'streamlit:rerun'}, '*');
        
        // Fallback: reload the parent page
        setTimeout(function() {
            window.parent.location.href = newUrl;
        }, 100);
    } catch (e) {
        // Cross-origin fallback - just reload with params
        alert('Opening add form for this person. The sidebar will open.');
        window.parent.location.href = window.parent.location.pathname + '?' + params.toString();
    }
}

// Keyboard shortcuts
document.addEventListener('keydown', function(e) {
    if (e.key === '+' || e.key === '=') {
        zoomIn();
    } else if (e.key === '-') {
        zoomOut();
    } else if (e.key === '0') {
        resetView();
    }
});

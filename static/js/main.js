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
 * Navigate to a URL - handles Streamlit iframe context
 * @param {string} url - URL to navigate to
 */
function goToUrl(url) {
    // Get the base URL from the parent window
    var baseUrl = '';
    try {
        baseUrl = window.parent.location.origin + window.parent.location.pathname.split('?')[0];
    } catch(e) {
        try {
            baseUrl = window.top.location.origin + window.top.location.pathname.split('?')[0];
        } catch(e2) {
            baseUrl = '';
        }
    }

    var fullUrl = baseUrl + url;

    // Try parent window navigation
    try {
        window.parent.location.href = fullUrl;
        return;
    } catch(e) {
        console.log('parent.location failed:', e);
    }

    // Try top window
    try {
        window.top.location.href = fullUrl;
        return;
    } catch(e) {
        console.log('top.location failed:', e);
    }

    // Last resort - current window
    window.location.href = fullUrl;
}

/**
 * Open the edit form for a person (when clicking on photo)
 * @param {string} personId - The ID of the person to edit
 */
function openEditForm(personId) {
    goToUrl('/?mode=edit&target=' + encodeURIComponent(personId));
}

/**
 * Open the add form to add a relative (when clicking + button)
 * @param {string} personId - The ID of the person to add relative to
 */
function openAddForm(personId) {
    goToUrl('/?mode=add&target=' + encodeURIComponent(personId));
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

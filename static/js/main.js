// Family Tree JavaScript

// DOM Elements
var canvas = document.getElementById('canvas');
var inner = document.getElementById('inner');

// State
var scale = 1;
var selectedId = null;

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
 * Open the add person modal
 * @param {string|null} pid - Person ID to add relative to, or null for new root
 */
function openModal(pid) {
    selectedId = pid;
    document.getElementById('mtitle').textContent = pid ? 'Add Relative' : 'Add Person';
    document.getElementById('frel').value = pid ? 'child' : 'root';
    document.getElementById('modal').classList.add('open');
    document.getElementById('fname').focus();
}

/**
 * Close the modal and reset form
 */
function closeModal() {
    document.getElementById('modal').classList.remove('open');
    document.getElementById('fname').value = '';
    document.getElementById('fbirth').value = '';
    document.getElementById('fdeath').value = '';
}

/**
 * Submit the form to add a new person
 */
function submit() {
    var name = document.getElementById('fname').value.trim();
    if (!name) {
        alert('Name is required');
        return;
    }

    var params = new URLSearchParams();
    params.set('action', 'add');
    params.set('name', name);
    params.set('gender', document.getElementById('fgender').value);
    params.set('relation', document.getElementById('frel').value);

    if (selectedId) {
        params.set('target', selectedId);
    }

    var birth = document.getElementById('fbirth').value.trim();
    var death = document.getElementById('fdeath').value.trim();
    if (birth) params.set('birth', birth);
    if (death) params.set('death', death);

    // Navigate to trigger Streamlit rerun with query params
    // Use top-level window to handle both local and deployed Streamlit
    try {
        var targetWindow = window.top || window.parent;
        var baseUrl = targetWindow.location.origin + targetWindow.location.pathname;
        targetWindow.location.href = baseUrl + '?' + params.toString();
    } catch (e) {
        // Fallback for cross-origin restrictions
        window.parent.location.href = window.parent.location.pathname + '?' + params.toString();
    }
}

// Event Listeners
document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape') {
        closeModal();
    }
    if (e.key === 'Enter' && document.getElementById('modal').classList.contains('open')) {
        submit();
    }
});

// Close modal when clicking outside
document.getElementById('modal').addEventListener('click', function(e) {
    if (e.target === this) {
        closeModal();
    }
});

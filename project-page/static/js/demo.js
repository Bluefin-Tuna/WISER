// ============================================================================
// Configuration Constants
// ============================================================================
const API_BASE_URL = 'https://semantic-world-models.com'; // Backend API URL
const MIN_POINT_DISTANCE = 12; // Minimum pixel distance between points
const MAX_POINT_DISTANCE = 16; // Maximum pixel distance between points (interpolate if exceeded)
const MAX_POINTS = 20; // Maximum number of points allowed
const POINT_RADIUS = 4; // Radius of drawn points
const LINE_WIDTH = 3; // Width of drawn lines
const POINT_BORDER_WIDTH = 2; // Width of point borders
const PEG_REGION_RADIUS = 20; // Radius of valid starting region around peg
const COLORS = {
    primary: '#6366f1',
    success: '#10b981',
    danger: '#ef4444',
    white: 'white',
    pegRegion: 'rgba(16, 185, 129, 0.3)',
    pegRegionBorder: '#10b981'
};

// ============================================================================
// State Management
// ============================================================================
let currentPairLoaded = false;
let sessionId = null; // Current session ID from backend
let drawingPaths = [];
let currentPath = [];
let canvas = null;
let ctx = null;
let isDrawing = false;
let pegLocation = null; // Peg location in image coordinates
let imageDimensions = null; // Original image dimensions from backend

// ============================================================================
// DOM Elements
// ============================================================================
const elements = {
    randomPairBtn: document.getElementById('randomPairBtn'),
    questionInput: document.getElementById('questionInput'),
    askBtn: document.getElementById('askBtn'),
    initialImageWrapper: document.getElementById('initialImageWrapper'),
    finalImageWrapper: document.getElementById('finalImageWrapper'),
    sampleQuestions: document.getElementById('sampleQuestions'),
    answerSection: document.getElementById('answerSection'),
    loadingOverlay: document.getElementById('loadingOverlay'),
    drawingControls: document.getElementById('drawingControls'),
    clearDrawingBtn: document.getElementById('clearDrawingBtn'),
    submitActionsBtn: document.getElementById('submitActionsBtn'),
    pointsCounter: document.getElementById('pointsCounter'),
    finalStateBox: document.querySelector('.final-state-box')
};

// ============================================================================
// Event Listeners
// ============================================================================
elements.randomPairBtn.addEventListener('click', loadRandomPair);
elements.askBtn.addEventListener('click', askQuestion);
elements.clearDrawingBtn.addEventListener('click', clearDrawing);
elements.submitActionsBtn.addEventListener('click', submitActions);
elements.questionInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !elements.askBtn.disabled) {
        askQuestion();
    }
});

// ============================================================================
// Utility Functions
// ============================================================================
function showLoading() {
    elements.loadingOverlay.classList.add('active');
}

function hideLoading() {
    elements.loadingOverlay.classList.remove('active');
}

function showError(message) {
    alert('Error: ' + message);
}

function getTotalPoints() {
    return drawingPaths.reduce((sum, path) => sum + path.length, 0) + currentPath.length;
}

function calculateDistance(point1, point2) {
    return Math.sqrt(
        Math.pow(point1.x - point2.x, 2) + 
        Math.pow(point1.y - point2.y, 2)
    );
}

function getCanvasCoordinates(e, rect) {
    // Convert from display coordinates to canvas internal coordinates
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    return {
        x: (e.clientX - rect.left) * scaleX,
        y: (e.clientY - rect.top) * scaleY
    };
}

function getImageCoordinates(canvasPoint) {
    // Canvas coordinates are already in image pixel space
    // (since we set canvas.width/height to match image dimensions)
    return {
        x: Math.round(canvasPoint.x),
        y: Math.round(canvasPoint.y)
    };
}

function getAllDrawnPoints() {
    // Get all points from all paths in image coordinates
    const allPoints = [];
    drawingPaths.forEach(path => {
        path.forEach(point => {
            allPoints.push(getImageCoordinates(point));
        });
    });
    // Include current path being drawn
    currentPath.forEach(point => {
        allPoints.push(getImageCoordinates(point));
    });
    return allPoints;
}

// ============================================================================
// API Functions
// ============================================================================

async function loadRandomPair() {
    showLoading();
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/random_pair`, {
            credentials: 'include'
        });
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'Failed to load random pair');
        }
        
        // Store session ID
        sessionId = data.session_id;
        console.log('Session ID:', sessionId);
        
        // Update initial image
        elements.initialImageWrapper.innerHTML = `<img src="${data.initial_image}" alt="Initial state" id="initialImage">`;
        
        // Wait for image to load before setting up canvas
        const img = document.getElementById('initialImage');
        img.onload = () => setupDrawingCanvas(img);
        
        // Hide final state section until future is generated
        elements.finalStateBox.style.display = 'none';
        
        // Store peg location and image dimensions
        pegLocation = data.peg_location;
        imageDimensions = data.image_dimensions;
        
        console.log('Image dimensions:', imageDimensions);
        console.log('Peg location (image coords):', pegLocation);
        
        // Clear sample questions
        elements.sampleQuestions.innerHTML = '<div class="placeholder-text">Generate future state to see questions</div>';
        
        // Disable question input until future is generated
        elements.questionInput.disabled = true;
        elements.askBtn.disabled = true;
        elements.questionInput.value = '';
        elements.questionInput.placeholder = 'Generate future state first';
        
        // Hide answer section
        elements.answerSection.style.display = 'none';
        
        currentPairLoaded = true;
        
    } catch (error) {
        showError(error.message);
    } finally {
        hideLoading();
    }
}

function populateSampleQuestions(sampleQuestions) {
    elements.sampleQuestions.innerHTML = '';
    
    if (!sampleQuestions || sampleQuestions.length === 0) return;
    
    // Use first sample question as placeholder
    elements.questionInput.placeholder = 'e.g., ' + sampleQuestions[0].question;
    
    // Create toggle button
    const toggleBtn = document.createElement('div');
    toggleBtn.className = 'sample-questions-toggle';
    toggleBtn.innerHTML = `
        <span>Sample questions (click to use)</span>
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M6 9l6 6 6-6"/>
        </svg>
    `;
    
    // Create content container
    const content = document.createElement('div');
    content.className = 'sample-questions-content';
    
    // Add question items to content
    sampleQuestions.forEach(item => {
        const questionDiv = createSampleQuestionElement(item);
        content.appendChild(questionDiv);
    });
    
    // Add toggle functionality
    toggleBtn.addEventListener('click', () => {
        toggleBtn.classList.toggle('collapsed');
        content.classList.toggle('collapsed');
    });
    
    elements.sampleQuestions.appendChild(toggleBtn);
    elements.sampleQuestions.appendChild(content);
}

function createSampleQuestionElement(item) {
    const questionDiv = document.createElement('div');
    questionDiv.className = 'sample-question';
    
    const questionText = document.createElement('span');
    questionText.className = 'question-text';
    questionText.textContent = item.question;
    
    questionDiv.appendChild(questionText);
    
    questionDiv.addEventListener('click', () => {
        elements.questionInput.value = item.question;
        elements.questionInput.focus();
    });
    
    return questionDiv;
}

async function askQuestion() {
    let question = elements.questionInput.value.trim();
    
    // If empty, use the placeholder text as the example question
    if (!question) {
        const placeholderText = elements.questionInput.placeholder;
        question = placeholderText.replace(/^e\.g\.,\s*/i, '').trim();
        elements.questionInput.value = question;
    }
    
    if (!currentPairLoaded || !sessionId) {
        showError('Please load a data pair first');
        return;
    }
    
    showLoading();
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/ask_question`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            credentials: 'include',
            body: JSON.stringify({ 
                session_id: sessionId,
                question: question 
            })
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.detail || data.error || 'Failed to get answer');
        }
        
        // Collapse sample questions after first question is asked
        const toggleBtn = elements.sampleQuestions.querySelector('.sample-questions-toggle');
        const content = elements.sampleQuestions.querySelector('.sample-questions-content');
        if (toggleBtn && content && !toggleBtn.classList.contains('collapsed')) {
            toggleBtn.classList.add('collapsed');
            content.classList.add('collapsed');
        }
        
        // Display answer
        displayAnswer(data);
        
    } catch (error) {
        showError(error.message);
    } finally {
        hideLoading();
    }
}

function displayAnswer(data) {
    const answerResult = document.getElementById('answerResult');
    const confidenceFill = document.getElementById('confidenceFill');
    const confidenceMarker = document.getElementById('confidenceMarker');
    const initialLine = document.getElementById('initialLine');
    const confidenceText = document.getElementById('confidenceText');
    const yesProb = document.getElementById('yesProb');
    const noProb = document.getElementById('noProb');
    
    // Set answer
    answerResult.textContent = data.answer;
    answerResult.className = 'answer-result ' + data.answer.toLowerCase();
    
    // Calculate positions (0 = No, 100 = Yes)
    const yesPercentage = data.yes_probability * 100;
    const initialYesPercentage = data.yes_probability_initial * 100;
    
    // Update confidence bar to span between initial and final states
    const minPercentage = Math.min(yesPercentage, initialYesPercentage);
    const maxPercentage = Math.max(yesPercentage, initialYesPercentage);
    const fillWidth = maxPercentage - minPercentage;
    
    confidenceFill.style.left = minPercentage + '%';
    confidenceFill.style.width = fillWidth + '%';
    confidenceMarker.style.left = yesPercentage + '%';
    
    // Update initial state red line
    initialLine.style.left = initialYesPercentage + '%';
    
    // Update confidence text
    confidenceText.textContent = `Confidence: ${(data.confidence * 100).toFixed(1)}%`;
    
    // Update probabilities
    yesProb.textContent = (data.yes_probability * 100).toFixed(2) + '%';
    noProb.textContent = (data.no_probability * 100).toFixed(2) + '%';
    
    // Show answer section with animation
    elements.answerSection.style.display = 'block';
    elements.answerSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// ============================================================================
// Drawing Functions
// ============================================================================
function setupDrawingCanvas(img) {
    // Remove existing canvas if any
    const existingCanvas = document.getElementById('drawingCanvas');
    if (existingCanvas) {
        existingCanvas.remove();
    }
    
    // Clear drawing state
    drawingPaths = [];
    currentPath = [];
    updatePointsCounter();
    
    // Create canvas with original image dimensions as internal resolution
    canvas = document.createElement('canvas');
    canvas.id = 'drawingCanvas';
    
    // Set canvas internal resolution to match original image dimensions
    if (imageDimensions) {
        canvas.width = imageDimensions.width;
        canvas.height = imageDimensions.height;
    } else {
        // Fallback to natural image dimensions
        canvas.width = img.naturalWidth || img.width;
        canvas.height = img.naturalHeight || img.height;
    }
    
    console.log('Canvas internal resolution:', canvas.width, 'x', canvas.height);
    console.log('Image natural dimensions:', img.naturalWidth, 'x', img.naturalHeight);
    
    // Position canvas over image (styled to 100% to match display size)
    canvas.style.position = 'absolute';
    canvas.style.top = '0';
    canvas.style.left = '0';
    canvas.style.width = '100%';
    canvas.style.height = '100%';
    canvas.style.cursor = 'not-allowed';
    canvas.style.pointerEvents = 'auto';
    
    // Get context
    ctx = canvas.getContext('2d');
    
    // Add canvas to wrapper
    elements.initialImageWrapper.style.position = 'relative';
    elements.initialImageWrapper.appendChild(canvas);
    
    // Show drawing controls
    elements.drawingControls.style.display = 'flex';
    
    // Add event listeners for drawing
    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', handleMouseMove);
    canvas.addEventListener('mouseup', stopDrawing);
    canvas.addEventListener('mouseleave', stopDrawing);
    
    // Touch events for mobile
    canvas.addEventListener('touchstart', handleTouchStart);
    canvas.addEventListener('touchmove', handleTouchMove);
    canvas.addEventListener('touchend', stopDrawing);
    
    // Draw initial peg region
    redrawCanvas();
}

function startDrawing(e) {
    const point = getCanvasPoint(e);
    
    // Check if starting point is within peg region
    if (!isPointInPegRegion(point)) {
        return;
    }
    
    isDrawing = true;
    // Clear all previous paths and start a new one
    drawingPaths = [];
    currentPath = [];
    addPoint(point);
}

function handleMouseMove(e) {
    const point = getCanvasPoint(e);
    
    // Update cursor based on whether we're in the peg region
    if (!isDrawing) {
        canvas.style.cursor = isPointInPegRegion(point) ? 'crosshair' : 'not-allowed';
    }
    
    // Continue drawing if mouse is down
    if (isDrawing) {
        addPoint(point);
    }
}

function stopDrawing() {
    if (isDrawing && currentPath.length > 0) {
        // Save the current path to the paths array
        drawingPaths.push([...currentPath]);
        currentPath = [];
    }
    isDrawing = false;
}

function handleTouchStart(e) {
    e.preventDefault();
    const point = getTouchPoint(e);
    
    // Check if starting point is within peg region
    if (!isPointInPegRegion(point)) {
        // showError('Please start drawing from the green peg region');
        return;
    }
    
    isDrawing = true;
    // Clear all previous paths and start a new one
    drawingPaths = [];
    currentPath = [];
    addPoint(point);
}

function handleTouchMove(e) {
    e.preventDefault();
    if (!isDrawing) return;
    const point = getTouchPoint(e);
    addPoint(point);
}

function getCanvasPoint(e) {
    const rect = canvas.getBoundingClientRect();
    return getCanvasCoordinates(e, rect);
}

function getTouchPoint(e) {
    const rect = canvas.getBoundingClientRect();
    return getCanvasCoordinates(e.touches[0], rect);
}

function addPoint(point) {
    // Check if we've reached the maximum number of points
    if (getTotalPoints() >= MAX_POINTS) return;
    
    // Check if this point needs interpolation or is valid
    if (currentPath.length > 0) {
        const lastPoint = currentPath[currentPath.length - 1];
        const distance = calculateDistance(point, lastPoint);
        
        // Too close - skip this point
        if (distance < MIN_POINT_DISTANCE) {
            return;
        }
        
        // Too far - interpolate intermediate points
        if (distance > MAX_POINT_DISTANCE) {
            const numIntermediatePoints = Math.floor(distance / MAX_POINT_DISTANCE);
            
            // Add interpolated points
            for (let i = 1; i <= numIntermediatePoints; i++) {
                if (getTotalPoints() >= MAX_POINTS) break;
                
                const t = i / (numIntermediatePoints + 1);
                const interpolatedPoint = {
                    x: lastPoint.x + (point.x - lastPoint.x) * t,
                    y: lastPoint.y + (point.y - lastPoint.y) * t
                };
                currentPath.push(interpolatedPoint);
            }
        }
    }
    
    // Add the actual point (if we haven't exceeded max)
    if (getTotalPoints() < MAX_POINTS) {
        currentPath.push(point);
        
        // Log first point for debugging
        if (currentPath.length === 1) {
            console.log('First point (canvas coords):', point);
            console.log('First point (image coords):', getImageCoordinates(point));
        }
    }
    
    redrawCanvas();
    updatePointsCounter();
}

function drawPath(path) {
    if (path.length === 0) return;
    
    // Set drawing style
    ctx.strokeStyle = COLORS.primary;
    ctx.lineWidth = LINE_WIDTH;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    
    // Draw lines
    ctx.beginPath();
    ctx.moveTo(path[0].x, path[0].y);
    for (let i = 1; i < path.length; i++) {
        ctx.lineTo(path[i].x, path[i].y);
    }
    ctx.stroke();
    
    // Draw points
    path.forEach((point, index) => {
        drawPoint(point, index === 0);
    });
}

function drawPoint(point, isStartPoint = false) {
    // Fill point
    ctx.fillStyle = isStartPoint ? COLORS.success : COLORS.primary;
    ctx.beginPath();
    ctx.arc(point.x, point.y, POINT_RADIUS, 0, Math.PI * 2);
    ctx.fill();
    
    // Draw border
    ctx.strokeStyle = COLORS.white;
    ctx.lineWidth = POINT_BORDER_WIDTH;
    ctx.stroke();
}

function redrawCanvas() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw peg region indicator
    drawPegRegion();
    
    // Draw all completed paths
    drawingPaths.forEach(path => drawPath(path));
    
    // Draw current path being drawn
    if (currentPath.length > 0) {
        drawPath(currentPath);
    }
}

function drawPegRegion() {
    if (!pegLocation) return;
    
    const pegX = pegLocation.x;
    const pegY = pegLocation.y;
    
    // Draw center dot (2x larger)
    ctx.fillStyle = COLORS.success;
    ctx.beginPath();
    ctx.arc(pegX, pegY, 8, 0, Math.PI * 2);
    ctx.fill();
}

function isPointInPegRegion(point) {
    if (!pegLocation) return true; // Allow drawing if no peg location set
    
    const distance = calculateDistance(point, pegLocation);
    return distance <= PEG_REGION_RADIUS;
}

function clearDrawing() {
    drawingPaths = [];
    currentPath = [];
    if (ctx && canvas) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
    updatePointsCounter();
    redrawCanvas();
}

// Export function for getting drawn points (useful for debugging and later API calls)
window.getDrawnPoints = getAllDrawnPoints;

function updatePointsCounter() {
    const totalPoints = getTotalPoints();
    const isMaxReached = totalPoints >= MAX_POINTS;
    
    elements.pointsCounter.textContent = `Actions: ${totalPoints} / ${MAX_POINTS}`;
    elements.pointsCounter.style.color = isMaxReached ? COLORS.danger : '';
    elements.pointsCounter.style.fontWeight = isMaxReached ? '700' : '';
    
    // Enable/disable submit button based on whether there are points
    elements.submitActionsBtn.disabled = totalPoints < 2;
}

async function submitActions() {
    if (!currentPairLoaded || !sessionId) {
        showError('Please load a data pair first');
        return;
    }
    
    const drawnPoints = getAllDrawnPoints();
    
    if (drawnPoints.length < 2) {
        showError('Please draw at least 2 points');
        return;
    }
    
    showLoading();
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/submit_actions`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            credentials: 'include',
            body: JSON.stringify({ 
                session_id: sessionId,
                points: drawnPoints 
            })
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.detail || data.error || 'Failed to generate future state');
        }
        
        // Show the final state section
        elements.finalStateBox.style.display = 'block';
        
        // Update the final image
        elements.finalImageWrapper.innerHTML = `<img src="${data.final_image}" alt="Generated final state">`;
        
        // Update sample questions with new questions from environment
        populateSampleQuestions(data.sample_questions);
        
        // Enable question input now that future is generated
        elements.questionInput.disabled = false;
        elements.askBtn.disabled = false;
        elements.questionInput.placeholder = 'e.g., ' + data.sample_questions[0].question;
        elements.questionInput.focus();
        
        console.log('Generated future state with', data.num_actions, 'actions');
        
    } catch (error) {
        showError(error.message);
    } finally {
        hideLoading();
    }
}

// ============================================================================
// Instructions Toggle
// ============================================================================
const instructionsToggle = document.getElementById('instructionsToggle');
const instructionsContent = document.getElementById('instructionsContent');

if (instructionsToggle && instructionsContent) {
    instructionsToggle.addEventListener('click', () => {
        const isExpanded = instructionsContent.classList.contains('expanded');
        
        if (isExpanded) {
            instructionsContent.classList.remove('expanded');
            instructionsToggle.classList.remove('active');
        } else {
            instructionsContent.classList.add('expanded');
            instructionsToggle.classList.add('active');
        }
    });
}

// ============================================================================
// Initialization
// ============================================================================
console.log('Semantic World Modeling Demo - Ready!');


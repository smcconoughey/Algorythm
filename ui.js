// UI-related functions

// Canvas and context references
const canvas = document.getElementById('network-canvas');
const ctx = canvas.getContext('2d');
const chartCanvas = document.getElementById('accuracy-chart');
const chartCtx = chartCanvas ? chartCanvas.getContext('2d') : null;

// UI state
let particles = [];
let isDeployingToken = false;
let activePowerUps = {};
let accuracyHistory = [];
let lossHistory = [];

// Calculate node positions for visualization
function calculateNodePositions() {
    if (!canvas || !canvas.width) return;
    
    const padding = 80; // Padding from edges
    
    // Calculate layer spacing
    const layerSpacing = (canvas.width - 2 * padding) / (network.layers.length - 1);
    
    // Calculate node positions for each layer
    for (let l = 0; l < network.layers.length; l++) {
        const layer = network.layers[l];
        const nodeCount = layer.size;
        
        // Calculate vertical spacing for nodes
        const nodeSpacing = (canvas.height - 2 * padding) / (nodeCount - 1 || 1);
        const positions = [];
        
        // Calculate position for each node
        for (let i = 0; i < nodeCount; i++) {
            positions.push({
                x: padding + l * layerSpacing,
                y: nodeCount > 1 ? padding + i * nodeSpacing : canvas.height / 2
            });
        }
        
        layer.positions = positions;
    }
}

// Draw the neural network on canvas
function drawNetwork() {
    if (!canvas || !canvas.width || !canvas.height) return;
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    calculateNodePositions();
    
    // Draw connections between nodes
    for (let l = 0; l < network.layers.length - 1; l++) {
        const currentLayer = network.layers[l];
        const nextLayer = network.layers[l + 1];
        
        for (let i = 0; i < currentLayer.size; i++) {
            for (let j = 0; j < nextLayer.size; j++) {
                const sourceNode = currentLayer.positions[i];
                const targetNode = nextLayer.positions[j];
                const weight = network.weights[l][i][j];
                
                // Determine line width based on weight
                const absWeight = Math.abs(weight);
                ctx.lineWidth = absWeight * 3 + 0.5;
                
                // Determine color based on weight sign
                const alpha = Math.min(absWeight + 0.2, 1);
                if (weight >= 0) {
                    ctx.strokeStyle = `rgba(79, 195, 247, ${alpha})`;
                } else {
                    ctx.strokeStyle = `rgba(255, 87, 34, ${alpha})`;
                }
                
                // Draw connection line
                ctx.beginPath();
                ctx.moveTo(sourceNode.x, sourceNode.y);
                ctx.lineTo(targetNode.x, targetNode.y);
                ctx.stroke();
            }
        }
    }
    
    // Draw nodes
    for (let l = 0; l < network.layers.length; l++) {
        const layer = network.layers[l];
        
        for (let i = 0; i < layer.size; i++) {
            const position = layer.positions[i];
            const activation = network.activations[l][i];
            const isFocused = network.nodeFocus[l][i];
            
            // Node base appearance
            ctx.beginPath();
            ctx.arc(position.x, position.y, 15, 0, Math.PI * 2);
            
            // Fill based on activation level
            const activationLevel = Math.max(0.1, activation);
            ctx.fillStyle = isFocused ? 
                `rgba(124, 179, 66, ${activationLevel})` : 
                `rgba(79, 195, 247, ${activationLevel})`;
            ctx.fill();
            
            // Node border
            ctx.lineWidth = 2;
            ctx.strokeStyle = isFocused ? '#8bc34a' : '#29b6f6';
            ctx.stroke();
            
            // Node label
            ctx.fillStyle = '#ffffff';
            ctx.font = '10px Arial';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            
            // Label with layer and node index
            const layerNames = ['I', 'H', 'H', 'O']; // Input, Hidden, Output
            const layerName = l === 0 ? 'I' : (l === network.layers.length - 1 ? 'O' : 'H');
            const label = `${layerName}${i+1}`;
            ctx.fillText(label, position.x, position.y);
        }
    }
    
    // Draw particles
    particles.forEach((particle, index) => {
        ctx.beginPath();
        ctx.arc(particle.x, particle.y, particle.size, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(79, 195, 247, ${particle.opacity})`;
        ctx.fill();
        
        // Update particle position
        particle.x += particle.vx;
        particle.y += particle.vy;
        particle.opacity -= 0.01;
        particle.size *= 0.98;
        
        // Remove faded particles
        if (particle.opacity <= 0 || particle.size <= 0.5) {
            particles.splice(index, 1);
        }
    });
}

// Create activation particles
function createParticles(sourceX, sourceY, targetX, targetY, count) {
    for (let i = 0; i < count; i++) {
        const t = i / count;
        const x = sourceX + (targetX - sourceX) * t + (Math.random() - 0.5) * 5;
        const y = sourceY + (targetY - sourceY) * t + (Math.random() - 0.5) * 5;
        
        particles.push({
            x: x,
            y: y,
            size: 2 + Math.random() * 3,
            opacity: 0.7 + Math.random() * 0.3,
            vx: (targetX - sourceX) * 0.01 + (Math.random() - 0.5) * 0.5,
            vy: (targetY - sourceY) * 0.01 + (Math.random() - 0.5) * 0.5
        });
    }
}

// Update accuracy chart
function updateAccuracyChart() {
    if (!chartCanvas || !chartCtx || !chartCanvas.width || !chartCanvas.height) return;
    
    // Clear chart
    chartCtx.clearRect(0, 0, chartCanvas.width, chartCanvas.height);
    
    // Draw background
    chartCtx.fillStyle = 'rgba(0, 0, 0, 0.3)';
    chartCtx.fillRect(0, 0, chartCanvas.width, chartCanvas.height);
    
    // Draw target accuracy line
    chartCtx.beginPath();
    chartCtx.moveTo(0, chartCanvas.height * (1 - targetAccuracy / 100));
    chartCtx.lineTo(chartCanvas.width, chartCanvas.height * (1 - targetAccuracy / 100));
    chartCtx.strokeStyle = 'rgba(255, 235, 59, 0.5)';
    chartCtx.lineWidth = 1;
    chartCtx.stroke();
    
    // Draw accuracy line
    if (accuracyHistory.length > 1) {
        chartCtx.beginPath();
        
        // Start point
        chartCtx.moveTo(0, chartCanvas.height * (1 - accuracyHistory[0] / 100));
        
        // Plot points
        for (let i = 1; i < accuracyHistory.length; i++) {
            const x = (i / (accuracyHistory.length - 1)) * chartCanvas.width;
            const y = chartCanvas.height * (1 - accuracyHistory[i] / 100);
            chartCtx.lineTo(x, y);
        }
        
        chartCtx.strokeStyle = '#4fc3f7';
        chartCtx.lineWidth = 2;
        chartCtx.stroke();
    }
}

// Update stats panel
function updateStatsPanel() {
    // Skip if elements don't exist
    if (!document.getElementById('stat-learning-rate')) return;
    
    // Update stat values
    document.getElementById('stat-learning-rate').textContent = 
        document.getElementById('learning-rate') ? document.getElementById('learning-rate').value : '0.1';
    
    document.getElementById('stat-batch-size').textContent = 
        document.getElementById('batch-size') ? document.getElementById('batch-size').value : '16';
    
    // Calculate average gradient magnitude
    let totalGradient = 0;
    let gradientCount = 0;
    
    for (let l = 0; l < network.gradients.weights.length; l++) {
        for (let i = 0; i < network.gradients.weights[l].length; i++) {
            for (let j = 0; j < network.gradients.weights[l][i].length; j++) {
                totalGradient += Math.abs(network.gradients.weights[l][i][j]);
                gradientCount++;
            }
        }
    }
    
    const avgGradient = gradientCount > 0 ? totalGradient / gradientCount : 0;
    document.getElementById('stat-gradient').textContent = avgGradient.toFixed(4);
    
    // Calculate average weight magnitude
    let totalWeight = 0;
    let weightCount = 0;
    
    for (let l = 0; l < network.weights.length; l++) {
        for (let i = 0; i < network.weights[l].length; i++) {
            for (let j = 0; j < network.weights[l][i].length; j++) {
                totalWeight += Math.abs(network.weights[l][i][j]);
                weightCount++;
            }
        }
    }
    
    const avgWeight = weightCount > 0 ? totalWeight / weightCount : 0;
    document.getElementById('stat-weight-avg').textContent = avgWeight.toFixed(4);
    
    // Update accuracy chart
    updateAccuracyChart();
}

// Generate mini-network visualization for level cards
function generateMiniNetworkHTML(layerSizes) {
    let html = '';
    
    for (let i = 0; i < layerSizes.length; i++) {
        const nodeCount = layerSizes[i];
        
        // Create node group
        html += `<div class="node-group">`;
        
        // Add nodes (limit display to 5 max for visual clarity)
        const displayCount = Math.min(nodeCount, 5);
        for (let j = 0; j < displayCount; j++) {
            html += `<div class="mini-node"></div>`;
        }
        
        // Add indicator for more nodes if needed
        if (nodeCount > 5) {
            html += `<div style="font-size: 0.7rem; color: #a0a0a0;">+${nodeCount - 5}</div>`;
        }
        
        html += `</div>`;
        
        // Add connections between layers
        if (i < layerSizes.length - 1) {
            html += `<div class="connections">
                <svg width="20" height="40">
                    <line x1="0" y1="20" x2="20" y2="20" stroke="#4fc3f7" stroke-width="1" />
                </svg>
            </div>`;
        }
    }
    
    return html;
}

// Initialize parameter controls based on level
function initializeControls() {
    const currentLevel = gameData.levels[gameData.player.currentLevel];
    const controlsContainer = document.getElementById('parameter-controls');
    if (!controlsContainer) return;
    
    controlsContainer.innerHTML = '';
    
    currentLevel.availableControls.forEach(control => {
        const controlGroup = document.createElement('div');
        controlGroup.className = 'control-group';
        
        const controlLabel = document.createElement('div');
        controlLabel.className = 'control-label';
        controlLabel.innerHTML = control.name + 
            `<span class="info-icon" data-concept="${control.id}">ⓘ</span>`;
        
        controlGroup.appendChild(controlLabel);
        
        const sliderContainer = document.createElement('div');
        sliderContainer.className = 'slider-container';
        
        if (control.type === 'range') {
            const slider = document.createElement('input');
            slider.type = 'range';
            slider.id = control.id;
            slider.min = control.min;
            slider.max = control.max;
            slider.step = control.step;
            slider.value = control.value;
            
            const valueDisplay = document.createElement('span');
            valueDisplay.id = control.id + '-value';
            valueDisplay.textContent = control.value;
            
            slider.addEventListener('input', (e) => {
                valueDisplay.textContent = parseFloat(e.target.value).toFixed(
                    control.step < 0.01 ? 4 : (control.step < 0.1 ? 2 : 1)
                );
            });
            
            sliderContainer.appendChild(slider);
            sliderContainer.appendChild(valueDisplay);
        } else if (control.type === 'select') {
            const select = document.createElement('select');
            select.id = control.id;
            
            control.options.forEach(option => {
                const optionElement = document.createElement('option');
                optionElement.value = option;
                optionElement.textContent = option.charAt(0).toUpperCase() + option.slice(1);
                if (option === control.value) {
                    optionElement.selected = true;
                }
                select.appendChild(optionElement);
            });
            
            sliderContainer.appendChild(select);
        }
        
        controlGroup.appendChild(sliderContainer);
        controlsContainer.appendChild(controlGroup);
    });
    
    // Add event listeners for info buttons
    document.querySelectorAll('.info-icon').forEach(icon => {
        icon.addEventListener('click', (e) => {
            e.stopPropagation();
            const conceptId = e.target.getAttribute('data-concept');
            showConceptInfo(conceptId);
        });
    });
}

// Show concept information popup
function showConceptInfo(conceptId) {
    let conceptKey = conceptId;
    
    // Map control IDs to concept keys
    const conceptMap = {
        'learning-rate': 'learningRate',
        'activation-fn': 'activationFunctions',
        'batch-size': 'batchSize',
        'dropout-rate': 'dropout',
        'l2-regularization': 'l2Regularization',
        'optimizer': 'optimizers',
        'adversarial-training': 'adversarialTraining'
    };
    
    if (conceptMap[conceptId]) {
        conceptKey = conceptMap[conceptId];
    }
    
    const concept = gameData.concepts[conceptKey];
    
    if (concept) {
        document.getElementById('concept-title').textContent = concept.title;
        document.getElementById('concept-content').innerHTML = concept.content;
        
        const popup = document.getElementById('concept-popup');
        
        // Position near the clicked icon if possible
        const icon = document.querySelector(`[data-concept="${conceptId}"]`);
        if (icon) {
            const rect = icon.getBoundingClientRect();
            popup.style.top = (rect.bottom + 10) + 'px';
            popup.style.left = rect.left + 'px';
        }
        
        popup.style.display = 'block';
        
        // Close when clicking elsewhere
        document.addEventListener('click', closeConceptOnClick);
    }
}

// Close concept popup when clicking outside
function closeConceptOnClick(e) {
    const popup = document.getElementById('concept-popup');
    const closeButton = document.getElementById('close-concept');
    
    if (!popup.contains(e.target) || e.target === closeButton) {
        popup.style.display = 'none';
        document.removeEventListener('click', closeConceptOnClick);
    }
}

// Visualize training process
function visualizeTraining() {
    // Create particles between nodes to visualize activations
    for (let l = 0; l < network.layers.length - 1; l++) {
        const currentLayer = network.layers[l];
        const nextLayer = network.layers[l + 1];
        
        for (let i = 0; i < currentLayer.size; i++) {
            for (let j = 0; j < nextLayer.size; j++) {
                const weight = network.weights[l][i][j];
                const activation = network.activations[l][i];
                
                // Only visualize significant connections
                if (Math.abs(weight) > 0.3 && activation > 0.3) {
                    const sourceNode = currentLayer.positions[i];
                    const targetNode = nextLayer.positions[j];
                    
                    // Number of particles based on connection strength
                    const particleCount = Math.ceil(Math.abs(weight) * 5);
                    
                    if (Math.random() > 0.7) { // Randomize to avoid too many particles
                        createParticles(
                            sourceNode.x, sourceNode.y, 
                            targetNode.x, targetNode.y, 
                            particleCount
                        );
                    }
                }
            }
        }
    }
}

// Update power-ups display
function updatePowerUps() {
    const container = document.getElementById('power-up-container');
    if (!container) return;
    
    container.innerHTML = '';
    
    // Add power-up buttons for each available power-up
    Object.keys(gameData.player.powerUps).forEach(powerUpId => {
        const count = gameData.player.powerUps[powerUpId];
        
        if (count > 0) {
            const powerUp = gameData.powerUps[powerUpId];
            
            const button = document.createElement('button');
            button.className = 'power-up-button';
            button.id = `power-up-${powerUpId}`;
            button.innerHTML = `${powerUp.name} <span class="power-up-count">${count}</span>`;
            button.title = powerUp.description;
            
            button.addEventListener('click', () => activatePowerUp(powerUpId));
            
            container.appendChild(button);
        }
    });
}

// Activate a power-up
function activatePowerUp(powerUpId) {
    if (gameData.player.powerUps[powerUpId] > 0 && !activePowerUps[powerUpId]) {
        // Decrease count
        gameData.player.powerUps[powerUpId]--;
        
        // Mark as active
        activePowerUps[powerUpId] = true;
        
        // Visual effect
        const button = document.getElementById(`power-up-${powerUpId}`);
        if (button) {
            button.classList.add('pulse-animation');
            setTimeout(() => {
                button.classList.remove('pulse-animation');
            }, 1500);
        }
        
        // Update UI
        updatePowerUps();
        
        // Apply effect (actual implementation in training)
        const effect = gameData.powerUps[powerUpId].effect(network);
        
        // Show effect as tooltip
        const tooltip = document.getElementById('tooltip');
        if (tooltip) {
            tooltip.innerHTML = `<div><strong>${gameData.powerUps[powerUpId].name} Activated!</strong></div><div>${effect}</div>`;
            tooltip.style.display = 'block';
            tooltip.style.left = (canvas.width / 2) + 'px';
            tooltip.style.top = (canvas.height / 2) + 'px';
            
            // Hide tooltip after delay
            setTimeout(() => {
                tooltip.style.display = 'none';
            }, 3000);
        }
    }
}

// Initialize event listeners for UI components
function initUIEventListeners() {
    console.log('Initializing UI event listeners');

    // Close button for concept popup
    const closeConceptButton = document.getElementById('close-concept');
    if (closeConceptButton) {
        closeConceptButton.addEventListener('click', () => {
            document.getElementById('concept-popup').style.display = 'none';
            document.removeEventListener('click', closeConceptOnClick);
        });
    }
    
    // Toggle stats panel
    const toggleStatsButton = document.getElementById('toggle-stats');
    if (toggleStatsButton) {
        toggleStatsButton.addEventListener('click', () => {
            const statsPanel = document.getElementById('stats-panel');
            const isVisible = statsPanel.style.display === 'block';
            
            statsPanel.style.display = isVisible ? 'none' : 'block';
            toggleStatsButton.textContent = isVisible ? 'Show Stats' : 'Hide Stats';
        });
    }
    
    // Handle training token deployment
    const trainingTokenButton = document.getElementById('training-token');
    if (trainingTokenButton) {
        trainingTokenButton.addEventListener('click', () => {
            if (tokensLeft <= 0) {
                alert('No training tokens left!');
                return;
            }
            
            isDeployingToken = true;
            canvas.style.cursor = 'pointer';
        });
    }
    
    // Handle node selection for token deployment
    if (canvas) {
        canvas.addEventListener('click', (e) => {
            if (!isDeployingToken) return;
            
            const rect = canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            
            // Check if click is on a node
            let nodeFound = false;
            for (let l = 0; l < network.layers.length; l++) {
                const layer = network.layers[l];
                
                for (let i = 0; i < layer.size; i++) {
                    const node = layer.positions[i];
                    const distance = Math.sqrt(Math.pow(x - node.x, 2) + Math.pow(y - node.y, 2));
                    
                    if (distance <= 15) { // Node radius is 15
                        nodeFound = true;
                        
                        // Toggle focus state
                        if (network.nodeFocus[l][i]) {
                            network.nodeFocus[l][i] = false;
                            tokensLeft++; // Return token
                        } else {
                            network.nodeFocus[l][i] = true;
                            tokensLeft--; // Use token
                        }
                        
                        document.getElementById('tokens-value').textContent = tokensLeft;
                        break;
                    }
                }
                
                if (nodeFound) break;
            }
            
            if (!nodeFound) {
                isDeployingToken = false;
                canvas.style.cursor = 'default';
            }
        });
        
        // Tooltip functionality
        canvas.addEventListener('mousemove', (e) => {
            const tooltip = document.getElementById('tooltip');
            if (!tooltip) return;
            
            const rect = canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            
            // Check if mouse is over a node
            let tooltipContent = null;
            for (let l = 0; l < network.layers.length; l++) {
                const layer = network.layers[l];
                
                for (let i = 0; i < layer.size; i++) {
                    const node = layer.positions[i];
                    const distance = Math.sqrt(Math.pow(x - node.x, 2) + Math.pow(y - node.y, 2));
                    
                    if (distance <= 15) { // Node radius is 15
                        const activation = network.activations[l][i].toFixed(2);
                        const isFocused = network.nodeFocus[l][i] ? 'Yes' : 'No';
                        
                        tooltipContent = `
                            <div><strong>${layer.name} Node ${i+1}</strong></div>
                            <div>Activation: ${activation}</div>
                            <div>Training Focus: ${isFocused}</div>
                        `;
                        
                        tooltip.style.left = (e.clientX + 10) + 'px';
                        tooltip.style.top = (e.clientY + 10) + 'px';
                        break;
                    }
                }
                
                if (tooltipContent) break;
            }
            
            if (tooltipContent) {
                tooltip.innerHTML = tooltipContent;
                tooltip.style.display = 'block';
            } else {
                tooltip.style.display = 'none';
            }
        });
    }
    
    // Handle responsive canvas
    window.addEventListener('resize', () => {
        if (canvas) {
            const container = document.querySelector('.network-container');
            if (container) {
                canvas.width = container.offsetWidth;
                canvas.height = container.offsetHeight;
                
                if (chartCanvas) {
                    const chartContainer = document.querySelector('.chart-container');
                    if (chartContainer) {
                        chartCanvas.width = chartContainer.offsetWidth;
                        chartCanvas.height = chartContainer.offsetHeight;
                    }
                }
                
                drawNetwork();
                updateAccuracyChart();
            }
        }
    });
    
    // Add direct access to level 1 button
    const startLevel1Button = document.getElementById('start-level-1-btn');
    if (startLevel1Button) {
        startLevel1Button.addEventListener('click', () => {
            console.log('Start Level 1 button clicked');
            selectLevel(0);
        });
    }
    
    console.log('UI event listeners initialized');
}
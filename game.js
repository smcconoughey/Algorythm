// Main game logic

// Game state variables
let currentLevel;
let tokensLeft = 10;
let targetAccuracy = 85;
let levelComplete = false;
let isTraining = false;
let trainingInterval;
let frameId;

// Initialize level selection
function initLevelSelection() {
    const container = document.getElementById('level-select-container');
    if (!container) {
        console.error('Level select container not found');
        return;
    }
    
    container.innerHTML = '';
    console.log('Initializing level selection with', gameData.levels.length, 'levels');
    
    gameData.levels.forEach((level, index) => {
        const isUnlocked = gameData.player.unlocked.includes(level.id);
        const isCompleted = gameData.player.completedLevels.includes(level.id);
        
        const levelCard = document.createElement('div');
        levelCard.className = `level-card ${isCompleted ? 'completed' : ''} ${!isUnlocked ? 'locked' : ''}`;
        
        if (isUnlocked) {
            levelCard.onclick = () => selectLevel(level.id);
        }
        
        const networkStructure = level.networkStructure;
        
        levelCard.innerHTML = `
            <h4 class="level-card-title">${level.name}</h4>
            <div class="level-card-description">${level.description}</div>
            <div class="level-card-network">
                ${generateMiniNetworkHTML(networkStructure)}
            </div>
            <div class="level-card-details">
                <span>Target: ${level.targetAccuracy}%</span>
                <span>${isCompleted ? 'Completed ✓' : (isUnlocked ? 'Available' : 'Locked 🔒')}</span>
            </div>
        `;
        
        container.appendChild(levelCard);
        console.log('Added level card:', level.name);
    });
}

// Select level and initialize game
function selectLevel(levelId) {
    console.log('Selecting level', levelId);
    gameData.player.currentLevel = levelId;
    currentLevel = gameData.levels.find(level => level.id === levelId);
    
    document.getElementById('level-select-overlay').style.display = 'none';
    document.getElementById('game-view').style.display = 'flex';
    
    // Show tutorial for first-time level visit
    if (!gameData.player.completedLevels.includes(levelId)) {
        showTutorial(0); // Show first tutorial for this level
    } else {
        initializeLevel();
    }
}

// Show tutorial for current level
function showTutorial(tutorialIndex) {
    const tutorials = currentLevel.tutorials;
    
    if (tutorials && tutorials.length > 0 && tutorialIndex < tutorials.length) {
        const tutorial = tutorials[tutorialIndex];
        
        document.getElementById('tutorial-title').textContent = tutorial.title;
        document.getElementById('tutorial-content').innerHTML = tutorial.content;
        
        // Set up button for next tutorial or start game
        const nextButton = document.getElementById('tutorial-next');
        
        if (tutorialIndex < tutorials.length - 1) {
            nextButton.textContent = "Next";
            nextButton.onclick = () => showTutorial(tutorialIndex + 1);
        } else {
            nextButton.textContent = "Start Training";
            nextButton.onclick = () => {
                document.getElementById('tutorial-overlay').style.display = 'none';
                initializeLevel();
            };
        }
        
        document.getElementById('tutorial-overlay').style.display = 'flex';
    } else {
        document.getElementById('tutorial-overlay').style.display = 'none';
        initializeLevel();
    }
}

// Initialize level parameters and network
function initializeLevel() {
    console.log('Initializing level');
    // Reset game state
    levelComplete = false;
    tokensLeft = currentLevel.maxTokens;
    accuracy = 0;
    loss = 0;
    epoch = 0;
    accuracyHistory = [];
    lossHistory = [];
    activePowerUps = {};
    
    // Update UI
    document.getElementById('level-title').textContent = currentLevel.name;
    document.getElementById('level-description').textContent = currentLevel.description;
    document.getElementById('accuracy-value').textContent = accuracy + '%';
    document.getElementById('loss-value').textContent = loss.toFixed(2);
    document.getElementById('epoch-value').textContent = epoch;
    document.getElementById('tokens-value').textContent = tokensLeft;
    document.getElementById('target-value').textContent = currentLevel.targetAccuracy + '%';
    targetAccuracy = currentLevel.targetAccuracy;
    
    // Initialize control panel
    initializeControls();
    
    // Initialize network
    initializeNetwork();
    
    // Update power-ups
    updatePowerUps();
    
    // Generate training data
    generateTrainingData();
    
    // Initialize stats panel
    const statsPanel = document.getElementById('stats-panel');
    if (statsPanel) {
        statsPanel.style.display = 'none';
    }
    
    const toggleStats = document.getElementById('toggle-stats');
    if (toggleStats) {
        toggleStats.textContent = 'Show Stats';
    }
    
    updateStatsPanel();
    
    // Start animation loop
    startAnimation();
    
    console.log('Level initialized');
}

// Animation loop
function startAnimation() {
    console.log('Starting animation loop');
    if (frameId) {
        cancelAnimationFrame(frameId);
    }
    
    function animate() {
        drawNetwork();
        frameId = requestAnimationFrame(animate);
    }
    
    animate();
}

// Run single training step
function runTrainingStep() {
    if (levelComplete) return;
    
    // Get batch size
    const batchSizeInput = document.getElementById('batch-size');
    const batchSize = batchSizeInput ? parseInt(batchSizeInput.value) : 16;
    
    // Create batch
    const batch = [];
    for (let i = 0; i < batchSize; i++) {
        const idx = Math.floor(Math.random() * trainingData.length);
        batch.push(trainingData[idx]);
    }
    
    // Train on batch
    loss = trainOnBatch(batch);
    
    // Evaluate network
    const evaluation = evaluateNetwork();
    accuracy = evaluation.accuracy;
    
    // Visualize activations
    visualizeTraining();
    
    // Update UI
    document.getElementById('accuracy-value').textContent = Math.round(accuracy) + '%';
    document.getElementById('loss-value').textContent = loss.toFixed(2);
    document.getElementById('epoch-value').textContent = epoch;
    
    // Update history
    accuracyHistory.push(accuracy);
    lossHistory.push(loss);
    
    // Limit history length
    if (accuracyHistory.length > 20) {
        accuracyHistory.shift();
        lossHistory.shift();
    }
    
    // Update stats panel
    updateStatsPanel();
    
    // Check if level is complete
    if (accuracy >= targetAccuracy && !levelComplete) {
        levelComplete = true;
        showLevelComplete();
    }
    
    // Increment epoch
    epoch++;
}

// Start/stop training loop
function toggleTraining() {
    if (isTraining) {
        clearInterval(trainingInterval);
        document.getElementById('train-button').textContent = 'Run Training';
        isTraining = false;
    } else {
        trainingInterval = setInterval(runTrainingStep, 100);
        document.getElementById('train-button').textContent = 'Stop Training';
        isTraining = true;
    }
}

// Show level complete screen
function showLevelComplete() {
    // Stop training if running
    if (isTraining) {
        clearInterval(trainingInterval);
        isTraining = false;
        document.getElementById('train-button').textContent = 'Run Training';
    }
    
    // Add level to completed levels if not already there
    if (!gameData.player.completedLevels.includes(currentLevel.id)) {
        gameData.player.completedLevels.push(currentLevel.id);
        
        // Unlock next level if available
        const nextLevelId = currentLevel.id + 1;
        if (nextLevelId < gameData.levels.length && !gameData.player.unlocked.includes(nextLevelId)) {
            gameData.player.unlocked.push(nextLevelId);
        }
        
        // Add rewards
        const rewards = currentLevel.rewards;
        if (rewards) {
            if (rewards.tokens) {
                gameData.player.tokens += rewards.tokens;
            }
            
            if (rewards.powerUps) {
                for (const [powerUp, count] of Object.entries(rewards.powerUps)) {
                    gameData.player.powerUps[powerUp] += count;
                }
            }
        }
        
        // Display rewards
        const rewardsList = document.getElementById('rewards-list');
        rewardsList.innerHTML = '';
        
        if (rewards) {
            if (rewards.tokens) {
                rewardsList.innerHTML += `- ${rewards.tokens} Training Tokens<br>`;
            }
            
            if (rewards.powerUps) {
                for (const [powerUp, count] of Object.entries(rewards.powerUps)) {
                    const powerUpName = gameData.powerUps[powerUp].name;
                    rewardsList.innerHTML += `- ${count} ${powerUpName} Power-up<br>`;
                }
            }
        }
    }
    
    document.getElementById('level-complete-overlay').style.display = 'flex';
}

// Set up event listeners
function setupEventListeners() {
    console.log('Setting up game event listeners');
    
    // Training button
    const trainButton = document.getElementById('train-button');
    if (trainButton) {
        trainButton.addEventListener('click', toggleTraining);
    }
    
    // Reset button
    const resetButton = document.getElementById('reset-button');
    if (resetButton) {
        resetButton.addEventListener('click', () => {
            // Stop training if running
            if (isTraining) {
                clearInterval(trainingInterval);
                isTraining = false;
                document.getElementById('train-button').textContent = 'Run Training';
            }
            
            // Reset network and start fresh
            initializeLevel();
        });
    }
    
    // Level complete navigation
    const levelCompleteNext = document.getElementById('level-complete-next');
    if (levelCompleteNext) {
        levelCompleteNext.addEventListener('click', () => {
            document.getElementById('level-complete-overlay').style.display = 'none';
            
            // Go to next level if available
            const nextLevelId = currentLevel.id + 1;
            if (nextLevelId < gameData.levels.length) {
                selectLevel(nextLevelId);
            } else {
                // Show level selection if no next level
                document.getElementById('game-view').style.display = 'none';
                document.getElementById('level-select-overlay').style.display = 'flex';
                initLevelSelection();
            }
        });
    }
    
    // Menu button to go back to level selection
    const menuButton = document.getElementById('menu-button');
    if (menuButton) {
        menuButton.addEventListener('click', () => {
            // Stop training if running
            if (isTraining) {
                clearInterval(trainingInterval);
                isTraining = false;
            }
            
            document.getElementById('game-view').style.display = 'none';
            document.getElementById('level-select-overlay').style.display = 'flex';
            initLevelSelection();
        });
    }
    
    // Initialize UI-specific event listeners
    initUIEventListeners();
    
    console.log('Game event listeners set up');
}

// Initialize the game
function initGame() {
    // Add debug output
    console.log('Initializing game...');
    
    // Force level selection to be visible initially
    const levelSelectOverlay = document.getElementById('level-select-overlay');
    const gameView = document.getElementById('game-view');
    
    if (levelSelectOverlay && gameView) {
        levelSelectOverlay.style.display = 'flex';
        gameView.style.display = 'none';
    }
    
    // Set up event listeners
    setupEventListeners();
    
    // Initialize level selection
    setTimeout(() => {
        initLevelSelection();
        console.log('Level selection initialized');
    }, 100);
    
    // Debug: create a direct way to access level 0 if selection doesn't work
    console.log('Adding emergency level access');
    window.startLevel0 = function() {
        selectLevel(0);
    };
    console.log('Type window.startLevel0() in the console to start the first level directly if needed');
}

// Start the game when DOM is fully loaded
document.addEventListener('DOMContentLoaded', () => {
    console.log('DOM fully loaded, starting game initialization');
    initGame();
});
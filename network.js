// Neural network implementation

// Network state
let network = {
    layers: [],
    weights: [],
    biases: [],
    activations: [],
    preActivations: [],
    nodeFocus: [],
    gradients: {
        weights: [],
        biases: []
    },
    momentum: {
        weights: [],
        biases: []
    }
};

// Training data
let trainingData = [];
let validationData = [];
let accuracy = 0;
let loss = 0;
let epoch = 0;

// Activation functions
const activationFunctions = {
    sigmoid: {
        forward: function(x) {
            return 1 / (1 + Math.exp(-x));
        },
        backward: function(x) {
            const sigmoid = 1 / (1 + Math.exp(-x));
            return sigmoid * (1 - sigmoid);
        }
    },
    relu: {
        forward: function(x) {
            return Math.max(0, x);
        },
        backward: function(x) {
            return x > 0 ? 1 : 0;
        }
    },
    tanh: {
        forward: function(x) {
            return Math.tanh(x);
        },
        backward: function(x) {
            const tanh = Math.tanh(x);
            return 1 - tanh * tanh;
        }
    },
    leaky_relu: {
        forward: function(x) {
            return x > 0 ? x : 0.01 * x;
        },
        backward: function(x) {
            return x > 0 ? 1 : 0.01;
        }
    },
    elu: {
        forward: function(x) {
            return x > 0 ? x : Math.exp(x) - 1;
        },
        backward: function(x) {
            return x > 0 ? 1 : Math.exp(x);
        }
    }
};

// Optimizers
const optimizers = {
    sgd: function(weights, biases, gradWeights, gradBiases, learningRate) {
        // Simple SGD update
        for (let l = 0; l < weights.length; l++) {
            for (let i = 0; i < weights[l].length; i++) {
                for (let j = 0; j < weights[l][i].length; j++) {
                    weights[l][i][j] -= learningRate * gradWeights[l][i][j];
                }
            }
            
            for (let i = 0; i < biases[l].length; i++) {
                biases[l][i] -= learningRate * gradBiases[l][i];
            }
        }
    },
    
    adam: function(weights, biases, gradWeights, gradBiases, learningRate, network, t = 1) {
        const beta1 = 0.9;
        const beta2 = 0.999;
        const epsilon = 1e-8;
        
        // Initialize momentum and velocity if not already present
        if (!network.adam) {
            network.adam = {
                m_weights: [],
                v_weights: [],
                m_biases: [],
                v_biases: []
            };
            
            for (let l = 0; l < weights.length; l++) {
                const m_layer_weights = [];
                const v_layer_weights = [];
                
                for (let i = 0; i < weights[l].length; i++) {
                    const m_node_weights = new Array(weights[l][i].length).fill(0);
                    const v_node_weights = new Array(weights[l][i].length).fill(0);
                    
                    m_layer_weights.push(m_node_weights);
                    v_layer_weights.push(v_node_weights);
                }
                
                network.adam.m_weights.push(m_layer_weights);
                network.adam.v_weights.push(v_layer_weights);
                
                network.adam.m_biases.push(new Array(biases[l].length).fill(0));
                network.adam.v_biases.push(new Array(biases[l].length).fill(0));
            }
        }
        
        // Update with Adam
        for (let l = 0; l < weights.length; l++) {
            for (let i = 0; i < weights[l].length; i++) {
                for (let j = 0; j < weights[l][i].length; j++) {
                    // Update biased first moment and second moment estimates
                    network.adam.m_weights[l][i][j] = beta1 * network.adam.m_weights[l][i][j] + (1 - beta1) * gradWeights[l][i][j];
                    network.adam.v_weights[l][i][j] = beta2 * network.adam.v_weights[l][i][j] + (1 - beta2) * gradWeights[l][i][j] * gradWeights[l][i][j];
                    
                    // Compute bias-corrected first moment and second moment estimates
                    const m_corrected = network.adam.m_weights[l][i][j] / (1 - Math.pow(beta1, t));
                    const v_corrected = network.adam.v_weights[l][i][j] / (1 - Math.pow(beta2, t));
                    
                    // Update weight
                    weights[l][i][j] -= learningRate * m_corrected / (Math.sqrt(v_corrected) + epsilon);
                }
            }
            
            for (let i = 0; i < biases[l].length; i++) {
                // Update biased first moment and second moment estimates
                network.adam.m_biases[l][i] = beta1 * network.adam.m_biases[l][i] + (1 - beta1) * gradBiases[l][i];
                network.adam.v_biases[l][i] = beta2 * network.adam.v_biases[l][i] + (1 - beta2) * gradBiases[l][i] * gradBiases[l][i];
                
                // Compute bias-corrected first moment and second moment estimates
                const m_corrected = network.adam.m_biases[l][i] / (1 - Math.pow(beta1, t));
                const v_corrected = network.adam.v_biases[l][i] / (1 - Math.pow(beta2, t));
                
                // Update bias
                biases[l][i] -= learningRate * m_corrected / (Math.sqrt(v_corrected) + epsilon);
            }
        }
    },
    
    rmsprop: function(weights, biases, gradWeights, gradBiases, learningRate, network) {
        const decayRate = 0.9;
        const epsilon = 1e-8;
        
        // Initialize cache for RMSprop if not already present
        if (!network.rmsprop) {
            network.rmsprop = {
                cache_weights: [],
                cache_biases: []
            };
            
            for (let l = 0; l < weights.length; l++) {
                const cache_layer_weights = [];
                
                for (let i = 0; i < weights[l].length; i++) {
                    const cache_node_weights = new Array(weights[l][i].length).fill(0);
                    cache_layer_weights.push(cache_node_weights);
                }
                
                network.rmsprop.cache_weights.push(cache_layer_weights);
                network.rmsprop.cache_biases.push(new Array(biases[l].length).fill(0));
            }
        }
        
        // Update with RMSprop
        for (let l = 0; l < weights.length; l++) {
            for (let i = 0; i < weights[l].length; i++) {
                for (let j = 0; j < weights[l][i].length; j++) {
                    // Update cache
                    network.rmsprop.cache_weights[l][i][j] = decayRate * network.rmsprop.cache_weights[l][i][j] + 
                                                            (1 - decayRate) * gradWeights[l][i][j] * gradWeights[l][i][j];
                    
                    // Update weight
                    weights[l][i][j] -= learningRate * gradWeights[l][i][j] / 
                                        (Math.sqrt(network.rmsprop.cache_weights[l][i][j]) + epsilon);
                }
            }
            
            for (let i = 0; i < biases[l].length; i++) {
                // Update cache
                network.rmsprop.cache_biases[l][i] = decayRate * network.rmsprop.cache_biases[l][i] + 
                                                    (1 - decayRate) * gradBiases[l][i] * gradBiases[l][i];
                
                // Update bias
                biases[l][i] -= learningRate * gradBiases[l][i] / 
                                (Math.sqrt(network.rmsprop.cache_biases[l][i]) + epsilon);
            }
        }
    }
};

// Initialize network based on level configuration
function initializeNetwork() {
    const currentLevel = gameData.levels[gameData.player.currentLevel];
    const structure = currentLevel.networkStructure;
    
    // Initialize network structure
    network.layers = [];
    for (let i = 0; i < structure.length; i++) {
        network.layers.push({
            size: structure[i],
            positions: []
        });
    }
    
    // Initialize weights, biases, and activations
    network.weights = [];
    network.biases = [];
    network.activations = [];
    network.preActivations = [];
    network.nodeFocus = [];
    network.gradients.weights = [];
    network.gradients.biases = [];
    network.momentum.weights = [];
    network.momentum.biases = [];
    
    // Initialize all activations arrays
    for (let l = 0; l < network.layers.length; l++) {
        const layerSize = network.layers[l].size;
        
        // Initialize activations for this layer
        network.activations.push(new Array(layerSize).fill(0));
        network.preActivations.push(new Array(layerSize).fill(0));
        network.nodeFocus.push(new Array(layerSize).fill(false));
        
        // Initialize biases for all but input layer
        if (l > 0) {
            // Random biases between -0.1 and 0.1
            const layerBiases = Array.from({ length: layerSize }, () => Math.random() * 0.2 - 0.1);
            network.biases.push(layerBiases);
            
            // Initialize bias gradients
            network.gradients.biases.push(new Array(layerSize).fill(0));
            network.momentum.biases.push(new Array(layerSize).fill(0));
        }
    }
    
    // Initialize weights between layers
    for (let l = 0; l < network.layers.length - 1; l++) {
        const currentLayerSize = network.layers[l].size;
        const nextLayerSize = network.layers[l + 1].size;
        
        const layerWeights = [];
        const layerGradients = [];
        const layerMomentum = [];
        
        for (let i = 0; i < currentLayerSize; i++) {
            const nodeWeights = [];
            const nodeGradients = [];
            const nodeMomentum = [];
            
            for (let j = 0; j < nextLayerSize; j++) {
                // Xavier/Glorot initialization for better initial weights
                const scale = Math.sqrt(2.0 / (currentLayerSize + nextLayerSize));
                nodeWeights.push((Math.random() * 2 - 1) * scale);
                nodeGradients.push(0);
                nodeMomentum.push(0);
            }
            
            layerWeights.push(nodeWeights);
            layerGradients.push(nodeGradients);
            layerMomentum.push(nodeMomentum);
        }
        
        network.weights.push(layerWeights);
        network.gradients.weights.push(layerGradients);
        network.momentum.weights.push(layerMomentum);
    }
    
    // Calculate node positions for visualization
    calculateNodePositions();
}

// Forward pass through the network
function forwardPass(input) {
    // Set input layer activations
    for (let i = 0; i < network.layers[0].size; i++) {
        network.activations[0][i] = input[i];
    }
    
    // Process activation function selection
    const activationFnSelect = document.getElementById('activation-fn');
    const activationFnName = activationFnSelect ? activationFnSelect.value : 'sigmoid';
    const activationFn = activationFunctions[activationFnName];
    
    // Process each layer
    for (let l = 1; l < network.layers.length; l++) {
        const layerSize = network.layers[l].size;
        const prevLayerSize = network.layers[l - 1].size;
        
        // For each node in current layer
        for (let j = 0; j < layerSize; j++) {
            let sum = network.biases[l - 1][j];
            
            // Sum weighted inputs from previous layer
            for (let i = 0; i < prevLayerSize; i++) {
                sum += network.activations[l - 1][i] * network.weights[l - 1][i][j];
            }
            
            // Store pre-activation value
            network.preActivations[l][j] = sum;
            
            // Apply activation function
            network.activations[l][j] = activationFn.forward(sum);
        }
    }
    
    // Return output layer activations
    return network.activations[network.layers.length - 1];
}

// Calculate loss for a single sample (cross entropy)
function calculateLoss(predicted, target) {
    let loss = 0;
    for (let i = 0; i < predicted.length; i++) {
        // Prevent log(0) with small epsilon
        const p = Math.max(predicted[i], 1e-15);
        loss += target[i] * Math.log(p);
    }
    return -loss;
}

// Calculate accuracy for classification
function calculateAccuracy(predicted, target) {
    const predictedClass = predicted.indexOf(Math.max(...predicted));
    const targetClass = target.indexOf(Math.max(...target));
    return predictedClass === targetClass ? 1 : 0;
}

// Backpropagation
function backpropagation(input, target) {
    // Forward pass
    forwardPass(input);
    
    // Process activation function selection
    const activationFnSelect = document.getElementById('activation-fn');
    const activationFnName = activationFnSelect ? activationFnSelect.value : 'sigmoid';
    const activationFn = activationFunctions[activationFnName];
    
    // Initialize arrays to store deltas
    const deltas = [];
    for (let l = 0; l < network.layers.length; l++) {
        deltas.push(new Array(network.layers[l].size).fill(0));
    }
    
    // Calculate output layer deltas (assuming cross-entropy loss and softmax output)
    const outputLayer = network.layers.length - 1;
    for (let i = 0; i < network.layers[outputLayer].size; i++) {
        // For cross entropy loss, delta is simply (prediction - target)
        deltas[outputLayer][i] = network.activations[outputLayer][i] - target[i];
    }
    
    // Backpropagate deltas to hidden layers
    for (let l = outputLayer - 1; l > 0; l--) {
        for (let i = 0; i < network.layers[l].size; i++) {
            let delta = 0;
            
            // Sum up deltas from next layer
            for (let j = 0; j < network.layers[l + 1].size; j++) {
                delta += deltas[l + 1][j] * network.weights[l][i][j];
            }
            
            // Multiply by derivative of activation function
            delta *= activationFn.backward(network.preActivations[l][i]);
            
            deltas[l][i] = delta;
        }
    }
    
    // Calculate gradients for weights and biases
    for (let l = outputLayer; l > 0; l--) {
        for (let i = 0; i < network.layers[l].size; i++) {
            // Bias gradient is simply the delta
            network.gradients.biases[l - 1][i] = deltas[l][i];
            
            // Calculate weight gradients
            for (let j = 0; j < network.layers[l - 1].size; j++) {
                network.gradients.weights[l - 1][j][i] = deltas[l][i] * network.activations[l - 1][j];
            }
        }
    }
    
    return calculateLoss(network.activations[outputLayer], target);
}

// Apply dropout to network
function applyDropout(rate) {
    if (rate <= 0) return;
    
    // Store dropout masks
    if (!network.dropoutMasks) {
        network.dropoutMasks = [];
        
        // Create masks for all but input and output layers
        for (let l = 1; l < network.layers.length - 1; l++) {
            const layerSize = network.layers[l].size;
            network.dropoutMasks.push(new Array(layerSize).fill(1));
        }
    }
    
    // Apply dropout masks
    for (let l = 0; l < network.dropoutMasks.length; l++) {
        const mask = network.dropoutMasks[l];
        
        for (let i = 0; i < mask.length; i++) {
            // Keep with probability (1 - rate)
            mask[i] = Math.random() >= rate ? 1 / (1 - rate) : 0;
            
            // Apply mask to activations
            network.activations[l + 1][i] *= mask[i];
        }
    }
}

// Clear dropout masks
function clearDropout() {
    if (!network.dropoutMasks) return;
    
    for (let l = 0; l < network.dropoutMasks.length; l++) {
        const mask = network.dropoutMasks[l];
        
        for (let i = 0; i < mask.length; i++) {
            mask[i] = 1;
        }
    }
}

// Train on batch of samples
function trainOnBatch(batch) {
    // Get batch size control value
    const batchSizeInput = document.getElementById('batch-size');
    const batchSize = batchSizeInput ? parseInt(batchSizeInput.value) : 16;
    
    // Get learning rate control value
    const learningRateInput = document.getElementById('learning-rate');
    const learningRate = learningRateInput ? parseFloat(learningRateInput.value) : 0.1;
    
    // Get dropout rate if available
    const dropoutRateInput = document.getElementById('dropout-rate');
    const dropoutRate = dropoutRateInput ? parseFloat(dropoutRateInput.value) : 0;
    
    // Get L2 regularization if available
    const l2RegInput = document.getElementById('l2-regularization');
    const l2Reg = l2RegInput ? parseFloat(l2RegInput.value) : 0;
    
    // Get optimizer if available
    const optimizerSelect = document.getElementById('optimizer');
    const optimizerName = optimizerSelect ? optimizerSelect.value : 'sgd';
    
    // Check for power-ups
    const boostFactor = activePowerUps.gradientBoost ? 2 : 1;
    const useRegularization = activePowerUps.regularization || l2Reg > 0;
    const useBatchNorm = activePowerUps.batchNormalization;
    const useMomentum = activePowerUps.momentum;
    
    // Reset gradients
    for (let l = 0; l < network.gradients.weights.length; l++) {
        for (let i = 0; i < network.gradients.weights[l].length; i++) {
            for (let j = 0; j < network.gradients.weights[l][i].length; j++) {
                network.gradients.weights[l][i][j] = 0;
            }
        }
        
        for (let i = 0; i < network.gradients.biases[l].length; i++) {
            network.gradients.biases[l][i] = 0;
        }
    }
    
    // Process each sample in batch
    let batchLoss = 0;
    for (let i = 0; i < batch.length; i++) {
        // Apply dropout for training
        applyDropout(dropoutRate);
        
        // Run backpropagation
        const sampleLoss = backpropagation(batch[i].input, batch[i].output);
        batchLoss += sampleLoss;
        
        // Clear dropout for next sample
        clearDropout();
    }
    
    // Average gradients over batch
    for (let l = 0; l < network.gradients.weights.length; l++) {
        for (let i = 0; i < network.gradients.weights[l].length; i++) {
            for (let j = 0; j < network.gradients.weights[l][i].length; j++) {
                network.gradients.weights[l][i][j] /= batch.length;
            }
        }
        
        for (let i = 0; i < network.gradients.biases[l].length; i++) {
            network.gradients.biases[l][i] /= batch.length;
        }
    }
    
    // L2 regularization gradient contribution
    if (useRegularization) {
        const l2Factor = activePowerUps.regularization ? 0.01 : l2Reg;
        
        for (let l = 0; l < network.weights.length; l++) {
            for (let i = 0; i < network.weights[l].length; i++) {
                for (let j = 0; j < network.weights[l][i].length; j++) {
                    // Add L2 regularization gradient: lambda * w
                    network.gradients.weights[l][i][j] += l2Factor * network.weights[l][i][j];
                }
            }
        }
    }
    
    // Apply boosted learning rate
    const effectiveLR = learningRate * boostFactor;
    
    // Apply gradient descent update with chosen optimizer
    if (optimizerName === 'adam') {
        optimizers.adam(network.weights, network.biases, network.gradients.weights, network.gradients.biases, effectiveLR, network, epoch + 1);
    } else if (optimizerName === 'rmsprop') {
        optimizers.rmsprop(network.weights, network.biases, network.gradients.weights, network.gradients.biases, effectiveLR, network);
    } else {
        // Default SGD, possibly with momentum
        if (useMomentum) {
            const momentumFactor = 0.9;
            
            // Apply momentum update
            for (let l = 0; l < network.weights.length; l++) {
                for (let i = 0; i < network.weights[l].length; i++) {
                    for (let j = 0; j < network.weights[l][i].length; j++) {
                        // Update momentum
                        network.momentum.weights[l][i][j] = 
                            momentumFactor * network.momentum.weights[l][i][j] - 
                            effectiveLR * network.gradients.weights[l][i][j];
                        
                        // Apply momentum update
                        network.weights[l][i][j] += network.momentum.weights[l][i][j];
                    }
                }
                
                for (let i = 0; i < network.biases[l].length; i++) {
                    // Update momentum
                    network.momentum.biases[l][i] = 
                        momentumFactor * network.momentum.biases[l][i] - 
                        effectiveLR * network.gradients.biases[l][i];
                    
                    // Apply momentum update
                    network.biases[l][i] += network.momentum.biases[l][i];
                }
            }
        } else {
            // Standard SGD
            optimizers.sgd(network.weights, network.biases, network.gradients.weights, network.gradients.biases, effectiveLR);
        }
    }
    
    // Return average loss
    return batchLoss / batch.length;
}

// Run evaluation on validation data
function evaluateNetwork() {
    let totalLoss = 0;
    let correctPredictions = 0;
    
    for (let i = 0; i < validationData.length; i++) {
        const input = validationData[i].input;
        const target = validationData[i].output;
        
        // Forward pass
        const predicted = forwardPass(input);
        
        // Calculate loss
        totalLoss += calculateLoss(predicted, target);
        
        // Calculate accuracy
        if (calculateAccuracy(predicted, target) === 1) {
            correctPredictions++;
        }
    }
    
    const avgLoss = totalLoss / validationData.length;
    const avgAccuracy = correctPredictions / validationData.length * 100;
    
    return {
        loss: avgLoss,
        accuracy: avgAccuracy
    };
}

// Generate training data based on level data model
function generateTrainingData() {
    const currentLevel = gameData.levels[gameData.player.currentLevel];
    const dataModel = currentLevel.dataModel;
    trainingData = [];
    validationData = [];
    
    const totalSamples = dataModel.sampleSize;
    const validationSplit = dataModel.testingSplit || 0.2;
    const trainSamples = Math.floor(totalSamples * (1 - validationSplit));
    const valSamples = totalSamples - trainSamples;
    
    switch (dataModel.type) {
        case 'binary_classification':
            generateBinaryClassificationData(trainSamples, valSamples, dataModel);
            break;
        case 'multi_class':
            generateMultiClassData(trainSamples, valSamples, dataModel);
            break;
        case 'noisy_classification':
            generateNoisyClassificationData(trainSamples, valSamples, dataModel);
            break;
        case 'complex_multi_class':
            generateComplexMultiClassData(trainSamples, valSamples, dataModel);
            break;
        case 'adversarial':
            generateAdversarialData(trainSamples, valSamples, dataModel);
            break;
        default:
            generateBinaryClassificationData(trainSamples, valSamples, dataModel);
    }
}

// Generate binary classification data
function generateBinaryClassificationData(trainSamples, valSamples, dataModel) {
    const features = dataModel.features;
    const noiseLevel = dataModel.noiseLevel || 0.1;
    
    // Generate training data
    for (let i = 0; i < trainSamples; i++) {
        // Random classification: 0 or 1
        const classLabel = Math.floor(Math.random() * 2);
        
        // Generate features with pattern related to class
        const input = [];
        for (let j = 0; j < features; j++) {
            // Add pattern: even features are higher for class 1, odd features higher for class 0
            let featureValue = 0;
            if (j % 2 === classLabel) {
                featureValue = 0.7 + Math.random() * 0.3;
            } else {
                featureValue = Math.random() * 0.3;
            }
            
            // Add noise
            featureValue += (Math.random() * 2 - 1) * noiseLevel;
            featureValue = Math.max(0, Math.min(1, featureValue)); // Clamp to [0,1]
            
            input.push(featureValue);
        }
        
        // Create one-hot encoded output
        const output = [0, 0];
        output[classLabel] = 1;
        
        trainingData.push({ input, output });
    }
    
    // Generate validation data
    for (let i = 0; i < valSamples; i++) {
        const classLabel = Math.floor(Math.random() * 2);
        const input = [];
        
        for (let j = 0; j < features; j++) {
            let featureValue = 0;
            if (j % 2 === classLabel) {
                featureValue = 0.7 + Math.random() * 0.3;
            } else {
                featureValue = Math.random() * 0.3;
            }
            
            featureValue += (Math.random() * 2 - 1) * noiseLevel;
            featureValue = Math.max(0, Math.min(1, featureValue));
            
            input.push(featureValue);
        }
        
        const output = [0, 0];
        output[classLabel] = 1;
        
        validationData.push({ input, output });
    }
}

// Generate multi-class data
function generateMultiClassData(trainSamples, valSamples, dataModel) {
    const features = dataModel.features;
    const classes = dataModel.classes;
    const noiseLevel = dataModel.noiseLevel || 0.15;
    
    // Generate training data
    for (let i = 0; i < trainSamples; i++) {
        const classLabel = Math.floor(Math.random() * classes);
        const input = [];
        
        // Generate features with more complex pattern
        for (let j = 0; j < features; j++) {
            // Feature pattern: value depends on class and feature index
            let featureValue = 0;
            
            // Each class has a distinct pattern of high/low features
            if ((j + classLabel) % classes === 0) {
                featureValue = 0.8 + Math.random() * 0.2; // High
            } else if ((j + classLabel) % classes === 1) {
                featureValue = 0.6 + Math.random() * 0.2; // Medium-high
            } else if ((j + classLabel) % classes === 2) {
                featureValue = 0.3 + Math.random() * 0.2; // Medium-low
            } else {
                featureValue = Math.random() * 0.2; // Low
            }
            
            // Add noise
            featureValue += (Math.random() * 2 - 1) * noiseLevel;
            featureValue = Math.max(0, Math.min(1, featureValue));
            
            input.push(featureValue);
        }
        
        // Create one-hot encoded output
        const output = new Array(classes).fill(0);
        output[classLabel] = 1;
        
        trainingData.push({ input, output });
    }
    
    // Generate validation data with similar pattern
    for (let i = 0; i < valSamples; i++) {
        const classLabel = Math.floor(Math.random() * classes);
        const input = [];
        
        for (let j = 0; j < features; j++) {
            let featureValue = 0;
            
            if ((j + classLabel) % classes === 0) {
                featureValue = 0.8 + Math.random() * 0.2;
            } else if ((j + classLabel) % classes === 1) {
                featureValue = 0.6 + Math.random() * 0.2;
            } else if ((j + classLabel) % classes === 2) {
                featureValue = 0.3 + Math.random() * 0.2;
            } else {
                featureValue = Math.random() * 0.2;
            }
            
            featureValue += (Math.random() * 2 - 1) * noiseLevel;
            featureValue = Math.max(0, Math.min(1, featureValue));
            
            input.push(featureValue);
        }
        
        const output = new Array(classes).fill(0);
        output[classLabel] = 1;
        
        validationData.push({ input, output });
    }
}

// Generate noisy classification data prone to overfitting
function generateNoisyClassificationData(trainSamples, valSamples, dataModel) {
    const features = dataModel.features;
    const classes = dataModel.classes;
    const noiseLevel = dataModel.noiseLevel || 0.25;
    
    // Generate training data with high noise
    for (let i = 0; i < trainSamples; i++) {
        const classLabel = Math.floor(Math.random() * classes);
        const input = [];
        
        // Only the first 3 features are truly informative
        for (let j = 0; j < features; j++) {
            let featureValue;
            
            if (j < 3) {
                // Informative features
                if (j === classLabel % 3) {
                    featureValue = 0.8 + Math.random() * 0.2;
                } else {
                    featureValue = Math.random() * 0.3;
                }
            } else {
                // Non-informative features (pure noise)
                featureValue = Math.random();
            }
            
            // Add noise
            featureValue += (Math.random() * 2 - 1) * noiseLevel;
            featureValue = Math.max(0, Math.min(1, featureValue));
            
            input.push(featureValue);
        }
        
        // Create one-hot encoded output
        const output = new Array(classes).fill(0);
        output[classLabel] = 1;
        
        trainingData.push({ input, output });
    }
    
    // Generate validation data with same pattern but different noise
    for (let i = 0; i < valSamples; i++) {
        const classLabel = Math.floor(Math.random() * classes);
        const input = [];
        
        for (let j = 0; j < features; j++) {
            let featureValue;
            
            if (j < 3) {
                // Informative features
                if (j === classLabel % 3) {
                    featureValue = 0.8 + Math.random() * 0.2;
                } else {
                    featureValue = Math.random() * 0.3;
                }
            } else {
                // Non-informative features (pure noise)
                featureValue = Math.random();
            }
            
            // Add noise
            featureValue += (Math.random() * 2 - 1) * noiseLevel;
            featureValue = Math.max(0, Math.min(1, featureValue));
            
            input.push(featureValue);
        }
        
        const output = new Array(classes).fill(0);
        output[classLabel] = 1;
        
        validationData.push({ input, output });
    }
}

// Generate complex multi-class data
function generateComplexMultiClassData(trainSamples, valSamples, dataModel) {
    const features = dataModel.features;
    const classes = dataModel.classes;
    const noiseLevel = dataModel.noiseLevel || 0.2;
    
    // Generate training data with complex patterns
    for (let i = 0; i < trainSamples; i++) {
        const classLabel = Math.floor(Math.random() * classes);
        const input = [];
        
        // Create complex feature interactions
        for (let j = 0; j < features; j++) {
            let featureValue;
            
            // Different classes have different feature interactions
            if (classLabel < classes / 2) {
                // First half of classes: XOR-like patterns
                if ((j % 3 === 0 && j % 2 === 0) || (j % 3 !== 0 && j % 2 !== 0)) {
                    featureValue = (classLabel % 2 === 0) ? 0.8 + Math.random() * 0.2 : Math.random() * 0.3;
                } else {
                    featureValue = (classLabel % 2 === 1) ? 0.8 + Math.random() * 0.2 : Math.random() * 0.3;
                }
            } else {
                // Second half of classes: Cluster-like patterns
                if (j % classes === classLabel % (classes / 2)) {
                    featureValue = 0.7 + Math.random() * 0.3;
                } else if (Math.abs(j % classes - classLabel % (classes / 2)) === 1) {
                    featureValue = 0.5 + Math.random() * 0.3;
                } else {
                    featureValue = Math.random() * 0.3;
                }
            }
            
            // Add noise
            featureValue += (Math.random() * 2 - 1) * noiseLevel;
            featureValue = Math.max(0, Math.min(1, featureValue));
            
            input.push(featureValue);
        }
        
        // Create one-hot encoded output
        const output = new Array(classes).fill(0);
        output[classLabel] = 1;
        
        trainingData.push({ input, output });
    }
    
    // Generate validation data
    for (let i = 0; i < valSamples; i++) {
        const classLabel = Math.floor(Math.random() * classes);
        const input = [];
        
        for (let j = 0; j < features; j++) {
            let featureValue;
            
            if (classLabel < classes / 2) {
                if ((j % 3 === 0 && j % 2 === 0) || (j % 3 !== 0 && j % 2 !== 0)) {
                    featureValue = (classLabel % 2 === 0) ? 0.8 + Math.random() * 0.2 : Math.random() * 0.3;
                } else {
                    featureValue = (classLabel % 2 === 1) ? 0.8 + Math.random() * 0.2 : Math.random() * 0.3;
                }
            } else {
                if (j % classes === classLabel % (classes / 2)) {
                    featureValue = 0.7 + Math.random() * 0.3;
                } else if (Math.abs(j % classes - classLabel % (classes / 2)) === 1) {
                    featureValue = 0.5 + Math.random() * 0.3;
                } else {
                    featureValue = Math.random() * 0.3;
                }
            }
            
            featureValue += (Math.random() * 2 - 1) * noiseLevel;
            featureValue = Math.max(0, Math.min(1, featureValue));
            
            input.push(featureValue);
        }
        
        const output = new Array(classes).fill(0);
        output[classLabel] = 1;
        
        validationData.push({ input, output });
    }
}

// Generate adversarial data
function generateAdversarialData(trainSamples, valSamples, dataModel) {
    const features = dataModel.features;
    const classes = dataModel.classes;
    const noiseLevel = dataModel.noiseLevel || 0.15;
    const adversarialRatio = dataModel.adversarialRatio || 0.2;
    
    // Generate clean data first
    const cleanTrainSamples = Math.floor(trainSamples * (1 - adversarialRatio));
    const adversarialTrainSamples = trainSamples - cleanTrainSamples;
    
    // Generate clean training data
    for (let i = 0; i < cleanTrainSamples; i++) {
        const classLabel = Math.floor(Math.random() * classes);
        const input = [];
        
        // Generate features
        for (let j = 0; j < features; j++) {
            let featureValue;
            
            // Simple cluster pattern
            if (j % classes === classLabel) {
                featureValue = 0.8 + Math.random() * 0.2;
            } else {
                featureValue = Math.random() * 0.3;
            }
            
            // Add noise
            featureValue += (Math.random() * 2 - 1) * noiseLevel;
            featureValue = Math.max(0, Math.min(1, featureValue));
            
            input.push(featureValue);
        }
        
        // Create one-hot encoded output
        const output = new Array(classes).fill(0);
        output[classLabel] = 1;
        
        trainingData.push({ input, output, isAdversarial: false });
    }
    
    // Generate adversarial examples
    for (let i = 0; i < adversarialTrainSamples; i++) {
        const classLabel = Math.floor(Math.random() * classes);
        const targetLabel = (classLabel + 1) % classes; // Target a different class
        const input = [];
        
        // Generate base features for the original class
        for (let j = 0; j < features; j++) {
            let featureValue;
            
            // Simple cluster pattern
            if (j % classes === classLabel) {
                featureValue = 0.8 + Math.random() * 0.2;
            } else {
                featureValue = Math.random() * 0.3;
            }
            
            // Add normal noise
            featureValue += (Math.random() * 2 - 1) * noiseLevel;
            
            // Add adversarial perturbation: push feature toward target class
            if (j % classes === targetLabel) {
                featureValue += 0.2 + Math.random() * 0.2; // Push up
            }
            if (j % classes === classLabel) {
                featureValue -= 0.2 + Math.random() * 0.1; // Push down
            }
            
            featureValue = Math.max(0, Math.min(1, featureValue));
            
            input.push(featureValue);
        }
        
        // The true label is still the original class, not the target
        const output = new Array(classes).fill(0);
        output[classLabel] = 1;
        
        trainingData.push({ input, output, isAdversarial: true });
    }
    
    // Generate validation data (all clean)
    for (let i = 0; i < valSamples; i++) {
        const classLabel = Math.floor(Math.random() * classes);
        const input = [];
        
        for (let j = 0; j < features; j++) {
            let featureValue;
            
            if (j % classes === classLabel) {
                featureValue = 0.8 + Math.random() * 0.2;
            } else {
                featureValue = Math.random() * 0.3;
            }
            
            featureValue += (Math.random() * 2 - 1) * noiseLevel;
            featureValue = Math.max(0, Math.min(1, featureValue));
            
            input.push(featureValue);
        }
        
        const output = new Array(classes).fill(0);
        output[classLabel] = 1;
        
        validationData.push({ input, output });
    }
    
    // Shuffle training data to mix adversarial and clean examples
    trainingData = shuffleArray(trainingData);
}

// Utility function to shuffle array
function shuffleArray(array) {
    const newArray = [...array];
    for (let i = newArray.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [newArray[i], newArray[j]] = [newArray[j], newArray[i]];
    }
    return newArray;
}
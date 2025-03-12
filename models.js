// Game data models
const gameData = {
    player: {
        completedLevels: [],
        tokens: 10,
        powerUps: {
            gradientBoost: 0,
            regularization: 0,
            batchNormalization: 0,
            momentum: 0,
            dataAugmentation: 0
        },
        currentLevel: 0,
        unlocked: [0]
    },
    
    levels: [
        {
            id: 0,
            name: "Level 1: Classification Basics",
            description: "Train your first neural network to classify simple binary patterns.",
            targetAccuracy: 85,
            maxTokens: 10,
            networkStructure: [4, 6, 2],
            dataModel: {
                type: "binary_classification",
                classes: 2,
                features: 4,
                sampleSize: 100,
                noiseLevel: 0.1
            },
            availableControls: [
                { 
                    id: "learning-rate", 
                    name: "Learning Rate", 
                    type: "range", 
                    min: 0.01, 
                    max: 1, 
                    step: 0.01, 
                    value: 0.1,
                    description: "Controls how quickly the network updates weights. Higher values make bigger changes."
                },
                { 
                    id: "activation-fn", 
                    name: "Activation Function", 
                    type: "select", 
                    options: ["sigmoid", "relu", "tanh"],
                    value: "sigmoid",
                    description: "The function that determines how node values are calculated. Different functions have different properties."
                },
                { 
                    id: "batch-size", 
                    name: "Batch Size", 
                    type: "range", 
                    min: 1, 
                    max: 64, 
                    step: 1, 
                    value: 16,
                    description: "Number of samples processed before updating weights. Larger batches are more stable but slower to adapt."
                }
            ],
            rewards: {
                tokens: 20,
                powerUps: {
                    gradientBoost: 1
                }
            },
            tutorials: [
                {
                    title: "Welcome to Classification",
                    content: "In this level, you'll train a neural network to classify data into two categories. This is one of the most common tasks in machine learning.<br><br>Your network has 4 input nodes (features), 6 hidden nodes (for processing), and 2 output nodes (classes).<br><br>Start by deploying training tokens on nodes you want to focus on, then adjust the parameters and run training.",
                },
                {
                    title: "Understanding Learning Rate",
                    content: "The <strong>Learning Rate</strong> controls how quickly your network adapts to the training data.<br><br>Higher values (>0.5) cause faster learning but might overshoot the optimal solution.<br><br>Lower values (<0.1) provide more stable but slower learning.<br><br>Try different values to see how it affects training!"
                }
            ]
        },
        {
            id: 1,
            name: "Level 2: Pattern Recognition",
            description: "Train a network to recognize visual patterns with more complex hidden layer dynamics.",
            targetAccuracy: 80,
            maxTokens: 12,
            networkStructure: [6, 8, 4],
            dataModel: {
                type: "multi_class",
                classes: 4,
                features: 6,
                sampleSize: 150,
                noiseLevel: 0.15
            },
            availableControls: [
                { 
                    id: "learning-rate", 
                    name: "Learning Rate", 
                    type: "range", 
                    min: 0.01, 
                    max: 1, 
                    step: 0.01, 
                    value: 0.1,
                    description: "Controls how quickly the network updates weights. Higher values make bigger changes."
                },
                { 
                    id: "activation-fn", 
                    name: "Activation Function", 
                    type: "select", 
                    options: ["sigmoid", "relu", "tanh"],
                    value: "relu",
                    description: "The function that determines how node values are calculated. Different functions have different properties."
                },
                { 
                    id: "batch-size", 
                    name: "Batch Size", 
                    type: "range", 
                    min: 1, 
                    max: 64, 
                    step: 1, 
                    value: 16,
                    description: "Number of samples processed before updating weights. Larger batches are more stable but slower to adapt."
                },
                { 
                    id: "dropout-rate", 
                    name: "Dropout Rate", 
                    type: "range", 
                    min: 0, 
                    max: 0.5, 
                    step: 0.05, 
                    value: 0,
                    description: "Randomly deactivates nodes during training to prevent overfitting. Higher values provide more regularization."
                }
            ],
            rewards: {
                tokens: 25,
                powerUps: {
                    regularization: 1
                }
            },
            tutorials: [
                {
                    title: "Pattern Recognition Challenge",
                    content: "This level focuses on recognizing visual patterns with more complex relationships. Your network needs to classify inputs into 4 different categories.<br><br>Notice the larger network structure, with 6 inputs, 8 hidden nodes, and 4 outputs.<br><br>You now have access to Dropout Rate, which helps prevent overfitting."
                }
            ]
        },
        {
            id: 2,
            name: "Level 3: Overfitting Challenge",
            description: "Train a network on noisy data while avoiding overfitting to the training samples.",
            targetAccuracy: 75,
            maxTokens: 15,
            networkStructure: [6, 10, 4],
            dataModel: {
                type: "noisy_classification",
                classes: 4,
                features: 6,
                sampleSize: 200,
                noiseLevel: 0.25,
                testingSplit: 0.3
            },
            availableControls: [
                { 
                    id: "learning-rate", 
                    name: "Learning Rate", 
                    type: "range", 
                    min: 0.001, 
                    max: 0.5, 
                    step: 0.001, 
                    value: 0.05,
                    description: "Controls how quickly the network updates weights. Higher values make bigger changes."
                },
                { 
                    id: "activation-fn", 
                    name: "Activation Function", 
                    type: "select", 
                    options: ["sigmoid", "relu", "tanh", "leaky_relu"],
                    value: "relu",
                    description: "The function that determines how node values are calculated. Different functions have different properties."
                },
                { 
                    id: "batch-size", 
                    name: "Batch Size", 
                    type: "range", 
                    min: 1, 
                    max: 64, 
                    step: 1, 
                    value: 32,
                    description: "Number of samples processed before updating weights. Larger batches are more stable but slower to adapt."
                },
                { 
                    id: "dropout-rate", 
                    name: "Dropout Rate", 
                    type: "range", 
                    min: 0, 
                    max: 0.5, 
                    step: 0.05, 
                    value: 0.2,
                    description: "Randomly deactivates nodes during training to prevent overfitting. Higher values provide more regularization."
                },
                { 
                    id: "l2-regularization", 
                    name: "L2 Regularization", 
                    type: "range", 
                    min: 0, 
                    max: 0.1, 
                    step: 0.005, 
                    value: 0.01,
                    description: "Penalizes large weights to prevent overfitting. Higher values result in smaller weights overall."
                }
            ],
            rewards: {
                tokens: 30,
                powerUps: {
                    batchNormalization: 1
                }
            },
            tutorials: [
                {
                    title: "Overfitting Challenge",
                    content: "In this level, your data contains significant noise. The challenge is to build a model that generalizes well rather than memorizing the training examples.<br><br>Overfitting occurs when your model performs well on training data but poorly on new data.<br><br>Use regularization techniques like Dropout and L2 Regularization to combat overfitting."
                }
            ]
        },
        {
            id: 3,
            name: "Level 4: Multi-Class Classification",
            description: "Train a deeper network for complex multi-class classification tasks.",
            targetAccuracy: 70,
            maxTokens: 20,
            networkStructure: [8, 12, 6],
            dataModel: {
                type: "complex_multi_class",
                classes: 6,
                features: 8,
                sampleSize: 300,
                noiseLevel: 0.2
            },
            availableControls: [
                { 
                    id: "learning-rate", 
                    name: "Learning Rate", 
                    type: "range", 
                    min: 0.001, 
                    max: 0.5, 
                    step: 0.001, 
                    value: 0.01,
                    description: "Controls how quickly the network updates weights. Higher values make bigger changes."
                },
                { 
                    id: "activation-fn", 
                    name: "Activation Function", 
                    type: "select", 
                    options: ["sigmoid", "relu", "tanh", "leaky_relu", "elu"],
                    value: "leaky_relu",
                    description: "The function that determines how node values are calculated. Different functions have different properties."
                },
                { 
                    id: "batch-size", 
                    name: "Batch Size", 
                    type: "range", 
                    min: 1, 
                    max: 128, 
                    step: 1, 
                    value: 64,
                    description: "Number of samples processed before updating weights. Larger batches are more stable but slower to adapt."
                },
                { 
                    id: "dropout-rate", 
                    name: "Dropout Rate", 
                    type: "range", 
                    min: 0, 
                    max: 0.5, 
                    step: 0.05, 
                    value: 0.2,
                    description: "Randomly deactivates nodes during training to prevent overfitting. Higher values provide more regularization."
                },
                { 
                    id: "l2-regularization", 
                    name: "L2 Regularization", 
                    type: "range", 
                    min: 0, 
                    max: 0.1, 
                    step: 0.005, 
                    value: 0.01,
                    description: "Penalizes large weights to prevent overfitting. Higher values result in smaller weights overall."
                },
                { 
                    id: "optimizer", 
                    name: "Optimizer", 
                    type: "select", 
                    options: ["sgd", "adam", "rmsprop"],
                    value: "adam",
                    description: "Algorithm used to update weights. Different optimizers have different properties and convergence rates."
                }
            ],
            rewards: {
                tokens: 40,
                powerUps: {
                    momentum: 1
                }
            },
            tutorials: [
                {
                    title: "Multi-Class Classification",
                    content: "This level challenges you to classify inputs into 6 different categories. You're working with a larger network and more complex data relationships.<br><br>You now have access to different optimizer algorithms, which affect how weights are updated during training.<br><br>Adam is an adaptive optimizer that often performs well without much tuning, while SGD (Stochastic Gradient Descent) gives you more control but requires careful parameter selection."
                }
            ]
        },
        {
            id: 4,
            name: "Level 5: Adversarial Challenge",
            description: "Train a robust network that can handle adversarial examples designed to fool classifiers.",
            targetAccuracy: 65,
            maxTokens: 25,
            networkStructure: [8, 14, 8, 4],
            dataModel: {
                type: "adversarial",
                classes: 4,
                features: 8,
                sampleSize: 400,
                adversarialRatio: 0.2,
                noiseLevel: 0.15
            },
            availableControls: [
                { 
                    id: "learning-rate", 
                    name: "Learning Rate", 
                    type: "range", 
                    min: 0.0001, 
                    max: 0.1, 
                    step: 0.0001, 
                    value: 0.005,
                    description: "Controls how quickly the network updates weights. Higher values make bigger changes."
                },
                { 
                    id: "activation-fn", 
                    name: "Activation Function", 
                    type: "select", 
                    options: ["sigmoid", "relu", "tanh", "leaky_relu", "elu"],
                    value: "elu",
                    description: "The function that determines how node values are calculated. Different functions have different properties."
                },
                { 
                    id: "batch-size", 
                    name: "Batch Size", 
                    type: "range", 
                    min: 16, 
                    max: 256, 
                    step: 16, 
                    value: 128,
                    description: "Number of samples processed before updating weights. Larger batches are more stable but slower to adapt."
                },
                { 
                    id: "dropout-rate", 
                    name: "Dropout Rate", 
                    type: "range", 
                    min: 0, 
                    max: 0.5, 
                    step: 0.05, 
                    value: 0.3,
                    description: "Randomly deactivates nodes during training to prevent overfitting. Higher values provide more regularization."
                },
                { 
                    id: "l2-regularization", 
                    name: "L2 Regularization", 
                    type: "range", 
                    min: 0, 
                    max: 0.1, 
                    step: 0.005, 
                    value: 0.02,
                    description: "Penalizes large weights to prevent overfitting. Higher values result in smaller weights overall."
                },
                { 
                    id: "optimizer", 
                    name: "Optimizer", 
                    type: "select", 
                    options: ["sgd", "adam", "rmsprop"],
                    value: "adam",
                    description: "Algorithm used to update weights. Different optimizers have different properties and convergence rates."
                },
                { 
                    id: "adversarial-training", 
                    name: "Adversarial Training", 
                    type: "range", 
                    min: 0, 
                    max: 1, 
                    step: 0.1, 
                    value: 0.5,
                    description: "Percentage of adversarial examples to include in training. Higher values improve robustness but may reduce overall accuracy."
                }
            ],
            rewards: {
                tokens: 50,
                powerUps: {
                    dataAugmentation: 1
                }
            },
            tutorials: [
                {
                    title: "Adversarial Challenge",
                    content: "In this advanced level, you'll face adversarial examples - inputs specifically designed to fool neural networks.<br><br>Your network has an additional hidden layer to help process these complex patterns. Notice the structure: 8 inputs, 14 hidden nodes, 8 more hidden nodes, and 4 outputs.<br><br>The Adversarial Training parameter controls how much your network learns from these tricky examples."
                }
            ]
        }
    ],

    powerUps: {
        gradientBoost: {
            name: "Gradient Boost",
            description: "Temporarily increases the effect of weight updates, helping convergence happen faster.",
            effect: function(network) {
                // Implementation in training function
                return "Gradient updates boosted by 2x for this training cycle";
            }
        },
        regularization: {
            name: "Regularization Shield",
            description: "Applies L2 regularization to prevent overfitting, even if not explicitly configured.",
            effect: function(network) {
                // Implementation in training function
                return "Applied temporary L2 regularization to all weights";
            }
        },
        batchNormalization: {
            name: "Batch Normalization",
            description: "Normalizes activations to improve training stability and convergence speed.",
            effect: function(network) {
                // Implementation in training function
                return "Applied batch normalization to all layers";
            }
        },
        momentum: {
            name: "Momentum Accelerator",
            description: "Adds momentum to weight updates, helping escape local minima.",
            effect: function(network) {
                // Implementation in training function
                return "Added momentum to weight updates for this training cycle";
            }
        },
        dataAugmentation: {
            name: "Data Augmentation",
            description: "Generates additional training examples by adding small variations to existing data.",
            effect: function(network) {
                // Implementation in training function
                return "Generated augmented training data for this cycle";
            }
        }
    },
    
    // Concepts for educational content
    concepts: {
        learningRate: {
            title: "Learning Rate",
            content: "The learning rate controls how quickly the network updates its weights during training. It's one of the most important hyperparameters to tune.<br><br><strong>Higher learning rates</strong> cause faster learning but may overshoot the optimal solution, leading to unstable training.<br><br><strong>Lower learning rates</strong> provide more stable learning but take longer to converge.<br><br>Finding the right learning rate is often a balancing act - too high and the network never converges, too low and training takes forever."
        },
        activationFunctions: {
            title: "Activation Functions",
            content: "Activation functions introduce non-linearity into neural networks, allowing them to learn complex patterns.<br><br><strong>Sigmoid:</strong> Maps values to range (0,1). Good for output layers in binary classification, but can cause vanishing gradients.<br><br><strong>ReLU:</strong> f(x) = max(0,x). Fast to compute and helps prevent vanishing gradients, but can cause 'dying neurons'.<br><br><strong>Tanh:</strong> Similar to sigmoid but maps to (-1,1). Generally performs better than sigmoid but still has vanishing gradient issues.<br><br><strong>Leaky ReLU:</strong> Like ReLU but allows small negative values, helping prevent dying neurons."
        },
        batchSize: {
            title: "Batch Size",
            content: "Batch size determines how many training examples are processed before the model weights are updated.<br><br><strong>Larger batches</strong> provide more stable gradient estimates but require more memory and may converge to poorer solutions.<br><br><strong>Smaller batches</strong> introduce more noise in the training process, which can help escape local minima, but can make training unstable.<br><br>A good batch size balances computational efficiency with generalization performance."
        },
        dropout: {
            title: "Dropout",
            content: "Dropout is a regularization technique that prevents overfitting by randomly deactivating a percentage of neurons during training.<br><br>This forces the network to develop redundant representations and not rely too heavily on any particular neuron, leading to better generalization.<br><br>Typical dropout rates range from 0.2 to 0.5 (meaning 20-50% of neurons are randomly dropped).<br><br>Dropout is only applied during training, not during inference/testing."
        },
        l2Regularization: {
            title: "L2 Regularization",
            content: "L2 regularization (also called weight decay) prevents overfitting by penalizing large weights.<br><br>It works by adding a term to the loss function that is proportional to the sum of squares of all weights.<br><br>This encourages the network to use smaller weights and distribute the importance across more features, which often leads to better generalization.<br><br>The regularization strength parameter controls how much to penalize large weights."
        },
        optimizers: {
            title: "Optimizers",
            content: "Optimizers determine how to update weights based on the computed gradients.<br><br><strong>SGD (Stochastic Gradient Descent):</strong> Simple but requires careful tuning of learning rate.<br><br><strong>Adam:</strong> Adaptive optimizer that maintains per-parameter learning rates, often works well out of the box.<br><br><strong>RMSprop:</strong> Adapts learning rates based on recent gradient magnitudes, good for recurrent networks.<br><br>Different optimizers work better for different problems, and their effectiveness often depends on proper hyperparameter settings."
        },
        adversarialTraining: {
            title: "Adversarial Training",
            content: "Adversarial examples are inputs specifically designed to fool neural networks, often by adding carefully crafted perturbations that are imperceptible to humans.<br><br>Adversarial training improves network robustness by incorporating these examples during training.<br><br>This technique forces the network to learn more robust features and decision boundaries, making it less susceptible to attacks and more reliable in real-world situations.<br><br>However, there's often a trade-off between robustness to adversarial examples and overall accuracy on clean data."
        }
    }
};
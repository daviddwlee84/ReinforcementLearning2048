import tensorflow as tf

INPUT_NODE = 16 # 4x4 grid
OUTPUT_NODE = 4 # up, down, left, right

DEPTH = 2
ONEHOT_METRICS_PER_BATCH = 16 # 0, 2, 4, 8, ..., 2^15 => 16 grids

def calcBatchNum(depth=DEPTH):
    batch_num = 0
    for i in range(depth, 0, -1):
        batch_num += OUTPUT_NODE**i
    return batch_num

BATCH_NUM = calcBatchNum(DEPTH) # if depth = 2, batch_num = 4 + 4^2
ONEHOT_INPUT = BATCH_NUM * ONEHOT_METRICS_PER_BATCH * INPUT_NODE # 5120

# Policy Gradient
LAYER1_NODE = 256
LAYER2_NODE = 256

# DQN
DQN_LAYER1 = 1024
DQN_LAYER2 = 512
DQN_LAYER3 = 256

WEIGHT_INIT_SCALE = 0.01
REGULARIZER = 0.001

ACTIVATION_FUNCTION = tf.nn.relu

def get_weight(shape, regularizer=REGULARIZER):
    """Generate random initial weight
    
    Arguments:
        shape {list} -- Shape of the weight. In FCNN is [LastLayerShape, CurrentLayerShape].
    
    Keyword Arguments:
        regularizer {int} -- Constant of regularizer (default: {REGULARIZER}).
    
    Returns:
        tf.Variable -- Weight generated.
    """

    w = tf.Variable(tf.truncated_normal(shape, stddev=WEIGHT_INIT_SCALE), name='weights')
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w

def get_bias(shape):
    """Generate baias
    
    Arguments:
        shape {list} -- Shape of bias. Usually a single value list with CurrentLayerShape.
    
    Returns:
        tf.Variable -- Bias generated (initial 0).
    """

    b = tf.Variable(tf.zeros(shape), name='biases')
    return b

def denseLayer(name, input_tensor, input_size, layer_size, activation_function=lambda x: x):
    """Generate a dense layer (fully connected layer)
    
    Arguments:
        name {str} -- Name of the layer.
        input_tensor {tf.Variable} -- Tensor from the last layer.
        input_size {int} -- Number of input units. Usually the last layer size.
        layer_size {int} -- Number of units in this layer.
    
    Keyword Arguments:
        activation_function {function} -- Activation function. Default is None (itself) (default: {lambdax:x})
    
    Returns:
        tf.Variable -- Output of the layer
    """

    with tf.name_scope(name):
        weights = get_weight([input_size, layer_size])
        biases = get_bias([layer_size])
        output_tensor = activation_function(tf.matmul(input_tensor, weights) + biases)

        tf.summary.histogram("Weights " + name, weights)
        tf.summary.histogram("Biases " + name, biases)
        tf.summary.histogram("Activations " + name, output_tensor)

    return output_tensor

# Policy Gradient
def forward(system_input):
    """Forward propagation (i.e. the model structure)
    
    Arguments:
        system_input {tf.Variable} -- Input of the 2048 grid.
    
    Returns:
        tf.Variable -- Inference output of the model.
    """

    y1 = denseLayer('HiddenLayer1', system_input, INPUT_NODE, LAYER1_NODE, activation_function=ACTIVATION_FUNCTION)
    y2 = denseLayer('HiddenLayer2', y1, LAYER1_NODE, LAYER2_NODE, activation_function=ACTIVATION_FUNCTION)
    y  = denseLayer('InferenceLayer', y2, LAYER2_NODE, OUTPUT_NODE)
    y_prob = tf.nn.softmax(y)

    return y, y_prob

# DQN with one-hot input
def DQN_onehot_forward(system_input):
    y1 = denseLayer('HiddenLayer1', system_input, ONEHOT_INPUT, DQN_LAYER1, activation_function=ACTIVATION_FUNCTION)
    y2 = denseLayer('HiddenLayer2', y1, DQN_LAYER1, DQN_LAYER2, activation_function=ACTIVATION_FUNCTION)
    y3 = denseLayer('HiddenLayer3', y2, DQN_LAYER2, DQN_LAYER3, activation_function=ACTIVATION_FUNCTION)
    y  = denseLayer('InferenceLayer', y3, DQN_LAYER3, OUTPUT_NODE) # linear activation funciton
    y_prob = tf.nn.softmax(y)

    return y, y_prob

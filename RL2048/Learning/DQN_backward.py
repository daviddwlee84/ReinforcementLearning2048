import tensorflow as tf
import numpy as np
import os
import RL2048.Learning.forward as forward
from RL2048.Game.StateEval import Metrics, Strategy
from RL2048.Game.Game import Game, ACTION
from RL2048.Learning.GridPreprocessing import OnehotGridPreprocessing
from RL2048.Learning.Memory import Memory, saveMemory, loadMemory, BATCH_SIZE
from enum import Enum

ACT_DICT = {0: ACTION.LEFT, 1: ACTION.UP, 2: ACTION.RIGHT, 3: ACTION.DOWN}

class TRAIN(Enum):
    BY_ITSELF = 0    # Train by itself
    # As previous experience
    # If learning by itself that it will keep doing invalid move
    # and cause the training process very unefficient and meanless
    # So it need some guide or teacher to take it
    # either explore the world or teach it how to play
    WITH_RANDOM = 1  # Train with epsilon decay random
    TEST = -1        # Test mode

OPTIMIZER_CLASS = tf.train.AdamOptimizer

LEARNING_RATE_BASE = 1e-4  # Initial learning rate
LEARNING_RATE_DECAY = 0.98 # Learning rate decate per 100K

MODEL_SAVE_PATH = "./model_dqn/"
MODEL_NAME = "2048ai_dqn"

LOG_FILE = "training_dqn.log"
GAME_STATUS_YAML = "training_game_dqn.yaml" # store last game status

PLAY_GAMES = 1000000 # Total traing rounds (excluding exploring games)

GAMMA = 0.99

EPSILON_GREED = 0.8 # Maximum prob that take control by others
EPSILON_MIN = 0.1   # Minimum prob

GOOD_SAMPLE_PROB = 0.8
BAD_SAMPLE_PROB = 0.2


def epsilon_greedy_prob(step):
    """Decay of epsilon parameter    
    from EPSILON_GREED to EPSILON_MIN
    """
    c = PLAY_GAMES / 2 / np.log(EPSILON_GREED / EPSILON_MIN)
    x = EPSILON_GREED * np.exp(-step / c)
    return max(x, EPSILON_MIN)

def get_update_targetnet_op():
    """
    Set evaluation network's weights as target network's weights
    """
    target_params = tf.get_collection(forward.TARGET_NET_COLLECTION) # Get parameters of target network
    eval_params = tf.get_collection(forward.EVAL_NET_COLLECTION) # Get parameters of eval network
    
    replace_target_op = [tf.assign(t, e) for t, e in zip (target_params, eval_params)]
    return replace_target_op

def get_loss(target_y_placeholder, logits):
    """Calculate negative reward as loss
    
    Arguments:
        target_y_placeholder{tf.Tensor} -- Placeholder of 
        logits {tf.Tensor} -- Output from eval_network
    
    Returns:
        tf.Tensor -- loss
    """
    return tf.reduce_mean(tf.square(target_y_placeholder - logits))

def move_and_get_reward(action_num, last_reward, game_obj, cumulate_penalty, forceAction=None):
    """Do action and calculate reward and penalty
    
    Arguments:
        action_num {int} -- Aciton number corresponding to ACTION enum
        last_reward {float} -- Rewards from last aciton
        game_obj {Game} -- Game object
        cumulate_penalty {float} -- Cumulated penalty
        forceAction {ACTION} - ACTION force to do (default: None)
    
    Returns:
        reward -- Reward minus penalty
        penalty -- Penalty
    """

    if forceAction is None:
        isShift = game_obj.doAction(ACT_DICT[action_num])
    else:
        isShift = game_obj.doAction(forceAction)
    grid = game_obj.getCopyGrid()
    
    ## Reward
    # Because of penalty, last reward might be a negative value
    # But it shouldn't be a kind of 'reward' since we minus it.
    # So if it's negative, set to zero.
    # reward = min(-last_reward, 0)
    # reward += float(Metrics(grid).ThreeEvalValueWithScore(10, 10, 2, 7))
    
    # reward = float(grid.getScore()) # Use pure score as reward

    ## Penalty
    # penalty = cumulate_penalty
    if not isShift:
        # Invalid movement
        reward = 0.0 # set reward to -1
    else:
        reward = float(grid.getScore())

    return reward

def get_train_op(loss, global_step, Optimizer):

    # Applies exponential decay to the learning rate
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        100000, 
        LEARNING_RATE_DECAY)
    
    optimizer = Optimizer(learning_rate)
    train_op = optimizer.minimize(loss, global_step=global_step)
    
    return train_op

def backward(training_mode=TRAIN.BY_ITSELF, verbose=False):
    # Define placeholders
    state_batch_placeholder = tf.placeholder(tf.float32, shape=[None, forward.INPUT_NODE], name="state_batch")
    state_after_batch_placeholder = tf.placeholder(tf.float32, shape=[None, forward.INPUT_NODE], name="state__after_batch")
    target_y_placeholder = tf.placeholder(tf.float32, shape=[None, forward.OUTPUT_NODE], name="target_y")
    # eval_y_placeholder = tf.placeholder(tf.float32, shape=[None, forward.OUTPUT_NODE], name="eval_y")

    # train_batch_placeholder = tf.placeholder(tf.float32, shape=[None, forward.INPUT_NODE],name="train_batch")

    # Construct Q and Target Q network
    logits = forward.DQN_raw_forward(state_batch_placeholder) # Q network
    target_logits = forward.DQN_raw_delay_forward(state_after_batch_placeholder) # Target Q network

    #Q_j = tf.reduce_sum(tf.one_hot(tf.argmax(logits), 4) * logits, axis=1)
    loss = get_loss(target_y_placeholder, logits)

    global_step = tf.Variable(0, name='global_step', trainable=False)

    # Generate operations
    get_action_num_op = tf.argmax(logits, 1)
    
    train_op = get_train_op(loss, global_step, OPTIMIZER_CLASS)
    # train_op = get_train_op(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=target_logits), global_step, OPTIMIZER_CLASS)

    update_targetnet_op = get_update_targetnet_op()

    # Currently has bug
    # tf.summary.histogram("States", state_batch_placeholder)
    # tf.summary.scalar("Rewards", rewards_placeholder)
    # summary_op = tf.summary.merge_all()
    
    saver = tf.train.Saver()

    with tf.Session() as sess:

        game = Game() # Create a game

        # Restore saved session
        ckpt = tf.train.latest_checkpoint(MODEL_SAVE_PATH)
        if ckpt:
            saver.restore(sess, ckpt)
            game.restoreGame(GAME_STATUS_YAML) # restore last game status
            memory = loadMemory()
        else:
            init_global_op = tf.global_variables_initializer()
            init_local_op = tf.local_variables_initializer()
            sess.run(init_global_op)
            sess.run(init_local_op)
            sess.run(update_targetnet_op) # Set Q and target Q the same initial weight
            memory = Memory()
            
        reward = 0.0 # must be float to feed into placeholder
        penalty = 0.0
        forceAction = None

        # Fill minimal memory
        while memory.getMemoryNum(good=False) < BATCH_SIZE * BAD_SAMPLE_PROB: # Get enough bad memory
            state_t = game.getCopyGrid().getState()
            action = Strategy(game.getCopyGrid()).RandomValidMove()
            reward = move_and_get_reward(0, reward, game, penalty, action)
            state_t_1 = game.getCopyGrid().getState()
            isAlive = not game.gameOver
            memory.append(state_t, action.value, reward, state_t_1, isAlive)
            if not isAlive:
                game.newGame()

        # Training
        for i in range(PLAY_GAMES):
            forceMove = 0 # Force move in current game
            while not game.gameOver:
                state_batch = np.reshape(game.getCopyGrid().getState(), (1, forward.INPUT_NODE))
                # state_batch = OnehotGridPreprocessing(game.getCopyGrid()).FlattenBatch(2, onehot=True)            
                # Decision by Network
                actionNum = sess.run(get_action_num_op, feed_dict={state_batch_placeholder: state_batch})
                # Epsilon-greedy
                if training_mode == TRAIN.WITH_RANDOM:
                    # Take random action under some probability
                    if np.random.uniform() < epsilon_greedy_prob(i):
                        forceAction = Strategy(game.getCopyGrid()).RandomValidMove()
                        forceMove += 1
                    else:
                        forceAction = None

                if forceAction:
                    target_action = forceAction.value
                else:
                    target_action = actionNum[0]
                

                state_t = game.getCopyGrid().getState()
                reward = move_and_get_reward(actionNum[0], reward, game, penalty, forceAction)
                state_t_1 = game.getCopyGrid().getState()
                isAlive = not game.gameOver

                # Store current transction to memory
                memory.append(state_t, target_action, reward, state_t_1, isAlive)

                # Get a batch of transactions from memory
                sample = memory.getSampleBatch()
                state_before_batch = sample['state_before']
                state_before_batch = np.reshape(state_before_batch, (BATCH_SIZE, forward.INPUT_NODE))
                state_after_batch = sample['state_after']
                state_after_batch = np.reshape(state_after_batch, (BATCH_SIZE, forward.INPUT_NODE))

                Qtarget = sess.run(target_logits,feed_dict={state_after_batch_placeholder: state_after_batch})

                target_y = np.zeros((BATCH_SIZE, forward.OUTPUT_NODE))
                max_a_Q = np.max(Qtarget, axis=1)

                for j in range(BATCH_SIZE):
                    target_y[j] = sample['reward'][j] + GAMMA * max_a_Q[j] * sample['isAlive'][j] * np.eye(forward.OUTPUT_NODE)[sample['action'][j]]
                # target_y = sample['reward'] + GAMMA * np.max(Qtarget, axis=1) * sample['isAlive']

                if training_mode != TRAIN.TEST:
                    # Training mode
                    step, _ = sess.run([global_step, train_op],
                                        feed_dict={
                                            state_batch_placeholder: state_before_batch,
                                            state_after_batch_placeholder: state_after_batch,
                                            target_y_placeholder: np.reshape(target_y, (BATCH_SIZE, forward.OUTPUT_NODE)), # shape must be (?, 4)
                                        })

                # Not sure if random move will mislead network to learn the wrong aciton
                # that it thought it's good aciton since the random move did change the board
                # and get the reward (not deserve to the network output)

                if step % 200 == 0: # For every 200 steps, update target net's weight
                    sess.run(update_targetnet_op)

                if verbose:
                    os.system('clear')
                    print('Reward:', reward)
                    if training_mode != TRAIN.TEST:
                        print('Loss:', sess.run(loss, feed_dict={state_batch_placeholder: state_before_batch, target_y_placeholder: np.reshape(target_y, (BATCH_SIZE, forward.OUTPUT_NODE))}))
                    if forceAction:
                        print('Action:', forceAction, '(forced)')
                    else:
                        print('Action:', ACT_DICT[actionNum[0]])
                    game.printGrid()
            else:
                # When a game is over
                game.dumpLog(LOG_FILE)
                if training_mode in (TRAIN.WITH_RANDOM,):
                    with open(LOG_FILE, 'a') as log:
                        log.write(f"(Use {forceMove} force steps with epsilon ({epsilon_greedy_prob(i)}))\n")

                print('Epsilon:', epsilon_greedy_prob(i))
                game.newGame()
            
            print("Current Training Round", i+1, "Game Over\n\n")
            saver.save(sess, MODEL_SAVE_PATH + MODEL_NAME)
            game.saveGame(GAME_STATUS_YAML)
            saveMemory(memory)

def main():
    backward(training_mode=TRAIN.WITH_RANDOM, verbose=False)

if __name__ == '__main__':
    main()

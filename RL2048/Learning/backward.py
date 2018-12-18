import tensorflow as tf
import numpy as np
import os
import RL2048.Learning.forward as forward
from RL2048.Game.StateEval import Metrics, Strategy
from RL2048.Game.Game import Game, ACTION
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
    WITH_MCTS = 2    # Train with Monte Carlo Tree Search (as a teacher)
    TEST = -1        # Test mode

OPTIMIZER_CLASS = tf.train.AdamOptimizer

LEARNING_RATE_BASE = 1e-4  # Initial learning rate
LEARNING_RATE_DECAY = 0.98 # Learning rate decate per 100K

MODEL_SAVE_PATH = "./model/"
MODEL_NAME = "2048ai"

LOG_FILE = "training.log"
GAME_STATUS_YAML = "training_game.yaml" # store last game status

PLAY_GAMES = 1000000

EPSILON_GREED = 0.8 # Maximum prob that take control by others
EPSILON_MIN = 0.1   # Minimum prob

def epsilon_greedy_prob(step):
    """Decay of epsilon parameter
    
    from EPSILON_GREED to EPSILON_MIN
    """
    c = PLAY_GAMES / 2 / np.log(EPSILON_GREED / EPSILON_MIN)
    x = EPSILON_GREED * np.exp(-step / c)
    if x > EPSILON_MIN:
    	return x
    else:
    	return EPSILON_MIN
    
def get_loss(logits, action_take_placeholder, rewards_placeholder):
    """Calculate negative reward as loss
    
    Arguments:
        logits {tf.Tensor} -- Output from network
        action_take_placeholder {tf.Tensor} -- Placeholder of action taken
        rewards_placeholder {tf.Tensor} -- Placehoder of reward value
    
    Returns:
        tf.Tensor -- loss
    """

    neg_log_prob = tf.reduce_sum(-tf.log(tf.nn.softmax(logits)) * tf.one_hot(action_take_placeholder, forward.OUTPUT_NODE), axis=1)
    return tf.reduce_mean(neg_log_prob * rewards_placeholder)

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
    
    ## Penalty
    penalty = cumulate_penalty
    if not isShift:
        # Invalid movement
        penalty += 20.0
    else:
        penalty = 0.0
    
    ## Reward
    # Because of penalty, last reward might be a negative value
    # But it shouldn't be a kind of 'reward' since we minus it.
    # So if it's negative, set to zero.
    reward = min(-last_reward, 0)
    reward -= penalty
    reward += float(Metrics(grid).ThreeEvalValueWithScore(10, 10, 2, 7))
    # reward += float(grid.getScore()) # Use pure score as reward

    return reward, penalty
    
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
    state_batch_placeholder = tf.placeholder(tf.float32, shape=[None, forward.INPUT_NODE], name="state_batch")
    target_action_placeholder = tf.placeholder(tf.int32, shape=[None, 1], name="aciton_target")
    rewards_placeholder = tf.placeholder(tf.float32, shape=[None, 1], name="rewards")

    logits, action_prob = forward.forward(state_batch_placeholder)

    get_action_num_op = tf.argmax(action_prob, 1)

    loss = get_loss(logits, target_action_placeholder, rewards_placeholder)

    global_step = tf.Variable(0, name='global_step', trainable=False)

    train_op = get_train_op(loss, global_step, OPTIMIZER_CLASS)

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
        else:
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            
        reward = 0.0 # must be float to feed into placeholder
        penalty = 0.0
        forceAction = None

        for i in range(PLAY_GAMES):
            forceMove = 0
            while not game.gameOver:
                state_batch = np.reshape(game.getCopyGrid().getState(), (1, forward.INPUT_NODE))
                reward = float(game.getCurrentScore())

                # Decision by Network
                actionNum = sess.run(get_action_num_op, feed_dict={state_batch_placeholder: state_batch})

                if forceAction:
                    target_action = forceAction.value
                else:
                    target_action = actionNum[0]
                
                if training_mode != TRAIN.TEST:
                    # Training mode
                    step, _ = sess.run([global_step, train_op],
                                        feed_dict={
                                            state_batch_placeholder: state_batch,
                                            target_action_placeholder: [[target_action]], # shape must be (?, 1)
                                            rewards_placeholder: [[reward]] # shape must be (?, 1)
                                        })

                # Not sure if random move will mislead network to learn the wrong aciton
                # that it thought it's good aciton since the random move did change the board
                # and get the reward (not deserve to the network output)
                if training_mode == TRAIN.WITH_RANDOM:
                    # Take random action under some probability
                    if np.random.uniform() < epsilon_greedy_prob(i):
                        forceAction = Strategy(game.getCopyGrid()).RandomValidMove()
                        forceMove += 1
                    else:
                        forceAction = None
                elif training_mode == TRAIN.WITH_MCTS:
                    if np.random.uniform() < epsilon_greedy_prob(i):
                        forceAction = Strategy(game.getCopyGrid()).MCTSDFS()
                        forceMove += 1
                    else:
                        forceAction = None

                reward, penalty = move_and_get_reward(actionNum[0], reward, game, penalty, forceAction)
                if verbose:
                    os.system('clear')
                    print('Reward:', reward)
                    if training_mode != TRAIN.TEST:
                        print('Loss:', sess.run(loss, feed_dict={state_batch_placeholder: state_batch, rewards_placeholder: [[reward]], target_action_placeholder: [[target_action]]}))
                    if forceAction:
                        print('Action:', forceAction, '(forced)')
                    else:
                        print('Action:', ACT_DICT[actionNum[0]])
                    game.printGrid()

            else:
                # When a game is over
                game.dumpLog(LOG_FILE)
                if training_mode in (TRAIN.WITH_RANDOM, TRAIN.WITH_MCTS):
                    with open(LOG_FILE, 'a') as log:
                        log.write(f"(Use {forceMove} force steps with epsilon ({epsilon_greedy_prob(i)}))\n")

                print('Epsilon:', epsilon_greedy_prob(i))
                game.printStatus()
                game.printGrid()
                game.newGame()
            
            print("Current Training Round", i+1, "Game Over\n\n")
            saver.save(sess, MODEL_SAVE_PATH + MODEL_NAME)
            game.saveGame(GAME_STATUS_YAML)

def main():
    backward(training_mode=TRAIN.WITH_MCTS, verbose=True)

if __name__ == '__main__':
    main()

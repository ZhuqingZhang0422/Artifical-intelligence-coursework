# multiAgents_partner.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util
import pacman
from game import Agent
import util
import random

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best
        "Add more of your code here if you want to"
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        curFood = currentGameState.getFood()
        newFood = successorGameState.getFood()
        curPos = currentGameState.getPacmanPosition()
        newPos = successorGameState.getPacmanPosition()
        "*** YOUR CODE HERE ***"

        # Initialize relative scores
        score = successorGameState.getScore() - currentGameState.getScore()
        # Evaluating the food left
        cur_food_res = currentGameState.getFood().asList()
        suc_food_res = successorGameState.getFood().asList()
        if len(cur_food_res) > len(suc_food_res):
            score += 100
        else:
            score -= 100

        # Calculate the Pacman-remaining food distance
        cur_food_dist = 0
        cur_food_res = curFood.asList()
        for food_pos in cur_food_res:
            food_dist = util.manhattanDistance(food_pos,curPos)
            cur_food_dist += food_dist

        suc_food_dist = 0
        suc_food_res = newFood.asList()
        for food_pos in suc_food_res:
            food_dist = util.manhattanDistance(food_pos, newPos)
            suc_food_dist += food_dist

        if cur_food_dist > suc_food_dist:
            score += 100
        else:
            score -= 100

        # Calculate current state Pacman-Ghost distance
        ghost_dist_cur = float('inf')
        for ghost in currentGameState.getGhostStates():
            dist_cur = util.manhattanDistance(ghost.getPosition(),curPos)
            ghost_dist_cur = min(dist_cur,ghost_dist_cur)

        # Calculate successor state Pacman-Ghost distance
        ghost_dist_suc = float('inf')
        for ghost in successorGameState.getGhostStates():
            dist_suc = util.manhattanDistance(ghost.getPosition(),newPos)
            ghost_dist_suc = min(dist_suc,ghost_dist_suc)

        if ghost_dist_cur > ghost_dist_suc:
            score -= 50
        else:
            score += 50
        # If the successor GameState is the win state, stop here
        if successorGameState.isWin():
            return float('inf')
        # Give penalty if the pacman stopped
        if action == Directions.STOP:
            # Penalty for stop
            score -= 10
        return score

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """
    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        d = self.depth
        take_action = ''
        val_prev = -float('inf')
        actions = gameState.getLegalActions(0)
        for action in actions:
            suc = gameState.generateSuccessor(0, action)
            val_cur = self.min_lev(suc, d - 1,1)
            if val_cur > val_prev:
                take_action = action
                val_prev = val_cur
        return take_action

    def max_lev(self,gameState, d):
        v_max = -float('inf')
        if d == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        actions = gameState.getLegalActions(0)
        for action in actions:
            suc = gameState.generateSuccessor(0, action)
            v_max = max(self.min_lev(suc,d-1,1), v_max)
        return v_max

    def min_lev(self, gameState, d,agent_index):
        v_min = float('inf')
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        actions = gameState.getLegalActions(agent_index)
        for action in actions:
            suc = gameState.generateSuccessor(agent_index, action)
            if agent_index == gameState.getNumAgents() - 1:
                v_min = min(v_min, self.max_lev(suc, d))
            else:
                v_min = min(v_min, self.min_lev(suc, d, agent_index + 1))
        return v_min

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

    def max_lev(self, gameState, d, alpha, beta):
        v_max = -float('inf')
        if d == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        actions = gameState.getLegalActions(0)
        for action in actions:
            suc = gameState.generateSuccessor(0, action)
            v_max = max(self.min_lev(suc, d - 1, 1), v_max)
        return v_max

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

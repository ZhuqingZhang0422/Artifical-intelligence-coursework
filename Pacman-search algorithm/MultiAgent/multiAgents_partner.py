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

#
#from pacman import GameState


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
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        #return successorGameState.getScore()

        curFood = currentGameState.getFood()
        curPos = currentGameState.getPacmanPosition()
        # Initialize relative scores
        score = successorGameState.getScore() - currentGameState.getScore()
        # Evaluating the food left
        cur_food_res = currentGameState.getFood().asList()
        suc_food_res = successorGameState.getFood().asList()
        if len(cur_food_res) > len(suc_food_res):
            score += 100
        else:
            score -= 100 #?
        # Calculate the Pacman-remaining food distance
        cur_food_dist = 0
        cur_food_res = curFood.asList()
        for food_pos in cur_food_res:
            food_dist = util.manhattanDistance(food_pos, curPos)
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
            dist_cur = util.manhattanDistance(ghost.getPosition(), curPos)
            ghost_dist_cur = min(dist_cur, ghost_dist_cur)

        # Calculate successor state Pacman-Ghost distance
        ghost_dist_suc = float('inf')
        for ghost in successorGameState.getGhostStates():
            dist_suc = util.manhattanDistance(ghost.getPosition(), newPos)
            ghost_dist_suc = min(dist_suc, ghost_dist_suc)

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
        #util.raiseNotDefined()

        def isend(s): # how do we know the terminal
            return s.isWin() or s.isLose()
        #def Player(s):
            #return 0
        # v1
        '''
        def value(s,d,player):
            if d==self.depth or isend(s):
                return self.evaluationFunction(s),[]
            elif player ==0: # this is the pacman turn
                return max_value(s,d)
            else: # this is the ghost turn
                return min_value(s,d,player)


        def max_value(s,d): # this is for the agent 0 move
            v=[]
            for each_action in s.getLegalActions(agentIndex=0):
                successor_state = s.generateSuccessor(agentIndex=0, action=each_action)
                temp=value(successor_state,d,1)[0]
                if v==[]:
                    v.append(temp)
                    actiontotake = each_action
                else:
                    if temp>v[0]:
                        v[0]=temp
                        actiontotake=each_action

            return (v[0],actiontotake)


        def min_value(s, d, ghost_index):  # this is for the ghost 1 move
            v = []
            new_index=(ghost_index+1)%s.getNumAgents()
            new_d=d+(ghost_index+1)//s.getNumAgents()

            for each_action in s.getLegalActions(agentIndex=ghost_index):
                successor_state = s.generateSuccessor(agentIndex=ghost_index,action=each_action)
                temp=value(successor_state, new_d,new_index)[0]
                if v == []:
                    v.append(temp)
                    actiontotake=each_action
                else:
                    if temp<v[0]:
                        v [0]= temp
                        actiontotake = each_action

            return v[0],actiontotake
        '''
        #v2 Simplified version from v1/ uniform value since we know there will be 0 max and n min total n+1
        def value(s,d,player):
            if d==self.depth or isend(s):
                return self.evaluationFunction(s),[]
            else:
                v = []
                new_index = (player + 1) % s.getNumAgents()
                new_d = d + (player + 1) // s.getNumAgents()

                for each_action in s.getLegalActions(agentIndex=player):
                    successor_state = s.generateSuccessor(agentIndex=player, action=each_action)
                    temp = value(successor_state, new_d, new_index)[0]
                    if v == []:
                        v.append(temp)
                        actiontotake = each_action
                    elif player==0:
                        if temp > v[0]:
                            v[0] = temp
                            actiontotake = each_action
                    else:
                        if temp < v[0]:
                            v[0] = temp
                            actiontotake = each_action

                return v[0], actiontotake

        the_value,action_final= value(s=gameState, d=0, player=0)

        return action_final

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        def isend(s):  # how do we know the terminal
            return s.isWin() or s.isLose()
        def value(s,d,player,alpha,beta):
            if d==self.depth or isend(s):
                return self.evaluationFunction(s),[]
            elif player ==0: # this is the pacman turn
                return max_value(s,d,alpha,beta)
            else: # this is the ghost turn
                return min_value(s,d,player,alpha,beta)


        def max_value(s,d,alpha,beta): # this is for the agent 0 move
            v=[]
            for each_action in s.getLegalActions(agentIndex=0):
                successor_state = s.generateSuccessor(agentIndex=0, action=each_action)
                temp=value(successor_state,d,1,alpha,beta)[0]
                if v==[]:
                    v.append(temp)
                    actiontotake = each_action
                else:
                    if temp>v[0]:
                        v[0]=temp
                        actiontotake=each_action
                if v[0]>beta: # or >=
                    return (v[0],actiontotake)
                alpha=max(alpha,v[0])
            return (v[0],actiontotake)


        def min_value(s, d, ghost_index,alpha,beta):  # this is for the ghost 1 move
            v = []
            new_index=(ghost_index+1)%s.getNumAgents()
            new_d=d+(ghost_index+1)//s.getNumAgents()

            for each_action in s.getLegalActions(agentIndex=ghost_index):
                successor_state = s.generateSuccessor(agentIndex=ghost_index,action=each_action)
                temp=value(successor_state, new_d,new_index,alpha,beta)[0]
                if v == []:
                    v.append(temp)
                    actiontotake=each_action
                else:
                    if temp<v[0]:
                        v [0]= temp
                        actiontotake = each_action
                if v[0]<alpha: #? or <=
                    return (v[0],actiontotake)
                beta=min(beta,v[0])
            return v[0],actiontotake

        the_value,action_final= value(s=gameState, d=0, player=0, alpha=float('-inf'), beta=float('inf'))

        return action_final


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
        #util.raiseNotDefined()
        def isend(s): # how do we know the terminal
            return s.isWin() or s.isLose()
        #def Player(s):
            #return 0
        # v1

        def value(s,d,player):
            if d==self.depth or isend(s):
                return self.evaluationFunction(s),[]
            elif player ==0: # this is the pacman turn
                return max_value(s,d)
            else: # this is the ghost turn
                return min_value(s,d,player)


        def max_value(s,d): # this is for the agent 0 move
            v=[]
            for each_action in s.getLegalActions(agentIndex=0):
                successor_state = s.generateSuccessor(agentIndex=0, action=each_action)
                temp=value(successor_state,d,1)[0]
                if v==[]:
                    v.append(temp)
                    actiontotake = each_action
                else:
                    if temp>v[0]:
                        v[0]=temp
                        actiontotake=each_action

            return (v[0],actiontotake)


        def min_value(s, d, ghost_index):  # this is for the ghost 1 move
            v=0
            n=0
            new_index=(ghost_index+1)%s.getNumAgents()
            new_d=d+(ghost_index+1)//s.getNumAgents()

            for each_action in s.getLegalActions(agentIndex=ghost_index):
                successor_state = s.generateSuccessor(agentIndex=ghost_index,action=each_action)
                temp=value(successor_state, new_d,new_index)[0]
                v+=temp
                n+=1
            v=v/n

            return v,[]

        the_value,action_final= value(s=gameState, d=0, player=0)

        return action_final
def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    #util.raiseNotDefined()
    # Useful information you can extract from a GameState (pacman.py)
    curFood = currentGameState.getFood()
    curPos = currentGameState.getPacmanPosition()
    # Initialize relative scores
    score = currentGameState.getScore()
    # Evaluating the food left
    cur_food_res = currentGameState.getFood().asList()
    # Calculate the Pacman-remaining food distance
    cur_food_dist = []
    cur_food_res = curFood.asList()
    total_food_number=len(cur_food_res)
    for food_pos in cur_food_res:
        food_dist = util.manhattanDistance(food_pos, curPos)
        cur_food_dist.append(food_dist)
    # Calculate current state Pacman-Ghost distancE
    ghost_dist_cur=[util.manhattanDistance(ghost.getPosition(), curPos) for ghost in currentGameState.getGhostStates()]
    # ghost scare
    GhostStates = currentGameState.getGhostStates()
    ScaredTimes = [ghostState.scaredTimer for ghostState in GhostStates]
    if cur_food_dist :
        score += 1 / max(cur_food_dist)+total_food_number
    if sum(ScaredTimes)>0:
        score+=sum(ScaredTimes)/len(ScaredTimes)-max(ghost_dist_cur)
    else:
        score+=min(ghost_dist_cur)
    return score

# Abbreviation
better = betterEvaluationFunction

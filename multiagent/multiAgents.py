# multiAgents.py
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

from game import Agent

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
        some Directions.X for some X in the set {North, South, West, East, Stop}
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
        # newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        # compute dist to closest ghost
        ghostDist = "inf"
        for ghostState in newGhostStates:
            # print type(ghostState), dir(ghostState)
            # <type 'instance'>
            # ['__doc__', '__eq__', '__hash__', '__init__', '__module__', '__str__', 'configuration',
            # 'copy', 'getDirection', 'getPosition', 'isPacman', 'numCarrying', 'numReturned', 'scaredTimer', 'start']
            ghostDist =  min(ghostDist, manhattanDistance(newPos, ghostState.getPosition()))
        penalty = 0
        if ghostDist < 2:
            penalty = 1000

        # compute dist to closest food
        foodDist = 1000
        for food in newFood.asList():
            # print type(newFood), dir(newFood)
            #<type 'instance'>
            # ['CELLS_PER_INT', '__doc__', '__eq__', '__getitem__', '__hash__', '__init__', '__module__',
            # '__setitem__', '__str__', '_cellIndexToPosition', '_unpackBits', '_unpackInt',
            # 'asList', 'copy', 'count', 'data', 'deepCopy', 'height', 'packBits', 'shallowCopy', 'width']
            # print "food", food
            foodDist = min(foodDist, manhattanDistance(newPos, food))

        return successorGameState.getScore() - penalty - 0.01*foodDist

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
        # print "\nSTART"
        def merge_values(agent, state, remaining_depth, mode):
            """
            mode = 0 => minimizer
            mode = 1 => maximizer
            """
            successors = [state.generateSuccessor(agent, action) for action in state.getLegalActions(agent)]
            nextAgent = (agent + 1) % state.getNumAgents()

            rem_depth = remaining_depth
            if not nextAgent: # (not ghost)
                rem_depth -= 1 # decrement depth
            # print "min_value nextAgent", nextAgent, "sucs_num = ", len(successors)

            if mode:
                best_score = float("-inf")
                for suc in successors:
                    best_score = max(best_score, mm_value(nextAgent, suc, rem_depth))
            else:
                best_score = float("inf")
                for suc in successors:
                    best_score = min(best_score, mm_value(nextAgent, suc, rem_depth))
            # print "min_value local best_score", best_score
            return best_score

        def mm_value(agent, state, remaining_depth):
            # print "mm_value agent", agent, "remdepth", remaining_depth
            if (state.isWin() or state.isLose() or not remaining_depth):
                # print "win?", state.isWin(), "lose?", state.isLose(), "score =", self.evaluationFunction(state)
                return self.evaluationFunction(state)
            mode = 1   # maximizer if pacman
            if agent:  # minimizer if ghost
                mode = 0
            return merge_values(agent, state, remaining_depth, mode)


        pacActions = gameState.getLegalActions(0)
        scores = {}
        for action in pacActions:
            scores[action] = mm_value(1, gameState.generateSuccessor(0, action), self.depth)
        # print scores
        ## find max score
        # best_act, best_score  = None, self.evaluationFunction(gameState)
        best_act, best_score  = None, float("-inf")
        for act, value in scores.iteritems():
            # print "act, value, best_score", act, value, best_score
            if value > best_score:
                # print "new best_score"
                best_act = act
                best_score = value
        # print "best_act", best_act, "best_score", best_score
        return best_act


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # print "\nSTART"
        # print "gameState.getNumAgents()", gameState.getNumAgents()
        # print "func_name evaluationFunction", self.evaluationFunction.func_name
        def merge_values(agent, state, remaining_depth, mode, a, b):
            """
            mode = 0 => minimizer
            mode = 1 => maximizer
            """
            # best_act, best_score  = None, self.evaluationFunction(gameState)

            # successors = [state.generateSuccessor(agent, action) for action in state.getLegalActions(agent)]
            nextAgent = (agent + 1) % state.getNumAgents()

            rem_depth = remaining_depth
            if not nextAgent: # (not ghost)
                rem_depth -= 1 # decrement depth
            # print "merge_values mode =", mode, "nextAgent =", nextAgent, "sucs_num = ", len(successors), "a =", a, "b =",b

            if mode:
                best_score = float("-inf")
                for action in state.getLegalActions(agent):
                    suc = state.generateSuccessor(agent, action)
                # for suc in successors: # GREAT ERROR! DO NOT EXPAND SUCCESSORS, AFTER 1st ACTION THEY CAN BE PRUNED!
                    new_score = mm_value(nextAgent, suc, rem_depth, a, b)
                    # print "maximize", "action = ", action, "a =", a, "b =",b, "best_score =", best_score, "new_score", new_score
                    best_score = max(best_score, new_score)
                    if best_score > b: return best_score
                    a = max(a, best_score)
            else:
                best_score = float("inf")
                for action in state.getLegalActions(agent):
                    suc = state.generateSuccessor(agent, action)
                    new_score = mm_value(nextAgent, suc, rem_depth, a, b)
                    # print "minimize", "action = ", action, "a =", a, "b =",b, "best_score =", best_score, "new_score", new_score
                    best_score = min(best_score, new_score)
                    if best_score < a: return best_score
                    b = min(b, best_score)
            # print "min_value local best_score", best_score
            return best_score

        def mm_value(agent, state, remaining_depth, a ,b):
            # print "mm_value agent", agent, "remdepth", remaining_depth
            if (state.isWin() or state.isLose() or not remaining_depth):
                # print "win?", state.isWin(), "lose?", state.isLose(), "score =", self.evaluationFunction(state)
                # print "mm_value returned", self.evaluationFunction(state)
                return self.evaluationFunction(state)
            mode = 1   # maximizer if pacman
            if agent:  # minimizer if ghost
                mode = 0
            return merge_values(agent, state, remaining_depth, mode, a, b)


        pacActions = gameState.getLegalActions(0)
        scores = {}
        # initially do not let pruning
        a = float("-inf")
        b = float("inf")
        for action in pacActions:
            score = mm_value(1, gameState.generateSuccessor(0, action), self.depth, a, b)
            scores[action] = score
            # print "returned score to root ", score, "a was =", a, "b was =",b
            a = max(a, score) # a and only a!
        # print scores

        ## find max score
        best_act, best_score  = None, float("-inf")
        for act, value in scores.iteritems():
            # print "act, value, best_score", act, value, best_score
            if value > best_score:
                # print "new best_score"
                best_act = act
                best_score = value
        # print "best_act", best_act, "best_score", best_score
        return best_act

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
        def merge_values(agent, state, remaining_depth, mode):
            """
            mode = 0 => minimizer
            mode = 1 => maximizer
            """
            successors = [state.generateSuccessor(agent, action) for action in state.getLegalActions(agent)]
            nextAgent = (agent + 1) % state.getNumAgents()

            rem_depth = remaining_depth
            if not nextAgent: # (not ghost)
                rem_depth -= 1 # decrement depth
            # print "min_value nextAgent", nextAgent, "sucs_num = ", len(successors)

            if mode:
                best_score = float("-inf")
                for suc in successors:
                    best_score = max(best_score, mm_value(nextAgent, suc, rem_depth))
            else:
                score_sum = 0.0
                leng = len(successors)
                for suc in successors:
                    score_sum +=  mm_value(nextAgent, suc, rem_depth)
                best_score = float(score_sum)/leng
            # print "min_value local best_score", best_score
            return best_score
        def mm_value(agent, state, remaining_depth):
            # print "mm_value agent", agent, "remdepth", remaining_depth
            if (state.isWin() or state.isLose() or not remaining_depth):
                # print "win?", state.isWin(), "lose?", state.isLose(), "score =", self.evaluationFunction(state)
                return self.evaluationFunction(state)
            mode = 1   # maximizer if pacman
            if agent:  # minimizer if ghost
                mode = 0
            return merge_values(agent, state, remaining_depth, mode)


        pacActions = gameState.getLegalActions(0)
        scores = {}
        for action in pacActions:
            scores[action] = mm_value(1, gameState.generateSuccessor(0, action), self.depth)
        # print scores
        ## find max score
        best_act, best_score  = None, float("-inf")
        for act, value in scores.iteritems():
            # print "act, value, best_score", act, value, best_score
            if value > best_score:
                # print "new best_score"
                best_act = act
                best_score = value
        # print "best_act", best_act, "best_score", best_score
        return best_act

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).
      DESCRIPTION:
      currentGameState.getScore() I subtract:
      - ghostPenalty ( = 0 if closest ghost is not closer than 1 step,
                     = 1000 if its close (dangerous position!,
                     - -1000 if it is close and scared (catch him now!);
       - 0.01*foodDist (distance to closest food so ceteris paribus we prefer way to food )
       and
       - 0.01*PelletDist (the same logic as food)
    """
    "*** YOUR CODE HERE ***"
    # Useful information you can extract from a GameState (pacman.py)
    # print dir(currentGameState) # ['__doc__', '__eq__', '__hash__', '__init__', '__module__', '__str__',
    #  'data', 'deepCopy', 'explored', 'generatePacmanSuccessor', 'generateSuccessor', 'getAndResetExplored',
    #  'getCapsules', 'getFood', 'getGhostPosition', 'getGhostPositions', 'getGhostState', 'getGhostStates',
    # 'getLegalActions', 'getLegalPacmanAct
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    pellets = currentGameState.getCapsules()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    # compute dist to closest ghost
    ghostDist = "inf"
    closestGhost = "undefined"
    for i, ghostState in enumerate(newGhostStates):
        # print type(ghostState), dir(ghostState)
        # <type 'instance'>
        # ['__doc__', '__eq__', '__hash__', '__init__', '__module__', '__str__', 'configuration',
        # 'copy', 'getDirection', 'getPosition', 'isPacman', 'numCarrying', 'numReturned', 'scaredTimer', 'start']
        ghostDist =  min(ghostDist, manhattanDistance(newPos, ghostState.getPosition()))
        closestGhost = i

    ghostPenalty = 0
    if newGhostStates: # ghost exists
        if ghostDist < 2: # if it's close to us (else ignore it)
            if newScaredTimes[closestGhost]: # if closest ghost scared, its very good position!
                ghostPenalty= -1000
            else: # its very bad, don't go there!
                ghostPenalty = 1000


    # compute dist to closest food
    foodDist = 1000
    for food in newFood.asList():
        # print type(newFood), dir(newFood)
        #<type 'instance'>
        # ['CELLS_PER_INT', '__doc__', '__eq__', '__getitem__', '__hash__', '__init__', '__module__',
        # '__setitem__', '__str__', '_cellIndexToPosition', '_unpackBits', '_unpackInt',
        # 'asList', 'copy', 'count', 'data', 'deepCopy', 'height', 'packBits', 'shallowCopy', 'width']
        # print "food", food
        foodDist = min(foodDist, manhattanDistance(newPos, food))

    # compute dist to closest pellet
    PelletDist = 1000
    for pelletPos in pellets:
        PelletDist = min(PelletDist, manhattanDistance(newPos, pelletPos))
    if PelletDist < 2:
        PelletDist *= -100
    # myScore of -q q5 --no graphics = 1115.1
    return currentGameState.getScore() - ghostPenalty - 0.01*foodDist - 0.01*PelletDist

# Abbreviation
better = betterEvaluationFunction
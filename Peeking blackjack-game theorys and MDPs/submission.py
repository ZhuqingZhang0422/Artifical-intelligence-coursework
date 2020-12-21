import collections, util, math, random

############################################################
# Problem 4.1.1
def computeQ(mdp, V, state, action):
    """
    Return Q(state, action) based on V(state).  Use the properties of the
    provided MDP to access the discount, transition probabilities, etc.
    In particular, MDP.succAndProbReward() will be useful (see util.py for
    documentation).  Note that |V| is a dictionary.  
    """
    q = 0
    for suc, pro, rew in mdp.succAndProbReward(state,action):
        q += pro * (rew + mdp.discount()*V[suc])
    return q

############################################################
# Problem 4.1.2

def policyEvaluation(mdp, V, pi, epsilon=0.001):
    """
    Return the value of the policy |pi| up to error tolerance |epsilon|.
    Initialize the computation with |V|.  Note that |V| and |pi| are
    dictionaries.
    """
    # BEGIN_YOUR_CODE (around 7 lines of code expected)
    while True:
        error = 0
        for state in mdp.states:
            q = computeQ(mdp, V, state, pi[state])
            error = max(error, abs(q - V[state]))
            V[state] = q
        if error < epsilon:
            break
    return V

############################################################
# Problem 4.1.3

def computeOptimalPolicy(mdp, V):
    """
    Return the optimal policy based on V(state).
    You might find it handy to call computeQ().  Note that |V| is a
    dictionary.
    """
    pi = {}
    for state in mdp.states:
        val_new = -float("inf")
        for action in mdp.actions(state):
            val, val_new = val_new, computeQ(mdp,V,state,action)
            if val < val_new:
                pi[state] = action
    return pi

############################################################
# Problem 4.1.4
class PolicyIteration(util.MDPAlgorithm):
    def solve(self, mdp, epsilon=0.001):
        mdp.computeStates()
        # compute |V| and |pi|, which should both be dicts
        # BEGIN_YOUR_CODE (around 8 lines of code expected)
        # raise Exception("Not implemented yet")
        #  Initialize
        V = {state:0 for state in mdp.states}
        pi = computeOptimalPolicy(mdp,V)
        while True:
            V = policyEvaluation(mdp, V, pi, epsilon)
            pi_new = computeOptimalPolicy(mdp,V)
            if pi == pi_new:
                break
            else:
                pi = pi_new
        self.pi = pi
        self.V = V
############################################################
# Problem 4.1.5
class ValueIteration(util.MDPAlgorithm):
    def solve(self, mdp, epsilon=0.001):
        mdp.computeStates()
        V = collections.Counter()
        while True:
            V_new, error = {}, 0
            for state in mdp.states:
                V_new[state] = max(computeQ(mdp, V, state, action) for action in mdp.actions(state))
                error = max(error,abs(V[state]-V_new[state]))
            if error < epsilon:
                V = V_new
                break
            V = V_new
        pi = computeOptimalPolicy(mdp, V)
        self.pi = pi
        self.V = V

############################################################
# Problem 4.1.6
# If you decide 1f is true, prove it in writeup.pdf and put "return None" for
# the code blocks below.  If you decide that 1f is false, construct a
# counterexample by filling out this class and returning an alpha value in
# counterexampleAlpha().
class CounterexampleMDP(util.MDP):
    def __init__(self):
        raise Exception("Not implemented yet")

    def startState(self):
        raise Exception("Not implemented yet")

    # Return set of actions possible from |state|.
    def actions(self, state):
        raise Exception("Not implemented yet")

    # Return a list of (newState, prob, reward) tuples corresponding to edges
    # coming out of |state|.
    def succAndProbReward(self, state, action):
        raise Exception("Not implemented yet")

    def discount(self):
        raise Exception("Not implemented yet")

def counterexampleAlpha():
    raise Exception("Not implemented yet")

############################################################
# Problem 4.2.1

class BlackjackMDP(util.MDP):
    def __init__(self, cardValues, multiplicity, threshold, peekCost):
        """
        cardValues: list of integers (face values for each card included in the deck)
        multiplicity: single integer representing the number of cards with each face value
        threshold: maximum number of points (i.e. sum of card values in hand) before going bust
        peekCost: how much it costs to peek at the next card
        """
        self.cardValues = cardValues
        self.multiplicity = multiplicity
        self.threshold = threshold
        self.peekCost = peekCost

    # Return the start state.
    # Look closely at this function to see an example of state representation for our Blackjack game.
    # Each state is a tuple with 3 elements:
    #   -- The first element of the tuple is the sum of the cards in the player's hand.
    #   -- If the player's last action was to peek, the second element is the index
    #      (not the face value) of the next card that will be drawn; otherwise, the
    #      second element is None.
    #   -- The third element is a tuple giving counts for each of the cards remaining
    #      in the deck, or None if the deck is empty or the game is over (e.g. when
    #      the user quits or goes bust).

    def startState(self):
        # First:    int--- sum of cards
        # Second:   Null/int --- index of the next card in self.cardValues
        # Third:    tuple --- Eg:(10,10,10) numbers of each type of cards remaining in the deck
        return (0, None, len(self.cardValues)*(self.multiplicity,))

    # Return set of actions possible from |state|.
    # You do not need to modify this function.
    # All logic for dealing with end states should be placed into the succAndProbReward function below.
    def actions(self, state):
        # No constrains for taking actions at any state
        return ['Quit','Peek','Take']

    # Given a |state| and |action|, return a list of (newState, prob, reward) tuples
    # corresponding to the states reachable from |state| when taking |action|.
    # A few reminders:
    # * Indicate a terminal state (after quitting, busting, or running out of cards)
    #   by setting the deck to None.
    # * If |state| is an end state, you should return an empty list [].
    # * When the probability is 0 for a transition to a particular new state,
    #   don't include that state in the list returned by succAndProbReward.
    def succAndProbReward(self, state, action):
        card_sum = state[0]
        peek_last = state[1]
        deck_cur = state[2]
        if deck_cur == None:
            card_count = 0
        else:
            cards_count = sum(deck_cur)
        # res contain list of tuples (suc,prob,rew)
        res = []

        def deck_remove(deck,ind):
            '''
            :param mul: current state of cards in deck
            :param ind: index of card value
            :return: updated card state in deck
            '''
            # remove card in deck and
            deck_new = []
            for i in range(len(deck)):
                if i == ind:
                    deck_new.append(deck[i] - 1)
                else:
                    deck_new.append(deck[i])
            return tuple(deck_new)

        # If out of cards the res is set as []
        if deck_cur == None:
            return res

        # If quit the game
        if action == 'Quit':
            # if sumCards <= self.threshold:
            suc = (card_sum, None, None)
            res.append((suc, 1, card_sum))

        if action == 'Take':
            if peek_last == None:
                # Calculate the None peek successors
                for i in range(len(deck_cur)):
                    card_left = deck_cur[i]
                    prob = float(card_left)/cards_count
                    if card_left > 0:
                        card_sum_new = card_sum + self.cardValues[i]
                        if card_sum_new > self.threshold:
                            suc = (card_sum_new, None, None)
                            res.append((suc,prob,0))
                        elif cards_count == 1:
                            suc = (card_sum_new,None,None)
                            rew = card_sum_new
                            res.append((suc,prob,rew))
                        else:
                            deck_update = deck_remove(deck_cur,i)
                            suc = (card_sum_new, None, deck_update)
                            res.append((suc, prob, 0))
            else:
                # Calculate the peek successors
                card_sum_new = card_sum + self.cardValues[peek_last]
                if card_sum_new > self.threshold:
                    suc = (card_sum_new, None, None)
                    res.append((suc, 1, 0))
                else:
                    deck_update = deck_remove(deck_cur,peek_last)
                    if card_sum == 1:
                        suc = (card_sum_newm, None, None)
                        rew = card_sum_new
                        res.append((suc,1,rew))
                    else:
                        deck_update = deck_remove(deck_cur,peek_last)
                        suc = (card_sum_new, None, deck_update)
                        res.append((suc,1,0))

        if action == 'Peek':
            # if peek_last != None we don't have to consider the successor since it is already covered in the action of quit
            if peek_last == None:
                for i in range(len(deck_cur)):
                    rew = -self.peekCost
                    card_left = deck_cur[i]
                    prob = float(card_left)/float(cards_count)
                    if card_left > 0:
                        suc = (card_sum, i, deck_cur)
                        res.append((suc, prob, rew))
        return res

    def discount(self):
        #No discount factor for this problem
        return 1

############################################################
# Problem 4.2.2

def peekingMDP():
    """
    Return an instance of BlackjackMDP where peeking is the optimal action at
    least 10% of the time.
    """
    mdp = BlackjackMDP(cardValues = [5,10,20], multiplicity=10,
                       threshold = 20, peekCost=1)
    return mdp


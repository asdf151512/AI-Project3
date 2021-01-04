# wumpus_planners.py
# ------------------
# Licensing Information:
# Please DO NOT DISTRIBUTE OR PUBLISH solutions to this project.
# You are free to use and extend these projects for EDUCATIONAL PURPOSES ONLY.
# The Hunt The Wumpus AI project was developed at University of Arizona
# by Clay Morrison (clayton@sista.arizona.edu), spring 2013.
# This project extends the python code provided by Peter Norvig as part of
# the Artificial Intelligence: A Modern Approach (AIMA) book example code;
# see http://aima.cs.berkeley.edu/code.html
# In particular, the following files come directly from the AIMA python
# code: ['agents.py', 'logic.py', 'search.py', 'utils.py']
# ('logic.py' has been modified by Clay Morrison in locations with the
# comment 'CTM')
# The file ['minisat.py'] implements a slim system call wrapper to the minisat
# (see http://minisat.se) SAT solver, and is directly based on the satispy
# python project, see https://github.com/netom/satispy .

from wumpus_environment import *
from wumpus_kb import *
import search

#-------------------------------------------------------------------------------
# Distance fn
#-------------------------------------------------------------------------------

def manhattan_distance_with_heading(current, target):
    """
    Return the Manhattan distance + any turn moves needed
        to put target ahead of current heading
    current: (x,y,h) tuple, so: [0]=x, [1]=y, [2]=h=heading)
    heading: 0:^:north 1:<:west 2:v:south 3:>:east
    """
    md = abs(current[0] - target[0]) + abs(current[1] - target[1])
    if current[2] == 0:   # heading north
        # Since the agent is facing north, "side" here means
        # whether the target is in a row above or below (or
        # the same) as the agent.
        # (Same idea is used if agent is heading south)
        side = (current[1] - target[1])
        if side > 0:
            md += 2           # target is behind: need to turns to turn around
        elif side <= 0 and current[0] != target[0]:
            md += 1           # target is ahead but not directly: just need to turn once
        # note: if target straight ahead (curr.x == tar.x), no turning required
    elif current[2] == 1: # heading west
        # Now the agent is heading west, so "side" means
        # whether the target is in a column to the left or right
        # (or the same) as the agent.
        # (Same idea is used if agent is heading east)
        side = (current[0] - target[0])
        if side < 0:
            md += 2           # target is behind
        elif side >= 0 and current[1] != target[1]:
            md += 1           # target is ahead but not directly
    elif current[2] == 2: # heading south
        side = (current[1] - target[1])
        if side < 0:
            md += 2           # target is behind
        elif side >= 0 and current[0] != target[0]:
            md += 1           # target is ahead but not directly
    elif current[2] == 3: # heading east
        side = (current[0] - target[0])
        if side > 0:
            md += 2           # target is behind
        elif side <= 0 and current[1] != target[1]:
            md += 1           # target is ahead but not directly
    return md


#-------------------------------------------------------------------------------
# Plan Route
#-------------------------------------------------------------------------------

def plan_route(current, heading, goals, allowed):
    """
    Given:
       current location: tuple (x,y)
       heading: integer representing direction
       gaals: list of one or more tuple goal-states
       allowed: list of locations that can be moved to
    ... return a list of actions (no time stamps!) that when executed
    will take the agent from the current location to one of (the closest)
    goal locations
    You will need to:
    (1) Construct a PlanRouteProblem that extends search.Problem
    (2) Pass the PlanRouteProblem as the argument to astar_search
        (search.astar_search(Problem)) to find the action sequence.
        Astar returns a node.  You can call node.solution() to exract
        the list of actions.
    NOTE: represent a state as a triple: (x, y, heading)
          where heading will be an integer, as follows:
          0='north', 1='west', 2='south', 3='east'
    """

    # Ensure heading is a in integer form
    if isinstance(heading,str):
        heading = Explorer.heading_str_to_num[heading]

    if goals and allowed:
        prp = PlanRouteProblem((current[0], current[1], heading), goals, allowed)
        # NOTE: PlanRouteProblem will include a method h() that computes
        #       the heuristic, so no need to provide here to astar_search()
        node = search.astar_search(prp)
        if node:
            return node.solution()
    
    # no route can be found, return empty list
    return []

#-------------------------------------------------------------------------------

class PlanRouteProblem(search.Problem):
    def __init__(self, initial, goals, allowed):
        """ Problem defining planning of route to closest goal
        Goal is generally a location (x,y) tuple, but state will be (x,y,heading) tuple
        initial = initial location, (x,y) tuple
        goals   = list of goal (x,y) tuples
        allowed = list of state (x,y) tuples that agent could move to """
        self.initial = initial # initial state
        self.goals = goals     # list of goals that can be achieved
        self.allowed = allowed # the states we can move into

    def h(self, node):
        """
        Heuristic that will be used by search.astar_search()
        """
        distanceToGoals = [manhattan_distance_with_heading(node.state,goal) for goal in self.goals]
        return min(distanceToGoals)

    def actions(self, state):
        """
        Return list of allowed actions that can be made in state
        0 = North
        1 = West
        2 = South
        3 = East
        Computing the distance of forward/TurnLeft/TurnRight
        Then choose the best route if possible
        If all three routes are the same, then choose randomly
        """
        
        doingactions = ['Forward','TurnRight','TurnLeft']
        forward_state=()
        left_state=()
        right_state=()
        #remember the new state of going forward/turn left/turn right in order to compute hamming distance
        if state[2] == 0:
            forward_state = (state[0],state[1]+1,state[2])
            left_state=(state[0],state[1],1)
            right_state=(state[0],state[1],3)
        elif state[2] == 1:
            forward_state = (state[0]-1,state[1],state[2])
            left_state=(state[0],state[1],2)
            right_state=(state[0],state[1],0)
        elif state[2] == 2:
            forward_state = (state[0],state[1]-1,state[2])
            left_state=(state[0],state[1],3)
            right_state=(state[0],state[1],1)
        elif state[2] == 3:
            forward_state = (state[0]+1,state[1],state[2])
            left_state=(state[0],state[1],0)
            right_state=(state[0],state[1],2)
        forward_distance = float('inf')
        left_distance = float('inf')
        right_distance = float('inf')
        for goal in self.goals:
            if (forward_state[0],forward_state[1]) in self.allowed:
                forward_distance = min(forward_distance,manhattan_distance_with_heading(forward_state,goal))
            left_distance = min(left_distance,manhattan_distance_with_heading(left_state,goal))
            right_distance = min(right_distance,manhattan_distance_with_heading(right_state,goal))
        #if forward is nearest then go forward
        if forward_distance <= min(left_distance,right_distance):
            return ['Forward']
        #otherwise turn
        elif left_distance <= min(forward_distance,right_distance):
            return ['TurnLeft']
        elif right_distance <= min(forward_distance,left_distance):
            return ['TurnRight']
        else:
            return [random.choice(doingactions)]
        
        

    def result(self, state, action):
        """
        Return the new state after applying action to state
        0 = North
        1 = West
        2 = South
        3 = East
        """
        #Brute Force to show the result
        new_state = ()
        if action == 'Forward':
            if state[2] == 0:
                new_state = (state[0],state[1] + 1,state[2])
            elif state[2] == 1:
                new_state = (state[0]-1,state[1],state[2])
            elif state[2] == 2:
                new_state = (state[0],state[1]-1,state[2])
            elif state[2] == 3:
                new_state = (state[0]+1,state[1],state[2])
        elif action == 'TurnRight':
            if state[2] == 0:
                new_state = (state[0],state[1],3)
            elif state[2] == 1:
                new_state = (state[0],state[1],0)
            elif state[2] == 2:
                new_state =(state[0],state[1],1)
            elif state[2] == 3:
                new_state = (state[0],state[1],2)
        elif action == 'TurnLeft':
            if state[2] == 0:
                new_state = (state[0],state[1],1)
            elif state[2] == 1:
                new_state = (state[0],state[1],2)
            elif state[2] == 2:
                new_state = (state[0],state[1],3)
            elif state[2] == 3:
                new_state = (state[0],state[1],0)
        #print '{0} to {1}'.format(state,new_state)
        return new_state

        

    def goal_test(self, state):
        """
        Return True if state is a goal state
        """
        return state[0:2] in self.goals

#-------------------------------------------------------------------------------

def test_PRP(initial):
    """
    The 'expected initial states and solution pairs' below are provided
    as a sanity check, showing what the PlanRouteProblem soluton is
    expected to produce.  Provide the 'initial state' tuple as the
    argument to test_PRP, and the associate solution list of actions is
    expected as the result.
    The test assumes the goals are [(2,3),(3,2)], that the heuristic fn
    defined in PlanRouteProblem uses the manhattan_distance_with_heading()
    fn above, and the allowed locations are:
        [(0,0),(0,1),(0,2),(0,3),
        (1,0),(1,1),(1,2),(1,3),
        (2,0),            (2,3),
        (3,0),(3,1),(3,2),(3,3)]
    
    Expected intial state and solution pairs:
    (0,0,0) : ['Forward', 'Forward', 'Forward', 'TurnRight', 'Forward', 'Forward']
    (0,0,1) : ['TurnRight', 'Forward', 'Forward', 'Forward', 'TurnRight', 'Forward', 'Forward']
    (0,0,2) : ['TurnLeft', 'Forward', 'Forward', 'Forward', 'TurnLeft', 'Forward', 'Forward']
    (0,0,3) : ['Forward', 'Forward', 'Forward', 'TurnLeft', 'Forward', 'Forward']
    """
    return plan_route((initial[0],initial[1]), initial[2],
                      # Goals:
                      [(2,3),(3,2)],
                      # Allowed locations:
                      [(0,0),(0,1),(0,2),(0,3),
                       (1,0),(1,1),(1,2),(1,3),
                       (2,0),            (2,3),
                       (3,0),(3,1),(3,2),(3,3)])


#-------------------------------------------------------------------------------
# Plan Shot
#-------------------------------------------------------------------------------

def plan_shot(current, heading, goals, allowed):
    """ Plan route to nearest location with heading directed toward one of the
    possible wumpus locations (in goals), then append shoot action.
    NOTE: This assumes you can shoot through walls!!  That's ok for now. """
    if goals and allowed:
        psp = PlanShotProblem((current[0], current[1], heading), goals, allowed)
        node = search.astar_search(psp)
        if node:
            plan = node.solution()
            plan.append(action_shoot_str(None))
            # HACK:
            # since the wumpus_alive axiom asserts that a wumpus is no longer alive
            # when on the previous round we perceived a scream, we
            # need to enforce waiting so that itme elapses and knowledge of
            # "dead wumpus" can then be inferred...
            plan.append(action_wait_str(None))
            return plan

    # no route can be found, return empty list
    return []

#-------------------------------------------------------------------------------

class PlanShotProblem(search.Problem):
    def __init__(self, initial, goals, allowed):
        """ Problem defining planning to move to location to be ready to
              shoot at nearest wumpus location
        NOTE: Just like PlanRouteProblem, except goal is to plan path to
              nearest location with heading in direction of a possible
              wumpus location;
              Shoot and Wait actions is appended to this search solution
        Goal is generally a location (x,y) tuple, but state will be (x,y,heading) tuple
        initial = initial location, (x,y) tuple
        goals   = list of goal (x,y) tuples
        allowed = list of state (x,y) tuples that agent could move to """
        self.initial = initial # initial state
        self.goals = goals     # list of goals that can be achieved
        self.allowed = allowed # the states we can move into

    def h(self,node):
        """
        Heuristic that will be used by search.astar_search()
        """
        #compute the shooting location we want to go
        shootLocations = []
        for location in self.allowed:
            for goal in self.goals:
                if location[0] == goal[0] and location[1] > goal[1]:
                    shootLocations.append((location[0],location[1],2))
                elif location[0] == goal[0] and location[1] < goal[1]:
                    shootLocations.append((location[0],location[1],0))
                elif location[0] > goal[0] and location[1] == goal[1]:
                    shootLocations.append((location[0],location[1],1))
                elif location[0] < goal[0] and location[1] == goal[1]:
                    shootLocations.append((location[0],location[1],3))
        "*** YOUR CODE HERE ***"
        distanceToGoals = [manhattan_distance_with_heading(node.state,goal) for goal in shootLocations]
        return min(distanceToGoals)

    def actions(self, state):
        """
        Return list of allowed actions that can be made in state
        0 = North
        1 = West
        2 = South
        3 = East
        """
        "*** YOUR CODE HERE ***"
        """
        mindistance = float('inf')
        shootinglocation =()
        for goal in shootLocations:
            dis = manhattan_distance_with_heading(state,goal)
            if dis<mindistance:
                mindistance = dis
                shootinglocation = goal
        #compute if I can go straight to shooting location
        """
        #Check if we are at shoot direction
        shootLocations = []
        for location in self.allowed:
            for goal in self.goals:
                if location[0] == goal[0] and location[1] > goal[1]:
                    shootLocations.append((location[0],location[1],2))
                elif location[0] == goal[0] and location[1] < goal[1]:
                    shootLocations.append((location[0],location[1],0))
                elif location[0] > goal[0] and location[1] == goal[1]:
                    shootLocations.append((location[0],location[1],1))
                elif location[0] < goal[0] and location[1] == goal[1]:
                    shootLocations.append((location[0],location[1],3))
        for shootLocation in shootLocations:
            if state[0]==shootLocation[0] and state[1]==shootLocation[1]:
                if state[2] == shootLocation[2]:
                    return ['Shoot','Wait']
                elif state[2] == (shootLocation[2]+3)%4:
                    return ['TurnLeft','Shoot','Wait']
                elif state[2] == (shootLocation[2]+1)%4:
                    return ['TurnRight','Shoot','Wait']
                else:
                    return ['TurnRight','TurnRight','Shoot','Wait']
        #if not then we compute as usual to find the wumpus location
        forward_state=()
        left_state=()
        right_state=()
        if state[2] == 0:
            forward_state = (state[0],state[1]+1,state[2])
            left_state=(state[0],state[1],1)
            right_state=(state[0],state[1],3)
        elif state[2] == 1:
            forward_state = (state[0]-1,state[1],state[2])
            left_state=(state[0],state[1],2)
            right_state=(state[0],state[1],0)
        elif state[2] == 2:
            forward_state = (state[0],state[1]-1,state[2])
            left_state=(state[0],state[1],3)
            right_state=(state[0],state[1],1)
        elif state[2] == 3:
            forward_state = (state[0]+1,state[1],state[2])
            left_state=(state[0],state[1],0)
            right_state=(state[0],state[1],2)
        forward_distance = float('inf')
        left_distance = float('inf')
        right_distance = float('inf')
        for goal in shootLocations:
            if (forward_state[0],forward_state[1]) in self.allowed:
                forward_distance = min(forward_distance,manhattan_distance_with_heading(forward_state,goal))
            left_distance = min(left_distance,manhattan_distance_with_heading(left_state,goal))
            right_distance = min(right_distance,manhattan_distance_with_heading(right_state,goal))
        #if forward is correct then go forward
        if forward_distance < min(left_distance,right_distance):
            return ['Forward']
        #otherwise, Turn
        elif left_distance <= min(forward_distance,right_distance):
            return ['TurnLeft']
        else:
            return ['TurnRight']
        

    def result(self, state, action):
        """
        Return the new state after applying action to state
        """
        "*** YOUR CODE HERE ***"
        #Brute force to show the new state 
        new_state = ()
        if action == 'Forward':
            if state[2] == 0:
                new_state = (state[0],state[1] + 1,state[2])
            elif state[2] == 1:
                new_state = (state[0]-1,state[1],state[2])
            elif state[2] == 2:
                new_state = (state[0],state[1]-1,state[2])
            elif state[2] == 3:
                new_state = (state[0]+1,state[1],state[2])
        elif action == 'TurnRight':
            if state[2] == 0:
                new_state = (state[0],state[1],3)
            elif state[2] == 1:
                new_state = (state[0],state[1],0)
            elif state[2] == 2:
                new_state =(state[0],state[1],1)
            elif state[2] == 3:
                new_state = (state[0],state[1],2)
        elif action == 'TurnLeft':
            if state[2] == 0:
                new_state = (state[0],state[1],1)
            elif state[2] == 1:
                new_state = (state[0],state[1],2)
            elif state[2] == 2:
                new_state = (state[0],state[1],3)
            elif state[2] == 3:
                new_state = (state[0],state[1],0)
        elif action == 'Wait':
            new_state = state
        elif action == 'Shoot':
            new_state = state
        return new_state
        #pass

    def goal_test(self, state):
        """
        Return True if state is a goal state
        """
        "*** YOUR CODE HERE ***"
        #compute the shoot location and only when we're at shooting location is true
        shootLocations = []
        for location in self.allowed:
            for goal in self.goals:
                if location[0] == goal[0] and location[1] > goal[1]:
                    shootLocations.append((location[0],location[1],2))
                elif location[0] == goal[0] and location[1] < goal[1]:
                    shootLocations.append((location[0],location[1],0))
                elif location[0] > goal[0] and location[1] == goal[1]:
                    shootLocations.append((location[0],location[1],1))
                elif location[0] < goal[0] and location[1] == goal[1]:
                    shootLocations.append((location[0],location[1],3))
        return state in shootLocations
        #return True

#-------------------------------------------------------------------------------

def test_PSP(initial = (0,0,3)):
    """
    The 'expected initial states and solution pairs' below are provided
    as a sanity check, showing what the PlanShotProblem soluton is
    expected to produce.  Provide the 'initial state' tuple as the
    argumetn to test_PRP, and the associate solution list of actions is
    expected as the result.
    The test assumes the goals are [(2,3),(3,2)], that the heuristic fn
    defined in PlanShotProblem uses the manhattan_distance_with_heading()
    fn above, and the allowed locations are:
        [(0,0),(0,1),(0,2),(0,3),
        (1,0),(1,1),(1,2),(1,3),
        (2,0),            (2,3),
        (3,0),(3,1),(3,2),(3,3)]
    
    Expected intial state and solution pairs:
    (0,0,0) : ['Forward', 'Forward', 'TurnRight', 'Shoot', 'Wait']
    (0,0,1) : ['TurnRight', 'Forward', 'Forward', 'TurnRight', 'Shoot', 'Wait']
    (0,0,2) : ['TurnLeft', 'Forward', 'Forward', 'Forward', 'TurnLeft', 'Shoot', 'Wait']
    (0,0,3) : ['Forward', 'Forward', 'Forward', 'TurnLeft', 'Shoot', 'Wait']
    """
    return plan_shot((initial[0],initial[1]), initial[2],
                     # Goals:
                     [(2,3),(3,2)],
                     # Allowed locations:
                     [(0,0),(0,1),(0,2),(0,3),
                      (1,0),(1,1),(1,2),(1,3),
                      (2,0),            (2,3),
                      (3,0),(3,1),(3,2),(3,3)])
    
#-------------------------------------------------------------------------------

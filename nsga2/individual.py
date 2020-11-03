import numpy as np

class Individual(object):
    
    """
    A class used to describe an individual portfolio
    ...

    Attributes
    ----------
    rank : int
        rank of the solution
        
    crowding_distance : float
        crowding distance in the front
        
    domination_count : int
        number of dominating portfolios
        
    dominated_solutions : int
        number of dominated portfolios

    features : array
        portfolio weights 
    
    objectives : list
        [risk,return]
        
    Methods
    -------
    __eq__(other)
        verifies equality with another solution
        
    dominates(other_individual)
        domination operator
        
    """
    

    def __init__(self):
               
        """
        Parameters
        ----------

        """
        self.rank = None
        self.crowding_distance = None
        self.domination_count = None
        self.dominated_solutions = None
        self.features = None
        self.objectives = None

    def __eq__(self, other):
        
               
        """
        Parameters
        ----------
        other : Individual
            other portfolio
        
        Returns
        -------
        boolean
            indicates whether portfolios are equal or not
       
        """
        
        if isinstance(self, other.__class__):
            return np.array_equal(self.features,other.features)
        return False

    """def dominates(self, other_individual):
        and_condition = True
        or_condition = False
        for first, second in zip(self.objectives, other_individual.objectives):
            and_condition = and_condition and first <= second
            or_condition = or_condition or first < second
        return (and_condition and or_condition)"""
    
    def dominates(self, other_individual):
        
        """
        Parameters
        ----------
        other_individual : Individual
            other portfolio
        
        Returns
        -------
        boolean
            indicates whether the current portfolio domintes the other portflio or not
       
        """
        
        X=self.objectives
        Y=other_individual.objectives
    
        if (X[0]<Y[0] and X[1]>Y[1]):
            return True
        else :
            return False
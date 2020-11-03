import numpy as np
from nsga2.individual import Individual
from nsga2.utils import NSGA2Utils

class Problem:
    
    """
    A class used to define the optimization problem
    ...

    Attributes
    ----------
    num_of_objectives : int
        number of objective functions
        
    portfolio_size : int
        number of risky assets
        
    objectives : list
        objective functions
        
    Q : float
        quantity limit for risky assets

    C : int
        cardinality limit 
    
    Q_rfa : float
        quantity limit for RFA
        
    Methods
    -------
    generate_individual()
        generates a new portfolio
        
    calculate_objectives(individual)
        calculates risk and return of a portfolio
        
    init_population(portfolio_size, N, Q,Q_rfa, C)
        initializes a new population
        
    """

    def __init__(self,objectives,portfolio_size,Q,Q_rfa,C,initial_portfolio=None):
        
                
        """
        Parameters
        ----------
        objectives : list
            list of objective functions [risk,return]

        portfolio_size : int
            number of risky assets
            
        Q : float
            quantity limit for risky assets

        C : int
            cardinality limit 

        Q_rfa : float
            quantity limit for RFA

        """
        
        if C*Q+Q_rfa<1:
            raise "Values do not add up"
        if C>portfolio_size:
            raise "check cardinality"
        self.num_of_objectives = len(objectives)
        self.portfolio_size = portfolio_size
        self.objectives = objectives
        self.Q=Q
        self.C=C
        self.Q_rfa=Q_rfa
        self.initial_portfolio=initial_portfolio
     
    
    
    
    def generate_individual(self):
                 
        """
        Parameters
        ----------
        
        Returns
        -------
        individual
            a new random portfolio
       
        """
        
        individual = Individual()
        individual.features = self.init_population(self.portfolio_size, 1, self.Q, self.Q_rfa, self.C)[0]
        return individual

    def calculate_objectives(self, individual):
        
                        
        """
        Parameters
        ----------
        individual : individual
            a portfolio
        
        Returns
        -------
        list
            list of objectives [risk,return]
       
        """
        
        individual.objectives = [f(individual.features) for f in self.objectives]
        
        
    def init_population(self,portfolio_size, N, Q,Q_rfa, C):
        
                        
        """
        Parameters
        ----------
        portfolio_size : int
            number of risky assets
            
        N : int
            population size
            
        Q : float
            quantity limit for risky assets

        C : int
            cardinality limit 

        Q_rfa : float
            quantity limit for RFA
    
        Returns
        -------
        list
            list of individuals
        """
        
        solutions=[]
        for i in range(N):
            if(np.all(self.initial_portfolio==None)):
                solution=[0+(Q-0)*np.random.random() for j in range(portfolio_size)]
                solution.append(0+(Q_rfa-0)*np.random.random())
            
            else:
                solution=self.initial_portfolio
            
            solutions.append(NSGA2Utils.check_fix_solution(self,np.array(solution),portfolio_size,C,Q,Q_rfa))


        return solutions 
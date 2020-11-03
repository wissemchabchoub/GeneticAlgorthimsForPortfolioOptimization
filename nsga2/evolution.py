from nsga2.population import Population
from nsga2.utils import NSGA2Utils
import numpy as np
import copy


class Evolution:
    
    
    """
    A class used to evolve genrations and find the Pareto set
    ...

    Attributes
    ----------
    utils : NSGA2Utils
        class of utile functions
        
    population : Population
        population class
        
    num_of_generations : int
        number of generations
        
    num_of_individuals : int
        population size

    problem : Problem
        problem definition class
        
    Methods
    -------
    evolve()
        main evolving method
        
    """
    
    def __init__(self, problem, num_of_generations=1000, num_of_individuals=100, num_of_tour_particips=2, tournament_prob=0.9, p_c=0.6, p_m=0.3,p_risked=0.5,full=True):
        
        
        """

        Attributes
        ----------
        utils : NSGA2Utils
            class of utile functions

        population : Population
            population class

        num_of_generations : int
            number of generations

        num_of_individuals : int
            population size

        problem : Problem
            problem definition class
            
        num_of_tour_particips : int
            number of indivisuals per selection tournament
            
        tournament_prob : float
            paramters in binary selection tournament (not used for now)
                
        p_c : float
            crossover probability 

        p_m : float
            mutaion probability

        p_risked : float
            probability of doing the crossover on the risky assets

        full : boolean
            True: cross over on the whole portfolio , False : cross over on a part of the portfolio

        """        

        
        self.utils = NSGA2Utils(problem, num_of_individuals, num_of_tour_particips, tournament_prob, p_c, p_m,p_risked,full)
        self.population = None
        self.num_of_generations = num_of_generations
        self.on_generation_finished = []
        self.num_of_individuals = num_of_individuals
        self.problem=problem

    def evolve(self):
        
        """main evolving method

            Parameters
            ----------

            Returns
            -------
            array
                array of optimal portfolios
        """
        
        
        self.population = self.utils.create_initial_population()
        self.utils.fast_nondominated_sort(self.population)
        for front in self.population.fronts:
            self.utils.calculate_crowding_distance(front)
        children = self.utils.create_children(self.population)
        returned_population = None
        for i in range(self.num_of_generations):
            self.population.extend(children)
            
            for pop in self.population.population:
                self.problem.calculate_objectives(pop)
            
            #Eliminate duplicates of  after breeding
            self.population.population=self.utils.eliminate_duplicates(self.population.population)
            
            self.utils.fast_nondominated_sort(self.population)
            new_population = Population()

            front_num = 0
            while len(new_population) + len(self.population.fronts[front_num]) <= self.num_of_individuals:
                self.utils.calculate_crowding_distance(self.population.fronts[front_num])
                new_population.extend(self.population.fronts[front_num])
                front_num += 1
            self.utils.calculate_crowding_distance(self.population.fronts[front_num])
            self.population.fronts[front_num].sort(key=lambda individual: individual.crowding_distance, reverse=True)
            new_population.extend(self.population.fronts[front_num][0:self.num_of_individuals-len(new_population)])
            
            #returned_population = self.population
            R=copy.deepcopy(self.population.fronts[0])
            
            self.population = new_population
            self.utils.fast_nondominated_sort(self.population)
            for front in self.population.fronts:
                self.utils.calculate_crowding_distance(front)
            children = self.utils.create_children(self.population)
        return R
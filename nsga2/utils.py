from nsga2.population import Population
import numpy as np
import random
import copy


class NSGA2Utils:
    
    """
    A class used for various utile methods
    ...

    Attributes
    ----------

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
        
    Q : float
        quantity limit for risky assets

    C : int
        cardinality limit 
    
    Q_rfa : float
        quantity limit for RFA
        

    Methods
    -------
    create_initial_population()
        creates initial population
    
    fast_nondominated_sort(population)
        performs fast nd sorting on a population
        
    calculate_crowding_distance(front)
        calculates crowding distance front wise
        
    crowding_operator(individual, other_individual)
        crowding distance operator 
        
    create_children(population)
        breeding function
        
    eliminate_duplicates(population)
        eliminates duplicates from a population
        
    __crossover(parent1, parent2)
        performs crossover
        
    cross_over(parent1,parent2,portfolio_size,C,Q,Q_rfa,p_c=0.4,assets='Risked',full=True)
        crossover between two parents with probability p_c
        
    __mutate(child)
        performs mutation
        
    mutate(solution,portfolio_size,p_m=0.2)
        performs mutation with a pobability p_m

    __tournament(population)
        performes tournanment selection
        
        
    __wheel(population,num_selected=1)
        performes wheel selection
        
    __choose_with_prob(prob)
        chooses with a probability
        
    check_fix_solution(solution,portfolio_size,C,Q,Q_rfa)
        adjust portfolio weights according to constraints
        
    return_portfolio(self,solution,returns):
        calculates portfolio's (solution) return
        
    risk_portfolio(self,solution,cov):
        calculates portfolio's (solution) risk
        
    """

    def __init__(self, problem, num_of_individuals=100,
                 num_of_tour_particips=2, tournament_prob=0.9, p_c=0.2, p_m=0.5,p_risked=0.5,full=True):
        
        """
            Parameters
            ----------
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

        self.problem = problem
        self.num_of_individuals = num_of_individuals
        self.num_of_tour_particips = num_of_tour_particips
        self.tournament_prob = tournament_prob
        self.p_c = p_c
        self.p_m = p_m
        self.Q=problem.Q
        self.Q_rfa=problem.Q_rfa

        self.C=problem.C
        self.portfolio_size=problem.portfolio_size
        self.full=full
        self.p_risked=p_risked
        

    def create_initial_population(self):
        
        """creates initial population

            Parameters
            ----------

            Returns
            -------
            Population
                initial population
        """
        
        population = Population()
        for _ in range(self.num_of_individuals):
            individual = self.problem.generate_individual()
            self.problem.calculate_objectives(individual)
            population.append(individual)
        return population
        

    def fast_nondominated_sort(self, population):
        
        """performs fast nd sorting on a population

            Parameters
            ----------
            population : Population
                population to sort
                
            Returns
            -------

        """
        
        population.fronts = [[]]
        for individual in population:
            individual.domination_count = 0
            individual.dominated_solutions = []
            for other_individual in population:
                if individual.dominates(other_individual):
                    individual.dominated_solutions.append(other_individual)
                elif other_individual.dominates(individual):
                    individual.domination_count += 1
            if individual.domination_count == 0:
                individual.rank = 0
                population.fronts[0].append(individual)
        i = 0
        while len(population.fronts[i]) > 0:
            temp = []
            for individual in population.fronts[i]:
                for other_individual in individual.dominated_solutions:
                    other_individual.domination_count -= 1
                    if other_individual.domination_count == 0:
                        other_individual.rank = i+1
                        temp.append(other_individual)
            i = i+1
            population.fronts.append(temp)

    def calculate_crowding_distance(self, front):
        
        
        """calculates crowding distance front wise

            Parameters
            ----------
            front : array of Individual
                portflio front
                
            Returns
            -------

        """    
        
        if len(front) > 0:
            solutions_num = len(front)
            for individual in front:
                individual.crowding_distance = 0

            for m in range(len(front[0].objectives)):
                front.sort(key=lambda individual: individual.objectives[m])
                front[0].crowding_distance = 10**9
                front[solutions_num-1].crowding_distance = 10**9
                m_values = [individual.objectives[m] for individual in front]
                scale = max(m_values) - min(m_values)
                if scale == 0: scale = 1
                for i in range(1, solutions_num-1):
                    front[i].crowding_distance += (front[i+1].objectives[m] - front[i-1].objectives[m])/scale

    def crowding_operator(self, individual, other_individual):
        
        """crowding distance operator 
        
            Parameters
            ----------
            individual : Individual
                1st portflio
                
            other_individual : Individual
                2nd portflio
                
            Returns
            -------
            
            int:
                1 if individual > other_individual
                elif -1 

        """    
        
        if (individual.rank < other_individual.rank) or \
            ((individual.rank == other_individual.rank) and (individual.crowding_distance > other_individual.crowding_distance)):
            return 1
        else:
            return -1

    def create_children(self, population):
        
        
        """breeding function

            Parameters
            ----------
            population : Population
                parent population
                
            Returns
            -------
            list
                children

        """    
        
        children = []
        while len(children) < len(population):
            
            mutation_candidate = copy.copy(self.__wheel(population))
            self.__mutate(mutation_candidate)
            self.problem.calculate_objectives(mutation_candidate)
            children.append(mutation_candidate)
            
            parent1 = self.__wheel(population)
            parent2 = parent1
            while parent1 == parent2:
                parent2 = self.__wheel(population)
            child1, child2 = self.__crossover(parent1, parent2)
            #self.__mutate(child1)
            #self.__mutate(child2)
            self.problem.calculate_objectives(child1)
            self.problem.calculate_objectives(child2)
            children.append(child1)
            children.append(child2)
            
        return children
    
    
    def eliminate_duplicates(self, population):
        
        """eliminates duplicates from a population

            Parameters
            ----------
            population : array
                
                
            Returns
            -------
            list
                population without duplicates
                

        """    
        
        result = []
        seen = []
        for sol in population:
            if sol.objectives not in seen:
                result.append(sol)
                seen.append(sol.objectives)
        
            
        return result
    

    """def __crossover(self, individual1, individual2):
        child1 = self.problem.generate_individual()
        child2 = self.problem.generate_individual()
        num_of_features = len(child1.features)
        genes_indexes = range(num_of_features)
        for i in genes_indexes:
            beta = self.__get_beta()
            x1 = (individual1.features[i] + individual2.features[i])/2
            x2 = abs((individual1.features[i] - individual2.features[i])/2)
            child1.features[i] = x1 + beta*x2
            child2.features[i] = x1 - beta*x2
        return child1, child2"""
    
    def __crossover(self,parent1, parent2):
        
        """performs crossover

            Parameters
            ----------
            parent1 : Individual
                1st parent

            parent2 : Individual
                2nd parent
                
            Returns
            -------
            child1 : Individual
                1st child

            child2 : Individual
                2nd child

        """    
        
        child1 = self.problem.generate_individual()
        child2 = self.problem.generate_individual()
        proba=np.random.random()
        if(proba<self.p_risked):
            child1.features,child2.features=self.cross_over(parent1.features,parent2.features,
                   self.portfolio_size,self.C,self.Q,self.Q_rfa,self.p_c,assets='Risked',full=self.full)
        else:
            child1.features,child2.features=self.cross_over(parent1.features,parent2.features,
                   self.portfolio_size,self.C,self.Q,self.Q_rfa,self.p_c,assets='Risk Free',full=False)
        
        return child1,child2
        
        
    #crossover on features    
    def cross_over(self,parent1,parent2,portfolio_size,C,Q,Q_rfa,p_c=0.4,assets='Risked',full=True):
        
        """crossover between two parents with probability p_c

            Parameters
            ----------
            parent1 : array
                1st parent weights

            parent2 : array
                2nd parent weights
                
            p_c : float
                crossover probability 

            assets : str
                'Risked' : crossover on the risky assets
                'Risk Free' : crossover on the RFA asset

            full : boolean
                True: cross over on the whole portfolio , False : cross over on a part of the portfolio

            Q : float
                quantity limit for risky assets

            C : int
                cardinality limit 

            Q_rfa : float
                quantity limit for RFA    
                
            Returns
            -------
            child1 : array
                1st child weights

            child2 : array
                2nd child weights

        """    
        
        if assets=='Risk Free' and full==True:
            raise 'Error'

        r=np.random.random()
        if(r>p_c):
            return parent1,parent2

        if assets=='Risked':
            N0=0
            N=portfolio_size-1
        elif assets=="Risk Free":
            N0=portfolio_size
            N=portfolio_size+1

        beta=np.random.random()
        if full==False:
            i=np.random.randint(N0,N)
            j=np.random.randint(i+1,N+1)
        elif full==True:
            i=0
            j=portfolio_size
        child1=np.copy(parent1)
        child1[i:j]=beta*parent1[i:j]+(1-beta)*parent2[i:j]

        beta=np.random.random()
        if full==False:
            i=np.random.randint(N0,N)
            j=np.random.randint(i+1,N+1)
        elif full==True:
            i=0
            j=portfolio_size

        child2=np.copy(parent2)
        child2[i:j]=beta*parent2[i:j]+(1-beta)*parent1[i:j]

        child1=self.check_fix_solution(child1,portfolio_size,C,Q,Q_rfa)
        child2=self.check_fix_solution(child2,portfolio_size,C,Q,Q_rfa)

        return child1,child2
        
    """def __get_beta(self):
        u = random.random()
        if u <= 0.5:
            return (2*u)**(1/(self.crossover_param+1))
        return (2*(1-u))**(-1/(self.crossover_param+1))"""

    """def __mutate(self, child):
        num_of_features = len(child.features)
        for gene in range(num_of_features):
            u, delta = self.__get_delta()
            if u < 0.5:
                child.features[gene] += delta*(child.features[gene] - self.problem.variables_range[gene][0])
            else:
                child.features[gene] += delta*(self.problem.variables_range[gene][1] - child.features[gene])
            if child.features[gene] < self.problem.variables_range[gene][0]:
                child.features[gene] = self.problem.variables_range[gene][0]
            elif child.features[gene] > self.problem.variables_range[gene][1]:
                child.features[gene] = self.problem.variables_range[gene][1]"""
    
    def __mutate(self,child):
        
        """performs mutation

            Parameters
            ----------
            child : Individual
                portflio to be mutated
                
            Returns
            -------

        """    
        
        child.features=self.mutate(child.features,self.portfolio_size,self.p_m)
    
    #mutaion on features
    def mutate(self,solution,portfolio_size,p_m=0.2):
        
        """performs mutation with a pobability p_m

            Parameters
            ----------
            solution : array
                portfolio weights
                
            portfolio_size : int
                portfolio size
            
            p_m : float
                mutation probability
                
            Returns
            -------
            array
                mutant portfolio weights

        """    
        
        r=np.random.random()
        if(r>p_m):
            return solution

        #only risked assets
        i=np.random.randint(0,portfolio_size-1)
        j=np.random.randint(i+1,portfolio_size)

        proba=np.random.random()

        #swap
        if proba>0.5:
            solution[i],solution[j]=solution[j],solution[i]

        #shift
        else:
            solution[i:j+1]=np.roll(solution[i:j+1],1)

        return(solution)
    
    """def __get_delta(self):
        u = random.random()
        if u < 0.5:
            return u, (2*u)**(1/(self.mutation_param + 1)) - 1
        return u, 1 - (2*(1-u))**(1/(self.mutation_param + 1))"""

    def __tournament(self, population):

        """performes tournanment selection

            Parameters
            ----------
            population : Population
                population to select from
                
            Returns
            -------
            Individual :
                selected individual

        """    
        
        participants = random.sample(population.population, self.num_of_tour_particips)
        best = None
        for participant in participants:
            if best is None or (self.crowding_operator(participant, best) == 1 and self.__choose_with_prob(self.tournament_prob)):
                best = participant

        return best
    
    def __wheel(self, population,num_selected=1):
        
        """performes wheel selection

            Parameters
            ----------
            population : Population
                population to select from
                
            num_selected : int
                number of individuaks to be selected
                
            Returns
            -------
            Individual or array of Individual:
                selected individual

        """              
        scores=[]
        participants = random.sample(population.population, self.num_of_tour_particips)
        ranks=np.array([ind.rank for ind in participants])
        ranks=np.max(ranks)-ranks
        total = sum(ranks)
        if total!=0:
            selection_probs = [r/total for r in ranks]
            best=participants[np.random.choice(len(participants), p=selection_probs)]
        else:
            best=participants[np.random.choice(len(participants))]
        return best

    def __choose_with_prob(self, prob):
        
        """chooses with a probability

            Parameters
            ----------
            prob : float
                probability
                
            Returns
            -------
            boolean:
                True or Flase according to prob

        """             
        
        if random.random() <= prob:
            return True
        return False
        
        
    def check_fix_solution(self,solution,portfolio_size,C,Q,Q_rfa):
        
        """adjust portfolio weights according to constraints

            Parameters
            ----------
            solution : array
                portfolio weights

            portfolio_size : int
                portfolio size
                
            Q : float
                quantity limit for risky assets

            C : int
                cardinality limit 

            Q_rfa : float
                quantity limit for RFA    
                
            Returns
            -------
            array 
                new portfolio weights
                
        """    
    
        #Weight_constraints    
        weight_constraints=np.ones(portfolio_size+1)*Q
        weight_constraints[-1]=Q_rfa

        #Cardinality limit
        if (np.sum(solution[:-1]!=0)>C):
            selected_indices=np.argwhere(solution[:-1]>0).flatten()
            choice=np.random.choice(selected_indices,size=np.sum(solution[:-1]!=0)-C,replace=False)
            solution[:-1][choice]=0

            weight_constraints[choice]=0

        #fixing the weight_constraints
        selected_indices=np.argwhere(np.logical_and(solution[:-1]==0,weight_constraints[:-1]!=0)).flatten()
        choice=np.random.choice(selected_indices,size=np.sum(weight_constraints[:-1]!=0)-C,replace=False)
        weight_constraints[:-1][choice]=0



        #Quantity
        surplus=solution-weight_constraints
        solution[surplus>0]=weight_constraints[surplus>0]

        #Sum==1
        surplus=solution-weight_constraints
        #surplus<0: there is still space
        #surplus>0 : we exceeded the limit

        delta=np.sum(solution)-1
        while(abs(delta)>0.001):

            weights=solution/np.sum(solution)
            weighted_delta=weights*delta

            if (delta>0): #more than one
                solution=np.maximum(solution-weighted_delta,0)

            if (delta<0): #less than one
                solution[surplus<0]=np.minimum(solution[surplus<0]-weighted_delta[surplus<0],weight_constraints[surplus<0])

            delta=np.sum(solution)-1
            surplus=solution-weight_constraints

        return solution
        
        
    #objective functions
    def return_portfolio(self,solution,returns):
        
        """calculates portfolio's (solution) return

            Parameters
            ----------
            solution : array
                portfolio
            returns : array
                an array of expected returns

            Returns
            -------
            float
                portfolio's return
        """
        
        return np.sum(solution*returns)

    def risk_portfolio(self,solution,cov):

        """calculates portfolio's (solution) risk

            Parameters
            ----------
            solution : array
                portfolio
            cov : array
                variance covariance matrix

            Returns
            -------
            float
                portfolio's risk
        """ 
        w=solution
        return np.matmul(w,np.matmul(cov,w.T))
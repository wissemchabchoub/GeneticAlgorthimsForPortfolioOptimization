class Population:
    
    """
    A class used to describe an individual portfolio
    ...

    Attributes
    ----------
    population : list
        list of portfolios
        
    fronts : list
        list of fronts
        
    Methods
    -------
    __len__()
        size of population
        
    __iter__(other_individual)
        iterator over a population
   
    extend(new_individuals)
        adds new individuals to population
   
    append(new_individual)
        adds new individual to population
        
    """

    def __init__(self):
        
        """
        Parameters
        ----------
       
        """
        
        self.population = []
        self.fronts = []

    def __len__(self):
        
        """
        Parameters
        ----------
        
        Returns
        -------
        int
            size of population
       
        """
        
        return len(self.population)

    def __iter__(self):
        
        """
        Parameters
        ----------
        
        Returns
        -------
        iterator
            iterator over a population
       
        """
        
        return self.population.__iter__()

    def extend(self, new_individuals):
        
        """
        Parameters
        ----------
        new_individuals : list
            new individuals to be added to the population
        
        Returns
        -------
        """
        
        self.population.extend(new_individuals)

    def append(self, new_individual):
        
        """
        Parameters
        ----------
        new_individual : individual
            new individual to be added to the population
        
        Returns
        -------
        """
        
        self.population.append(new_individual)
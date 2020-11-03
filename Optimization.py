import numpy as np
import pandas as pd
from nsga2.problem import Problem
from nsga2.evolution import Evolution
import pickle
import os
import matplotlib.pyplot as plt

class Optimization:
    
    
    """
    A class used for portfolio optimization

    ...

    Attributes
    ----------
    data_path : str
        path for the market_data
        
    portfolios_path : str
        path for portfolios directory

    rf : float
        risk free rate
    
    data : dataframe
        prices dataframe
    
    N : int
        Total number of assets

    assets : array
        array of assets
    
    Methods
    -------
    closest_point(node,nodes)
        returns the index closest point in nodes to node
        
    get_all_real_returns()
        return the dataframe of 10-days historical returns
        
    return_portfolio(solution,R)
        calculates portfolio's (solution) return
    
    risk_portfolio(solution,cov)
        calculates portfolio's (solution) risk
        
    select_portfolio(front,function1,function2,policy)
        selects a portfolio from the front following a risk or return policy
        
    optimize(predictions,run_day,C=0,Q=0.6,Q_rfa=1,num_of_generations=250, num_of_individuals=200, num_of_tour_particips=10,tournament_prob=1, p_c=0.5,p_m=0.5,p_risked=0.8,full=True,policy='return')
        optimizes the portfolio following constraints, predictions and historical risk
    
        
       
    """

    import numpy as np

    def __init__(self,data_path,portfolios_path,rf):
        
        """
        Parameters
        ----------
        data_path : str
            path for the market_data
            
        portfolios_path : str
            path of portfilios directory

        rf : float
            risk free rate in percentage
        """
        
        self.data_path=data_path
        self.portfolios_path=portfolios_path
        self.rf=rf

        files=os.listdir(data_path)
        tables=[]
        for f in files:
            table=pd.read_csv(data_path+'/'+f)
            table.index=pd.to_datetime(table.Date)
            tables.append(table)

        data=pd.DataFrame(index=tables[0].index)
        for f,table in zip(files,tables):
            data[f]=table.Open

        self.data=data
        self.assets=data.columns
        self.N=len(self.assets)
        
    def closest_point(self,node,nodes):
        
        """returns the index closest point in nodes to node

        Parameters
        ----------
        node : float
            the node to approach
        nodes : array
            nodes to select from

        Returns
        -------
        int
            the index of the selected node
        """
        
        nodes = np.asarray(nodes)
        dist_2 =(nodes - node)**2
        return np.argmin(dist_2)

        
    def get_all_real_returns(self):
        
        
        """returns the dataframe of 10-days historical returns

        Parameters
        ----------

        Returns
        -------
        dataframe
            the dataframe of 10-days historical returns
        """

        all_real_returns=pd.DataFrame(index=self.data.iloc[::10].index,columns=self.assets.copy())

        for i in range(self.N):
            all_real_returns.loc[:,self.assets[i]]=self.data.loc[:,self.assets[i]].iloc[::10].pct_change().values

        all_real_returns=all_real_returns.iloc[1:]
        all_real_returns['rf']=self.rf*0.01
        all_real_returns=all_real_returns.fillna(0)
        
        return all_real_returns
        
    def return_portfolio(self,solution,R):
        """calculates portfolio's (solution) return

            Parameters
            ----------
            solution : array
                portfolio
            R : array
                an array of expected returns

            Returns
            -------
            float
                portfolio's return
        """

        return np.sum(solution*R)
        
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
        


    def select_portfolio(self,front,function1,function2,policy):

        """selects a portfolio from the front following a risk or return policy

            Parameters
            ----------
            front : array
                array of optimal portfolios
            function1 : array
                array of corresponfing portfolios risks
            function2 : array
                array of corresponfing portfolios returns
            policy : str
                selection policy

            Returns
            -------
            array
                the selected portfolio
        """

        if policy=='return':
            return front[np.argmax(function2)]
        if policy=='risk':
            return front[np.argmin(function1)]
        


    def choose_portfolio(self,front,function1,function2,policy,L,B):
        if policy=='mean':
            mean_risk=np.mean(function1)
            selected_portfolio=front[self.closest_point(mean_risk,function1)]
            return selected_portfolio
        else :
            return self.select_portfolio(front,function1,function2,policy)

        
    def optimize(self,predictions,run_day,C=0,Q=0.6,Q_rfa=1,num_of_generations=250, num_of_individuals=200, num_of_tour_particips=10,tournament_prob=1, p_c=0.5,p_m=0.5,p_risked=0.8,full=True,policy='return'):
        
        """optimizes the portfolio following constraints, predictions and historical risk

            Parameters
            ----------
            predictions : array
                array of optimal portfolios
                
            run_day : datetime
                array of corresponfing portfolios risks
                
            C : int
                array of corresponfing portfolios returns
                
            Q : float
                maximal quantity to be invested in a risky asset
                
            Q_rfa : sloat
                maximal quantity to be invested in the risk free asset
                
            num_of_generations : int
                number of NSGAII generations 
                
            num_of_individuals : int
                NSGAII population 
                
            num_of_tour_particips : int
                NSGAII tournament participants 
                
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
                
            policy : str
                selection policy


            Returns
            -------
            array
                the selected optimal portfolio
        """
        
        L=[]
        B=[]
        
        all_real_returns=self.get_all_real_returns()
        
        cov=all_real_returns[all_real_returns.index<=run_day].cov().values


        def f1(x):
            return self.risk_portfolio(x,cov)

        def f2(x):
            return self.return_portfolio(x,predictions)

        portfolio_size=self.N
        Q=Q
        Q_rfa=Q_rfa
        if(C==0):
            C=portfolio_size

        problem = Problem(portfolio_size=portfolio_size, objectives=[f1, f2],Q=Q,Q_rfa=Q_rfa,C=C)
        evo = Evolution(problem, num_of_generations, num_of_individuals, num_of_tour_particips,tournament_prob, p_c,p_m,p_risked,full=True)
        solutions=evo.evolve()
        func = [i.objectives for i in solutions]
        function1 = [i[0] for i in func]
        function2 = [i[1] for i in func]

        plt.scatter(function1,function2)
        plt.title('Paret Set of Optimar Portfolios')
        plt.xlabel('Risk')
        plt.ylabel('Return')

        plt.savefig(self.portfolios_path+'/front_'+str(run_day)+'.png')


        selected_portfolio=self.choose_portfolio(solutions,function1,function2,policy,L,B)
        
        with open(self.portfolios_path+'/portfolio_'+str(run_day)+'.p', 'wb') as f:
            pickle.dump(selected_portfolio, f)
        
        
        return selected_portfolio
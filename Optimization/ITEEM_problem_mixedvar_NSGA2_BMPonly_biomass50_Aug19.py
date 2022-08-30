# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 18:50:25 2021

@author: Shaobin
"""
import autograd.numpy as anp
import numpy as np
from ITEEM import ITEEM
from pymoo.model.problem import Problem

from pymoo.model.repair import Repair
# from pymoo.model.callback import Callback
from pymoo.optimize import minimize
from pymoo.algorithms.nsga2 import NSGA2

# get ref_dirs points
# ref_dirs = get_reference_directions("energy", 4, 100, seed=1)
# get_visualization("scatter").add(ref_dirs).show()
# create the reference directions to be used for the optimization
# ref_dirs = get_reference_directions("das-dennis", 4, n_partitions=6)
# plt.scatter(ref_dirs)

# set up decision variables
n_bmp = 7
n_wwt = 0
n_gp  = 0
n_var = n_bmp*10 + n_wwt + n_gp
x_lower = np.zeros((n_var,))
x_upper = np.ones((n_var,))
x_upper[-4] = 4
for i in range(1,11):
    x_upper[n_bmp*i-1] = 0.25
    

class ITEEM_problem(Problem):
    def __init__(self, **kwargs):
        super().__init__(n_var=n_var, n_obj=5, n_constr=2, elementwise_evaluation=True, **kwargs)
        self.xl = x_lower
        self.xu = x_upper
               
    def _evaluate(self, x, out, *args, **kwargs):
        '''
        x1,x2,x3,x4,x5,x6: contiuous variable ranging from 0 to 1
        '''
        tech_wwt_list = ['AS', 'ASCP', 'EBPR_basic', 'EBPR_acetate', 'EBPR_StR']
        # tech_wwt_list[4]
        tech_GP_list = [int(1), int(2)]
        
        # sw start from 1
        cluster1 = [9,10,11,29,41]
        cluster2 = [19,32,34,37]
        cluster3 = [4,6,12]
        cluster4 = [28,30,33]
        cluster5 = [35,36,39,43,44,45]
        cluster6 = [7,16,20,21,22,24,25,26,27,38,40,42]
        cluster7 = [1,2,3]
        cluster8 = [13,14,18,23]
        cluster9 = [17]
        cluster10 = [5,15]
        clusters = [cluster1, cluster2, cluster3, cluster4, cluster5, cluster6, cluster7, cluster8, cluster9, cluster10]
        
        # sw start from 0
        landuse_matrix = np.zeros((45,62)) # BMP 37, 39, 46, 47, 48, 55         
        i = 0
        for cluster in clusters:
            for sw in cluster:
                landuse_matrix[sw-1, 0] = x[i]
                landuse_matrix[sw-1,37] = x[i+1]
                landuse_matrix[sw-1,39] = x[i+2]
                landuse_matrix[sw-1,46] = x[i+3]
                landuse_matrix[sw-1,47] = x[i+4]
                landuse_matrix[sw-1,48] = x[i+5]
                landuse_matrix[sw-1,55] = x[i+6]
            i = i + 7
            
        landuse_matrix[7,:] = 0
        if x[-1]==2 or x[-2]==2 or x[-3]==2:
            landuse_matrix[7,1] = 1.0  # assign BMP for subwatershed 8
        else: landuse_matrix[7,19] = 1.0 # 
        
        solution = ITEEM(landuse_matrix = landuse_matrix, tech_wwt='AS',
                        limit_N = 10.0, tech_GP1=1, 
                        tech_GP2=1, tech_GP3=1)
        
        output = solution.run_ITEEM_opt(sg_price=0.05)
        f1 = output[0]
        f2 = output[1]
        f3 = output[2]
        f4 = output[3]
        f5 = output[4]
        g1 = (output[5] - 774310*0.85)/(7*(10**6))   # 10% reduction for P
        g2 =  (output[6] - 7927670*0.85)/(7*(10**6)) # 15% reduction
        
        out['F'] = anp.column_stack([f1, f2, f3, f4, f5])
        out['G'] = anp.column_stack([g1, g2])
        
        
# add equal constraints
class MyRepair(Repair):
    def _do(self, problem, pop, **kwargs):
        for ind in pop:
            # ind.X[:5] = ind.X[:5] / ind.X[:5].sum()
            for i in range(10):
                ind.X[7*i:7*(i+1)] = ind.X[7*i:7*(i+1)] / ind.X[7*i:7*(i+1)].sum()

                if ind.X[7*i+6] > 0.25:
                    extra = ind.X[6+7*i] - 0.25
                    ind.X[7*i+6] = 0.25
                    # for j in range(5):
                    ind.X[7*i]   = ind.X[7*i]   + (extra*ind.X[7*i])  /ind.X[7*i:7*(i+1)-1].sum()
                    ind.X[7*i+1] = ind.X[7*i+1] + (extra*ind.X[7*i+1])/ind.X[7*i:7*(i+1)-1].sum()
                    ind.X[7*i+2] = ind.X[7*i+2] + (extra*ind.X[7*i+2])/ind.X[7*i:7*(i+1)-1].sum()
                    ind.X[7*i+3] = ind.X[7*i+3] + (extra*ind.X[7*i+3])/ind.X[7*i:7*(i+1)-1].sum()
                    ind.X[7*i+4] = ind.X[7*i+4] + (extra*ind.X[7*i+4])/ind.X[7*i:7*(i+1)-1].sum()
                    ind.X[7*i+5] = ind.X[7*i+5] + (extra*ind.X[7*i+5])/ind.X[7*i:7*(i+1)-1].sum()       
        return pop

# part 2: optimization
# if the first variable is an integer and the second a real value
mask1 = ['real' for i in range(n_bmp*10)]
# mask2 = ['int', 'int', 'int', 'int']
mask = mask1

from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.operators.mixed_variable_operator import MixedVariableSampling, MixedVariableMutation, MixedVariableCrossover

# from pymoo.operators.mixed_variable_operator import MixedVariableSelection
sampling = MixedVariableSampling(mask, {
    "real": get_sampling("real_random"),
    "int": get_sampling("int_random")
})

crossover = MixedVariableCrossover(mask, {
    "real": get_crossover("real_sbx", prob=0.9, eta=3.0),
    "int": get_crossover("int_sbx", prob=0.9, eta=3.0)
})

mutation = MixedVariableMutation(mask, {
    "real": get_mutation("real_pm", eta=3.0),
    "int": get_mutation("int_pm", eta=3.0)
})

# the optimization method is called with the operators defined above
from pymoo.factory import get_crossover, get_mutation, get_sampling

'''set up parallelization using starmap interface'''
# from multiprocessing.pool import ThreadPool
# the number of threads to be used
# n_threads = 8
# initialize the pool
# pool = ThreadPool(n_threads)
# define the problem by passing the starmap interface of the thread pool
# problem = ITEEM_problem(parallelization = ('starmap', pool.starmap))
problem = ITEEM_problem()

from pymoo.util.termination.default import MultiObjectiveDefaultTermination

termination = MultiObjectiveDefaultTermination(
    # x_tol=1e-8,
    # cv_tol=1e-6,
    f_tol=0.005,
    nth_gen=5,
    n_last=20,
    n_max_gen=500,
    n_max_evals=20000)

# set terminatin
algorithm = NSGA2(
    pop_size=100,
    # ref_dirs=ref_dirs,
    n_offsprings=40,
    repair=MyRepair(),
    sampling=sampling,
    crossover=crossover,
    mutation=mutation,
    eliminate_duplicates=True)

res = minimize(
    problem,
    algorithm,
    termination,
    # =('n_gen',100),
    # ('n_gen', 500),
    # seed=1,
    verbose=True,
    # return_least_infeasible=True,
    save_history=True)
# pool.close()

print("Best solution found: %s" % res.X)
print("Function value: %s" % res.F)
# print("Aggregated constraint violation: %s" % res.CV)
# print('Running time is {:.1f} seconds'.format(end-start))
print("Optimization time (mins) is {:.1f} mins".format(res.exec_time/60))
# print("Time per run (s): %s" % (res.exec_time))
# print('sum of BMPs: %s' %res.X[:5].sum())

# for i in range(10):
#     for j in range(res.F.shape[0]):
#         print('sum of BMPs', res.X[0,6*i:6*(i+1)].sum())

# print(res_s1.X[:6].sum())
# for ind in res_s1.pop:
#     print(ind.X[:6].sum())
# res_s1.history[0].pop
n_evals = np.array([e.evaluator.n_eval for e in res.history])

# opt_F: solution in each generation
# opt_F2 = np.array([e.opt[0].F for e in res.history])
opt_F = res.F
# res_s1.history[-1].opt[-1].X  #
opt_X = res.X.astype('float')   #(n_nds,64)
opt_landuse = opt_X[:,:n_bmp*10]

'''save data'''
import scipy.io
scipy.io.savemat('C:\ITEEM\Optimization\solutions\opt_F_NSGA2_BMPsonly_biomass50_Sept13_2021.mat', mdict={'out': opt_F}, oned_as='row')
scipy.io.savemat('C:\ITEEM\Optimization\solutions\opt_X_NSGA2_BMPsonly_biomass50_Sept13_2021.mat', mdict={'out': opt_X}, oned_as='row')

'''post analysis'''
opt_X = scipy.io.loadmat(r'C:\ITEEM\Optimization\solutions\opt_X_NSGA2_BMPsonly_biomass50_Sept13_2021.mat')['out']
# opt_X_BMP_Tech = scipy.io.loadmat(r'C:\ITEEM\Optimization\solutions\opt_X_NSGA2_BMPs_Tech_biomass50_Aug19_2021.mat')['out']

# opt_landuse = opt_X[:,:60]
#  10 clusters 
cluster1 = [9,10,11,29,41]
cluster2 = [19,32,34,37]
cluster3 = [4,6,12]
cluster4 = [28,30,33]
cluster5 = [35,36,39,43,44,45]
cluster6 = [7,16,20,21,22,24,25,26,27,38,40,42]
cluster7 = [1,2,3]
cluster8 = [13,14,18,23]
cluster9 = [17]
cluster10 = [5,15]
clusters = [cluster1, cluster2, cluster3, cluster4, cluster5, 
            cluster6, cluster7, cluster8, cluster9, cluster10]

landuse_matrix = np.zeros((opt_X.shape[0],45,62)) # BMP 37, 39, 46, 47, 48, 55
for j in range(opt_X.shape[0]):
    opt_landuse = opt_X[j,:70].astype('float')    
    i = 0
    for cluster in clusters:
        for sw in cluster:
            # print(sw)
            landuse_matrix[j,sw-1, 1] = opt_landuse[i]
            landuse_matrix[j,sw-1,37] = opt_landuse[i+1]
            landuse_matrix[j,sw-1,39] = opt_landuse[i+2]
            landuse_matrix[j,sw-1,46] = opt_landuse[i+3]
            landuse_matrix[j,sw-1,47] = opt_landuse[i+4]
            landuse_matrix[j,sw-1,48] = opt_landuse[i+5]
            landuse_matrix[j,sw-1,55] = opt_landuse[i+6]
        i = i + 7
    landuse_matrix[:,7,0] = 1
    landuse_matrix[:,30,0] = 1

landuse_matrix_sum = landuse_matrix.sum(axis=2)
# from Optimization.plot_map_ITEEM_opt_ave import opt_landuse
# landuse_matrix = opt_landuse('NSGA2_BMPs_Tech_biomass50_June2021')

'''average land use calculations'''
opt_X_mean = opt_X.mean(axis=0)
bmp_list =[1, 37, 39, 46, 47, 48, 55]
a = 0
for i in bmp_list: 
    a += landuse_matrix[:,:,i].mean()
    print(landuse_matrix[:,:,i].mean())
    
landuse_matrix_sum = landuse_matrix.sum(axis=2)
    
from Submodel_SWAT.SWAT_functions import basic_landuse
land_total, land_agri = basic_landuse()
land_total = land_total.iloc[:,-1]

landuse_matrix_ha = np.zeros((opt_X.shape[0],45,62))
for i in range(opt_X.shape[0]):
    landuse_matrix_ha[i,:,:] = np.multiply(landuse_matrix[i,:,:],np.repeat(land_agri,62,1))

landuse_matrix_ha_mean = landuse_matrix_ha.mean(axis=0)
landuse_matrix_ha_mean2 = landuse_matrix_ha_mean.sum(axis=1)
landuse_matrix_ha_bmp = landuse_matrix_ha_mean.sum(axis=0)

total = landuse_matrix_ha_mean2.sum() - landuse_matrix_ha_mean2[7] - landuse_matrix_ha_mean2[31]

a = 0
for i in bmp_list:
    a += landuse_matrix_ha_bmp[i]/total
    print(landuse_matrix_ha_bmp[i]/total)

landuse_matrix_ha_sumbybmp = landuse_matrix_ha.sum(axis=2)
landuse_matrix_ha_sum = landuse_matrix_ha_sumbybmp.sum(axis=1)

a=0
for i in bmp_list: 
    a += landuse_matrix_ha[:,:,i].mean()
    print(landuse_matrix_ha[:,:,i].mean())

    
'''Start: Nov 24, 2021'''
lanuse_matrix_allocation_per_run = np.zeros((100,7))
for i in range(100):
    k=0
    for j in bmp_list:
        print(k)
        lanuse_matrix_allocation_per_run[i,k] = landuse_matrix_ha[i,:,j].sum()/total
        k+=1

  
lanuse_matrix_allocation_per_run.sum(axis=1)
'''End: Nov 24, 2021'''



'''BMPs only'''
import pandas as pd
from Optimization.plot_map_ITEEM_opt_ave import opt_landuse
landuse_matrix_BMPonly = opt_landuse('NSGA2_BMPsonly_biomass50_Sept13_2021')
# landuse_matrix_BMPonly = landuse_matrix
df_spider_output = pd.DataFrame()
df_obj = pd.DataFrame()

for i in range(100):
    print(i)
    output = ITEEM(landuse_matrix_BMPonly[i,:,:], tech_wwt='AS', limit_N=10.0, tech_GP1=1, tech_GP2=1, tech_GP3=1)
    run_ITEEM_opt = output.run_ITEEM_opt(sg_price=0.05)
    df_spider_output['run'+str(1+i)] = run_ITEEM_opt[-1]
    df_obj['run'+str(1+i)] = [run_ITEEM_opt[0], run_ITEEM_opt[1], run_ITEEM_opt[2], run_ITEEM_opt[3], run_ITEEM_opt[4]]
mean = df_spider_output.mean(axis=1)

df = np.array(df_obj.T)
scipy.io.savemat('C:\ITEEM\Optimization\solutions\opt_F_NSGA2_BMPsonly_biomass50_Sept13_2021.mat', mdict={'out':df}, oned_as='row')

'''Start section: calculate fold of WTP'''
baseline_benefit = 529783859.9  # $/yr
fold50 = []
for i in range(df_spider_output.shape[1]):
    a = (baseline_benefit - df_spider_output.iloc[-4,i])/df_spider_output.iloc[-5,i]
    fold50.append(a)
    # print(fold)
sum(fold50)/len(fold50)
'''End section: calculate fold of WTP'''






'''Convergence'''
n_evals = []    # corresponding number of function evaluations\
F       = []    # the objective space values in each generation
cv      = []    # constraint violation in each generation

# iterate over the deepcopies of algorithms
for algorithm in res.history:    
    # store the number of function evaluations
    n_evals.append(algorithm.evaluator.n_eval)
    # retrieve the optimum from the algorithm
    opt = algorithm.opt
    # store the least contraint violation in this generation
    cv.append(opt.get("CV").min())
    # filter out only the feasible and append
    feas = np.where(opt.get("feasible"))[0]
    _F = opt.get("F")[feas]
    F.append(_F)

'''Convergence: Hypervolume'''
pf = problem.pareto_front(flatten=True, use_cache=False)
import matplotlib.pyplot as plt
from pymoo.performance_indicator.hv import Hypervolume

# MODIFY - this is problem dependend
ref_point = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
# ref_point = np.array([1.0, 1.0])

# create the performance indicator object with reference point
metric = Hypervolume(ref_point=ref_point, normalize=False)

# calculate for each generation the HV metric
hv = [metric.calc(f) for f in F]
# visualze the convergence curve
plt.plot(n_evals, hv, '-o', markersize=4, linewidth=2)
plt.title("Convergence")
plt.xlabel("Function Evaluations")
plt.ylabel("Hypervolume")
plt.show()

import scipy.io
hv_n_evals = [hv, n_evals]
# scipy.io.savemat('C:\ITEEM\Optimization\solutions\hv_NSGA2_BMPonly_biomass50_Aug19_2021.mat', mdict={'out':hv_n_evals}, oned_as='row')

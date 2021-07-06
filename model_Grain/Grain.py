# -*- coding: utf-8 -*-
"""
Author: Shaobin Li (shaobin@illinois.edu)
Project: INFEWS - ITEEM

Purpose: P recovery and techno-economic anlaysis for grain processing
"""

# import packages
import pandas as pd
import numpy_financial as npf
from model_Economics.discount_functions import annuity_factor

# set up global variables
df_wetmilling = pd.read_excel(r'C:\ITEEM\Submodel_Grain\Grain_lookup.xlsx', sheet_name='Wet_milling',
                             index_col=0, nrows=40, usecols = 'B:H')
df_drygrind = pd.read_excel(r'C:\ITEEM\Submodel_Grain\Grain_lookup.xlsx', sheet_name='Dry_grind',
                             index_col=0, nrows=40, usecols = 'B:M')

depreciation_rate1 = [0.1429, 0.2449, 0.1749, 0.1249, 0.0892, 0.0893, 0.0893, 0.0446] # MACRS Depreciation (8 years)
depreciation_rate2 = [0 for i in range(12)]
depreciation_rate = depreciation_rate1 + depreciation_rate2

class Grain(object):  
    def __init__(self, plant_type, plant_capacity, tech_GP):
        '''
        plant_type (int) = {1:'corn_wet', 2:'corn_dry', 3:'soybean'}
        plant_capacity (metric ton corn/yr for corn_wet and million gallon ethanol/yr for corn_dry) (float) 
                        [0.67, 0.95, 1.7, 2.1, 5.0, 40, 80, 120, 240, 300]
        tech_GP (int) = {1: 'base case', 2: 'P recovery case'}
        '''
        self.plant_type = plant_type
        self.plant_capacity = plant_capacity
        self.column = str(self.plant_capacity) + 'M'
        if plant_type == 2:
            self.plant_capacity = 8580*plant_capacity/(10**6) # convert MGD of ethanol to metric ton of corn
        self.tech_GP = tech_GP
        tech_GPs = {1: 'base case', 2: 'P recovery case'}      
        plant_types = {1:'corn_wet', 2:'corn_dry', 3:'soybean'}
        plant_capacities = [0.67, 0.95, 1.7, 2.1, 5.0, 40, 80, 120, 240, 300]
        if self.tech_GP not in tech_GPs.keys():
            raise ValueError("Invalid scenario")
        if self.plant_type not in plant_types.keys():
            raise ValueError("Invalid plant_type")

        if self.plant_type == 1:
            self.df = df_wetmilling
        elif self.plant_type ==2:
            self.df = df_drygrind
        
    def get_product(self):       
        '''return a list of products'''
        product = []
        if self.plant_type == 1:
            starch = self.df[self.column][3]*self.plant_capacity*10**6   # ton/yr
            # cgf = self.df[self.column][4]*self.plant_capacity*10**6    # ton/yr 
            cgm = self.df[self.column][5]*self.plant_capacity*10**6      # ton/yr
            dried_germ = self.df[self.column][6]*self.plant_capacity*10**6  # ton/yr
            if self.tech_GP==1:
                cgf = 0.159*self.plant_capacity*10**6      # ton/yr, 0.159 Corn Gluten Feed (MT CGF/MT corn grain)
            else:
                cgf = 0.155*self.plant_capacity*10**6      # ton/yr, 0.155 Corn Gluten Feed (MT CGF/MT corn grai
            product = [starch, cgf, cgm, dried_germ]
        
        if self.plant_type == 2:
            ethanol = self.df[self.column][3]*self.plant_capacity*10**6     # gal/yr
            # ddgs = self.df[self.column][4]*self.plant_capacity*10**3        # ton/yr
            corn_oil = self.df[self.column][4]*self.plant_capacity*10**3    # ton/yr
            if self.tech_GP==1:
                ddgs = 289.2*self.plant_capacity*10**3    # ton/yr, 289.2 DDGS (kg /MT corn grain)
            else:
                ddgs = 285.6*self.plant_capacity*10**3    # ton/yr, 285.6 DDGS (kg /MT corn grain)
            product = [ethanol, ddgs, corn_oil]
            
        return product
    
    def get_energy_use(self):
        '''return a list of energy_use, including natural gas and electricity in MJ/yr'''
        energy_use = []
        op_ele_base = self.df[self.column][7]*self.plant_capacity*1000*3.6*10**6 # 1 kWh = 3.6 MJ
        op_ng_base = self.df[self.column][8]*self.plant_capacity*1000*10**6
        op_tot_base = op_ele_base + op_ng_base
        if self.tech_GP==1:
            energy_use = [op_tot_base]
        elif self.tech_GP==2:
            op_ele_rp = self.df[self.column][30]*self.plant_capacity*3.6*10**6 # 1 kWh = 3.6 MJ
            op_ng_rp = self.df[self.column][31]*self.plant_capacity*10**6
            op_tot_rp = op_ele_rp + op_ng_rp
            op_tot = op_tot_base + op_tot_rp
            energy_use = [op_tot_base, op_tot_rp, op_tot]
        return energy_use
    
    def get_water_use(self):
        '''return a list of water_use in m3/year'''
        water_use = []
        water_base = self.df[self.column][9]*self.plant_capacity*10**6
        if self.tech_GP==1:
            water_use = [water_base]
        elif self.tech_GP==2:
            water_rP = self.df[self.column][32]*self.plant_capacity*10**6
            water_use = [water_base, water_rP, water_base + water_rP]
        return water_use
    
    def get_rP(self):
        '''return P recovered (kg/yr)'''
        if self.tech_GP==1:
            rP = [0, 0]
        elif self.tech_GP==2:
            rP_phytin = self.df[self.column][18]
            rp_P_complex = self.df[self.column][19]
            rP = [rP_phytin, rp_P_complex]
        return rP
    
    def get_cost(self, feedstock_index=1.0, chem_index=1.0, utility_index=1.0):
        '''
        return a list: gross operating cost ($/year): sum of total raw materials, utilities, labor, facility dependent cost.
        0.136 $/kg yellow dent corn in 2017
        '''
        cost = []
        cost_utility = self.df[self.column][14]*utility_index      # $/yr
        cost_feedstock = self.df[self.column][15]*feedstock_index  # $/yr
        cost_facility_dependent = self.df[self.column][16]         # $/yr
        cost_chemical = self.df[self.column][17]*chem_index        # $/yr
        cost_op_base = cost_feedstock + cost_chemical + cost_utility + cost_facility_dependent # $/yr
        cost_labor = self.df[self.column][13] - self.df[self.column][14] - self.df[self.column][15] \
        - self.df[self.column][16] - self.df[self.column][17]  # $/yr
        cost_op_base = cost_op_base + cost_labor
        
        if self.tech_GP==1:
            cost = [cost_op_base]
        elif self.tech_GP==2:
            cost_utility_rp = self.df[self.column][36]*utility_index      # $/yr
            cost_feedstock_rp = self.df[self.column][37]*feedstock_index  # $/yr
            cost_facility_dependent_rp = self.df[self.column][38]         # $/yr
            cost_chemical_rp = self.df[self.column][39]*chem_index        # $/yr

            cost_op_rP = cost_feedstock_rp + cost_chemical_rp + cost_utility_rp + cost_facility_dependent_rp # $/yr
            cost_op_total = cost_op_base + cost_op_rP                     # $/yr
            cost = [cost_op_base, cost_op_rP, cost_op_total]
        return cost

    def get_revenue(self, starch_price=320, ethanol_price=1.32, rp_price=393.4, p_credit=0.0, grain_product_index = 1.0, rP_index=1.0):
        '''
        return revenue gross revenue ($/yr)
        starch_price = 320 $/ton; rP_price = 393.4 $/ton; CGF = 98.7 $/ton; CGM = 454.5 $/ton; Dried germ = 300 $/ton
        DDGS_price = 152.8 $/ton; Corn oil = 750 $/ton; Ethanol = 1.32 $/gal; p_credit=5% price increase for byproduct
        '''
        revenue = []
        plant_capacity = self.df[self.column][0]  # MT corn/yr
        if self.plant_type == 1:
            revenue_base = plant_capacity*(self.df[self.column][3]*starch_price + self.df[self.column][4]*98.7 + 
                                           self.df[self.column][5]*454.5 + self.df[self.column][6]*300)*grain_product_index
            if self.tech_GP==1:
                revenue = [revenue_base]
            elif self.tech_GP==2:
                revenue_tot = plant_capacity*(self.df[self.column][25]*starch_price \
                + self.df[self.column][26]*98.7*(1+p_credit) \
                + self.df[self.column][27]*454.5 \
                + self.df[self.column][28]*300)*grain_product_index \
                + self.df[self.column][19]*rp_price*rP_index/1000 \
                + plant_capacity*self.df[self.column][26]/2276*232*p_credit # 2726 ton of CGF or DDGS required for StoneRidge feedlot; 232kg P reduction/yr for StoneRidge
                
                p_credit2 = self.df[self.column][26]*98.7/1000*p_credit
                revenue_rP = revenue_tot - revenue_base
                revenue = [revenue_base, revenue_rP, p_credit2, revenue_tot]
        elif self.plant_type == 2:
            revenue_base = plant_capacity*(self.df[self.column][3]*ethanol_price + self.df[self.column][4]*152.8/1000 + 
                                           self.df[self.column][5]/1000*750)*grain_product_index
            if self.tech_GP==1:
                revenue = [revenue_base]
            elif self.tech_GP==2:
                revenue_tot = plant_capacity*(self.df[self.column][25]*ethanol_price \
                + self.df[self.column][26]*152.8/1000*(1+p_credit) \
                + self.df[self.column][27]/1000*750)*grain_product_index \
                + self.df[self.column][19]*rp_price*rP_index/1000 \
                + plant_capacity*self.df[self.column][26]/2276/1000*232*p_credit
                # 1029600*0.3/2276*232*20
                p_credit2 = self.df[self.column][26]*152.8/1000*p_credit
                revenue_rP = revenue_tot - revenue_base
                revenue = [revenue_base, revenue_rP, p_credit2, revenue_tot]       
        return revenue
    
    def get_profit(self, r, starch_price=320, ethanol_price=1.32, rp_price=393.4, p_credit=0.0, grain_product_index=1.0, 
                   rP_index=1.0, feedstock_index=1.0, chem_index=1.0, utility_index=1.0, tax=0.35):
        '''return net profit as annualized value, $/yr '''
        revenue = self.get_revenue(starch_price=starch_price, ethanol_price=ethanol_price, p_credit=p_credit, rp_price=rp_price, 
                                          grain_product_index=grain_product_index, rP_index=rP_index)[-1]
        cost = self.get_cost(feedstock_index=feedstock_index, chem_index=chem_index, utility_index=utility_index)[-1]
        taxable_income = revenue - cost
        net_profit = taxable_income*(1-tax)           # assume 40% as default tax
        depreciation_cost = self.df[self.column][16]  # $/yr
        cash_flow = net_profit + depreciation_cost    # $/yr
        if self.tech_GP==1:
            direct_fixed_cost = self.df[self.column][12]  # $/yr
        elif self.tech_GP==2:
            direct_fixed_cost = self.df[self.column][12] + self.df[self.column][34]  # $/yr
        working_capital = direct_fixed_cost*0.05      # assume 5%
        cash_flow_list = [-(direct_fixed_cost*tax+working_capital), -(direct_fixed_cost*(1-tax))]
        for i in depreciation_rate:
            cash_flow_i = (revenue-(cost-depreciation_cost) - \
                           direct_fixed_cost*i)*(1-tax) + direct_fixed_cost*i
            cash_flow_list.append(cash_flow_i)
        npv = npf.npv(r, cash_flow_list)
        irr = npf.irr(cash_flow_list)
        annualized_value = npv/annuity_factor(20, r)
        
        a = [cost, revenue, taxable_income, net_profit]
        
        return a, irr, npv, annualized_value

# wet_1 = Grain(plant_type=1, plant_capacity=5.0, tech_GP=2)
# (Grain(plant_type=1, plant_capacity=2.1, tech_GP=1).get_energy_use()[-1] + \
#     Grain(plant_type=1, plant_capacity=5.0, tech_GP=1).get_energy_use()[-1] + \
#         Grain(plant_type=2, plant_capacity=120, tech_GP=1).get_energy_use()[-1])/(10**6)

# Grain(plant_type=1, plant_capacity=2.1, tech_GP=2).get_energy_use()
# wet_1_base = Grain(plant_type=1, plant_capacity=2.1, tech_GP=1).get_profit(r=0.07, p_credit=0.0)[0]
# wet_1_rp = Grain(plant_type=1, plant_capacity=2.1, tech_GP=2).get_profit(r=0.07, p_credit=0.0)[-1]
# wet_1_base - wet_1_rp

# wet_2_base = Grain(plant_type=1, plant_capacity=5.0, tech_GP=1).get_profit(r=0.07, p_credit=0.0)[-1]
# wet_2_rp = Grain(plant_type=1, plant_capacity=5.0, tech_GP=2).get_profit(r=0.07, p_credit=0.0)[-1]
# wet_2_base - wet_2_rp

# dry_1_base = Grain(plant_type=2, plant_capacity=120, tech_GP=1).get_profit(r=0.07, p_credit=0.0)[-1]
# dry_1_rp = Grain(plant_type=2, plant_capacity=120, tech_GP=2).get_profit(r=0.07, p_credit=0.0)[-1]
# dry_1_base - dry_1_rp

# annuity_factor(20, 0.07)
# chem_index = 1.0
# feedstock_index = 1.0
# grain_product_index = 1.2
# utility_index = 1.0
# rP_index = 1.2

# wet_1.get_revenue(grain_product_index = grain_product_index, rP_index=rP_index)[-1] \
#     + wet_2.get_revenue(grain_product_index = grain_product_index,rP_index=rP_index)[-1] \
#         + dry_1.get_revenue(grain_product_index = grain_product_index, rP_index=rP_index)[-1]

# wet_1.get_cost(chem_index=chem_index, feedstock_index=feedstock_index, utility_index=utility_index)[-1] \
#     + wet_2.get_cost(chem_index=chem_index, feedstock_index=feedstock_index, utility_index=utility_index)[-1] \
#         + dry_1.get_cost(chem_index=chem_index, feedstock_index=feedstock_index, utility_index=utility_index)[-1]

# wet_1.get_profit(r=0.07, chem_index=chem_index, feedstock_index=feedstock_index, grain_product_index=grain_product_index)[-1] \
#     + wet_2.get_profit(r=0.07, chem_index=chem_index, feedstock_index=feedstock_index, grain_product_index=grain_product_index)[-1] \
#         + dry_1.get_profit(r=0.07, chem_index=chem_index, feedstock_index=feedstock_index, grain_product_index=grain_product_index)[-1]

# wet_1.get_cost(chem_index=1.2)
# wet_1.get_revenue()
# wet_1_rP = Grain(plant_type=1, plant_capacity=2.1, tech_GP=2, raw_P=1)
# wet_1.get_revenue()[-1] + wet_2.get_revenue()[-1] + dry_1.get_revenue()[-1]
# wet_1_rP.get_revenue()[-1]
# wet_1_base.get_revenue()[-1] - wet_1_base.get_cost()[-1]
# wet_2.get_profit()
# wet_1.get_cost()[-1] + wet_2.get_cost()[-1] + dry_1.get_cost()[-1]

# wet_1_rP.get_energy_use()
# wet_2_rP = GP(plant_type=1, plant_capacity=5.0, tech_GP=2, raw_P=1)
# wet_2_rP.get_energy_use()
# dry_rP = Grain(plant_type=2, plant_capacity=120, tech_GP=2, raw_P=1)
# dry_rP.get_energy_use()
# dry_rP.get_rP_phytin()
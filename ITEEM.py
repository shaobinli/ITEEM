# -*- coding: utf-8 -*-
"""
Project: NSF INFEWS project (Award Abstract #1739788)
PI: Ximing Cai

Author: Shaobin Li (shaobin@illinois.edu)

Purpose:
The ITEEM that includes the five component models:
1) SWAT: represented by a response matrix method
2) Wastewater treatment (WWT): represented by neural netowrks to represent different wastewater treatment technologies
3) Grain processing (GP): represented by a lookup table with different P recovery technologies
4) Economics: economics of crop yield and willingness to pay by farmer and public 
5） Dringkin water treatment (DWT): energy and chemicals needed to treat different N conc. in drinking water
"""

# load general packages 
import numpy as np
import numpy_financial as npf
import pandas as pd
import time

# load new packages developed for ITEEM
from Submodel_WWT.SDD_analysis.wwt_model_SDD import WWT_SDD
from Submodel_SWAT.SWAT_functions import loading_outlet_USRW, sediment_instream, get_P_riverine, get_P_biosolid, loading_outlet_USRW_opt_v2 
from Submodel_SWAT.crop_yield import get_yield_crop, get_crop_cost, get_P_fertilizer, get_P_crop
from Submodel_Grain.Grain import Grain
from Submodel_DWT.DWT_daily import DWT
from Submodel_Economics.Economics import Economics
from Submodel_Economics.discount_functions import annuity_factor

class ITEEM(object):
    '''
    landuse_matrix: land use decision for BMPs (45,56)
    tech_wwt = ['AS', 'ASCP', 'EBPR_basic', 'EBPR_acetate', 'EBPR_StR']
    limit_N = policy on nitrate concentration in drinking water, default: 10 mg/L
    tech_GP1: for wet milling plant 1, decision values: [1,2]
    tech_GP2: for wet milling plant 2, decision values: [1,2]
    tech_GP3: for dry grind plant, decision values: [1,2]
    '''
    def __init__(self, landuse_matrix, tech_wwt, limit_N, tech_GP1, tech_GP2, tech_GP3):
        self.landuse_matrix = landuse_matrix
        self.tech_wwt = tech_wwt
        self.limit_N = limit_N
        self.tech_GP1 = tech_GP1
        self.tech_GP2 = tech_GP2
        self.tech_GP3 = tech_GP3

    def get_N_outlet(self, nutrient_index, flow_index):
        N_loading = loading_outlet_USRW('nitrate', self.landuse_matrix, self.tech_wwt, nutrient_index, flow_index)
        N_outlet = N_loading[:,:,33]
        return N_outlet
    
    def get_P_outlet(self, nutrient_index, flow_index):
        TP_loading = loading_outlet_USRW('phosphorus', self.landuse_matrix, self.tech_wwt, nutrient_index, flow_index)
        TP_outlet = TP_loading[:,:,33]
        return TP_outlet
    
    def get_streamflow_outlet(self):
        streamflow_loading = loading_outlet_USRW('streamflow', self.landuse_matrix)
        streamflow_outlet = streamflow_loading[:,:,33]
        return streamflow_outlet  

    def get_sediment_outlet(self):
        sediment_outlet = sediment_instream(33, self.landuse_matrix)
        return sediment_outlet
    
    def get_corn(self):
        '''return corn production per year, kg/yr'''
        corn = get_yield_crop('corn', self.landuse_matrix)[1]
        corn = corn.sum(axis=1).mean()
        return corn
    
    def get_soybean(self):
        '''return soybean production per year, kg/yr'''
        soybean = get_yield_crop('soybean', self.landuse_matrix)[1]
        soybean = soybean.sum(axis=1).mean()
        return soybean
    
    def get_biomass(self):
        '''return soybean production per year, kg/yr'''
        biomass = get_yield_crop('switchgrass', self.landuse_matrix)[1]
        biomass = biomass.sum(axis=1).mean()
        return biomass

    def get_cost_energy(self, r=0.07, n_wwt=40, nutrient_index=1.0, flow_index=1.0, 
                        chem_index=1.0, utility_index=1.0, rP_index=1.0, feedstock_index=1.0, crop_index=1.0):
        '''return a numpy array (energy_wwt, energy_grain, energy_water): Million MJ/yr
        7% interest rate, 40 year of lifespan'''
        
        '''*** energy of drinking water in MJ***'''
        DWT_Decatur = DWT(self.limit_N, self.landuse_matrix)
        energy_dwt = DWT_Decatur.get_nitrate_energy()[2].sum()/16
        
        '''*** energy of GP in Million MJ ***'''
        wet_1 = Grain(plant_type=1, plant_capacity=2.1, tech_GP=self.tech_GP1)
        wet_2 = Grain(plant_type=1, plant_capacity=5.0, tech_GP=self.tech_GP2)
        dry_1 = Grain(plant_type=2, plant_capacity=120, tech_GP=self.tech_GP3)  
        energy_grain = wet_1.get_energy_use()[-1] + wet_2.get_energy_use()[-1] + dry_1.get_energy_use()[-1]
        
        '''*** cost in $/yr ***'''
        cost_grain = wet_1.get_cost(feedstock_index, chem_index, utility_index)[-1] \
        + wet_2.get_cost(feedstock_index, chem_index, utility_index)[-1] \
        + dry_1.get_cost(feedstock_index, chem_index, utility_index)[-1]
        
        cost_dwt = DWT_Decatur.get_cost(r, chem_index, utility_index)
        wwt_SDD = WWT_SDD(self.tech_wwt, multiyear=True, start_yr = 2003, end_yr=2018)
        cost_energy_nutrient = wwt_SDD.get_cost_energy_nutrient(1000, self.landuse_matrix, r, n_wwt, 
                                                                nutrient_index, flow_index, 
                                                                chem_index, utility_index,
                                                                rP_index)
        cost_wwt = cost_energy_nutrient[0]
        energy_wwt = cost_energy_nutrient[4]
        rP_amount = cost_energy_nutrient[-4]
        revenue_rP = cost_energy_nutrient[-3]        
        outlet_nitrate = cost_energy_nutrient[-2]
        outlet_tp = cost_energy_nutrient[-1]

        cost_crop = Economics(self.landuse_matrix).get_crop_cost_acf(r)[-1]   # annualized cost, $/yr
        cost_total = cost_dwt + cost_grain + cost_wwt + cost_crop
        return [energy_dwt/(10**6), energy_grain/(10**6), energy_wwt/(10**6),
                cost_dwt, cost_grain, cost_wwt, cost_crop, cost_total, 
                rP_amount, revenue_rP, outlet_nitrate, outlet_tp]
    
    def get_system_revenue(self, r=0.07, grain_product_index = 1.0, rP_index=1.0, 
                           feedstock_index=1.0, chem_index=1.0, utility_index=1.0, crop_index=1.0, sg_price=0.05, cost_SA_EBT=1.0):
        '''return annualized benefit from all submodels'''
        wet_1 = Grain(plant_type=1, plant_capacity=2.1, tech_GP=self.tech_GP1)
        wet_2 = Grain(plant_type=1, plant_capacity=5.0, tech_GP=self.tech_GP2)
        dry_1 = Grain(plant_type=2, plant_capacity=120, tech_GP=self.tech_GP3)
        revenue_GP = wet_1.get_revenue(grain_product_index=grain_product_index, rP_index=rP_index)[-1] \
        + wet_2.get_revenue(grain_product_index=grain_product_index, rP_index=rP_index)[-1] \
        + dry_1.get_revenue(grain_product_index=grain_product_index, rP_index=rP_index)[-1]
        
        cost_GP1, profit_GP1 = wet_1.get_profit(r, grain_product_index=grain_product_index, rP_index=rP_index, 
                                     feedstock_index=feedstock_index, chem_index=chem_index, utility_index=utility_index, cost_SA_EBT=cost_SA_EBT)
        cost_GP2, profit_GP2 = wet_2.get_profit(r, grain_product_index=grain_product_index, rP_index=rP_index, 
                                     feedstock_index=feedstock_index, chem_index=chem_index, utility_index=utility_index, cost_SA_EBT=cost_SA_EBT)
        cost_GP3, profit_GP3 = dry_1.get_profit(r, grain_product_index=grain_product_index, rP_index=rP_index, 
                                     feedstock_index=feedstock_index, chem_index=chem_index, utility_index=utility_index, cost_SA_EBT=cost_SA_EBT)
        cost_GP = cost_GP1 + cost_GP2 + cost_GP3
        profit_GP = profit_GP1 + profit_GP2 + profit_GP3
        
        revenue_crop = Economics(self.landuse_matrix, sg_price=sg_price).get_crop_revenue_acf(r=r, crop_index=crop_index)[-1]
        revenue_total = revenue_GP + revenue_crop
        return profit_GP, cost_GP, revenue_GP, revenue_crop, revenue_total
    
    def get_rP(self):
        '''return rP in kg/yr'''
        rP_1 = Grain(plant_type=1, plant_capacity=2.1, tech_GP=self.tech_GP1).get_rP()[1]
        rP_2 = Grain(plant_type=1, plant_capacity=5.0, tech_GP=self.tech_GP2).get_rP()[1]
        rp_3 = Grain(plant_type=2, plant_capacity=120, tech_GP=self.tech_GP3).get_rP()[1]
        rP = rP_1 + rP_2 + rp_3
        return rP
    
    def get_P_flow(self):
        '''calculate P flow between submodels, metric ton/yr'''
        '''P_riverine'''
        # P_nonpoint, P_point, P_reservoir, P_instream_store, P_total_outlet, struvite
        P_nonpoint, P_point, P_reservoir, P_instream_store, P_total_outlet, struvite = get_P_riverine(self.landuse_matrix, self.tech_wwt)
        P_SDD_influent = 676.8 # MT/yr
        # P_point_baseline = 582.4 # MT/yr
        # P_nonpoint_baseline = 292.9 # MT/yr
        # in_stream_load = P_nonpoint + P_point
        
        '''P_biosolid'''
        P_in_biosolid, P_crop_biosolid, P_riverine_biosolid, P_soil_biosolid = get_P_biosolid(self.tech_wwt)
        
        '''P_crop & P_fertilizer'''
        P_fertilizer = get_P_fertilizer('corn', self.landuse_matrix) # MT/yr
        P_corn_self, _, P_soybean, P_sg = get_P_crop(self.landuse_matrix)
        P_corn_local = P_corn_self + P_crop_biosolid
        P_corn_import = 17966 - P_corn_local
        # P_crop_list = [P_corn_self, P_corn_import, P_soybean, P_sg]
        # P_manure_list = [P_manure, P_manure_runoff, P_manure_soil, P_CGF]
        # P_fertilizer_net = P_fertilizer - P_crop_biosolid

        
        '''P to wastewater and soybean'''
        P_corn_to_wastewater = 1.3 + 1.3*5/2.1  # 1.3 MT P/yr for plant capacity 2.1
        P_human_to_waswater = 67.4              # 67.4 MT/yr from SDD report, Table 3.3.1
        P_soy_to_wastewater = P_SDD_influent - P_corn_to_wastewater - P_human_to_waswater
        P_soy_biorefinery = 1040.6  
        P_soybean_exported = P_soybean - P_soy_biorefinery # MT/yr
        P_soy_product = 458.6 # MT/yr    
                
        '''P_corn_biorefinery'''
        P_in1, P_product1, P_other1, rP1 = Grain(plant_type=1, plant_capacity=2.1, tech_GP=self.tech_GP1).get_P_flow()
        P_in2, P_product2, P_other2, rP2 = Grain(plant_type=1, plant_capacity=5.0, tech_GP=self.tech_GP2).get_P_flow()
        P_in3, P_product3, P_other3, rP3 = Grain(plant_type=2, plant_capacity=120, tech_GP=self.tech_GP3).get_P_flow()
        P_cb_in = P_in1 + P_in2 + P_in3
        P_cb_rP = rP1 + rP2 + rP3

        '''P_manure'''
        P_corn_silage = 24.7                 # 10487*908.6*0.26/100/1000 #10487 kg/ha, 908.6 ha, assume 0.26%
        if self.tech_GP1==1 and self.tech_GP2==1 and self.tech_GP3==1:
            P_CGF = 2726*12/1000             # 2726 ton/yr, total CGF demand for StoneDairy; 12mg/g
            P_manure = 67.8
            P_manure_runoff = 1.932
            P_manure_soil = P_manure - P_manure_runoff - P_corn_silage
        else:
            P_CGF = 2726*2.5/1000            # 2726 ton/yr, total CGF demand for StoneDairy; 12mg/g
            P_manure = 67.8 - (2726*12/1000-2726*2.5/1000)  #
            P_manure_runoff = 1.700
            P_manure_soil = P_manure - P_manure_runoff - P_corn_silage
        
        P_cb_product = P_cb_in - P_cb_rP - P_corn_to_wastewater - P_CGF
        P_rP = P_cb_rP + struvite
        P_soil = P_soil_biosolid + P_manure_soil # P_soil_biosolid highly uncertain
        P_soil_fertilizer = P_fertilizer - P_corn_self - P_soybean - P_sg - P_soil_biosolid - P_nonpoint
        # P_soil_adj = 
        
        '''P_list'''
        P_in_list  = [P_corn_import, P_fertilizer, P_manure, P_human_to_waswater]
        P_out_list = [P_cb_product, P_rP, P_soybean_exported, P_corn_silage, P_soil, P_soil_fertilizer,
                     P_total_outlet,  P_reservoir,  P_instream_store]

        '''adjustment coefficient'''
        P_in = sum(P_in_list); P_out = sum(P_out_list); coef = (P_out - P_in)/P_in
        P_out_list_adj = [(1-coef)*x for x in P_out_list]
        
        if P_soil_fertilizer > 0:
            output_list = [P_corn_import, P_nonpoint, P_corn_self, P_soybean, P_soil_fertilizer, P_sg,
                           P_manure_runoff, P_corn_silage, P_manure_soil, 
                           P_point,  P_crop_biosolid, struvite, P_soil_biosolid,
                           P_cb_product, P_cb_rP,
                           P_total_outlet, P_reservoir, P_instream_store, P_corn_local, 
                           P_soy_biorefinery, P_soybean_exported, P_soy_product, 
                           P_soy_to_wastewater, P_human_to_waswater, P_corn_to_wastewater, P_CGF
                           ]
            
            source = ['Imported corn', 'Fertilizer', 'Fertilizer', 'Fertilizer', 'Fertilizer', 'Fertilizer',  
                      'Manure', 'Manure', 'Manure', 
                      'Wastewater', 'Wastewater', 'Wastewater', 'Wastewater', 
                      'Corn biorefinery', 'Corn biorefinery',
                      'In-stream load', 'In-stream load', 'In-stream load', 'Corn (local)', 
                      'Soybean (local)', 'Soybean (local)', 'Soybean biorefinery', 
                      'P_soy_to_wastewater', 'Human wastewater', 'Corn biorefinery', 'Corn biorefinery'
                      ]
            
            target = ['Corn biorefineries', 'In-stream load', 'Corn (local)', 'Soybean', 'Soil',
                      'Biomass', 'In-stream load', 'Corn silage', 'Soil', 'In-stream load', 
                      'Corn (local)', 'recovered P', 'Soil', 'Products from CBs', 'recovered P', 
                      'Riverine export', 'Reservoir trapping', 'In-stream storage', 'Corn biorefineries',
                      'Soybean biorefinery', 'Soybean (exported)', 'Products from soybean biorefinery', 
                      'Wastewater', 'Wastewater', 'Wastewater', 'Manure'
                      ]
            
        elif P_soil_fertilizer < 0:
            output_list = [P_corn_import, P_nonpoint, P_corn_self+P_soil_fertilizer*0.65, P_soybean+P_soil_fertilizer*0.35, 
                           P_sg, P_manure_runoff, P_corn_silage, 
                           P_point,  P_crop_biosolid, struvite, P_soil_biosolid,
                           P_cb_product, P_cb_rP,
                           P_total_outlet, P_reservoir, P_instream_store,
                           P_corn_self+P_crop_biosolid, P_soil_fertilizer*-0.65, P_soil_fertilizer*-0.35,
                           P_soy_biorefinery, P_soybean_exported, P_soy_product, 
                           P_soy_to_wastewater, P_human_to_waswater, P_corn_to_wastewater, P_CGF
                           ]
            
            source = ['Imported corn', 'Fertilizer', 'Fertilizer', 'Fertilizer', 'Fertilizer', 'Manure', 'Manure',
                      'Wastewater', 'Wastewater', 'Wastewater', 'Wastewater', 'Corn biorefinery', 'Corn biorefinery',
                      'In-stream load', 'In-stream load', 'In-stream load', 'Corn (local)', 'Soil', 'Soil',
                      'Soybean (local)', 'Soybean (local)', 'Soybean biorefinery', 
                      'P_soy_to_wastewater', 'Human wastewater', 'Corn biorefinery', 'Corn biorefinery'
                      ]
            
            target = ['Corn biorefineries', 'In-stream load', 'Corn (local)', 'Soybean', 'Biomass',
                      'In-stream load', 'Corn silage', 'In-stream load', 'Corn (local)', 
                      'recovered P', 'Biosolid', 'Products from CBs', 'recovered P', 
                      'Riverine export', 'Reservoir trapping', 'In-stream storage',
                      'Corn biorefineries', 'Corn (local)', 'Soybean',
                      'Soybean biorefinery', 'Soybean (exported)', 'Products from soybean biorefinery', 
                      'Wastewater', 'Wastewater', 'Wastewater', 'Manure'
                      ]

        return P_in_list, P_out_list_adj, source, target, P_soil_fertilizer, output_list
    
    def run_ITEEM(self, r=0.07, n_wwt=40, nutrient_index=1.0, flow_index=1.0, chem_index=1.0, rP_index=1.0, 
                  utility_index=1.0, grain_product_index=1.0, feedstock_index=1.0, crop_index=1.0, unit_pay=0.95):
        '''
        return a list containg multiple outputs of N, P, streamflow, sediment, 
        energy_dwt, energy_grain, energy_wwt,
        cost_dwt, cost_grain, rP
        '''
        streamflow = self.get_streamflow_outlet()
        streamflow_outlet = streamflow.sum(axis=1).mean()
        sediment_outlet = self.get_sediment_outlet().sum(axis=1).mean()
        sediment_outlet_landscape = loading_outlet_USRW('sediment', self.landuse_matrix)[:,:,33].sum(axis=1).mean()
        
        # cost_dwt, cost_GP, cost_wwt, cost_crop, cost_total = self.get_system_cost(r)
        cost_energy = self.get_cost_energy(r=r, n_wwt=n_wwt, nutrient_index=nutrient_index, flow_index=flow_index, 
                                           chem_index=chem_index, utility_index=utility_index, rP_index=rP_index)
        energy_dwt = cost_energy[0]
        energy_grain = cost_energy[1]
        energy_wwt = cost_energy[2]
        cost_dwt = cost_energy[3]
        cost_grain = cost_energy[4]
        revenue_rP = cost_energy[9]
        cost_wwt = cost_energy[5] - revenue_rP
        cost_crop = cost_energy[6]
        # cost_total = cost_energy[7]
        rP_amount = cost_energy[8]
        outlet_nitrate = cost_energy[-2]
        outlet_tp = cost_energy[-1]
        N_outlet = outlet_nitrate[:,:,33].sum(axis=1).mean()
        P_outlet = outlet_tp[:,:,33].sum(axis=1).mean()
        profit_GP, revenue_GP, revenue_crop, revenue_total = self.get_system_revenue(r=r, grain_product_index=grain_product_index,
                                                                                     rP_index=rP_index, feedstock_index=feedstock_index, 
                                                                                     chem_index=chem_index, utility_index=utility_index, 
                                                                                     crop_index=crop_index)

        nitrate_impro_prt = ((7240 - N_outlet/1000)/7240)/0.45  # baseline nitrate load =7240 Mg/yr
        if nitrate_impro_prt > 0 and nitrate_impro_prt <1.0:
            wtp_nitrate = nitrate_impro_prt*unit_pay*100*113700 # $0.95/1% nitrate improvement, 113700 household
        elif nitrate_impro_prt > 1.0:
            wtp_nitrate = unit_pay*100*113700
        else:
            wtp_nitrate = 0
        tp_impro_prt = ((324 - P_outlet/1000)/324)/0.45  # baseline TP load = 324 Mg/yr
        if  tp_impro_prt > 0 and tp_impro_prt < 1.0:
            wtp_tp = tp_impro_prt*unit_pay*100*113700 # 113700 households
        elif tp_impro_prt > 1.0: 
            wtp_tp = unit_pay*100*113700
        else:
            wtp_tp = 0
            
        wtp = 0.5*wtp_nitrate + 0.5*wtp_tp
        profit_crop = revenue_crop - cost_crop
        system_net_benefit = wtp + revenue_crop + profit_GP - cost_crop - cost_wwt - cost_dwt

        rP_P_complex = self.get_rP()  # kg/yr
        corn = self.get_corn()        # kg/yr
        soybean = self.get_soybean()  # kg/yr
        biomass = self.get_biomass()  # kg/yr
        
        environment = [N_outlet, P_outlet, sediment_outlet_landscape, sediment_outlet, streamflow_outlet]
        energy = [energy_dwt, energy_grain, energy_wwt.mean(), biomass]
        economics = [cost_dwt, cost_grain, cost_wwt, cost_crop,
                     revenue_GP, revenue_crop, profit_crop, profit_GP, wtp, system_net_benefit]
        food = [rP_P_complex, rP_amount, corn, soybean]
        spider_output = [N_outlet, P_outlet, sediment_outlet,streamflow_outlet,
                         energy_dwt, energy_grain, energy_wwt.mean(), biomass,
                         cost_dwt,cost_wwt,profit_crop, profit_GP, wtp, system_net_benefit,
                         rP_P_complex + rP_amount, corn, soybean]
        
        return environment, energy, economics, food, spider_output
    
    def run_ITEEM_opt(self, sg_price=0.05, wtp_price=0.95, cost_SA_EBT=1.0, cost_SA_BMP=1.0):
        '''
        return: net EAC ($/yr); nitrate ($/yr), TP loading ($/yr)
        '''
        # water quality and quantity
        streamflow = self.get_streamflow_outlet()
        # low_flow = streamflow[:,7:10].mean()  # average monthly flow of Aug, Sept, Oct
        streamflow_outlet = streamflow.sum(axis=1).mean()  # annual flow
        sediment_outlet_instream = sediment_instream(33, self.landuse_matrix).sum(axis=1).mean()
        sediment_decautr_instream = sediment_instream(32, self.landuse_matrix).sum(axis=1).mean()
        # energy and cost
        # energy_dwt, energy_grain, energy_wwt, cost_dwt, cost_grain, cost_wwt, cost_crop, cost_total, outlet_nitrate, outlet_tp = self.get_cost_energy()
        profit_GP, cost_GP, revenue_GP, revenue_crop, revenue_total = self.get_system_revenue(sg_price=sg_price,cost_SA_EBT=cost_SA_EBT)   # annualized revenue for crop
        
        '''start: simplified calculation on WWT: no running ML'''
        wet_1 = Grain(plant_type=1, plant_capacity=2.1, tech_GP=self.tech_GP1)
        wet_2 = Grain(plant_type=1, plant_capacity=5.0, tech_GP=self.tech_GP2)
        dry_1 = Grain(plant_type=2, plant_capacity=120, tech_GP=self.tech_GP3)  
        energy_grain = (wet_1.get_energy_use()[-1] + wet_2.get_energy_use()[-1] + dry_1.get_energy_use()[-1])/(10**6)

        if self.tech_GP1 or self.tech_GP2 or self.tech_GP3 ==2:
            p_reduction = 232 # 232 kg/yr P reduction
        else: p_reduction=0
        
        if self.tech_GP1 ==2:
            p_credit1 = wet_1.get_revenue()[-2]
        else: p_credit1 = 0
        
        if self.tech_GP2 ==2:
            p_credit2 = wet_2.get_revenue()[-2]
        else: p_credit2 = 0
        
        if self.tech_GP3 ==2:
            p_credit3 = dry_1.get_revenue()[-2]
        else: p_credit3 = 0
        p_credit = (p_credit1 + p_credit2 + p_credit3)*(1-0.4)   # 40 tax as default
        p_credit_ac = npf.npv(0.07, [p_credit for i in range(16)])/annuity_factor(16, 0.07) # 16 years
        
        if self.tech_wwt == 'AS':
            cost_wwt = 19071338  # annualized cost
            energy_wwt = 51.7  # TJ/yr
        elif self.tech_wwt == 'ASCP':
            cost_wwt = 20159685  # annualized cost
            energy_wwt = 52.2 # TJ/yr
        elif self.tech_wwt == 'EBPR_basic':
            cost_wwt = 20842504  # annualized cost
            energy_wwt = 40.9 # TJ/yr
        elif self.tech_wwt == 'EBPR_acetate':
            cost_wwt = 24096776  # annualized cost
            energy_wwt = 45.0 # TJ/yr
        elif self.tech_wwt == 'EBPR_StR':
            cost_wwt = 22055418  # annualized cost
            energy_wwt = 38.6 # TJ/yr
            
        # dwt = DWT(limit_N=10, landuse_matrix=self.landuse_matrix)
        # cost_dwt = dwt.get_cost() # $/yr, simplified dwt cost
        # energy_dwt = dwt.get_nitrate_energy()[-1].sum(axis=1).mean()/(10**6) # TJ/yr
        cost_dwt = 0; energy_dwt = 0; 
        
        # cost_grain = wet_1.get_cost()[-1] + wet_2.get_cost()[-1] + dry_1.get_cost()[-1] # averaged cost
        cost_crop = Economics(self.landuse_matrix).get_crop_cost_acf(r=0.07, cost_SA_BMP=cost_SA_BMP)[-1]  # annulized cost
        # cost_total = cost_dwt + cost_grain + cost_wwt + cost_crop  # cost_dwt, cost_grain: averaged annual cost; cost_wwt, cost_crop: annualized cost 
        
        outlet_nitrate, outlet_tp = loading_outlet_USRW_opt_v2(self.landuse_matrix, self.tech_wwt)
        N_outlet = outlet_nitrate[:,:,33].sum(axis=1).mean()
        P_outlet = outlet_tp[:,:,33].sum(axis=1).mean() + p_reduction
        # N_decatur = outlet_nitrate[:,:,32].sum(axis=1).mean()
        # P_decatur = outlet_tp[:,:,32].sum(axis=1).mean() + p_reduction
        '''end: simplified calculation'''
        
        nitrate_impro_prt = ((7240 - N_outlet/1000)/7240)/0.45  # baseline nitrate load =7240 Mg/yr
        if nitrate_impro_prt > 0 and nitrate_impro_prt < 1.0:
            wtp_nitrate = nitrate_impro_prt*0.95*100*113700 # $0.95/1% nitrate improvement, 113700 household
        elif nitrate_impro_prt > 1.0:
            wtp_nitrate = 0.95*100*113700
        else:
            wtp_nitrate = 0
        
        tp_impro_prt = ((324 - P_outlet/1000)/324)/0.45  # baseline TP load = 324 Mg/yr
        if  tp_impro_prt > 0 and tp_impro_prt < 1.0:
            wtp_tp = tp_impro_prt*wtp_price*100*113700 # 113700 households
        elif tp_impro_prt > 1.0: 
            wtp_tp = wtp_price*100*113700          # 59600 households (new)
        else:
            wtp_tp = 0
        wtp = 0.5*wtp_nitrate + 0.5*wtp_tp
        wtp_npv = npf.npv(0.07, [wtp]*16)
        wtp_acf = wtp_npv/annuity_factor(16, 0.07)
        
        sediment_credit = (27455*0.7 - sediment_decautr_instream*0.7)*21.2   # $/yr， 21.2 $/ton, 70% trapped
        sediment_credit_ac = npf.npv(0.07, [sediment_credit for i in range(16)])/annuity_factor(16, 0.07) # 16 years
        
        # 27276 is the baseline sediment load; 21.2 $/ton if sediment is avoided
        system_net_benefit = wtp_acf + profit_GP + revenue_crop + sediment_credit_ac - cost_crop - cost_dwt*cost_SA_EBT - cost_wwt*cost_SA_EBT 
        
        ''' P recovery and food production '''
        rP_P_complex = self.get_rP()*0.264       # 26.4% P for wet milling and 31.5 for dry-grind, kg/yr
        
        if self.tech_wwt == 'EBPR_StR':
            rP_struvite = 1283150*0.1262   # 12.62% P in struvite, kg/yr
        else: rP_struvite = 0
        
        corn = self.get_corn()
        soybean = self.get_soybean()
        biomass = self.get_biomass()  # kg/yr
        
        energy_total = energy_grain + energy_dwt + energy_wwt
        rP = rP_P_complex + rP_struvite
        
        N_outlet_scaled = (N_outlet - 4200713)/(7927670 - 4200713)  # kg/yr
        P_outle_scaled = (P_outlet - 182204)/(774310 - 182204)      # kg/yr
        # sediment_scaled = (sediment_outlet_instream - 25747)/(31405 - 25747) # ton/yr

        obj_water_quality = N_outlet_scaled*0.5 + P_outle_scaled*0.5 #+sediment_scaled*0.2          
        corn_scaled = (1708600000-corn)/(1708600000-1273972052)        # min = 1273972052 kg/yr
        soybean_scaled = (510333000-soybean)/(510333000-372719776)     # min = 372719776 kg/yr   
        obj_food = (corn_scaled + soybean_scaled)/2
        obj_eco = (529.8 - system_net_benefit/(10**6))/(529.8 - 474.1) # $ million/yr        
        obj_energy = (energy_total - 22884)/(23219 - 22884)                  # TJ/yr;
        obj_rP = (12880653 - rP)/(12880653 - 0)                        # kg/yr
        
        output = [N_outlet, P_outlet, sediment_outlet_instream, streamflow_outlet, energy_dwt, energy_grain, energy_wwt, energy_total, biomass, 
                  cost_dwt, cost_wwt*cost_SA_EBT, cost_crop, revenue_crop-cost_crop, cost_GP, profit_GP, p_credit_ac, 
                  sediment_credit_ac, wtp_acf, system_net_benefit, corn, soybean, rP]

        return obj_water_quality, obj_food, obj_eco, obj_energy, obj_rP, P_outlet, N_outlet, output
    
# start = time.time()
# landuse_matrix_baseline = np.zeros((45,62))
# landuse_matrix_baseline[:,1] = 1
# landuse_matrix_baseline[:,55] = 0.5
# landuse_matrix_baseline[:,47] = 1
# baseline = ITEEM(landuse_matrix_baseline, tech_wwt='AS', limit_N=10.0, tech_GP1=1, tech_GP2=1, tech_GP3=1)
# output = baseline.run_ITEEM_opt(cost_SA_EBT=1.0, cost_SA_BMP=1.0)
# end = time.time()
# print('Simulation time is: ', end - start)

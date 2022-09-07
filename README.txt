This repository contains the outputs from the integrated technology-environment-economics model (ITEEM) and genetic optimization algorithm used for evaluating optimal portfolios of agricultural practices and engineering technologies's performances food-energy-water systems in Corn Belt Watersheds. Strucutre of codes and data is provided below:
ITEEM - root directory
	|
	|-> ITEEM.py - script used for simulating the integrated technology-environment-economics model.
	|-> model_SWAT - folder containing script and data for running SWAT model using a modified response matrix approach.
	|-> model_WWT - folder containing script and data for runnning wastewater treatment model using artificial neural netowrks.
	|-> model_Grain - folder containing script and data for running simulation results of grain biorefinries 
	|-> model_Economics - folder containing script and data used for estiamting the total system benefits and environmental benefits (willingess to pay)
	|-> model_DWT - folder containing script and data for running estimating energy and cost of drinking water treatment model for treating nitrate and suspended solids.
	|-> Shapefiles_USRW - folder contains shapefile for creating map of testbed watershed
	|-> Optimization - folder contains detailed genetic optimization algorithm scripts and optimization results.

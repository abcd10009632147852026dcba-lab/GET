#!/bin/bash

###No Visualization
##bus
python valid.py --config ./configs/bus_valid.yaml

##busi
#python valid.py --config ./configs/busi_valid.yaml

## ham10000
#python valid.py --config ./configs/ham10000_valid.yaml

## glas
#python valid.py --config ./configs/glas_valid.yaml

## kvasir-instrument
#python valid.py --config ./configs/kvasir-instrument_valid.yaml


### With Visualization
## bus
#python valid.py --config ./configs/bus_valid.yaml --vis separate
#python valid.py --config ./configs/bus_valid.yaml --vis composite

## busi
#python valid.py --config ./configs/busi_valid.yaml --vis separate
#python valid.py --config ./configs/busi_valid.yaml --vis composite

## ham10000
#python valid.py --config ./configs/ham10000_valid.yaml --vis separate
#python valid.py --config ./configs/ham10000_valid.yaml --vis composite

## glas
#python valid.py --config ./configs/glas_valid.yaml --vis separate
#python valid.py --config ./configs/glas_valid.yaml --vis composite

## kvasir-instrument
#python valid.py --config ./configs/kvasir-instrument_valid.yaml --vis separate
#python valid.py --config ./configs/kvasir-instrument_valid.yaml --vis composite


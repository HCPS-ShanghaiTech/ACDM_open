#!/bin/bash

python load_world.py
python main.py -t "short" -s "ttc" -n "acdm" -e "cdm"
python load_world.py
python main.py -t "short" -s "ttc" -n "d2rl" -e "cdm"
python load_world.py
python main.py -t "short" -s "ttc" -n "idm" -e "na_cdm"
python load_world.py
python main.py -t "short" -s "ttc" -n "acdm" -e "idm"
python load_world.py
python main.py -t "short" -s "ttc" -n "d2rl" -e "idm"
python load_world.py
python main.py -t "short" -s "ttc" -n "idm" -e "na_idm"
python load_world.py
python main.py -t "short" -s "action" -n "d2rl" -e "cdm"
python load_world.py
python main.py -t "short" -s "action" -n "acdm" -e "cdm"
python load_world.py
python main.py -t "short" -s "action" -n "idm" -e "na_cdm"
python load_world.py
python main.py -t "short" -s "action" -n "acdm" -e "idm"
python load_world.py
python main.py -t "short" -s "action" -n "d2rl" -e "idm"
python load_world.py
python main.py -t "short" -s "action" -n "idm" -e "na_idm"
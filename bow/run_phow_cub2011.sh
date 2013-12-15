#!/bin/sh
# script to run phow_cub2011.m
nohup matlab -r "phow_cub2011;exit;" 1>phow_cub2011.log 2>phow_cub2011.err

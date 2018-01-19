#!/usr/bin/python
import thspace as ths

def _complex_concat(a, b):
    tmp = []
    for i in a:
        for j in b:
            tmp.append(i+j)
    return tmp

def _add_prefix(a):
    tmp = []
    for idx, val in enumerate(a):
        tmp.append("w_" + val)
        # tmp.append("b_" + val)
    return tmp

# Pruning threshold setting (90 % off)
th = ths.th80
r = 0.05
#th = 0.9

# CNN settings for pruned training
#target_layer = ["conv1","conv2"]
retrain_iterations = 1000

target_layer = ["conv1", "conv2", "fc1", "fc2"]
all_layer = ["w_fc1","w_fc2","w_conv1","w_conv2","b_fc1","b_fc2","b_conv1","b_conv2"]
target_w_layer = _add_prefix(target_layer)
# Output data lists: do not change this

target_dat = _complex_concat(target_w_layer, [".dat"])
target_p_dat = _complex_concat(target_w_layer, ["_p.dat"])
target_tp_dat = _complex_concat(target_w_layer, ["_tp.dat"])

all_dat = _complex_concat(all_layer, [".dat"])
all_p_dat = _complex_concat(all_layer, ["_p.dat"])
all_tp_dat = _complex_concat(all_layer, ["_tp.dat"])

weight_all = target_dat + target_p_dat + target_tp_dat
syn_all = ["in_conv1.syn", "in_conv2.syn", "in_fc1.syn", "in_fc2.syn"]

# Data settings
show_zero = False

# Graph settings
alpha = 0.75
color = "green"
pdf_prefix = ""

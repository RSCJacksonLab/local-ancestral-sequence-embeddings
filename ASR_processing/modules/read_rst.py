from Bio import AlignIO,SeqIO
import numpy as np
import pandas as pd

def read_rst(infile, node_point, seq_length):
    data = open(infile, 'r')
    datas = data.readlines()
    cat_data = []
    for i in range(len(datas)):
        node_data = []
        if 'Prob distribution at node' in datas[i]:
            if node_point > 999:
                if ',' in datas[i][26:30]:
                    continue
                else:
                    if int(datas[i][26:30]) == node_point:
                        node_data.append(datas[i+4:i+4+int(seq_length)])
                        node = datas[i][26:30]
                        node_data = node_data[0]
                        nodes = (node, node_data)
                        cat_data.append(nodes)
            else:
                if int(datas[i][26:29]) == node_point:
                    node_data.append(datas[i+4:i+4+int(seq_length)])
                    node = datas[i][26:29]
                    node_data = node_data[0]
                    nodes = (node, node_data)
                    cat_data.append(nodes)
    return cat_data
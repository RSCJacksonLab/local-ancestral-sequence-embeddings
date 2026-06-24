from Bio import AlignIO,SeqIO
import numpy as np
import pandas as pd
import re

def pd_post_dist(rst):    
    for i in rst:
        data = i[1]
    aa_site_list = []
    aa_post_dict = []
    for site, i in enumerate(data):
        z = []
        for m, n in enumerate(i.split(':')):
            if m == 1:
                z.append(n.strip())
                d = {site+1:z}
                aa_post_dict.append(d)
    for ind, dist in enumerate(aa_post_dict):
        if ind == 0:
            df = pd.DataFrame.from_dict(dist).transpose()
        else:
            df = pd.concat([df, pd.DataFrame.from_dict(dist).transpose()], axis = 0, sort = False) 
    df = df[0].str.split(' ', n = 19, expand = True)
    for col in range(20):    
        test_list = df[col].to_list()
        z = []
        for i in test_list:
            z.append(re.search('\(([^)]+)', i).group(1))
        df[col] = z
    return df
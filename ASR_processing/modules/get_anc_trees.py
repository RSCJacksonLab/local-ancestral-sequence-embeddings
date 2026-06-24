import re

def get_anc_trees(infile):
    trees = []
    with open(infile) as codeML:
        for line in codeML:
            if line.startswith('(') and line.endswith(';\n'):
                trees.append(line.strip())    
    anc_nwk_bl = trees[0]
    anc_nwk_lab = trees[2]
    return(anc_nwk_lab, anc_nwk_bl)
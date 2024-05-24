from copy import deepcopy

def multiCast(coreBase, starId, coreBit, mapping):
    cores = set()
    cores.add(coreBase)
    for i in range(coreBit):
        if (starId >> i) & 1:
            tmpCores = deepcopy(cores)
            star = 1 << i
            for core in tmpCores:
                cores.add(core ^ star)
    if mapping is not None:
        newCores = set()
        for core in cores:
            newCores.add(mapping[core])
        return newCores
    else:
        return cores

def getStar(cores):
    baseCore = -1
    star = 0
    for core in cores:
        if baseCore < 0:
            baseCore = core
        star |= (baseCore ^ core)
    return star

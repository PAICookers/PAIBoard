# from copy import deepcopy



# def getStar(cores):
#     baseCore = -1
#     star = 0
#     for core in cores:
#         if baseCore < 0:
#             baseCore = core
#         star |= (baseCore ^ core)
#     return star
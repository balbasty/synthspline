#import torch

#class Thingy(object):

#    def __init__(self):
#        self.random_number = torch.randint(1, 6, [])



#for i in range(10):
#    thing = Thingy().random_number
 #   print(int(thing))



nb_levels = []

for i in range(5):
    max_level1 = i
    nb_levels += [max_level1] * len([1, 5, 3, 2, 5])

print(nb_levels)
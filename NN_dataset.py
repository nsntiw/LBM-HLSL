import numpy as np
from functools import lru_cache

@lru_cache(maxsize=10, typed=False)
def get_sample_range(px, samples):
   return range(px*samples)

px_x, px_y, samples = 4, 4, 1
dataset = []

#ALL pairs pt0: 4*4, pt1: 4*4 = 256
#Unique permutations: 256 - 4*4 = 240
#Unique combinations: 240/2 = 120
pt_pairs = [(x, y) for x in range(px_x*samples) for y in range(px_y*samples)]
pt_pairs = [(pt_0, pt_1) for pt_0 in pt_pairs for pt_1 in pt_pairs if pt_0 < pt_1] #Only unique pt pairs containing unique pts. 'pt_0 < pt1' good

for pts in pt_pairs:
   #int((abs(pts[1][0] - pts[0][0])**2 + abs(pts[1][1] - pts[0][1])**2)**0.5)
   num_steps = max(abs(pts[1][0] - pts[0][0]), abs(pts[1][1] - pts[0][1])) + 1

   #line = np.arange(pts[0], pts[1], step=num_steps, dtype=int)
   line = np.linspace(pts[0], pts[1], num=num_steps, dtype=int)

   bit_mx = [[0 for _ in range(px_x*samples)] for _ in range(px_y*samples)]
   for x, y in line:
      #print(x,y)
      bit_mx[x][y]=1
   dataset.append('')

   #print(pt_pair[0], pt_pair[1], num_steps)
   #print(line)
   print(*bit_mx, sep='\n')
   print('---------------------------------------------------')
   if pts[1][1] > 10 and pts[1][0]>10:
      break

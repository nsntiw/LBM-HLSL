import numpy as np
#import scipy as sp
from functools import lru_cache

@lru_cache(maxsize=10, typed=False)
def get_sample_range(px, samples):
   return range(px*samples)

def generate_bit_mx(pt0, pt1): #Assumes pt1>pt2
   #num_steps = max(abs(pt0[0] - pt0[0]), abs(pt1[1] - pt0[1])) + 1
   #num = abs(pt0[0] - pt1[0]) + abs(pt0[1] - pt1[1])
   x_axis_px = pt1[0] - pt0[0]
   y_axis_px = pt1[1] - pt0[1]
   '''
   if x_axis_px > y_axis_px:
      y = np.interp(range(min(pt0[0], pt1[0]), max(pt0[0], pt1[0]), 1),[pt0[0], pt1[0]], [pt0[1], pt1[1]])
      if y_axis_px == 0:
         x = np.interp(range(pt0[0], pt1[0], 1),[pt0[0], pt1[0]], [pt0[0], pt1[0]])
      else:
         x = np.interp(y, [pt0[1], pt1[1]], [pt0[0], pt1[0]])
   elif x_axis_px < y_axis_px:
      x = np.interp(range(min(pt0[1], pt1[1]), max(pt0[1], pt1[1]), 1),[pt0[1], pt1[1]], [pt0[0], pt1[0]])
      if x_axis_px == 0:
         y = np.interp(range(pt0[1], pt1[1], 1),[pt0[1], pt1[1]], [pt0[1], pt1[1]])
      else:
         y = np.interp(x, [pt0[0], pt1[0]], [pt0[1], pt1[1]])
   else:
      x = np.interp(range(pt0[1], pt1[1], 1),[pt0[0], pt1[0]], [pt0[1], pt1[1]])
      y = np.interp(range(pt0[0], pt1[0], 1),[pt0[1], pt1[1]], [pt0[0], pt1[0]])
   '''

   print(pt0, pt1)
   print(x, y)
   #np.interp([0, 1, 2, 3, 4], pts[0][1], pts[1][1])
   #line = np.linspace(pts[0], pts[1], num=50, dtype=int)
   bit_mx = [[0 for _ in range(px_x*samples)] for _ in range(px_y*samples)]
   for x, y in zip(x, y): #line:
      bit_mx[int(x)][int(y)]=1

   bit_mx[pt0[0]][pt0[1]]=1
   bit_mx[pt1[0]][pt1[1]]=1

   print(*bit_mx, sep='\n')
   print('-------------------------------')
   return bit_mx

px_x, px_y, samples = 4, 4, 3
dataset = []

#ALL permutations: (4*4)*(4*4) = 256, Unique permutations: 256-4*4 = 240, Unique combinations: 240/2 = 120
pt_pairs = [(x, y) for x in range(px_x*samples) for y in range(px_y*samples)]
pt_pairs = [(pt_0, pt_1) for pt_0 in pt_pairs for pt_1 in pt_pairs if pt_0 < pt_1] #Only unique pt pairs containing unique pts. 'pt_0 < pt1' good

supersampled_bit_mx = [generate_bit_mx(*pts) for pts in pt_pairs]


#generate_bit_mx(*((0, 0),(12, 0)))
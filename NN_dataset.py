import numpy as np
#import scipy as sp
from functools import lru_cache
from multiprocessing import Pool
from time import time

@lru_cache(maxsize=10, typed=False)
def get_sample_range(px, samples):
   return range(px*samples)

def interpolate_line(pt0, pt1):
   """
   Return a list of (x, y) coords from pt0 to pt1, stepping one pixel
   (in the major axis) at a time, including both endpoints.
   Uses a single parametric np.interp for x and y.
   """
   x0, y0 = pt0
   x1, y1 = pt1

   # Determine number of samples from the longer span
   dx, dy = abs(x1 - x0), abs(y1 - y0)
   num = int(max(dx, dy)) + 1
   num = max(num, 2)

   # Parametric t from 0→1
   t = np.linspace(0.0, 1.0, num=num)

   # Interpolate both coordinates in one go
   xs = np.interp(t, [0.0, 1.0], [x0, x1])
   ys = np.interp(t, [0.0, 1.0], [y0, y1])

   return list(zip(xs, ys))

def generate_bit_mx(pt_pair): #Assumes pt1>pt2
   pt0, pt1 = pt_pair
   bit_mx = [[0 for _ in range(px_x*samples)] for _ in range(px_y*samples)]
   for x, y in interpolate_line(pt0, pt1):
      bit_mx[round(y)][round(x)] = 1
   print(*bit_mx, sep='\n')
   return bit_mx

px_x, px_y, samples = 4, 4, 4
dataset = []

#ALL permutations: (4*4)*(4*4) = 256, Unique permutations: 256-4*4 = 240, Unique combinations: 240/2 = 120
pt_pairs = [(x, y) for x in range(px_x*samples) for y in range(px_y*samples)]
pt_pairs = [(pt_0, pt_1) for pt_0 in pt_pairs for pt_1 in pt_pairs if pt_0 < pt_1] #Only unique pt pairs containing unique pts. 'pt_0 < pt1' good

supersampled_bit_mx = [generate_bit_mx(pts) for pts in pt_pairs]


#if __name__ == '__main__':
#    with Pool(5) as p:
#        supersampled_bit_mx = p.map(generate_bit_mx, pt_pairs)



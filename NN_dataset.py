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

def generate_bit_mx(pt_pair):
   pt0, pt1 = pt_pair
   bit_mx = [[0 for _ in range(px_x*samples)] for _ in range(px_y*samples)]
   for x, y in interpolate_line(pt0, pt1):
      bit_mx[round(y)][round(x)] = 1
   #print(*bit_mx, sep='\n')
   return bit_mx

def downsample(bit_mx, px_x, px_y, samples):
   downsampled_bit_mx = [[0 for _ in range(px_x)] for _ in range(px_y)]
   for y in range(px_y):
      sy0 = y * samples
      sy1 = sy0 + samples

      for x in range(px_x):
         sx0 = x * samples
         sx1 = sx0 + samples

         # Sum directly over the known block range using slicing
         count = sum(
               sum(bit_mx[sy][sx0:sx1])  # sum the slice directly
               for sy in range(sy0, sy1)
         )

         # Set output pixel if majority of samples are 1s
         if count >= (samples * samples) // 2 + 1:
               downsampled_bit_mx[y][x] = 1
         if count > 1:
            downsampled_bit_mx[y][x] = 1
   #print(*downsampled_bit_mx, sep='\n')
   #print('----------------------')
   return downsampled_bit_mx

def downsample_with_jitter(bit_mx, px_x, px_y, samples):
   """
   Downsamples a supersampled bit matrix using jittered sampling.

   Parameters:
      bit_mx: 2D list (supersampled), size (px_y*samples) x (px_x*samples)
      px_x, px_y: target dimensions
      samples: supersampling factor
   """
   H = px_y * samples
   W = px_x * samples

   # Define 9 jitter offsets: center and 8 neighbors
   #jitter_offsets = [(0, 0), (-1, 0), (-1, 1), (0, 1),
   #                  (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
   jitter_offsets = [(0, 0), (-1, 0), (0, 1), (1, 0), (0, -1)]

   downsampled = [[0 for _ in range(px_x)] for _ in range(px_y)]

   for y in range(px_y):
      for x in range(px_x):
         count = 0
         for dy_j, dx_j in jitter_offsets:
               for sy in range(samples):
                  sy_global = y * samples + sy + dy_j
                  if not (0 <= sy_global < H):
                     continue
                  for sx in range(samples):
                     sx_global = x * samples + sx + dx_j
                     if not (0 <= sx_global < W):
                           continue
                     count += bit_mx[sy_global][sx_global]

         total_samples = samples * samples * len(jitter_offsets)
         if count >= total_samples // 4 + 1:
               downsampled[y][x] = 1
         if count >= 4:
               downsampled[y][x] = 1

   return downsampled

def remove_deduplicate(matrices):
   """
   Remove duplicate 2D bit matrices from a list.

   Args:
      matrices: list of 2D lists (each a binary matrix)

   Returns:
      List of unique 2D matrices (still as list-of-lists)
   """
   seen = set()
   unique = []

   for mat in matrices:
      hashable = tuple(tuple(row) for row in mat)
      if hashable not in seen:
         seen.add(hashable)
         unique.append(mat)  # keep original list-of-lists

   return unique

import math
from itertools import combinations

def label_bit_mx(bit_mx):
   # 1) collect points
   pts = [(r, c)
         for r, row in enumerate(bit_mx)
         for c, v in enumerate(row) if v == 1]
   if len(pts) < 2:
      # fewer than 2 pixels → no well-defined slope
      return 0

   # 2) find the pair with max euclidean distance
   def dist2(p, q):
      return (p[0] - q[0])**2 + (p[1] - q[1])**2

   p, q = max(combinations(pts, 2), key=lambda pair: dist2(*pair))

   # 3) orient so start has smaller column
   if p[1] <= q[1]:
      start, end = p, q
   else:
      start, end = q, p

   dr = end[0] - start[0]
   dc = end[1] - start[1]

   # 4) check for vertical/horizontal
   if dc == 0 or dr == 0:
      return 0
   # upslope if dr*dc < 0, downslope if dr*dc > 0
   return 1 if dr*dc < 0 else -1


px_x, px_y, samples = 4, 4, 4

pt_pairs = [(x, y) for x in range(px_x*samples) for y in range(px_y*samples)] #Good
pt_pairs = [(pt_0, pt_1) for pt_0 in pt_pairs for pt_1 in pt_pairs if pt_0 <= pt_1] #'pt_0 < pt1': Unique combinations with unique pts, 'pt_0<=pt_1' Unique combinations with duplicate pts also
supersampled_bit_mx = [generate_bit_mx(pts) for pts in pt_pairs]
downsampled_bit_mx = [downsample(i, px_x, px_y, samples) for i in supersampled_bit_mx]
dataset = remove_deduplicate(downsampled_bit_mx)
labels = [label_bit_mx(i) for i in dataset]

for mx, label in zip(dataset, labels):
   print(*mx, sep='\n')
   print(label)
   print("--------------------------")




#if __name__ == '__main__':
#    with Pool(5) as p:
#        supersampled_bit_mx = p.map(generate_bit_mx, pt_pairs)



import math
x, mu = 0, 0.1
sinX, cosX = 0, 1
sinMu, cosMu = math.sin(mu), math.cos(mu)

estimated_sinX = []
true_sinX = []
for i in range(int(math.pi/0.1)):
   new_sinX = sinX*cosMu + cosX*sinMu
   new_cosX = cosX*cosMu- sinX*sinMu
   estimated_sinX.append(sinX)
   true_sinX.append(math.sin(i*0.1))
   sinX, cosX = new_sinX, new_cosX


print(estimated_sinX, true_sinX, sep='\n')

mse = sum([(j-i)**2 for i, j in zip(estimated_sinX, true_sinX)])/len(estimated_sinX)
print(mse)

print('=================')
#Quantized 4 bit version (0-15)
x, mu = 0, 0.1*15
sinX, cosX = 0, 1*15
sinMu, cosMu = int(math.sin(0.1)*15), int(math.cos(0.1)*15)
estimated_sinX.clear()


for i in range(int(math.pi/0.1)):
   new_sinX = sinX*cosMu//15 + cosX*sinMu//15
   new_cosX = cosX*cosMu//15 - sinX*sinMu//15
   sinX = max(0, min(15, new_sinX))
   cosX = max(0, min(15, new_cosX))
   estimated_sinX.append(sinX)
   print(sinX)

print(list(map(lambda x: x, estimated_sinX)), true_sinX, sep='\n')

mse = sum([(j-i/15)**2 for i, j in zip(estimated_sinX, true_sinX)])/len(estimated_sinX)
print(mse)
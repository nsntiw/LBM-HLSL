class bool_op():
   @staticmethod
   def truncate_to_binary(a):
      a = max(min(1, a), 0)
      return a

   @staticmethod
   def add(a1, a2):
      return 1 if a1+a2 == 1 else 0

   @staticmethod
   def subtract(a1, a2):
      return max(a1-a2,0)

class test():
   def __init__(self, operation, num_bits):
      self.operation = operation
      self.num_bits = num_bits
      self.sequence = []
   
   def generate(self):
      return


opA, opB = [1]*4, [1]*4

print(opA, opB)
print(bool_op.subtract(1,1))
print(bool_op.subtract(0,1))
print(bool_op.subtract(1,0))
print(bool_op.subtract(0,0))
a = 0
print(a)
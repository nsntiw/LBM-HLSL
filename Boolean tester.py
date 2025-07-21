class bool_op():
   @staticmethod
   def truncate_to_binary(a):
      a = max(min(1, a), 0)
      return a

   @staticmethod
   def ADD(a1, a2):
      return 1 if a1+a2 == 1 else 0

   @staticmethod
   def SUBTRACT(a1, a2):
      return max(a1-a2,0)

   @staticmethod
   def AND(a1, a2):
      return 1 if a1==a2==1 else 0

   @staticmethod
   def OR(a1, a2):
      return bool_op.truncate_to_binary(a1+a2)

   @staticmethod
   def NOT(a):
      return bool_op.truncate_to_binary(abs(1-a))

class test():
   def __init__(self, operation, num_bits):
      self.operation = operation
      self.num_bits = num_bits
      self.sequence = []
   
   def generate(self):
      return


opA, opB = [1]*4, [1]*4
print(opA, opB)

print(bool_op.OR(0,0))
print(bool_op.OR(0,1))
print(bool_op.OR(1,0))
print(bool_op.OR(1,1))

print(bool_op.NOT(0))
print(bool_op.NOT(1))
class bool_op():
   @staticmethod
   def truncate_to_binary(a): #OK
      a = max(min(1, a), 0)
      return a

   @staticmethod
   def ADD(a1, a2): #OK
      return 1 if a1+a2 == 1 else 0

   @staticmethod #OK
   def SUBTRACT(a1, a2):
      return max(a1-a2,0)

   @staticmethod
   def AND(a1, a2): #OK
      return 1 if a1==a2==1 else 0

   @staticmethod
   def OR(a1, a2): #OK
      return bool_op.truncate_to_binary(a1+a2)

   @staticmethod
   def NOT(a): #OK
      return bool_op.truncate_to_binary(abs(1-a))

class TESTER():
   def __init__(self, operation, num_bits):
      self.operation = operation
      self.num_bits = num_bits
      self.sequence = []
   
   def generate(self):
      return


opA, opB = [1]*4, [1]*4
print(opA, opB)

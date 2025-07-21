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

   @staticmethod
   def XOR(a1, a2): #OK
      return 1 if a1+a2==1 else 0

class TESTER():
   def __init__(self, operation, bit_width, bit_split=[]):
      self.operation = operation
      self.bit_width = bit_width
      self.sequence = self.genbit(bit_width)
      self.bit_split = bit_split
      return
   
   @staticmethod
   def genbit(n): #OK
      def helper(n, bs=''): #https://stackoverflow.com/questions/64890117/what-is-the-best-way-to-generate-all-binary-strings-of-the-given-length-in-pytho
         if len(bs) == n:
            bs_lst.append(bs)
         else:
            helper(n, bs + '0')
            helper(n, bs + '1')
   
      bs_lst = []
      helper(n)
      return bs_lst


a = TESTER('a', 3)
print(a.sequence)
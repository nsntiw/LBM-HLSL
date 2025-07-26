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
   '''
   Sample usage:
   temp_tester = TESTER(2, [1, 1], bool_op.XOR)
   temp_tester.generate_truth_table()
   '''
   def __init__(self, bit_width=0, bit_split=[], operation=bool_op.ADD):
      self.operation = operation
      self.bit_width = bit_width
      self.bitstr_lst = self.genbit(bit_width)
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

   def generate_truth_table(self):
      def get_inputs(bitstr, bit_split):
         result = []
         start = 0
         for i in bit_split:
            substring = bitstr[start:start+i]
            result.append(substring)
            start += i
         return result
      
      truth_table = []
      for bitstr in self.bitstr_lst:
         input_lst = get_inputs(bitstr, self.bit_split)
         output = self.operation(*tuple(list(map(int, input_lst))))
         truth_table.append([input_lst, output])

      print(*tuple(truth_table), sep='\n')
      return truth_table


carry_flag = 0

R3 = i = 1
R0 = a = 0, R1 = b = 0

while a < 16 and b < 16:
   b = b + a #Perform OP
   b = b - a #Reverse OP
   carry = 1 if a == 1111 else 0
   a = a + i #Increment R1
   b = b + carry



a = input("A = ")
b = input("B = ")
c = input("C = ")

total_sum = None
if int(a) == 13:
    total_sum = 0
elif int(b) == 13:
    total_sum = int(a)
elif int(c) == 13:
    total_sum = int(a) + int(b)
else:
    total_sum = int(a) + int(b) + int(c)

print("Sum is " + str(total_sum) + ".")

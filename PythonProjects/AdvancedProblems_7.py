a = abs(int(input("A = ")))
b = abs(int(input("B = ")))
c = abs(int(input("C = ")))

is_close_to_a = None

b_to_a = abs(a-b)
c_to_a = abs(a-c)
b_to_c = abs(b-c)


if b_to_a <= 1 and c_to_a >= 2 and b_to_c >= 2 or c_to_a <=1 and b_to_a >= 2 and b_to_c >= 2:
    is_close_to_a = True
else:
    is_close_to_a = False

print(is_close_to_a)

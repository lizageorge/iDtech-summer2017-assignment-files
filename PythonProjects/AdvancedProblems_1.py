ind_0 = input("Enter the 1st value in the array: ")
ind_1 = input("Enter the 2nd value in the array: ")
ind_2 = input("Enter the 3rd value in the array: ")
ind_3 = input("Enter the 4th value in the array: ")
ind_4 = input("Enter the 5th value in the array: ")
ind_5 = input("Enter the 6th value in the array: ")
ind_6 = input("Enter the 7th value in the array: ")
ind_7 = input("Enter the 8th value in the array: ")
ind_8 = input("Enter the 9th value in the array: ")
ind_9 = input("Enter the 10th value in the array: ")
target = input("Enter the target sum: ")

list_of_values = [int(ind_0), int(ind_1), int(ind_2), int(ind_3), int(ind_4), int(ind_5), int(ind_6), int(ind_7), int(ind_8), int(ind_9)]

print(list_of_values)

value1 = None
value2 = None

for i in range(0, 9):
    check_value = int(target) - list_of_values[i]
    for j in range(0, 9):
        if int(check_value) == list_of_values[j]:
            value1 = i
            value2 = j
            print(list_of_values[i], i, "+", int(list_of_values[j]), j)
            break
    # break

print("------------------")
print("The final indices are ", str(value1), "and", str(value2))


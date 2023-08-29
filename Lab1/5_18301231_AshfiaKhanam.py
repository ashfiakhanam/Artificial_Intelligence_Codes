#task1
f_input = open("input.txt","r")
f_output = open("output1.txt","w")
arr = []
printed = []
counted = []

for i in f_input:
    arr.append(i.split())

def finder(arr, row, col, count):
    affect = []
    count += 1
    printed.append([row, col])

    if col - 1 >= 0:
        if arr[row][col] == arr[row][col - 1]:
            affect.append([row, col - 1])
        
    if col + 1 < len(arr[0]):
        if arr[row][col] == arr[row][col + 1]:
            affect.append([row, col + 1])

    if row + 1 < len(arr):
        if col - 1 >= 0:
            if arr[row][col] == arr[row + 1][col - 1]:
                affect.append([row + 1, col - 1])
        if arr[row][col] == arr[row + 1][col]:
            affect.append([row + 1, col])
        if col + 1 < len(arr[0]):
            if arr[row][col] == arr[row + 1][col + 1]:
                affect.append([row + 1, col + 1])
        
    if row - 1 >= 0 and col - 1 >= 0:
        if arr[row][col] == arr[row - 1][col - 1]:
            affect.append([row - 1, col - 1])
        if arr[row][col] == arr[row - 1][col]:
            affect.append([row - 1, col])
        if col + 1 < len(arr[0]):
            if arr[row][col] == arr[row - 1][col + 1]:
                affect.append([row - 1, col + 1])

    
    for i, j in affect:
        if [i, j] not in printed:
            count = finder(arr, i, j, count)
    return count


for row in range(len(arr)):
    for col in range(len(arr[0])):
        count = 0
        if arr[row][col] == "Y":
            if [row, col] not in printed:
                counted.append(finder(arr, row, col, count))

f_output.write(str(max(counted)))
f_input.close()
f_output.close()


#task 2
from operator import itemgetter
f_input = open("Question2 input2.txt","r")
f_output = open("output2.txt","w")

row = int(f_input.readline())
col = int(f_input.readline())
arr = []
for i in range(row):
    arr.append(f_input.readline().split())

printed = []
a = [[idx, row.index("A"), 0] for idx, row in enumerate(arr) if "A" in row]
h = sum(row.count("H") for row in arr)

def aliens(arr, loc, Q, printed):
    a[loc][2] = Q[0][2] 
    if(Q[0][0]-1 >= 0 and arr[Q[0][0]-1][Q[0][1]] == 'H'):
        Q.append([Q[0][0]-1,Q[0][1], Q[0][2]+1])
        arr[Q[0][0]-1][Q[0][1]] = 'A'
    if(Q[0][1]-1 >= 0 and arr[Q[0][0]][Q[0][1]-1] == 'H'):
        Q.append([Q[0][0],Q[0][1]-1, Q[0][2]+1])
        arr[Q[0][0]][Q[0][1]-1] = 'A'
    if(Q[0][1]+1 < len(arr[0]) and arr[Q[0][0]][Q[0][1]+1] == 'H'):
        Q.append([Q[0][0],Q[0][1]+1, Q[0][2]+1])
        arr[Q[0][0]][Q[0][1]+1] = 'A'
    if(Q[0][0]+1 < len(arr) and arr[Q[0][0]+1][Q[0][1]] == 'H'):
        Q.append([Q[0][0]+1,Q[0][1], Q[0][2]+1])
        arr[Q[0][0]+1][Q[0][1]] = 'A'

    printed.append([Q[0][0],Q[0][1], Q[0][2]])
    Q.pop(0)

    if len(Q) != 0 :
        if Q[0] not in printed:
            aliens(arr, loc, Q, printed)

for loc in range(len(a)):
        Q = []
        Q.append([a[loc][0], a[loc][1], 0])
        aliens(arr, loc, Q, printed)

mt = [] 
for row in a:
    mt.append(row[2])
f_output.write("Time: "+str(max(mt))+" minutes\n")

sur = str(sum(row.count("H") for row in arr))

if sur != str(0):
    f_output.write(sur+" Survived")
else:
    f_output.write("No one Survived")

f_input.close()
f_output.close()

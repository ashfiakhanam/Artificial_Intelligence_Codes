import numpy as np

def Bank(arr, fitness, bins, mutation_threshold):
    gens = 0  
    while gens != len(arr)**2:
        new_bin = []
        fitlist = fitness(bins, arr)
        for i in range(1, len(bins)):
            x = (select(bins, min(fitlist)))
            y = (select(bins, min(fitlist)))
            c = crossover(x, y)

            if round(np.random.uniform(0, 1), 2) < mutation_threshold:
                c = mutate(c)
            new_bin.append(c) 
        
        new_fitlist = fitness(new_bin, arr)

        if 0 not in new_fitlist:
            res = -1
        else:
            res = new_bin[new_fitlist.index(0)]
            if res != "0"*len(arr):
                return res 
        gens += 1
    return -1

def fitness(bins, arr):
    fitlist = []
    for i in bins:
        score = 0
        for j in range(len(i)):
            if arr[j][0] == "d":
                if i[j] == "1":
                    score += int(arr[j][1])
            if arr[j][0] == "l":
                if i[j] == "1":
                    score -= int(arr[j][1])
        fitlist.append(score)
    return fitlist

def select(bins, fitness):
    return np.random.choice(bins)

def crossover(x, y):
    div = int(np.random.randint(1, len(x) - 1))
    return x[:div] + y[div:]

def mutate(child):
   return child.replace('1', 'temp').replace('0', '1').replace('temp', '0')

def binaries(arr):
    binaries = []        
    for i in range(len(arr)**2):
        binstr = ""
        for j in range(len(arr)):
            string = str(np.random.randint(0, 2))
            if string != "0"*len(arr):    
                binstr += string 
        if binstr not in binaries: 
                binaries.append(binstr)                
    return(binaries)

f_input = open("input.txt","r")
f_output = open("output.txt","w")

input_data = f_input.read().split("\n")
n = int(input_data[0])
arr = []
for i in range(1, n + 1):
    arr.append(input_data[i].split(" "))
mutation_threshold = 0.3
res= str(Bank(arr, fitness, binaries(arr), mutation_threshold))
print(res)

f_output.write(res)
f_input.close()
f_output.close()


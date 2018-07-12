import ana

with open("input.txt",'r') as f:
  while True :
    line = f.readline()
    if not line : break
    tmp = line.split()
    directory = tmp[0]
    inputFile = tmp[1]
    process = tmp[2]
    ana.ana(directory, inputFile, process)

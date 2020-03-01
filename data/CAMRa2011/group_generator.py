filesource = open('sorted.txt', 'r') 
sourceLines = filesource.readlines()
gmember = open('groupMember.txt', 'a')
grating = open('groupRating.txt', 'a')
count=0
for i in range (2, 100836):
    arr1 = sourceLines[i-2].split()
    arr2 = sourceLines[i-1].split()
    arr3 = sourceLines[i].split()
    if arr1[1]==arr2[1] and arr2[1]==arr3[1]:
        if arr1[2]==arr2[2] and arr2[2]==arr3[2] and arr1[2]>2:
            count+=1
            gmember.write(str(count)+" "+str(arr1[0])+","+str(arr2[0])+","+str(arr3[0])+"\n")
            grating.write(str(count)+" "+str(arr1[1])+" "+str(arr1[2])+"\n")

gmember.close()
grating.close()


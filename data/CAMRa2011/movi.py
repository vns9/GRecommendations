onehot = open('movies.txt', 'a')
sourceFile = open('mov.txt', 'r')
sourceLines = sourceFile.readlines()
for line in sourceLines:
    a = line.split()
    b = a[1].split('|')
    encode = [0]*18
    for genre in b:
        if genre=="Action":
            encode[0]=1
        if genre=="Adventure":
            encode[1]=1
        if genre=="Animation":
            encode[2]=1
        if genre=="Children":
            encode[3]=1
        if genre=="Comedy":
            encode[4]=1
        if genre=="Crime":
            encode[5]=1
        if genre=="Documentary":
            encode[6]=1
        if genre=="Drama":
            encode[7]=1
        if genre=="Fantasy":
            encode[8]=1
        if genre=="Film-Noir":
            encode[9]=1
        if genre=="Horror":
            encode[10]=1
        if genre=="Musical":
            encode[11]=1
        if genre=="Mystery":
            encode[12]=1
        if genre=="Romance":
            encode[13]=1
        if genre=="Sci-Fi":
            encode[14]=1
        if genre=="Thriller":
            encode[15]=1
        if genre=="War":
            encode[16]=1
        if genre=="Western":
            encode[17]=1
    k=""
    k = ''.join(str(e) for e in encode)
    onehot.write(a[0])
    onehot.write(" ")
    onehot.write(k)
    onehot.write("\n")

        
        
# 1* Action
# 2* Adventure
# 3* Animation
# 4* Children's
# 5* Comedy
# 6* Crime
# 7* Documentary
# 8* Drama
# 9* Fantasy
# 10* Film-Noir
# 11* Horror
# 12* Musical
# 13* Mystery
# 14* Romance
# 15* Sci-Fi
# 16* Thriller
# 17* War
# 18* Western
    
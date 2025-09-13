t = int(input())
for _ in range(t):
    n=int(input())
    a=[int(x) for x in input().split()]
    cuts=0
    even=[x for x in a if x%2==0]
    cuts+=sum(even)
    odd=[x for x in a if x%2==1]
    if len(odd)==0:
        print(0)
        continue
    odd.sort()
    cuts+=odd[0]
    odd.pop(0)
    i=0
    j=len(odd)-1
    status=True
    while i<len(odd):
        if status:
            status=False
            i+=1
        else:
            status=True
            cuts+=odd.pop()

    print(cuts)

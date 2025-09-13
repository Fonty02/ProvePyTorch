t = int(input())
for _ in range(t):
    n=int(input())
    a=[int(x) for x in input().split()]
    odd=[x for x in a if x%2==1]
    if len(odd)==0:
        print(0)
        continue
    # Sacrifice the smallest odd to start mower, cut everything else
    cuts = sum(a) - min(odd)
    print(cuts)

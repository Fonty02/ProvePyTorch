# Test to understand the current algorithm behavior
def analyze_current_algorithm():
    test_cases = [
        [1, 2, 3, 4, 5],  # Expected: sacrifice 1, cut 2+3+4+5=14? or optimize differently?
        [2, 4, 6, 8],     # Expected: 0 (no odd numbers)
        [1, 3, 5, 7, 9, 11], # Expected: sacrifice 1, cut 3+5+7+9+11=35? or optimize?
    ]
    
    for i, a in enumerate(test_cases):
        print(f"Test case {i+1}: {a}")
        
        even = [x for x in a if x%2==0]
        odd = [x for x in a if x%2==1]
        
        print(f"  Even: {even}, sum: {sum(even)}")
        print(f"  Odd: {odd}")
        
        if not odd:
            print(f"  Result: 0 (no odd numbers to start mower)")
            continue
            
        odd.sort()
        print(f"  Sorted odd: {odd}")
        
        # Current algorithm simulation
        cuts = sum(even)  # Add all even numbers
        cuts += odd[0]    # Sacrifice first (smallest) odd to start mower
        print(f"  After starting mower with {odd[0]}: cuts = {cuts}")
        
        remaining_odd = odd[1:]
        print(f"  Remaining odd: {remaining_odd}")
        
        # Current alternating logic simulation
        status = True
        i = 0
        temp_cuts = cuts
        temp_odd = remaining_odd[:]
        while i < len(temp_odd):
            if status:
                print(f"    Skip index {i}")
                status = False
                i += 1
            else:
                print(f"    Cut {temp_odd[-1]} (from end)")
                temp_cuts += temp_odd.pop()
                status = True
        
        print(f"  Final result: {temp_cuts}")
        print()
        
        # What should the optimal strategy be?
        # Strategy 1: Sacrifice smallest odd, cut everything else
        if odd:
            optimal1 = sum(a) - min(odd)
            print(f"  Optimal strategy 1 (sacrifice min odd): {optimal1}")
        
        print("-" * 50)

if __name__ == "__main__":
    analyze_current_algorithm()
# Given an array of integers nums and an integer target, 
# write a function to search target in nums. If target exists, then 
# return its index. Otherwise, return -1.

def func(A,a,tf):
    n=len(A)
    for x in range(n):
        if(A[x]==tf):
            return x
    return -1



A=[1,2,3,4,5]
ans=0
print(func(A,ans,4))

#Given an array of integers nums which is sorted in ascending order, 
#and an integer target, write a function to search target in nums.
#If target exists, then return its index. Otherwise, return -1.	

def func(A,a,tf):
    n=len(A)
    for x in range(n):
        if(A[x]==tf):
            return x
        elif(A[x]>tf):
            print("Element not found")
            break



A=[1,2,3,4,5,7]
ans=0
print(func(A,ans,6))


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

#Given a sorted array of n elements, possibly with duplicates,
#find the number of occurrences of the target element.	

def func(A,a,tf):
    n=len(A)
    c=0
    for x in range(n):
        if(A[x]==tf):
            c=c+1
    return c



A=[1,2,3,4,4,7]
ans=0
print(func(A,ans,4))

#Given a sorted array of n elements, possibly with duplicates,
#find the number of occurrences of the target element.	

def func(A,a):
    n=len(A)
    for x in range(n-1):
        if(A[x]>A[x+1] and A[x-1]<A[x]):
            return x


A=[1,2,3,4,8,7]
ans=0
print(func(A,ans))

# Python Program to search an element
# in a sorted and pivoted array

# Searches an element key in a pivoted
# sorted array arrp[] of size n
def pivotedBinarySearch(arr, n, key):

	pivot = findPivot(arr, 0, n-1)

	# If we didn't find a pivot,
	# then array is not rotated at all
	if pivot == -1:
		return binarySearch(arr, 0, n-1, key)

	# If we found a pivot, then first
	# compare with pivot and then
	# search in two subarrays around pivot
	if arr[pivot] == key:
		return pivot
	if arr[0] <= key:
		return binarySearch(arr, 0, pivot-1, key)
	return binarySearch(arr, pivot + 1, n-1, key)


# Function to get pivot. For array
# 3, 4, 5, 6, 1, 2 it returns 3
# (index of 6)
def findPivot(arr, low, high):

	# base cases
	if high < low:
		return -1
	if high == low:
		return low

	# low + (high - low)/2;
	mid = int((low + high)/2)

	if mid < high and arr[mid] > arr[mid + 1]:
		return mid
	if mid > low and arr[mid] < arr[mid - 1]:
		return (mid-1)
	if arr[low] >= arr[mid]:
		return findPivot(arr, low, mid-1)
	return findPivot(arr, mid + 1, high)

# Standard Binary Search function
def binarySearch(arr, low, high, key):

	if high < low:
		return -1

	# low + (high - low)/2;
	mid = int((low + high)/2)

	if key == arr[mid]:
		return mid
	if key > arr[mid]:
		return binarySearch(arr, (mid + 1), high,
							key)
	return binarySearch(arr, low, (mid - 1), key)


# Driver program to check above functions
# Let us search 3 in below array
if __name__ == '__main__':
	arr1 = [5, 6, 7, 8, 9, 10, 1, 2, 3]
	n = len(arr1)
	key = 3
	print("Index of the element is : ", \
		pivotedBinarySearch(arr1, n, key))


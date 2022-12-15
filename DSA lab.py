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

	
# Given an array arr of positive integers sorted in a strictly increasing order, and an integer k.
# Write a function to return the kth positive integer that is missing from this array.


def findKthPositive(self, arr, k: int) -> int:
		strt = 0
		end = len(arr)-1
		while strt <= end:
			mid = strt+(end-strt)//2
			if arr[mid]-mid-1 < k:
				strt = mid+1
			else:
				end = mid-1
		return strt+k

a = [1, 5, 11, 19]
 
# k-th missing element
# to be found in the array
k = 11
n = len(a)
 
# calling function to
# find missing element
missing = findKthPositive(n,a, k)
 
print(missing)

# Write a program to implement stack using array (Show all the operations like insertion, deletion and display)


stack = []
#insertion 
stack.append('a')
stack.append('b')
stack.append('c')
#display
print('Initial stack')
print(stack)
#deletion
print('\nElements popped from stack:')
print(stack.pop())
print(stack.pop())

print('\nStack after elements are popped:')
print(stack)


#Write a program to convert Infix expression into Postfix expression and also analyze its Complexity.

Operators = set(['+', '-', '*', '/', '(', ')', '^'])  # collection of Operators

Priority = {'+':1, '-':1, '*':2, '/':2, '^':3} # dictionary having priorities of Operators
 
 
def infixToPostfix(expression): 

    stack = [] # initialization of empty stack

    output = '' 

    

    for character in expression:

        if character not in Operators:  # if an operand append in postfix expression

            output+= character

        elif character=='(':  # else Operators push onto stack

            stack.append('(')

        elif character==')':

            while stack and stack[-1]!= '(':

                output+=stack.pop()

            stack.pop()

        else: 

            while stack and stack[-1]!='(' and Priority[character]<=Priority[stack[-1]]:

                output+=stack.pop()

            stack.append(character)

    while stack:

        output+=stack.pop()

    return output


expression = input('Enter infix expression ')

print('infix notation: ',expression)

print('postfix notation: ',infixToPostfix(expression))

# Python program to evaluate value of a postfix expression

# Class to convert the expression
class Evaluate:
	
	# Constructor to initialize the class variables
	def __init__(self, capacity):
		self.top = -1
		self.capacity = capacity
		# This array is used a stack
		self.array = []
	
	# check if the stack is empty
	def isEmpty(self):
		return True if self.top == -1 else False
	
	# Return the value of the top of the stack
	def peek(self):
		return self.array[-1]
	
	# Pop the element from the stack
	def pop(self):
		if not self.isEmpty():
			self.top -= 1
			return self.array.pop()
		else:
			return "$"
	
	# Push the element to the stack
	def push(self, op):
		self.top += 1
		self.array.append(op)


	# The main function that converts given infix expression
	# to postfix expression
	def evaluatePostfix(self, exp):
		
		# Iterate over the expression for conversion
		for i in exp:
			
			# If the scanned character is an operand
			# (number here) push it to the stack
			if i.isdigit():
				self.push(i)

			# If the scanned character is an operator,
			# pop two elements from stack and apply it.
			else:
				val1 = self.pop()
				val2 = self.pop()
				self.push(str(eval(val2 + i + val1)))

		return int(self.pop())
				

			
# Driver program to test above function
exp = "231*+9-"
obj = Evaluate(len(exp))
print ("postfix evaluation: %d"%(obj.evaluatePostfix(exp)))

#Write a program to implement Simple Queue using arrays
# (Show all the operations like insertion, deletion and display)


class Queue:
    def __init__(self,c):
        self.queue=[]
        self.front=self.rear=0
        self.cap=c

    def queueEnqueue(self,data):
         if(self.cap==self.rear):
            print("\Queue is Full")
         else:
            self.queue.append(data)
            self.rear+=1

    def queueDequeue(self):
        if(self.front==self.rear):
            print("\nQueue is empty")

        else:
            x=self.queue.pop(0)
            self.rear-=1
    def queueDisplay(self):
 
        if(self.front == self.rear):
            print("\nQueue is Empty")
        for i in self.queue:
            print(i, "<--", end='')
    def queueFront(self):
 
        if(self.front == self.rear):
            print("\nQueue is Empty")
 
        print("\nFront Element is:",
            self.queue[self.front])

if __name__ == '__main__':
    q = Queue(4)
    q.queueDisplay()
    q.queueEnqueue(20)
    q.queueEnqueue(30)
    q.queueEnqueue(40)
    q.queueEnqueue(50)
    q.queueDisplay()
    q.queueEnqueue(60)
    q.queueDisplay()
 
    q.queueDequeue()
    q.queueDequeue()
    print("\n\nafter two node deletion\n")
    q.queueDisplay()
    q.queueFront()

#Write a program to implement Circular Queue using arrays
# (Show all the operations like insertion, deletion and display)	


class CircularQueue:
    def __init__(self,c):
        self.size=c
        self.queue = [None for i in range(c)]
        self.front=self.rear=-1
        
    def enqueue(self,data):
         if ((self.rear + 1) % self.size == self.front):
            print("\Queue is Full")
         elif (self.front == -1):
            self.front = 0
            self.rear = 0
            self.queue[self.rear] = data
         else:
            self.rear = (self.rear + 1) % self.size
            self.queue[self.rear] = data

    def dequeue(self):
        if (self.front == -1):
            print("\nQueue is empty")
        elif (self.front == self.rear):
            temp=self.queue[self.front]
            self.front = -1
            self.rear = -1
            return temp
        else:
            temp = self.queue[self.front]
            self.front = (self.front + 1) % self.size
            return temp
    def display(self):
        if(self.front == -1):
            print ("Queue is Empty")
        elif (self.rear >= self.front):
            print("Elements in the circular queue are:",
                                              end = " ")
            for i in range(self.front, self.rear + 1):
                print(self.queue[i], end = " ")
            print ()
        else:
            print ("Elements in Circular Queue are:",
                                           end = " ")
            for i in range(self.front, self.size):
                print(self.queue[i], end = " ")
            for i in range(0, self.rear + 1):
                print(self.queue[i], end = " ")
            print ()
        if ((self.rear + 1) % self.size == self.front):
            print("Queue is Full")

if __name__ == '__main__':
    ob = CircularQueue(5)
    ob.enqueue(14)
    ob.enqueue(22)
    ob.enqueue(13)
    ob.enqueue(-6)
    ob.display()
    print ("Deleted value = ", ob.dequeue())
    print ("Deleted value = ", ob.dequeue())
    ob.display()
    ob.enqueue(9)
    ob.enqueue(20)
    ob.enqueue(5)
    ob.display()

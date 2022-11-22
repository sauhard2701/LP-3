# Python implementation of Quick Sort Using Deterministic Method

# Function to find the partition position
def partition(arr,start,stop):
	pivot = start # pivot
	
	# a variable to memorize where the
	i = start + 1
	
	# partition in the array starts from.
	for j in range(start + 1, stop + 1):
		if arr[j] <= arr[pivot]:
			arr[i] , arr[j] = arr[j] , arr[i]
			i = i + 1
	arr[pivot] , arr[i - 1] = arr[i - 1] , arr[pivot]
	pivot = i - 1
	return (pivot)

# function to perform quicksort
def quickSort(array, start, stop):
	if start < stop:
		pi = partition(array, start, stop)
		# Recursive call on the left of pivot
		quickSort(array, start, pi - 1)
		# Recursive call on the right of pivot
		quickSort(array, pi + 1, stop)

# Driver Code
if __name__ == "__main__":
	array = [10, 7, 8, 9, 1, 5]
	quickSort(array, 0, len(array) - 1)
	print(array)


# Python implementation of Quick Sort using Randomization Method

import random

# Function to find random pivot
def partitionrand(arr , start, stop):
	randpi = random.randrange(start, stop)

    # swapping starting element with random pivot
	arr[start], arr[randpi] = arr[randpi], arr[start]
	return partition(arr, start, stop)

# Function to find the partition position
def partition(arr,start,stop):
	pivot = start # pivot
	
	# a variable to memorize where the
	i = start + 1
	
	# partition in the array starts from.
	for j in range(start + 1, stop + 1):
		if arr[j] <= arr[pivot]:
			arr[i] , arr[j] = arr[j] , arr[i]
			i = i + 1
	arr[pivot] , arr[i - 1] = arr[i - 1] , arr[pivot]
	pivot = i - 1
	return (pivot)

def quickSort(arr, start , stop):
	if(start < stop):		
		pi = partitionrand(array, start, stop)
		# Recursive call on the left of pivot
		quickSort(array, start, pi - 1)
		# Recursive call on the right of pivot
		quickSort(array, pi + 1, stop)

# Driver Code
if __name__ == "__main__":
	array = [10, 7, 8, 9, 1, 5]
	quickSort(array, 0, len(array) - 1)
	print(array)

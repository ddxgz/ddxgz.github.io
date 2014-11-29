##Quick Sort
---
Quick sort is:

* a good choice in many cases. 
* O(nlogn), usually faster than other O(nlogn) sorting algorithm, but it will be O(n^2) in worst case.


The thought of quick sort is divide & conquer:

1. pick a pivot number in the array as "pivot", usually pick the last element.
2. rerange all the numbers, all the numbers smaller than the "pivot" will be put before the "pivot", and the numbers bigger than the "pivot" will be put after the "pivot".
3. recursively rerange all the numbers

The picture below is from "Introduction to Algorithm", it shows the process of quick sort.

![image](img/qsort.png)

Python code for memo:

```
# py 2.7

def partition(array, p, r):
    x = array[r]
    i = p-1
    for j in range(p, r):
        if array[j] <= x:
            i += 1
            tmp = array[i]
            array[i] = array[j]
            array[j] = tmp
    tmp = array[i+1]
    array[i+1] = array[r]
    array[r] = tmp
    return i+1


def qsort(array, p, r):
    if p<r:
        q = partition(array, p, r)
        qsort(array, p, q-1)
        qsort(array, q+1, r)

```
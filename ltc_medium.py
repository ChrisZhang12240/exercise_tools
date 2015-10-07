#Two Sum
def twoSum(nums, target):
    mp = {}
    ret = []
    for i in xrange(0, len(nums)):
        n = nums[i]
        if mp.has_key(n):
            return [mp[n] + 1, i+1]
        mp[target - n] = i

    return ret

#Add Two Numbers 
import linked_list
def addTwoNumbers(l1, l2):
    dummy = linked_list.Node(-1)
    pt = dummy
    extra = 0
    while l1 or l2 or extra:
        val = (l1.val if l1 else 0) + (l2.val if l2 else 0) + extra
        pt.next = linked_list.Node(val%10)
        pt = pt.next
        extra = val / 10
        l1 = l1.next if l1 else l1
        l2 = l2.next if l2 else l2
    return dummy.next

#Longest Substring Without Repeating Characters 
def lengthOfLongestSubstring(s):
    mp = {}
    pt = 0
    if len(s) == 0:
        return 0
        
    ret = 0
    for i in xrange(0, len(s)):
        c = s[i]
        if mp.has_key(c) and pt <= mp[c]:
            pt = mp[c] + 1
        
        mp[c] = i
        ret = max(ret, i - pt + 1)

    return ret
	
	
#Longest Palindromic Substring
def longestPalindrome(s):
	def check(s, a, b):
		while a < b:
			if s[a] != s[b]:
				return False
			a += 1
			b -= 1
		return True
		
	i = l = 0
	for j in xrange(0, len(s)):
		if check(s, j-l, j):
			i = j - l
			l += 1
		elif j - l - 1>=0 and check(s, j-l-1, j):
			i = j - l - 1
			l += 2
			
	return s[i:i + l]

#Container With Most Water 
def maxArea(height):
    area, left, right = 0, 0, len(height)-1
    while left < right:
        h = min(height[left], height[right])
        area = max(area, h * (right - left))
        while left < right and height[left] <= h:
            left += 1
        while left < right and height[right] <= h:
            right -= 1
    return area

#3Sum
def threeSum(nums):
    collection = []
    nums.sort()
    left, right = 0, len(nums) - 1
    while left < right:
        a, i, j = nums[left], left + 1, right
        while i < j:
            b, c = nums[i], nums[j]
            sumVal = a + b + c
            if sumVal == 0:
                collection.append([a, b, c])
            if sumVal <= 0:
                while i < j and nums[i] == b:
                    i += 1
            if sumVal >= 0:
                while i < j and nums[j] == c:
                    j -= 1
        while left < right and nums[left] == a:
            left += 1

    return collection

#3Sum Closest
def threeSumClosest(nums, target):
    nums.sort()
    
    ret = 0
    tempRet = None
    left, right = 0, len(nums)-1
    while left < right-1:
        a = nums[left]
        i, j = left + 1, right
        while i < j:
            b, c = nums[i], nums[j]
            sumVal = a + b + c
            if sumVal == target:
                return target
            elif sumVal < target:
                while i < j and nums[i] == b:
                    i += 1
            else:
                while i < j and nums[j] == c:
                    j -= 1
                    
            if tempRet == None or tempRet > abs(sumVal-target):
                tempRet = abs(sumVal-target)
                ret = sumVal
                
        while left < right and nums[left] == a:
            left += 1
            
    return ret

#Letter Combinations of a Phone Number
def letterCombinations(digits, dict):
    if len(digits) == 0:
        return []
    ret = [""]
    for n in digits:
        temp = []
        s = dict[int(n)]
        if not len(s):
            continue
        for c in s:
            for item in ret:
                temp.append( item+c )
        ret = temp
        
    return ret

#4sum
def fourSum(nums, target):
    collection = []
    nums.sort()
    pairs =[(x, y) for x in xrange(0, len(nums)-1) for y in xrange(x+1, len(nums))]
    mp = {}
    for pair in pairs:
        val = sum([nums[x] for x in pair])
        mp[val] = [pair] if not mp.has_key(val) else mp[val] + [pair]

    for val in mp.keys():
        val2 = target - val
        if val2 < val or not mp.has_key(val2):
            continue
        collection += [p1 + p2 for p1 in mp[val] for p2 in mp[val2] if p1[1] < p2[0]]

    ret = set()
    for item in collection:
        ret.add(tuple([nums[x] for x in item]))

    return [list(x) for x in ret]


#Generate Parentheses 
def generateParenthesis(n):
    def collect(collection, temp, leftNum, rightNum):
        if leftNum == 0 and rightNum == 0:
            if len(temp):
                collection.append(temp)
            return
        elif leftNum > rightNum:
            return
            
        if leftNum > 0:
            collect(collection, temp+"(", leftNum-1, rightNum)
        if rightNum > 0:
            collect(collection, temp+")", leftNum, rightNum-1)
    
    collection = []    
    collect(collection, "", n, n)
    
    return collection

#Seeking for a better solution
def swapPairs(head):
    if head == None:
        return None
    temp = head.next
    ret = head
    if temp:
        head.next = swapPairs(head.next.next)
        temp.next = head
        ret = temp
        
    return ret

import linked_list
def swapPairs2(head):
    dummy = linked_list.Node(-1)
    pt = dummy
    pt.next = head
    while pt.next and pt.next.next:
        a = pt.next
        b = pt.next.next
        pt.next, b.next, a.next, pt = b, a, b.next, a
    return dummy.next

#Divide Two Integers
def divide(a, b):
    maxVal = 2 ** 31 - 1
    if b == 0:
        return 0
    flip = -1 if a ^ b < 0 else 1
    a, b = abs(a), abs(b)
    ret = 0
    while a >= b:
        temp = b
        cnt = 1
        while (temp << 1) <= a:
            temp <<= 1
            cnt <<= 1

        if maxVal - ret < cnt:
            ret = maxVal
            break
        else:
            ret += cnt
        a -= temp

    ret *= flip
    ret = ret - 1 if (flip == -1 and a != 0) else ret
    return ret

#Next Permutation
def nextPermutation(nums):
        def reverse(lst, a, b):
            while a < b:
                lst[a], lst[b] = lst[b], lst[a]
                a += 1
                b -= 1

        i = len(nums) - 1
        while i > 0 and nums[i-1] >= nums[i]:
            i -= 1
        reverse(nums, i, len(nums)-1)
        k = i
        while 0 < k < len(nums) and nums[k] <= nums[i-1]:
            k += 1
        if 0 < k < len(nums):
            nums[i-1], nums[k] = nums[k], nums[i-1]

#Search for a Range 
def searchRange(nums, target):
    def search(lst, lo, hi, target):
        if hi - lo < 1:
            return lo
        mid = lo + (hi - lo) / 2
        if lst[mid] < target:
            return search(lst, mid+1, hi, target)
        else:
            return search(lst, lo, mid, target)

    r = search(nums, 0, len(nums), target)
    if not len(nums) or r >= len(nums) or nums[r] != target:
        return [-1, -1]
        
    return [r, search(nums, 0, len(nums), target+1)-1]

#Search Insert Position
def searchInsert(nums, target):
    def search(lst, lo, hi, target):
        if hi - lo < 1:
            return lo
        mid = lo + (hi - lo)/2
        if lst[mid] < target:
            return search(lst, mid+1, hi, target)
        else:
            return search(lst, lo, mid, target)
            
    return search(nums, 0, len(nums), target)

#Combination Sum 
def combinationSum(candidates, target):
    def collect(lst, collection, temp, target, pos):
        if target == 0:
            collection.append(temp)
            return
        
        for i in xrange(pos, len(lst)):
            val = lst[i]
            if val <= target:
                collect(lst, collection, temp + [val], target-val, i)
            else:
                break
                
    ret = []
    temp = set()
    map(temp.add, candidates)
    candidates = [x for x in temp]
    candidates.sort()
    collect(candidates, ret, [], target, 0)
    
    return ret


#Combination Sum II 
def combinationSum2(candidates, target):
    def collect(lst, ret, temp, target, pos):
        if target == 0:
            if temp:
                ret.append(temp)
            return
        
        for i in xrange(pos, len(lst)):
            val = lst[i]
            if i > pos and lst[i] == lst[i-1]:
                continue
            if val <= target:
                collect(lst, ret, temp+[val], target-val, i+1)
            else:
                break
            
    candidates.sort()
    ret = []
    collect(candidates, ret, [], target, 0)
    return ret
    
#Multiply Strings 
def multiplyString(num1, num2):
    product = [0] * (len(num1) + len(num2))
    pos = len(product)-1
    
    for n1 in reversed(num1):
        tempPos = pos
        for n2 in reversed(num2):
            product[tempPos] += int(n1) * int(n2)
            product[tempPos-1] += product[tempPos]/10
            product[tempPos] %= 10
            tempPos -= 1
        pos -= 1
        
    pt = 0
    while pt < len(product)-1 and product[pt] == 0:
        pt += 1

    return ''.join(map(str, product[pt:]))

#Permutations
def permute(nums):
    if not nums:
        return []
		
    ret = [ [] ]
    for n in nums:
        temp = []
        for item in ret:
            for i in xrange(0, len(item)+1):
                temp.append(item[:i] + [n] + item[i:])
        ret = temp
    
    return ret

#Rotate Image 
def rotateMatrix(matrix):
    n = len(matrix)
    for t in xrange(0, n):
        for i in xrange(t+1, n):
            matrix[t][i], matrix[i][t] = matrix[i][t], matrix[t][i]
    for i in xrange(0, n/2):
        for j in xrange(0, n):
            matrix[j][i], matrix[j][n-i-1] = matrix[j][n-i-1], matrix[j][i]

#Anagrams 
def anagrams(strs):
    mp = {}
    for s in strs:
        t = "".join(sorted([x for x in s]))
        if mp.has_key(t):
            mp[t].append(s)
        else:
            mp[t] = [s]
            
    ret = []
    for lst in mp.values():
        if len(lst) > 1:
            ret += lst
    
    return ret

#Pow
def pow(x, n):
    if n == 0:
        return 1
    if x == 0:
        return x

    if n < 0:
        n = -n
        x = 1.00000/x

    return pow(x*x, n/2) * x if n&1 else pow(x*x, n/2)

def pow2(x, n):
    if n == 0:
        return 1
    if x == 0:
        return x

    if n < 0:
        n = -n
        x = 1.00000/x

    ret = 1
    while n:
        if n&1:
            ret*=x
        x *= x
        n >>= 1

    return ret

#Maximum Subarray 
def maxSubArray(nums):
    if not nums:
        return 0
    sumVal = ret = 0
    for i in nums:
        sumVal = max(0, sumVal) + i
        ret = max(ret, sumVal)
    return max(nums) if ret == 0 else ret

#Spiral Matrix 
def spiralOrder(matrix):
    result = []

    while matrix and matrix[0]:
        if matrix[0]:
            result += matrix.pop(0)

        if matrix and matrix[0]:
            for row in matrix:
                result.append(row.pop())

        if matrix and matrix[-1]:
            result += matrix.pop()[::-1]

        if matrix and matrix[0]:
            for row in matrix[::-1]:
                result.append(row.pop(0))

    return result

#Jump Game
# Tag: Array Greedy
def canJump(nums):
    maxPos = 0
    for i in xrange(len(nums)):
        maxPos = max(maxPos, i + nums[i])
        if maxPos <= i or maxPos >= len(nums)-1:
            break
            
    return len(nums) and maxPos >= len(nums)-1

#Spiral Matrix II
#Tag: Array
def generateMatrix(n):
    A, lo = [], n*n+1
    while lo > 1:
        lo, hi = lo - len(A), lo
        A = [range(lo, hi)] + zip(*A[::-1])
    return [list(x) for x in A]

#Permutation Sequence 
#Tag: Backtracking Math
def getPermutation2(n, k):
    array = range(1, n + 1)
    lst = [1,1]
    for i in xrange(2, n + 1):
        lst += [i * lst[-1]]

    k -= 1
    if k > lst[n]:
        return ""
    permutation = []
    for i in xrange(n - 1, -1, -1):
        idx, k = divmod(k, lst[i])
        permutation.append(array.pop(idx))

    return "".join(map(str, permutation))

def getPermutation(n, k):
    arr = range(1, n + 1)
    lst = [1]
    for i in xrange(1,n+1):
        lst = [lst[0] * i] + lst
        
    k -= 1
    if k > lst[0]:
        return ""
    ret = []
    for i in xrange(0, len(lst)-1):
        idx, k = divmod(k, lst[i+1])
        ret.append(arr.pop(idx))

    return "".join(map(str, ret))


#Rotate List
#Tag: Linked List, Two Pointers
def rotateRight(head, k):
    if not head:
        return head

    temp = head
    size = 1
    while temp and temp.next:
        temp = temp.next
        size += 1
    
    temp.next = head

    k = size - k % size
    th = head
    
    while k > 1:
        k -= 1
        th = th.next

    temp = th.next
    th.next = None

    return temp

#Unique Paths 
#Tag: Array, Dynamic Programming
def uniquePaths(m, n):
    matrix = [ [1 for x in xrange(n)] for y in xrange(m)]
    for i in xrange(1, m):
        for j in xrange(1, n):
            matrix[i][j] = matrix[i-1][j] + matrix[i][j-1]
            
    return matrix[-1][-1]

def uniquePaths2(m, n):
    def c(a, b):
        ret = 1
        for x in xrange(a, a-b, -1):
            ret *= x
        for x in xrange(1, b+1):
            ret /= x
        return ret

    return c(m+n-2, m-1)


#Unique Paths II
#Tag: Array, Dynamic Programming
def uniquePathsWithObstacles(obstacleGrid):
    n = len(obstacleGrid)
    if n == 0:
        return 0
    dp = [0] * len(obstacleGrid[0])
    dp[0] = 1
    for row in obstacleGrid:
        for j in xrange(0, len(row)):
            if row[j] == 1:
                dp[j] = 0
            elif j > 0:
                dp[j] += dp[j-1]
    return dp[-1]


#Minimum Path Sum 
#Tag: Array, Dynamic Programming
def minPathSum(grid):
        if not grid:
            return 0
            
        dp = grid[0]
        for i in xrange(1, len(dp)):
            dp[i] += dp[i-1]

        for j in xrange(1, len(grid)):
            row = grid[j]
            for i in xrange(0, len(row)):
                if i == 0:
                    dp[i] += row[i]
                else:
                    dp[i] = row[i] + min(dp[i-1], dp[i])

        return dp[-1]

#Sqrt(x) 
#Tag: Math, Binary Search
def mySqrt(x):
    if x < 2:
        return x
    left, right = 0, x
    while left < right:
        mid = left + (right - left)/2
        if mid > x/mid:
            right = mid
        elif mid < x/mid:
            left = mid + 1
        else:
            return mid

    return right - 1


def mySqrt2(x):
    ans = x
    while ans != 0 and x / ans < ans:
        ans = (ans + x / ans) / 2
    return ans

#Set Matrix Zeroes
#tag: Array
def setZeroes(matrix):
    mark = False
    for i in xrange(0, len(matrix)):
        if matrix[i][0] == 0:
            mark = True
        for j in xrange(1, len(matrix[i])):
            if matrix[i][j] == 0:
                matrix[i][0] = matrix[0][j] = 0

    for i in xrange(len(matrix)-1, -1, -1):
        for j in xrange(1, len(matrix[i])):
            if matrix[0][j] == 0 or matrix[i][0] == 0:
                matrix[i][j] = 0
        if mark:
            matrix[i][0] = 0




#Search a 2D Matrix 
#Tad: Binary Search, Array
def searchMatrix(matrix, target):
    def search(lst, lo, hi, target):
        if hi - lo < 1:
            return False, lo-1 if 0 < lo <= len(lst) else None 
        mid = lo + (hi - lo)/2
        val = lst[mid] if isinstance(lst[mid], int) else lst[mid][0]
        if val == target:
            return True, mid
        elif val < target:
            return search(lst, mid+1, hi, target)
        return search(lst, lo, mid, target)

    found, idx = search(matrix, 0, len(matrix), target)
    return True if found else idx != None and search(matrix[idx], 0, len(matrix[idx]), target)[0]

#Sort Colors
#Tag: Array, Two Pointers, Sort
def sortColors(nums):
    left, right, pt = 0, len(nums)-1, 0
    while pt <= right:
        if nums[pt] == 0:
            nums[pt], nums[left] = nums[left], nums[pt]
            pt += 1
            left += 1
        elif nums[pt] == 1:
            pt += 1
        else:
            nums[pt], nums[right] = nums[right], nums[pt]
            right -= 1

def sortColors2(nums):
    cnt = [0,0,0]
    for n in nums:
        cnt[n] += 1
    r = 0
    for i in xrange(0, len(cnt)):
        for j in xrange(0, cnt[i]):
            nums[r] = i
            r += 1


#Combinations
#Tag: Backtracking 
def numCombine(n, k):
    def collect(ret, temp, start, n, k):
        if k == 0:
            if temp:
                ret.append(temp)
            return
        for i in xrange(start, n+1):
            collect(ret, temp + [i], i+1, n, k-1)
            
    ret = []    
    collect(ret, [], 1, n, k)
    return ret

def numCombine2(n, k):
    if k == 0:
        return []
    combs = [[]]
    for _ in range(k):
        combs = [[i] + c for c in combs for i in range(1, c[0] if c else n+1)]
    return combs

   
#Subsets
#Tag: Array, Backtracking, Bit Manipulation
def subsets(nums):
    nums.sort()
    ret = []
    for n in nums:
        for i in xrange(len(ret)):
            ret.append(ret[i] + [n])
        ret.append([n])
    ret.append([])
    return ret

#https://leetcode.com/discuss/46668/recursive-iterative-manipulation-solutions-explanations
def subsets2(nums):
    nums.sort()
    num_subset = 1 << len(nums)
    res = [[] for x in xrange(0, num_subset)]
    for i in xrange(0, len(nums)):
        step = 1 << i
        for j in xrange(step, num_subset, step*2):
            for k in xrange(j, j+step):
                    res[k].append(nums[i])
    return res


#Word Search
#Tag:  Array, Backtracking
def wordSearch(board, word):
    def check(board, i, j, word, pos):
        if i < 0 or i >= len(board) or\
            j < 0 or j >= len(board[0]) or\
            board[i][j] != word[pos]:
                return False
        elif pos == len(word) - 1:
            return True
        
        old, board[i][j] = board[i][j], '*'    
        
        checkConnection = check(board, i-1, j, word, pos+1) or \
            check(board, i+1, j, word, pos+1) or \
            check(board, i, j-1, word, pos+1) or \
            check(board, i, j+1, word, pos+1)
            
        board[i][j] = old
        return checkConnection
        
    for i in xrange(0, len(board)):
        for j in xrange(0, len(board[i])):
            if check(board, i, j, word, 0):
                return True
                
    return False

#Remove Duplicates from Sorted Array II 
#tag: Array, Two Pointers
def removeDuplicates2(nums):
    cnt = 2
    for i in xrange(2, len(nums)):
        if nums[i] != nums[cnt-2]:
            nums[cnt] = nums[i]
            cnt += 1
    return min(cnt, len(nums))


#Remove Duplicates from Sorted List II 
def deleteDuplicates(head):
    flag = 0
    while head and head.next and head.val == head.next.val:
        head, flag = head.next, 1
    if flag:
        head = deleteDuplicates(head.next)
    elif head:
        head.next = deleteDuplicates(head.next)
    return head

def deleteDuplicates2(head):
    dummy = pt = linked_list.Node(-1)
    while head:
        temp = head.next
        while temp and temp.val == head.val:
            temp = temp.next
        if temp == head.next:
            pt.next = head
            pt = pt.next
        head.next = temp
        head = head.next
    pt.next = None
    return dummy.next

#Search in Rotated Sorted Array II 
#Tag: Array, Binary Search
def searchInRotatedSortedArray(nums, target):
    lo, hi = 0, len(nums)-1
    while lo < hi:
        mid = lo + (hi - lo)/2
        if nums[mid] == target:
            return True
        if nums[mid] > nums[hi]:
            if nums[lo] <= target < nums[mid]:
                hi = mid
            else:
                lo = mid + 1
        elif nums[mid] < nums[hi]:
            if nums[mid] < target <= nums[hi]:
                lo = mid + 1
            else:
                hi = mid
        else:
            hi -= 1
    return lo < len(nums) and nums[lo] == target

#Partition List
#Tag: Linked List, Two Pointers
def partitionList(head, x):
    lst1 = pt1 = linked_list.Node(-1)
    lst2 = pt2 = linked_list.Node(-1)
    while head:
        if head.val < x:
            pt1.next = head
            pt1 = pt1.next
        else:
            pt2.next = head
            pt2 = pt2.next
        head = head.next
        
    pt2.next = None
    pt1.next = lst2.next
    return lst1.next

#Gray Code
#Tag: Backtracking
def grayCode(n):
    ret = [0]
    for i in xrange(0, n):
        ret += [ (1 << i) + x for x in ret[::-1]]
    return ret

def grayCode2(n):
    ret = []
    for i in xrange(0, 1 << n):
        ret.append(i ^ (i/2))
    return ret
   

#Subsets II 
#Array, Backtracking
def subsetsWithDup(nums):
    nums.sort()
    ret = [[]]
    cnt = size = 0
    for i in xrange(len(nums)):
        n = nums[i]
        cnt = size if i > 0 and nums[i-1] == nums[i] else 0
        size = len(ret)
        ret += [ret[i] + [n] for i in xrange(cnt, len(ret))]
    return ret

#Decode Ways
#Dynamic Programming, String
def numDecodings(s):
    if not s:
        return 0
    dp = [0] * len(s) + [1]
    dp[-2] = 1 if 0 < int(s[-1]) <= 26 else 0
    for i in xrange(len(s)-2, -1, -1):
        if s[i] == '0':
            continue
        if 0 < int(s[i:i+2]) <= 26:
            dp[i] += dp[i+2]
        
        dp[i] += dp[i+1]
        
    return dp[0]

#Reverse Linked List II 
#Tag: Linked list
def reverseBetween(head, m, n):
    dummy = pt = linked_list.Node(-1)
    pt.next = head

    for i in xrange(1, m):
        pt = pt.next

    n -= m
    temp = pt.next
    for i in xrange(n):
        x = temp.next
        temp.next, x.next, pt.next = x.next, pt.next, x

    return dummy.next

#Restore IP Addresses
#Tag: Backtracking, String
def restoreIpAddresses(s):
    def restore(s, ret, temp, cnt):
        if cnt == 4 and len(temp) == 4 and len(s) == 0:
            ret.append(".".join(temp))
            return ret
        for j in xrange(1, 4):
            if j > len(s) or (s[0] == '0' and j > 1) or int(s[:j]) > 255:
                return ret
            restore(s[j:], ret, temp + [s[:j]], cnt + 1)
        return ret
        
    if len(s) < 4 or  len(s) > 12:
        return []
    return restore(s, [], [], 0)

#Binary Tree Inorder Traversal
#Tag: Tree, Hash Table, Stack
def inorderTraversal(root):
    ret = []
    stack = []
    while root:
        stack.append(root)
        root = root.left
        
    while stack:
        node = stack.pop()
        temp = node.right
        while temp:
            stack.append(temp)
            temp = temp.left
        ret.append(node.val)
    return ret

def inorderTraversal2(root):
    ret = []
    while root:
        if root.left:
            temp = root.left
            while temp and temp.right and temp.right != root:
                temp = temp.right
            if temp.right == None:
                temp.right = root
                root = root.left
                continue
            else:
                temp.right = None
        ret.append(root.val)
        root = root.right
    return ret


#Unique Binary Search Trees
#Tree, Dynamic Programming
def numTrees(n):
    dp = [0] * (n + 1)
    dp[0] = dp[1] = 1
    for i in xrange(2, n+1):
        for j in xrange(1, i+1):
            dp[i] += dp[j-1] * dp[i-j]
    
    return dp[n]

#Unique Binary Search Trees II
#Tree, Dynamic Programming
import binary_tree
def generateTrees(n):
    def generateSubtree(s, e):
        if s >= e:
            return [None]
        ret = []
        for i in xrange(s, e):
            left = generateSubtree(s, i)
            right = generateSubtree(i+1, e)
            for l in left:
                for r in right:
                    node = binary_tree.Node(i)
                    node.left = l
                    node.right = r
                    ret.append(node)
        
        return ret
    return generateSubtree(1, n+1)

#Validate Binary Search Tree
# Tree, Depth-first Search
def isValidBST(root):
    prev = None
    while root:
        if root.left:
            temp = root.left
            while temp and temp.right != None and temp.right != root:
                temp = temp.right
            if temp.right == None:
                temp.right = root
                root = root.left
                continue
        if prev != None and prev >= root.val:
            return False
        prev = root.val
        root = root.right
            
    return True

#Binary Tree Zigzag Level Order Traversal
#Tag: Tree, Breadth-first Search, Stack
def zigzagLevelOrder(root):
    if root == None:
        return []

    ret = []
    queue = [root]
    flip = False
    while queue:
        size = len(queue)
        row = [0] * size
        for i in xrange(size):
            node, queue = queue[0], queue[1:]
            idx = i if not flip else size - i - 1
            row[idx] = node.val
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        ret.append(row)
        flip = not flip

    return ret

#Construct Binary Tree from Preorder and Inorder Traversal
#Tag: Tree, Array, Depth-first Search
def buildTreeFromPreIn(preorder, inorder):
    root = binary_tree.Node(preorder.pop(0))
    stack = [root]
    prev = None
    while preorder:
        if stack and stack[-1].val == inorder[0]:
            inorder.pop(0)
            prev = stack.pop()
        elif prev:
            prev.right = binary_tree.Node(preorder.pop(0))
            stack.append(prev.right)
            prev = None
        else:
            newNode = binary_tree.Node(preorder.pop(0))
            if stack:
                stack[-1].left = newNode
            stack.append(newNode)

    return root

#Construct Binary Tree from Inorder and Postorder Traversal
#Tag: Tree, Array, Depth-first Search
def buildTreeFromPostIn(postorder, inorder):
    if not inorder:
        return None

    idx = inorder.index(postorder.pop())
    node = binary_tree.Node(inorder[idx])
    node.right = buildTreeFromPostIn(postorder, inorder[idx+1:])
    node.left = buildTreeFromPostIn(postorder, inorder[:idx])
    return node

#Convert Sorted Array to Binary Search Tree
#Tag:Tree, Depth-first Search
def sortedArrayToBST(nums):
    if not nums:
        return None
        
    mid = (len(nums)-1)/2
    node = binary_tree.Node(nums[mid])
    node.left = sortedArrayToBST(nums[:mid])
    node.right = sortedArrayToBST(nums[mid+1:])
    
    return node

#Convert Sorted List to Binary Search Tree
#Tag: Depth-first Search, Linked List
def sortedListToBST(head):
    def buildTree(wrapper, lo, hi ):
        if hi - lo < 1:
            return None
        mid = lo + (hi - lo - 1)/2
        leftNode = buildTree(wrapper, lo, mid)
        thisNode = binary_tree.Node(wrapper[0].val)
        wrapper[0] = wrapper[0].next
        rightNode = buildTree(wrapper, mid+1, hi)
        thisNode.left = leftNode
        thisNode.right = rightNode
        
        return thisNode
        
    size = 0
    th = head
    while th:
        th = th.next
        size += 1
        
    return buildTree([head], 0, size)

#Flatten Binary Tree to Linked List
#Tag: Tree, Depth-first Search 
def flatten(root):
    th = root
    while th:
        if th.left:
            temp = th.left
            while temp.right:
                temp = temp.right
            temp.right = th.right
            th.right = th.left
            th.left = None
        th = th.right

def flatten2(root):
    def imp(root, prev):
        if(root==None):
            return prev
        prev=imp(root.right,prev)  
        prev=imp(root.left,prev)
        root.right=prev
        root.left=None
        return root
    return imp(root, None)

#Path Sum II
#Tag: Tree, Depth-first Search
def pathSum(root, sum):
    def collect(root, ret, temp, val):
        if root == None:
            return ret
            
        val -= root.val
        if val == 0 and root.left == None and root.right == None:
            ret.append(temp + [root.val])
            return ret
        
        collect(root.left, ret, temp + [root.val], val)
        collect(root.right, ret, temp + [root.val], val)
            
        return ret
        
    return collect(root, [], [], sum)


#Populating Next Right Pointers in Each Node
#Tag: Tree, Depth-first Search
def connectTree(root):
    if not root:
        return
    
    tRoot = root
    dummy = pt = binary_tree.TreeLinkNode(-1)
    while tRoot:
        if tRoot.left:
            pt.next = tRoot.left
            pt = pt.next
        if tRoot.right:
            pt.next = tRoot.right
            pt = pt.next

        tRoot = tRoot.next
        if tRoot == None:
            tRoot = dummy.next
            dummy = pt = binary_tree.TreeLinkNode(-1)

#Triangle
#Array, Dynamic Programming
def minimumTotal(triangle):
    if not triangle:
        return 0
    dp = triangle[-1]
    for i in xrange(len(triangle)-2, -1, -1):
        row = triangle[i]
        for j in xrange(0, len(row)):
            dp[j] = row[j] + min(dp[j], dp[j+1])
            
    return dp[0]

#Best Time to Buy and Sell Stock
#Tag: Array, Dynamic Programming
def maxProfit(prices):
    if len(prices) <= 1:
        return 0
        
    ret = sumVal = 0
    for i in xrange(1, len(prices)):
        sumVal = max(0, sumVal + (prices[i]-prices[i-1]))
        ret = max(ret, sumVal)
    return ret

#Best Time to Buy and Sell Stock II
#Tag: Array, Greedy
def maxProfit2(prices):
    if len(prices) <= 1:
        return 0
        
    maxVal = 0
    for i in xrange(1, len(prices)):
        diff = max(0, prices[i] - prices[i-1])
        maxVal += diff
        
    return maxVal

#Sum Root to Leaf Numbers
#Tag: Tree, Depth-first Search
def sumNumbers(root):
    def calc(root, val):
        if root == None:
            return 0
        val = val * 10 + root.val
        if root.left == None and root.right == None:
            return val
        return calc(root.left, val) + calc(root.right, val)
    return calc(root, 0)

#Word Ladder
def wordLadder(beginWord, endWord, wordList):
    front = set([beginWord])
    back = set([endWord])
    wordList = set(wordList)
    wordList.discard(beginWord)
    length = 2
    while front:
        front = wordList & set(word[:idx] + c + word[idx+1:] for word in front for idx in xrange(len(word)) for c in 'abcdefghijklmnopqrstuvwxyz')
        if front & back:
            return length
        length += 1
        if len(front) > len(back):
            front, back = back, front
        wordList -= front
        
    return 0

#Surrounded Regions
#Tag:Breadth-first Search, Union Find
def surroundedRegons(board):
    if not any(board):
        return
    m, n = len(board), len(board[0])
    queue = [ij for k in range(m+n) for ij in ((0, k), (m-1, k), (k, 0), (k, n-1))]
    while queue:
        x,y = queue.pop()
        if not(0 <= x < m and 0 <= y < n and board[x][y] == 'O'):
            continue
        board[x][y] = 'M'
        queue += [(x+1, y), (x-1, y), (x, y-1), (x, y+1)]
        

    board[:] = [['XO'[c == 'M'] for c in row] for row in board] 

#Palindrome Partitioning
#Tag:Backtracking
def palindromePartition(s):
    def imp(s, mp):
        if not mp.has_key(s):
            ret = [[s[:i]] + item for i in xrange(1, len(s)+1) for item in imp(s[i:], mp) if s[:i] == s[i-1::-1]]
            mp[s] = ret if ret else [[]]
        return mp[s]
    return imp(s, {})

#Clone Graph
#Tag:Depth-first Search, Breadth-first Search, Graph
import graphic
def cloneGraph(node):
    def clone(node, mp):
        if node == None:
            return None
        if not mp.has_key(node.val):
            c = graphic.UndirectedGraphNode(node.val)
            mp[node.val] = c
            for neighbor in node.adj:
                c.adj.append(clone(neighbor, mp))
        return mp[node.val]
    
    return clone(node, {})

#Gas Station
#tag: Greedy
def canCompleteCircuit(gas, cost):
    total = sumVal = start = 0
    for i in xrange(len(gas)):
        total += gas[i] - cost[i]
        sumVal += gas[i] - cost[i]
        if sumVal < 0:
            start = i + 1
            sumVal = 0
            
    return start if total >= 0 else -1

#Single Number
#Tag: Hash Table, Bit Manipulation
def singleNumber(nums):
    ret = 0
    for n in nums:
        ret ^= n
    return ret

#Single Number II
#Tag:Bit Manipulation
def singleNumber2(nums):
    ret = 0
    ep1 = ep2 = ep3 = 0
    for n in nums:
        ep3 = (ep3 ^ n) & ep2 & ep1
        ep2 = (ep2 ^ n) & ep1
        ep1 = (ep1 | n) & ~ep3
    return ep1

#Word Break
#Tag: Dynamic Programming
def wordBreak(s, wordDict):
    dp = [0] * (len(s) + 1)
    dp[0] = 1
    for i in xrange(len(s)):
        for j in xrange(i, -1, -1):
            if s[j:i+1] in wordDict and dp[j] == 1:
                dp[i+1] = 1
                break
            
    return dp[-1] == 1

#Linked List Cycle
#Tag:Linked List, Two Pointers
def hasCycle(head):
    p1, p2 = head, head
    while p2 and p2.next:
        p2 = p2.next.next
        p1 = p1.next
        if p1 == p2:
            return True
        
    return False 
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
        for j in xrange(len(matrix[i])-1, 0, -1):
            if matrix[0][j] == 0 or matrix[i][0] == 0:
                matrix[i][j] = 0
        if mark:
            matrix[i][0] = 0




#Search a 2D Matrix 
#Tad: Binary Search, Array
def searchMatrix(matrix, target):
    if not matrix:
        return False
        
    lo, hi = 0, len(matrix)
    while lo < hi:
        mid = lo + (hi-lo)/2
        if matrix[mid][0] == target:
            return True
        elif matrix[mid][0] < target:
            lo = mid + 1
        else:
            hi = mid
            
    i = max(0, min(lo-1, len(matrix)-1))

    lo, hi = 0, len(matrix[i])
    while lo < hi:
        mid = lo + (hi - lo)/2
        if matrix[i][mid] == target:
            return True
        elif matrix[i][mid] < target:
            lo = mid + 1
        else:
            hi = mid
    
    return False

   


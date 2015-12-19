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
    lo, hi = 0, len(nums)
    while lo < hi:
        mid = lo + (hi - lo) / 2
        if nums[mid] < target:
            lo = mid + 1
        else:
            hi = mid

    return lo
            

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
    if not s:
        return True if not any(wordDict) else False
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

#Linked List Cycle II
#Tag:Linked List, Two Pointers
def detectCycle(head):
    p1 = p2 = head
    while p2:
        p2 = p2.next.next if p2.next else p2.next
        p1 = p1.next
        if p1 == p2 and p2 != None:
            p1 = head
            while p1 != p2:
                p1 = p1.next
                p2 = p2.next
            return p1

    return None

#Reorder List
#Tag: linked list
def reorderList(head):
    def reverse(head):
        temp = linked_list.Node(-1)
        temp.next = head
        while head and head.next:
            x = head.next
            head.next = x.next
            x.next = temp.next
            temp.next = x
        return temp.next
        
        
    if not head:
        return head
        
    th1, th2 = head, head
    while th2 and th2.next and th2.next.next:
        th2 = th2.next.next
        th1 = th1.next
        
    
    th2 = reverse(th1.next)
    th1.next = None
    th1 = head
    while th2:
        temp = th2.next
        th2.next = th1.next
        th1.next = th2
        th1 = th1.next.next
        th2 = temp

#Binary Tree Preorder Traversal
#Tag:Tree, Stack
def preorderTraversal(root):
    ret = []
    while root:
        if root.left:
            temp = root.left
            while temp.right and temp.right != root:
                temp = temp.right
            if temp.right:
                temp.right = None
                root = root.right
            else:
                temp.right = root
                ret.append(root.val)
                root = root.left
        else:
            ret.append(root.val)
            root = root.right
    return ret

def preorderTraversal2(root):
    ret = []
    stack = [root]
    while stack:
        node = stack.pop()
        if node:
            ret.append(node.val)
            stack.append(node.right)
            stack.append(node.left)
    return ret

#Binary Tree Postorder Traversal
#Tag:Tree, Stack
def postorderTraversal(root):
    stack = [root]
    ret = []
    while stack:
        node = stack.pop()
        if node:
            ret.append(node.val)
            stack.append(node.left)
            stack.append(node.right)
    return ret[::-1]

def postorderTraversal2(root):
    ret = []
    while root:
        if root.right:
            temp = root.right
            while temp.left and temp.left != root:
                temp = temp.left
            if temp.left:
                temp.left = None
                root = root.left
            else:
                temp.left = root
                ret.append(root.val)
                root = root.right
        else:
            ret.append(root.val)
            root = root.left
            
    return ret[::-1]

#Single Number III
#Tag:  Bit Manipulation
def singleNumber3(nums):
    temp = 0
    for n in nums:
        temp ^= n
    temp &= -temp
    ret = [0,0]
    for n in nums:
        if n & temp == 0:
            ret[0] ^= n
        else:
            ret[1] ^= n
    return ret

#Insertion Sort List
#Linked List, Sort
def insertionSortList(head):
    dummy = linked_list.Node(-1)
    dummy.next = head
    pt = head
    prev = None
    while pt:
        if prev == None or prev.val <= pt.val:
            prev = pt
            pt = pt.next
        else:
            mark = pt.next
            temp = dummy
            while temp.next and temp.next.val <= pt.val:
                temp = temp.next
            pt.next = temp.next
            temp.next = pt
            pt = mark
            prev.next = mark
    
    return dummy.next

#Sort List
#Tag:Linked List, Sort
def quickSortLinkedList(head):
    def sort(head, tail):
        if head.next == tail or head.next.next == tail:
            return

        h1, h2, h3 = head, head.next, linked_list.Node(-1)
        t1, t2, t3 = h1, h2, h3
        pt, val = h2.next, h2.val
        while pt and pt != tail:
            if pt.val < val:
                t1.next, t1, pt = pt, pt, pt.next
            elif pt.val == val:
                t2.next, t2, pt = pt, pt, pt.next
            else:
                t3.next, t3, pt = pt, pt, pt.next

        t1.next = h2
        t3.next = tail
        t2.next = h3.next
        
        sort(h1, h2.next)
        sort(t2, tail)

    dummy = pt = linked_list.Node(-1)
    pt.next = head
    sort(pt, None)

    return dummy.next



def mergeSortLinkedList(head):
    def merge(lst1, lst2, head):
        while lst1 and lst2:
            if lst1.val < lst2.val:
                head.next = lst1
                lst1 = lst1.next
            else:
                head.next = lst2
                lst2 = lst2.next
            head = head.next
        head.next = lst1 if not lst2 else lst2
        while head.next:
            head = head.next
        return head

    def split(head, cnt):
        while cnt > 1 and head:
            head = head.next
            cnt -= 1
        if not head:
            return None

        temp, head.next = head.next, None
        return temp

    def sort(head):
        th = head.next
        size = 0
        while th:
            th = th.next
            size += 1

        i = 1
        while i < size:
            cur = head.next
            tail = head
            while cur:
                left = cur
                right = split(cur, i)
                cur = split(right, i)
                tail = merge(left, right, tail)
            i *= 2

        return head.next

    dummy = linked_list.Node(-1)
    dummy.next = head
    return sort(dummy)


#Maximum Product Subarray
#Array, Dynamic Programming
def maxProduct(nums):
    if not nums:
        return 0
    maxVal = minVal = ret = nums[0]
    for n in nums[1:]:
        maxVal, minVal = max( maxVal * n, minVal * n, n ), min( maxVal * n, minVal * n, n )
        ret = max(maxVal, ret)
    return ret

#Evaluate Reverse Polish Notation
#Tag: Stack
def evalRPN(tokens):
    stack = []
    ops = {'+':lambda x, y: x+y, '-':lambda x, y: x-y, '*':lambda x, y: x*y, '/':lambda x, y: x/y}
    for s in tokens:
        try:
            stack.append( float( s ) )
        except:
            stack.append( int( ops[s]( stack.pop(-2), stack.pop(-1) ) ) )
    return int( stack[-1] )

#Reverse Words in a String
#Tag:String
def reverseWords(s):
    ret = []
    for c in s:
        if not ret or (c == ' ' and ret[-1] != ""):
            ret.append("")
        if c != ' ':
            ret[-1] += c
            
    ret = ret[:-1] if len(ret) and ret[-1] == "" else ret
    return " ".join(ret[::-1])

#Find Minimum in Rotated Sorted Array II
#Tag: Array, Binary Search
def findMinInRotatedSortedArray(nums):
    lo, hi = 0, len(nums)-1
    while lo < hi:
        mid = lo + (hi - lo)/2
        if nums[mid] < nums[hi]:
            hi = mid
        elif nums[mid] > nums[hi]:
            lo = mid + 1
        else:
            hi -= 1
    
    return nums[lo] if lo < len(nums) else 0

#Find Peak Element
#Tag: Array, Binary Search
def findPeakElement(nums):
    for i in xrange(len(nums)):
        cur = nums[i]
        prev, nxt = nums[i-1] if i > 0 else cur-1, nums[i+1] if i < len(nums)-1 else cur - 1
        if prev < cur > nxt:
            return i
    return -1

#Fraction to Recurring Decimal
#Tag: Hash Table, Math
def fractionToDecimal(numerator, denominator):
    if numerator == 0 or denominator == 0:
        return '0'
    ret = "-" if (numerator < 0) ^ (denominator < 0) else ""
    numerator, denominator = abs(numerator), abs(denominator)
    val, numerator = divmod(numerator, denominator)
    ret += '%d' % val  + ('.' if numerator else "")
    
    mp = {}
    right = ""
    while numerator:
        if mp.has_key(numerator):
            i = mp[numerator]
            right = right[:i] + '(' + right[i:] + ')'
            break
        
        mp[numerator] = len(right)
        numerator *= 10
        val, numerator = divmod(numerator, denominator)
        right += "%d" % val
    
    return ret + right

#Largest Number
#Tag: sort
def largestNumber(nums):
    lst = map(str, nums)
    lst.sort(cmp = lambda x, y: cmp(x+y, y+x), reverse = True)
    ret = "".join(lst) 
    return ret.lstrip('0') or '0'


#Repeated DNA Sequences
#Tag: Hash Table, Bit Manipulation
def findRepeatedDnaSequences(s):
    mask, val, ret, mp = 2 ** 30 - 1, 0, [], {}
    for i in xrange(0, len(s)):
        val = (val << 3) | (ord(s[i]) & 7)
        if i >= 9:
            val = val & mask
            mp[val] = 1 if not mp.has_key(val) else mp[val] + 1
            if mp[val] == 2:
                ret.append(s[i-9:i+1])
    return ret

#Binary Tree Right Side View
#Tag:Tree, Depth-first Search, Breadth-first Search
def rightSideView(root):
    def collect(root, ret, level):
        if root == None:
            return ret
        if level == len(ret):
            ret.append(root.val)
        collect(root.right, ret, level+1)
        collect(root.left, ret, level+1)
        return ret
        
    return collect(root, [], 0)


#Number of Islands
#Tag:Depth-first Search, Breadth-first Search, Union Find
def numIslands(grid):
    def sink(x, y):
        if not (0 <= x < len(grid) and 0 <= y < len(grid[0]) and grid[x][y] == '1'):
            return 0
        grid[x][y] = '*'
        map(sink, (x,x,x+1,x-1), (y-1,y+1,y,y))
        return 1
        
    return sum(sink(x,y) for x in xrange(len(grid)) for y in xrange(len(grid[0])))

#Bitwise AND of Numbers Range
#Tag:Bit Manipulation
def rangeBitwiseAnd(m, n):
    while n != 0 and n > m:
        n &= n-1
    return n

#Course Schedule
#Tag: Depth-first Search, Breadth-first Search, Graph, Topological Sort
def courseScheduleBFS(numCourses, prerequisites):
    degree = [0] * numCourses
    links = {}
    for pair in prerequisites:
        degree[pair[1]] += 1
        if not links.has_key(pair[0]):
            links[pair[0]] = []
        links[pair[0]].append(pair[1])
        
    queue = [i for i in xrange(numCourses) if degree[i] == 0]
    while queue:
        idx = queue.pop(0)
        if not links.has_key(idx):
            continue
        adj = links[idx]
        for i in xrange(len(adj)):
            degree[adj[i]] -= 1
            if degree[adj[i]] == 0:
                queue.append(adj[i])
          
    return sum(degree) == 0 if len(degree) else False


#Implement Trie (Prefix Tree)
#Tag:Trie, Design
class TrieNode(object):
    def __init__(self):
        self.end = False
        self.lst = [None] * 26
        
class Trie(object):

    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node, i = self.find(word, True)
        node.end = True
            
    def find(self, word, create):
        node = self.root
        if len(word) == 0:
            return node, 0
        for i in xrange(len(word)):
            idx = ord(word[i]) - ord('a')
            if node.lst[idx] == None:
                if create:
                    node.lst[idx] = TrieNode()
                else:
                    return node, i
            node = node.lst[idx]
        return node, len(word)
            
    def search(self, word):
        node, i = self.find(word, False)
        return node.end and i == len(word)


    def startsWith(self, prefix):
        node, i = self.find(prefix, False)
        return i == len(prefix)

#Minimum Size Subarray Sum
#Tag: Array, Two Pointers, Binary Search
def minSubArrayLen(s, nums):
    if not nums or sum(nums) < s:
        return 0

    left = sumVal = 0
    ret = len(nums)
    for i in xrange(len(nums)):
        sumVal += nums[i]
        while sumVal >= s:
            ret = min(i - left + 1, ret)
            sumVal -= nums[left]
            left += 1
            
    return ret

#Course Schedule II
#Tag:Depth-first Search, Breadth-first Search, Graph, Topological Sort
def courseScheduleFindCourses(numCourses, prerequisites):
    def dfs(i, visited, graph, ret):
        if visited[i] == 1:
            return True
        if visited[i] == -1:
            return False
            
        visited[i] = -1
        for n in graph[i]:
            if not dfs(n, visited, graph, ret):
                return False
        ret.append(i)
                
        visited[i] = 1
        return True
        
    visited = [0] * numCourses
    graph = {x:[] for x in xrange(numCourses)}
    for p in prerequisites:
        graph[p[1]].append(p[0])
        
    ret = []
    for i in xrange(numCourses):
        if not dfs(i, visited, graph, ret):
            return []
            
    return ret[::-1]

#Kth Largest Element in an Array
#Tag: Divide and Conquer, Heap
def findKthLargest(nums, k):
    def search(lst, lo, hi, idx):
        if hi - lo < 1:
            return -1
        left, pt, right = lo, lo, hi - 1
        val = lst[lo]
        while pt <= right:
            if lst[pt] == val:
                pt += 1
            elif lst[pt] > val:
                lst[left], lst[pt] = lst[pt], lst[left]
                left += 1
                pt += 1
            else:
                lst[right], lst[pt] = lst[pt], lst[right]
                right -= 1
                
        if left <= idx <= right:
            return lst[idx]
        elif left > idx:
            return search(lst, lo, left, idx)
        return search(lst, pt, hi, idx)
        
    return search(nums, 0, len(nums), k-1)

import heapq
def findKthLargest2(nums, k):
    heap = []
    for n in nums:
        if len(heap) == k:
            heapq.heappushpop(heap, n)
        else:
            heapq.heappush(heap, n)
    return heapq.heappop(heap) if heap else 0


#Contains Duplicate III
#Tag: Binary Search Tree
def containsNearbyAlmostDuplicate(nums, k, t):
    if not nums or k <= 0 or t < 0:
        return False
        
    mp = {}
    for i, val in enumerate(nums):
        bucket = val/(t + 1)
        for idx in xrange(bucket-1, bucket+2):
            if mp.has_key(idx) and abs(mp[idx] - val) <= t:
                return True
        mp[bucket] = val
        if i >= k:
            del mp[nums[i-k]/(t+1)]
    
    return False

#Bulls and Cows
#Tag: Hash Table
def bullsAndCow(secret, guess):
    cnt, a, b = [0] * 10, 0, 0
    for i in xrange(len(secret)):
        x, y = int(secret[i]), int(guess[i])
        if x == y:
            a += 1
        else:
            b += (cnt[x] < 0) + (cnt[y] > 0)
            cnt[x] += 1
            cnt[y] -= 1

    return "%dA%dB" % (a, b)

#Longest Increasing Subsequence
#Tag: Dynamic Programming Binary Search
def lengthOfLIS(nums):
    def search(lst, lo, hi, target):
        if hi - lo < 1:
            return lo
        mid = lo + (hi - lo)/2
        return search(lst, mid+1, hi, target) if lst[mid] < target else search(lst, lo, mid, target)
        
    seq = []
    for n in nums:
        pos = search(seq, 0, len(seq), n)
        if pos >= len(seq):
            seq.append(n)
        else:
            seq[pos] = min(seq[pos], n)
            
    return len(seq)

#Number of Digit One
#Tag: Math
def countDigitOne(n):
    ret, m = 0, 1
    while m <= n:
        b, a, c = n/m/10, n%m, n/m % 10
        ret += (b + (c > 1)) * m + (c == 1) * (a + 1)
        m *= 10
    return ret

#Count Complete Tree Nodes
#Tag:Tree, Binary Search
def countCompleteTreeNodes(root):
    def leftCnt(root):
        cnt = 0
        while root:
            cnt += 1
            root = root.left
        return cnt
            
    if not root:
        return 0
        
    l, r = leftCnt(root.left), leftCnt(root.right)
    if l == r:
        return (1 << l) + countCompleteTreeNodes(root.right)
    return (1 << r) + countCompleteTreeNodes(root.left)

#Remove Invalid Parentheses
#Tag:Depth-first Search, Breadth-first Search
def removeInvalidParentheses(s):
    def check(s):
        n = 0
        for c in s:
            n = n + 1 if c == '(' else n - 1 if c == ')' else n
            if n < 0:
                return False
        return n == 0
            
    queue = {s}
    ret = []
    while queue:
        ret = filter(check, queue)
        if ret:
            return ret
        queue = {s[:i] + s[i+1:] for s in queue for i in xrange(len(s))}
    
    return ret

#Maximal Square
#Tag: Dynamic Programming
def maximalSquare(matrix):
    if not matrix:
        return 0
        
    m, n = len(matrix), len(matrix[0])
    dp = [[0 for y in xrange(n+1)] for x in xrange(m+1)]
    maxVal = 0
    for i in xrange(1, m+1):
        for j in xrange(1, n+1):
            if matrix[i-1][j-1] == '1':
                dp[i][j] = min(dp[i-1][j-1], dp[i-1][j], dp[i][j-1]) + 1
                maxVal = max(maxVal, dp[i][j])

    return maxVal * maxVal

#Basic Calculator
#Tag:Stack, Math
def basicCalculator(s):
    nums, ops = [], []
    num = rst = 0
    sign = 1

    for c in s:
        if c == ' ':
            continue
        elif c.isdigit():
            num = num * 10 + int(c)
        elif c == '(':
            nums.append(rst)
            ops.append(sign)
            num = rst = 0
            sign = 1
        elif c == ')':
            num = rst + sign * num
            sign, rst = ops.pop(), nums.pop()
        else:
            rst += sign * num
            num = 0
            sign = [-1, 1][c == '+']
            
    return rst + sign * num

#Basic Calculator II
#Tag:String
def basicCalculator2(s):
    num, op, stack = 0, '+', [0]
    ops = {'+':lambda x, y: y, '-':lambda x, y: -y, '*':lambda x, y: x*y, '/':lambda x, y: (int)(float(x)/float(y))}
    for i, c in enumerate(s):
        if c.isdigit():
            num = num * 10 + int(c)
        if not c.isdigit() and c != ' ' or i == len(s)-1:
            prev = 0 if op in '+-' else stack.pop()
            stack.append(ops[op](prev, num))
            num, op = 0, c
    return sum(stack)

#Range Sum Query - Immutable
#Tag:Dynamic Programming
class NumArray(object):
    def __init__(self, nums):
        self.dp = nums
        for i in xrange(1, len(nums)):
            self.dp[i] += self.dp[i-1]

    def sumRange(self, i, j):
        return self.dp[j] - (self.dp[i-1] if i > 0 else 0)


#Majority Element II
#Tag:Array
def majorityElement2(nums):
    a, b = None, None
    cnt1, cnt2 = 0, 0
    for n in nums:
        if n == a:
            cnt1 += 1
        elif n == b:
            cnt2 += 1
        elif cnt1 == 0:
            a = n
            cnt1 += 1
        elif cnt2 == 0:
            b = n
            cnt2 += 1
        else:
            cnt1 -= 1
            cnt2 -= 1

    cnt1 = cnt2 = 0
    for n in nums:
        if n == a:
            cnt1 += 1
        elif n == b:
            cnt2 += 1

    return ([a] if cnt1 > len(nums)/3 else [])  + ([b] if cnt2 > len(nums)/3 else [])

#Lowest Common Ancestor of a Binary Tree
#Tag: tree
def lowestCommonAncestor(root, p, q):
    if root == None or root == p or root == q:
        return root
    l, r = lowestCommonAncestor(root.left, p, q), lowestCommonAncestor(root.right, p, q)
    return root if l and r else l or r 


#Delete Node in a Linked List
#Tag: Linked list
def deleteNode(node):
    node.val = node.next.val
    node.next = node.next.next
    
def deleteNode2(head, idx):
    dummy = linked_list.Node(-1)
    dummy.next = head
    prev = dummy
    while head and idx:
        prev = head
        head = head.next
        idx -= 1
    if head == None:
        return dummy.next
    prev.next = head.next
    return dummy.next

#Product of Array Except Self
#Tag: Array
def productExceptSelf(nums):
    size = len(nums)
    out = [1] * size
    t1 = t2 = 1
    for i in xrange(size-1):
        t1 *= nums[i]
        t2 *= nums[size-1-i]
        out[i+1] *= t1
        out[size - i - 2] *= t2
        
    return out
        

#Search a 2D Matrix II
#Tag:Divide and Conquer, Binary Search
def searchMatrix2(matrix, target):
    if not matrix:
        return False

    i = 0
    j = len(matrix[0]) - 1
    while i < len(matrix) and j >= 0:
        if matrix[i][j] == target:
            return True
        elif matrix[i][j] < target:
            i += 1
        else:
            j -= 1
            
    return False

#Different Ways to Add Parentheses
#Tag:  Divide and Conquer
def diffWaysToCompute(input):
    def collect(s, cache):
        ops = {'+':lambda x, y:x+y, '-':lambda x, y:x-y, '*':lambda x, y:x*y}
        ret = []
        if 1 != 2:
            for i, c in enumerate(s):
                if c in '+-*':
                    for p in collect(s[:i], cache):
                        for n in collect(s[i+1:], cache):
                            ret.append(ops[c](p, n))
                        
            if not ret:
                ret.append(int(s))
            
            
        return ret
        
    return collect(input, {})


#Ugly Number
#Tag: Math
def isUgly(num):
    for x in 2,3,5:
        while num > 0 and num % x == 0:
            num /= x
    return num == 1

#Sliding Window Maximum
#Tag: Heap
def maxSlidingWindow(nums, k):
    queue = []
    ret = []
    for i in xrange(len(nums)):
        if queue and queue[0] <= i - k:
            queue.pop(0)
        while queue and nums[queue[-1]] < nums[i]:
            queue.pop()
        queue.append(i)
            
        if i >= k - 1:
            ret.append(nums[queue[0]])
            
    return ret

#Ugly Number II
#Tag:Dynamic Programming, Heap, Math
def nthUglyNumber(n):
    ret = [1]
    a, b, c = 2, 3, 5
    i, j, k = 0, 0, 0 
    while len(ret) < n:
        v1, v2, v3 = a *ret[i], b * ret[j], c * ret[k]
        val = min(v1, v2, v3)
        i += 1 if val == v1 else 0
        j += 1 if val == v2 else 0
        k += 1 if val == v3 else 0
        ret.append(val)
        
    return ret[-1]

#Perfect Squares
#Tag:  Dynamic Programming, Breadth-first Search, Math
def perfectSquares(n):
    if n <= 0:
        return 0
    lst = []
    i = 1
    while i * i <= n:
        lst.append(i * i)
        i += 1
        
    queue = {n}
    level = 0
    while queue:
        temp = set()
        level += 1
        for x in queue:
            for y in lst:
                if x < y:
                    break
                if x == y:
                    return level
                temp.add(x-y)
        queue = temp
            
    return level

def perfectSquares2(n):
    if n <= 0:
        return 0
    
    dp = [65535] * (n + 1)
    dp[0] = 0
    for i in xrange(1, n+1):
        minVal = None
        for j in xrange(1, i+1):
            val = j * j
            if val <= i:
                minVal = min(minVal, dp[i - val] + 1) if minVal else (dp[i- val] + 1)
        dp[i] = minVal
        
    return dp[-1]


#Best Time to Buy and Sell Stock with Cooldown
def maxProfitWithCooldown(prices):
    if len(prices) <= 1:
        return 0
        
    buy = [0] * len(prices)
    sell = [0] * len(prices)
    buy[0] = -prices[0]
    buy[1] = max(-prices[0], -prices[1])
    sell[1] = max(0, prices[1] + buy[0])
    for i in xrange(2, len(prices)):
        buy[i] = max(buy[i-1], sell[i-2] - prices[i])
        sell[i] = max(prices[i] + buy[i-1], sell[i-1])
        
    return sell[-1]

#Regular Expression Matching
#Tag:Dynamic Programming, Backtracking, String
def reMatch2(s, p):
    def match(s, p, mp):
        if not p:
            return not s
            
        if not mp.has_key(tuple([s, p])):
            ret = False
            if len(p) > 1 and p[1] == '*':
                ret = match(s, p[2:], mp) or ( (len(s) > 0 and (s[0] == p[0] or p[0] == '.')) and match(s[1:], p, mp) )
            elif p[0] not in '.*':
                ret = len(s) > 0 and s[0] == p[0] and match(s[1:], p[1:], mp)
            elif p[0] == '.':
                ret = len(s) > 0 and match(s[1:], p[1:], mp)
            mp[tuple([s, p])] = ret
        
        return mp[tuple([s, p])]

    return match(s, p, {})

def reMatch(s, p):
    m, n = len(s), len(p)
    dp = [[False] * (n + 1) for x in xrange(m + 1)] 
    dp[0][0] = True
    for i in xrange(m+1):
        for j in xrange(1, n + 1):
            if p[j-1] == '*':
                dp[i][j] = dp[i][j-2] or (i > 0 and (s[i-1] == p[j-2] or p[j-2] == '.') and dp[i-1][j])
            else:
                dp[i][j] = i > 0 and (dp[i-1][j-1] and (s[i-1] == p[j-1] or p[j-1] == '.'))
    
    return dp[m][n]

#Best Time to Buy and Sell Stock IV
#Tag: Dynamic Programming
def maxProfitWithKTransactions(k, prices):
    if k == 0:
        return 0
        
    if k >= len(prices)/2:
        ret = 0
        for i in xrange(1, len(prices)):
            ret += max(0, prices[i] - prices[i-1])
        return ret
        
    dp = [[0] * (len(prices) + 1) for x in xrange(k + 1)]
    for i in xrange(1, k + 1):
        temp = -prices[0]
        for j in xrange(1, len(prices)):
            dp[i][j] = max(dp[i][j-1], temp + prices[j])
            temp = max(temp, dp[i-1][j] - prices[j])
            
    return dp[k][len(prices)-1]



#Find Median from Data Stream
#Tag:  Heap, Design
class MedianFinder:
    def __init__(self):
        self.heaps = [], []

    def addNum(self, num):
        small, large = self.heaps
        heapq.heappush(small, -heapq.heappushpop(large, num))
        if len(small) > len(large):
            heapq.heappush(large, -heapq.heappop(small))

    def findMedian(self):
        small, large = self.heaps
        return (-small[0] + large[0]) * 1.0 / 2 if len(small) == len(large) else large[0]

#Reverse Nodes in k-Group
#Tag: Linked list
def reverseKGroup(head, k):
    cnt = 0
    cur = head
    while cur and cnt < k:
        cur = cur.next
        cnt += 1
        
    if cnt == k:
        cur = reverseKGroup(cur, k)
        while cnt > 0:
            cnt -= 1
            temp = head.next
            head.next = cur
            cur = head
            head = temp
        head = cur
    
    return head

#Longest Substring with At Most Two Distinct Characters
#Tag:Hash Table, Two Pointers, String
def lengthOfLongestSubstringTwoDistinct(s):
    x, j, k = 0, -1, 0
    ret = 0
    for i in xrange(1, len(s)):
        if s[i] == s[i-1]:
            continue
        if j >= 0 and s[j] != s[i]:
           ret = max(ret, i - x)
           x = j + 1
        j = i - 1
        
    return max(ret, len(s) - x)



#Read N Characters Given Read4 II - Call multiple times
#Tag:String
class BuffReader:
    def __init__(self):
        self.buffer = ['', '', '', '']
        self.bPos = 0
        self.cnt = 0

    def read(self, buf, n, read4):
        pos = 0
        while pos < n:
            if self.bPos == 0:
                self.cnt = read4(self.buffer)
            if self.cnt == 0:
                break
            while pos < n and self.bPos < self.cnt:
                buf[pos] = self.buffer[self.bPos]
                pos += 1
                self.bPos += 1
            
            self.bPos %= self.cnt
            
        return pos

#Edit Distance
#Tag: Dynamic Programming, String
def editDistance(word1, word2):
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for x in xrange(m + 1)]
    for i in xrange(n + 1):
        dp[0][i] = i
    for i in xrange(m + 1):
        dp[i][0] = i
    for i in xrange(1, m + 1):
        for j in xrange(1, n + 1):
            if word1[i-1] == word2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i][j-1], dp[i-1][j], dp[i-1][j-1]) + 1
                
    return dp[-1][-1]

#Super Ugly Number
#Tag: Math, Heap
def nthSuperUglyNumber(n, primes):
    uglys = [1]
    def gen(prime):
            for i in uglys:
                yield prime * i
    merged = heapq.merge(*map(gen, primes))
    while len(uglys) < n:
        i = next(merged)
        if i != uglys[-1]:
            uglys.append(i)
                
    return uglys[-1]

#One Edit Distance
#Tag: String
def isOneEditDistance(s, t):
    if s == t:
        return False
    i = 0
    while i < len(s) and i < len(t):
        if s[i] != t[i]:
            break
        i += 1
    return s[i+1:] == t[i:] or s[i:] == t[i + 1:] or s[i+1:] == t[i+1:]


#Missing Ranges
#Tag: Array
def findMissingRanges(nums, lower, upper):
    prev = lower - 1
    ret = []
    for i in xrange(len(nums) + 1):
        nxt = upper + 1 if i == len(nums) else nums[i]
        if prev + 2 == nxt:
            ret.append('%d' % (prev + 1))
        elif prev + 2 < nxt:
            ret.append('%d->%d' % (prev + 1, nxt - 1))
        prev = nxt
    return ret


#Reverse Words in a String II
#Tag: String
def reverseWordsInplace(s):
    def reverse(s, i, j):
        while i < j:
            s[i], s[j] = s[j], s[i]
            i += 1
            j -= 1
    reverse(s, 0, len(s)-1)
    i = 0
    while i < len(s):
        j = i
        while i < len(s) and s[i] != ' ':
            i += 1
        reverse(s, j, i-1)
        i += 1

#Binary Tree Vertical Order Traversal
#Tag: Hash Table
def verticalOrder(root):
    import collections
    mp = collections.defaultdict(list)
    queue = [(root, 0)]
    for node, i in queue:
        if node:
            mp[i].append(node.val)
            queue += (node.left, i-1), (node.right, i+1)
    return [mp[i] for i in sorted(mp.keys())]

#Count of Smaller Numbers After Self
#Tag: Divide and Conquer, Binary Indexed Tree, Segment Tree, Binary Search Tree
def countSmaller(nums):
    def sort(nums, ret):
        if len(nums) <= 1:
            return nums
            
        mid = len(nums)/2
        left, right = sort(nums[:mid], ret), sort(nums[mid:], ret)
        i = j = 0
        while i < len(left) or j < len(right):
            v1 = left[i][1] if i < len(left) else float('inf')
            v2 = right[j][1] if j < len(right) else float('inf')
            if v1 <= v2:
                nums[i + j] = left[i]
                if i < len(left):
                    ret[left[i][0]] += j
                i += 1
            else:
                nums[i + j] = right[j]
                j += 1

        return nums
        
    ret = [0] * len(nums)
    sort([p for p in enumerate(nums)], ret)
    return ret

#Remove Duplicate Letters
#Tag: Stack, Greedy
import collections
def removeDuplicateLetters(s):
    if not s:
        return ""
    cnt = collections.Counter(list(s))
    pos = 0
    for i, c in enumerate(s):
        if c < s[pos]:
            pos = i
        cnt[c] -= 1
        if cnt[c] == 0:
            break
    return s[pos] + removeDuplicateLetters(s[pos:].replace(s[pos], ''))


#Shortest Word Distance II
#Tag: Hash Table, Design
import collections
class WordDistance(object):
    def __init__(self, words):
        self.words = collections.defaultdict(list)
        for i, word in enumerate(words):
            self.words[word].append(i)

    def shortest(self, word1, word2):
        lst1, lst2 = self.words[word1], self.words[word2]
        i1 = i2 = 0
        ret = float('inf')
        while i1 < len(lst1) and i2 < len(lst2):
            val1, val2 = lst1[i1], lst2[i2]
            if val1 < val2:
                i1 += 1
            else:
                i2 += 1
            ret = min(ret, abs(val1 - val2))
        return ret

#todo: Range Sum Query - Mutable

#Strobogrammatic Number
#Tag: Hash Table, Math
def isStrobogrammatic(num):
    return all(num[i] + num[~i] in '696 00 11 88' for i in range(len(num)/2+1))

#Strobogrammatic Number II
#Tag: Math, Recursion
def findStrobogrammatic(n):
    pairs = "00 11 88 69 96"
    
    def helper(n, m):
        if n == 0:
            return [""]
        if n == 1:
            return [x[0] for x in pairs.split()[:-2]]
            
        prev = helper(n-2, m)
        ret = []
        for item in prev:
            for pair in pairs.split():
                if n == m and pair == '00':
                    continue
                ret.append(pair[0] + item + pair[1])
        return ret
        
    return helper(n, n)

def countStrobogrammatic(n):
    if n == 0:
        return 0
    if n == 1:
        return 3
    ret = 4 * (5**(n/2 - 1))
    if n % 2 == 0:
        return ret
    return ret * 3

#Group Shifted Strings
#Tag:Hash Table, String
def groupStrings(strings):
    mp = collections.defaultdict(list)
    for s in strings:
        mp[tuple([(ord(c) - ord(s[0])) % 26 for c in s])].append(s)
    return map(sorted, mp.values())

#Count Univalue Subtrees
#Tag: Tree
def countUnivalSubtrees(root):
    def collect(root):
        if root == None:
            return True
        
        l, r = collect(root.left), collect(root.right)
        
        vL = root.left.val if root.left else root.val
        vR = root.right.val if root.right else root.val
        if root.val == vL == vR and l and r:
            cnt[0] += 1
            return True
        return False

    cnt = [0]
    collect(root)
    return cnt[0]

#Meeting Rooms II
#Tag:Heap, Greedy, Sort
def minMeetingRooms(intervals):
    mp = collections.defaultdict(int)
    for x, y in intervals:
        mp[x] += 1
        mp[y] -= 1

    ret = cnt = 0
    for i in sorted(mp.keys()):
        cnt += mp[i]
        ret = max(ret, cnt)
    return ret

#Factor Combinations
#Tag:  Backtracking
def getFactors(n):
    todo, combis = [(n, 2, [])], []
    while todo:
        n, i, combi = todo.pop()
        while i * i <= n:
            if n % i == 0:
                combis += combi + [i, n/i],
                todo += (n/i, i, combi+[i]),
            i += 1
    return combis

def getFactors2(n):
    def factor(n, i, combi, combis):
        while i * i <= n:
            if n % i == 0:
                combis += combi + [i, n/i],
                factor(n/i, i, combi+[i], combis)
            i += 1
        return combis
    return factor(n, 2, [], [])

#Maximum Product of Word Lengths
#Tag:  Bit Manipulation
def maxProductOfWordLength(words):
    def mask(s):
        return reduce(lambda x, y: x | y, [1 << (ord(c) - ord('a')) for c in s], 0)
    lst = sorted(map(lambda x: (mask(x), len(x)), words), cmp = lambda x, y: cmp(x[1], y[1]), reverse = True)
    
    ret = 0
    for i in xrange(len(lst)):
        for j in xrange(1 + i, len(lst)):
           if lst[i][0] & lst[j][0] == 0:
               ret = max(lst[i][1] * lst[j][1], ret)
               break
           
    return ret

#Paint House
#Tag: Dynamic Programming
def paintHouse(costs):
    for i in xrange(1, len(costs)):
        costs[i][0] += min(costs[i-1][1], costs[i-1][2])
        costs[i][1] += min(costs[i-1][0], costs[i-1][2])
        costs[i][2] += min(costs[i-1][1], costs[i-1][0])
        
    return min(costs[-1][0], costs[-1][1], costs[-1][2]) if costs else 0


#Verify Preorder Sequence in Binary Search Tree
#Tag: Tree, Stack
def verifyPreorder1(preorder):
    stack = []
    low = float('-inf')
    for p in preorder:
        if p < low:
            return False
        while stack and stack[-1] < p:
            low = stack.pop()
        stack.append(p)
    return True

def verifyPreorder2(preorder):
    low = float('-inf')
    i = -1
    for p in preorder:
        if p < low:
            return False
        while i >= 0 and preorder[i] < p:
            low = preorder[i]
            i -= 1
        i += 1
        preorder[i] = p
       
    return True

#Burst Balloons
#Tag:Divide and Conquer, Dynamic Programming
def maxCoins(nums):
    nums = [1] + [i for i in nums if i > 0] + [1]
    n = len(nums)
    if not n:
        return 0
        
    dp = [[0] * n for x in xrange(n)]
    for k in xrange(2, n):
        for left in xrange(0, n-k):
            right = left + k
            for i in xrange(left+1, right):
                dp[left][right] = max(dp[left][right], nums[i] * nums[left] * nums[right] + dp[left][i] + dp[i][right])
                
    return dp[0][-1]
            
#3Sum Smaller
#Tag:Array, Two Pointers
def threeSumSmaller(nums, target):
    nums.sort()
    count = 0
    for k in range(len(nums)):
        i, j = 0, k - 1
        while i < j:
            if nums[i] + nums[j] + nums[k] < target:
                count += j - i
                i += 1
            else:
                j -= 1
    return count
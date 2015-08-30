 # Compare Version Numbers
def compareVersion(version1, version2):
    lst1 = [int(x) for x in version1.split('.')]
    lst2 = [int(x) for x in version2.split('.')]

    pt = 0
    while True:
        val1 = lst1[pt] if pt < len(lst1) else 0
        val2 = lst2[pt] if pt < len(lst2) else 0
        if val1 < val2:
            return -1
        elif val1 > val2:
            return 1

        if pt >= len(lst1) and pt >= len(lst2):
            break
        pt += 1
    return 0


#happy number
def isHappyNumber( n ):
    def helper(n):
        ret = 0
        while n > 0:
            ret += (n % 10) ** 2
            n /= 10
        return ret
    n1 = n2 = n
    while True:
        n1 = helper(n1)
        n2 = helper( helper(n2) )
        if n1 == 1:
            return True
        elif n1 == n2:
            return False
    return False


#invert binary tree:
def invertTree(root):
    if root == None:
        return None
    root.left, root.right = invertTree(root.right), invertTree(root.left)
    return root


#string to interger:
def strToInterger(s):
    maxVal = 2 ** 31 - 1
    maxFlip = 2 ** 31

    pt = 0
    while pt < len(s) and s[pt] == ' ':
        pt += 1
    if pt >= len(s):
        return 0

    flip = 1
    if s[pt] == '-' or s[pt] == '+':
        flip = -1 if s[pt] == '-' else 1
        pt += 1

    ret = 0
    numLst = ['%d' % d for d in xrange(0, 10)]
    while pt < len(s) and s[pt] in numLst:
        val = int(s[pt])
        if ret > maxVal/10 or (ret == maxVal/10 and val > 7):
            ret = maxVal if flip == 1 else maxFlip
            break
        ret = ret * 10 + val
        pt += 1

    return ret * flip


#zigZag conversion:
def zigZagConvertion(s, numRows):
    if len(s) == 0 or numRows <= 1:
        return s
    diff = (numRows - 1) * 2
    ret = ""
    for row in xrange(0, numRows):
        for pt in xrange(row, len(s), diff):
            ret += s[pt]
            pt2 = pt + diff - row * 2
            if pt2 < len(s) and pt2 != pt and pt2 != pt + diff:
                ret += s[pt2]
    return ret


#reverse interget:
def reverseInt(x):
    maxVal = 2 ** 31 - 1
    flip = -1 if x < 0 else 1
    ret = 0
    x = abs(x)
    while x > 0:
        if ret > maxVal/10 or (ret == maxVal/10 and x % 10 > 7):
            return 0
        ret = ret * 10 + x % 10
        x /= 10
    return ret * flip


#palindrome number:
def palindromeNumber(x):
    x = abs(x)
    if x == 0 or x % 10 == 0:
        return False
    temp = 0
    while temp < x:
        temp = temp * 10 + x % 10
        x /= 10
    return temp == x or temp/10 == x


#longest common prefix:
def longestCommonPrefix(strs):
    if len(strs) == 0:
        return ""
    ret = strs[0]
    for s in strs:
        if len(ret) > len(s):
            ret = ret[:len(s)]
        length = len(ret)
        while ret[:length] != s[:length]:
            length -= 1
        if length == 0:
            return ""
        ret = ret[:length]
    return ret


#Remove Nth Node From End of List 
def removeNthFromEnd(head, n):
    if head == None or n == 0:
        return head

    th = head
    tn = n
    size = 0
    while tn > 0:
        tn -= 1
        size += 1
        th = th.next
        if th == None:
            tn = n % size
            if tn == 0:
                return head.next
            th = head
            size = 0

    th2 = head
    while th and th.next:
        th = th.next
        th2 = th2.next

    th2.next = th2.next.next if th2.next else th2.next
    return head

#Valid parentheses
def isValidParentheses(s):
    if len(s) == 0:
        return False

    stack = []
    mp = {')':'(','}':'{',']':'['}
    for c in s:
        if c in mp.values():
            stack.append(c)
        elif c in mp.keys():
            if len(stack) and stack[-1] == mp[c]:
                stack.pop()
            else:
                return False
        else:
            return False

    if len(stack):
        return False
    return True


#Merge two sorted lists
def mergeSortedList1(l1, l2):
    if l1 == None:
        return l2
    if l2 == None:
        return l1
    if l1.val <= l2.val:
        l1.next = mergeSortedList1(l1.next, l2)
        return l1
    else:
        l2.next = mergeSortedList1(l2.next, l1)
        return l2
    return None



import linked_list
def mergeSortedList2(l1, l2):
    if l1 == None:
        return l2
    if l2 == None:
        return l1
    dummy = linked_list.Node(-1)
    pt = dummy
    while l1 and l2:
        if l1.val <= l2.val:
            pt.next = l1
            pt = pt.next
            l1 = l1.next
        else:
            pt.next = l2
            pt = pt.next
            l2 = l2.next

        if l1 == None:
            pt.next = l2
        if l2 == None:
            pt.next = l1

    return dummy.next


#Remove Duplicates from Sorted Array
def removeDuplicates( A ):
    count = 1
    for i in xrange(1, len(A)):
        if A[i] != A[i-1]:
            A[count] = A[i]
            count += 1

    return min(count, len(A))

#Remove Element 
def removeElement(nums, val):
    count = 0
    for n in nums:
        if n != val:
            nums[count] = n
            count += 1
    return count


#Implement strStr()
def strStr(haystack, needle):
    if haystack == needle or needle == "":
        return 0

    for i in xrange(0, len(needle)):
        for idx in xrange(i, len(haystack), len(needle)):
            end = idx + len(needle)
            if end <= len(haystack) and haystack[idx:end] == needle:
                return idx

    return -1


#Count and Say          
def countAndSay(n):
    def nextNum(s):
        count = 1
        ret = ""
        for i in xrange(1, len(s)):
            if s[i] == s[i-1]:
                count += 1
            else:
                ret += "%d%s" % (count, s[i-1])
                count = 1

        ret += "%d%s" % (count, s[-1])
        return ret

    if n <= 0:
        return ""
    ret = '1'
    for i in xrange(1, n):
        ret = nextNum(ret)

    return ret


#Length of Last Word
def lengthOfLastWord(s):
    pt = len(s) - 1
    while pt >= 0 and s[pt] == ' ':
        pt -= 1
    count = 0
    while pt >= 0 and s[pt] != ' ':
        pt -= 1
        count += 1
    return count


#Plus One
    # @param digits, a list of integer digits
    # @return a list of integer digits
def plusOne(digits):
    if len(digits) == 0:
        return [1]
    digits[-1] += 1
    extra = 0
    for i in xrange(len(digits)-1, -1, -1):
        digits[i] += extra
        if digits[i] >= 10:
            extra = 1
            digits[i] %= 10
        else:
            extra = 0
        if extra == 0:
            break

    if extra:
        digits = [1] + digits
    return digits



#Add Binary
    # @param a, a string
    # @param b, a string
    # @return a string

def addBinary(a, b):
    extra = 0
    ret = ""
    while len(a) or len(b):
        val1 = int(a[-1]) if len(a) else 0
        val2 = int(b[-1]) if len(b) else 0
        cmb = val1 ^ val2 ^ extra
        if val1 & val2 or val1 & extra or val2 & extra:
            extra = 1
        else:
            extra = 0
        ret = "%d%s" % (cmb, ret)
        a = a[:-1] if len(a) else []
        b = b[:-1] if len(b) else []

    if extra:
        ret = "%d%s" % (extra, ret)

    return ret

#Climbing Stairs

def climbStairs1(n):
    a = 1
    b = 2
    
    if n <= 0:
        return 0
    elif n == 1:
        return a
        
    cur = b
    prevA = a
    prevB = b
    while n > 2:
        cur,prevA = prevA + prevB, prevB
        prevB = cur
        n -= 1
        
    return cur

def climbStairs2(n):
    def climb(n, cnt, mp):
        if n <= 0:
            return cnt
        if n == 1:
            return cnt + 1
        if n == 2:
            return cnt + 2
            
        if mp.has_key(n):
            return cnt + mp[n]
            
        cnt = climb(n-1, cnt, mp)
        cnt = climb(n-2, cnt, mp)
        
        mp[n] = cnt
        
        return cnt

    return climb(n, 0, {}) 


#merge sorted array
# @param A  a list of integers
# @param m  an integer, length of A
# @param B  a list of integers
# @param n  an integer, length of B
# @return nothing(void)
def mergeSortedArray(A, m, B, n):
    ptA = m - 1
    ptB = n - 1
    for x in xrange(m+n-1, -1, -1):
        val = 0
        if ptA >= 0 and ptB >= 0 and A[ptA] >= B[ptB]:
            val = A[ptA]
            ptA -= 1
        elif ptB >= 0:
            val = B[ptB]
            ptB -= 1
        else:
            break
        A[x] = val

#Remove Duplicates from Sorted List

# @param head, a ListNode
# @return a ListNode
def deleteDuplicates(head):
    pt = head
    while pt != None:
        pt2 = pt.next
        while pt2 != None and pt2.val == pt.val:
            pt2 = pt2.next
        pt.next = pt2
        pt = pt.next
    
    return head


#is same tree

def isSameTree(p, q):
    val1 = p.val if p else None
    val2 = q.val if q else None
    if val1 != val2:
        return False
    elif val1 == None:
        return True
        
    return isSameTree(p.left, q.left) and isSameTree(p.right, q.right)

#Symmetric Tree 
def isSymmetric(root):
    def checkTree(t1, t2):
        val1 = t1.val if t1 else None
        val2 = t2.val if t2 else None
        if val1 != val2:
            return False
        elif val1 == None:
            return True
            
        return checkTree(t1.right, t2.left) and checkTree(t1.left, t2.right)
        
    return checkTree(root, root)


#Binary Tree Level Order Traversal

def levelOrder(root):
    if root == None:
        return []
        
    ret = []
    queue = [root, None]
    temp = []
    while len(queue):
        node = queue[0]
        queue = queue[1:]
        
        if node:
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
            temp.append(node.val)
        else:
            ret.append(temp)
            temp = []
            if len(queue):
                queue.append(None)
            else:
                break
            
    return ret

#Maximum Depth of Binary Tree 

def maxDepth(root):
    def findMax(root, cnt):
        if root == None:
            return cnt
    
        cnt += 1
        return max(findMax(root.left, cnt), findMax(root.right, cnt))

    return findMax(root, 0)


#Balanced Binary Tree
def isBalanced(root):
    def check(root):
        if root == None:
            return 0
            
        dleft = check(root.left)
        dright = check(root.right)

        return max(dleft, dright) + 1 if (dleft != -1 and dright != -1 and abs(dleft - dright) <= 1) else -1

    return check(root) != -1

#Minimum Depth of Binary Tree
def minDepth(node):
    if node == None:
        return 0
        
    l = minDepth(node.left)
    r = minDepth(node.right)
    if l == 0:
        return r + 1
    if r == 0:
        return l + 1
    
    return min(l, r) + 1

#Path Sum
def hasPathSum(root, sum):
    if root == None:
        return False
        
    if root.left == None and root.right == None:
        return root.val == sum
        
    sum -= root.val
    return hasPathSum(root.left, sum) or hasPathSum(root.right, sum)

#Pascal's Triangle
def pascalTriangle(numRows):
    if numRows < 1:
        return []
        
    ret = [ [1] ]
    for row in xrange(1, numRows):
        prev = ret[-1]
        newLst = [prev[0]]
        
        for i in xrange(0, len(prev)):
            val = prev[i] + prev[i+1] if i < len(prev) - 1 else prev[i]
            newLst.append(val)
            
        ret.append(newLst)
        
    return ret

#Valid Palindrome 
def isPalindrome(s):
    if len(s) == 0:
        return True
        
    validStr = "0123456789abcdefghijklmnopqrstuvwxyz"
    left = 0
    right = len(s) - 1
    while True:
        while left < len(s) and s[left].lower() not in validStr:
            left += 1
        while right >= 0 and s[right].lower() not in validStr:
            right -= 1
            
        if left > right:
            break
        
        if s[left].lower() != s[right].lower():
            return False
            
        left += 1
        right -= 1
        
    return True

#Excel Sheet Column Title 
# @param {integer} n
# @return {string}
def convertToTitle(n):
    ret = ""
    lookup = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    while n:
        ret = lookup[(n - 1) % 26] + ret
        n = (n - 1) / 26
    return ret

#Majority Element
# @param {integer[]} nums
# @return {integer}
def majorityElement(nums):
    major = nums[0]
    cnt = 0
    
    for n in nums:
        cnt = cnt + 1 if major == n else cnt - 1
        if cnt == 0:
            major = n
            cnt = 1
                
    return major

#Excel Sheet Column Number

# @param {string} s
# @return {integer}
def titleToNumber(s):
    pt = len(s) - 1
    pos = 0
    ret = 0
    while pt >= 0:
        ret += (ord(s[pt]) - ord('A') + 1) * (26 ** pos)
        pt -= 1
        pos += 1
        
    return ret

#Factorial Trailing Zeroes
# @param {integer} n
# @return {integer}
def trailingZeroes(n):
    n = abs(n)
    ret = 0
    while n > 0:
        ret += n / 5
        n = n / 5
    return ret

#Rotate Array
def rotateArray(nums, k):
    def reverse(lst, a, b):
        while a < b:
            lst[a], lst[b] = lst[b], lst[a]
            a += 1
            b -= 1
   
    k = max(k, 0)
    if k == 0 or len(nums) == 0:
        return
    k %= len(nums)
    
    
    reverse(nums, 0, len(nums)-1)
    reverse(nums, 0, k-1)
    reverse(nums, k, len(nums)-1)

#Reverse Bits
def reverseBits(n):
    ret = 0
    for i in xrange(0, 32):
        ret = ret << 1 | (n % 2)
        n = n >> 1
    return ret

#Number of 1 Bits
def hammingWeight(n):
    ret = 0
    while n:
        n &= n-1 
        ret += 1
    return ret

#Intersection of Two Linked Lists
def getIntersectionNode(headA, headB):
        tA = headA
        tB = headB
        while tA != tB:
            tA = tA.next if tA else headB
            tB = tB.next if tB else headA

        return tA

#House Robber
def houseRob(nums):
    prevYes = 0
    prevNo = 0
    for n in nums:
        prevYes, prevNo = prevNo + n, max(prevNo, prevYes)
        
    return max(prevYes, prevNo)

#Remove Linked List Elements 
def removeElements(head, val):
    while head and head.val == val:
        head = head.next
        
    th = head
    while th:
        temp = th.next
        while temp and temp.val == val:
            temp = temp.next
        th.next = temp
        th = th.next
        
    return head

#Count primes
def countPrimes(n):
    if n <= 1:
        return 0
        
    ret = [True for i in xrange(0, n)]
    ret[1] = False
    i = 2
    while i * i < n:
        if ret[i] == False:
            i += 1
            continue
        for j in xrange(i*i, n, i):
            ret[j] = False
            
        i += 1
    cnt = 0
    for i in xrange(1, n):
        if ret[i]:
            cnt += 1
    return cnt


#Isomorphic Strings
def isIsomorphic(s, t):
    if len(s) != len(t):
        return False
        
    mp1 = {}
    mp2 = {}
    for i in xrange(0, len(s)):
        if not mp1.has_key(s[i]):
            mp1[s[i]] = i
        if not mp2.has_key(t[i]):
            mp2[t[i]] = i
            
        if mp1[s[i]] != mp2[t[i]]:
            return False
            
    return True

#Reverse Linked List 
def reverseList1(head):
    pt = None
    while head:
        temp = head.next
        head.next = pt
        pt = head
        head = temp
        
    return pt

def reverseList2(head):
    def reverseImp(head, nxt):
        if head == None:
            return nxt
        temp = head.next
        head.next = nxt
        return reverseImp(temp, head)
    return reverseImp(head, None)

#Contains Duplicate
def containsDuplicate(nums):
    collection = set()
    for n in nums:
        if n in collection:
            return True
        collection.add( n )
        
    return False

#Contains Duplicate II
def containsNearbyDuplicate(nums, k):
    mp = {}
    for i in xrange(0, len(nums)):
        n = nums[i]
        if not mp.has_key(n):
            mp[n] = i
        else:
            diff = i - mp[n]
            if diff <= k:
                return True
            mp[n] = i
            
    return False

#Rectangle Area
def computeArea(A, B, C, D, E, F, G, H):
    left = max(A, E)
    right = max(min(C,G), left)
    bottom = max(B,F)
    top = max(min(D,H), bottom)
    
    return (A-C) * (B-D) + (E-G) * (F-H) - (left-right) * (bottom-top)


#Summary Ranges 
def summaryRanges(nums):
    ranges = []
    for n in nums:
        if not ranges or n > ranges[-1][-1] + 1:
            ranges += [],
        ranges[-1][1:] = n,
    return ['->'.join(map(str, r)) for r in ranges]

#Power of Two
def isPowerOfTwo(n):
    return n > 0 and n & (n-1) == 0


def isPalindromeLinkedList(head):
    def reverse(head):
        temp = None
        while head:
            temp2 = head.next
            head.next = temp
            temp = head
            head = temp2
        return temp

    th, th2 = head, head
    while th2 and th2.next and th2.next.next:
        th2 = th2.next.next
        th = th.next

    th2 = reverse(th)
    th = head
    while th:
        if th.val != th2.val:
            return False
        th = th.next
        th2 = th2.next
    return True

#Lowest Common Ancestor of a Binary Search Tree
def lowestCommonAncestor(root, p, q):
    def find(root, val):
        if root == None:
            return False
        if root.val < val:
            return find(root.right, val)
        elif root.val > val:
            return find(root.left, val)
        
        return True

    if root == None:
        return None

    if p < root.val > q:
        return lowestCommonAncestor(root.left, p, q)
    elif p > root.val < q:
        return lowestCommonAncestor(root.right, p, q)
    else:
        other = p if root.val == q else q
        if find(root, other):
            return root.val

    return None


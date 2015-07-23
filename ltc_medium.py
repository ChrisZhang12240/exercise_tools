#Two Sum
def twoSum(nums, target):
    ret = []
    mp = {}
    for i in xrange(0, len(nums)):
        if not mp.has_key(nums[i]):
            mp[target - nums[i]] = i+1
        else:
            ret = [mp[nums[i]], i+1]
            
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
        
    ret = 1
    for i in xrange(0, len(s)):
        c = s[i]
        if mp.has_key(c) and pt < mp[c]:
            pt = mp[c] + 1
            ret = max(i - mp[c]+1, ret)
        mp[c] = i
    
    return ret

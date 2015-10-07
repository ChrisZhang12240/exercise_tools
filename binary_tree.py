
class Node:
	def __init__(self, val):
		self.left = None
		self.right = None
		self.val = val

class TreeLinkNode:
	def __init__(self, val):
		self.val = val
		self.left = self.right = self.next = None



def createTree(lst):
	def addNode(root, val):
		if root == None:
			return Node(val)
		if val < root.val:
			root.left = addNode(root.left, val)
		elif val > root.val:
			root.right = addNode(root.right, val)
		return root

	if not lst:
		return None

	root = addNode(None, lst[0])
	for i in xrange(1, len(lst)):
		root = addNode(root, lst[i])

	return root

def treeToLinkedTree(tree):
	if tree == None:
		return None
	node = TreeLinkNode(tree.val)
	node.left = treeToLinkedTree(tree.left)
	node.right = treeToLinkedTree(tree.right)
	return node


def createTreeFromStr2( s ):
	
	if len(s) == 0:
		return None

	lst = s.split(',')
			
	root = Node(int(lst[0]))
	lst = lst[1:]
	queue = [root]
	cnt = 0
	pt = 1
	while len(queue) and len(lst):
		if cnt == 2:
			cnt = 0
			queue = queue[1:]
			
		node = queue[0]
		nextNode = None
		if lst[0] != '#':
			nextNode = Node(int(lst[0]))
			queue.append(nextNode)
				
		lst = lst[1:]
		if cnt == 0:
			node.left = nextNode
		elif cnt == 1:
			node.right = nextNode
				
		cnt += 1

	return root

def createRandomTree(cnt):
	if cnt <= 0:
		return None
	lst = []
	while len(lst) < cnt:
		val = random.randint(1, cnt)
		if val not in lst:
			lst.append(val)

	root = Node(lst.pop())
	node = root
	collection = [root]
	while len(lst):
		node = collection[random.randint(0, len(collection)-1)]
		newNode = Node(lst.pop())
		if node.left == None and node.right == None:
			if random.randint(0, 1):
				node.left = newNode
			else:
				node.right = newNode
		else:
			if node.left == None:
				node.left = newNode
			if node.right == None:
				node.right = newNode
			collection.remove(node)
		collection.append(newNode)

	return root

def serialize2(root):
	ret = []
	if root == None:
		return ""

	queue = [root]
	quit = False
	while not quit:
		size = len(queue)
		quit = True
		temp = []
		for i in xrange(0, size):
			node, queue = queue[0], queue[1:]
			if node.val != -1 and (node.left or node.right):
				quit = False

			temp.append('#' if node.val == -1 else '%d' % node.val)
			if node.val == -1:
				continue

			if node.left:
				queue.append(node.left)
			else:
				queue.append(Node(-1))
			if node.right:
				queue.append(node.right)

		ret += temp

	i = len(ret) - 1
	while i > 0 and ret[i] == '#':
		i -= 1

	return ",".join(ret[:i+1])

def serialize(root):
	ret = []
	queue = [root]

	while queue:
		if queue.count(None) == len(queue):
			break
		node = queue.pop(0)
		if node == None:
			ret.append('#')
		else:
			nxtLeft = node.left if node.left else None
			nxtRight = node.right if node.right else None
			queue += [nxtLeft, nxtRight]
			ret.append('%d' % node.val)

	return ','.join(ret)

def createTreeFromStr(s):
	if not s:
		return None
	s = s.split(',')

	queue = [Node( int(s.pop(0)) )]
	root = queue[-1]
	left = True
	while s:
		nxt = s.pop(0)
		node = None if nxt == '#' else Node( int(nxt) )
		if node:
			queue.append(node)
		if left:
			queue[0].left = node
		else:
			queue[0].right = node
			queue.pop(0)
		left = not left

	return root


def inorder(root):
	if not root:
		return []
	return inorder(root.left) + [root.val] + inorder(root.right)

def preorder(root):
	if not root:
		return []
	return [root.val] + preorder(root.left) + preorder(root.right)

def postorder(root):
	if not root:
		return []
	return postorder(root.left) + postorder(root.right) + [root.val]


def sameTree(tree1, tree2):
	val1 = tree1.val if tree1 else None
	val2 = tree2.val if tree2 else None
	if val1 != val2:
		return False
	elif val1 == None:
		return True
	return sameTree(tree1.left, tree2.left) and sameTree(tree1.right, tree2.right)



import random
def test():
	def buildTreeFromPreIn(preorder, inorder):
	    root = Node(preorder.pop(0))
	    stack = [root]
	    prev = None
	    while preorder:
	        if stack and stack[-1].val == inorder[0]:
	            inorder.pop(0)
	            prev = stack.pop()
	        elif prev:
	            prev.right = Node(preorder.pop(0))
	            stack.append(prev.right)
	            prev = None
	        else:
	            newNode = Node(preorder.pop(0))
	            if stack:
	                stack[-1].left = newNode
	            stack.append(newNode)

	    return root

	def verify(tree):
		flip = False
		temp = []
		for i in xrange(30):
			if not flip:
				temp.append(serialize(tree))
			else:
				tree = createTreeFromStr(temp[-1])
			flip = not flip
		if len(set(temp)) != 1:
			raise RuntimeError("tree test failed!! ")

	def verify2(tree):
		flip = False
		temp = []
		for i in xrange(30):
			if not flip:
				temp.append(tree)
			else:
				tree = createTreeFromStr( serialize(temp[-1]) )
			flip = not flip
		
		for i in xrange(1, len(temp)):
			if not sameTree(temp[i], temp[i-1]):
				raise RuntimeError("tree test failed!! ")


	testCase = [
		([1, 2, 3, 6, 4, 5], [3, 2, 1, 4, 6, 5], "1,2,6,3,4,5"),
		([1, 3, 2, 6, 5, 4], [6, 2, 3, 5, 1, 4], "1,3,4,2,5,#,6"),
		([3, 2, 7, 6, 5, 1, 4], [6, 7, 5, 2, 4, 1, 3], "3,2,7,1,6,5,4"),
		([6, 2, 4, 7, 1, 3, 5], [4, 2, 1, 7, 6, 5, 3], "6,2,3,4,7,5,#,1"),
		([4, 5, 3, 2, 1], [3, 2, 5, 4, 1], "4,5,1,3,#,#,2"),
		([5, 2, 1, 4, 6, 3], [2, 4, 1, 5, 3, 6], "5,2,6,#,1,3,4"),
		([4, 3, 2, 7, 1, 6, 5], [3, 4, 1, 6, 7, 2, 5], "4,3,2,#,7,5,1,#,#,6"),
		([1, 6, 4, 2, 3, 5], [4, 6, 5, 3, 2, 1], "1,6,4,2,#,3,5"),
		([2, 1, 7, 3, 5, 6, 4], [1, 2, 3, 7, 5, 6, 4], "2,1,7,#,3,5,#,#,6,#,4"),
		([7, 3, 4, 2, 5, 6, 1], [7, 4, 3, 6, 5, 2, 1], "7,#,3,4,2,#,5,1,6"),
		([3, 1, 2, 4, 6, 5, 7], [1, 3, 2, 6, 4, 5, 7], "3,1,2,#,#,4,6,5,#,#,7"),
		([4, 2, 1, 3, 5], [4, 2, 5, 3, 1], "4,#,2,#,1,3,5"),
		([5, 3, 4, 2, 1], [3, 2, 4, 5, 1], "5,3,1,#,4,#,2"),
		([5, 2, 4, 3, 1, 6], [2, 5, 3, 1, 4, 6], "5,2,4,#,3,6,#,1"),
		([5, 6, 3, 2, 1, 7, 4], [6, 5, 2, 1, 7, 3, 4], "5,6,3,#,2,4,#,1,#,#,7"),
		([1, 2, 5, 4, 3], [5, 2, 4, 1, 3], "1,2,3,5,4"),
		([5, 2, 3, 1, 4], [2, 3, 5, 4, 1], "5,2,1,#,3,4"),
		([1, 3, 6, 4, 5, 2], [4, 6, 5, 3, 2, 1], "1,3,6,2,4,5"),
		([1, 3, 2, 4, 5], [3, 1, 5, 4, 2], "1,3,2,#,4,5"),
		([1, 3, 2, 4, 5], [3, 2, 1, 5, 4], "1,3,4,#,2,5"),
		([5, 2, 3, 1, 4], [5, 4, 1, 3, 2], "5,#,2,3,1,4"),
		([3, 5, 4, 1, 2], [1, 4, 5, 3, 2], "3,5,2,4,#,1"),
		([7, 3, 5, 6, 2, 1, 4], [3, 6, 2, 5, 7, 4, 1], "7,3,1,#,5,4,6,#,#,2"),
		([1, 6, 5, 4, 2, 3], [6, 2, 4, 5, 1, 3], "1,6,3,#,5,#,4,2"),
		([5, 4, 3, 1, 2], [1, 3, 4, 5, 2], "5,4,2,3,#,1"),
	]

	for case in testCase:
		tree = buildTreeFromPreIn(case[0], case[1])
		verify(tree)
		verify2(tree)
	print "step1 done!"
		

	for i in xrange(100):
		tree = createRandomTree(50 + random.randint(0,20))
		verify(tree)
		verify2(tree)
	print "step2 done!"

	print "tree test done!"

		
#test()



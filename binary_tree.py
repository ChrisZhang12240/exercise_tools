
class Node:
	def __init__(self, val):
		self.left = None
		self.right = None
		self.val = val


class Tree:
	def __init__(self):
		self.root = None

	def sameTree(self, other):
		if self.preorderStr() == other.preorderStr() and \
			self.inorderStr() == other.inorderStr():
			return True
		return False

	def findNode(self, node, val):
		if node == None:
			return None
		if node.val == val:
			return node
		elif node.val < val:
			return self.findNode(node.right, val)
		else:
			return self.findNode(node.left, val)

	def addNode(self, node, val):
		if node == None:
			return Node(val)
		if node.val == val:
			return node
		elif node.val < val:
			node.right = self.addNode(node.right, val)
		else:
			node.left = self.addNode(node.left, val)
		return node

	def addChild(self, childVal):
		self.root = self.addNode(self.root, childVal)



	def inorderStr(self):
		def pushToLeft(node, stack):
			while node:
				stack.append(node)
				node = node.left

		ret = ""
		stack = []
		pushToLeft(self.root, stack)
		while len(stack):
			node = stack.pop()
			ret += "%d" % node.val
			pushToLeft(node.right, stack)

		return ret

	def preorderStr(self):
		ret = ""
		stack = [self.root]
		while len(stack):
			node = stack.pop()
			if node.right:
				stack.append(node.right)
			if node.left:
				stack.append(node.left)
			ret += "%d" % node.val
		return ret
		
	def serialize(self):
		if self.root == None:
			return ""
		ret = ""
		queue = [self.root, Node(-1)]
		temp = []
		while len(queue):
			node = queue[0]
			queue = queue[1:]
			if node != None and node.val == -1:
				done = True
				for c in temp:
					if c != '#':
						done = False
						break
				if done:
					break
				
				if len(ret):
					ret += ','
				ret += ','.join(temp)
				temp = []
				if len(queue):
					queue.append(Node(-1))
			else:
				if node == None:
					temp.append('#')
				else:
					queue.append(node.left)
					queue.append(node.right)
					temp.append('%d' % node.val)
		return ret
		

def createTree( lst ):
	tree = Tree()
	for i in lst:
		tree.addChild(i)
	return tree

def createTreeFromStr( s ):
	
	if len(s) == 0:
		return Tree()

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
			
	t = Tree()
	t.root = root
	
	return t

import random
def test():
	for i in xrange(0, 300):
		lst = [random.randint(1, 999) for x in xrange(0, 33)]
		tree = createTree(lst)
		tree2 = createTree(lst)
		tree3 = createTreeFromStr(tree.serialize())
		lst = list(set(lst))
		#lst.sort()
		if "".join(["%d" % i for i in sorted(lst)]) != tree.inorderStr():
			print "heheh", "".join(["%d" % i for i in sorted(lst)]), tree.inorderStr()
		if not tree.sameTree(tree2) or not tree.sameTree(tree3):
			print "not the same tree"

	print "finished!"

#test()

class Node:
	def __init__(self, val):
		self.val = val
		self.next = None

	def __str__(self):
		ret = "%s" % self.val 
		if self.next:
			ret += "," + str(self.next)
		return ret


def createLinkedLst( lst ):
	dummy = Node(-1)
	pt = dummy
	for i in lst:
		pt.next = Node(i)
		pt = pt.next
	return dummy.next
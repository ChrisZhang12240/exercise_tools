import importlib
import random
import binary_tree
import linked_list

test_module = None
def randomString(size):
	s = ""
	for i in xrange(0, size):
		s += chr(random.randint(97,122))
	return s


tests = [
	(testCompareVersion, "def compareVersion(version1, version2):"),
	(testHappyNumber, "def isHappyNumber( n ):"),
	(testInvertTree, "def invertTree(root):"),
	
]	

import os
import sys
import os
import sys

def askForQuit():
	while True:
		try:
			r = raw_input("\nquit? y/n: ")
			if r == 'y':
				exit()
			break
		except KeyboardInterrupt:
			exit()

def reloadModule():
	global test_module
	failed = False
	while True:
		try:
			if failed:
				raw_input("reolad test module failed..\n")
			reload(test_module)
			return
		except KeyboardInterrupt:
			exit()
		except:
			print sys.exc_info()[1]
			failed = True
			continue


def pause(msg, needReload = False):
	try:
		ret = raw_input(msg)
	except:
		exit()

	if needReload:
		reloadModule()

	return ret


def getQuestionCnt():
	while True:
		try:
			cnt = input("Please input number of questions:\n")
			break
		except KeyboardInterrupt:
			exit()
		except:
			continue

	return cnt

def initFuncs(testLst, fileName, cnt):
	cnt = min(len(testLst), cnt)
	f = open(fileName, "a")
	funcs = []
	for i in xrange(0, cnt):
		item = testLst.pop()
		print item[0].__name__
		funcs.append(item[0])
		f.write(item[1])
		f.write("\n\n\n\n\n\n\n\n\n")
	f.close()

	return funcs

def callFunc(func):
	global test_module

	while True:
		try:
			func()
			break
		except KeyboardInterrupt:
			askForQuit()
			reloadModule()
			continue
		except:
			print sys.exc_info()[1]
			pause("Press any keys to try again...\n", True)
			continue


def testFuncs(funcs):

	while True:

		pause("Press any keys when ready....\n", True)

		idx = 0
		while idx < len(funcs):
			callFunc(funcs[idx])
			idx += 1

		if pause("try again? y/n\n") == 'n':
			break



def testEx():
	global test_module
	testLst = []
	while len(tests):
		idx = random.randint(0, len(tests)-1)
		testLst.append( tests.pop(idx) )
	
	tempName = os.path.dirname(os.path.abspath( __file__ ));
	tempName = os.path.join(tempName, "test_temp.py")
	
	f = open(tempName, "w")
	f.close()

	test_module = importlib.import_module('test_temp')


	cnt = getQuestionCnt()
	totalCnt = len(testLst)
	progress = 0
	while len(testLst):
		try:
			funcs = initFuncs(testLst, tempName, cnt)
			testFuncs(funcs)
			progress += len(funcs)
			print "====================(%d/%d)==================" % (progress, totalCnt)
		except KeyboardInterrupt:
			askForQuit()
				
	print "done!"
		
	


def test():
	global test_module
	test_module = importlib.import_module('ltc_easy')
	for test in tests:
		test[0]()
	print "\n\n\n"


test()
testEx()
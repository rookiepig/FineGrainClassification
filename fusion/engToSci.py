#
# File: engToSci.py
# Desc: convert english class names to science name from birdtree.org
# Author: Zhang Kang
# Date: 2013/12/30
#
import csv      # imports the csv module
import re
import difflib

# lcs function
def longest_common_substring(s1, s2):
    m = [[0] * (1 + len(s2)) for i in xrange(1 + len(s1))]
    longest, x_longest = 0, 0
    for x in xrange(1, 1 + len(s1)):
        for y in xrange(1, 1 + len(s2)):
            if s1[x - 1] == s2[y - 1]:
                m[x][y] = m[x - 1][y - 1] + 1
                if m[x][y] > longest:
                    longest = m[x][y]
                    x_longest = x
            else:
                m[x][y] = 0
    return s1[x_longest - longest: x_longest]

# remove all white characters


def urlify(s):
    # Remove all non-word characters (everything except numbers and letters)
    s = re.sub(r"[^\w\s]", '', s)
    # Replace all runs of whitespace with a single dash
    s = re.sub(r"\s+", '', s)
    s = re.sub(r"_+", '', s)
    return s

# main
print 'Hello'

fCls = open('classes.csv', 'rb')
fSci = open('sci.txt', 'w')
fBird = open('bird.csv', 'rb')
# for x in xrange( 1, 3 ) :
#     for y in xrange( 1, 3 ) :
#         print 'x = ', x, ' y = ', y

try:

    csvCls = csv.reader(fCls)
    clsList = []
    for clsRow in csvCls:
        clsList.append(clsRow)
    csvBird = csv.reader(fBird)
    birdList = []
    for birdRow in csvBird:
        birdList.append(birdRow)
    clsRowNum = 0
    for clsRow in clsList:
        # if clsRowNum > 2 :
        #    break
        # print clsRow
        clsName = clsRow[2].lower()
        clsName = urlify(clsName)
        birdRowNum = 0
        maxRatio = 0
        matchSci = ""
        matchEng = ""
        for birdRow in birdList:
            # print birdRow
            # if birdRowNum > 2 :
            #     break
            if birdRowNum > 0:
                engName = birdRow[6].lower()
                engName = urlify(engName)
                sciName = birdRow[1]
                #subStr = longest_common_substring( clsName, engName )
                curRatio = difflib.SequenceMatcher(
                    None, clsName, engName).ratio()
                if curRatio > maxRatio:
                    # print '%-10s <--> %-10s : %-10s' % ( clsName, engName, subStr )
                    # maxLen = len( subStr )
                    maxRatio = curRatio
                    matchEng = engName
                    matchSci = sciName
            birdRowNum += 1
        print '%-10s <--> %-10s (%.2f)' % (clsName, matchEng, maxRatio)
        fSci.write(matchSci)
        fSci.write('\n')
        clsRowNum += 1
finally:
    fBird.close()      # closing
    fCls.close()
    fSci.close()

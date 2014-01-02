#
# File: treeToGroup.py
# Desc: convert tree file to different grouping systems
# Author: Zhang Kang
# Date: 2013/12/30
#
import csv
from dendropy import Tree

# check is this node can be splitted


def IsSplitNode(node):
    if node.is_leaf():
        return False
    else:
        for chldNode in node.child_nodes():
            if chldNode.is_leaf():
                return False
        return True

# read class name and sci name
clsList = []
sciList = []
fCls = open("classes.csv", "rb")
csvCls = csv.reader(fCls)
for row in csvCls:
    clsList.append(row[2])
    sciList.append(row[4])

# read tree
tree = Tree.get_from_stream(open("CUB11.tre"), "nexus")

# generate grouping system
nodeList = [tree.seed_node]
grpSys = []
grpSys.append(nodeList)
GRP_SYS_NUM = 8
for g in xrange(1, GRP_SYS_NUM):
    nodeList = []
    for node in grpSys[g - 1]:
        if IsSplitNode(node):
            for chldNode in node.child_nodes():
                nodeList.append(chldNode)
        else:
            nodeList.append(node)
    grpSys.append(nodeList)

# 3 classes are not found in the tree
misCls = [51, 130, 136]
# generate matlab script
print "grp=cell(1,", GRP_SYS_NUM, ");"
for g in xrange(0, len(grpSys)):
    print "grp{", g + 1, "}.cluster=cell(1,", len(grpSys[g]) + 1, ");"
    clusterNum = 1
    for node in grpSys[g]:
        idxList = []
        for leafNode in node.leaf_nodes():
            idxList.append(sciList.index(leafNode.taxon.label) + 1)
        print "grp{", g + 1, "}.cluster{", clusterNum, "}=", idxList, ";"
        clusterNum += 1
    print "grp{", g + 1, "}.cluster{", clusterNum, "}=", misCls, ";"

fCls.close()


# print( clsList[ 1 : 3 ] )
# print( sciList[ 1 : 3 ] )
# print( clsList.index( 'Sooty_Albatross' ) )

# node = tree.find_node_with_taxon_label('Caprimulgus vociferus')
# print(node.description())
# print(node.level())

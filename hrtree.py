import nodes

global root

Bvalue = 4
num_of_category = 6


def handleOverFlow(node):
    global root

    # split node into two new nodes
    clus = node.split()

    #print([x.childList for x in clus])
    #print([x.range for x in clus])

    # if root node is overflow, new root need to build
    if node.paren == None:
        root = nodes.Branch(Bvalue, num_of_category, node.level + 1, clus[0])
        root.addChild(clus[0])
        root.addChild(clus[1])
        root.childList[0].paren = root
        root.childList[1].paren = root
    else:
        # update the parent node
        parent = node.paren
        parent.childList.remove(node)
        parent.childList += clus
        # check whether parent node is overflow
        if parent.isOverFlow():
            handleOverFlow(parent)


# insert a point to a node
def insert(node, point):
    # if the node is a leaf, add this point
    if isinstance(node, nodes.Leaf):
        node.addChild(point)
        if node.isOverFlow():
            handleOverFlow(node)

    # if the node is a branch, choose a child to add this point
    elif isinstance(node, nodes.Branch):
        node.update(point)
        childNode = node.chooseChild(point)
        insert(childNode, point)

    else:
        pass



# check all nodes and points in a r-tree
def checktree(nodes):
    checkBranch(nodes)
    print('Finished checking R-tree')


# check the correctness of a leaf node in r-tree
def checkLeaf(leaf):
    # check whether a point is inside of a leaf
    def insideLeaf(x, y, parent):
        if x < parent[0] or x > parent[1] or y < parent[2] or y > parent[3]:
            return False
        else:
            return True

    # general check
    checkNode(leaf)
    # check whether each child point is inside of leaf's range
    for point in leaf.childList:
        if not insideLeaf(point.x, point.y, leaf.range):
            print('point(', point.x, point.y, 'is not in leaf range:', leaf.range)


# check the correctness of a branch node in r-tree
def checkBranch(branch):
    # check whether a branch is inside of another branch
    def insideBranch(child, parent):
        if child[0] < parent[0] or child[1] > parent[1] or child[2] < parent[2] or child[3] > parent[3]:
            return False
        else:
            return True

    # general check
    checkNode(branch)
    # check whether child's range is inside of this node's range
    for child in branch.childList:
        if not insideBranch(child.range, branch.range):
            print('child range:', child.range, 'is not in node range:', branch.range)
        # check this child
        if isinstance(child, nodes.Branch):
            # if child is still a branch node, check recursively
            checkBranch(child)
        elif isinstance(child, nodes.Leaf):
            # if child is a leaf node
            checkLeaf(child)


# general check for both branch and leaf node
def checkNode(node):

    length = len(node.childList)
    # check whether is empty
    if length == 0:
        print('empty node. node level:', node.level, 'node range:', node.range)
    # check whether overflow
    if length > Bvalue:
        print('overflow. node level:', node.level, 'node range:', node.range)

    # check whether the centre is really in the centre of the node's range
    r = node.range
    if (r[0] + r[1]) / 2 != node.centre[0] or (r[2] + r[3]) / 2 != node.centre[1]:
        print('wrong centre. node level:', node.level, 'node range:', node.range)
    if r[0] > r[1] or r[2] > r[3]:
        print('wrong range. node level:', node.level, 'node range:', node.range)


def isIntersect(range1, range2):
    if range1[0]>range2[1] or range1[1]<range2[0] or range1[2]>range2[3] or range1[3]<range2[2]:
        return False
    else:
        return True


def range_query(q_p, distance, root, collected):
    res = [1 for __ in range(num_of_category)]
    q = q_p
    dist = distance
    collected = collected

    def __helper(mbr):
        nonlocal res
        # If whole mbr in q + distance
        if q[0] - dist <= mbr.range[0] and q[1] - dist <= mbr.range[2] and \
                q[0] + dist >= mbr.range[1] and q[1] + dist >= mbr.range[3]:
            res = [res[i] * mbr.zeta[i] for i in range(num_of_category)]
            if mbr.ps.intersection(collected):
                print("Intersect", mbr.ps.intersection(collected))
        else:
            for each_child in mbr.childList:
                # If current mbr is Leaf-node and all its children would be Points
                if isinstance(each_child, nodes.Tree_Point):
                    if (q[0] - each_child.x) * (q[0] - each_child.x) + \
                            (q[1] - each_child.y) * (q[1] - each_child.y) <= dist * dist and \
                            each_child.id not in collected:
                        res = [res[i] * (1 - each_child.score[i]) for i in range(num_of_category)]
                else:
                    if isIntersect(each_child.range, [q[0] - dist, q[0] + dist, q[1] - dist, q[1] + dist]):
                        __helper(each_child)

    __helper(root)

    return [1-i for i in res]


def main():

    global root

    t_point = nodes.Tree_Point([[0.1, 0.2, 0.3, 0.3, 0, 0.1], 1, 2, 'o'])

    root = nodes.Leaf(Bvalue, num_of_category, 1, t_point)

    root.addChild(t_point)

    insert(root, nodes.Tree_Point([[0.1, 0.2, 0.3, 0.3, 0, 0.1], 2, 4, 'a']))
    insert(root, nodes.Tree_Point([[0.1, 0.2, 0.3, 0.3, 0, 0.1], 5, 6, 'b']))
    insert(root, nodes.Tree_Point([[0.1, 0.2, 0.3, 0.3, 0, 0.1], 9, 5, 'c']))
    insert(root, nodes.Tree_Point([[0.1, 0.2, 0.3, 0.3, 0, 0.1], 10, 20, 'd']))
    insert(root, nodes.Tree_Point([[0.1, 0.2, 0.3, 0.3, 0, 0.1], 1, 3, 'e']))

    checktree(root)

    print([x.ps for x in root.childList])

    print(range_query((0, 0), 8, root, set('b')))


if __name__ == "__main__":
    main()

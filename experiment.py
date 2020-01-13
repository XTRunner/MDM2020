from lda_learner import places_reviews
import os, csv, pickle, random
from gensim.test.utils import datapath
from gensim import models
import greedy_search

import matplotlib.pyplot as plt
from matplotlib import collections as mc

from geopy.distance import geodesic

import graph_construct

import nodes

global root
Bvalue = 10
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
    print('Finished checking HR-tree')


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


def isIntersect(mbr_range, q_p, dist):
    if (mbr_range[0] > q_p[0] and geodesic((q_p[1], q_p[0]), (q_p[1], mbr_range[0])).m > dist) or \
            (mbr_range[1] < q_p[0] and geodesic((q_p[1], mbr_range[1]), (q_p[1], q_p[0])).m > dist) or \
            (q_p[1] < mbr_range[2] and geodesic((q_p[1], q_p[0]), (mbr_range[2], q_p[0])).m > dist) or \
            (q_p[1] > mbr_range[3] and geodesic((mbr_range[3], q_p[0]), (q_p[1], q_p[0])).m > dist):
        return False
    else:
        return True


def range_query(q_p, distance, root, collected, place_feature):
    res = [1 for __ in range(num_of_category)]
    q = q_p
    dist = distance
    collected = collected
    place_feature = place_feature

    def __helper(mbr):
        nonlocal res
        # geodesic distance (lat, lng)
        '''
        One example:
                                mbr.range[1], mbr.range[3]
                |-------------------------|
                |                         |
                |                         |
                |                         |
                |            o            |
                |      (q[0], q[1])       |
                |                         |
                |                         |
                |-------------------------|
        mbr.range[0], mbr.range[2]
        '''
        if geodesic((q[1], q[0]), (q[1], mbr.range[0])).m <= dist and \
                geodesic((q[1], q[0]), (q[1], mbr.range[1])).m <= dist and \
                geodesic((q[1], q[0]), (mbr.range[2], q[0])).m <= dist and \
                geodesic((q[1], q[0]), (mbr.range[3], q[0])).m <= dist:
            res = [res[i] * mbr.zeta[i] for i in range(num_of_category)]
            if mbr.ps.intersection(collected):
                remove_ps = mbr.ps.intersection(collected)
                remove_res = [1 for _ in range(num_of_category)]
                for each_p in remove_ps:
                    p_feature = place_feature[each_p]
                    remove_res = [remove_res[i] * (1 - p_feature[i]) for i in range(num_of_category)]
                res = [res[i]/remove_res[i] for i in range(num_of_category)]
        else:
            for each_child in mbr.childList:
                # If current mbr is Leaf-node and all its children would be Points
                if isinstance(each_child, nodes.Tree_Point):
                    if geodesic((each_child.y, each_child.x), (q[1], q[0])).m <= dist and \
                            each_child.id not in collected:
                        res = [res[i] * (1 - each_child.score[i]) for i in range(num_of_category)]
                else:
                    if isIntersect(each_child.range, q, dist):
                        __helper(each_child)

    __helper(root)

    return [1-i for i in res]


def visual_res(ns, es, res_path):
    '''
    :param ns: Nodes in road graph {id1: {'lng': -74, 'lat': 41, 'sites':set(['Park, NY, USA', ...])}, ...}
    :param es: Edges in road graph {(id1, id2): 20.1, ...}
    :param res_path: [id1, id2, ...]
    :return:
    '''
    m_x, m_y = [], []
    poi_node_x, poi_node_y = [], []
    for each_n, each_corr in ns.items():
        if each_corr['sites']:
            poi_node_x.append(each_corr['lng'])
            poi_node_y.append(each_corr['lat'])
        else:
            m_x.append(each_corr['lng'])
            m_y.append(each_corr['lat'])

    lines = []
    for each_n, each_corr in es.items():
        # lines = [[(0, 1), (1, 1)], [(2, 3), (3, 3)], [(1, 2), (1, 3)]]
        lines.append([(ns[each_n[0]]['lng'], ns[each_n[0]]['lat']),
                      (ns[each_n[1]]['lng'], ns[each_n[1]]['lat'])])

    lc = mc.LineCollection(lines)
    fig, ax = plt.subplots()
    ax.add_collection(lc)

    plt.plot(m_x, m_y, 'ro')

    plt.plot(poi_node_x, poi_node_y, 'bo', markersize=12)

    #plt.show()
    if res_path:
        plt.plot(ns[res_path[0]]['lng'], ns[res_path[0]]['lat'], 'Dm', markersize=12)

    for i in range(len(res_path)-1):
        print_path_x, print_path_y = [], []

        print_path_x.append(ns[res_path[i]]['lng'])
        print_path_x.append(ns[res_path[i+1]]['lng'])

        print_path_y.append(ns[res_path[i]]['lat'])
        print_path_y.append(ns[res_path[i + 1]]['lat'])

        plt.plot(print_path_x, print_path_y, 'g-', linewidth=5)

    plt.show()


def main():
    place_name = 'New_York_city'
    # Memphis, New_York_city
    state_name = 'New York'
    # Tennessee, New York

    if os.path.exists('LDA_model/' + place_name + '_cleaned_review.csv'):
        print("Start Loading Cleaned Data...")
        with open('LDA_model/' + place_name + '_cleaned_review.csv', 'r') as handle:
            rfile = csv.reader(handle, delimiter='|')
            reviews = {}

            for each_row in rfile:
                reviews[each_row[0]] = reviews.get(each_row[0], []) + [each_row[1]]
    else:
        reviews = places_reviews('trip_advisor_crawler/tripadvisor_attractions_' + place_name + '.csv')

        with open('LDA_model/' + place_name + '_cleaned_review.csv', 'a', newline='') as whandle:
            spamwriter = csv.writer(whandle, delimiter='|')

            for k, v in reviews.items():
                for each_review in v:
                    spamwriter.writerow([k, each_review])

    print('--------------------------------')

    temp_file = datapath("C:/Users/xuteng/PycharmProjects/MDM20/LDA_model/lda_trained_model")
    lda_model = models.LdaModel.load(temp_file)

    dict_w = lda_model.id2word

    osm_g = graph_construct.PoI_Graph(" ".join(place_name.split("_")) + ', ' + state_name + ', USA', 'drive')

    pois_lng_lat = pickle.load(open(place_name + '_attractions_lng_lat.pkl', 'rb'))

    ns, es = osm_g.poi_overlay(pois_lng_lat)

    tree_flag = 0

    global root

    print('Start Numerating Reviews by LDA and Construct HR-tree')

    poi_score = {}

    for k, v in ns.items():
        if v['sites']:
            for each_p in v['sites']:
                p_lng, p_lat = v['lng'], v['lat']

                #p_reviews = reviews[each_p.replace(', ' + place_name + ', USA', '')]

                p_reviews = reviews[each_p.replace(', New York, USA', '')]

                corpus = dict_w.doc2bow((" ".join(p_reviews)).split())

                lda_score = [x[1] for x in lda_model[corpus]]
                poi_score[each_p] = lda_score

                if tree_flag == 0:
                    point = nodes.Tree_Point([lda_score, p_lng, p_lat, each_p])
                    root = nodes.Leaf(Bvalue, num_of_category, 1, point)
                    root.addChild(point)

                    tree_flag += 1
                else:
                    insert(root, nodes.Tree_Point([lda_score, p_lng, p_lat, each_p]))
                    tree_flag += 1

                print('Done with ', tree_flag, 'out of', len(reviews))

    checktree(root)

    print('HR-tree has been built. Highest level is:', root.level)

    init_node = 40000

    # root, init_node, d_range, ns, es, place_feature, k
    res_node = greedy_search.greedy_search_alg(root, init_node, 3000, ns, es, poi_score, 3, graph_f=True)

    visual_res(ns, es, res_node.print_path())


if __name__ == '__main__':
    main()
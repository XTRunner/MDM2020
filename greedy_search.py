import heapq
import itertools
import experiment


class Search_Node:
    def __init__(self, node_id, priority, d_to_start, res=None, parent=None):
        '''
        :param node_id: ID of node (same in road network)
        :param priority: Priority (Heuristic value) of node
        :param d_to_start: The distance from start to current node
        :param res: The result SET gotten until now
        :param parent: Previous node of current node
        '''
        if not res:
            res = set()
        self.id = node_id
        self.priority = priority
        self.d_to_start = d_to_start
        self.parent = parent
        self.res = res

    def get_node(self):
        return self.id

    def print_path(self):
        print("The solution (Start -> End) has been found: ")

        path, path_n = [], self

        while path_n:
            path.append(path_n.get_node())
            path_n = path_n.parent

        path.reverse()

        print("----------------------------")
        print(" -> ".join([str(x) for x in path]))
        print("----------------------------")

        return path

    def __lt__(self, other):
        return (self.priority, -self.d_to_start) < (other.priority, -other.d_to_start)


class Priority_Queue:
    def __init__(self, order='min', f=lambda x: x):
        self.heap = []
        # Default order of heapq is ascending
        if order == 'min':
            self.f = f
        elif order == 'max':
            self.f = lambda x: -f(x)
        else:
            raise ValueError("Queue Order is either 'min' or 'max'!")

    def append(self, item):
        heapq.heappush(self.heap, (self.f(item.priority), item))

    def pop(self):
        if self.heap:
            return heapq.heappop(self.heap)[1]
        else:
            raise Exception("Queue has already been empty!")

    def __delitem__(self, key):
        try:
            del self.heap[[item.get_node() == key for _, item in self.heap].index(True)]
        except ValueError:
            raise KeyError("Delete Fail! Cannot find key")

        heapq.heapify(self.heap)

    def __contains__(self, key):
        return any([item.get_node() == key for _, item in self.heap])

    def __getitem__(self, key):
        for val, item in self.heap:
            if item.get_node() == key:
                return val
        raise KeyError(str(key) + " cannot be found in Priority Queue")

    def __len__(self):
        return len(self.heap)


def div_score(res_set, category_num=6):
    '''
    Example:
    :param res_set: [[0.1, 0.2, ..., 0.2], [0.8, 0.7, ..., 0.1], ...]
    :param category_num: Default 6. Only useful if res_set is empty
    :return: Diversity score [0.34, 0.21, ..., 0.11]
    '''

    if not res_set:
        return [0]*category_num
    else:
        category_num = len(res_set[0])

        res = [0] * category_num

        for i in range(category_num):
            tmp_product = 1

            for each_res in res_set:
                tmp_product *= (1 - each_res[i])

            res[i] = 1 - tmp_product

        return res


def swap_res(cur_res, pois, place_feature, k):
    '''
    Example:
    :param cur_res: set(['Park, NY, USA', 'Zoo, NY, USA', ...])
    :param pois: New Sites. set(['Square, NY, USA', ...])
    :param place_feature: Hashtable {'Park, NY, USA': [0.1, 0.2, ...], 'Zoo, NY, USA': [0, 0.8, ...], ...}
    :param k: Limit of result set
    :return: set(['Zoo, NY, USA', 'Square, NY, USA', ...])
    '''
    total_set = cur_res | pois

    res = set()

    # Get all possible combination (k out of k+1)
    subset_idxs = list(itertools.combinations({x for x in range(k+1)}, k))

    for each_poi in total_set:
        if len(res) < k:
            res.add(each_poi)
        else:
            res.add(each_poi)

            res_list = list(res)

            max_score = float('-inf')

            for each_comb in subset_idxs:
                score_list = [place_feature[res_list[idx]] for idx in each_comb]

                if max_score < sum(div_score(score_list)):
                    max_score = sum(div_score(score_list))
                    res = set([ res_list[idx] for idx in each_comb ])

    return res


def adj_priority_calculate(res_list, potential_feature, k):
    '''
    Example
    :param res_list: [[0.1, 0.2, ..., 0.2], [0.8, 0.7, ..., 0.1], ...]
    :param potential_feature: [0.75, 0.5, ...]
    :param k: Limit of result set
    :return: Max possible Div score
    '''
    total_list = [x for x in res_list]
    total_list.append(potential_feature)

    if len(total_list) <= k:
        return sum(div_score(total_list))
    else:
        subset_idxs = list(itertools.combinations({x for x in range(k + 1)}, k))

        max_score = float('-inf')

        for each_comb in subset_idxs:
            score_list = [total_list[idx] for idx in each_comb]

            max_score = max(max_score, sum(div_score(score_list)))

        return max_score


def goal_test(res_set, k):
    # The div score cannot be larger than k or number of category. If already got there, then stop earlier
    if not res_set:
        return False
    else:
        if sum(div_score(res_set)) >= min(k, len(res_set[0])):
            return True

    return False


def max_res(new_node, place_feature, cur_res, cur_res_score):
    if cur_res_score < sum(div_score([place_feature[x] for x in new_node.res])):
        cur_res_score = sum(div_score([place_feature[x] for x in new_node.res]))
        cur_res = new_node

    return cur_res, cur_res_score


def greedy_search_alg(root, init_node, d_range, ns, es, place_feature, k, graph_f=True):
    '''
    :param root: HR-tree
    :param init_node: Starting node -- id
    :param d_range: Distance limit
    :param ns: Nodes in road graph {id1: {'lng': -74, 'lat': 41, 'sites':set(['Park, NY, USA', ...])}, ...}
    :param es: Edges in road graph {(id1, id2): 20.1, ...}
    :param place_feature: Hashtable {'Park, NY, USA': [0.1, 0.2, ...], 'Zoo, NY, USA': [0, 0.8, ...], ...}
    :param k: Limit of result set
    :param graph_f: Graph search or Tree search
    :return:
    '''

    # Priority queue waiting for exploration
    frontier = Priority_Queue(order='max')

    # returned results
    res_res, max_res_score = None, float('-inf')

    if ns[init_node]['sites']:
        # node_id, priority, d_to_start, res=None, parent=None
        start_node = Search_Node(init_node, 0, 0, res=swap_res(set(), ns[init_node]['sites'], place_feature, k))
        print("Found Site(s)!!!")
        res_res, max_res_score = max_res(start_node, place_feature, res_res, max_res_score)
    else:
        start_node = Search_Node(init_node, 0, 0)

    frontier.append(start_node)

    # All the visited node
    if graph_f:
        explored = set()

    while frontier:
        print("Awaiting unexplored nodes", len(frontier))
        print("Current Diversity", max_res_score)
        cur_node = frontier.pop()

        if goal_test([place_feature[x] for x in cur_node.res], k):
            print("///////////////////////////////////////////")
            print("Stop Earlier! Already found the MAX diversity")
            print(cur_node.res)
            cur_node.print_path()
            return cur_node

        if graph_f:
            explored.add(cur_node.get_node())

        # Get all adjacent nodes of current node
        adj_nodes = set([k[1] for k in es if k[0] == cur_node.get_node()])

        update_f = False

        for each_adj_n in adj_nodes:
            if graph_f and each_adj_n in explored:
                update_f = True
                continue

            used_d = cur_node.d_to_start + es[(cur_node.get_node(), each_adj_n)]

            # Next adjacent node is within distance range
            if used_d <= d_range:
                potential_div = experiment.range_query((ns[each_adj_n]['lng'], ns[each_adj_n]['lat']),
                                                       d_range-used_d,
                                                       root,
                                                       cur_node.res,
                                                       place_feature)

                cur_div = div_score([place_feature[x] for x in cur_node.res])

                # If the further diversity cannot contribute more, then no need to explore this direction
                if any([cur_div[i] < potential_div[i] for i in range(len(potential_div))]):
                    adj_priority = adj_priority_calculate([place_feature[x] for x in cur_node.res],
                                                          potential_div,
                                                          k)

                    if adj_priority > max_res_score:
                        if ns[each_adj_n]['sites']:
                            adj_res = swap_res(cur_node.res, ns[each_adj_n]['sites'], place_feature, k)
                            print("Found Site(s)!!!")
                        else:
                            adj_res = cur_node.res

                        print("---------------------------------------")
                        print("Next direction: ", cur_node.get_node(), '->', each_adj_n,
                              'with potential', potential_div)

                        # node_id, priority, d_to_start, res=None, parent=None
                        next_node = Search_Node(each_adj_n, adj_priority, used_d, res=adj_res, parent=cur_node)

                        # Check explored list earlier
                        '''
                        if graph_f:
                            if not next_node.get_node() in frontier:
                                frontier.append(next_node)
                                #res_res, max_res_score = max_res(next_node, place_feature, res_res, max_res_score)
                            elif next_node.get_node() in frontier:
                                if adj_priority > frontier[next_node.get_node()]:
                                    del frontier[next_node.get_node()]
                                    frontier.append(next_node)
                                    #res_res, max_res_score = max_res(next_node, place_feature, res_res, max_res_score)
                        else:
                            if not next_node.get_node() in frontier:
                                frontier.append(next_node)
                                #res_res, max_res_score = max_res(next_node, place_feature, res_res, max_res_score)
                            elif next_node.get_node() in frontier:
                                if adj_priority > frontier[next_node.get_node()]:
                                    del frontier[next_node.get_node()]
                                    frontier.append(next_node)
                                    #res_res, max_res_score = max_res(next_node, place_feature, res_res, max_res_score)
                        '''

                        if not next_node.get_node() in frontier:
                            frontier.append(next_node)
                            # res_res, max_res_score = max_res(next_node, place_feature, res_res, max_res_score)
                        elif next_node.get_node() in frontier:
                            if adj_priority > frontier[next_node.get_node()]:
                                del frontier[next_node.get_node()]
                                frontier.append(next_node)
                                # res_res, max_res_score = max_res(next_node, place_feature, res_res, max_res_score)
                else:
                    '''
                    if max_res_score < sum(cur_div):
                        max_res_score = sum(cur_div)
                        res_res = cur_node
                    '''
                    update_f = True
            else:
                '''
                if max_res_score < sum(div_score([place_feature[x] for x in cur_node.res])):
                    max_res_score = sum(div_score([place_feature[x] for x in cur_node.res]))
                    res_res = cur_node
                '''
                update_f = True

        if update_f:
            if max_res_score < sum(div_score([place_feature[x] for x in cur_node.res])):
                max_res_score = sum(div_score([place_feature[x] for x in cur_node.res]))
                res_res = cur_node

        if max_res_score >= min(k, len(list(place_feature.values())[0])):
            print("///////////////////////////////////////////")
            print("Stop Earlier! Already found the MAX diversity")
            print(res_res.res)
            res_res.print_path()
            return res_res

    print("///////////////////////////////////////////")
    print("Explored graph and found the path with MAX diversity", max_res_score)
    print(res_res.res)
    res_res.print_path()
    return res_res








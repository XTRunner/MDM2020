import pickle
import matplotlib.pyplot as plt
from matplotlib import collections  as mc
import graph_construct


def main():

    osm_g = graph_construct.PoI_Graph('New York City, New York, USA', 'drive')

    pois = pickle.load(open('NY_attractions_lng_lat.pkl', 'rb'))

    p_x, p_y = [], []
    for each_n, each_corr in osm_g.node_col.items():
        p_x.append(each_corr['lng'])
        p_y.append(each_corr['lat'])
    '''
    lines = []
    for each_n, each_corr in osm_g.edge_col.items():
        #lines = [[(0, 1), (1, 1)], [(2, 3), (3, 3)], [(1, 2), (1, 3)]]
        lines.append([(osm_g.node_col[each_n[0]]['lng'], osm_g.node_col[each_n[0]]['lat']),
                      (osm_g.node_col[each_n[1]]['lng'], osm_g.node_col[each_n[1]]['lat'])])

    lc = mc.LineCollection(lines)
    fig, ax = plt.subplots()
    ax.add_collection(lc)
    '''


    poi_x, poi_y = [], []
    for each_n, each_corr in pois.items():
        poi_x.append(each_corr[0])
        poi_y.append(each_corr[1])

    plt.plot(p_x, p_y, 'ro')
    plt.plot(poi_x, poi_y, 'go')

    plt.show()

    ns, es = osm_g.poi_overlay(pois)

    m_x, m_y = [], []
    for each_n, each_corr in ns.items():
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

    plt.show()


if __name__ == '__main__':
    main()
import pickle
import osmnx as ox
import matplotlib.pyplot as plt


def map_osm(place, network_type):

    try:
        graph = ox.graph_from_place(place,
                                    network_type=network_type,
                                    truncate_by_edge=True,
                                    simplify=True)
    except:
        graph = ox.graph_from_place(place,
                                    network_type=network_type,
                                    which_result=2,
                                    truncate_by_edge=True,
                                    simplify=True)

    return graph


def main():
    osm_g = map_osm('Memphis, Tennessee, USA', 'drive')

    fig, ax = ox.plot_graph(osm_g, show=False, close=False)

    pois = pickle.load(open("Memphis_attractions_lng_lat.pkl", "rb"))

    lng_lat_s = pois.values()

    for lng_lat in lng_lat_s:
        ax.scatter(lng_lat[0], lng_lat[1], c='red', s=50)

    plt.show()



if __name__ == '__main__':
    main()

import data_collect
import hrtree
import pickle

def main():
    ns = pickle.load(open("London_overlay_node.pkl", "rb"))
    es = pickle.load(open("London_overlay_edge.pkl", "rb"))
    pois = pickle.load(open("poi_pos.pkl", "rb"))
    '''
    ns: {
            id: (lat, lng)
        }
    
    es: {
            (id1, id2): length
        }
        
    pois: {
            id: [PoI1, ...]
        }
        PoI.coord - (lat, lng)      PoI.texts - "..."       PoI.pd - [0,0,...,0]
    '''

    ### 127832 nodes, 320584 edges (directed), 3617 PoIs
    ### Every PoI has at least 8 words for textual description




if __name__ == '__main__':
    main()
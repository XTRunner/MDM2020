import csv
from geopy.geocoders import Nominatim, ArcGIS
import pickle
from geopy.exc import GeocoderTimedOut

city = 'Chicago, USA'


def get_lng_lat(geolocator, address):
    try:
        return geolocator.geocode(address, timeout=10)
    except GeocoderTimedOut:
        return get_lng_lat(geolocator, address)


def main():
    places = set()
    with open("trip_advisor_crawler/tripadvisor_attractions_Chicago.csv", "r", encoding='utf-8') as handle:
        rfile = csv.reader(handle)

        fields = next(rfile)

        idx = fields.index('place')

        for each_row in rfile:
            place = each_row[idx] + ', ' + city
            places.add(place)

    #print(len(places))

    res = {}
    osm_geolocator = Nominatim(user_agent="ta_mdm")
    arcgis_geolocator = ArcGIS()
    
    for each_one in places:
        print(each_one.encode("utf-8"))
        location = get_lng_lat(osm_geolocator, each_one.encode("utf-8"))
        if location:
            res[each_one] = (location.longitude, location.latitude)
        else:
            location = get_lng_lat(arcgis_geolocator, each_one.encode("utf-8"))
            if location:
                res[each_one] = (location.longitude, location.latitude)
            else:
                print("Not Found")

        print(res[each_one])
        print("Done: ", len(res), " / ", len(places))

    pickle.dump(res, open("Chicago_attractions_lng_lat.pkl", "wb"))


if __name__ == '__main__':
    main()
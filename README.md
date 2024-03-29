# MDM2020

Implementation in Python 3.5 of MDM 2020 work. Please check requirements.txt for package dependencies

Major Codes: 

-- hrtree.py: Diversity Aggregated R-Tree. Besides, nodes.py defines the node class in DAR-Tree

-- graph_construct.py: Build PoI Network from Road Network and PoI database by using R-Tree

-- greedy_search.py: Include NSS-kDPQ, ESS-kDPQ, Dijkstra's algorithm (by using priority queue) and Random Walk with Restart

-- lda_learner.py: Perform Natrual Language Processing, including clean the raw text and train Latent topic based model

-- experiment.py: Produce the experimental results in MDM 2020 paper

-- trip_advisor_crawler/trip_advisor_crawler/spiders/trip_advisor_attraction_review_NY.py: Web HTML crawler for collecting reviews information of attratctions from TripAdvisor (https://www.tripadvisor.com)

Major Data:

-- LDA_model/lda_trained_model: Trained learning model used in MDM 2020 paper

-- LDA_model/train_cleaned_text.csv: Dataset for training the model

-- LDA_model/xxx_cleaned_review.csv: Cleaned/Stemmed reviews of attractions in xxx (city name)

-- experiment_related: All the randomly picked query points in each city (for reproducing experimental results)

Note: If you are interested in using TripAdvisor web crawler or attraction reviews dataset proposed in this work, please consider cite this work, https://ieeexplore.ieee.org/document/9162294

X. Teng, G. Trajcevski, J. Kim and A. Züfle, "Semantically Diverse Path Search," 2020 21st IEEE International Conference on Mobile Data Management (MDM), Versailles, France, 2020, pp. 69-78, doi: 10.1109/MDM48529.2020.00028.

If you have any question regarding this work, please feel free to reach out to me through xuteng@iastate.edu. Thanks for your interest:)

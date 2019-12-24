import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import graphviz 
from sklearn import tree, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score




# def main():
#     df = pd.read_csv("youtube-new/USvideos.csv", escapechar="\\",skipinitialspace= True)
    
#     s_plot = pd.DataFrame(columns=['X_data', 'Y_data'])

#     #PREPROCESSING
#     df[["val1", "val2", "val3"]] = df["publish_time"].str.split(pat = "-", expand=True)
#     df[["time1", "time2", "time3"]] = df["trending_date"].str.split(pat = ".", expand=True)
#     df["days_to_trend"] = df["time2"].astype('int32') - df["val2"].astype('int32')
#     df = df.drop(columns=["val1", "val2", "val3", "time1", "time2", "time3"])
    
#     column_name = df.columns[7:11]
#     train_attr, test_attr, train_tar, test_tar = train_test_split(df.loc[0:,column_name],df["days_to_trend"],test_size=0.5)
#     clf = tree.DecisionTreeClassifier()
    
#     # train_clf = 
#     # pred_clf = train_clf.predict(test_attr)
#     # acc_score = accuracy_score(test_tar, pred_clf)
    
#     tree.plot_tree(clf.fit(train_attr, train_tar))
#     dot_data = tree.export_graphviz(clf, out_file=None) 
#     graph = graphviz.Source(dot_data) 
#     graph.render("youtube_data") 

def main():
    us_videos = pd.read_csv("youtube-new/USvideos.csv", escapechar="\\",skipinitialspace= True)
    us_videos_categories = pd.read_json('../input/US_category_id.json')
    
    


if __name__ == "__main__":
    main()
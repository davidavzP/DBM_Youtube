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
    us_videos = pd.read_csv("youtube-new/USvideos.csv")
    us_videos_categories = pd.read_json('youtube-new/US_category_id.json')
    us_videos.category_id = us_videos.category_id.astype('category')
    
    #converting tending date to datetieme format
    us_videos['trending_date'] = pd.to_datetime(us_videos['trending_date'], format='%y.%d.%m').dt.date
    us_videos.trending_date.value_counts().sort_index(inplace=True)
    # print(us_videos.head().to_string())

    #converting puplish_time to datetime format
    publish_time = pd.to_datetime(us_videos.publish_time, format='%Y-%m-%dT%H:%M:%S.%fZ')
    us_videos['publish_date'] = publish_time.dt.date

    #get days trending
    us_videos['days_to_trending'] = (us_videos.trending_date - us_videos.publish_date).dt.days
    us_videos.drop(['publish_time','title','channel_title','publish_date', 'thumbnail_link', 'tags', 'trending_date', 'description'], axis=1, inplace=True)

    # print(us_videos.columns)
    # print(us_videos.head().to_string())
    # print(us_videos.describe(percentiles=[.05,.25,.5,.75,.95]).to_string())


    #create index
    # us_videos.set_index(['days_to_trending', 'video_id'], inplace=True)
    # print(us_videos.head())

    #dislike percentage
    us_videos['dislike_percentage'] = us_videos['dislikes'] / (us_videos['dislikes'] + us_videos['likes'])
    us_videos['dislike_percentage'] = us_videos['dislike_percentage'].astype('float32')
    # print(us_videos.dislike_percentage.describe(percentiles=[.05,.25,.5,.75,.95]))

    #clean up
    us_videos = us_videos[~us_videos.video_error_or_removed]
    us_videos.drop(['video_error_or_removed'], axis=1, inplace=True)
    print(us_videos.info())

    column_name = us_videos.columns[1:5].to_list()
    # column_name.append(us_videos.columns[9])
    clf = tree.DecisionTreeClassifier()

    msk = np.random.rand(len(us_videos)) < 0.9
    train = us_videos[msk]
    test = us_videos[~msk]
    clf_train = clf.fit(train.loc[0:,column_name], train["days_to_trending"])

    clf_pred = clf_train.predict(test.loc[0:,column_name])

    print(accuracy_score(clf_pred,test["days_to_trending"]))

    # tree.plot_tree(clf_train)

    
    
    
if __name__ == "__main__":
    main()

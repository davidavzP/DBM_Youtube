import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import graphviz 
from sklearn import tree, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import seaborn as sns


def main():
    us_videos = pd.read_csv("youtube-new/USvideos.csv")
    us_videos_categories = pd.read_json('youtube-new/US_category_id.json')
    us_videos.category_id = us_videos.category_id.astype('category')
    categories = {int(category['id']): category['snippet']['title'] for category in us_videos_categories['items']}
    
    ##Category Graph

    sns.set(font_scale=1.5,rc={'figure.figsize':(16,9)})
    sns_ax = sns.countplot([categories[i] for i in us_videos.category_id])
    _, labels = plt.xticks()
    _ = sns_ax.set_xticklabels(labels, rotation=30)
    plt.show()
    
    #converting tending date to datetieme format
    us_videos['trending_date'] = pd.to_datetime(us_videos['trending_date'], format='%y.%d.%m').dt.date
    us_videos.trending_date.value_counts().sort_index(inplace=True)

    #converting puplish_time to datetime format
    publish_time = pd.to_datetime(us_videos.publish_time, format='%Y-%m-%dT%H:%M:%S.%fZ')
    us_videos['publish_date'] = publish_time.dt.date

    print(us_videos.info())

    #get days trending
    us_videos['days_to_trending'] = (us_videos.trending_date - us_videos.publish_date).dt.days
    us_videos.drop(['publish_time','title','publish_date', 'thumbnail_link', 'tags', 'trending_date', 'description'], axis=1, inplace=True)

    us_videos['channel_title'] = us_videos["channel_title"].str.decode("UTF-8").astype("category")
    us_videos["channel_title"] = us_videos["channel_title"].cat.codes

    us_videos['video_id'] = us_videos["video_id"].str.decode("UTF-8").astype("category")
    us_videos["video_id"] = us_videos["video_id"].cat.codes

    #create index
    # us_videos.set_index(['days_to_trending', 'video_id'], inplace=True)
    # print(us_videos.head())

    #clean up
    us_videos = us_videos[~us_videos.video_error_or_removed]
    us_videos.drop(['video_error_or_removed'], axis=1, inplace=True)

    # groups = us_videos.groupby(["category_id"])
    #
    # for name, group in groups:
    #     us_videos = group
    #     column_name = us_videos.columns[1:5].to_list()
    #     clf = tree.DecisionTreeClassifier()
    #
    #     msk = np.random.rand(len(us_videos)) < 0.5
    #     train = us_videos[msk]
    #     test = us_videos[~msk]
    #     clf_train = clf.fit(train.loc[0:,column_name], train["days_to_trending"])
    #
    #     clf_pred = clf_train.predict(test.loc[0:,column_name])
    #
    #     print(accuracy_score(clf_pred,test["days_to_trending"]))

    # print(us_videos.columns)
    column_name = us_videos.columns[0:9].to_list()
    clf = tree.DecisionTreeClassifier()

    msk = np.random.rand(len(us_videos)) < 0.5
    train = us_videos[msk]
    test = us_videos[~msk]
    print(column_name)
    clf_train = clf.fit(train.loc[0:,column_name], train["days_to_trending"])

    clf_pred = clf_train.predict(test.loc[0:,column_name])

    print(accuracy_score(clf_pred,test["days_to_trending"]))

    # tree.plot_tree(clf_train)
    # dot_data = tree.export_graphviz(clf, out_file=None) 
    # graph = graphviz.Source(dot_data) 
    # graph.render("youtube_data_tree_1") 

    ###PREDICTING CATEGORY based on video stats
    us_videos = us_videos.drop(["video_id"], axis=1, inplace=False)
    X = us_videos.drop(['category_id'], axis=1, inplace=False)
    X.fillna(X.mean(), inplace=True)
    y = us_videos['category_id']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=44, shuffle =True)
    
    DTCM = DecisionTreeClassifier(criterion='entropy',max_depth=20,random_state=33) #criterion can be entropy
    DTCM = DTCM.fit(X_train, y_train)
    
    DTCM_pred = DTCM.predict(X_test)

    print(accuracy_score(DTCM_pred,y_test))

    
if __name__ == "__main__":
    main()

#coding=utf-8

COLUMNS = ['cf', 'lfm', 'pr', 'userid', 'docid', 'label', 'date', 'duration', 'user_duration', 'commentCount', 'dislikeCount', 'norm_duration', 'likeCount', 'tags', 'viewCount', 'title', 'channel', \
            'publishTimeNormalized', 'click_num', 'click_show_rate', 'click_show_rate_prob', 'duration_prob', 'click_pos', 'topic_same_rate', 'item_vec']
MODEL_O = ['userid','docid','duration_prob','norm_duration','click_num','click_show_rate','click_show_rate_prob','topic_same_rate',"click_pos",'likeCount', 'tags', 'viewCount', 'title', 'channel']
MODEL_T = MODEL_O + ["video_ids"]

FIELD_INDEX = dict([(key,str(index)) for index,key in enumerate(MODEL_T)])

COMMAN_FIELDS = {'11': '11', '10': '10', '13': '13', '12': '12', '1': '1', '0': '0', '3': '3', '2': '2', '5': '5', '4': '4', '7': '7', '6': '6', '9': '9', '8': '8'}
MULTI_FIELDS = {'14':('videoIds','14')}

LOCAL_PATH = "/mnt1/local_v1/vector_recall/"
INDEX_FILE = LOCAL_PATH + "tranform/index_file"
DATA_ORIGIN = LOCAL_PATH + "train_data/origin"
DATA_ADDIDS = LOCAL_PATH + "train_data/addids"
DATA_LIB = LOCAL_PATH + "train_data/libtest"
DATA_TFRECORD = LOCAL_PATH + "train_data/"

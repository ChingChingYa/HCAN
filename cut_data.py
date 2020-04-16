from load_data import cut_dataset, cut_dataset2



if __name__ == '__main__':
    print('cut:prouduct_rating_data_1')
    cut_dataset('./data/prouduct_rating_data_1.pickle')
    print('cut:review_comment_data_1')
    cut_dataset2('./data/review_comment_data_1.pickle')
    print('cut:review_comment_bywords_1')
    cut_dataset2('./data/review_comment_bywords_1.pickle')
    print('cut success')

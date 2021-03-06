import matplotlib.pyplot as plt
import numpy as np
from load_data import load_data, read_dataset, cut_data_len
from sklearn.model_selection import train_test_split
from pmf import PMF
from sklearn.utils import shuffle
# from test import Config
import pickle

train_list = []
test_list = []
test_rmse = []
# config = Config()
aspect_vec = []
sentiment_labels = []

# PMF parameter
ratio = 0.8
lambda_U = 0.01
lambda_V = 0.01
latent_size = 6
learning_rate = 3e-5  # 3e-5
iterations = 1000


lambda_value_list = []
lambda_value_list.append([0.01, 0.01])

if __name__ == "__main__":

    alldata = load_data('./data/prouduct_rating_data_1.pickle')
    train, test = read_dataset('./data/prouduct_rating_data_11.pickle')
    num_users = cut_data_len(alldata, 'reviewerID')
    num_items = cut_data_len(alldata, 'asin')

    fp = open("log.txt", "a")
    fp.write("dataset:" + "Musical_Instruments_5"+"\n")
    fp.write("ratio:" + str(ratio)+"\n")
    fp.write("latent_factor:" + str(latent_size)+"\n")
    fp.write("learning_rate:" + str(learning_rate)+"\n")

    for lambda_value in lambda_value_list:
        lambda_U = lambda_value[0]
        lambda_V = lambda_value[1]
        # initialization
        pmf_model = PMF(U=None, V=None, lambda_U=lambda_U,
                        lambda_V=lambda_V, latent_size=latent_size,
                        momentum=0.8, learning_rate=learning_rate,
                        iterations=iterations)

        s = ('parameters are:ratio={:f}, reg_u={:f}, reg_v={:f}, latent_size={:d},'
             + 'learning_rate={:f}, iterations={:d}')
        print(s.format(ratio, lambda_U, lambda_V, latent_size,
              learning_rate, iterations))

        U = None
        V = None

        fp.write("=============================== Lambda Value ============="+"\n")
        fp.write("lambda_U:" + str(lambda_U)+"\n")
        fp.write("lambda_V:" + str(lambda_V)+"\n")
        # rmse均方根誤差
        rmse_minus = 1
        rmse_temp = 0
        round_value = 0

        model_loss = 0
        loss_minus = 1
        loss_temp = 0


        while loss_minus > 0.001:
            print("=============================== Round =================================")
            print(round_value)
            fp.write("=============================== Round ================================="+"\n")
            fp.write(str(round_value)+"\n")

            if round_value == 0:
                U, V, train_loss = pmf_model(num_users=num_users, num_items=num_items,
                                            train_data=train, aspect_vec=None, U=U, V=V, flag=round_value, 
                                            lambda_U=lambda_U, lambda_V=lambda_V)
            else:   
                U, V, train_loss = pmf_model(num_users=num_users, num_items=num_items, 
                                            train_data=train, aspect_vec=aspect_vec, U=U, V=V, flag=round_value,
                                            lambda_U=lambda_U, lambda_V=lambda_V)
        
            aspect_vec.clear()
            sentiment_labels.clear()

            for data in (train):
                s = np.multiply(U[int(data[0])], V[int(data[1])])
                sentiment_labels.append(s)

            print('testing PMFmodel.......')
            preds = pmf_model.predict(data=test)
            test_rmse = pmf_model.RMSE(preds, test[:, 2])
           
            print("=============================== RMSE =================================")
            print('test rmse:{:f}'.format(test_rmse))
            fp.write("================================ RMSE =========================="+"\n")
            fp.write(str('test rmse:{:f}'.format(test_rmse))+"\n")

            print("=============================== LOSS =================================")
            print('loss:{:f}'.format(train_loss+model_loss))
            fp.write("================================ LOSS =========================="+"\n")
            fp.write(str('loss:{:f}'.format(train_loss+model_loss))+"\n")

            
      
            # abs 絕對值
            rmse_minus = abs(rmse_temp - test_rmse)
            rmse_temp = test_rmse

            # abs 絕對值
            loss_minus = abs(loss_temp - train_loss+model_loss)
            loss_temp = train_loss+model_loss

            print('loss_minus='+str(loss_minus)) 

            if loss_minus > 0.001:
                aspect_vec, aspects, loss = config.Setup(sentiment_labels)
                model_loss = loss
                pickle.dump((aspects),
                            open('./data/aspect_vec'+str(round_value)+'.pickle', 'wb'))  
                # aspect_vec.append(sentiment_labels)

            round_value = round_value + 1

    fp.close()

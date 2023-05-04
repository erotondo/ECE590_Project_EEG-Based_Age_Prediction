# Sample cmd command to run CNN Model Framework

python auto_train_and_eval.py --name CNN_healthy_only --epochs 50 --batch_size 2 --arch CNNAgePrediction --group healthy --condition restEC

# Sample cmd command to run LSTM Model Framework

python auto_train_and_eval.py --name LSTM_healthy_only --epochs 50 --batch_size 2 --arch LSTMAgePrediction --group healthy --condition restEC

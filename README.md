# c4android
Runs a PyTorch Mobile model for detecting the state of a Connect 4 board, displays predictions and suggests best move.


This project deploys a PyTorch Mobile model trained with Detectron2 Go for the purpose of detecting the state of a Connect 4 board. The pipeline for training and exporting such a model is found at https://github.com/tho121/c4insseg. 

The state of the board is used to suggest the best move for the current player. The intended device for this app is the Pixel 6 running the Android OS. The camera from the back of the smart phone is used to stream images and passed to the model. Model predictions are then displayed on the phone screen, along with the camera stream. The recommended move will be indicated by changing the color of column box to purple.

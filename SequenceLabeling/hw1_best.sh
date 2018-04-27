#Download model
git clone https://gitlab.com/yujheli/model.git
#Training
#python model_best.py $1
#Tesing
python test_pad.py $1 $2 model/best.model

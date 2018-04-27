git clone https://gitlab.com/yujheli/ADLxMLDS_hw2_special_model.git
python hw2_seq2seq_attention.py --test $1 $2 "ADLxMLDS_hw2_special_model/model_last-1300"
python hw2_seq2seq_attention.py --peer $1 $3 "ADLxMLDS_hw2_special_model/model_last-1300"

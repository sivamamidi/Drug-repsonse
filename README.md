# Drug-repsonse
The concept behind precision medicine is to cure diseases based on a patient's genetic profile, lifestyle, and environmental factors. This method has been proven to increase clinical trial success rates and accelerate drug regulatory approval. However, previous  applications of precision medicine .A issue of the vital important for precision medicine is predicting the vulnerability of tumors to certain anti-cancer therapy. In the treatment of cancer, drug combinations have been shown to be quite helpful. They reduce drugs resistances and enhance treatments efficiency. The experiments in  all these therapeutic combination has become costly and time-consuming as a result of the increasing number of anti-cancer drugs. Large-scale drug response testing on cancer cell lines might help us understand way drugs react with cancer cells. I propose graph attention network for the
drug response prediction
![image](https://user-images.githubusercontent.com/83269163/234543398-e3f307e2-b940-49e3-bfde-bcd8809d5c15.png)
for preprocess to call function 
python preprocess.py --choice 0 
for training to call the python function
python training.py --model 0 --train_batch 1024 --val_batch 1024 --test_batch 1024 --lr 0.0001 --num_epoch 300 --log_interval 20 --cuda_name "cuda:0"
for saliency map to call the function
python saliency_map.py --model 0 --num_feature 10 --processed_data_file "data/processed/GDSC_bortezomib.pt" --model_file "model_GINConvNet_GDSC.model" --cuda_name "cuda:0"

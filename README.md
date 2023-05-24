# Drug-repsonse
# Precision Medicine: Drug Response Prediction using Graph Attention Network

## Introduction

This project focuses on predicting the vulnerability of tumors to certain anti-cancer therapy using precision medicine techniques. Precision medicine aims to cure diseases based on a patient's genetic profile, lifestyle, and environmental factors. By tailoring treatments to individual patients, precision medicine has shown promising results in increasing clinical trial success rates and accelerating drug regulatory approval.

One of the challenges in precision medicine is predicting the response of tumors to specific anti-cancer therapies. Drug combinations have been found to be effective in cancer treatment as they reduce drug resistance and enhance treatment efficiency. However, conducting experiments to test all possible therapeutic combinations has become costly and time-consuming due to the increasing number of anti-cancer drugs available.

To address this challenge, this project proposes the use of a Graph Attention Network (GAT) for drug response prediction. The GAT model leverages graph neural networks to analyze large-scale drug response testing data on cancer cell lines. By understanding how drugs interact with cancer cells, the model aims to predict the effectiveness of certain drug combinations for individual patients.

## Project Structure

The project is organized into the following components:

1. Preprocessing: The `preprocess.py` script is used to preprocess the input data. Run the following command to preprocess the data:

python preprocess.py --choice 0

2. Training: The `training.py` script trains the GAT model using the preprocessed data. Use the following command to start the training process:

python training.py --model 0 --train_batch 1024 --val_batch 1024 --test_batch 1024 --lr 0.0001 --num_epoch 300 --log_interval 20 --cuda_name "cuda:0"

3. Saliency Map: The `saliency_map.py` script generates saliency maps to visualize the important features for drug response prediction. Run the following command to generate the saliency map:

python saliency_map.py --model 0 --num_feature 10 --processed_data_file "data/processed/GDSC_bortezomib.pt" --model_file "model_GINConvNet_GDSC.model" --cuda_name "cuda:0

## Installation

1. Clone the repository:


2. Install the required dependencies:

pip install -r requirements.txt
## Usage

1. Preprocess the data using the `preprocess.py` script.

2. Train the GAT model using the `training.py` script.

3. Generate saliency maps to visualize important features using the `saliency_map.py` script.

4. Customize the scripts and parameters as per your specific requirements.

## Contributing

Contributions are welcome! If you find any issues or want to add new features, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).


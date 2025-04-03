"""
-- Created by: Ashok Kumar Pant
-- Email: asokpant@gmail.com
-- Created on: 03/04/2025
"""

import torch

from kullm.service.surname_classifier import SurnameDataset, SurnameClassifier, SurnameInference


def inference_example():
    data_path = './data/surnames-by-nationality.csv'  # Update with actual path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = SurnameDataset(data_path)
    model = SurnameClassifier(len(dataset.vocab), embed_dim=32, hidden_dim=32,
                              output_dim=len(dataset.nationality_vocab))
    inference = SurnameInference(model, dataset.vocab, model_path="./models/surname-model.bin", device=device)
    for name in ['McMahan', 'Nakamoto', 'Wan', 'Cho', "Pant", "aayush", "Ansan"]:
        prediction = inference.predict(name)
        prediction_label = dataset.nationality_vocab.idx_to_token[prediction]
        print(f"Predicted class for '{name}': {prediction_label}")


if __name__ == '__main__':
    inference_example()

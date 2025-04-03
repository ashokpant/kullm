"""
-- Created by: Ashok Kumar Pant
-- Email: asokpant@gmail.com
-- Created on: 03/04/2025
"""

import torch
from kullm.domain.surname_req_res import ClassifySurnameRequest, ClassifySurnameResponse

from kullm.service.surname_classifier import SurnameDataset, SurnameClassifier, SurnameInference


class SurnameClassificationService:
    def __init__(self):
        self.model = None
        self.dataset = None
        self._load_model()

    def _load_model(self):
        data_path = './data/surnames-by-nationality.csv'  # Update with actual path
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.dataset = SurnameDataset(data_path)
        model = SurnameClassifier(len(self.dataset.vocab), embed_dim=32, hidden_dim=32,
                                  output_dim=len(self.dataset.nationality_vocab))
        self.model = SurnameInference(model, self.dataset.vocab, model_path="./models/surname-model.bin", device=device)

    def predict(self, req: ClassifySurnameRequest) -> ClassifySurnameResponse:
        prediction = self.model.predict(req.query)
        prediction_label = self.dataset.nationality_vocab.idx_to_token[prediction]
        print(f"Predicted class for '{req.query}': {prediction_label}")
        res = ClassifySurnameResponse(error=False, category=prediction_label)
        return res

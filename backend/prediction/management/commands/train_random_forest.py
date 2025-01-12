from django.core.management.base import BaseCommand
from frauddetection.randomForest import FraudDetectionModel 
from frauddetection.prepareData import get_labeled_data
from pathlib import Path
import os

class Command(BaseCommand):
    help = 'Train and save the fraud detection model'

    def handle(self, *args, **kwargs):
        model_path = os.path.join(Path(__file__).resolve().parent.parent.parent, 'fraud_model.pkl')

        if os.path.exists(model_path):
            self.stdout.write("Loading existing model...")
            model = FraudDetectionModel.load_model(model_path)
        else:
            self.stdout.write("No model found. Training a new model...")
            model = FraudDetectionModel()

            train_data, test_data = get_labeled_data()

            model.train(train_data, test_data)
            model.save_model(model_path)
            self.stdout.write("Model trained and saved successfully.")
        self.stdout.write("Model ready for use.")
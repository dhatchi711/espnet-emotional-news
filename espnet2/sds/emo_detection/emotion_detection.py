from warnings import warn
import torch
from simpletransformers.classification import ClassificationModel

class EmotionDetection:
    def __init__(self, device: str="cuda"):
        label_list = ["Angry", "Happy", "Sad"]

        if device == "cpu":
            use_cuda = False
        elif device.startswith("cuda"):
            use_cuda = torch.cuda.is_available()
        
            if not(use_cuda): 
                warn(f"{device} was selected, but cuda is not available. Use cpu.")

        detector = ClassificationModel(
            "roberta",
            "distilroberta-base",
            num_labels= len(label_list),
            use_cuda=use_cuda,
            args={
                "num_train_epochs": 4,
                "labels_list": label_list,
                "max_seq_length": 256,
                "learning_rate": 1e-5,
                "overwrite_output_dir": True
            }
        )

        detector.model.from_pretrained("RusCucumber/good-news-everyon-emotion-detection")

        self.detector = detector

    def forward(self, transcript: str) -> str:
        """
        Args:
            transcript (str):
        
        Returns:
            emotion label (str):
                - model classifies into Angry, Happy, or Sad
        """
        
        pred, _ = self.detector.predict([transcript])
        return pred[0]
    
if __name__ == "__main__":
    emotion_detection = EmotionDetection(device="cuda")
    emotion = emotion_detection.forward("Dam breaking: New Epstein accuser comes forward.")

    print(emotion)
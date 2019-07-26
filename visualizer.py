from fastai import *
from fastai.vision import *
import cv2


class JaguarVisualizer(object):

    def __init__(self, data_path):
        self.data_path = data_path
        self.learner = None
        return

    def load_learner(self, path=None):
        if not path:
            path = Path(self.data_path)
        self.learner = load_learner(path)
        return

    def evaluate_single_image(self, image_path):
        if not self.learner:
            self.load_learner()
        pred = ""
        pred_class, pred_idx, outputs = self.learner.predict(open_image(image_path))
        if pred_idx == 0:
            pred = 'car'
        elif pred_idx == 1:
            pred = 'animal'
        cv2.waitKey()
        return pred

    def display_batch_prediction(self, images):
        preds = []
        for i in range(len(images)):
            pred = self.evaluate_single_image(images[i])
            preds.append(pred)
            img = cv2.imread(images[i])
            cv2.putText(img, pred, (10, 30), \
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow("prediction_{}".format(i), img)
        print(preds)
        cv2.waitKey()
        return preds


def main():
    vis = JaguarVisualizer('./data')
    vis.load_learner()
    vis.display_batch_prediction(['./download.jpeg', './image.jpeg'])
    return

if __name__ == '__main__':
    main()

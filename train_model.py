from fastai.vision import *

def create_ImageBunch(path):
    data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.2, \
            ds_tfms=get_transforms(), size=224, num_workers=0, bs=16).normalize(imagenet_stats)
    return data

def train_model(data):
    learn = cnn_learner(data, models.resnet34, metrics=error_rate)
    learn.fit_one_cycle(4)
    learn.save('stage-1')
    return learn

def main():
    path = Path('./data')
    data = create_ImageBunch(path)
    print(data.classes)
    learn = train_model(data)

    return


if __name__ == '__main__':
    np.random.seed(42)
    main()

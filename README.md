# 데이콘 컴퓨터 비전 알고리즘 활용 이상치 탐지 대회(Dacon Anomaly Detection Project)  

## 주요 파라미터(Main Parameters)
- Epoch : Total 30 Epochs  
          ~10 : For the First Training
          ~20 : More Data Training (I added datas twice the size of the previous model..)
          ~30 : 
- Optimizer : AdamW
- lr_scheduler : Depreciated (not performed well..) 
- Batch Size : 32

## 사용한 모델 (Classification Model)
- 컴퓨팅 파워의 한계로 아래의 모델을 사용했습니다.
- Efficient Net V2 B3
- Efficient Net V1 B3 

## 학습 환경 (Training Environment)
- Colab... 
- Colab Pro (도중에 Pro로 바꿔 진행했습니다)

## 학습 전략 (Training Strategy)
### Augmentation
- 사용한 Transforms  

          train_transform = transforms.Compose([
        transforms.Resize((img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(), ## added
        transforms.RandomRotation(0.5, expand=False),
        transforms.RandomResizedCrop(size = (img_size), scale = (0.7, 1)), ## added
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1), ## added
        transforms.RandomAffine((-20,20)), ## added
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.433038, 0.403458, 0.394151],
        std = [0.181572, 0.174035, 0.163234]),])

- 전략은 Random으로 Augmentation되는 것에 대한 의존이 큽니다. 주요 가정은
  "같은 이미지라도 Random으로 Augmentation되면 모델 입장에선 충분히 다른 이미지로 생각하고 학습할 것이다!"
  였습니다.
- 때문에 Random 요소가 많은 Transforms를 많이 사용했습니다.

### Ensemble 
- 학습된 EFN V2 B3와 EFN V1 B3를 각각 앙상블 시켰습니다.
- 최종 모델에선 Epoch 중 가장 학습이 잘 된 모델 5개씩을 선정하여 총 10개의 Classfier를 Ensenble 시켰습니다.
- 단순히 더 많은 모델 = 더 좋은 성능 이라는 단순한 기대에서 진행했지만 결과는 큰 차이가 없었습니다.

### TTA
- 이번 프로젝트를 진행하면서 TTA를 처음 사용해봤습니다.
- 사용한 tta_transforms는 다음과 같습니다.
          tta_transforms = tta.Compose(
    [tta.HorizontalFlip(),
    tta.VerticalFlip(),
    tta.Rotate90(angles=[0, 90, 180, 270]),
    tta.Rotate90(angles=[0, 90, 180, 270]),
    tta.Multiply([0.9, 1])
    ])
- 사용 전과 비교했을 때 성능이 약 3% 정도 상승했습니다. 

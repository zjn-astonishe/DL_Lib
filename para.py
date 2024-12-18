from model.MyViT import my_vit_xx_small
from torchinfo import summary


if __name__ == '__main__':
    # model = mobile_vit_xx_small(num_classes=10)
    model = my_vit_xx_small(num_classes=10)
    # X = torch.rand(2, 3, 224, 224)   # B. C. H. W
    # model = MyViTBlock(16, 64, 32)
    # print("MyViTBlock")
    # print(model(X).shape)
    summary(model, input_size=(2, 3, 224, 224))
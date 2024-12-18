import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model.MyViT import my_vit_xx_small as create_model
from data.load_dataset import read_test_data
from data.my_dataset import MyDataSet

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    img_size = 32
    data_transform1 = transforms.Compose(
        [transforms.Resize(int(img_size * 1.14)),
         transforms.CenterCrop(img_size),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # load image
    test_images_path, test_images_label = read_test_data('/root/dataset/cifar-10/train_valid_test/test')
    img_size = 32
    data_transform = {
        "test": transforms.Compose([transforms.RandomResizedCrop(img_size), 
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    }
    test_dataset = MyDataSet(images_path=test_images_path, images_class=test_images_label, transform=data_transform['test'])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False, pin_memory=True, collate_fn=test_dataset.collate_fn)

    img_path = "/root/dataset/cifar-10/test/6.png"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform1(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # create model
    model = create_model(num_classes=10).to(device)
    # load model weights
    model_weight_path = "./weights/best_model.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    with torch.no_grad():
        # print(test_images_path)
        pred_labels = []
        for step, data in enumerate(test_loader):
            images, labels = data
            pred = model(images.to(device))
            pred_classes = torch.max(pred, dim=1)[1]
            pred_labels.extend(pred_classes.cpu().detach().numpy())
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    # for i in pred_labels:
    #     print(class_indict[str(i)])

        

    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    plt.title(print_res)
    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                  predict[i].numpy()))
    plt.show()

    with open('submission.csv', 'w') as output_file:
        output_file.write('id,' + 'label' + '\n')
        for i, out in zip(test_dataset.images_path, pred_labels):
            output_file.write(i.split('.')[0] + ',' + class_indict[str(out)] + '\n')
    

if __name__ == '__main__':
    main()

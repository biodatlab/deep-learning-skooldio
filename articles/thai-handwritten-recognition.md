# Thai Digit Handwritten Recognition

ในบทความนี้เราจะมาสรุปการเขียน Pytorch เพื่อใช้ในการสร้าง Neural network สำหรับ
Thai digit handwritten recognition

## ดาวน์โหลดและสรุปข้อมูล

เราเริ่มจากการ clone ข้อมูลจาก Github โดยสามารถตรวจสอบได้จาก tab folder ด้านข้าง
จากนั้นทำการสรุปข้อมูลโดยการนับจำนวน parent ของภาพทั้งหมดใน folder

```sh
!git clone https://github.com/biodatlab/deep-learning-skooldio
```

```py
from pathlib import Path
from collections import Counter
directory = "deep-learning-skooldio/"
paths = glob(op.join(directory, "thai-handwritten-dataset", "*", "*"))
Counter([Path(p).parent.name for p in paths])
```

- `pathlib` เป็น library หนึ่งที่นิยมใช้ในการดึงข้อมูลจาก path เช่น ชื่อไฟล์, parent (folder ของไฟล์) และอื่นๆ
- `Counter` ใช้ในการนับจำนวนของข้อมูลที่เป็น list ได้อย่างรวดเร็ว

## ทดลองดูภาพจาก dataset

ไลบรารี่ที่นิยมใช้ในการอ่านภาพคือ `PIL` และ `OpenCV` โดยในบทเรียนเราใช้ `PIL` ในการอ่านภาพ

```py
from PIL import Image
idx = 460
print(Path(paths[idx]).parent.name)
Image.open(paths[idx])
```

## แบ่ง Dataset เป็น train, validation และ test

ในบทเรียนที่เรียน เราแบ่งข้อมูลเป็นเพียง training set และ validation set เท่านั้น

- training set คือข้อมูลที่ใช้ในการเทรนโมเดล
- validation set คือข้อมูลที่ไม่ได้ใช้ในการเทรนโมเดล แต่ว่าใช้ในการประเมินความแม่นยำของโมเดลระหว่างเทรน
- test set คือข้อมูลที่ไม่เคยเห็นมาก่อน เราแยกไว้ใช้ประเมินโมเดลที่เทรนเสร็จแล้ว

## สร้าง Dataset และ DataLoader

เมื่อได้ชุดข้อมูลเรียบร้อย ถัดไปคือการสร้าง `Dataset` และ `DataLoader`

`DataSet` ประกอบด้วย 3 methods หลักๆคือ

- `__init__` โดยส่วนมากจะใส่ความสัมพันธ์ระหว่าง input และ output ของโมเดล
- `__len__` เพื่อคืนค่าจำนวนข้อมูลทั้งหมด
- `__getitem__` รับค่า index และคืนค่าของข้อมูลที่ตำแหน่ง index นั้นๆ อาจจะเป็น input และ output ของโมเดล
  หรือ input อย่างเดียวก็ได้

ด้านล่างเราโชว์ตัวอย่างการสร้าง `Dataset` สำหรับข้อมูล Thai digit ที่เราได้เตรียมไว้ให้
เริ่มต้นจากการใส่ `img_dir` เพื่อทำการเปลี่ยนเป็น `img_labels` คือเป็นลิสต์ของ path และ label ของภาพ

```py
class ThaiDigitDataset(Dataset):
    def __init__(self, img_dir: str, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_labels = [(p, Path(p).parent.name) for p in glob(op.join(img_dir, "*", "*"))]
```

ถัดมาคือจำนวนของข้อมูลทั้งหมด โดยหาได้จาก `len` ของ `img_labels`

```py
    def __len__(self):
        return len(self.img_labels)
```

สุดท้ายคือการคืนค่าของข้อมูลที่ตำแหน่ง index นั้นๆ โดยเราจะเปิดภาพด้วย `PIL.Image`
และทำการแปลงเป็น `tensor` ด้วย `transform` หักลบด้วย 1 เพื่อเปลี่ยนให้ตัวอักษรเป็นสีขาวส่วนพื้นหลังเป็นสีดำ

```py
    def __getitem__(self, idx):
        image, label = self.img_labels[idx]
        label = int(label)
        image = Image.open(image)
        if self.transform:
            image = 1 - self.transform(image)
        return image, label
```

เมื่อสร้าง class แล้วเรียบร้อยเราสามารถสร้าง instance ของ class ได้ดังนี้

```py
train_thaidigit_dataset = ThaiDigitDataset("data/train/", transform=transform)
val_thaidigit_dataset = ThaiDigitDataset("data/validation/", transform=transform)
```

`DataLoader` สามารถนำมาครอบ dataset เพื่อทำให้เราสามารถดึงข้อมูลเป็น batch สำหรับใช้ในการเทรนโมเดลได้
`batch_size` กำหนดขนาดของ batch ที่เราต้องการ และ `shuffle` กำหนดว่าจะทำการสลับตำแหน่งของข้อมูลใน batch หรือไม่
ส่วนมากจะใช้ `shuffle=True` สำหรับการเทรนโมเดล และ `shuffle=False` สำหรับการ validate หรือ test โมเดล

```py
train_loader = DataLoader(train_thaidigit_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_thaidigit_dataset, batch_size=16, shuffle=False)
```

## สร้าง Model ด้วย `nn.Module`

จากนั้นเราสามารถสร้าง Neural network โดยการประกอบ layers ต่างๆเข้าด้วยกันได้ โดยใช้ `nn.Module` ซึ่งเป็น class
ที่มี method ที่ชื่อว่า `forward` ซึ่งจะรับ input และคืนค่า output ของโมเดล

```py
import torch.nn as nn
import torch.nn.functional as F

class ThaiDigitNet(nn.Module):
    def __init__(self):
        super(ThaiDigitNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
```

เมื่อสร้างเสร็จแล้วลองสร้าง instance ของโมเดล แล้วส่งข้อมูลเข้าไปดูว่าโมเดลสามารถทำนายได้ไหม input, output dimension ถูกต้องมั้ย

```py
net = ThaiDigitNet()
images, labels = next(iter(train_loader))
pred = net(images)
print(image.shape, pred.shape)
```

ถ้า dimension ถูกต้อง สเต็บถัดไปเราจะเริ่มเทรนโมเดลได้

**ข้อสังเกต** จะเห็นว่าเลเยอร์สุดท้ายของโมเดลเราไม่ได้ใส่ softmax เพื่อเปลี่ยนให้ logits เป็น probability เนื่องจากเราจะใช้
`nn.CrossEntropyLoss` ซึ่งมีการใช้ softmax ในตัวเองแล้วก่อนหา loss จึงไม่ได้ใส่ softmax เพิ่ม

สำหรับวิธีปกติคือการใช้ `nn.NLLLoss` แต่เพิ่ม `softmax` layer เข้าไปในโมเดลตอนท้าย
แต่การใช้ `nn.CrossEntropyLoss` ช่วยให้เวลาเทรนโมเดลมีความเสถียรกว่าเมื่อใช้ `nn.NLLLoss` และ `softmax` เอง

## เทรนโมเดล

สเต็บหลักๆของการเทรนโมเดลประกอบด้วย loss function, optimizer, data loader, และ model ที่เราสร้างไว้
หลักจากนั้นเราจะเขียน training loop ได้ดังนี้

```py
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
n_epochs = 50

for epoch in range(n_epochs):
    # training
    net.train()
    for batch_idx, (images, labels) in enumerate(train_loader):
        pred = net(images)
        loss = loss_fn(pred, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

ในที่นี้เราเทรนโมเดลจำนวน 50 epochs โดยใช้ SGD optimizer และ cross entropy loss
ถัดมาเราสามารถเพิ่ม validation loop ได้เพื่อดูว่าโมเดลเราเทรนไปได้ดีแค่ไหน

```py
    net.eval()
    val_loss, correct = 0, 0
    n_val = len(val_loader.dataset)
    for images, labels in val_loader:
        pred = net(images)
        val_loss += loss_fn(pred, labels).item()
        correct += (pred.argmax(1) == labels).float().sum().item()
```

โดยเราต้องเปลี่ยนโมเดลเป็น evaluation mode ก่อน โดยใช้ `net.eval()` จากนั้นก็คำนวณหา loss และ accuracy ของ validation set
ได้ตามปกติ

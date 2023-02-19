# Thai Digit Handwritten Recognition: Experiments

หลักจากที่เราได้สร้างโมเดลแรกของ Thai Digit Handwritten Recognition แล้ว
มีวิธีการอีกมากมายที่สามารถทดลองเพื่อปรับแก้ให้โมเดลมีประสิทธิภาพดียิ่งขึ้นกว่าเดิม

## เริ่มจากการสร้าง script ในการเทรนโมเดล

เราจะเริ่มจากการนำโค้ดที่ได้จากครั้งที่แล้วมาสร้าง script ในการเทรนโมเดล และทำการเก็บ log ของการเทรนโมเดล
เพื่อเปรียบเทียบ training loss, validation loss, training accuracy และ validation accuracy ของโมเดลหลังจากการเทรน

```py
def train(
    model,
    n_epochs,
    loss_function,
    optimizer,
    train_loader,
    validation_loader
):
    ...
    return model, training_logs
```

หลังจากเขียน script แล้วเรียบร้อย เราได้ทดลองสิ่งต่างๆดังนี้

1. เพิ่มจำนวน Layer ให้โมเดล
2. เพิ่ม Dropout เพื่อลดโอกาสการเกิด Overfitting
3. เพิ่ม Augmentation

## เพิ่มจำนวน Layer ให้โมเดล

น่าจะเป็นสิ่งที่ทดลองได้ง่ายที่สุด คือการลองเพิ่มจำนวน Layer ให้โมเดล เช่น เปลี่ยนจาก

```py
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

ให้กลายเป็น

```py
class ThaiDigitMoreLayers(nn.Module):
    def __init__(self):
        super(ThaiDigitMoreLayers, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 392)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(392, 196)
        self.fc3 = nn.Linear(196, 98)
        self.fc4 = nn.Linear(98, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        return x
```

การเพิ่มจำนวนเลเยอร์อาจจะทำให้โมเดลเราเรียนรู้ได้ดีขึ้น แต่อาจจะทำให้เกิด Overfitting หรือปรากฎการณ์ที่โมเดลเรียนรู้ที่จะทำนายข้อมูลใน training set
ได้แม่นยำมาก แต่ไม่สามารถทำนายข้อมูลใน validation set ได้ดี ซึ่งวิธีการแก้ไขอย่างหนึ่งคือการใส่ Dropout เข้าไปในโมเดล

## ใส่ Dropout layer

```py
class DropoutThaiDigit(nn.Module):
    def __init__(self):
        super(DropoutThaiDigit, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 392)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(392, 196)
        self.fc3 = nn.Linear(196, 98)
        self.fc4 = nn.Linear(98, 10)
        self.dropout = nn.Dropout(0.1) # ใส่เพิ่ม

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc4(x)
        return x
```

การใส่เลเยอร์ `nn.Dropout` ทำได้ง่ายมาก และ Dropout layer สามารถใช้ซ้ำได้เนื่องจากเป็นการสุ่มตัวเลขในการเปิด-ปิด activation ของเลเยอร์
0.1 แปลว่าเราทำการสุ่ม drop activation ไป 10% จากทั้งหมดระหว่างเทรนโมเดล

## ใช้ Image augmentation

ในการทดลองของเรา Image augmentation เพิ่มประสิทธิภาพของโมเดลได้มากที่สุด
โดย Image augmentation ที่ใช้คือเราทำการเพิ่มการสุ่มการเปลี่ยนแปลงของภาพ เช่น สุ่มการหมุน
การย้าย และการเปลี่ยนขนาดของภาพ เพื่อเพิ่มความหลากหลายของภาพ (data distribution)

Image augmentation สามารถทำได้งสะดวกมากโดยใช้ `torchvision.transforms` ซึ่งเราสามารถทำการสร้าง transform ได้ง่ายๆดังนี้

```py
train_transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(),
    transforms.RandomAffine(degrees=(-15, 15), translate=(0.05, 0.1), scale=(1, 1)),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(),
    transforms.ToTensor(),
])
```

โดยปกติการใส่ `transforms` จะทำเพียงแค่ใน trainging set เพราะเราต้องการคง validation set เอาไว้เช่นเดิม
ในตัวอย่างเราทำการเชื่อม dataset เดิมกับ dataset ที่มีการ transform แล้วด้วยโดยใช้

```py
train_dataset = ConcatDataset([train_thaidigit_dataset, augmented_train_dataset])
```

ในการทำจริงๆอาจจะใช้แค่ `augmented_train_dataset` โดยไม่ Concatenate กับ `train_thaidigit_dataset` ก็ได้

## Image augmentation

Image augmentation เป็นวิธีการแบบหนึ่งของ data augmentation ซึ่งมีส่วนสำคัญมากๆในการเพิ่มความหลากหลายของ input ระหว่างเทรนโมเดล
โดยผลดีของการทำ image augmentation มีหลายอย่างเช่น การลดโอกาสการเกิด overfitting ของโมเดลเนื่องจากโมเดลจะเห็น input ที่มีความหลากหลายขึ้นกว่าเดิม
นอกจากนั้นคือการทำให้โมเดลมีความยืดหยุ่น (generalized) กับ test data มากขึ้น เนื่องจากเราได้ป้อนข้อมูลที่มีความหลากหลายมากขึ้นระหว่างเทรนโมเดล
ทั้งนี้เราอาจจะต้องเทรนโมเดลด้วยจำนวน epoch มากขึ้นเนื่องจากโมเดลอาจจะต้องเรียนรู้ชุดข้อมูลที่มีความหลากหลายมากขึ้น

ไลบรารี่ที่สามารถทำ Image augmentation ที่เป็นที่นิยมได้แก่ [`torchvision`](https://pytorch.org/vision/stable/index.html) และ
[`albumentations`](https://albumentations.ai/) โดยในบทเรียนเราใช้ `torchvision` เพื่อทำการ transform ภาพในชุดข้อมูล train ดังนี้

```py
train_transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(),
    transforms.RandomAffine(degrees=(-10, 10), translate=(0.0, 0.1), scale=(1, 1)),
    transforms.ToTensor(),
])
```

- `Resize((28, 28))` ทำการ resize ขนาดภาพให้เป็น 28x28 pixel
- `Grayscale()` เปลี่ยนภาพสีให้เป็นภาพขาวดำ
- `RandomAffine(degrees=(-10, 10), translate=(0.0, 0.1), scale=(1, 1))` ใส่ 3 input parameters ได้แก่
  - `degrees` เพื่อสุ่มให้ภาพหมุนไปทางซ้ายหรือขวาได้ตามจำนวนองศาที่กำหนด
  - `translate` เพื่อสุ่มให้ภาพเลื่อนไปทางซ้ายหรือขวาด้วยสัดส่วนกำหนด โดย 0 คือไม่เลื่อนเลย
  - `scale` เพื่อสุ่มให้ภาพขยายหรือซูมออกได้ตามสเกลที่กำหนด
- `ToTensor()` เปลี่ยนสเกลของภาพจาก 0 ถึง 255 (ดำ-ขาว) ให้อยู่ในสเกล 0 ถึง 1 และแปลงภาพให้เป็น tensor เพื่อใส่เข้าไปในโมเดลได้

จากนั้นเราสามารถนำ `Compose` มารวม transform ที่สร้างเข้ามาด้วยกัน นอกจากนั้นยังมี `transforms` อื่นๆที่ใช้ได้เช่น

- `RandomCrop` เพื่อสุ่มการ crop ภาพ
- `RandomHorizontalFlip` เพื่อกลับภาพตามแนวนอนจากซ้ายไปขาว (เหมาะกับภาพทั่วไป แต่อาจจะไม่เหมาะกับภาพที่อ่านได้ทางเดียวเช่นภาพตัวเลข)
- `RandomErasing` ลบบางส่วนของภาพด้วยสี่เหลี่ยมสีดำ

## แล้วยังมีวิธีอื่นๆนอกเหนือจากบทเรียนมั้ย

ยังมีอีกหลายส่วนที่สามารถทำเพิ่มเติมได้จากบทเรียน

### ในเชิงการสร้างโมเดล

ยังมีอีกหลากหลายวิธีในการเทรนโมเดลที่ทำให้เราได้โมเดลและเก็บโมเดลที่ดีที่สุดระหว่างการเทรนโมเดล ได้แก่

- เซฟโมเดลที่ดีที่สุดระหว่างเทรน (Save best checkpoint): เซฟโมเดลที่ทำนายผลดีที่สุดใน validation set ไว้เพื่อใช้ในการทำนายในครั้งถัดไป
- ใช้ learning rate scheduler เพื่อปรับปรุงการเรียนรู้ของโมเดล: ปรับปรุงการเรียนรู้ของโมเดลโดยการปรับ learning rate ให้มีค่าต่ำลงเมื่อโมเดลเรียนรู้ไปเรื่อยๆ
  เช่นใช้ [`LinearLR`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LinearLR.html)
- ปรับขนาดข้อมูลภาพและโมเดลให้มีขนาดใหญ่ขึ้น (Increase image size and model size): ในบทเรียนนี้เริ่มต้นจากการใช้ขนาดของภาพเป็น 28x28 และโมเดลเป็น 2 layers แต่ในการทำจริงๆอาจจะใช้ขนาดของภาพใหญ่ขึ้นเช่น 64x64 และโมเดลมีจำนวน layers มากขึ้นเช่น 4 layers หรือ 8 layers ก็ได้
- ปรับปรุงโมเดลที่ไม่ใช้ Dense layer: ข้อเสียของ Dense layer ที่เราทำคือจำเป็นต้องยืด pixels ให้มีขนาด 784 ซึ่งเราอาจจะทำให้สูญเสียข้อมูลเชิงพื้นที่ไป
  ในอนาคตเราจะได้เรียนวิธีใช้ Convolution layer แทน ซึ่งสามารถดึงข้อมูลเชิงพื้นที่ออกมาได้

ใครที่สนใจจูนโมเดลเพิ่มเติม อาจศึกษาเพิ่มได้จาก [`tuning_playbook`](https://github.com/google-research/tuning_playbook) ของ Google Research
เพิ่มเติมได้ ซึ่งเค้าจะแนะนำ tips and tricks ในการเทรนโมเดลที่มากกว่าที่สอนในบทเรียนนี้

### ในเชิงข้อมูล

- ใช้ `confusion_matrix` หรือ loss function ในการดูความผิดพลาดของโมเดลว่าทำนายผิดมากที่สุดที่ไหน เพื่อปรับปรุงข้อมูลในอนาคต (เช่น เลข 3 ไทยอาจจะคล้ายเลข 7)
- ทดลอง augment ข้อมูลในแบบต่างๆเพิ่มเติม

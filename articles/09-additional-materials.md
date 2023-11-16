# คำอธิบายเพิ่มเติม

## ทบทวนการสร้าง Class ในภาษา Python

การเขียน `Dataset` และ `nn.Module` จำเป็นต้องใช้องค์ความรู้เบื้องต้นเกี่ยวกับการเขียน Class ในภาษา Python

การสร้าง class ของ Python จะมี built-in function `__init__()` ที่ใช้ในการเก็บข้อมูล
เมื่อ object ถูกสร้างขึ้น ตัวอย่างด้านล่างจะเป็นการเขียน class ชื่อ `Person` โดยมีข้อมูลชื่อและอายุ

```py
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    def display_name(self):
        print(f"Name: {self.name}")

p1 = Person("John", 36)  # สร้าง class ของ Person ที่มีชื่อ John, อายุ 36
p1.display_name()  # เรียกคำสั่ง display_name() เพื่อแสดงชื่อ
# >>> Name: John
```

นอกจากนั้นจะเห็นว่ามีพารามิเตอร์ที่ชื่อว่า `self` ซึ่งใช้ในการ reference ถึง instance ของคลาส เช่นการสร้าง instance ที่ชื่อ `p1` ขึ้นมา เราจะสามารถเข้าถึง `p1.name`, `p1.age` ได้ การใช้ชื่อ `self` เป็นมาตรฐานหลักของการเขียน Class
ของโปรแกรมภาษา Python

เมื่อเราสร้าง Class มาเรียบร้อยจะสามารถสร้าง Class เพิ่มเติมจากเดิม ("inheritance") โดยที่อาจจะเพิ่ม function อื่นๆ หรือปรับเปลี่ยน Class เดิมได้

```py
class Student(Person):
  def __init__(self, name, age, grad_year):
    super().__init__(name, age)
    self.grad_year = grad_year

s1 = Student("Jane", 18, 2022)
s1.display_name()  # Student ที่ inherit จาก Person ใช้คำสั่งจาก Person ได้
print(s1.name)  # เข้าถึง name, age ได้
print(s1.age)
print(s1.grad_year)  # เข้าถึงปีที่จบการศึกษา ซึ่งเป็นค่าใหม่ได้
```

ในที่นี้เราสามารถสร้างคลาสที่ชื่อว่า `Student` ที่รับทั้ง `name`, `age`, และปีที่จบ `grad_year` โดยจะเห็นว่าตัวแปร `s1` ที่สร้างขึ้นจะสามารถเข้าถึง `s1.name`, `s1.age` ที่มีในคลาส `Person` และ `s1.grad_year` ที่มีในคลาส `Student` ได้

คอนเซ็ปต์เดียวกันถูกนำมาใช้สำหรับการสร้างชุดข้อมูลและโมเดลด้วย Pytorch โดยเราสามารถสร้าง Class จาก `nn.Module` และ `torch.utils.data.Dataset`

- สำหรับ `nn.Module` ที่เป็น class พื้นฐานของการสร้างโมเดล Neural netowkr  เราจะต้องเขียนฟังก์ชั่นหลักคือ `__init__()` เพื่อประกาศเลเยอร์ของโมเดล และ `forward()`  ที่นำเลเยอร์มาประกอบเพื่อใช้ในการคำนวณจากข้อมูลขาเข้า
- สำหรับ `torch.utils.data.Dataset` เป็น class พื้นฐานของการสร้างชุดข้อมูล โดยเราจะนำ เป็นต้องเขียน `__init__()`, `__len__()` และ `__getitem__()` เพื่อใช้สำหรับชุดข้อมูลนั่นเอง

ในปัจจุบันมีไลบรารี่ที่ใช้ในการสร้างชุดข้อมูลที่สะดวกมากยิ่งขึ้น เช่น ไลบรารี่ [Datasets](https://huggingface.co/docs/datasets/index)


## Softmax operation and cross entropy loss

Softmax เป็น operation ที่ใช้เปลี่ยนจาก logits ซึ่งเป็นผลลัพธ์ที่ได้ในเลเยอร์สุดท้ายของ Neural network ให้กลายเป็นความน่าจะเป็น
สำหรับบทเรียนที่นำเสนอ จะเห็นว่าเราไม่ได่้เขียน softmax ในเลเยอร์สุดท้ายเนื่องจากเราเลือกใช้ loss ของ Pytorch `CrossEntropy` ที่ทำการคำนวณ softmax และคำนวณ loss ให้เลย

โดยการใช้ `CrossEntropy` loss ยังมีข้อดีเพิ่มเติมเกี่ยวกับการคำนวณหลังบ้าน Pytorch อีกด้วย

โดยการใช้ Cross entropy loss จะเทียบเท่ากับการใส่ `LogSoftmax` ไปในเลเยอร์สุดท้ายของ Neural Network ร่วมกับการใช้ Negative Log Likelihood loss (`NLLLoss`) ของ Pytorch

สำหรับคนที่สนใจอ่านข้อมูลเพิ่มเติม สามารถอ่านได้ใน documentation [`CrossEntropyLoss`](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) ของ Pytorch


ตัวอย่างการใช้ฟังก์ชั่น softmax อาจจะใช้ผ่าน `torch.nn.functional` หรือใช้โดยตรงผ่าน tensor เลยผ่านคำสั่ง `.softmax(dim=1)` ก็ได้ ยกตัวอย่างเช่น

```py
import torch
import torch.nn.functional as F

logits = torch.tensor([[-0.5, 6.0], [3.0, 0.7]])
F.softmax(logits, dim=1)  # คำนวณ softmax ในแกนแถว (ตามแนวนอน)
# tensor([[0.0015, 0.9985],
#         [0.9089, 0.0911]])

logits.softmax(dim=1)
# tensor([[0.0015, 0.9985],
#         [0.9089, 0.0911]])
```

จะเห็นว่าผลบวกตามแกนแถวจะเท่ากับ 1 โดยคำสั่ง Softmax ถูกใช้บ่อยครั้งมากในการเปลี่ยนผลลัพธ์จาก Neural network ให้กลายเป็นความน่าจะเป็น

ในตัวอย่างของการทำนายตัวเลขลายมือ เราใช้ softmax เพื่อเปลี่ยนผลลัพธ์เป็นความน่าจะเป็นของเลข 10 ตัว จาก 0 ถึง 9 ก่อนแสดงผลในแอพพลิเคชั่น


## การเลือก Optimizer สำหรับการเทรนโมเดล

การอัพเดทพารามิเตอร์ของโมเดล ใช้วิธี back propagation ซึ่งเป็นเทคนิคที่ใช้ gradient ในการอัพเดทพารามิเตอร์ของโมเดล
สำหรับตัวอย่างในบทเรียนเราได้ทดลองใช้อัลกอริทึมของ Optimizer ที่ชื่อ Stochastic Gradient Descent (SGD) และ Adaptive Moment (Adam)

ผู้ที่สนใจเพิ่มเติม Optimizer ที่ใช้การอัพเดทพารามิเตอร์ของโมเดล Neural Network ที่ Pytorch ได้เขียนไว้มีให้เลือกหลากหลายมาก เช่น `Adadelta`, `Adagrad`, `AdamW`, `Adamax`, `RMSprop` โดยแต่ละวิธีจะมีข้อดีข้อเสียต่างกันไป

โดยทั่วไปทั้ง `SGD` และ `AdamW` ได้รับความนิยมสูงที่สุดในการนำมา optimizer โมเดล แต่ว่าวิธีการอื่นก็ใช้ได้เช่นกัน ในปัจจุบันยังไม่มีแนวปฏิบัติว่า optimizer ตัวใดทำงานได้ดีที่สุด ดังนั้นถ้ายังเลือกไม่ได้ แนะนำให้ทดลองใช้ `AdamW` และ `SGD` ก่อน


## ไลบรารี่ต่างๆที่พบในบทเรียน

ไลบรารี่ที่เราพบในบทเรียนบ่อยครั้ง จะเป็นไลบรารี่เกี่ยวกับการลิสต์ไฟล์ (`glob`), ไลบรารี่ที่ช่วยในเกี่ยวกับ operating system interfaces เช่น การเชื่อม path และไฟล์ (`os`), ไลบรารี่เกี่ยวกับการดึงค่าต่างๆจาก path (`pathlib`) ไลบรารี่ที่มีตำสั่งเกี่ยวกับการดำเนินการเกี่ยวกับไฟล์  (`shutil`) และไลบรารี่ที่ช่วยในการติดตามความคืบหน้า (`tqdm`)

- `glob`, `glob` เป็นไลบรารี่ที่นิยมใช้สำหรับลิสต์ไฟล์จาก directory เช่น

``` py
from glob import glob
glob("thai-handwritten-dataset/*") # เป็นการดึง path ของไฟล์ทั้งหมดที่อยู่ใน folder `thai-handwritten-dataset`
glob("thai-handwritten-dataset/*/*") # เป็นการดึง path และไฟล์ทั้งหมดที่อยู่ในทุก subfolder ของ `thai-handwritten-dataset`
glob("thai-handwritten-dataset/*/*.jpg") # เป็นการดึง path และไฟล์ทั้งหมดที่อยู่ในทุก subfolder ของ `thai-handwritten-dataset` ที่มีสกุล jpg
```

การใช้ `*` แปลว่าเลือกทุกๆ subfolder


- `os` และ `os.path` ถูกนำมาใช้ในการเชื่อม directory เป็นหลัก

```py
import os.path as op
directory = "deep-learning-skooldio/"
op.join(directory, "thai-handwritten-dataset", "*", "*")
# >>> 'deep-learning-skooldio/thai-handwritten-dataset/*/*'
```

การใช้ `os.path` มีข้อดีกว่าการต่อ string โดยใช้ `+` เนื่องจากระบบปฏิบัติการเช่น Windows, MacOS, หรือ Ubuntu จะใช้วิธีการต่อไฟล์ต่างกัน โดย `os.path` จะทำการต่อไฟล์ให้เป็นไปตามระบบปฏิบัติการที่เราใช้อยู่

- `pathlib` ช่วยในการดึงชื่อไฟล์ ชื่อ path โดยไม่ต้องเขียนโค้ดเยอะ และเป็นระเบียบกว่าการแบ่ง string โดยฟังก์ชั่นที่พบบ่อย เช่น

```py
from pathlib import Path
path = 'deep-learning-skooldio/thai-handwritten-dataset/1/001.jpg'
p = Path(path)
print(str(p.parent)) # >>> 'deep-learning-skooldio/thai-handwritten-dataset/1'
print(p.parent.name) # >>> "1" ได้ชื่อ parent folder นั่นคือโฟลเดอร์ "1 " นั่นเอง
print(p.name) # >>> 001.jpg ได้ชื่อไฟล์พร้อมสกุลของไฟล์
print(p.stem) # >>> 001  ได้ชื่อไฟล์แบบไม่มีสกุล
```

ในตัวอย่าง `p.parent.name` ถูกนำมาใช้ในการดึงชื่อโฟลเดอร์เพื่อใช้เป็น label ของภาพในโฟลเดอร์, `p.name` ถูกนำมาใช้ในการดึงชื่อไฟล์เพื่อ copy, `p.stem` ถูกใช้บ่อยเช่นกันในการดึงชื่อไฟล์ แต่ไม่ต้องการสกุลของไฟล์

- `shutil` เป็นไลบรารี่ที่มีตำสั่งเกี่ยวกับการดำเนินการเกี่ยวกับไฟล์ เช่น การ copy, การย้ายไฟล์ ในบทเรียนเราใช้คำสั่ง
`shutil.copy(src, dsc)` เพื่อ copy ไฟล์จากต้นทางไปยังปลายทางหลังจากที่เราแบ่ง paths เป็น train paths และ validation paths

- `tqdm` มีที่มาจากภาษาอารบิก taqaddum (تقدّم) ที่แปลว่าความคืบหน้า (“progress") ดังนั้นเราจึงเห็นว่า `tqdm` ถูกนำมาใช้ครอบเพื่อติดตามความคืบหน้าของ for loop นั่นเอง ตัวอย่างการใช้ `tqdm` เพียงนำมาครอบระหว่าง for loop ก็สามารถ
ติดตามความคืบหน้าของการรันโค้ดได้ เช่น

```py
import time
from tqdm.auto import tqdm
for i in tqdm(range(5)):
    time.sleep(0.1)
print("Done")
```

การใช้ `from tqdm.auto import tqdm` จะเป็นการเลือกให้อัตโนมัติว่าเรากำลังรันใน ipython หรีือ Jupyter notebook
ซึ่งการแสดงผลของ progress bar จะถูกปรับให้แสดงผลได้อย่างเหมาะสมใน environment รูปแบบต่างๆ


## Built-in อื่นๆ ที่พบในบทเรียน

- `enumerate` สามารถนำไปครอบ for-loop เพื่อสร้าง index ให้กับ element ในลิสต์

```py
for e in ["a", "b", "c"]:
    print(e)

for i, e in enumerate(["a", "b", "c"]):
    print(i, e)  # ได้ i ซึ่งเป็น index ของลิสต์ด้วย
```

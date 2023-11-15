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

เมื่อเราสร้าง Class มาเรียบร้อยจะสามารถสร้าง Class เพิ่มเติมจากเดิม ("inheritance") โดยที่อาจจะเพิ่ม function อื่นๆ หรือปรับเปลี่ยน Class เดิมได้

```py
class Student(Person):
  def __init__(self, name, age, grad_year):
    super().__init__(name, age)
    self.grad_year = grad_year
s1 = Student("Jane", 18, 2022)
s1.display_name()
print(s1.name)
print(s1.age)
print(s1.grad_year)
```

ในที่นี้เราสามารถสร้างคลาสที่ชื่อว่า `Student` ที่รับทั้ง `name`, `age`, และปีที่จบ `grad_year` โดยจะเห็นว่าตัวแปร `s1` ที่สร้างขึ้นจะสามารถเข้าถึง `s1.name`, `s1.age` ที่มีในคลาส `Person` และ `s1.grad_year` ที่มีในคลาส `Student` ได้

คอนเซ็ปต์เดียวกันถูกนำมาใช้สำหรับการสร้างชุดข้อมูลและโมเดลด้วย Pytorch โดยเราสามารถสร้าง Class จาก `nn.Module` และ `torch.utils.data.Dataset`

- สำหรับ `nn.Module` ที่เป็น class พื้นฐานของการสร้างโมเดล Neural netowkr  เราจะต้องเขียนฟังก์ชั่นหลักคือ `__init__()` และ `forward()` เพื่อประกาศเลเยอร์ของโมเดล และการคำนวณ
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

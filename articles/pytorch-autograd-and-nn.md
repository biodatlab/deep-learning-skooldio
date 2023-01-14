# Autograd and Neural network

## Autograd

ส่วนหนึ่งที่สำคัญของ Pytorch คือการมี engine ที่เรียกว่า Autograd ที่สามารถคำนวณ gradient หรือความชันของฟังก์ชัน
ได้โดยอัตโนมัติ ยกตัวอย่างเช่น ถ้าเรามีฟังก์ชัน `f(x) = x^2 - 4 x` และเราต้องการหาค่า gradient ของ `f(x)` ต่อ `x` ที่ค่า
`x = 10` สามารถทำได้ด้วย Pytorch ได้ดังนี้

```py
x = torch.tensor(10., requires_grad=True)
f = torch.sum(x * x - 4 * x)
f.backward()
x.grad
```

จะได้ค่า `x.grad` เท่ากับ 16 ซึ่งเป็นค่าที่คำนวณได้ด้วยสูตร `2 x - 4` เมื่อแทนค่า `x = 10` เข้าไป

ทั้งนี้ autograd จึงเป็นส่วนสำคัญของการทำ back propagation ใน Neural network เพื่อใช้อัพเดทค่า weight ในแต่ละ layer
เพื่อให้ loss ลดลง

## Neural network

ในการเขียน Pytorch เราสามารถเขียน Neural network layer และประกอบร่างกันด้วย Neural network module

**โดยปกติแล้ว Pytorch จะสมมติให้ dimension แรกเป็น batch size** ดังนั้นสมมติถ้าเรามี input ขนาด `(10, 5)`
Pytorch จะเห็นว่าเป็น input ของ 10 ตัวอย่าง และมี dimension ของแต่ละตัวอย่างเท่ากับ 5
เมื่อผ่าน linear layer ขนาด `(5, 3)` จะได้ output ขนาด `(10, 3)` ทดลองได้ดังนี้

```py
import torch
import torch.nn as nn

x = torch.randn((10, 5))
fc = nn.Linear(5, 3)
h = fc(x)
print(h.shape) # ได้ (10, 3)
```

ในอนาคตเราอาจจะพบกับ layer อื่นๆ อีกมากมาย เช่น Convolutional layer ซึ่งเวลาเขียนจะใช้การเขียนแบบเดียวกันคือ batch size
เป็น dimension แรกเสมอ

```py
x = torch.randn((10, 1, 28, 28)) # batch size, channel, height, width
conv = nn.Conv2d(1, 8, kernel_size=3, padding=1) # เปลี่ยนจาก channel 1 เป็น 8
h = conv(x)
print(h.shape) # ได้ (10, 8, 28, 28)
```

จะเห็นว่า batch size จะมีขนาดเท่ากับ input คือ 10 แต่จะมี channel เพิ่มเป็น 8 แทน
ในคลาสนี้ยังไม่ได้ลงลึกเกี่ยวกับ Convolutional layer แต่จะเห็นภาพรวมเกี่ยวกับการเขียนและการใช้งาน Pytorch ได้เผื่อในอนาคต

**ฝึกบ่อยๆ:** การฝึกสร้าง Neural network layer แล้วป้อน input เพื่อดู output dimension จะทำให้เราสามารถเขียน
Neural network module ได้คล่องขึ้นในอนาคต

## Neural network functional

functions ที่พบบ่อยได้แก่การใช้ activation function, การทำ softmax เช่น

```py
x = torch.tensor([10.])
F.relu(x)

x = torch.tensor([-10.])
F.relu(x)

x = torch.tensor([0.])
torch.sigmoid(x)
```

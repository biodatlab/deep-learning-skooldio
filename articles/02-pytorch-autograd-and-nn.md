# Autograd and Neural network

## การหาจุดต่ำสุดของฟังก์ชัน

ย้อนไปสมัยเรียนมัธยมปลาย ถ้าสมมติอยากจะหาค่าของ `x` ที่ทำให้ `f(x) = x ** 2 − 4x` มีค่าต่ำที่สุด
เราก็จะหา first derivative (หรือเรียกสั้นๆว่าดิฟ) ของ `f(x)` นั่นคือ `f′(x) = 2x − 4`
แล้วแก้สมการ `f′(x) = 2x − 4 = 0` เพื่อได้ค่าต่ำสุดของฟังก์ชัน นั่นคือ `x = 2` นั่นเอง

หรืออีกวิธีนั้นเราอาจจะจัดรูปสมการ `f(x)= (x ** 2 − 4x + 4) − 4 = (x − 2)**2 − 4`
ก็จะรู้ว่าค่าต่ำที่สุดของ `f(x)` คือ -4 เมื่อแทนค่า `x = 2`

แล้วถ้าจะใช้คอมพิวเตอร์หาค่าต่ำสุดของฟังก์ชันนั้น จะต้องใช้วิธีอะไร? คำตอบคือ Gradient descent นั่นเอง

## Gradient descent

การเคลื่อนลงตามความชัน (Gradient descent) เป็นแก่นของการแก้เพื่อหาค่าที่เหมาะสมที่สุดให้กับฟังก์ชั่นหรือ Cost function
จากการคำนวณหาความชัน (slope, gradint) ณ จุดที่เราอยู่แล้วพยายามเดินไปทางตรงข้ามกับความชัน
ถ้านึกภาพก็เหมือนเวลาเราเดินลงเขา จะเห็นว่าเราจะเดินไปทางตรงข้ามกับ ความชันไปเรื่องๆเราอาจจะลงไปถึงจุดต่ำสุดของเขาได้นั่นเอง
โดยเราสามารถอัพเดทด้วยสมการ:

```py
x = x - lr * gradient
```

- `lr` คือ learning rate ใช้กำหนดความเร็วในการเดินลงเขา หรือค่า scale ในการอัพเดทค่า `x` ในแต่ละรอบ
- `gradient` คือ ความชันของฟังก์ชันที่เราต้องการหาค่าต่ำสุด ณ​ จุด `x` ที่เราอยู่

โดยปกติการหาความชัน (gradient) นี้จะต้องใช้การคำนวณด้วยมือ เช่น สมการ `f(x) = x**2 - 4x` คำนวณ gradient ได้เท่ากับ
`f'(x) = 2x - 4` แต่ไม่ต้องห่วงว่าเราจะต้องคำนวณ gradient ด้วยมือระหว่างการสร้าง Neural network เนื่องจากใน Pytorch มี engine
ที่ชื่อว่า Autograd ที่สามารถคำนวณ gradient ได้อัตโนมัติ

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

## ใช้งาน Autograd ร่วมกับ Gradient descent

ถ้าเราใช้ autograd ร่วมกับ gradient descent เพื่อหาค่าต่ำสุดของ function `f(x) = x^2 - 4x` จะได้โค้ดดังนี้

```py
import torch

alpha = 0.02 # กำหนดพารามิเตอร์สำหรับการอัพเดทค่า x
x = torch.tensor(10, dtype=torch.float, requires_grad=True)  # กำหนดค่า x ที่เริ่มต้น ใช้  requires_grad=True เพื่อให้ autograd คำนวณ gradient ได้
f = torch.sum(x * x - 4 * x) # ฟังก์ชันสำหรับกราฟพาราโบลา x ^ 2 + 4 x

# รัน gradient descent algorithm 1000 ครั้ง
for _ in range(1000):
    f.backward(retain_graph=True) # คำนวณความชันหรือ gradient โดยใช้ autograd ``.backward()``
    x.data.sub_(alpha * x.grad) # เทียบเท่ากับ x = x - alpha * gradient โดย x.grad เก็บ gradient ไว้
    x.grad.data.zero_() # หลังจากเราคำนวณ gradient แล้ว เราต้องตั้งค่ากลับไปที่ 0 อีกครั้งหนึ่งเพื่อคำนวณใหม่
print(x) # เราจะได้ค่า x ต่ำที่สุดที่ 2
```

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

ปัจจุบัน `relu` มักเป็นที่นิยมในการใช้สำหรับ non-linear layer ของ Neural Network

## การใส่ non-linear layer ใน Neural Network

ลองนึกภาพในกรณีที่เราไม่ใส่ non-linear layer หรือ activation function ระหว่างเลเยอร์ของ Neural Network ก่อน

กรณีที่เรามี 2 เลเยอร์ เมื่อใส่ input `x` เข้าไปในโมเดล เราจะได้ผลลัพธ์หลังจากใส่ input เป็น
`pred = fc2(fc1(x))` ซึ่งความสัมพันธ์ระหว่าง input กับ output แบบไม่มี non-linear layer อาจจะเขียนเป็น `pred = fc3(x)`
ได้เพราะว่า `fc2` กับ `fc1` สามารถนำมาคูณกันได้ตรงๆ ดังนั้นการ optimize โมเดลแบบไม่มี non-linear layer จึงอาจจะไม่สามารถ
หาความสัมพันธ์ของข้อมูลกรณีที่มีความซับซ้อนมากได้

ดังนั้นการใส่ non-linear layer ทำให้ Neural Network สามารถเรียนรู้แพทเทิร์นระหว่าง input กับ output แบบ non-linear ได้นั่นเอง

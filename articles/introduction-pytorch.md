# Introduction to Pytorch

ในบทเรียนที่ผ่านมาเราได้เรียนรู้การสร้าง tensor ซึ่งใช้เป็นตัวกลางในการเก็บข้อมูลของไลบรารี่ Pytorch และได้
ลองสร้าง tensor operate โดยใช้วิธีต่างๆ รวมถึงลองเพิ่มหรือลด dimension ของ tensor

## Create tensors

วิธีการสร้าง tensor สามารถทำได้หลายรูปแบบ เช่น

- ใช้ `torch.tensor` เพื่อ cast จาก list เป็น tensor

```py
v = torch.tensor([1, 2, 3]) # ถ้า list ที่ใช้ในการสร้าง tensor มีค่าเป็น int จะได้ tensor ที่มีค่าเป็น int
vf = torch.tensor([1., 2., 3.]) # ถ้า list ที่ใช้ในการสร้าง tensor มีค่าเป็น float จะได้ tensor ที่มีค่าเป็น float
```

- ใช้ `torch.from_numpy` เพื่อ cast จาก numpy array เป็น tensor บางครั้งเหมาะกับการ

```py
v_np = np.array([1., 2., 3.])
v2 = torch.from_numpy(v_np)
```

- หรืออาจจะสร้าง tensor โดยใช้ `torch.zeros` หรือ `torch.ones`

```py
v3 = torch.zeros(3) # tensor ที่มีค่า 0 ขนาดเท่ากับ 3
v4 = torch.ones(3) # tensor ที่มีค่า 1 ขนาดเท่ากับ 3
```

- หรือสามารถสร้าง tensor ที่มีค่าเป็นตัวเลขสุ่มได้ด้วย `torch.rand`

```py
v5 = torch.rand(3) # tensor ที่มีค่าเป็นตัวเลขสุ่มระหว่าง 0 ถึง 1 ขนาดเท่ากับ 3
v6 = torch.randn(5) # tensor ที่มีค่าเป็นตัวเลขสุ่มจากการแจกแจงปกติ ขนาดเท่ากับ 5
v7 = torch.randn((16, 1, 30)) # tensor ที่มีค่าเป็นตัวเลขสุ่มจากการแจกแจงปกติ ขนาดเท่ากับ (16, 1, 30)
```

- หรือสามารถสร้าง tensor ที่มีค่าเป็นตัวเลขเรียงต่อกันได้ด้วย `torch.arange`

```py
v8 = torch.arange(0, 10, 0.01) # tensor ที่มีขนาดตั้งแต่ 0 ถึง 10 เว้นช่วงทีละ 0.01
```

## Operations

มี operations ที่เราสามารถนำมาใช้จัดการ tensor ได้หลายอย่าง เช่น

### Math Operations

- การบวก tensor เช่น

```py
A = torch.tensor([[1., 2.], [3., 4.]])
B = torch.tensor([[1., 1.], [0., 2.]])

A + B
A.add(B)
torch.add(A, B)
A.add_(B)
```

เพียงแค่บวก tensor สามารถเขียนได้หลากหลายแบบมาก โดย method `add_` เป็น method พิเศษที่ใช้
inplace เช่น `A.add_(B)` เทียบเท่ากับ `A = A + B` (แต่ใช้ `A.add_(B)` จะ efficient กว่า)

- การคูณ tensor แบบทีละ element

```py
A * B
A.multiply(B) # หรือ A.mul(B)
torch.multiply(A, B)
A.mul_(B)
```

- การคูณ tensor แบบ matrix multiplication

```py
A @ B
A.matmul(B)
A.mm(B)
torch.matmul(A,B)
torch.mm(A,B)
```

การเขียน operations เป็นไปได้หลายรูปแบบมาก ในชีวิตจริงเราจะเจอการเขียนหลากหลายรูปแบบ

### Changing dimensions

Operation ที่เกี่ยวกับการเปลี่ยน dimensions ของ tensor เช่น

- การเชื่อม tensor

```py
A = torch.tensor([[1, 2], [3, 4]])
B = torch.tensor([[5, 6], [7, 8]])
torch.stack((A, B), dim=0) # เชื่อม dim 0 ได้ขนาด (2, 2, 2)
torch.cat((A, B)) # เชื่อมแนวตั้ง ได้ขนาด (4, 2) คนนิยมใช้ในการเชื่อม tensor เวลาสร้าง neural network
torch.vstack((A, B)) # เชื่อมแนวตั้ง ได้ขนาด (4, 2)
torch.hstack((A, B)) # เชื่อมแนวนอน ได้ขนาด (2, 4)
```

- การ transpose tensor หรือ สลับ dim

```py
A = torch.randn((20, 30, 1))
A.transpose(0, 2) # สลับ dim 0 กับ dim 2 ได้ขนาด (1, 30, 20)
```

- การ reshape tensor

```py
A = torch.randn(6)
A.reshape((3, 2)) # ได้ขนาด (3, 2)
A.view(-1, 2) # กี่ row ก็ได้ แต่ต้องมี 2 columns ได้ขนาด (3, 2)
```

- การ squeeze tensor และ unsqueeze tensor เรามักจะเห็นการใช้ unsqueeze กับ dim 0 เพื่อเพิ่ม dim ของ tensor
  ให้ใส่เข้าไปใน Neural Network ได้ และมักเห็นการใช้ squeeze tensor เวลาเอา output ของ neural network มาใช้ต่อ
  เช่น นำมาพล็อต หรือหาตำแหน่งที่สูงหรือต่ำที่สุด

```py
A.squeeze(0) # ตัด dimension 0 ออก
A.unsqueeze(0) # เพิ่ม dimension 0 เข้าไป
```
